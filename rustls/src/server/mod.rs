use crate::builder::{ConfigBuilder, WantsCipherSuites};
use crate::conn::{CommonState, Connection, ConnectionCommon, IoState, Reader, State, Writer};
use crate::error::Error;
use crate::keylog::KeyLog;
use crate::kx::SupportedKxGroup;
#[cfg(feature = "logging")]
use crate::log::trace;
use crate::msgs::base::PayloadU8;
#[cfg(feature = "quic")]
use crate::msgs::enums::AlertDescription;
use crate::msgs::enums::ProtocolVersion;
use crate::msgs::enums::SignatureScheme;
use crate::msgs::handshake::{ConvertProtocolNameList, ServerExtension};
use crate::msgs::message::Message;
use crate::sign;
use crate::suites::SupportedCipherSuite;
use crate::verify;
#[cfg(feature = "quic")]
use crate::{conn::Protocol, quic};

use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};
use std::sync::Arc;
use std::{fmt, io};

#[macro_use]
mod hs;
pub(crate) mod builder;
mod common;
pub(crate) mod handy;
mod tls12;
mod tls13;

/// A trait for the ability to store server session data.
///
/// The keys and values are opaque.
///
/// Both the keys and values should be treated as
/// **highly sensitive data**, containing enough key material
/// to break all security of the corresponding sessions.
///
/// Implementations can be lossy (in other words, forgetting
/// key/value pairs) without any negative security consequences.
///
/// However, note that `take` **must** reliably delete a returned
/// value.  If it does not, there may be security consequences.
///
/// `put` and `take` are mutating operations; this isn't expressed
/// in the type system to allow implementations freedom in
/// how to achieve interior mutability.  `Mutex` is a common
/// choice.
pub trait StoresServerSessions: Send + Sync {
    /// Store session secrets encoded in `value` against `key`,
    /// overwrites any existing value against `key`.  Returns `true`
    /// if the value was stored.
    fn put(&self, key: Vec<u8>, value: Vec<u8>) -> bool;

    /// Find a value with the given `key`.  Return it, or None
    /// if it doesn't exist.
    fn get(&self, key: &[u8]) -> Option<Vec<u8>>;

    /// Find a value with the given `key`.  Return it and delete it;
    /// or None if it doesn't exist.
    fn take(&self, key: &[u8]) -> Option<Vec<u8>>;
}

/// A trait for the ability to encrypt and decrypt tickets.
pub trait ProducesTickets: Send + Sync {
    /// Returns true if this implementation will encrypt/decrypt
    /// tickets.  Should return false if this is a dummy
    /// implementation: the server will not send the SessionTicket
    /// extension and will not call the other functions.
    fn enabled(&self) -> bool;

    /// Returns the lifetime in seconds of tickets produced now.
    /// The lifetime is provided as a hint to clients that the
    /// ticket will not be useful after the given time.
    ///
    /// This lifetime must be implemented by key rolling and
    /// erasure, *not* by storing a lifetime in the ticket.
    ///
    /// The objective is to limit damage to forward secrecy caused
    /// by tickets, not just limiting their lifetime.
    fn lifetime(&self) -> u32;

    /// Encrypt and authenticate `plain`, returning the resulting
    /// ticket.  Return None if `plain` cannot be encrypted for
    /// some reason: an empty ticket will be sent and the connection
    /// will continue.
    fn encrypt(&self, plain: &[u8]) -> Option<Vec<u8>>;

    /// Decrypt `cipher`, validating its authenticity protection
    /// and recovering the plaintext.  `cipher` is fully attacker
    /// controlled, so this decryption must be side-channel free,
    /// panic-proof, and otherwise bullet-proof.  If the decryption
    /// fails, return None.
    fn decrypt(&self, cipher: &[u8]) -> Option<Vec<u8>>;
}

/// How to choose a certificate chain and signing key for use
/// in server authentication.
pub trait ResolvesServerCert: Send + Sync {
    /// Choose a certificate chain and matching key given simplified
    /// ClientHello information.
    ///
    /// Return `None` to abort the handshake.
    fn resolve(&self, client_hello: ClientHello) -> Option<Arc<sign::CertifiedKey>>;
}

/// A struct representing the received Client Hello
pub struct ClientHello<'a> {
    server_name: Option<webpki::DnsNameRef<'a>>,
    signature_schemes: &'a [SignatureScheme],
    alpn: Option<Vec<&'a [u8]>>,
}

impl<'a> ClientHello<'a> {
    /// Creates a new ClientHello
    fn new<'sni: 'a, 'sigs: 'a, 'alpn: 'a>(
        sni: &'sni Option<webpki::DnsName>,
        signature_schemes: &'sigs [SignatureScheme],
        alpn: Option<&'alpn Vec<PayloadU8>>,
    ) -> Self {
        let server_name = sni
            .as_ref()
            .map(webpki::DnsName::as_ref);
        let alpn = alpn.map(|protos| protos.to_slices());

        trace!("sni {:?}", server_name);
        trace!("sig schemes {:?}", signature_schemes);
        trace!("alpn protocols {:?}", alpn);

        ClientHello {
            server_name,
            signature_schemes,
            alpn,
        }
    }

    /// Get the server name indicator.
    ///
    /// Returns `None` if the client did not supply a SNI.
    pub fn server_name(&self) -> Option<&str> {
        self.server_name
            .as_ref()
            .and_then(|s| std::str::from_utf8(s.as_ref()).ok())
    }

    /// Get the compatible signature schemes.
    ///
    /// Returns standard-specified default if the client omitted this extension.
    pub fn signature_schemes(&self) -> &[SignatureScheme] {
        self.signature_schemes
    }

    /// Get the alpn.
    ///
    /// Returns `None` if the client did not include an ALPN extension
    pub fn alpn(&self) -> Option<&[&[u8]]> {
        self.alpn.as_deref()
    }
}

/// Common configuration for a set of server sessions.
///
/// Making one of these can be expensive, and should be
/// once per process rather than once per connection.
///
/// These must be created via the [`ServerConfig::builder()`] function.
///
/// # Defaults
///
/// * [`ServerConfig::max_fragment_size`]: the default is `None`: TLS packets are not fragmented to a specific size.
/// * [`ServerConfig::session_storage`]: the default stores 256 sessions in memory.
/// * [`ServerConfig::alpn_protocols`]: the default is empty -- no ALPN protocol is negotiated.
/// * [`ServerConfig::key_log`]: key material is not logged.
#[derive(Clone)]
pub struct ServerConfig {
    /// List of ciphersuites, in preference order.
    cipher_suites: Vec<SupportedCipherSuite>,

    /// List of supported key exchange groups.
    ///
    /// The first is the highest priority: they will be
    /// offered to the client in this order.
    kx_groups: Vec<&'static SupportedKxGroup>,

    /// Ignore the client's ciphersuite order. Instead,
    /// choose the top ciphersuite in the server list
    /// which is supported by the client.
    pub ignore_client_order: bool,

    /// The maximum size of TLS message we'll emit.  If None, we don't limit TLS
    /// message lengths except to the 2**16 limit specified in the standard.
    ///
    /// rustls enforces an arbitrary minimum of 32 bytes for this field.
    /// Out of range values are reported as errors from ServerConnection::new.
    ///
    /// Setting this value to the TCP MSS may improve latency for stream-y workloads.
    pub max_fragment_size: Option<usize>,

    /// How to store client sessions.
    pub session_storage: Arc<dyn StoresServerSessions + Send + Sync>,

    /// How to produce tickets.
    pub ticketer: Arc<dyn ProducesTickets>,

    /// How to choose a server cert and key.
    pub cert_resolver: Arc<dyn ResolvesServerCert>,

    /// Protocol names we support, most preferred first.
    /// If empty we don't do ALPN at all.
    pub alpn_protocols: Vec<Vec<u8>>,

    /// Supported protocol versions, in no particular order.
    /// The default is all supported versions.
    versions: crate::versions::EnabledVersions,

    /// How to verify client certificates.
    verifier: Arc<dyn verify::ClientCertVerifier>,

    /// How to output key material for debugging.  The default
    /// does nothing.
    pub key_log: Arc<dyn KeyLog>,

    /// Amount of early data to accept; 0 to disable.
    #[cfg(feature = "quic")] // TLS support unimplemented
    #[doc(hidden)]
    pub max_early_data_size: u32,
}

impl ServerConfig {
    /// Create builder to build up the server configuration.
    ///
    /// For more information, see the [`ConfigBuilder`] documentation.
    pub fn builder() -> ConfigBuilder<Self, WantsCipherSuites> {
        ConfigBuilder {
            state: WantsCipherSuites(()),
            side: PhantomData::default(),
        }
    }

    #[doc(hidden)]
    /// We support a given TLS version if it's quoted in the configured
    /// versions *and* at least one ciphersuite for this version is
    /// also configured.
    pub fn supports_version(&self, v: ProtocolVersion) -> bool {
        self.versions.contains(v)
            && self
                .cipher_suites
                .iter()
                .any(|cs| cs.version().version == v)
    }
}

/// This represents a single TLS server connection.
///
/// Send TLS-protected data to the peer using the `io::Write` trait implementation.
/// Read data from the peer using the `io::Read` trait implementation.
pub struct ServerConnection {
    inner: ConnectionCommon<ServerConnectionData>,
}

impl ServerConnection {
    /// Make a new ServerConnection.  `config` controls how
    /// we behave in the TLS protocol.
    pub fn new(config: Arc<ServerConfig>) -> Result<Self, Error> {
        Self::from_config(config, vec![])
    }

    fn from_config(
        config: Arc<ServerConfig>,
        extra_exts: Vec<ServerExtension>,
    ) -> Result<Self, Error> {
        let common = CommonState::new(config.max_fragment_size, false)?;
        Ok(Self {
            inner: ConnectionCommon::new(
                Box::new(hs::ExpectClientHello::new(config, extra_exts)),
                ServerConnectionData::default(),
                common,
            ),
        })
    }

    /// Retrieves the SNI hostname, if any, used to select the certificate and
    /// private key.
    ///
    /// This returns `None` until some time after the client's SNI extension
    /// value is processed during the handshake. It will never be `None` when
    /// the connection is ready to send or process application data, unless the
    /// client does not support SNI.
    ///
    /// This is useful for application protocols that need to enforce that the
    /// SNI hostname matches an application layer protocol hostname. For
    /// example, HTTP/1.1 servers commonly expect the `Host:` header field of
    /// every request on a connection to match the hostname in the SNI extension
    /// when the client provides the SNI extension.
    ///
    /// The SNI hostname is also used to match sessions during session
    /// resumption.
    pub fn sni_hostname(&self) -> Option<&str> {
        self.inner.data.get_sni_str()
    }

    /// Application-controlled portion of the resumption ticket supplied by the client, if any.
    ///
    /// Recovered from the prior session's `set_resumption_data`. Integrity is guaranteed by rustls.
    ///
    /// Returns `Some` iff a valid resumption ticket has been received from the client.
    pub fn received_resumption_data(&self) -> Option<&[u8]> {
        self.inner
            .data
            .received_resumption_data
            .as_ref()
            .map(|x| &x[..])
    }

    /// Set the resumption data to embed in future resumption tickets supplied to the client.
    ///
    /// Defaults to the empty byte string. Must be less than 2^15 bytes to allow room for other
    /// data. Should be called while `is_handshaking` returns true to ensure all transmitted
    /// resumption tickets are affected.
    ///
    /// Integrity will be assured by rustls, but the data will be visible to the client. If secrecy
    /// from the client is desired, encrypt the data separately.
    pub fn set_resumption_data(&mut self, data: &[u8]) {
        assert!(data.len() < 2usize.pow(15));
        self.inner.data.resumption_data = data.into();
    }

    /// Explicitly discard early data, notifying the client
    ///
    /// Useful if invariants encoded in `received_resumption_data()` cannot be respected.
    ///
    /// Must be called while `is_handshaking` is true.
    pub fn reject_early_data(&mut self) {
        assert!(
            self.is_handshaking(),
            "cannot retroactively reject early data"
        );
        self.inner.data.reject_early_data = true;
    }
}

impl Connection for ServerConnection {
    fn read_tls(&mut self, rd: &mut dyn io::Read) -> io::Result<usize> {
        self.inner.read_tls(rd)
    }

    fn process_new_packets(&mut self) -> Result<IoState, Error> {
        self.inner.process_new_packets()
    }

    fn export_keying_material(
        &self,
        output: &mut [u8],
        label: &[u8],
        context: Option<&[u8]>,
    ) -> Result<(), Error> {
        self.inner
            .export_keying_material(output, label, context)
    }

    fn writer(&mut self) -> Writer {
        Writer::new(&mut self.inner)
    }

    fn reader(&mut self) -> Reader {
        self.inner.reader()
    }
}

impl fmt::Debug for ServerConnection {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("ServerConnection")
            .finish()
    }
}

impl Deref for ServerConnection {
    type Target = CommonState;

    fn deref(&self) -> &Self::Target {
        &self.inner.common_state
    }
}

impl DerefMut for ServerConnection {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner.common_state
    }
}

/// Handle on a server-side connection before configuration is available.
///
/// The `Acceptor` allows the caller to provide a [`ServerConfig`] that is customized
/// based on the [`ClientHello`] of the incoming connection. Using the [`ResolvesServerConfig`]
/// trait, the caller can build a [`ServerConfig`] and [`sign::CertifiedKey`] based on the
/// `ClientHello`.
pub struct Acceptor {
    resolver: Arc<dyn ResolvesServerConfig>,
    inner: ConnectionCommon<ServerConnectionData>,
}

impl Acceptor {
    /// Create a new `Acceptor`.
    pub fn new(
        resolver: Arc<dyn ResolvesServerConfig>,
        max_fragment_size: Option<usize>,
    ) -> Result<Self, Error> {
        let common = CommonState::new(max_fragment_size, false)?;
        let state = Box::new(Accepting);
        Ok(Self {
            resolver,
            inner: ConnectionCommon::new(state, Default::default(), common),
        })
    }

    /// Read TLS content from `rd`.
    ///
    /// For more details, see [`Connection::read_tls()`].
    pub fn read_tls(&mut self, rd: &mut dyn io::Read) -> Result<usize, io::Error> {
        self.inner.read_tls(rd)
    }

    /// Check if a `ClientHello` message has been received.
    ///
    /// Returns an error if the `ClientHello` message is invalid or `Ok(None)` if no complete
    /// `ClientHello` has been received yet. If `Some(Accepted)` is returned, the [`Accepted`] token
    /// type can be used to turn the `Acceptor` into a [`ServerConnection`] using
    /// [`Acceptor::into_connection()`].
    pub fn read_client_hello(&mut self) -> Result<Option<Accepted>, Error> {
        let msg = match self.inner.first_handshake_message() {
            Ok(Some(msg)) => msg,
            Ok(None) => return Ok(None),
            Err(e) => return Err(e),
        };

        let (sig_schemes, client_hello) = hs::process_client_hello(
            &msg,
            false,
            &mut self.inner.common_state,
            &mut self.inner.data,
        )?;

        let ch = ClientHello::new(
            &self.inner.data.sni,
            &sig_schemes,
            client_hello.get_alpn_extension(),
        );

        let (config, key) = match self.resolver.resolve(ch) {
            Some((config, key)) => (config, key),
            None => return Err(Error::General("no server config resolved".to_string())),
        };

        let state = hs::ExpectClientHello::new(config, Vec::new());
        let mut cx = hs::ServerContext {
            common: &mut self.inner.common_state,
            data: &mut self.inner.data,
        };

        let new = state.with_certified_key(key, sig_schemes, client_hello, &msg, &mut cx)?;
        self.inner.replace_state(new);
        Ok(Some(Accepted(())))
    }

    /// Turn the [`Acceptor`] into a [`ServerConnection`].
    ///
    /// Requires an [`Accepted`] token as returned from [`Acceptor::read_client_hello`].
    pub fn into_connection(self, _accepted: Accepted) -> ServerConnection {
        ServerConnection { inner: self.inner }
    }
}

impl Deref for Acceptor {
    type Target = CommonState;

    fn deref(&self) -> &Self::Target {
        &self.inner.common_state
    }
}

impl DerefMut for Acceptor {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner.common_state
    }
}

/// Trait that lets the implementer provide a [`ServerConfig`] based on the [`ClientHello`].
pub trait ResolvesServerConfig: Send + Sync {
    /// Resolve a [`ServerConfig`] and [`sign::CertifiedKey`] based on the [`ClientHello`].
    ///
    /// If this method returns `None`, the handshake will fail.
    fn resolve(
        &self,
        hello: ClientHello<'_>,
    ) -> Option<(Arc<ServerConfig>, Arc<sign::CertifiedKey>)>;
}

/// Token type used by the [`Acceptor`] to signal the connection can continue.
#[derive(Debug)]
pub struct Accepted(());

struct Accepting;

impl State<ServerConnectionData> for Accepting {
    fn handle(
        self: Box<Self>,
        _cx: &mut hs::ServerContext<'_>,
        _m: Message,
    ) -> Result<Box<dyn State<ServerConnectionData>>, Error> {
        unreachable!()
    }
}

/// State associated with a server connection.
#[derive(Default)]
struct ServerConnectionData {
    sni: Option<webpki::DnsName>,
    received_resumption_data: Option<Vec<u8>>,
    resumption_data: Vec<u8>,
    #[allow(dead_code)] // only supported for QUIC currently
    /// Whether to reject early data even if it would otherwise be accepted
    reject_early_data: bool,
}

impl ServerConnectionData {
    fn get_sni_str(&self) -> Option<&str> {
        self.sni.as_ref().map(AsRef::as_ref)
    }

    fn get_sni(&self) -> Option<verify::DnsName> {
        self.sni
            .as_ref()
            .map(|name| verify::DnsName(name.clone()))
    }
}

#[cfg(feature = "quic")]
impl quic::QuicExt for ServerConnection {
    fn quic_transport_parameters(&self) -> Option<&[u8]> {
        self.inner
            .common_state
            .quic
            .params
            .as_ref()
            .map(|v| v.as_ref())
    }

    fn zero_rtt_keys(&self) -> Option<quic::DirectionalKeys> {
        Some(quic::DirectionalKeys::new(
            self.inner
                .common_state
                .suite
                .and_then(|suite| suite.tls13())?,
            self.inner
                .common_state
                .quic
                .early_secret
                .as_ref()?,
        ))
    }

    fn read_hs(&mut self, plaintext: &[u8]) -> Result<(), Error> {
        self.inner.read_quic_hs(plaintext)
    }

    fn write_hs(&mut self, buf: &mut Vec<u8>) -> Option<quic::Keys> {
        quic::write_hs(&mut self.inner.common_state, buf)
    }

    fn alert(&self) -> Option<AlertDescription> {
        self.inner.common_state.quic.alert
    }

    fn next_1rtt_keys(&mut self) -> Option<quic::PacketKeySet> {
        quic::next_1rtt_keys(&mut self.inner.common_state)
    }
}

/// Methods specific to QUIC server sessions
#[cfg(feature = "quic")]
pub trait ServerQuicExt {
    /// Make a new QUIC ServerConnection. This differs from `ServerConnection::new()`
    /// in that it takes an extra argument, `params`, which contains the
    /// TLS-encoded transport parameters to send.
    fn new_quic(
        config: Arc<ServerConfig>,
        quic_version: quic::Version,
        params: Vec<u8>,
    ) -> Result<ServerConnection, Error> {
        if !config.supports_version(ProtocolVersion::TLSv1_3) {
            return Err(Error::General(
                "TLS 1.3 support is required for QUIC".into(),
            ));
        }

        if config.max_early_data_size != 0 && config.max_early_data_size != 0xffff_ffff {
            return Err(Error::General(
                "QUIC sessions must set a max early data of 0 or 2^32-1".into(),
            ));
        }

        let ext = match quic_version {
            quic::Version::V1Draft => ServerExtension::TransportParametersDraft(params),
            quic::Version::V1 => ServerExtension::TransportParameters(params),
        };
        let mut new = ServerConnection::from_config(config, vec![ext])?;
        new.inner.common_state.protocol = Protocol::Quic;
        Ok(new)
    }
}

#[cfg(feature = "quic")]
impl ServerQuicExt for ServerConnection {}
