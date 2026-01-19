//! LLM wrapper for CQuests.
//!
//! Phase 3 goal: provide a stable interface for the GM agent:
//! - create sessions (KV-cache-like continuity)
//! - stream generation (token chunks)
//! - cancellation
//! - bounded session memory (LRU eviction)
//!
//! Default backend is a mock implementation so the workspace compiles and tests
//! run without model assets.

use std::sync::{
	atomic::{AtomicBool, AtomicU64, Ordering},
	Arc,
};
use std::thread;
use std::time::{Duration, Instant};

use crossbeam_channel::Receiver;
use lru::LruCache;
use serde::{Deserialize, Serialize};
use thiserror::Error;
#[cfg(feature = "burn")]
use toml;

/// Opaque handle for an LLM session (KV cache, rolling context, etc.).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct LlmSessionHandle(pub u64);

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChatMessage {
	pub role: ChatRole,
	pub content: String,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ChatRole {
	System,
	User,
	Assistant,
	Tool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SamplingParams {
	pub temperature: f32,
	pub top_p: f32,
	pub top_k: u32,
	pub repetition_penalty: f32,
}

impl Default for SamplingParams {
	fn default() -> Self {
		Self {
			temperature: 0.7,
			top_p: 0.9,
			top_k: 50,
			repetition_penalty: 1.05,
		}
	}
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GenerateRequest {
	pub system_prompt: String,
	pub messages: Vec<ChatMessage>,
	pub sampling: SamplingParams,
	pub stop_sequences: Vec<String>,
	pub max_new_tokens: u32,
}

impl GenerateRequest {
	pub fn new(system_prompt: impl Into<String>) -> Self {
		Self {
			system_prompt: system_prompt.into(),
			messages: Vec::new(),
			sampling: SamplingParams::default(),
			stop_sequences: Vec::new(),
			max_new_tokens: 256,
		}
	}
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Usage {
	pub prompt_tokens: u32,
	pub completion_tokens: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
	Stop,
	Length,
	Cancelled,
	Error,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum TokenEvent {
	TokenChunk { text: String },
	/// A candidate block that looks like a tool call (raw JSON or a block).
	ToolCallCandidate { raw: String },
	Finished { usage: Usage, reason: FinishReason },
	Error { message: String },
}

/// Streaming token receiver.
///
/// Implemented as a channel so we don't force an async runtime in Phase 3.
pub struct TokenStream {
	rx: Receiver<TokenEvent>,
}

impl TokenStream {
	pub fn recv(&self) -> Option<TokenEvent> {
		self.rx.recv().ok()
	}
}

impl Iterator for TokenStream {
	type Item = TokenEvent;
	fn next(&mut self) -> Option<Self::Item> {
		self.recv()
	}
}

#[derive(Debug, Error)]
pub enum LlmError {
	#[error("unknown session")]
	UnknownSession,
	#[error("backend not available: {0}")]
	BackendUnavailable(String),
	#[error("invalid request: {0}")]
	InvalidRequest(String),
	#[error("internal error: {0}")]
	Internal(String),
}

pub type LlmResult<T> = Result<T, LlmError>;

/// Stable interface for the GM agent.
pub trait LlmClient: Send + Sync {
	fn create_session(&self) -> LlmSessionHandle;
	fn reset_session(&self, session: LlmSessionHandle) -> LlmResult<()>;
	fn abort(&self, session: LlmSessionHandle) -> LlmResult<()>;
	fn generate_stream(&self, session: LlmSessionHandle, req: GenerateRequest) -> LlmResult<TokenStream>;
}

/// Configuration for session caching / LRU.
#[derive(Clone, Debug)]
pub struct SessionCacheConfig {
	pub max_sessions: usize,
}

impl Default for SessionCacheConfig {
	fn default() -> Self {
		Self { max_sessions: 16 }
	}
}

/// Minimal per-session state for Phase 3.
#[derive(Clone, Debug)]
struct SessionState {
	/// Rolling transcript for the mock backend; for burn backend this is where
	/// KV-cache handle and metadata will live.
	transcript: String,
	last_used: Instant,
	cancelled: Arc<AtomicBool>,
}

/// Default client: mock streaming generator with cancellation and LRU session cap.
pub struct MockLlm {
	next_id: AtomicU64,
	inner: parking_lot::Mutex<LruCache<u64, SessionState>>,
	cfg: SessionCacheConfig,
}

impl MockLlm {
	pub fn new(cfg: SessionCacheConfig) -> Self {
		let cap = std::num::NonZeroUsize::new(cfg.max_sessions.max(1)).unwrap();
		Self {
			next_id: AtomicU64::new(1),
			inner: parking_lot::Mutex::new(LruCache::new(cap)),
			cfg,
		}
	}

	fn touch(&self, session: LlmSessionHandle) -> LlmResult<()> {
		let mut inner = self.inner.lock();
		let Some(state) = inner.get_mut(&session.0) else {
			return Err(LlmError::UnknownSession);
		};
		state.last_used = Instant::now();
		Ok(())
	}

	fn ensure_session(&self, session: LlmSessionHandle) -> LlmResult<SessionState> {
		let mut inner = self.inner.lock();
		let Some(state) = inner.get(&session.0) else {
			return Err(LlmError::UnknownSession);
		};
		Ok(state.clone())
	}
}

impl Default for MockLlm {
	fn default() -> Self {
		Self::new(SessionCacheConfig::default())
	}
}

impl LlmClient for MockLlm {
	fn create_session(&self) -> LlmSessionHandle {
		let id = self.next_id.fetch_add(1, Ordering::Relaxed);
		let mut inner = self.inner.lock();
		inner.put(
			id,
			SessionState {
				transcript: String::new(),
				last_used: Instant::now(),
				cancelled: Arc::new(AtomicBool::new(false)),
			},
		);
		LlmSessionHandle(id)
	}

	fn reset_session(&self, session: LlmSessionHandle) -> LlmResult<()> {
		let mut inner = self.inner.lock();
		let Some(state) = inner.get_mut(&session.0) else {
			return Err(LlmError::UnknownSession);
		};
		state.transcript.clear();
		state.cancelled.store(false, Ordering::Relaxed);
		state.last_used = Instant::now();
		Ok(())
	}

	fn abort(&self, session: LlmSessionHandle) -> LlmResult<()> {
		let mut inner = self.inner.lock();
		let Some(state) = inner.get_mut(&session.0) else {
			return Err(LlmError::UnknownSession);
		};
		state.cancelled.store(true, Ordering::Relaxed);
		Ok(())
	}

	fn generate_stream(&self, session: LlmSessionHandle, req: GenerateRequest) -> LlmResult<TokenStream> {
		if req.max_new_tokens == 0 {
			return Err(LlmError::InvalidRequest("max_new_tokens must be > 0".into()));
		}

		self.touch(session)?;
		let state = self.ensure_session(session)?;

		let (tx, rx) = crossbeam_channel::unbounded();

		// Mock generation: echoes a short, deterministic response based on input,
		// and emits it in small chunks with cancellation support.
		let cancelled = state.cancelled.clone();
		let prompt_snapshot = build_prompt_snapshot(&req);
		let prior = state.transcript;

		thread::spawn(move || {
			let start = Instant::now();

			let response = mock_response(&prompt_snapshot, &prior);
			let mut emitted = 0u32;
			let chunk_size = 16usize;

			for chunk in response.as_bytes().chunks(chunk_size) {
				if cancelled.load(Ordering::Relaxed) {
					let _ = tx.send(TokenEvent::Finished {
						usage: Usage {
							prompt_tokens: estimate_tokens(&prompt_snapshot) as u32,
							completion_tokens: emitted,
						},
						reason: FinishReason::Cancelled,
					});
					return;
				}

				let text = String::from_utf8_lossy(chunk).to_string();
				emitted += estimate_tokens(&text) as u32;
				let _ = tx.send(TokenEvent::TokenChunk { text });
				thread::sleep(Duration::from_millis(10));
			}

			let _elapsed = start.elapsed();
			let _ = tx.send(TokenEvent::Finished {
				usage: Usage {
					prompt_tokens: estimate_tokens(&prompt_snapshot) as u32,
					completion_tokens: emitted,
				},
				reason: if emitted >= req.max_new_tokens {
					FinishReason::Length
				} else {
					FinishReason::Stop
				},
			});
		});

		Ok(TokenStream { rx })
	}
}

fn build_prompt_snapshot(req: &GenerateRequest) -> String {
	let mut s = String::new();
	s.push_str("[SYSTEM]\n");
	s.push_str(&req.system_prompt);
	s.push_str("\n");
	for m in &req.messages {
		s.push_str(match m.role {
			ChatRole::System => "[SYSTEM]",
			ChatRole::User => "[USER]",
			ChatRole::Assistant => "[ASSISTANT]",
			ChatRole::Tool => "[TOOL]",
		});
		s.push('\n');
		s.push_str(&m.content);
		s.push('\n');
	}
	s
}

fn estimate_tokens(text: &str) -> usize {
	// Placeholder heuristic: 1 token ~ 4 chars.
	// Burn backend will report real usage.
	(text.len().max(1) + 3) / 4
}

fn mock_response(prompt: &str, prior_transcript: &str) -> String {
	// Deterministic “response” that demonstrates:
	// - strict tool-call protocol
	// - session continuity (via prior transcript)
	// - forming valid MCP tool arguments (session_id/player_id from system prompt)

	fn parse_u64_after(label: &str, s: &str) -> Option<u64> {
		let idx = s.find(label)?;
		let rest = &s[idx + label.len()..];
		let digits: String = rest.chars().skip_while(|c| *c == ' ' || *c == '=').take_while(|c| c.is_ascii_digit()).collect();
		digits.parse().ok()
	}

	let lower = prompt.to_ascii_lowercase();
	let session_id = parse_u64_after("session_id", &lower).or_else(|| parse_u64_after("session_id=", &lower));
	let player_id = parse_u64_after("player_id", &lower).or_else(|| parse_u64_after("player_id=", &lower));

	// Heuristic: focus on the most recent user message if present.
	let user_text = if let Some(idx) = lower.rfind("[user]") {
		lower[idx + "[user]".len()..].trim().to_string()
	} else {
		lower.clone()
	};

	// Helper to emit strict tool call (single line, no prefix text).
	let tool_call = |name: &str, arguments: serde_json::Value| -> String {
		format!(
			"TOOL_CALL: {}",
			serde_json::json!({"name": name, "arguments": arguments}).to_string()
		)
	};

	// For tests/demos we always want to form valid args; if ids are missing, narrate.
	let Some(session_id) = session_id else {
		return "NARRATIVE: I need a session context before I can act.".into();
	};
	let Some(player_id) = player_id else {
		return "NARRATIVE: I need a player context before I can act.".into();
	};

	// Intents
	// If a tool was already executed, narrate from the result.
	if lower.contains("[tool]") && lower.contains("tool_result") {
		let mut body = String::new();
		if !prior_transcript.is_empty() {
			body.push_str("(continuing) ");
		}
		body.push_str("I interpret the tool results and continue the scene.");
		return format!("NARRATIVE: {body}");
	}

	if user_text.contains("move") && (user_text.contains("room b") || user_text.contains(" to b") || user_text.contains("destination 2")) {
		return tool_call(
			"move",
			serde_json::json!({"session_id": session_id, "player_id": player_id, "destination_id": 2}),
		);
	}
	if user_text.contains("move") && (user_text.contains("room a") || user_text.contains(" to a") || user_text.contains("destination 1")) {
		return tool_call(
			"move",
			serde_json::json!({"session_id": session_id, "player_id": player_id, "destination_id": 1}),
		);
	}
	if user_text.contains("attack") || user_text.contains("hit") {
		// Demo world: goblin entity id 11
		return tool_call(
			"attack",
			serde_json::json!({"session_id": session_id, "player_id": player_id, "target_id": 11}),
		);
	}
	if user_text.contains("inspect") || user_text.contains("look") {
		// Default: inspect current location id 1 (demo world starts in A)
		return tool_call(
			"inspect",
			serde_json::json!({"session_id": session_id, "player_id": player_id, "target": {"kind": "location", "id": 1}}),
		);
	}
	if user_text.contains("roll") || user_text.contains("dice") || user_text.contains("tool_call") {
		return tool_call(
			"roll",
			serde_json::json!({"session_id": session_id, "dice_expr": "d20"}),
		);
	}

	let mut body = String::new();
	if !prior_transcript.is_empty() {
		body.push_str("(continuing) ");
	}
	body.push_str("I hear you. The world awaits your next move.");
	format!("NARRATIVE: {body}")
}

/// Backend selector for the CLI and future GM.
pub enum Backend {
	Mock(MockLlm),

	#[cfg(feature = "burn")]
	Burn(BurnLlm),
}

impl Backend {
	pub fn mock() -> Self {
		Self::Mock(MockLlm::default())
	}

	#[cfg(feature = "burn")]
	pub fn burn_from_manifest_path(path: &str) -> LlmResult<Self> {
		Ok(Self::Burn(BurnLlm::from_manifest_path(path)?))
	}
}

impl LlmClient for Backend {
	fn create_session(&self) -> LlmSessionHandle {
		match self {
			Backend::Mock(m) => m.create_session(),

			#[cfg(feature = "burn")]
			Backend::Burn(b) => b.create_session(),
		}
	}

	fn reset_session(&self, session: LlmSessionHandle) -> LlmResult<()> {
		match self {
			Backend::Mock(m) => m.reset_session(session),

			#[cfg(feature = "burn")]
			Backend::Burn(b) => b.reset_session(session),
		}
	}

	fn abort(&self, session: LlmSessionHandle) -> LlmResult<()> {
		match self {
			Backend::Mock(m) => m.abort(session),

			#[cfg(feature = "burn")]
			Backend::Burn(b) => b.abort(session),
		}
	}

	fn generate_stream(&self, session: LlmSessionHandle, req: GenerateRequest) -> LlmResult<TokenStream> {
		match self {
			Backend::Mock(m) => m.generate_stream(session, req),

			#[cfg(feature = "burn")]
			Backend::Burn(b) => b.generate_stream(session, req),
		}
	}
}

// --- Burn backend (feature-gated) ---

#[cfg(feature = "burn")]
pub struct BurnLlm {
	next_id: AtomicU64,
	// Per-session mutable state (KV cache, transcript, cancellation flag).
	sessions: parking_lot::Mutex<std::collections::HashMap<u64, Arc<parking_lot::Mutex<BurnSession>>>>,

	// Model config loaded from ./models/manifest.toml
	manifest: ModelManifest,
}

#[cfg(feature = "burn")]
impl BurnLlm {
	/// Load model config from `models/manifest.toml` (repo-root relative).
	pub fn from_default_models_dir() -> LlmResult<Self> {
		Self::from_manifest_path("models/manifest.toml")
	}

	pub fn from_manifest_path(path: &str) -> LlmResult<Self> {
		let text = std::fs::read_to_string(path)
			.map_err(|e| LlmError::Internal(format!("failed to read manifest {path}: {e}")))?;
		let mut manifest: ModelManifest = toml::from_str(&text)
			.map_err(|e| LlmError::InvalidRequest(format!("invalid manifest: {e}")))?;

		// Resolve relative paths relative to the manifest file location.
		let base_dir = std::path::Path::new(path)
			.parent()
			.unwrap_or(std::path::Path::new("."));
		manifest.model.checkpoint = resolve_manifest_path(base_dir, &manifest.model.checkpoint);
		manifest.model.tokenizer = resolve_manifest_path(base_dir, &manifest.model.tokenizer);

		Ok(Self {
			next_id: AtomicU64::new(1),
			sessions: parking_lot::Mutex::new(std::collections::HashMap::new()),
			manifest,
		})
	}
}

#[cfg(feature = "burn")]
impl Default for BurnLlm {
	fn default() -> Self {
		Self::from_default_models_dir().expect("models/manifest.toml must exist for burn backend")
	}
}

#[cfg(feature = "burn")]
impl LlmClient for BurnLlm {
	fn create_session(&self) -> LlmSessionHandle {
		let id = self.next_id.fetch_add(1, Ordering::Relaxed);
		let mut sessions = self.sessions.lock();
		sessions.insert(
			id,
			Arc::new(parking_lot::Mutex::new(BurnSession {
				llama: None,
				transcript: String::new(),
				bos_used: false,
				cancelled: Arc::new(AtomicBool::new(false)),
			})),
		);
		LlmSessionHandle(id)
	}

	fn reset_session(&self, session: LlmSessionHandle) -> LlmResult<()> {
		let sess = {
			let sessions = self.sessions.lock();
			sessions.get(&session.0).cloned().ok_or(LlmError::UnknownSession)?
		};
		let mut sess = sess.lock();
		sess.cancelled.store(false, Ordering::Relaxed);
		sess.transcript.clear();
		sess.bos_used = false;
		if let Some(llama) = sess.llama.as_mut() {
			llama.reset();
		}
		Ok(())
	}

	fn abort(&self, session: LlmSessionHandle) -> LlmResult<()> {
		let sess = {
			let sessions = self.sessions.lock();
			sessions.get(&session.0).cloned().ok_or(LlmError::UnknownSession)?
		};
		sess.lock().cancelled.store(true, Ordering::Relaxed);
		Ok(())
	}

	fn generate_stream(&self, session: LlmSessionHandle, req: GenerateRequest) -> LlmResult<TokenStream> {
		if req.max_new_tokens == 0 {
			return Err(LlmError::InvalidRequest("max_new_tokens must be > 0".into()));
		}

		let sess = {
			let sessions = self.sessions.lock();
			sessions.get(&session.0).cloned().ok_or(LlmError::UnknownSession)?
		};

		let manifest = self.manifest.clone();

		let (tx, rx) = crossbeam_channel::unbounded();
		thread::spawn(move || {
			let mut sess = sess.lock();

			let delta = match format_llama3_delta(&sess.transcript.is_empty(), &req.system_prompt, &req.messages) {
				Ok(d) => d,
				Err(e) => {
					let _ = tx.send(TokenEvent::Error { message: e });
					let _ = tx.send(TokenEvent::Finished {
						usage: Usage {
							prompt_tokens: 0,
							completion_tokens: 0,
						},
						reason: FinishReason::Error,
					});
					return;
				}
			};

			// Ensure model is loaded for this session.
			if sess.llama.is_none() {
				match burn_backend_load_llama(&manifest) {
					Ok(llama) => sess.llama = Some(llama),
					Err(e) => {
						let _ = tx.send(TokenEvent::Error { message: e });
						let _ = tx.send(TokenEvent::Finished {
							usage: Usage {
								prompt_tokens: 0,
								completion_tokens: 0,
							},
							reason: FinishReason::Error,
						});
						return;
					}
				}
			}

			let cancelled = sess.cancelled.clone();

			// Run streaming generation.
			let start = Instant::now();
			let result = burn_streaming_generate(
				&mut *sess,
				&delta,
				&req,
				manifest.sampling.seed,
				manifest.sampling.top_p,
				&tx,
			);
			let _elapsed = start.elapsed();

			match result {
				Ok(finish) => {
					let _ = tx.send(TokenEvent::Finished {
						usage: Usage {
							prompt_tokens: estimate_tokens(&delta) as u32,
							completion_tokens: finish.emitted_tokens,
						},
						reason: finish.reason,
					});
				}
				Err(e) => {
					let _ = tx.send(TokenEvent::Error { message: e });
					let _ = tx.send(TokenEvent::Finished {
						usage: Usage {
							prompt_tokens: estimate_tokens(&delta) as u32,
							completion_tokens: 0,
						},
						reason: if cancelled.load(Ordering::Relaxed) {
							FinishReason::Cancelled
						} else {
							FinishReason::Error
						},
					});
				}
			}
		});

		Ok(TokenStream { rx })
	}
}

// ---- Burn backend helpers (feature gated) ----

// Enforce selecting a single Burn backend.
#[cfg(all(feature = "burn-tch", feature = "burn-cuda"))]
compile_error!("Select only one burn backend feature: burn-tch | burn-cuda | burn-vulkan | burn-ndarray");
#[cfg(all(feature = "burn-tch", feature = "burn-vulkan"))]
compile_error!("Select only one burn backend feature: burn-tch | burn-cuda | burn-vulkan | burn-ndarray");
#[cfg(all(feature = "burn-tch", feature = "burn-ndarray"))]
compile_error!("Select only one burn backend feature: burn-tch | burn-cuda | burn-vulkan | burn-ndarray");
#[cfg(all(feature = "burn-cuda", feature = "burn-vulkan"))]
compile_error!("Select only one burn backend feature: burn-tch | burn-cuda | burn-vulkan | burn-ndarray");
#[cfg(all(feature = "burn-cuda", feature = "burn-ndarray"))]
compile_error!("Select only one burn backend feature: burn-tch | burn-cuda | burn-vulkan | burn-ndarray");
#[cfg(all(feature = "burn-vulkan", feature = "burn-ndarray"))]
compile_error!("Select only one burn backend feature: burn-tch | burn-cuda | burn-vulkan | burn-ndarray");

#[cfg(feature = "burn")]
#[derive(Clone, Debug, Deserialize)]
struct ModelManifest {
	pub model: ModelSection,
	#[serde(default)]
	pub sampling: SamplingSection,
}

#[cfg(feature = "burn")]
#[derive(Clone, Debug, Deserialize)]
struct ModelSection {
	pub variant: String,
	pub checkpoint: String,
	pub tokenizer: String,
	pub max_seq_len: usize,
}

#[cfg(feature = "burn")]
#[derive(Clone, Debug, Deserialize)]
struct SamplingSection {
	#[serde(default = "default_temp")]
	pub temperature: f64,
	#[serde(default = "default_top_p")]
	pub top_p: f64,
	#[serde(default = "default_seed")]
	pub seed: u64,
}

#[cfg(feature = "burn")]
impl Default for SamplingSection {
	fn default() -> Self {
		Self {
			temperature: default_temp(),
			top_p: default_top_p(),
			seed: default_seed(),
		}
	}
}

#[cfg(feature = "burn")]
fn resolve_manifest_path(base_dir: &std::path::Path, p: &str) -> String {
	let path = std::path::Path::new(p);
	if path.is_absolute() {
		p.to_string()
	} else {
		base_dir.join(path).to_string_lossy().to_string()
	}
}

#[cfg(feature = "burn")]
fn default_temp() -> f64 {
	0.6
}

#[cfg(feature = "burn")]
fn default_top_p() -> f64 {
	0.9
}

#[cfg(feature = "burn")]
fn default_seed() -> u64 {
	42
}

#[cfg(feature = "burn")]
fn format_llama3_instruct_prompt(system_prompt: &str, messages: &[ChatMessage]) -> Result<String, String> {
	// Match the llama-burn example template (see llama-burn/examples/chat.rs).
	let mut out = String::new();

	out.push_str("<|start_header_id|>system<|end_header_id|>\n\n");
	if system_prompt.trim().is_empty() {
		out.push_str(
			"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.",
		);
	} else {
		out.push_str(system_prompt.trim());
	}
	out.push_str("<|eot_id|>");

	for m in messages {
		match m.role {
			ChatRole::User => {
				out.push_str("<|start_header_id|>user<|end_header_id|>\n\n");
				out.push_str(m.content.trim());
				out.push_str("<|eot_id|>");
			}
			ChatRole::Assistant => {
				out.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
				out.push_str(m.content.trim());
				out.push_str("<|eot_id|>");
			}
			ChatRole::System => {
				// We already provided one system section at the top; fold additional system
				// messages into the conversation as user-visible instructions.
				out.push_str("<|start_header_id|>system<|end_header_id|>\n\n");
				out.push_str(m.content.trim());
				out.push_str("<|eot_id|>");
			}
			ChatRole::Tool => {
				// Phase 3: tools are encoded as assistant messages in the prompt.
				out.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
				out.push_str(m.content.trim());
				out.push_str("<|eot_id|>");
			}
		}
	}

	// Open assistant turn.
	out.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
	Ok(out)
}

#[cfg(feature = "burn")]
struct BurnSession {
	llama: Option<BurnLlama>,
	/// Full formatted prompt transcript already committed into the KV cache.
	transcript: String,
	/// Whether we've already added a BOS token for this session.
	bos_used: bool,
	cancelled: Arc<AtomicBool>,
}

#[cfg(feature = "burn")]
struct BurnFinish {
	emitted_tokens: u32,
	reason: FinishReason,
}

#[cfg(feature = "burn")]
type BurnLlama = burn_backend::LlamaType;

#[cfg(feature = "burn")]
mod burn_backend {
	use super::*;
	use burn::tensor::Device;
	use llama_burn::llama::{Llama, LlamaConfig};
	use llama_burn::tokenizer::Tiktoken;

	// One-backend selection is enforced by compile_error! earlier in the file.
	#[cfg(feature = "burn-tch")]
	pub type Backend = burn::backend::LibTorch<burn::tensor::f16>;
	#[cfg(feature = "burn-tch")]
	pub type DeviceType = burn::backend::libtorch::LibTorchDevice;

	#[cfg(feature = "burn-cuda")]
	pub type Backend = burn::backend::Cuda<burn::tensor::f16, i32>;
	#[cfg(feature = "burn-cuda")]
	pub type DeviceType = burn::backend::cuda::CudaDevice;

	#[cfg(feature = "burn-vulkan")]
	pub type Backend = burn::backend::wgpu::Vulkan<burn::tensor::f16, i32>;
	#[cfg(feature = "burn-vulkan")]
	pub type DeviceType = burn::backend::wgpu::WgpuDevice;

	#[cfg(feature = "burn-ndarray")]
	pub type Backend = burn::backend::ndarray::NdArray<f32>;
	#[cfg(feature = "burn-ndarray")]
	pub type DeviceType = ();

	pub type LlamaType = Llama<Backend, Tiktoken>;

	pub fn device() -> Device<Backend> {
		#[cfg(feature = "burn-tch")]
		{
			return DeviceType::Cpu;
		}
		#[cfg(feature = "burn-cuda")]
		{
			return DeviceType::default();
		}
		#[cfg(feature = "burn-vulkan")]
		{
			return DeviceType::default();
		}
		#[cfg(feature = "burn-ndarray")]
		{
			return Default::default();
		}
	}

	pub fn load(manifest: &ModelManifest) -> Result<LlamaType, String> {
		#[cfg(not(feature = "burn-llama3"))]
		{
			let _ = manifest;
			return Err("burn backend requires feature burn-llama3".into());
		}

		#[cfg(feature = "burn-llama3")]
		{
			let device = device();
			let checkpoint = manifest.model.checkpoint.as_str();
			let tokenizer_path = manifest.model.tokenizer.as_str();
			let max_seq_len = manifest.model.max_seq_len;

			match manifest.model.variant.as_str() {
				"llama3_1_8b_instruct" => {
					LlamaConfig::load_llama3_1_8b::<Backend>(checkpoint, tokenizer_path, max_seq_len, &device)
				}
				"llama3_8b_instruct" => {
					LlamaConfig::load_llama3_8b::<Backend>(checkpoint, tokenizer_path, max_seq_len, &device)
				}
				"llama3_2_1b_instruct" => {
					LlamaConfig::load_llama3_2_1b::<Backend>(checkpoint, tokenizer_path, max_seq_len, &device)
				}
				"llama3_2_3b_instruct" => {
					LlamaConfig::load_llama3_2_3b::<Backend>(checkpoint, tokenizer_path, max_seq_len, &device)
				}
				other => Err(format!("unknown model.variant: {other}")),
			}
		}
	}
}

#[cfg(feature = "burn")]
fn burn_backend_load_llama(manifest: &ModelManifest) -> Result<BurnLlama, String> {
	burn_backend::load(manifest)
}

#[cfg(feature = "burn")]
fn format_llama3_delta(is_new_session: &bool, system_prompt: &str, messages: &[ChatMessage]) -> Result<String, String> {
	let mut out = String::new();

	if *is_new_session {
		out.push_str("<|start_header_id|>system<|end_header_id|>\n\n");
		if system_prompt.trim().is_empty() {
			out.push_str(
				"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.",
			);
		} else {
			out.push_str(system_prompt.trim());
		}
		out.push_str("<|eot_id|>");
	}

	for m in messages {
		match m.role {
			ChatRole::User => {
				out.push_str("<|start_header_id|>user<|end_header_id|>\n\n");
				out.push_str(m.content.trim());
				out.push_str("<|eot_id|>");
			}
			ChatRole::Assistant => {
				out.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
				out.push_str(m.content.trim());
				out.push_str("<|eot_id|>");
			}
			ChatRole::System => {
				out.push_str("<|start_header_id|>system<|end_header_id|>\n\n");
				out.push_str(m.content.trim());
				out.push_str("<|eot_id|>");
			}
			ChatRole::Tool => {
				out.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
				out.push_str(m.content.trim());
				out.push_str("<|eot_id|>");
			}
		}
	}

	// Open assistant turn.
	out.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
	Ok(out)
}

#[cfg(feature = "burn")]
fn burn_streaming_generate(
	sess: &mut BurnSession,
	delta: &str,
	req: &GenerateRequest,
	default_seed: u64,
	default_top_p: f64,
	tx: &crossbeam_channel::Sender<TokenEvent>,
) -> Result<BurnFinish, String> {
	use burn::tensor::backend::Backend as BurnBackendTrait;
	use burn::tensor::{activation::softmax, Int, Shape, Tensor, TensorData};
	use burn_backend::Backend as B;
	use llama_burn::sampling::Sampler;
	use llama_burn::tokenizer::Tokenizer;

	let llama = sess.llama.as_mut().ok_or("llama not loaded")?;

	let temperature = req.sampling.temperature as f64;
	let top_p = req.sampling.top_p as f64;
	let seed = default_seed;

	let mut sampler = if temperature > 0.0 {
		Sampler::new_top_p(if top_p > 0.0 { top_p } else { default_top_p }, seed)
	} else {
		Sampler::Argmax
	};

	let stop_ids = llama.tokenizer.stop_ids().clone();
	let mut generated_token_ids: Vec<u32> = Vec::new();
	let mut last_decoded = String::new();
	let mut emitted_tokens: u32 = 0;

	// Prefill KV cache with the new delta (only BOS on first-ever prefill).
	let bos = !sess.bos_used;
	let delta_token_ids = llama.tokenizer.encode(delta, bos, false);
	sess.bos_used = true;

	if delta_token_ids.is_empty() {
		return Err("delta prompt produced no tokens".into());
	}

	let x_prompt: Tensor<B, 2, Int> = Tensor::from_data(
		TensorData::new(delta_token_ids.clone(), Shape::new([1, delta_token_ids.len()])),
		&llama.device,
	);

	let mut logits = llama.model.forward(x_prompt, &mut llama.cache, &llama.rope);

	// Generate token-by-token.
	for i in 0..(req.max_new_tokens as usize) {
		if sess.cancelled.load(Ordering::Relaxed) {
			// Close out the assistant turn so the next user message can follow.
			burn_feed_text(llama, "<|eot_id|>").ok();
			sess.transcript.push_str(delta);
			sess.transcript.push_str(&last_decoded);
			sess.transcript.push_str("<|eot_id|>");
			return Ok(BurnFinish {
				emitted_tokens,
				reason: FinishReason::Cancelled,
			});
		}

		let [batch_size, seq_len, _vocab_size] = logits.dims();
		let mut next_token_logits = logits
			.slice([0..batch_size, seq_len - 1..seq_len])
			.squeeze_dim(1); // [1, vocab]

		if temperature > 0.0 {
			next_token_logits = softmax(next_token_logits / temperature, 1);
		}

		let next_token = sampler.sample(next_token_logits).squeeze_dim::<1>(0); // [1]
		let raw_token = *next_token
			.clone()
			.into_data()
			.as_slice::<<B as BurnBackendTrait>::IntElem>()
			.map_err(|_| "failed to read sampled token")?
			.first()
			.ok_or("empty sampled token")?;
		let next_token_id = u32::try_from(raw_token).map_err(|_| "sampled token id out of range")?;

		if stop_ids.contains(&next_token_id) {
			// Advance cache with the stop token, but don't emit it.
			burn_feed_token(llama, next_token_id).ok();
			let reason = if i + 1 >= req.max_new_tokens as usize {
				FinishReason::Length
			} else {
				FinishReason::Stop
			};
			// Commit delta and any generated text (if any) to the transcript.
			sess.transcript.push_str(delta);
			sess.transcript.push_str(&last_decoded);
			sess.transcript.push_str("<|eot_id|>");
			return Ok(BurnFinish {
				emitted_tokens,
				reason,
			});
		}

		generated_token_ids.push(next_token_id);
		let decoded = llama.tokenizer.decode(generated_token_ids.clone());
		let to_send = if decoded.starts_with(&last_decoded) {
			decoded[last_decoded.len()..].to_string()
		} else {
			decoded.clone()
		};
		last_decoded = decoded;

		emitted_tokens += 1;
		if !to_send.is_empty() {
			let _ = tx.send(TokenEvent::TokenChunk { text: to_send });
		}

		// Feed the generated token back in to advance KV cache and get next logits.
		let x_next: Tensor<B, 2, Int> = Tensor::from_data(
			TensorData::new(vec![next_token_id], Shape::new([1, 1])),
			&llama.device,
		);
		logits = llama.model.forward(x_next, &mut llama.cache, &llama.rope);
	}

	// Max tokens reached; close assistant turn and commit transcript.
	burn_feed_text(llama, "<|eot_id|>").ok();
	sess.transcript.push_str(delta);
	sess.transcript.push_str(&last_decoded);
	sess.transcript.push_str("<|eot_id|>");
	Ok(BurnFinish {
		emitted_tokens,
		reason: FinishReason::Length,
	})
}

#[cfg(feature = "burn")]
fn burn_feed_token(llama: &mut BurnLlama, token_id: u32) -> Result<(), String> {
	use burn::tensor::{Int, Shape, Tensor, TensorData};
	use burn_backend::Backend as B;

	let x_next: Tensor<B, 2, Int> = Tensor::from_data(
		TensorData::new(vec![token_id], Shape::new([1, 1])),
		&llama.device,
	);
	let _ = llama.model.forward(x_next, &mut llama.cache, &llama.rope);
	Ok(())
}

#[cfg(feature = "burn")]
fn burn_feed_text(llama: &mut BurnLlama, text: &str) -> Result<(), String> {
	use burn::tensor::{Int, Shape, Tensor, TensorData};
	use burn_backend::Backend as B;
	use llama_burn::tokenizer::Tokenizer;

	let token_ids = llama.tokenizer.encode(text, false, false);
	if token_ids.is_empty() {
		return Ok(());
	}
	let token_len = token_ids.len();
	let x: Tensor<B, 2, Int> = Tensor::from_data(
		TensorData::new(token_ids, Shape::new([1, token_len])),
		&llama.device,
	);
	let _ = llama.model.forward(x, &mut llama.cache, &llama.rope);
	Ok(())
}

