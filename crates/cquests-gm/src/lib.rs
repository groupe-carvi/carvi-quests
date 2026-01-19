//! strict tool-call protocol + bounded tool loop.
//!
//! The GM agent sits between the player text and the MCP router tools:
//! - It prompts the LLM with the strict protocol.
//! - It parses either NARRATIVE or TOOL_CALL.
//! - It executes tools via `cquests-mcp` and feeds results back as tool messages.
//! - It enforces a bounded tool loop (prevents runaway tool calling).
//! - It maintains lightweight memory scaffolding (summary + facts).

use std::collections::HashMap;

use cquests_core::{PlayerId, SessionId};
use cquests_llm::{ChatMessage, ChatRole, FinishReason, GenerateRequest, LlmClient, LlmError, LlmSessionHandle, TokenEvent};
use cquests_mcp::{auth_gm, McpRouter, PROMPT_GM_TURN, TOOL_SEND_TO_PLAYERS};
use cquests_server::ActionService;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum GmError {
    #[error("llm error: {0}")]
    Llm(#[from] LlmError),

    #[error("mcp error: {0}")]
    Mcp(#[from] cquests_mcp::McpError),

    #[error("protocol violation: {0}")]
    Protocol(String),

    #[error("tool loop limit reached")]
    ToolLoopLimit,
}

pub type GmResult<T> = Result<T, GmError>;

/// Strict tool-call protocol:
///
/// - NARRATIVE: <text>
/// - TOOL_CALL: {"name": "...", "arguments": { ... }}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ToolCall {
    pub name: String,
    pub arguments: Value,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GmReply {
    Narrative(String),
    ToolCall(ToolCall),
}

/// Result of one GM turn, including tool-loop trace.
#[derive(Clone, Debug)]
pub struct GmTurnOutcome {
    pub narrative: String,
    /// In order of execution.
    pub tool_trace: Vec<(ToolCall, Value)>,
}

pub fn parse_gm_reply_strict(text: &str) -> GmResult<GmReply> {
    let t = text.trim();
    if let Some(rest) = t.strip_prefix("NARRATIVE:") {
        let body = rest.trim();
        if body.is_empty() {
            return Err(GmError::Protocol("empty NARRATIVE".into()));
        }
        return Ok(GmReply::Narrative(body.to_string()));
    }
    if let Some(rest) = t.strip_prefix("TOOL_CALL:") {
        let raw = rest.trim();
        let v: Value = serde_json::from_str(raw)
            .map_err(|e| GmError::Protocol(format!("invalid TOOL_CALL json: {e}")))?;
        let obj = v
            .as_object()
            .ok_or_else(|| GmError::Protocol("TOOL_CALL must be a JSON object".into()))?;
        let name = obj
            .get("name")
            .and_then(|v| v.as_str())
            .ok_or_else(|| GmError::Protocol("TOOL_CALL.name must be a string".into()))?
            .to_string();
        let arguments = obj
            .get("arguments")
            .cloned()
            .ok_or_else(|| GmError::Protocol("TOOL_CALL.arguments missing".into()))?;
        if !arguments.is_object() {
            return Err(GmError::Protocol("TOOL_CALL.arguments must be a JSON object".into()));
        }
        return Ok(GmReply::ToolCall(ToolCall { name, arguments }));
    }

    Err(GmError::Protocol(
        "output must start with exactly NARRATIVE: or TOOL_CALL:".into(),
    ))
}

#[derive(Clone, Debug, Default)]
pub struct MemoryScaffold {
    /// A short running recap.
    pub summary: String,
    /// Stable facts to keep consistent across turns.
    pub facts: Vec<String>,
}

impl MemoryScaffold {
    fn add_fact(&mut self, fact: impl Into<String>) {
        let f = fact.into();
        if f.trim().is_empty() {
            return;
        }
        if !self.facts.iter().any(|x| x == &f) {
            self.facts.push(f);
        }
    }
}

#[derive(Clone, Debug)]
struct GmSessionState {
    llm_session: LlmSessionHandle,
    memory: MemoryScaffold,
}

/// Generic over the server implementation so we can test with `InMemoryActionService`.
pub struct GmAgent<L: LlmClient, S: ActionService> {
    llm: L,
    router: McpRouter<S>,
    state: Mutex<HashMap<SessionId, GmSessionState>>,
    max_tool_calls: usize,
    max_new_tokens: u32,
}

impl<L: LlmClient, S: ActionService> GmAgent<L, S> {
    pub fn new(llm: L, router: McpRouter<S>) -> Self {
        Self {
            llm,
            router,
            state: Mutex::new(HashMap::new()),
            max_tool_calls: 4,
            // Default to a reasonably sized completion so narratives don't truncate.
            // Can be lowered for very slow CPU-only backends.
            max_new_tokens: 512,
        }
    }

    pub fn with_max_tool_calls(mut self, n: usize) -> Self {
        self.max_tool_calls = n.max(1);
        self
    }

    pub fn with_max_new_tokens(mut self, n: u32) -> Self {
        self.max_new_tokens = n.max(32);
        self
    }

    fn get_or_create_state(&self, session_id: SessionId) -> GmSessionState {
        let mut map = self.state.lock();
        map.entry(session_id)
            .or_insert_with(|| GmSessionState {
                llm_session: self.llm.create_session(),
                memory: MemoryScaffold::default(),
            })
            .clone()
    }

    fn put_state(&self, session_id: SessionId, st: GmSessionState) {
        self.state.lock().insert(session_id, st);
    }

    fn build_system_prompt(&self, session_id: SessionId, player_id: PlayerId) -> String {
        // Use the MCP prompt as the baseline contract, then make it stricter.
        let base = self
            .router
            .get_prompt(PROMPT_GM_TURN)
            .unwrap_or_else(|_| "You are the Game Master.".into());

        let st = self.get_or_create_state(session_id);

        let mut p = String::new();
        p.push_str(&base);
        p.push_str("\n\nSTRICT OUTPUT FORMAT (must match exactly):\n");
        p.push_str("- NARRATIVE: <text>\n");
        p.push_str("- TOOL_CALL: {\"name\":\"...\",\"arguments\":{...}}\n");
        p.push_str("No other prefixes. No markdown. No extra commentary.\n");

        p.push_str("\nSTYLE GUIDANCE:\n");
        p.push_str("- The player can attempt anything in the fiction. Treat requests as in-world attempts; narrate outcomes and consequences.\n");
        p.push_str("- Dark comedy and mature themes are OK when appropriate, but keep descriptions non-graphic and non-explicit.\n");
        p.push_str("- If the player pushes for explicit sexual content, do a fade-to-black / cutaway and continue with aftermath and plot momentum.\n");
        p.push_str("Do NOT call tools for normal conversation.\n");
        p.push_str("Only call tools when the player explicitly asks for an in-world action, or you must fetch state.\n\n");

        p.push_str("AVAILABLE_TOOLS (do not invent others):\n");
        for t in self.router.list_tools() {
            p.push_str("- ");
            p.push_str(&t.name);
            p.push('\n');
        }
        p.push('\n');

        // Provide stable identifiers so even the mock backend can form valid arguments.
        p.push_str(&format!("SESSION_ID={}\nPLAYER_ID={}\n\n", session_id.get(), player_id.get()));

        if !st.memory.summary.trim().is_empty() {
            p.push_str("MEMORY_SUMMARY:\n");
            p.push_str(st.memory.summary.trim());
            p.push_str("\n\n");
        }
        if !st.memory.facts.is_empty() {
            p.push_str("MEMORY_FACTS:\n");
            for f in &st.memory.facts {
                p.push_str("- ");
                p.push_str(f);
                p.push('\n');
            }
            p.push('\n');
        }

        p
    }

    fn is_known_tool(&self, name: &str) -> bool {
        self.router.list_tools().iter().any(|t| t.name == name)
    }

    fn autofill_tool_arguments(&self, args: &mut Value, session_id: SessionId, player_id: PlayerId) {
        let Value::Object(map) = args else {
            return;
        };
        // Many tools require session_id (and some also require player_id). The model
        // sometimes forgets these even though we provide them in the system prompt.
        // We inject them when missing to keep the protocol robust.
        map.entry("session_id".to_string())
            .or_insert_with(|| serde_json::json!(session_id.get()));
        map.entry("player_id".to_string())
            .or_insert_with(|| serde_json::json!(player_id.get()));
    }

    fn call_tool_lossy(
        &self,
        auth: &cquests_server::AuthContext,
        name: &str,
        args: Value,
    ) -> Value {
        match self.router.call_tool(auth, name, args) {
            Ok(v) => v,
            Err(e) => serde_json::json!({
                "error": "tool_call_failed",
                "tool": name,
                "details": e.to_string(),
            }),
        }
    }

    fn run_llm_once(&self, llm_session: LlmSessionHandle, req: GenerateRequest) -> GmResult<String> {
        let mut stream = self.llm.generate_stream(llm_session, req)?;
        let mut out = String::new();
        let mut finished = None;
        for ev in &mut stream {
            match ev {
                TokenEvent::TokenChunk { text } => out.push_str(&text),
                TokenEvent::ToolCallCandidate { .. } => {
                    // Current backends don't emit these reliably; parsing is done at the end.
                }
                TokenEvent::Finished { reason, .. } => {
                    finished = Some(reason);
                    break;
                }
                TokenEvent::Error { message } => return Err(GmError::Protocol(message)),
            }
        }
        match finished {
            Some(FinishReason::Cancelled) => Err(GmError::Protocol("llm cancelled".into())),
            Some(FinishReason::Error) => Err(GmError::Protocol("llm error".into())),
            _ => Ok(out),
        }
    }

    fn run_llm_once_streaming<F>(&self, llm_session: LlmSessionHandle, req: GenerateRequest, mut on_delta: F) -> GmResult<String>
    where
        F: FnMut(&str),
    {
        let mut stream = self.llm.generate_stream(llm_session, req)?;
        let mut out = String::new();
        let mut finished = None;

        // None: undecided (haven't seen full prefix yet)
        // Some(true): narrative (emit deltas)
        // Some(false): tool call (do not emit deltas)
        let mut mode: Option<bool> = None;
        let mut emitted_len: usize = 0;

        for ev in &mut stream {
            match ev {
                TokenEvent::TokenChunk { text } => {
                    out.push_str(&text);
                    if mode.is_none() {
                        let t = out.trim_start();
                        if t.starts_with("NARRATIVE:") {
                            mode = Some(true);
                        } else if t.starts_with("TOOL_CALL:") {
                            mode = Some(false);
                        }
                    }

                    if mode == Some(true) {
                        let t = out.trim_start();
                        if let Some(rest) = t.strip_prefix("NARRATIVE:") {
                            let body = rest.trim_start();
                            if body.len() > emitted_len {
                                on_delta(&body[emitted_len..]);
                                emitted_len = body.len();
                            }
                        }
                    }
                }
                TokenEvent::ToolCallCandidate { .. } => {
                    // Parsing is done at the end with strict rules.
                }
                TokenEvent::Finished { reason, .. } => {
                    finished = Some(reason);
                    break;
                }
                TokenEvent::Error { message } => return Err(GmError::Protocol(message)),
            }
        }

        match finished {
            Some(FinishReason::Cancelled) => Err(GmError::Protocol("llm cancelled".into())),
            Some(FinishReason::Error) => Err(GmError::Protocol("llm error".into())),
            _ => Ok(out),
        }
    }

    fn observe_tool_result_for_memory(mem: &mut MemoryScaffold, tool_name: &str, tool_result: &Value) {
        mem.add_fact(format!("last_tool={tool_name}"));

        // Keep this intentionally minimal/deterministic for the moment
        if let Some(events) = tool_result.get("events") {
            if let Some(arr) = events.as_array() {
                if !arr.is_empty() {
                    mem.add_fact(format!("events_emitted={}", arr.len()));
                }
            }
        }
}

    /// Handle one player turn. Returns a narrative string (without the `NARRATIVE:` prefix).
    pub fn handle_player_turn(
        &self,
        session_id: SessionId,
        player_id: PlayerId,
        player_text: &str,
    ) -> GmResult<String> {
        Ok(self
            .handle_player_turn_with_trace(session_id, player_id, player_text)?
            .narrative)
    }

    /// Handle one player turn and return a trace of tool calls + tool results.
    pub fn handle_player_turn_with_trace(
        &self,
        session_id: SessionId,
        player_id: PlayerId,
        player_text: &str,
    ) -> GmResult<GmTurnOutcome> {
        let mut st = self.get_or_create_state(session_id);
        let system_prompt = self.build_system_prompt(session_id, player_id);
        let auth = auth_gm();
        let mut tool_trace: Vec<(ToolCall, Value)> = Vec::new();

        let mut messages: Vec<ChatMessage> = vec![ChatMessage {
            role: ChatRole::User,
            content: player_text.to_string(),
        }];

        for _step in 0..self.max_tool_calls {
            let mut req = GenerateRequest::new(system_prompt.clone());
            req.messages = messages.clone();
			// Keep turns fast and deterministic by default.
			// On CPU backends, sampling (softmax/top-p) can be very slow for large vocabularies.
			req.sampling.temperature = 0.0; // Argmax
			req.sampling.top_p = 0.0;
            req.max_new_tokens = self.max_new_tokens;
            let llm_text = self.run_llm_once(st.llm_session, req)?;
            let reply = parse_gm_reply_strict(&llm_text)?;

            match reply {
                GmReply::Narrative(n) => {
                    // Update memory scaffolding in a deterministic, lightweight way.
                    if st.memory.summary.trim().is_empty() {
                        st.memory.summary = format!("Player: {}", player_text.trim());
                    } else {
                        st.memory.summary = format!("{} | Player: {}", st.memory.summary, player_text.trim());
                    }
                    st.memory.add_fact(format!("last_player_said={}", player_text.trim()));
                    self.put_state(session_id, st);
                    return Ok(GmTurnOutcome {
                        narrative: n,
                        tool_trace,
                    });
                }
                GmReply::ToolCall(tc) => {
                    let mut tc = tc;
                    self.autofill_tool_arguments(&mut tc.arguments, session_id, player_id);

                    // Reject hallucinated/unknown tools without failing the whole turn.
                    if !self.is_known_tool(&tc.name) {
                        let known: Vec<String> = self.router.list_tools().into_iter().map(|t| t.name).collect();
                        let out = serde_json::json!({
                            "error": "unknown tool",
                            "name": tc.name,
                            "known_tools": known,
                        });

                        messages.push(ChatMessage {
                            role: ChatRole::Assistant,
                            content: format!("TOOL_CALL: {}", serde_json::to_string(&tc).unwrap()),
                        });
                        tool_trace.push((tc.clone(), out.clone()));
                        messages.push(ChatMessage {
                            role: ChatRole::Tool,
                            content: format!("TOOL_RESULT: {}", out),
                        });
                        continue;
                    }

                    // Record tool call (as assistant) and execute.
                    messages.push(ChatMessage {
                        role: ChatRole::Assistant,
                        content: format!("TOOL_CALL: {}", serde_json::to_string(&tc).unwrap()),
                    });

                    let out = self.call_tool_lossy(&auth, &tc.name, tc.arguments.clone());
                    Self::observe_tool_result_for_memory(&mut st.memory, &tc.name, &out);
                    tool_trace.push((tc.clone(), out.clone()));

                    // Communication tool: return immediately with the text we just "sent".
                    if tc.name == TOOL_SEND_TO_PLAYERS {
                        if let Some(text) = tc.arguments.get("text").and_then(|v| v.as_str()) {
                            let text = text.trim();
                            if !text.is_empty() {
                                self.put_state(session_id, st);
                                return Ok(GmTurnOutcome {
                                    narrative: text.to_string(),
                                    tool_trace,
                                });
                            }
                        }
                    }

                    messages.push(ChatMessage {
                        role: ChatRole::Tool,
                        content: format!("TOOL_RESULT: {}", out),
                    });
                }
            }
        }

        Err(GmError::ToolLoopLimit)
    }

    /// Like `handle_player_turn_with_trace`, but also streams assistant narrative deltas.
    ///
    /// The callback is invoked only for strict `NARRATIVE:` generations, never for tool calls.
    pub fn handle_player_turn_streaming_with_trace<F>(
        &self,
        session_id: SessionId,
        player_id: PlayerId,
        player_text: &str,
        mut on_delta: F,
    ) -> GmResult<GmTurnOutcome>
    where
        F: FnMut(&str),
    {
        let mut st = self.get_or_create_state(session_id);
        let system_prompt = self.build_system_prompt(session_id, player_id);
        let auth = auth_gm();
        let mut tool_trace: Vec<(ToolCall, Value)> = Vec::new();

        let mut messages: Vec<ChatMessage> = vec![ChatMessage {
            role: ChatRole::User,
            content: player_text.to_string(),
        }];

        for _step in 0..self.max_tool_calls {
            let mut req = GenerateRequest::new(system_prompt.clone());
            req.messages = messages.clone();
			// Keep turns fast and deterministic by default.
			// On CPU backends, sampling (softmax/top-p) can be very slow for large vocabularies.
			req.sampling.temperature = 0.0; // Argmax
			req.sampling.top_p = 0.0;
            req.max_new_tokens = self.max_new_tokens;
            let llm_text = self.run_llm_once_streaming(st.llm_session, req, &mut on_delta)?;
            let reply = parse_gm_reply_strict(&llm_text)?;

            match reply {
                GmReply::Narrative(n) => {
                    if st.memory.summary.trim().is_empty() {
                        st.memory.summary = format!("Player: {}", player_text.trim());
                    } else {
                        st.memory.summary =
                            format!("{} | Player: {}", st.memory.summary, player_text.trim());
                    }
                    st.memory.add_fact(format!("last_player_said={}", player_text.trim()));
                    self.put_state(session_id, st);
                    return Ok(GmTurnOutcome {
                        narrative: n,
                        tool_trace,
                    });
                }
                GmReply::ToolCall(tc) => {
                    let mut tc = tc;
                    self.autofill_tool_arguments(&mut tc.arguments, session_id, player_id);

                    if !self.is_known_tool(&tc.name) {
                        let known: Vec<String> = self.router.list_tools().into_iter().map(|t| t.name).collect();
                        let out = serde_json::json!({
                            "error": "unknown tool",
                            "name": tc.name,
                            "known_tools": known,
                        });

                        messages.push(ChatMessage {
                            role: ChatRole::Assistant,
                            content: format!("TOOL_CALL: {}", serde_json::to_string(&tc).unwrap()),
                        });
                        tool_trace.push((tc.clone(), out.clone()));
                        messages.push(ChatMessage {
                            role: ChatRole::Tool,
                            content: format!("TOOL_RESULT: {}", out),
                        });
                        continue;
                    }

                    messages.push(ChatMessage {
                        role: ChatRole::Assistant,
                        content: format!("TOOL_CALL: {}", serde_json::to_string(&tc).unwrap()),
                    });

                    let out = self.call_tool_lossy(&auth, &tc.name, tc.arguments.clone());
                    Self::observe_tool_result_for_memory(&mut st.memory, &tc.name, &out);
                    tool_trace.push((tc.clone(), out.clone()));

                    if tc.name == TOOL_SEND_TO_PLAYERS {
                        if let Some(text) = tc.arguments.get("text").and_then(|v| v.as_str()) {
                            let text = text.trim();
                            if !text.is_empty() {
                                on_delta(text);
                                self.put_state(session_id, st);
                                return Ok(GmTurnOutcome {
                                    narrative: text.to_string(),
                                    tool_trace,
                                });
                            }
                        }
                    }

                    messages.push(ChatMessage {
                        role: ChatRole::Tool,
                        content: format!("TOOL_RESULT: {}", out),
                    });
                }
            }
        }

        Err(GmError::ToolLoopLimit)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cquests_core::LocationId;
    use cquests_mcp::{auth_player, TOOL_GET_VISIBLE_STATE, TOOL_MOVE};
    use cquests_server::InMemoryActionService;
    use std::sync::Arc;
    use cquests_llm::{TokenStream, TokenEvent, Usage, FinishReason};

    #[test]
    fn player_prompt_triggers_tool_and_state_change_then_narrative() {
        let service = InMemoryActionService::new();
        let session_id = service.create_session(
            123,
            InMemoryActionService::default_demo_world(PlayerId::new(1)),
        );
        let router = McpRouter::new(service.clone());

        let gm = GmAgent::new(cquests_llm::MockLlm::default(), router.clone());
        let narrative = gm
            .handle_player_turn(session_id, PlayerId::new(1), "Move to room B")
            .expect("gm turn");
        assert!(!narrative.trim().is_empty());

        // Verify the move happened.
        let auth = auth_player(1);
        let visible = router
            .call_tool(
                &auth,
                TOOL_GET_VISIBLE_STATE,
                serde_json::json!({"session_id": session_id, "player_id": 1}),
            )
            .unwrap();
        let visible: cquests_core::VisibleState = serde_json::from_value(visible).unwrap();
        assert_eq!(visible.location.id, LocationId::new(2));

        // Also ensure the tool itself is callable (sanity).
        let _ = router
            .call_tool(
                &auth,
                TOOL_MOVE,
                serde_json::json!({"session_id": session_id, "player_id": 1, "destination_id": 1}),
            )
            .unwrap();
    }

    #[test]
    fn tool_loop_is_bounded() {
        let service = InMemoryActionService::new();
        let session_id = service.create_session(
            123,
            InMemoryActionService::default_demo_world(PlayerId::new(1)),
        );
        let router = McpRouter::new(service);

        // With max_tool_calls=1, a prompt that (deterministically) triggers a tool call
        // and then another tool call should hit the limit.
        let gm = GmAgent::new(cquests_llm::MockLlm::default(), router).with_max_tool_calls(1);
        let err = gm
            .handle_player_turn(session_id, PlayerId::new(1), "tool_call tool_call")
            .unwrap_err();
        match err {
            GmError::ToolLoopLimit | GmError::Protocol(_) => {}
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn streaming_emits_deltas_for_narrative() {
        let service = InMemoryActionService::new();
        let session_id = service.create_session(
            123,
            InMemoryActionService::default_demo_world(PlayerId::new(1)),
        );
        let router = McpRouter::new(service);
        let gm = GmAgent::new(cquests_llm::MockLlm::default(), router);

        let mut streamed = String::new();
        let out = gm
            .handle_player_turn_streaming_with_trace(session_id, PlayerId::new(1), "Hello", |d| {
                streamed.push_str(d);
            })
            .expect("streaming gm turn");

        assert!(!out.narrative.trim().is_empty());
        assert!(!streamed.trim().is_empty());
        // The streamed text is derived from the narrative body; allow minor trimming differences.
        assert!(out.narrative.contains(streamed.trim()) || streamed.trim().contains(out.narrative.trim()));
    }

    struct CaptureReqLlm {
        last_max_new_tokens: Arc<parking_lot::Mutex<Option<u32>>>,
    }

    impl CaptureReqLlm {
        fn new(last_max_new_tokens: Arc<parking_lot::Mutex<Option<u32>>>) -> Self {
            Self { last_max_new_tokens }
        }
    }

    impl LlmClient for CaptureReqLlm {
        fn create_session(&self) -> LlmSessionHandle {
            LlmSessionHandle(1)
        }

        fn reset_session(&self, _session: LlmSessionHandle) -> cquests_llm::LlmResult<()> {
            Ok(())
        }

        fn abort(&self, _session: LlmSessionHandle) -> cquests_llm::LlmResult<()> {
            Ok(())
        }

        fn generate_stream(
            &self,
            _session: LlmSessionHandle,
            req: GenerateRequest,
        ) -> cquests_llm::LlmResult<TokenStream> {
            *self.last_max_new_tokens.lock() = Some(req.max_new_tokens);

            let (tx, rx) = crossbeam_channel::unbounded::<TokenEvent>();
            // Minimal strict narrative response.
            let _ = tx.send(TokenEvent::TokenChunk {
                text: "NARRATIVE: ok".into(),
            });
            let _ = tx.send(TokenEvent::Finished {
                usage: Usage {
                    prompt_tokens: 0,
                    completion_tokens: 0,
                },
                reason: FinishReason::Stop,
            });
            Ok(TokenStream::from_receiver(rx))
        }
    }

    #[test]
    fn gm_passes_configured_max_new_tokens_to_llm() {
        let service = InMemoryActionService::new();
        let session_id = service.create_session(
            123,
            InMemoryActionService::default_demo_world(PlayerId::new(1)),
        );
        let router = McpRouter::new(service);

        let last = Arc::new(parking_lot::Mutex::new(None));
        let llm = CaptureReqLlm::new(last.clone());

        let gm = GmAgent::new(llm, router).with_max_new_tokens(512);
        let _ = gm.handle_player_turn(session_id, PlayerId::new(1), "Hello").unwrap();

        assert_eq!(*last.lock(), Some(512));
    }
}
