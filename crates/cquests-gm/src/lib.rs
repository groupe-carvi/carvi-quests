//! GM agent (Phase 4): strict tool-call protocol + bounded tool loop.
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
use cquests_mcp::{auth_gm, McpRouter, PROMPT_GM_TURN};
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

/// Phase 4 GM agent.
///
/// Generic over the server implementation so we can test with `InMemoryActionService`.
pub struct GmAgent<L: LlmClient, S: ActionService> {
    llm: L,
    router: McpRouter<S>,
    state: Mutex<HashMap<SessionId, GmSessionState>>,
    max_tool_calls: usize,
}

impl<L: LlmClient, S: ActionService> GmAgent<L, S> {
    pub fn new(llm: L, router: McpRouter<S>) -> Self {
        Self {
            llm,
            router,
            state: Mutex::new(HashMap::new()),
            max_tool_calls: 4,
        }
    }

    pub fn with_max_tool_calls(mut self, n: usize) -> Self {
        self.max_tool_calls = n.max(1);
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
        p.push_str("If you need information or state changes, call a tool.\n\n");

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

    fn observe_tool_result_for_memory(mem: &mut MemoryScaffold, tool_name: &str, tool_result: &Value) {
        mem.add_fact(format!("last_tool={tool_name}"));

        // Keep this intentionally minimal/deterministic for Phase 4.
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
                    // Record tool call (as assistant) and execute.
                    messages.push(ChatMessage {
                        role: ChatRole::Assistant,
                        content: format!("TOOL_CALL: {}", serde_json::to_string(&tc).unwrap()),
                    });

                    let out = self.router.call_tool(&auth, &tc.name, tc.arguments.clone())?;
                    Self::observe_tool_result_for_memory(&mut st.memory, &tc.name, &out);
                    tool_trace.push((tc.clone(), out.clone()));

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
}
