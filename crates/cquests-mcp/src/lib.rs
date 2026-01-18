//! MCP layer (Phase 2): tools/resources/prompts.
//!
//! This crate intentionally models MCP concepts in a transport-agnostic way.
//! A later phase can expose the same router over stdio / websocket / HTTP.

use cquests_core::{EntityId, LocationId, PlayerId, SessionId};
use cquests_server::{ActionService, AuthContext, Role, ServiceError};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;

pub const TOOL_GET_VISIBLE_STATE: &str = "get_visible_state";
pub const TOOL_ROLL: &str = "roll";
pub const TOOL_MOVE: &str = "move";
pub const TOOL_INSPECT: &str = "inspect";
pub const TOOL_ATTACK: &str = "attack";

pub const PROMPT_GM_TURN: &str = "gm_turn";
pub const PROMPT_NPC_VOICE: &str = "npc_voice";
pub const PROMPT_SUMMARIZE_SESSION: &str = "summarize_session";

#[derive(Debug, Error)]
pub enum McpError {
    #[error("unknown tool: {0}")]
    UnknownTool(String),

    #[error("unknown resource: {0}")]
    UnknownResource(String),

    #[error("unknown prompt: {0}")]
    UnknownPrompt(String),

    #[error("invalid arguments: {0}")]
    InvalidArguments(String),

    #[error("service error: {0}")]
    Service(#[from] ServiceError),
}

pub type McpResult<T> = Result<T, McpError>;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ToolDescriptor {
    pub name: String,
    pub description: String,
}

#[derive(Clone)]
pub struct McpRouter<S: ActionService> {
    service: S,
}

impl<S: ActionService> McpRouter<S> {
    pub fn new(service: S) -> Self {
        Self { service }
    }

    pub fn list_tools(&self) -> Vec<ToolDescriptor> {
        vec![
            ToolDescriptor {
                name: TOOL_GET_VISIBLE_STATE.into(),
                description: "Get per-player filtered visible state".into(),
            },
            ToolDescriptor {
                name: TOOL_ROLL.into(),
                description: "Roll dice using server-owned deterministic RNG".into(),
            },
            ToolDescriptor {
                name: TOOL_MOVE.into(),
                description: "Move player to an adjacent destination".into(),
            },
            ToolDescriptor {
                name: TOOL_INSPECT.into(),
                description: "Inspect an entity or your current location".into(),
            },
            ToolDescriptor {
                name: TOOL_ATTACK.into(),
                description: "Attack a target entity (co-located)".into(),
            },
        ]
    }

    /// Call a tool by name with JSON arguments.
    pub fn call_tool(&self, auth: &AuthContext, name: &str, args: Value) -> McpResult<Value> {
        match name {
            TOOL_GET_VISIBLE_STATE => {
                let a: GetVisibleStateArgs = serde_json::from_value(args)
                    .map_err(|e| McpError::InvalidArguments(e.to_string()))?;
                let visible = self
                    .service
                    .get_visible_state(auth, a.session_id, a.player_id)?;
                Ok(serde_json::to_value(visible).expect("serializable"))
            }
            TOOL_ROLL => {
                let a: RollArgs = serde_json::from_value(args)
                    .map_err(|e| McpError::InvalidArguments(e.to_string()))?;
                let (result, events) = self
                    .service
                    .tool_roll(auth, a.session_id, a.dice_expr.as_str())?;
                Ok(serde_json::json!({"result": result, "events": events}))
            }
            TOOL_MOVE => {
                let a: MoveArgs = serde_json::from_value(args)
                    .map_err(|e| McpError::InvalidArguments(e.to_string()))?;
                let (result, events) = self.service.tool_move(
                    auth,
                    a.session_id,
                    a.player_id,
                    LocationId::new(a.destination_id),
                )?;
                Ok(serde_json::json!({"result": result, "events": events}))
            }
            TOOL_INSPECT => {
                let a: InspectArgs = serde_json::from_value(args)
                    .map_err(|e| McpError::InvalidArguments(e.to_string()))?;

                let (result, events) = match a.target {
                    InspectTargetArgs::Entity { id } => self.service.tool_inspect_entity(
                        auth,
                        a.session_id,
                        a.player_id,
                        EntityId::new(id),
                    )?,
                    InspectTargetArgs::Location { id } => self.service.tool_inspect_location(
                        auth,
                        a.session_id,
                        a.player_id,
                        LocationId::new(id),
                    )?,
                };
                Ok(serde_json::json!({"result": result, "events": events}))
            }
            TOOL_ATTACK => {
                let a: AttackArgs = serde_json::from_value(args)
                    .map_err(|e| McpError::InvalidArguments(e.to_string()))?;
                let (result, events) = self.service.tool_attack(
                    auth,
                    a.session_id,
                    a.player_id,
                    EntityId::new(a.target_id),
                )?;
                Ok(serde_json::json!({"result": result, "events": events}))
            }
            other => Err(McpError::UnknownTool(other.into())),
        }
    }

    /// Read a resource by URI.
    ///
    /// Implemented in Phase 2:
    /// - `world://lore/overview`
    /// - `session://{session_id}/visible/{player_id}`
    /// - `session://{session_id}/recent_events?limit=N`
    pub fn read_resource(&self, auth: &AuthContext, uri: &str) -> McpResult<Value> {
        if uri == "world://lore/overview" {
            return Ok(serde_json::json!({
                "title": "CQuests (placeholder lore)",
                "text": "A small realm of deterministic adventures. Real lore arrives in Phase 5."
            }));
        }

        if let Some(rest) = uri.strip_prefix("session://") {
            // rest = "{session_id}/..." OR "{session_id}/...?..."
            let (sid_s, remainder) = rest
                .split_once('/')
                .ok_or_else(|| McpError::UnknownResource(uri.into()))?;
            let sid: u64 = sid_s
                .parse()
                .map_err(|_| McpError::UnknownResource(uri.into()))?;

            // remainder starts with e.g. "visible/1" or "recent_events?limit=50"
            let (next_raw, tail) = match remainder.split_once('/') {
                Some((a, b)) => (a, b),
                None => (remainder, ""),
            };

            // Allow query string on the first path segment, e.g. "recent_events?limit=50".
            let (next, query_from_next) = match next_raw.split_once('?') {
                Some((p, q)) => (p, Some(q)),
                None => (next_raw, None),
            };

            if next == "visible" {
                let pid: u64 = tail
                    .parse()
                    .map_err(|_| McpError::UnknownResource(uri.into()))?;
                let visible = self.service.get_visible_state(
                    auth,
                    SessionId::new(sid),
                    PlayerId::new(pid),
                )?;
                return Ok(serde_json::to_value(visible).expect("serializable"));
            }

            if next == "recent_events" {
                // Accept:
                // - session://{sid}/recent_events/
                // - session://{sid}/recent_events/?limit=50
                // - session://{sid}/recent_events?limit=50  (legacy-ish)
                let (path, query) = if let Some(q) = query_from_next {
                    (tail, Some(q))
                } else {
                    match tail.split_once('?') {
                        Some((p, q)) => (p, Some(q)),
                        None => (tail, None),
                    }
                };

                if !path.is_empty() && path != "/" {
                    return Err(McpError::UnknownResource(uri.into()));
                }

                let mut limit = 50usize;
                if let Some(q) = query {
                    for kv in q.split('&') {
                        if let Some((k, v)) = kv.split_once('=') {
                            if k == "limit" {
                                if let Ok(n) = v.parse::<usize>() {
                                    limit = n;
                                }
                            }
                        }
                    }
                }

                let events = self.service.get_recent_events(auth, SessionId::new(sid), limit)?;
                return Ok(serde_json::to_value(events).expect("serializable"));
            }

            return Err(McpError::UnknownResource(uri.into()));
        }

        Err(McpError::UnknownResource(uri.into()))
    }

    /// Return prompt templates by name.
    pub fn get_prompt(&self, name: &str) -> McpResult<String> {
        match name {
            PROMPT_GM_TURN => Ok(
                "You are the Game Master. Output either:\n\
NARRATIVE: ...\n\
OR\n\
TOOL_CALL: {\"name\":...,\"arguments\":{...}}\n\
Never claim outcomes not returned by tools."
                    .into(),
            ),
            PROMPT_NPC_VOICE => Ok(
                "Speak as the NPC. Keep it short, in-character, and consistent with known facts.".into(),
            ),
            PROMPT_SUMMARIZE_SESSION => Ok(
                "Summarize the session into: (1) short recap, (2) facts list, (3) open threads.".into(),
            ),
            other => Err(McpError::UnknownPrompt(other.into())),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GetVisibleStateArgs {
    pub session_id: SessionId,
    pub player_id: PlayerId,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RollArgs {
    pub session_id: SessionId,
    pub dice_expr: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MoveArgs {
    pub session_id: SessionId,
    pub player_id: PlayerId,
    pub destination_id: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InspectArgs {
    pub session_id: SessionId,
    pub player_id: PlayerId,
    pub target: InspectTargetArgs,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum InspectTargetArgs {
    Entity { id: u64 },
    Location { id: u64 },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AttackArgs {
    pub session_id: SessionId,
    pub player_id: PlayerId,
    pub target_id: u64,
}

/// Convenience helper for tests.
pub fn auth_player(player_id: u64) -> AuthContext {
    AuthContext {
        role: Role::Player(PlayerId::new(player_id)),
    }
}

/// Convenience helper for tests.
pub fn auth_gm() -> AuthContext {
    AuthContext { role: Role::Gm }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cquests_server::InMemoryActionService;
    use pretty_assertions::assert_eq;

    #[test]
    fn player_can_call_tools_for_self() {
        let service = InMemoryActionService::new();
        let session_id = service.create_session(
            123,
            InMemoryActionService::default_demo_world(PlayerId::new(1)),
        );

        let router = McpRouter::new(service);
        let auth = auth_player(1);

        // get_visible_state
        let v = router
            .call_tool(
                &auth,
                TOOL_GET_VISIBLE_STATE,
                serde_json::json!({"session_id": session_id, "player_id": 1}),
            )
            .unwrap();
        let visible: cquests_core::VisibleState = serde_json::from_value(v).unwrap();
        assert_eq!(visible.player_id, PlayerId::new(1));

        // move
        let out = router
            .call_tool(
                &auth,
                TOOL_MOVE,
                serde_json::json!({
                    "session_id": session_id,
                    "player_id": 1,
                    "destination_id": 2
                }),
            )
            .unwrap();
        assert!(out.get("result").is_some());

        // resource: session visible
        let v2 = router
            .read_resource(&auth, &format!("session://{}/visible/1", session_id.get()))
            .unwrap();
        let visible2: cquests_core::VisibleState = serde_json::from_value(v2).unwrap();
        assert_eq!(visible2.player_id, PlayerId::new(1));
    }

    #[test]
    fn player_cannot_act_for_other_player() {
        let service = InMemoryActionService::new();
        let session_id = service.create_session(
            123,
            InMemoryActionService::default_demo_world(PlayerId::new(1)),
        );
        let router = McpRouter::new(service);
        let auth = auth_player(1);

        let err = router
            .call_tool(
                &auth,
                TOOL_MOVE,
                serde_json::json!({
                    "session_id": session_id,
                    "player_id": 2,
                    "destination_id": 2
                }),
            )
            .unwrap_err();

        match err {
            McpError::Service(ServiceError::Forbidden(_)) => {}
            other => panic!("expected forbidden, got {other:?}"),
        }
    }

    #[test]
    fn gm_can_read_any_visible_state() {
        let service = InMemoryActionService::new();
        let session_id = service.create_session(
            123,
            InMemoryActionService::default_demo_world(PlayerId::new(1)),
        );
        let router = McpRouter::new(service);
        let auth = auth_gm();

        let v = router
            .read_resource(&auth, &format!("session://{}/visible/1", session_id.get()))
            .unwrap();
        let visible: cquests_core::VisibleState = serde_json::from_value(v).unwrap();
        assert_eq!(visible.player_id, PlayerId::new(1));
    }

    #[test]
    fn recent_events_resource_is_bounded() {
        let service = InMemoryActionService::new();
        let session_id = service.create_session(
            123,
            InMemoryActionService::default_demo_world(PlayerId::new(1)),
        );
        let router = McpRouter::new(service);
        let auth = auth_player(1);

        // Generate some events.
        for _ in 0..10 {
            let _ = router.call_tool(
                &auth,
                TOOL_ROLL,
                serde_json::json!({"session_id": session_id, "dice_expr": "d20"}),
            );
        }

        let events_val = router
            .read_resource(
                &auth,
                &format!("session://{}/recent_events?limit=3", session_id.get()),
            )
            .unwrap();

        let events: Vec<cquests_core::Event> = serde_json::from_value(events_val).unwrap();
        assert!(events.len() <= 3);
    }
}
