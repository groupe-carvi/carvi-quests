//! Authoritative game-server primitives.
//!
//! Phase 2 scope (MCP layer support): an in-memory `ActionService` that owns
//! sessions, validates permissions, applies Actions deterministically via
//! `cquests-engine`, and stores an append-only event log.

use std::collections::HashMap;
use std::sync::Arc;

use cquests_core::{
    make_visible_state, Action, ActionFailure, ActionResult, DiceExpr, Entity, EntityId, ErrorCode,
    Event, Location, LocationId, Player, PlayerId, Seed, SessionId, VisibleState, WorldState,
};
use cquests_engine::{ApplyOutcome, Engine};
use parking_lot::Mutex;
use thiserror::Error;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Role {
    Player(PlayerId),
    Gm,
}

#[derive(Clone, Debug)]
pub struct AuthContext {
    pub role: Role,
}

impl AuthContext {
    pub fn player(player_id: PlayerId) -> Self {
        Self {
            role: Role::Player(player_id),
        }
    }

    pub fn gm() -> Self {
        Self { role: Role::Gm }
    }
}

#[derive(Debug, Error)]
pub enum ServiceError {
    #[error("not found: {0}")]
    NotFound(&'static str),

    #[error("forbidden: {0}")]
    Forbidden(&'static str),

    #[error("invalid input: {0}")]
    InvalidInput(String),

    #[error("action failure: {0:?}")]
    ActionFailure(ActionFailure),
}

pub type ServiceResult<T> = Result<T, ServiceError>;

/// Thin interface so `cquests-mcp` can proxy without depending on internal details.
pub trait ActionService: Send + Sync {
    fn create_session(&self, seed: Seed, initial: WorldState) -> SessionId;

    fn get_visible_state(
        &self,
        auth: &AuthContext,
        session_id: SessionId,
        player_id: PlayerId,
    ) -> ServiceResult<VisibleState>;

    fn get_recent_events(
        &self,
        auth: &AuthContext,
        session_id: SessionId,
        limit: usize,
    ) -> ServiceResult<Vec<Event>>;

    fn tool_roll(
        &self,
        auth: &AuthContext,
        session_id: SessionId,
        dice_expr: &str,
    ) -> ServiceResult<(ActionResult, Vec<Event>)>;

    fn tool_move(
        &self,
        auth: &AuthContext,
        session_id: SessionId,
        player_id: PlayerId,
        destination: LocationId,
    ) -> ServiceResult<(ActionResult, Vec<Event>)>;

    fn tool_inspect_entity(
        &self,
        auth: &AuthContext,
        session_id: SessionId,
        player_id: PlayerId,
        target: EntityId,
    ) -> ServiceResult<(ActionResult, Vec<Event>)>;

    fn tool_inspect_location(
        &self,
        auth: &AuthContext,
        session_id: SessionId,
        player_id: PlayerId,
        target: LocationId,
    ) -> ServiceResult<(ActionResult, Vec<Event>)>;

    fn tool_attack(
        &self,
        auth: &AuthContext,
        session_id: SessionId,
        player_id: PlayerId,
        target: EntityId,
    ) -> ServiceResult<(ActionResult, Vec<Event>)>;
}

#[derive(Clone)]
pub struct InMemoryActionService {
    inner: Arc<Mutex<Inner>>,
}

struct Inner {
    next_session: u64,
    sessions: HashMap<SessionId, Session>,
}

struct Session {
    seed: Seed,
    engine: Engine,
    state: WorldState,
    events: Vec<Event>,
}

impl InMemoryActionService {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(Inner {
                next_session: 1,
                sessions: HashMap::new(),
            })),
        }
    }

    pub fn default_demo_world(player_id: PlayerId) -> WorldState {
        // Simple two-room world with a player and a goblin in room A.
        let a = LocationId::new(1);
        let b = LocationId::new(2);

        WorldState {
            turn: 0,
            locations: vec![
                Location {
                    id: a,
                    name: "A".into(),
                    neighbors: vec![b],
                },
                Location {
                    id: b,
                    name: "B".into(),
                    neighbors: vec![a],
                },
            ],
            entities: vec![
                Entity {
                    id: EntityId::new(10),
                    name: "Hero".into(),
                    location: a,
                    hp: 10,
                    max_hp: 10,
                    attack_bonus: 3,
                    armor_class: 12,
                },
                Entity {
                    id: EntityId::new(11),
                    name: "Goblin".into(),
                    location: a,
                    hp: 6,
                    max_hp: 6,
                    attack_bonus: 2,
                    armor_class: 11,
                },
            ],
            players: vec![Player {
                id: player_id,
                name: "Player".into(),
                entity_id: EntityId::new(10),
            }],
        }
    }

    fn with_session_mut<T>(
        inner: &mut Inner,
        session_id: SessionId,
        f: impl FnOnce(&mut Session) -> ServiceResult<T>,
    ) -> ServiceResult<T> {
        let s = inner
            .sessions
            .get_mut(&session_id)
            .ok_or(ServiceError::NotFound("session"))?;
        f(s)
    }

    fn with_session<T>(
        inner: &Inner,
        session_id: SessionId,
        f: impl FnOnce(&Session) -> ServiceResult<T>,
    ) -> ServiceResult<T> {
        let s = inner
            .sessions
            .get(&session_id)
            .ok_or(ServiceError::NotFound("session"))?;
        f(s)
    }

    fn require_player_identity(auth: &AuthContext, player_id: PlayerId) -> ServiceResult<()> {
        match auth.role {
            Role::Gm => Ok(()),
            Role::Player(pid) if pid == player_id => Ok(()),
            Role::Player(_) => Err(ServiceError::Forbidden(
                "player role cannot act for another player",
            )),
        }
    }

    fn player_entity_id(state: &WorldState, player_id: PlayerId) -> ServiceResult<EntityId> {
        let player = state
            .players
            .iter()
            .find(|p| p.id == player_id)
            .ok_or(ServiceError::NotFound("player"))?;
        Ok(player.entity_id)
    }

    fn apply_and_record(session: &mut Session, action: Action) -> (ActionResult, Vec<Event>) {
        let ApplyOutcome { result, events } = session.engine.apply(&mut session.state, action);
        session.events.extend(events.clone());
        (result, events)
    }

    fn action_failure(code: ErrorCode, message: impl Into<String>) -> ActionFailure {
        ActionFailure {
            code,
            message: message.into(),
        }
    }
}

impl Default for InMemoryActionService {
    fn default() -> Self {
        Self::new()
    }
}

impl ActionService for InMemoryActionService {
    fn create_session(&self, seed: Seed, initial: WorldState) -> SessionId {
        let mut inner = self.inner.lock();
        let id = SessionId::new(inner.next_session);
        inner.next_session += 1;

        inner.sessions.insert(
            id,
            Session {
                seed,
                engine: Engine::new(seed),
                state: initial,
                events: Vec::new(),
            },
        );

        id
    }

    fn get_visible_state(
        &self,
        auth: &AuthContext,
        session_id: SessionId,
        player_id: PlayerId,
    ) -> ServiceResult<VisibleState> {
        Self::require_player_identity(auth, player_id)?;
        let inner = self.inner.lock();
        Self::with_session(&inner, session_id, |s| {
            make_visible_state(&s.state, player_id).ok_or(ServiceError::NotFound("visible_state"))
        })
    }

    fn get_recent_events(
        &self,
        auth: &AuthContext,
        session_id: SessionId,
        limit: usize,
    ) -> ServiceResult<Vec<Event>> {
        // For Phase 2 we allow:
        // - GM: always
        // - Player: only if that player exists in the session
        let inner = self.inner.lock();
        Self::with_session(&inner, session_id, |s| {
            match auth.role {
                Role::Gm => {}
                Role::Player(pid) => {
                    let _ = Self::player_entity_id(&s.state, pid)?;
                }
            }

            let cap = limit.clamp(0, 200);
            let start = s.events.len().saturating_sub(cap);
            Ok(s.events[start..].to_vec())
        })
    }

    fn tool_roll(
        &self,
        _auth: &AuthContext,
        session_id: SessionId,
        dice_expr: &str,
    ) -> ServiceResult<(ActionResult, Vec<Event>)> {
        let dice = DiceExpr::parse(dice_expr)
            .map_err(|_| ServiceError::InvalidInput("invalid dice".into()))?;

        let mut inner = self.inner.lock();
        Self::with_session_mut(&mut inner, session_id, |s| {
            Ok(Self::apply_and_record(s, Action::Roll { dice }))
        })
    }

    fn tool_move(
        &self,
        auth: &AuthContext,
        session_id: SessionId,
        player_id: PlayerId,
        destination: LocationId,
    ) -> ServiceResult<(ActionResult, Vec<Event>)> {
        Self::require_player_identity(auth, player_id)?;

        let mut inner = self.inner.lock();
        Self::with_session_mut(&mut inner, session_id, |s| {
            let entity_id = Self::player_entity_id(&s.state, player_id)?;
            Ok(Self::apply_and_record(
                s,
                Action::Move {
                    entity_id,
                    destination,
                },
            ))
        })
    }

    fn tool_inspect_entity(
        &self,
        auth: &AuthContext,
        session_id: SessionId,
        player_id: PlayerId,
        target: EntityId,
    ) -> ServiceResult<(ActionResult, Vec<Event>)> {
        Self::require_player_identity(auth, player_id)?;

        let mut inner = self.inner.lock();
        Self::with_session_mut(&mut inner, session_id, |s| {
            let entity_id = Self::player_entity_id(&s.state, player_id)?;
            Ok(Self::apply_and_record(
                s,
                Action::Inspect {
                    entity_id,
                    target: cquests_core::InspectTarget::Entity(target),
                },
            ))
        })
    }

    fn tool_inspect_location(
        &self,
        auth: &AuthContext,
        session_id: SessionId,
        player_id: PlayerId,
        target: LocationId,
    ) -> ServiceResult<(ActionResult, Vec<Event>)> {
        Self::require_player_identity(auth, player_id)?;

        let mut inner = self.inner.lock();
        Self::with_session_mut(&mut inner, session_id, |s| {
            let entity_id = Self::player_entity_id(&s.state, player_id)?;
            Ok(Self::apply_and_record(
                s,
                Action::Inspect {
                    entity_id,
                    target: cquests_core::InspectTarget::Location(target),
                },
            ))
        })
    }

    fn tool_attack(
        &self,
        auth: &AuthContext,
        session_id: SessionId,
        player_id: PlayerId,
        target: EntityId,
    ) -> ServiceResult<(ActionResult, Vec<Event>)> {
        Self::require_player_identity(auth, player_id)?;

        let mut inner = self.inner.lock();
        Self::with_session_mut(&mut inner, session_id, |s| {
            let attacker = Self::player_entity_id(&s.state, player_id)?;
            if attacker == target {
                return Ok((
                    ActionResult::Failure(Self::action_failure(
                        ErrorCode::InvalidMove,
                        "cannot attack self",
                    )),
                    vec![],
                ));
            }
            Ok(Self::apply_and_record(
                s,
                Action::Attack {
                    attacker,
                    target,
                },
            ))
        })
    }
}
