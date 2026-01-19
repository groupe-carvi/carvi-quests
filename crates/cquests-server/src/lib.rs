//! Authoritative game-server primitives.
//!
//! An in-memory `ActionService` that owns
//! sessions, validates permissions, applies Actions deterministically via
//! `cquests-engine`, and stores an append-only event log.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::thread;
use std::time::Duration;

use cquests_core::{
    make_visible_state, Action, ActionFailure, ActionResult, DiceExpr, Entity, EntityId, ErrorCode,
    Event, Location, LocationId, Player, PlayerId, Seed, SessionId, VisibleState, WorldState,
};
use cquests_engine::{ApplyOutcome, Engine};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json;
use thiserror::Error;

// ---- Persistence & scheduling ----

/// A deterministic snapshot of a session.
///
/// We persist the *action log* (not the RNG state) so we can restore the engine's
/// RNG progression by replaying actions from the initial world state.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct SessionSnapshot {
    pub session_id: SessionId,
    pub seed: Seed,
    pub initial: WorldState,
    pub actions: Vec<Action>,
}

pub trait SessionStore: Send + Sync {
    fn save(&self, snapshot: &SessionSnapshot) -> std::io::Result<()>;
    fn load(&self, session_id: SessionId) -> std::io::Result<SessionSnapshot>;
}

/// JSON file store for snapshots: `sessions/session_{id}.json`.
#[derive(Clone, Debug)]
pub struct JsonFileSessionStore {
    base_dir: PathBuf,
}

impl JsonFileSessionStore {
    pub fn new(base_dir: impl Into<PathBuf>) -> Self {
        Self {
            base_dir: base_dir.into(),
        }
    }

    fn file_path(&self, session_id: SessionId) -> PathBuf {
        self.base_dir
            .join(format!("session_{}.json", session_id.get()))
    }

    fn ensure_dir(&self) -> std::io::Result<()> {
        std::fs::create_dir_all(&self.base_dir)
    }
}

impl SessionStore for JsonFileSessionStore {
    fn save(&self, snapshot: &SessionSnapshot) -> std::io::Result<()> {
        self.ensure_dir()?;
        let path = self.file_path(snapshot.session_id);
        let json = serde_json::to_string_pretty(snapshot)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
        std::fs::write(path, json)
    }

    fn load(&self, session_id: SessionId) -> std::io::Result<SessionSnapshot> {
        let path = self.file_path(session_id);
        let s = std::fs::read_to_string(path)?;
        serde_json::from_str(&s)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))
    }
}

/// A simple autosave loop handle.
pub struct AutosaveHandle {
    stop: Arc<AtomicBool>,
    join: Option<thread::JoinHandle<()>>,
}

impl AutosaveHandle {
    pub fn stop(mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(j) = self.join.take() {
            let _ = j.join();
        }
    }
}

impl Drop for AutosaveHandle {
	fn drop(&mut self) {
		self.stop.store(true, Ordering::Relaxed);
		if let Some(j) = self.join.take() {
			let _ = j.join();
		}
	}
}

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
    initial: WorldState,
    actions: Vec<Action>,
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
        session.actions.push(action.clone());
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
                initial: initial.clone(),
                actions: Vec::new(),
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
        // Rightnow we allow:
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

impl InMemoryActionService {
    /// Export a deterministic snapshot (seed + initial world + actions).
    pub fn export_snapshot(&self, session_id: SessionId) -> ServiceResult<SessionSnapshot> {
        let inner = self.inner.lock();
        Self::with_session(&inner, session_id, |s| {
            Ok(SessionSnapshot {
                session_id,
                seed: s.seed,
                initial: s.initial.clone(),
                actions: s.actions.clone(),
            })
        })
    }

    /// Restore (or overwrite) a session from snapshot. Returns the snapshot session id.
    ///
    /// This replays actions from the stored initial state to restore:
    /// - authoritative world state
    /// - deterministic RNG progression inside the engine
    /// - event log
    pub fn restore_snapshot(&self, snapshot: SessionSnapshot) -> ServiceResult<SessionId> {
        let mut engine = Engine::new(snapshot.seed);
        let mut state = snapshot.initial.clone();
        let mut events = Vec::new();
        for a in snapshot.actions.iter().cloned() {
            let out = engine.apply(&mut state, a);
            events.extend(out.events);
        }

        let mut inner = self.inner.lock();
        // Ensure future auto-assigned ids don't collide.
        inner.next_session = inner
            .next_session
            .max(snapshot.session_id.get().saturating_add(1));

        inner.sessions.insert(
            snapshot.session_id,
            Session {
                seed: snapshot.seed,
                engine,
                initial: snapshot.initial,
                actions: snapshot.actions,
                state,
                events,
            },
        );

        Ok(snapshot.session_id)
    }

    /// Helper: create sessions dir store rooted at `./sessions`.
    pub fn default_store_dir() -> PathBuf {
        Path::new("sessions").to_path_buf()
    }

    /// Start an autosave thread for a single session.
    pub fn start_autosave(
        &self,
        session_id: SessionId,
        store: JsonFileSessionStore,
        interval: Duration,
    ) -> AutosaveHandle {
        let stop = Arc::new(AtomicBool::new(false));
        let stop2 = stop.clone();
        let service = self.clone();
        let join = thread::spawn(move || {
            while !stop2.load(Ordering::Relaxed) {
                thread::sleep(interval);
                let snapshot = match service.export_snapshot(session_id) {
                    Ok(s) => s,
                    Err(_) => continue,
                };
                let _ = store.save(&snapshot);
            }
        });
        AutosaveHandle {
            stop,
            join: Some(join),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapshot_roundtrip_restores_identical_state_and_events() {
        let service = InMemoryActionService::new();
        let player_id = PlayerId::new(1);
        let session_id = service.create_session(123, InMemoryActionService::default_demo_world(player_id));

        // Drive a mix of actions (including RNG-driven ones) so replay matters.
        let _ = service.tool_roll(&AuthContext::gm(), session_id, "d20").unwrap();
        let _ = service
            .tool_attack(&AuthContext::player(player_id), session_id, player_id, EntityId::new(11))
            .unwrap();
        let _ = service
            .tool_move(&AuthContext::player(player_id), session_id, player_id, LocationId::new(2))
            .unwrap();

        let snap1 = service.export_snapshot(session_id).unwrap();
        let vs1 = service
            .get_visible_state(&AuthContext::player(player_id), session_id, player_id)
            .unwrap();
        let ev1 = service.get_recent_events(&AuthContext::gm(), session_id, 200).unwrap();

        let service2 = InMemoryActionService::new();
        let restored_id = service2.restore_snapshot(snap1.clone()).unwrap();
        assert_eq!(restored_id, session_id);

        let vs2 = service2
            .get_visible_state(&AuthContext::player(player_id), restored_id, player_id)
            .unwrap();
        let ev2 = service2.get_recent_events(&AuthContext::gm(), restored_id, 200).unwrap();
        let snap2 = service2.export_snapshot(restored_id).unwrap();

        assert_eq!(vs1, vs2);
        assert_eq!(ev1, ev2);
        assert_eq!(snap1, snap2);
    }
}
