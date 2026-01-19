//! Core, stable contracts for CQuests.
//!
//! Current state: deterministic world state primitives, actions, events, and
//! validation-friendly error codes. No networking, no UI.

use serde::{Deserialize, Serialize};
use thiserror::Error;

pub type Seed = u64;

macro_rules! id_newtype {
    ($name:ident) => {
        #[derive(
            Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
        )]
        pub struct $name(pub u64);

        impl $name {
            pub const fn new(value: u64) -> Self {
                Self(value)
            }

            pub const fn get(self) -> u64 {
                self.0
            }
        }
    };
}

id_newtype!(SessionId);
id_newtype!(PlayerId);
id_newtype!(EntityId);
id_newtype!(LocationId);
id_newtype!(ItemId);
id_newtype!(QuestId);

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Location {
    pub id: LocationId,
    pub name: String,
    /// Adjacent locations (graph edges).
    pub neighbors: Vec<LocationId>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Entity {
    pub id: EntityId,
    pub name: String,
    pub location: LocationId,
    pub hp: i32,
    pub max_hp: i32,
    pub attack_bonus: i32,
    pub armor_class: i32,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Player {
    pub id: PlayerId,
    pub name: String,
    /// The in-world entity controlled by the player.
    pub entity_id: EntityId,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct WorldState {
    /// Monotonic turn counter advanced by the server/engine.
    pub turn: u64,
    pub locations: Vec<Location>,
    pub entities: Vec<Entity>,
    pub players: Vec<Player>,
}

impl WorldState {
    pub fn find_location(&self, id: LocationId) -> Option<&Location> {
        self.locations.iter().find(|l| l.id == id)
    }

    pub fn find_entity(&self, id: EntityId) -> Option<&Entity> {
        self.entities.iter().find(|e| e.id == id)
    }

    pub fn find_entity_mut(&mut self, id: EntityId) -> Option<&mut Entity> {
        self.entities.iter_mut().find(|e| e.id == id)
    }

    pub fn is_alive(&self, id: EntityId) -> bool {
        self.find_entity(id).is_some_and(|e| e.hp > 0)
    }
}

/// Player-specific filtered projection of `WorldState`.
///
/// TODO: can hide fog-of-war, secrets, etc.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct VisibleState {
    pub turn: u64,
    pub player_id: PlayerId,
    pub self_entity: Entity,
    pub location: Location,
    pub co_located_entities: Vec<Entity>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum Action {
    /// Deterministic dice roll driven by engine-owned RNG.
    Roll { dice: DiceExpr },
    Move {
        entity_id: EntityId,
        destination: LocationId,
    },
    Inspect {
        entity_id: EntityId,
        target: InspectTarget,
    },
    Attack {
        attacker: EntityId,
        target: EntityId,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum InspectTarget {
    Entity(EntityId),
    Location(LocationId),
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum ActionResult {
    Success(ActionOutcome),
    Failure(ActionFailure),
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum ActionOutcome {
    Rolled { total: i32 },
    Moved { from: LocationId, to: LocationId },
    InspectedEntity { entity: Entity },
    InspectedLocation { location: Location },
    Attacked {
        hit: bool,
        attack_roll: i32,
        damage: i32,
        target_hp_after: i32,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ActionFailure {
    pub code: ErrorCode,
    pub message: String,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum ErrorCode {
    NotFound,
    NotAlive,
    InvalidMove,
    OutOfRange,
    InvalidDice,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum Event {
    TurnAdvanced { turn: u64 },
    Rolled { dice: DiceExpr, total: i32 },
    Moved {
        entity_id: EntityId,
        from: LocationId,
        to: LocationId,
    },
    InspectedEntity { entity_id: EntityId, target: EntityId },
    InspectedLocation {
        entity_id: EntityId,
        target: LocationId,
    },
    AttackResolved {
        attacker: EntityId,
        target: EntityId,
        attack_roll: i32,
        hit: bool,
        damage: i32,
        target_hp_after: i32,
    },
    EntityDefeated { entity_id: EntityId },
}

#[derive(Debug, Error)]
pub enum DiceParseError {
    #[error("invalid dice expression")]
    Invalid,
    #[error("dice count must be >= 1")]
    Count,
    #[error("dice sides must be >= 2")]
    Sides,
}

/// Canonical dice expression: $N\mathrm{d}S + B$.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct DiceExpr {
    pub count: u32,
    pub sides: u32,
    pub bonus: i32,
}

impl DiceExpr {
    pub const fn new(count: u32, sides: u32, bonus: i32) -> Result<Self, DiceParseError> {
        if count < 1 {
            return Err(DiceParseError::Count);
        }
        if sides < 2 {
            return Err(DiceParseError::Sides);
        }
        Ok(Self {
            count,
            sides,
            bonus,
        })
    }

    /// Parse forms: `d20`, `1d6`, `2d8+3`, `3d10-2`.
    pub fn parse(input: &str) -> Result<Self, DiceParseError> {
        let s = input.trim().to_ascii_lowercase();
        let Some(d_pos) = s.find('d') else {
            return Err(DiceParseError::Invalid);
        };

        let (count_s, rest) = s.split_at(d_pos);
        let rest = &rest[1..];

        let count: u32 = if count_s.is_empty() {
            1
        } else {
            count_s.parse().map_err(|_| DiceParseError::Invalid)?
        };

        // rest = "<sides><(+|-)bonus>?"
        let mut sides_part = rest;
        let mut bonus = 0i32;
        if let Some(plus) = rest.find('+') {
            sides_part = &rest[..plus];
            bonus = rest[(plus + 1)..]
                .parse::<i32>()
                .map_err(|_| DiceParseError::Invalid)?;
        } else if let Some(minus) = rest.find('-') {
            sides_part = &rest[..minus];
            bonus = -rest[(minus + 1)..]
                .parse::<i32>()
                .map_err(|_| DiceParseError::Invalid)?;
        }

        let sides: u32 = sides_part.parse().map_err(|_| DiceParseError::Invalid)?;
        Self::new(count, sides, bonus)
    }
}

/// Hash the world state deterministically (used by replay tests and audit).
pub fn state_hash(state: &WorldState) -> [u8; 32] {
    let mut h = blake3::Hasher::new();
    hash_world_state(&mut h, state);
    h.finalize().into()
}

/// Hash the canonical event log deterministically.
pub fn events_hash(events: &[Event]) -> [u8; 32] {
    let mut h = blake3::Hasher::new();
    hash_u64(&mut h, events.len() as u64);
    for e in events {
        hash_event(&mut h, e);
    }
    h.finalize().into()
}

fn hash_u64(h: &mut blake3::Hasher, v: u64) {
    h.update(&v.to_le_bytes());
}

fn hash_i32(h: &mut blake3::Hasher, v: i32) {
    h.update(&v.to_le_bytes());
}

fn hash_u32(h: &mut blake3::Hasher, v: u32) {
    h.update(&v.to_le_bytes());
}

fn hash_str(h: &mut blake3::Hasher, s: &str) {
    // Length-prefix to avoid ambiguity.
    hash_u64(h, s.len() as u64);
    h.update(s.as_bytes());
}

fn hash_location_id(h: &mut blake3::Hasher, id: LocationId) {
    hash_u64(h, id.get());
}

fn hash_entity_id(h: &mut blake3::Hasher, id: EntityId) {
    hash_u64(h, id.get());
}

fn hash_player_id(h: &mut blake3::Hasher, id: PlayerId) {
    hash_u64(h, id.get());
}

fn hash_world_state(h: &mut blake3::Hasher, s: &WorldState) {
    hash_u64(h, s.turn);

    hash_u64(h, s.locations.len() as u64);
    for loc in &s.locations {
        hash_location(h, loc);
    }

    hash_u64(h, s.entities.len() as u64);
    for ent in &s.entities {
        hash_entity(h, ent);
    }

    hash_u64(h, s.players.len() as u64);
    for p in &s.players {
        hash_player(h, p);
    }
}

fn hash_location(h: &mut blake3::Hasher, loc: &Location) {
    hash_location_id(h, loc.id);
    hash_str(h, &loc.name);
    hash_u64(h, loc.neighbors.len() as u64);
    for n in &loc.neighbors {
        hash_location_id(h, *n);
    }
}

fn hash_entity(h: &mut blake3::Hasher, e: &Entity) {
    hash_entity_id(h, e.id);
    hash_str(h, &e.name);
    hash_location_id(h, e.location);
    hash_i32(h, e.hp);
    hash_i32(h, e.max_hp);
    hash_i32(h, e.attack_bonus);
    hash_i32(h, e.armor_class);
}

fn hash_player(h: &mut blake3::Hasher, p: &Player) {
    hash_player_id(h, p.id);
    hash_str(h, &p.name);
    hash_entity_id(h, p.entity_id);
}

fn hash_dice_expr(h: &mut blake3::Hasher, d: &DiceExpr) {
    hash_u32(h, d.count);
    hash_u32(h, d.sides);
    hash_i32(h, d.bonus);
}

fn hash_event(h: &mut blake3::Hasher, e: &Event) {
    // Tag each variant explicitly.
    match e {
        Event::TurnAdvanced { turn } => {
            hash_u32(h, 1);
            hash_u64(h, *turn);
        }
        Event::Rolled { dice, total } => {
            hash_u32(h, 2);
            hash_dice_expr(h, dice);
            hash_i32(h, *total);
        }
        Event::Moved { entity_id, from, to } => {
            hash_u32(h, 3);
            hash_entity_id(h, *entity_id);
            hash_location_id(h, *from);
            hash_location_id(h, *to);
        }
        Event::InspectedEntity { entity_id, target } => {
            hash_u32(h, 4);
            hash_entity_id(h, *entity_id);
            hash_entity_id(h, *target);
        }
        Event::InspectedLocation { entity_id, target } => {
            hash_u32(h, 5);
            hash_entity_id(h, *entity_id);
            hash_location_id(h, *target);
        }
        Event::AttackResolved {
            attacker,
            target,
            attack_roll,
            hit,
            damage,
            target_hp_after,
        } => {
            hash_u32(h, 6);
            hash_entity_id(h, *attacker);
            hash_entity_id(h, *target);
            hash_i32(h, *attack_roll);
            hash_u32(h, if *hit { 1 } else { 0 });
            hash_i32(h, *damage);
            hash_i32(h, *target_hp_after);
        }
        Event::EntityDefeated { entity_id } => {
            hash_u32(h, 7);
            hash_entity_id(h, *entity_id);
        }
    }
}

pub fn make_visible_state(state: &WorldState, player_id: PlayerId) -> Option<VisibleState> {
    let player = state.players.iter().find(|p| p.id == player_id)?;
    let self_entity = state.find_entity(player.entity_id)?.clone();
    let location = state.find_location(self_entity.location)?.clone();

    let co_located_entities = state
        .entities
        .iter()
        .filter(|e| e.location == self_entity.location && e.id != self_entity.id)
        .cloned()
        .collect();

    Some(VisibleState {
        turn: state.turn,
        player_id,
        self_entity,
        location,
        co_located_entities,
    })
}
