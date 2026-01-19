//! Deterministic action application engine.
//!

use cquests_core::{
    Action, ActionFailure, ActionOutcome, ActionResult, DiceExpr, EntityId, ErrorCode, Event,
    InspectTarget, LocationId, Seed, WorldState,
};
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum EngineError {
    #[error("action denied: {0:?}")]
    Denied(ActionFailure),
}

#[derive(Debug, Clone)]
pub struct ApplyOutcome {
    pub result: ActionResult,
    pub events: Vec<Event>,
}

/// Deterministic engine state.
///
/// The only stateful piece is the RNG; all authoritative world data lives in `WorldState`.
pub struct Engine {
    rng: ChaCha20Rng,
}

impl Engine {
    pub fn new(seed: Seed) -> Self {
        // ChaCha expects a 32-byte seed. Expand u64 deterministically.
        let mut bytes = [0u8; 32];
        bytes[..8].copy_from_slice(&seed.to_le_bytes());
        Self {
            rng: ChaCha20Rng::from_seed(bytes),
        }
    }

    pub fn apply(&mut self, state: &mut WorldState, action: Action) -> ApplyOutcome {
        // advance turn on every attempted action (success or failure) for clarity.
        // In later phases, this can be moved to the server scheduling layer.
        state.turn = state.turn.saturating_add(1);
        let mut events = vec![Event::TurnAdvanced { turn: state.turn }];

        let result = match action {
            Action::Roll { dice } => {
                let total = roll_dice(&dice, &mut self.rng);
                events.push(Event::Rolled { dice, total });
                ActionResult::Success(ActionOutcome::Rolled { total })
            }
            Action::Move {
                entity_id,
                destination,
            } => {
                match apply_move(state, entity_id, destination) {
                    Ok((from, to)) => {
                        events.push(Event::Moved {
                            entity_id,
                            from,
                            to,
                        });
                        ActionResult::Success(ActionOutcome::Moved { from, to })
                    }
                    Err(f) => ActionResult::Failure(f),
                }
            }
            Action::Inspect { entity_id, target } => {
                match apply_inspect(state, entity_id, target.clone()) {
                    Ok(outcome) => {
                        match target {
                            InspectTarget::Entity(t) => {
                                events.push(Event::InspectedEntity {
                                    entity_id,
                                    target: t,
                                });
                            }
                            InspectTarget::Location(t) => {
                                events.push(Event::InspectedLocation {
                                    entity_id,
                                    target: t,
                                });
                            }
                        }
                        ActionResult::Success(outcome)
                    }
                    Err(f) => ActionResult::Failure(f),
                }
            }
            Action::Attack { attacker, target } => {
                match apply_attack(state, &mut self.rng, attacker, target) {
                    Ok((outcome, event, defeated)) => {
                        events.push(event);
                        if let Some(defeated_id) = defeated {
                            events.push(Event::EntityDefeated {
                                entity_id: defeated_id,
                            });
                        }
                        ActionResult::Success(outcome)
                    }
                    Err(f) => ActionResult::Failure(f),
                }
            }
        };

        ApplyOutcome { result, events }
    }
}

fn failure(code: ErrorCode, message: impl Into<String>) -> ActionFailure {
    ActionFailure {
        code,
        message: message.into(),
    }
}

fn apply_move(
    state: &mut WorldState,
    entity_id: EntityId,
    destination: LocationId,
) -> Result<(LocationId, LocationId), ActionFailure> {
    let entity = state
        .find_entity(entity_id)
        .ok_or_else(|| failure(ErrorCode::NotFound, "entity not found"))?;
    if entity.hp <= 0 {
        return Err(failure(ErrorCode::NotAlive, "entity is not alive"));
    }

    let from = entity.location;
    let from_loc = state
        .find_location(from)
        .ok_or_else(|| failure(ErrorCode::NotFound, "current location not found"))?;

    if !from_loc.neighbors.iter().any(|&n| n == destination) {
        return Err(failure(
            ErrorCode::InvalidMove,
            "destination is not adjacent",
        ));
    }

    let _ = state
        .find_location(destination)
        .ok_or_else(|| failure(ErrorCode::NotFound, "destination not found"))?;

    // Mutate state
    let entity = state
        .find_entity_mut(entity_id)
        .ok_or_else(|| failure(ErrorCode::NotFound, "entity not found"))?;
    entity.location = destination;

    Ok((from, destination))
}

fn apply_inspect(
    state: &WorldState,
    entity_id: EntityId,
    target: InspectTarget,
) -> Result<ActionOutcome, ActionFailure> {
    let entity = state
        .find_entity(entity_id)
        .ok_or_else(|| failure(ErrorCode::NotFound, "entity not found"))?;
    if entity.hp <= 0 {
        return Err(failure(ErrorCode::NotAlive, "entity is not alive"));
    }

    match target {
        InspectTarget::Entity(target_id) => {
            let target_ent = state
                .find_entity(target_id)
                .ok_or_else(|| failure(ErrorCode::NotFound, "target entity not found"))?;

            // only inspect co-located entities. TODO: later phases may allow ranged inspection.
            if target_ent.location != entity.location {
                return Err(failure(ErrorCode::OutOfRange, "target not in range"));
            }

            Ok(ActionOutcome::InspectedEntity {
                entity: target_ent.clone(),
            })
        }
        InspectTarget::Location(target_id) => {
            if target_id != entity.location {
                return Err(failure(
                    ErrorCode::OutOfRange,
                    "can only inspect current location",
                ));
            }
            let loc = state
                .find_location(target_id)
                .ok_or_else(|| failure(ErrorCode::NotFound, "location not found"))?;
            Ok(ActionOutcome::InspectedLocation {
                location: loc.clone(),
            })
        }
    }
}

fn apply_attack(
    state: &mut WorldState,
    rng: &mut impl rand_core::RngCore,
    attacker: EntityId,
    target: EntityId,
) -> Result<(ActionOutcome, Event, Option<EntityId>), ActionFailure> {
    if attacker == target {
        return Err(failure(ErrorCode::InvalidMove, "cannot attack self"));
    }

    let attacker_ent = state
        .find_entity(attacker)
        .ok_or_else(|| failure(ErrorCode::NotFound, "attacker not found"))?
        .clone();
    let target_ent = state
        .find_entity(target)
        .ok_or_else(|| failure(ErrorCode::NotFound, "target not found"))?
        .clone();

    if attacker_ent.hp <= 0 {
        return Err(failure(ErrorCode::NotAlive, "attacker is not alive"));
    }
    if target_ent.hp <= 0 {
        return Err(failure(ErrorCode::NotAlive, "target is not alive"));
    }

    // only attack co-located entities. TODO: later phases may allow ranged attacks.
    if attacker_ent.location != target_ent.location {
        return Err(failure(ErrorCode::OutOfRange, "target not in range"));
    }

    // Attack roll: d20 + attack_bonus
    let attack_roll = DiceExpr::new(1, 20, attacker_ent.attack_bonus)
        .expect("static dice")
        ;
    let attack_roll = roll_dice(&attack_roll, rng);

    let hit = attack_roll >= target_ent.armor_class;

    let mut damage = 0;
    let mut hp_after = target_ent.hp;
    let mut defeated = None;

    if hit {
        // Damage: 1d6 (placeholder weapon model)
        let dmg_expr = DiceExpr::new(1, 6, 0).expect("static dice");
        damage = roll_dice(&dmg_expr, rng);
        hp_after = (target_ent.hp - damage).max(0);

        let target_mut = state
            .find_entity_mut(target)
            .ok_or_else(|| failure(ErrorCode::NotFound, "target not found"))?;
        target_mut.hp = hp_after;

        if hp_after == 0 {
            defeated = Some(target);
        }
    }

    let outcome = ActionOutcome::Attacked {
        hit,
        attack_roll,
        damage,
        target_hp_after: hp_after,
    };

    let event = Event::AttackResolved {
        attacker,
        target,
        attack_roll,
        hit,
        damage,
        target_hp_after: hp_after,
    };

    Ok((outcome, event, defeated))
}

fn roll_dice(expr: &DiceExpr, rng: &mut impl rand_core::RngCore) -> i32 {
    // Uniform 1..=sides for each die.
    let mut total = expr.bonus;
    for _ in 0..expr.count {
        let r = (rng.next_u32() % expr.sides) + 1;
        total += r as i32;
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;
    use cquests_core::{events_hash, state_hash, Entity, Location, LocationId};
    use pretty_assertions::assert_eq;

    fn tiny_world() -> WorldState {
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
            players: vec![],
        }
    }

    #[test]
    fn replay_is_deterministic() {
        let seed = 123_u64;
        let actions = vec![
            Action::Roll {
                dice: DiceExpr::parse("d20").unwrap(),
            },
            Action::Move {
                entity_id: EntityId::new(10),
                destination: LocationId::new(2),
            },
            Action::Move {
                entity_id: EntityId::new(10),
                destination: LocationId::new(1),
            },
            Action::Attack {
                attacker: EntityId::new(10),
                target: EntityId::new(11),
            },
        ];

        let mut s1 = tiny_world();
        let mut e1 = Engine::new(seed);
        let mut log1 = Vec::new();
        for a in actions.clone() {
            let out = e1.apply(&mut s1, a);
            log1.extend(out.events);
        }

        let mut s2 = tiny_world();
        let mut e2 = Engine::new(seed);
        let mut log2 = Vec::new();
        for a in actions {
            let out = e2.apply(&mut s2, a);
            log2.extend(out.events);
        }

        assert_eq!(state_hash(&s1), state_hash(&s2));
        assert_eq!(events_hash(&log1), events_hash(&log2));
        assert_eq!(log1, log2);
    }

    #[test]
    fn move_requires_adjacency() {
        let seed = 1_u64;
        let mut state = tiny_world();
        // Add a disconnected location
        state.locations.push(Location {
            id: LocationId::new(3),
            name: "C".into(),
            neighbors: vec![],
        });

        let mut engine = Engine::new(seed);
        let out = engine.apply(
            &mut state,
            Action::Move {
                entity_id: EntityId::new(10),
                destination: LocationId::new(3),
            },
        );

        match out.result {
            ActionResult::Failure(f) => assert_eq!(f.code, ErrorCode::InvalidMove),
            other => panic!("expected failure, got {other:?}"),
        }
    }
}
