# CARVI Quests

**CARVI Quests** is an open-source, project-based technical assessment framework built around a small RPG-style game.

It is designed to evaluate real-world engineering skills through hands-on work, realistic constraints, and incremental problem-solving — not through artificial puzzles or disconnected coding challenges.

The project is used internally at **Groupe CARVI** as part of our technical hiring process and evolves over time through candidate contributions.

---

## Purpose

Traditional technical interviews often fail to reflect how engineers actually work.

CARVI Quests aims to:
- Evaluate **practical engineering skills** in realistic conditions
- Observe **design thinking, trade-offs, and decision-making**
- Assess **code quality, adaptability, and technical communication**
- Create a **shared, evolving technical artifact** instead of throwaway tests

Each candidate works from an existing codebase and is asked to extend, adapt, or integrate part of the game depending on the role they are applying for.

---

## Assessment Philosophy

This project is intentionally:
- **Project-based**, not puzzle-based
- **Incremental**, not greenfield-only
- **Constraint-driven**, reflecting real hardware and system limits
- **Observable**, allowing discussion, reasoning, and explanation

We value:
- Clear thinking over clever tricks
- Trade-offs over perfection
- Communication over silent execution
- Engineering judgment over rote knowledge

---

## Typical Assessment Flow

A CARVI Quests assessment is usually conducted over a **4-hour on-site or live session**, divided into phases:

### 1. Design & Analysis
- Codebase exploration
- Constraint analysis
- Architecture and integration planning
- Whiteboard or verbal explanation

### 2. Implementation & Integration
- Feature development or system integration
- Adaptation to target constraints (e.g. embedded hardware)
- Incremental testing and validation

### 3. Refinement & Documentation
- Code cleanup and improvements
- UX or interaction refinements
- Short technical write-up documenting decisions and trade-offs

The exact scope varies depending on the role (embedded, backend, systems, tooling, etc.).

---

## Technical Stack (varies by scenario)

Depending on the assessment scenario, CARVI Quests may involve:
- **Rust**
- **Ratatui** (terminal UI)
- **Embedded targets** (e.g. ESP32 with touchscreen)
- Input systems (keyboard, touch, GPIO)
- Resource-constrained environments

Candidates are not expected to finish everything — the goal is to observe **how they work**, not just what they deliver.

---

## Open Source & Contributions

This repository is public by design.

Over time, candidates may:
- Add features
- Improve systems
- Introduce new scenarios
- Refactor or document parts of the project

This allows CARVI Quests to evolve organically while showcasing:
- Our technical culture
- Our hiring philosophy
- Real contributions from real engineers

---

## Disclaimer

This project is **not** a generic take-home test.

It is part of a guided technical assessment conducted with CARVI engineers and is not intended to be completed independently without context.

---

## License

[Add license here]
