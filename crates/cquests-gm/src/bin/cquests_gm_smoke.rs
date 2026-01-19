use std::io::{self, Write};

use cquests_core::{PlayerId, VisibleState};
use cquests_gm::GmAgent;
use cquests_llm::Backend;
use cquests_mcp::{auth_player, McpRouter, TOOL_GET_VISIBLE_STATE};
use cquests_server::{ActionService, InMemoryActionService};

fn usage_and_exit() -> ! {
	eprintln!(
		"cquests_gm_smoke [--backend mock|burn] [--manifest <path>] [--player-id <n>] [--once <text>]\n\n  --backend   Backend to use (default: mock).\n  --manifest  Manifest path (burn only; default: models/manifest.toml).\n  --player-id Player id (default: 1).\n  --once      Run a single turn and exit (otherwise interactive)."
	);
	std::process::exit(2)
}

fn main() {
	let mut backend_choice = "mock".to_string();
	let mut manifest = "models/manifest.toml".to_string();
	let mut player_id: u64 = 1;
	let mut once: Option<String> = None;

	let mut args = std::env::args().skip(1);
	while let Some(arg) = args.next() {
		match arg.as_str() {
			"--help" | "-h" => usage_and_exit(),
			"--backend" => backend_choice = args.next().unwrap_or_else(|| usage_and_exit()),
			"--manifest" => manifest = args.next().unwrap_or_else(|| usage_and_exit()),
			"--player-id" => {
				let raw = args.next().unwrap_or_else(|| usage_and_exit());
				player_id = raw.parse().unwrap_or_else(|_| usage_and_exit());
			}
			"--once" => once = Some(args.next().unwrap_or_else(|| usage_and_exit())),
			other => {
				eprintln!("unknown arg: {other}");
				usage_and_exit();
			}
		}

		// Keep these used in default builds.
		let _ = &manifest;
	}

	let llm: Backend = match backend_choice.as_str() {
		"mock" => Backend::mock(),
		"burn" => {
			#[cfg(feature = "burn")]
			{
				match Backend::burn_from_manifest_path(&manifest) {
					Ok(b) => b,
					Err(e) => {
						eprintln!("failed to initialize burn backend: {e:?}");
						std::process::exit(1);
					}
				}
			}
			#[cfg(not(feature = "burn"))]
			{
				eprintln!("burn backend not enabled (rebuild cquests-gm with --features burn ...) ");
				std::process::exit(1);
			}
		}
		_ => {
			eprintln!("invalid --backend: {backend_choice}");
			usage_and_exit();
		}
	};

	let service = InMemoryActionService::new();
	let session_id = service.create_session(
		123,
		InMemoryActionService::default_demo_world(PlayerId::new(player_id)),
	);
	let router = McpRouter::new(service);
	let gm = GmAgent::new(llm, router.clone());

	let player_id = PlayerId::new(player_id);

	if let Some(text) = once {
		run_one(&gm, &router, session_id, player_id, &text);
		return;
	}

	eprintln!("CQuests GM smoke — interactive. Type 'quit' to exit.");
	eprintln!("session_id={} player_id={} backend={}", session_id.get(), player_id.get(), backend_choice);

	let stdin = io::stdin();
	loop {
		print!("\n> player: ");
		let _ = io::stdout().flush();
		let mut line = String::new();
		if stdin.read_line(&mut line).is_err() {
			break;
		}
		let line = line.trim().to_string();
		if line.is_empty() {
			continue;
		}
		if line == "quit" || line == "exit" {
			break;
		}
		run_one(&gm, &router, session_id, player_id, &line);
	}
}

fn run_one(
	gm: &GmAgent<Backend, InMemoryActionService>,
	router: &McpRouter<InMemoryActionService>,
	session_id: cquests_core::SessionId,
	player_id: PlayerId,
	text: &str,
) {
	match gm.handle_player_turn_with_trace(session_id, player_id, text) {
		Ok(out) => {
			for (tc, result) in out.tool_trace {
				eprintln!("\n[tool_call] {} {}", tc.name, tc.arguments);
				eprintln!("[tool_result] {result}");
			}

			// Print a small state snapshot so state changes are obvious even without reading events.
			let auth = auth_player(player_id.get());
			match router.call_tool(
				&auth,
				TOOL_GET_VISIBLE_STATE,
				serde_json::json!({"session_id": session_id, "player_id": player_id.get()}),
			) {
				Ok(v) => {
					if let Ok(visible) = serde_json::from_value::<VisibleState>(v) {
						let names: Vec<String> = visible
							.co_located_entities
							.iter()
							.map(|e| format!("{}(id={},hp={})", e.name, e.id.get(), e.hp))
							.collect();
						eprintln!(
							"\n[state] turn={} location={} (id={}) self_hp={}/{} co_located=[{}]",
							visible.turn,
							visible.location.name,
							visible.location.id.get(),
							visible.self_entity.hp,
							visible.self_entity.max_hp,
							names.join(", ")
						);
					}
				}
				Err(e) => {
					eprintln!("\n[state] unable to fetch visible state: {e}");
				}
			}

			println!("\nGM: {}", out.narrative);
		}
		Err(e) => {
			eprintln!("\n[gm_error] {e}");
		}
	}
}
