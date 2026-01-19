use std::time::Duration;

use crossbeam_channel::{Receiver, Sender};
use eframe::egui;

use cquests_core::{PlayerId, SessionId, VisibleState};
use cquests_gm::GmAgent;
use cquests_llm::Backend as LlmBackend;
use cquests_mcp::McpRouter;
use cquests_server::{ActionService, AuthContext, InMemoryActionService, JsonFileSessionStore, SessionStore};

#[derive(Debug)]
enum PlayerCommand {
	Send(String),
}

#[derive(Debug)]
enum UiEvent {
	AssistantStart,
	AssistantDelta(String),
	ToolCall { name: String, args: serde_json::Value },
	ToolResult(serde_json::Value),
	VisibleState(VisibleState),
	NarrativeFinal(String),
	Error(String),
	Busy(bool),
}

struct ClientApp {
	tx_cmd: Sender<PlayerCommand>,
	rx_ev: Receiver<UiEvent>,

	input: String,
	chat: Vec<(String, String)>, // (role, text)
	streaming: String,
	visible: Option<VisibleState>,
	busy: bool,
	events: Vec<String>,
}

impl ClientApp {
	fn new(tx_cmd: Sender<PlayerCommand>, rx_ev: Receiver<UiEvent>) -> Self {
		Self {
			tx_cmd,
			rx_ev,
			input: String::new(),
			chat: Vec::new(),
			streaming: String::new(),
			visible: None,
			busy: false,
			events: Vec::new(),
		}
	}

	fn pump_events(&mut self) {
		while let Ok(ev) = self.rx_ev.try_recv() {
			match ev {
				UiEvent::AssistantStart => {
					self.streaming.clear();
				}
				UiEvent::AssistantDelta(s) => {
					self.streaming.push_str(&s);
				}
				UiEvent::NarrativeFinal(n) => {
					self.streaming.clear();
					self.chat.push(("GM".into(), n));
				}
				UiEvent::ToolCall { name, args } => {
					self.events.push(format!("tool_call: {name} {args}"));
				}
				UiEvent::ToolResult(v) => {
					self.events.push(format!("tool_result: {v}"));
				}
				UiEvent::VisibleState(vs) => {
					self.visible = Some(vs);
				}
				UiEvent::Error(e) => {
					self.events.push(format!("error: {e}"));
					self.streaming.clear();
				}
				UiEvent::Busy(b) => self.busy = b,
			}
		}
	}
}

impl eframe::App for ClientApp {
	fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
		self.pump_events();

		egui::TopBottomPanel::top("top").show(ctx, |ui| {
			ui.heading("CQuests — Local Client (Phase 5)");
			ui.horizontal(|ui| {
				ui.label(if self.busy { "GM: thinking…" } else { "GM: idle" });
				if ui.button("Clear log").clicked() {
					self.events.clear();
				}
			});
		});

		egui::SidePanel::right("state").resizable(true).show(ctx, |ui| {
			ui.heading("State");
			ui.separator();
			if let Some(v) = &self.visible {
				ui.label(format!("Turn: {}", v.turn));
				ui.label(format!("Location: {} (id={})", v.location.name, v.location.id.get()));
				ui.label(format!("Self: {} hp {}/{}", v.self_entity.name, v.self_entity.hp, v.self_entity.max_hp));
				ui.separator();
				ui.label("Co-located:");
				for e in &v.co_located_entities {
					ui.label(format!("- {} (id={}) hp={}", e.name, e.id.get(), e.hp));
				}
			} else {
				ui.label("(no state yet)");
			}
			ui.separator();
			ui.heading("Log");
			egui::ScrollArea::vertical().max_height(260.0).show(ui, |ui| {
				for line in self.events.iter().rev().take(200).rev() {
					ui.label(line);
				}
			});
		});

		egui::CentralPanel::default().show(ctx, |ui| {
			ui.heading("Chat");
			ui.separator();
			egui::ScrollArea::vertical().stick_to_bottom(true).show(ui, |ui| {
				for (role, text) in &self.chat {
					ui.horizontal_wrapped(|ui| {
						ui.strong(format!("{role}:"));
						ui.label(text);
					});
				}
				if !self.streaming.is_empty() {
					ui.horizontal_wrapped(|ui| {
						ui.strong("GM (streaming):");
						ui.label(&self.streaming);
					});
				}
			});

			ui.separator();
			ui.horizontal(|ui| {
				let send_clicked = ui.add_enabled(!self.busy, egui::Button::new("Send")).clicked();
				let resp = ui.add_enabled(!self.busy, egui::TextEdit::singleline(&mut self.input).hint_text("Type a command…"));
				let enter = resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
				if (send_clicked || enter) && !self.input.trim().is_empty() {
					let text = self.input.trim().to_string();
					self.chat.push(("Player".into(), text.clone()));
					self.streaming.clear();
					let _ = self.tx_cmd.send(PlayerCommand::Send(text));
					self.input.clear();
				}
			});
		});

		ctx.request_repaint_after(Duration::from_millis(16));
	}
}

fn main() -> eframe::Result<()> {
	let (tx_cmd, rx_cmd) = crossbeam_channel::unbounded::<PlayerCommand>();
	let (tx_ev, rx_ev) = crossbeam_channel::unbounded::<UiEvent>();

	// Worker thread: owns server + router + GM + persistence.
	std::thread::spawn(move || {
		let service = InMemoryActionService::new();
		let store = JsonFileSessionStore::new(InMemoryActionService::default_store_dir());

		let player_id = PlayerId::new(1);
		let session_id = SessionId::new(1);

		// Load existing snapshot if present; otherwise create new.
		let sid = match store.load(session_id) {
			Ok(snapshot) => service.restore_snapshot(snapshot).unwrap_or_else(|_| {
				service.create_session(123, InMemoryActionService::default_demo_world(player_id))
			}),
			Err(_) => service.create_session(123, InMemoryActionService::default_demo_world(player_id)),
		};

		// Background scheduled persistence (in addition to save-after-turn).
		let _autosave = service.start_autosave(sid, store.clone(), Duration::from_secs(2));

		let router = McpRouter::new(service.clone());
		let llm = {
			#[cfg(feature = "burn")]
			{
				match LlmBackend::burn_default() {
					Ok(b) => b,
					Err(e) => {
						let _ = tx_ev.send(UiEvent::Error(format!(
							"LLM burn backend failed to initialize: {e}.\n\
												Strict mode is enabled: not falling back to mock."
						)));
						let _ = tx_ev.send(UiEvent::Busy(false));
						return;
					}
				}
			}
			#[cfg(not(feature = "burn"))]
			{
				let _ = tx_ev.send(UiEvent::Error(
					"Strict mode is enabled but the burn backend is not compiled (feature 'burn' disabled).\n\
												Rebuild with --features burn,burn-llama3 and a backend (burn-tch|burn-cuda|burn-vulkan|burn-ndarray)."
						.into(),
				));
				let _ = tx_ev.send(UiEvent::Busy(false));
				return;
			}
		};
		let gm = GmAgent::new(llm, router.clone());

		// Initial visible state
		if let Ok(vs) = service.get_visible_state(&AuthContext::player(player_id), sid, player_id) {
			let _ = tx_ev.send(UiEvent::VisibleState(vs));
		}

		for cmd in rx_cmd.iter() {
			match cmd {
				PlayerCommand::Send(text) => {
					let _ = tx_ev.send(UiEvent::Busy(true));
					let _ = tx_ev.send(UiEvent::AssistantStart);
					let tx_ev_stream = tx_ev.clone();

					match gm.handle_player_turn_streaming_with_trace(sid, player_id, &text, |delta| {
						let _ = tx_ev_stream.send(UiEvent::AssistantDelta(delta.to_string()));
					}) {
						Ok(out) => {
							for (tc, result) in &out.tool_trace {
								let _ = tx_ev.send(UiEvent::ToolCall {
									name: tc.name.clone(),
									args: tc.arguments.clone(),
								});
								let _ = tx_ev.send(UiEvent::ToolResult(result.clone()));
							}

							// Save snapshot after the turn.
							if let Ok(snapshot) = service.export_snapshot(sid) {
								let _ = store.save(&snapshot);
							}

							if let Ok(vs) = service.get_visible_state(&AuthContext::player(player_id), sid, player_id) {
								let _ = tx_ev.send(UiEvent::VisibleState(vs));
							}
							let _ = tx_ev.send(UiEvent::NarrativeFinal(out.narrative));
						}
						Err(e) => {
							let _ = tx_ev.send(UiEvent::Error(e.to_string()));
						}
					}

					let _ = tx_ev.send(UiEvent::Busy(false));
				}
			}
		}
	});

	let options = eframe::NativeOptions::default();
	eframe::run_native(
		"CQuests Client",
		options,
		Box::new(|_cc| Ok(Box::new(ClientApp::new(tx_cmd, rx_ev)))),
	)
}
