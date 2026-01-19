use cquests_llm::{Backend, ChatMessage, ChatRole, GenerateRequest, LlmClient, TokenEvent};

fn usage_and_exit() -> ! {
    eprintln!(
        "cquests_llm_smoke [--backend mock|burn] [--manifest <path>] [--prompt <text>] [--cancel-after-chunks <n>]\n\n  --backend              Backend to use (default: mock).\n  --manifest             Manifest path (burn only; default: models/manifest.toml).\n  --prompt               User prompt (default: 'Hello GM!').\n  --cancel-after-chunks  Request cancellation after N streamed chunks (optional)."
    );
    std::process::exit(2)
}

fn main() {
    // Phase 3 smoke test:
    // - create session
    // - stream output
    // - demonstrate cancellation if requested
    let mut backend_choice = "mock".to_string();
    let mut manifest = "models/manifest.toml".to_string();
    let mut prompt = "Hello GM!".to_string();
	let mut cancel_after_chunks: Option<u32> = None;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--help" | "-h" => usage_and_exit(),
            "--backend" => {
                backend_choice = args.next().unwrap_or_else(|| usage_and_exit());
            }
            "--manifest" => {
                manifest = args.next().unwrap_or_else(|| usage_and_exit());
            }
            "--prompt" => {
                prompt = args.next().unwrap_or_else(|| usage_and_exit());
            }
			"--cancel-after-chunks" => {
				let raw = args.next().unwrap_or_else(|| usage_and_exit());
				cancel_after_chunks = Some(raw.parse().unwrap_or_else(|_| usage_and_exit()));
			}
            other => {
                eprintln!("unknown arg: {other}");
                usage_and_exit();
            }
        }

        // The manifest is only used when selecting the burn backend; keep it "used" in
        // default builds so we don't warn on unused variables.
        let _ = &manifest;
    }

    let backend = match backend_choice.as_str() {
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
                eprintln!("burn backend not enabled (rebuild with --features burn)");
                std::process::exit(1);
            }
        }
        _ => {
            eprintln!("invalid --backend: {backend_choice}");
            usage_and_exit();
        }
    };
    let session = backend.create_session();

    // Turn 1
    {
        let mut req = GenerateRequest::new(
            "You are a helpful Game Master. Output NARRATIVE:... or TOOL_CALL: {...}.",
        );
        req.messages.push(ChatMessage {
            role: ChatRole::User,
            content: prompt,
        });

        let mut stream = backend.generate_stream(session, req).expect("stream");
        let mut seen_chunks: u32 = 0;
        let mut cancel_at = cancel_after_chunks;
        for ev in &mut stream {
            match ev {
                TokenEvent::TokenChunk { text } => {
                    print!("{text}");
                    seen_chunks += 1;
                    if cancel_at.is_some_and(|n| seen_chunks >= n) {
                        let _ = backend.abort(session);
                        eprintln!("\n[cancel requested after {seen_chunks} chunks]");
                        cancel_at = None;
                    }
                }
                TokenEvent::ToolCallCandidate { raw } => eprintln!("\n[tool_call_candidate] {raw}"),
                TokenEvent::Finished { usage, reason } => {
                    eprintln!("\n\n[finished] reason={reason:?} usage={usage:?}");
                    break;
                }
                TokenEvent::Error { message } => {
                    eprintln!("\n[error] {message}");
                    break;
                }
            }
        }
    }

    eprintln!("\n\n--- second turn (same session) ---\n");

    // Turn 2: only provide the new user message; the backend should reuse session state.
    {
        let mut req = GenerateRequest::new(
            "You are a helpful Game Master. Output NARRATIVE:... or TOOL_CALL: {...}.",
        );
        req.messages.push(ChatMessage {
            role: ChatRole::User,
            content: "Continue the scene. What happens next?".into(),
        });

        let mut stream = backend.generate_stream(session, req).expect("stream");
        let mut seen_chunks: u32 = 0;
        let mut cancel_at = cancel_after_chunks;
        for ev in &mut stream {
            match ev {
                TokenEvent::TokenChunk { text } => {
                    print!("{text}");
                    seen_chunks += 1;
                    if cancel_at.is_some_and(|n| seen_chunks >= n) {
                        let _ = backend.abort(session);
                        eprintln!("\n[cancel requested after {seen_chunks} chunks]");
                        cancel_at = None;
                    }
                }
                TokenEvent::ToolCallCandidate { raw } => eprintln!("\n[tool_call_candidate] {raw}"),
                TokenEvent::Finished { usage, reason } => {
                    eprintln!("\n\n[finished] reason={reason:?} usage={usage:?}");
                    break;
                }
                TokenEvent::Error { message } => {
                    eprintln!("\n[error] {message}");
                    break;
                }
            }
        }
    }
}
