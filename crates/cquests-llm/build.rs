use std::env;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

fn main() {
	// Only do vendoring work when the burn backend is being compiled.
	if env::var_os("CARGO_FEATURE_BURN").is_none() {
		return;
	}

	let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR must be set"));
	let target_dir = target_dir_from_out_dir(&out_dir).unwrap_or_else(|| {
		// Fallback: infer from crate manifest dir by walking up to repo root.
		let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR must be set"));
		let repo_root = find_repo_root(&manifest_dir).unwrap_or(manifest_dir);
		repo_root.join("target")
	});

	// Pick a model variant.
	// Priority: env override -> models/manifest.toml if present -> default.
	let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR must be set"));
	let repo_root = find_repo_root(&manifest_dir).unwrap_or(manifest_dir);
	let models_manifest = repo_root.join("models").join("manifest.toml");

	let variant = env::var("CQUESTS_MODEL_VARIANT")
		.ok()
		.or_else(|| read_variant_from_manifest(&models_manifest))
		.unwrap_or_else(|| "llama3_1_8b_instruct".to_string());

	let (model_url, tokenizer_url) = pretrained_urls(&variant).unwrap_or_else(|| {
		panic!(
			"Unsupported CQUESTS_MODEL_VARIANT '{variant}'. Expected one of: \
llama3_1_8b_instruct | llama3_8b_instruct | llama3_2_1b_instruct | llama3_2_3b_instruct"
		)
	});

	let vendor_dir = target_dir.join("llama-burn").join(&variant);
	let checkpoint_path = vendor_dir.join("model.mpk");
	let tokenizer_path = vendor_dir.join("tokenizer.model");

	fs::create_dir_all(&vendor_dir).expect("failed to create vendor dir under target");

	// Feature toggle: only download when explicitly enabled.
	// Cargo exposes enabled features to build scripts as `CARGO_FEATURE_<FEATURE_NAME_IN_CAPS_WITH_UNDERSCORES>`.
	let should_download = env::var_os("CARGO_FEATURE_MODEL_DOWNLOAD").is_some();
	if should_download {
		if !checkpoint_path.is_file() {
			download_to_file(&model_url, &checkpoint_path).expect("failed to download model.mpk");
		}
		if !tokenizer_path.is_file() {
			download_to_file(&tokenizer_url, &tokenizer_path).expect("failed to download tokenizer.model");
		}
	} else {
		// Don't auto-download by default (CI/test friendly); provide a clear hint.
		if !checkpoint_path.is_file() || !tokenizer_path.is_file() {
			println!(
				"cargo:warning=LLM artifacts are not vendored yet. Rebuild with feature 'model-download' to download into: {}",
				vendor_dir.display()
			);
			println!("cargo:warning=  model: {model_url}");
			println!("cargo:warning=  tokenizer: {tokenizer_url}");
		}
	}

	// Generate a vendor manifest in target/ so runtime can load from there.
	let vendor_manifest_path = vendor_dir.join("manifest.toml");
	write_vendor_manifest(&vendor_manifest_path, &variant, &checkpoint_path, &tokenizer_path)
		.expect("failed to write vendor manifest");

	// Make the vendor manifest path available to the crate at compile time.
	println!(
		"cargo:rustc-env=CQUESTS_LLM_VENDOR_MANIFEST={}",
		vendor_manifest_path.to_string_lossy()
	);

	// Re-run if the local models manifest changes (e.g., variant switched).
	println!("cargo:rerun-if-env-changed=CQUESTS_MODEL_VARIANT");
	// Cargo features aren't tracked by rerun-if-env-changed, but that's fine;
	// builds will re-run when features change.
	println!("cargo:rerun-if-changed={}", models_manifest.display());
}

fn target_dir_from_out_dir(out_dir: &Path) -> Option<PathBuf> {
	// Typical layout: <repo>/target/<profile>/build/<crate-hash>/out
	// We walk up ancestors looking for a directory literally named "target".
	for a in out_dir.ancestors() {
		if a.file_name().is_some_and(|n| n == "target") {
			return Some(a.to_path_buf());
		}
		// Some setups have CACHEDIR.TAG in target root.
		if a.join("CACHEDIR.TAG").is_file() {
			return Some(a.to_path_buf());
		}
	}
	None
}

fn find_repo_root(start_dir: &Path) -> Option<PathBuf> {
	for dir in start_dir.ancestors() {
		if dir.join("Cargo.toml").is_file() {
			return Some(dir.to_path_buf());
		}
	}
	None
}

fn read_variant_from_manifest(path: &Path) -> Option<String> {
	let text = fs::read_to_string(path).ok()?;
	// Minimal parser: look for a line like: variant = "..."
	// under [model]. Good enough for build-time convenience.
	let mut in_model = false;
	for line in text.lines() {
		let l = line.trim();
		if l.starts_with('[') {
			in_model = l == "[model]";
			continue;
		}
		if in_model {
			if let Some(rest) = l.strip_prefix("variant") {
				let rest = rest.trim_start();
				let rest = rest.strip_prefix('=').unwrap_or(rest).trim();
				let rest = rest.trim_matches('"');
				if !rest.is_empty() {
					return Some(rest.to_string());
				}
			}
		}
	}
	None
}

fn pretrained_urls(variant: &str) -> Option<(String, String)> {
	// Keep in sync with tracel-ai/models llama-burn/src/pretrained.rs
	let (model, tok) = match variant {
		"llama3_1_8b_instruct" => (
			"https://huggingface.co/tracel-ai/llama-3.1-8b-instruct-burn/resolve/main/model.mpk?download=true",
			"https://huggingface.co/tracel-ai/llama-3.1-8b-instruct-burn/resolve/main/tokenizer.model?download=true",
		),
		"llama3_8b_instruct" => (
			"https://huggingface.co/tracel-ai/llama-3-8b-instruct-burn/resolve/main/model.mpk?download=true",
			"https://huggingface.co/tracel-ai/llama-3-8b-instruct-burn/resolve/main/tokenizer.model?download=true",
		),
		"llama3_2_1b_instruct" => (
			"https://huggingface.co/tracel-ai/llama-3.2-1b-instruct-burn/resolve/main/model.mpk?download=true",
			"https://huggingface.co/tracel-ai/llama-3.2-1b-instruct-burn/resolve/main/tokenizer.model?download=true",
		),
		"llama3_2_3b_instruct" => (
			"https://huggingface.co/tracel-ai/llama-3.2-3b-instruct-burn/resolve/main/model.mpk?download=true",
			"https://huggingface.co/tracel-ai/llama-3.2-3b-instruct-burn/resolve/main/tokenizer.model?download=true",
		),
		_ => return None,
	};
	Some((model.to_string(), tok.to_string()))
}

fn download_to_file(url: &str, path: &Path) -> Result<(), String> {
	let tmp = path.with_extension("tmp");
	if let Some(parent) = path.parent() {
		fs::create_dir_all(parent).map_err(|e| format!("create_dir_all failed: {e}"))?;
	}

	println!("cargo:warning=Downloading {url}");
	println!("cargo:warning=  -> {dest}", dest = path.display());

	// Let ureq follow redirects; HF resolve URLs typically redirect to a CDN.
	let agent: ureq::Agent = ureq::config::Config::builder()
		.max_redirects(10)
		// Model downloads can take a while.
		.timeout_global(Some(std::time::Duration::from_secs(60 * 60)))
		.build()
		.into();

	let mut resp = agent
		.get(url)
		// Some CDNs behave better with a clear UA; also avoid compressed responses.
		.header("User-Agent", "cquests-llm-build/0.1")
		.header("Accept-Encoding", "identity")
		.call()
		.map_err(|e| format!("HTTP GET failed for {url}: {e}"))?;

	if resp.body().mime_type() == Some("text/html") {
		return Err(format!(
			"Unexpected Content-Type 'text/html' while downloading {url}. This usually means we downloaded an HTML error page, not the binary artifact."
		));
	}

	let mut reader = resp.body_mut().as_reader();
	let mut out = fs::File::create(&tmp).map_err(|e| format!("File::create failed: {e}"))?;

	let bytes = io::copy(&mut reader, &mut out).map_err(|e| format!("copy failed: {e}"))?;
	// Very small downloads are almost always an error page.
	if bytes < 64 * 1024 {
		return Err(format!(
			"Downloaded only {bytes} bytes from {url} into {}. This is suspiciously small and likely not the model artifact.",
			path.display()
		));
	}
	out.flush().ok();
	// Best effort fsync
	let _ = out.sync_all();

	// Atomic-ish replace.
	fs::rename(&tmp, path).map_err(|e| format!("rename failed: {e}"))?;

	println!("cargo:warning=Downloaded {} bytes", bytes);
	Ok(())
}

fn write_vendor_manifest(
	path: &Path,
	variant: &str,
	checkpoint: &Path,
	tokenizer: &Path,
) -> Result<(), String> {
	// On Windows, absolute paths contain backslashes. TOML basic strings interpret
	// backslashes as escape sequences, so unescaped paths like `C:\Users\...` can
	// fail to parse (e.g. `\U` expects a unicode escape).
	// We write paths with forward slashes to keep the manifest valid cross-platform.
	let checkpoint_toml = checkpoint.to_string_lossy().replace('\\', "/");
	let tokenizer_toml = tokenizer.to_string_lossy().replace('\\', "/");

	let content = format!(
		"# Auto-generated by crates/cquests-llm/build.rs\n\
# Do not commit; artifacts live under target/.\n\n\
[model]\n\
variant = \"{variant}\"\n\
checkpoint = \"{checkpoint}\"\n\
tokenizer = \"{tokenizer}\"\n\
max_seq_len = 4096\n\n\
[sampling]\n\
temperature = 0.6\n\
top_p = 0.9\n\
seed = 42\n",
		checkpoint = checkpoint_toml,
		tokenizer = tokenizer_toml,
	);
	fs::write(path, content).map_err(|e| format!("write failed: {e}"))
}
