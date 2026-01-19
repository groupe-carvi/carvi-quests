# Local model artifacts

This folder is for **local-only** model artifacts.

- Keep large model files out of git.
- Configure paths in `models/manifest.toml`.

## Expected files for llama-burn

For Llama 3.x instruct models using `llama-burn`, you typically need:

- `model.mpk` (Burn checkpoint)
- `tokenizer.model` (SentencePiece tokenizer)

The `llama-burn` repository provides examples and pretrained download helpers (to `~/.cache/llama-burn`), but CQuests uses `./models/` so you can manage artifacts explicitly.

## Suggested layout

```
models/
  manifest.toml
  llama-3.1-8b-instruct/
    model.mpk
    tokenizer.model
```

## Fixing the error “model.checkpoint not found”

If `models/manifest.toml` points to:

- `models/llama-3.1-8b-instruct/model.mpk`
- `models/llama-3.1-8b-instruct/tokenizer.model`

then create the folder `models/llama-3.1-8b-instruct/` and place **exactly** those two files there.

The files are published on Hugging Face (Files tab):

- https://huggingface.co/tracel-ai/llama-3.1-8b-instruct-burn/tree/main

## Optional: build-time vendoring (no files under ./models)

If you compile with the Cargo feature `model-download`, the build script for `cquests-llm` can download the artifacts into:

`target/llama-burn/<variant>/` (for example `target/llama-burn/llama3_1_8b_instruct/`).

When the app uses `Backend::burn_default()`, it will prefer that vendored manifest automatically when present.
