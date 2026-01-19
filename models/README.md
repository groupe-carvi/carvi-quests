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
