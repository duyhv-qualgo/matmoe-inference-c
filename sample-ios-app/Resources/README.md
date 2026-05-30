# Resources/

Drop the three artifacts the app needs to translate:

| File                     | Where it comes from                                     | Size  |
|--------------------------|----------------------------------------------------------|-------|
| `encode_prefill.tflite`  | `matmoe-inference-c/models/dim-256/encode_prefill.tflite`| ~75 MB |
| `decode_step.tflite`     | `matmoe-inference-c/models/dim-256/decode_step.tflite`  | ~75 MB |
| `tokenizer.json`         | HuggingFace `tokenizer.json` you trained the model with | ~2 MB  |

These are intentionally gitignored — they're bundled into the `.app` at build
time as folder-reference resources.

The tokenizer must contain the direction tokens `<translate-en-vi>` and
`<translate-vi-en>` (and `<pad>`, `</s>` if using non-default pad/eos ids).
