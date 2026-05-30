# sample-ios-app

A minimal SwiftUI app that runs the **MatMoE en↔vi translator** on-device
via `SlmEngine.xcframework`. Lives inside the engine repo so it can ship and
version-bump with the engine it depends on.

```
[ direction picker  EN→VI | VI→EN ]
┌────────────────────────────────┐
│ Hello, how are you today?      │   ← English input
└────────────────────────────────┘
            [ Translate ]
┌────────────────────────────────┐
│ Chào, hôm nay anh khoẻ không?  │   ← Vietnamese output
└────────────────────────────────┘
        96 tokens · 480 ms · 5.0 ms/tok
```

---

## Layout

```
matmoe-inference-c/                 ← engine repo (this submodule)
├── include/  src/  scripts/        ← engine sources, build_ios.sh etc.
├── dist/                           ← scripts/build_ios.sh output lands here
│   └── SlmEngine.xcframework
└── sample-ios-app/                 ← THIS FOLDER
    ├── project.yml                 XcodeGen spec (one source of truth)
    ├── MatMoETranslator/
    │   ├── MatMoETranslatorApp.swift        @main
    │   ├── ContentView.swift                SwiftUI screen
    │   ├── TranslationViewModel.swift       owns engine + tokenizer
    │   ├── TranslationDirection.swift       en→vi / vi→en + prompt prefix
    │   ├── Tokenizer.swift                  loads tokenizer.json via swift-transformers
    │   ├── MatMoEBridge.h/.mm                Obj-C++ shim over matmoe::SlmEngine
    │   ├── MatMoETranslator-Bridging-Header.h
    │   └── Info.plist
    ├── Resources/                  ← drop .tflite + tokenizer.json here (gitignored)
    ├── Frameworks/                 ← bootstrap.sh drops SlmEngine.xcframework here
    └── scripts/
        ├── bootstrap.sh            build xcframework + xcodegen
        └── run_sim.sh              one-shot build + launch on iPhone sim
```

---

## Prereqs (Mac)

1. **macOS** — Xcode and the iOS Simulator are Mac-only.
2. **Xcode 15+** (iOS 15 SDK or newer). Open it once after install so the SDK
   gets unpacked.
3. **XcodeGen**:
   ```bash
   brew install xcodegen
   # optional: nicer xcodebuild output
   brew install xcbeautify
   ```

---

## Quick start

```bash
cd matmoe-inference-c/sample-ios-app

# 1. Drop the three artifacts in Resources/
#    (see Resources/README.md for what each one is)
cp /path/to/encode_prefill.tflite Resources/
cp /path/to/decode_step.tflite    Resources/
cp /path/to/tokenizer.json        Resources/

# 2. Build the xcframework (one level up at matmoe-inference-c/)
#    and generate MatMoETranslator.xcodeproj
./scripts/bootstrap.sh
#  first run: ~25 min (compiles TFLite + XNNPACK for iphoneos & sim)
#  subsequent runs: ~10 s

# 3a. Open in Xcode and hit Run
open MatMoETranslator.xcodeproj

# 3b. — OR — build + launch on a sim from the terminal
./scripts/run_sim.sh                 # default: "iPhone 15"
SIM_NAME="iPhone 15 Pro" ./scripts/run_sim.sh
```

---

## What the app does

1. On first appear, `TranslationViewModel.warmUpIfNeeded()` does:
   - `MatMoEBridge.load(encoderPath:decoderPath:threads:)` on a background
     thread → loads both `.tflite` graphs, attaches XNNPACK delegates.
   - `MatMoETokenizer.loadFromBundle()` copies `tokenizer.json` out of the
     read-only bundle into the caches dir (writes a tiny
     `tokenizer_config.json` next to it if missing), then calls
     `AutoTokenizer.from(modelFolder:)` from swift-transformers.

2. On **Translate**:
   - The input is prefixed with `<translate-en-vi>` (or `…vi-en>`), tokenized
     and padded to 128 int32 IDs + a 128-long attention mask.
   - The pair is handed to `MatMoEBridge.generate(srcIds:srcMask:...)` which
     calls `matmoe::SlmEngine::Generate(...)` on a detached Task.
   - The returned int32 IDs are decoded back to a string via the same
     tokenizer (`skipSpecialTokens: true`).

The engine runs on **4 CPU threads with XNNPACK**, greedy sampling by default.
Flip `method: .greedy` to `.topKTopP` in `TranslationViewModel` (or expose it
in the UI) for non-deterministic output.

---

## Wiring details (for porting into your own app)

### The C++ → Obj-C → Swift bridge

```
slm_engine.h  (C++)     ←—  matmoe-inference-c/include/
   ▲
   │  #include via xcframework Headers/
MatMoEBridge.mm  (Obj-C++)
   ▲
   │  bridging header
MatMoETranslator-Bridging-Header.h
   ▲
   │  exposed by SWIFT_OBJC_BRIDGING_HEADER
TranslationViewModel.swift
```

To use this in another app, copy the files in `MatMoETranslator/` whose
names start with `MatMoE` (the two bridge files + bridging header) plus
`Tokenizer.swift`, `TranslationDirection.swift`, and `TranslationViewModel.swift`.
Then wire your `project.yml` / `project.pbxproj` to:

1. Link `Frameworks/SlmEngine.xcframework` (do **not** embed — it's a static
   library wrapped as xcframework).
2. Set `SWIFT_OBJC_BRIDGING_HEADER` to the bridging header.
3. Add `Frameworks/SlmEngine.xcframework/{ios-arm64,ios-arm64_x86_64-simulator}/Headers`
   to `HEADER_SEARCH_PATHS` so the shim can `#include "slm_engine.h"`.
4. Add `-lc++` to `OTHER_LDFLAGS`.
5. Depend on the `Tokenizers` + `Hub` products of
   [`huggingface/swift-transformers`](https://github.com/huggingface/swift-transformers).

### Tokenizer requirements

The tokenizer.json **must** define the direction tokens as added or special
tokens (otherwise the encoder won't see the prefix as a single ID):

- `<translate-en-vi>`
- `<translate-vi-en>`

The pad / eos IDs in `TranslationViewModel` (`padId = 0`, `eosId = 1`) must
match what your model was trained with — they're the same defaults used in
`../export/export_tflite.py` and `../scripts/dump_prompt.py`.

---

## Troubleshooting

- **"tokenizer.json not found in the app bundle"** — drop it into
  `Resources/`, then rebuild. `Resources/` is added as a *folder reference*
  in `project.yml`, so anything dropped in becomes a bundled resource.
- **"encode_prefill.tflite not in bundle"** — same fix; check
  `Resources/README.md` for the file list.
- **"InvokeImpl failed" or shape mismatch on `Init`** — the `.tflite` was
  exported with a geometry that doesn't match the constants in
  `../include/slm_engine.h` (`kMaxSrcLen=128`, `kNumLayers=8`, …).
  Re-export with matching shapes or bump the constants and rebuild the
  xcframework.
- **Output is garbled / wrong language** — the prompt prefix probably isn't
  being tokenized as a single ID. Confirm `<translate-en-vi>` is in the
  tokenizer's added/special-token list.
- **Translation is very slow in the sim** — the sim doesn't have Neural
  Engine and x86_64 doesn't get the same XNNPACK paths as arm64. Expect
  noticeable but acceptable latency. Real hardware is ~5 ms/token.

---

## Why XcodeGen instead of a committed .xcodeproj?

`.xcodeproj` files are notoriously merge-hostile. `project.yml` is one human
file you can diff and review; the actual project is regenerated from it on
demand. This also means you can safely `rm -rf MatMoETranslator.xcodeproj`
and re-run `bootstrap.sh` whenever things drift.
