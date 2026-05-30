# matmoe-inference-c

Production C++ inference engine for the **MatMoE Small Language Model**
(en↔vi translation), running on mobile CPUs via LiteRT / TensorFlow Lite + the
XNNPACK delegate.

The model is a JAX-trained encoder–decoder with cross-attention and a Mixture
of Experts FFN, exported to two `.tflite` graphs:

```
encode_prefill.tflite : (source_ids, src_mask) → (cross_kv, direction_ids)
decode_step.tflite    : (token_id, step, self_kv, cross_kv, src_mask, direction_ids)
                        → (logits, self_kv_next)
```

`SlmEngine` runs the full decode loop in C++: encoder prefill once per
sentence, then a tight per-token loop that updates the self-attention
KV-cache in place and stops on EOS.

---

## Performance

| Metric | macOS arm64 (M-series) |
|---|---|
| `Init` (load both `.tflite`, attach XNNPACK ×2, custom KV allocation) | ~0.2 s |
| Encoder prefill | ~20 ms |
| Decode step | **~5 ms / token** (4 threads, XNNPACK) |
| Sample output (greedy, en→vi) | `Hello, how are you today?` → `Chào, hôm nay anh khoẻ không?` |

Hot loop is allocation-free: KV cache lives in a pre-allocated
`kv_cache_buffer_` owned by the engine, and only the freshly-produced K/V slot
at position `step` (32 KiB) is copied back from the TFLite output arena each
iteration — sliding-window pointer math instead of a full 4 MB blit.

---

## Project layout

```
matmoe-inference-c/
├── include/
│   └── slm_engine.h            ← public API
├── src/
│   ├── slm_engine.cpp          ← engine implementation
│   └── test_main.cc            ← CLI test harness
├── scripts/
│   ├── dump_prompt.py          ← text → prompt.bin
│   ├── decode_output.py        ← out_ids.bin → text
│   ├── build_ios.sh            ← ship to iOS (xcframework)
│   └── build_android.sh        ← ship to Android (libslm_engine.so)
├── models/
│   └── dim-256/                ← exported .tflite (~150 MB total)
│       ├── encode_prefill.tflite
│       └── decode_step.tflite
├── export/                     ← Python reference: export_tflite.py,
│                                 moe_inference.py, benchmark_tflite.py
├── litert/                     ← vendored TFLite source tree
│                                 (used as a CMake subdirectory; the build
│                                  FetchContents the matching TensorFlow source)
├── CMakeLists.txt              ← top-level build wiring
└── README.md
```

---

## Public API (`include/slm_engine.h`)

```cpp
#include "slm_engine.h"

matmoe::SlmEngine eng;
eng.Init("encode_prefill.tflite", "decode_step.tflite", /*threads=*/4);

matmoe::GenerationConfig cfg;
cfg.pad_id = 0;
cfg.eos_id = 1;
cfg.max_new_tokens = 60;

// Greedy
cfg.sampling.method = matmoe::SamplingOptions::kGreedy;

// — or — Sampling (matches Python: temperature → top-k → top-p → categorical)
cfg.sampling.method      = matmoe::SamplingOptions::kSample;
cfg.sampling.temperature = 0.8f;
cfg.sampling.top_k       = 40;
cfg.sampling.top_p       = 0.9f;
cfg.sampling.seed        = 42;

int32_t out[matmoe::kMaxTgtLen];
int n = eng.Generate(src_ids, src_mask, cfg, out, matmoe::kMaxTgtLen);
// out[0..n-1] = generated token IDs (NOT including a leading BOS pad;
//                INCLUDING a trailing EOS if the model produced one)
```

The engine is **tokenizer-free** by design. Tokenize / detokenize at the
caller (`tokenizers-cpp`, SentencePiece, or Python via `scripts/`).

---

## Build (desktop / macOS)

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --target test_slm
```

First build: **~25 min** (compiles TensorFlow Lite + XNNPACK from source via
`FetchContent`). After that, edits to `src/` rebuild in **~10-30 s**.

### Run the smoke test

```bash
./build/test_slm \
    models/dim-256/encode_prefill.tflite \
    models/dim-256/decode_step.tflite \
    4
```

### Real translation round-trip (Python tokenizer + C++ inference)

```bash
# 1. Tokenize a sentence to prompt.bin
python3 scripts/dump_prompt.py \
    --tokenizer /path/to/tokenizer.json \
    --text "<translate-en-vi> Hello, how are you today?" \
    --out prompt.bin

# 2. Run inference (greedy)
./build/test_slm \
    models/dim-256/encode_prefill.tflite \
    models/dim-256/decode_step.tflite 4 \
    --prompt-bin prompt.bin --out-bin out_ids.bin \
    --max-new 60 --method greedy

# 3. Detokenize
python3 scripts/decode_output.py \
    --tokenizer /path/to/tokenizer.json \
    --in out_ids.bin
```

For sampling: `--method sample --temperature 0.8 --top-k 40 --top-p 0.9 --seed 42`.

---

## Ship to mobile

### iOS

```bash
./scripts/build_ios.sh
```

Produces `dist/SlmEngine.xcframework` (iphoneos arm64 + iphonesimulator
arm64+x86_64) and `dist/include/slm_engine.h`. Drop the xcframework into
Xcode → "Frameworks, Libraries, and Embedded Content"; bridge from Swift via
an Objective-C++ `.mm` shim.

```swift
let bridge = MatMoEBridge()
bridge.load(encoderPath: encPath, decoderPath: decPath, threads: 4)

let (ids, mask) = tokenizer.encode("<translate-en-vi> Hello, how are you?")
let outIds = bridge.generate(srcIds: ids, srcMask: mask,
                             maxNewTokens: 60, padId: 0, eosId: 1,
                             method: .greedy,
                             temperature: 0.7, topK: 40, topP: 0.9, seed: 42)
let text   = tokenizer.decode(outIds.map { $0.int32Value })
```

A working end-to-end sample (SwiftUI app + Obj-C++ shim + swift-transformers
tokenizer + XcodeGen) lives in [`sample-ios-app/`](sample-ios-app/README.md).
Build + launch on an iPhone simulator with:

```bash
cd sample-ios-app
# drop encode_prefill.tflite, decode_step.tflite, tokenizer.json into Resources/
./scripts/bootstrap.sh     # builds xcframework + generates the .xcodeproj
./scripts/run_sim.sh       # boots iPhone sim, builds, installs, launches
```

### Android

```bash
./scripts/build_android.sh                  # arm64-v8a only (default)
./scripts/build_android.sh --with-emu       # + x86_64 for Studio emulator
./scripts/build_android.sh --with-armv7     # + armeabi-v7a for very old devices
```

Produces `dist/android/jniLibs/<abi>/libslm_engine.so` and the header. Drop
into `app/src/main/jniLibs/<abi>/` and write a JNI shim in
`app/src/main/cpp/slm_jni.cpp`. Kotlin call site:

```kotlin
class SlmEngine(encPath: String, decPath: String, threads: Int = 4) {
    private val handle: Long = nativeInit(encPath, decPath, threads)
    fun generate(srcIds: IntArray, srcMask: IntArray,
                 maxNew: Int = 60, padId: Int = 0, eosId: Int = 1): IntArray =
        nativeGenerate(handle, srcIds, srcMask, maxNew, padId, eosId)

    private external fun nativeInit(enc: String, dec: String, threads: Int): Long
    private external fun nativeGenerate(handle: Long, src: IntArray, mask: IntArray,
                                        maxNew: Int, padId: Int, eosId: Int): IntArray

    companion object { init { System.loadLibrary("slm_engine") } }
}
```

### Tokenizer on device

The engine ships **without** a tokenizer (intentional: matches the Python
split). For real translation in your app, link one of:

- [`tokenizers-cpp`](https://github.com/mlc-ai/tokenizers-cpp) — drop-in
  loader for the same `tokenizer.json` used here. Recommended.
- SentencePiece C++ (if you re-export the `.model` file).
- Native Swift/Kotlin port (e.g. `huggingface/swift-transformers`).

---

## Internals (one-line summary per file)

| File | What it does |
|---|---|
| `include/slm_engine.h`    | `SlmEngine` class, `GenerationConfig`, geometry constants, `SamplingOptions` |
| `src/slm_engine.cpp`      | Engine: TFLite + XNNPACK setup, tensor-index discovery, encode prefill, decode loop, greedy + Top-K/Top-P sampling |
| `src/test_main.cc`        | CLI driver: dummy smoke test + `--prompt-bin / --out-bin` round-trip |
| `scripts/dump_prompt.py`  | HuggingFace tokenizer → packed int32 `prompt.bin` (`src_ids`+`src_mask`) |
| `scripts/decode_output.py`| Reads `out_ids.bin`, detokenizes via HuggingFace tokenizer |
| `scripts/build_ios.sh`    | iOS device + simulator → merged `.a` → `SlmEngine.xcframework` |
| `scripts/build_android.sh`| Per-ABI cross-compile → single `libslm_engine.so` per ABI |
| `CMakeLists.txt`          | Wires `litert/tflite/` as a subdirectory, builds `slm_engine` lib + `test_slm` |

---

## Geometry (compile-time constants in `slm_engine.h`)

| Constant | Value | Source |
|---|---|---|
| `kMaxSrcLen` | 128 | `export/export_tflite.py:51` |
| `kMaxTgtLen` | 128 | same |
| `kDModel`    | 512 | same |
| `kNumLayers` | 8   | same |
| `kNumHeads`  | 8   | same |
| `kHeadDim`   | 64  | `kDModel / kNumHeads` |
| `vocab_size` | autodetected at `Init` from `decode_step` logits | runtime |

If you re-export the model with different geometry (different `max_seq_len`,
deeper stack, etc.), bump the constants in `slm_engine.h` and rebuild. The
engine **refuses to load** a graph whose shapes don't match the constants it
was compiled against.
