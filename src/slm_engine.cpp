// =============================================================================
//  slm_engine.cpp
//
//  Production-ready C++ implementation of the MatMoE SLM inference pipeline,
//  ported one-to-one from the JAX/TFLite Python reference at
//      export/export_tflite.py        (graph layout + tensor shapes)
//      export/moe_inference.py        (sampling: greedy + Top-K/Top-P)
//      export/benchmark_tflite.py     (decode loop driver)
//
//  Runtime  : LiteRT (TensorFlow Lite) C++ API.
//  Hardware : Mobile CPU + XNNPACK delegate (configurable thread count;
//             defaults to 4 per the project spec).
//
//  Hot-path discipline
//  -------------------
//   * No `new`, no `std::vector::resize`, no allocation of any kind inside
//     RunDecoderStep(). All buffers are sized once in Init() from
//     compile-time constants in slm_engine.h.
//   * The canonical self-attention KV state lives in `kv_cache_buffer_`. The
//     decoder graph reads its `self_kv` input via a custom allocation pointed
//     directly at this buffer (zero-copy IN). After each Invoke(), only the
//     *new* slot at position `step` for each of the 8 layers (32 KiB total)
//     is copied back from the TFLite output arena into the canonical buffer
//     (zero-copy OUT for everything except the freshly-produced slice).
//   * Encoder cross-attention K/V is computed once per source sentence and
//     parked in `cross_kv_buffer_` (zero-copy view into the decoder graph).
//
//  All shape / layout constants are duplicated as static_asserts against the
//  values used by the Python exporter so the engine refuses to load a model
//  whose geometry has silently drifted.
// =============================================================================

#include "slm_engine.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <vector>

// ---- LiteRT / TFLite C++ API ----
#include "tflite/core/interpreter_builder.h"
#include "tflite/interpreter.h"
#include "tflite/kernels/register.h"
#include "tflite/model_builder.h"

// ---- XNNPACK delegate (mobile CPU accelerator) ----
#include "tflite/delegates/xnnpack/xnnpack_delegate.h"

namespace matmoe {

namespace {

// `new float[N]` typically returns 8-16 byte aligned memory on common mobile
// toolchains. TFLite's SetCustomAllocationForTensor wants 64-byte alignment
// by default; we instead pass kTfLiteCustomAllocationFlagsSkipAlignCheck so
// our `new[]` / `delete[]` pair is safe with std::unique_ptr<T[]>.
constexpr int64_t kCustomAllocFlags = kTfLiteCustomAllocationFlagsSkipAlignCheck;

}  // namespace


// ============================================================================
//  Ctor / Dtor
// ============================================================================

SlmEngine::SlmEngine() : rng_(42) {}

SlmEngine::~SlmEngine() {
  // The Interpreters must be destroyed BEFORE the delegates they reference,
  // otherwise the delegate's PrepareNode pointers dangle during teardown.
  enc_interp_.reset();
  dec_interp_.reset();
  if (enc_xnnpack_) {
    TfLiteXNNPackDelegateDelete(enc_xnnpack_);
    enc_xnnpack_ = nullptr;
  }
  if (dec_xnnpack_) {
    TfLiteXNNPackDelegateDelete(dec_xnnpack_);
    dec_xnnpack_ = nullptr;
  }
}


// ============================================================================
//  Init
// ============================================================================

bool SlmEngine::Init(const std::string& encode_prefill_tflite_path,
                     const std::string& decode_step_tflite_path,
                     int num_threads) {
  last_error_.clear();

  // -------------------------------------------------------------------------
  // 1. Pre-allocate every working buffer ONCE.
  //    These outlive the interpreters and back the SetCustomAllocation calls.
  //    new[] pairs cleanly with std::unique_ptr<T[]>'s default delete[].
  // -------------------------------------------------------------------------
  kv_cache_buffer_.reset(new float[kSelfKvElems]);
  cross_kv_buffer_.reset(new float[kCrossKvElems]);
  src_mask_buffer_.reset(new int32_t[kMaxSrcLen]);
  if (!kv_cache_buffer_ || !cross_kv_buffer_ || !src_mask_buffer_) {
    last_error_ = "Failed to allocate KV-cache / src_mask buffers.";
    return false;
  }
  std::memset(kv_cache_buffer_.get(), 0, sizeof(float)   * kSelfKvElems);
  std::memset(cross_kv_buffer_.get(), 0, sizeof(float)   * kCrossKvElems);
  std::memset(src_mask_buffer_.get(), 0, sizeof(int32_t) * kMaxSrcLen);

  // -------------------------------------------------------------------------
  // 2. Build the two interpreters, each with its own XNNPACK delegate.
  // -------------------------------------------------------------------------
  if (!BuildEncoderInterpreter(encode_prefill_tflite_path, num_threads)) {
    return false;
  }
  if (!BuildDecoderInterpreter(decode_step_tflite_path, num_threads)) {
    return false;
  }

  // -------------------------------------------------------------------------
  // 3. Look up every input/output tensor by name (with shape fallback).
  // -------------------------------------------------------------------------
  if (!CacheEncoderTensorIndices()) return false;
  if (!CacheDecoderTensorIndices()) return false;

  // -------------------------------------------------------------------------
  // 4. Validate geometry against compile-time constants and read vocab_size_.
  // -------------------------------------------------------------------------
  {
    const TfLiteTensor* logits_t = dec_interp_->tensor(dec_out_logits_idx_);
    if (!logits_t || logits_t->dims == nullptr ||
        logits_t->dims->size != 2 || logits_t->dims->data[0] != 1) {
      last_error_ = "decode_step: unexpected logits tensor shape.";
      return false;
    }
    vocab_size_ = logits_t->dims->data[1];
  }

  // -------------------------------------------------------------------------
  // 5. Wire up custom allocations for zero-copy KV state.
  //
  //    * Decoder.self_kv  IN  -> kv_cache_buffer_   (read in place)
  //    * Decoder.cross_kv IN  -> cross_kv_buffer_   (read in place)
  //    * Decoder.src_mask IN  -> src_mask_buffer_   (read in place)
  //    * Encoder.cross_kv OUT -> cross_kv_buffer_   (written in place by
  //                                                  the encoder)
  //
  //    The self_kv_next OUTPUT stays in the TFLite arena: we only copy back
  //    the freshly-produced slot for position `step` after each Invoke().
  // -------------------------------------------------------------------------
  auto attach = [&](tflite::Interpreter* interp, int tensor_idx, void* data,
                    std::size_t bytes) -> bool {
    TfLiteCustomAllocation alloc{data, bytes};
    TfLiteStatus st = interp->SetCustomAllocationForTensor(
        tensor_idx, alloc, kCustomAllocFlags);
    return st == kTfLiteOk;
  };

  if (!attach(dec_interp_.get(), dec_in_self_kv_idx_,
              kv_cache_buffer_.get(),
              kSelfKvElems * sizeof(float))) {
    last_error_ = "Failed to attach kv_cache_buffer_ to decoder.self_kv.";
    return false;
  }
  if (!attach(dec_interp_.get(), dec_in_cross_kv_idx_,
              cross_kv_buffer_.get(),
              kCrossKvElems * sizeof(float))) {
    last_error_ = "Failed to attach cross_kv_buffer_ to decoder.cross_kv.";
    return false;
  }
  if (!attach(dec_interp_.get(), dec_in_src_mask_idx_,
              src_mask_buffer_.get(),
              kMaxSrcLen * sizeof(int32_t))) {
    last_error_ = "Failed to attach src_mask_buffer_ to decoder.src_mask.";
    return false;
  }
  if (!attach(enc_interp_.get(), enc_out_cross_kv_idx_,
              cross_kv_buffer_.get(),
              kCrossKvElems * sizeof(float))) {
    last_error_ = "Failed to attach cross_kv_buffer_ to encoder.cross_kv out.";
    return false;
  }

  // SetCustomAllocationForTensor requires re-running tensor allocation.
  if (enc_interp_->AllocateTensors() != kTfLiteOk) {
    last_error_ = "Encoder AllocateTensors() failed after custom allocations.";
    return false;
  }
  if (dec_interp_->AllocateTensors() != kTfLiteOk) {
    last_error_ = "Decoder AllocateTensors() failed after custom allocations.";
    return false;
  }

  // Seed the RNG with a deterministic value; users can override per-Generate.
  rng_.seed(42);

  return true;
}


// ============================================================================
//  Interpreter construction (XNNPACK, 4 threads, before AllocateTensors)
// ============================================================================

bool SlmEngine::BuildEncoderInterpreter(const std::string& path,
                                        int num_threads) {
  enc_model_ = tflite::FlatBufferModel::BuildFromFile(path.c_str());
  if (enc_model_ == nullptr) {
    last_error_ = "Failed to load TFLite model: " + path;
    return false;
  }

  tflite::ops::builtin::BuiltinOpResolver resolver;
  tflite::InterpreterBuilder builder(*enc_model_, resolver);
  builder.SetNumThreads(num_threads);
  if (builder(&enc_interp_) != kTfLiteOk || enc_interp_ == nullptr) {
    last_error_ = "Failed to build encoder Interpreter from: " + path;
    return false;
  }

  // -------- XNNPACK delegate, configured with the same thread count. --------
  TfLiteXNNPackDelegateOptions xopts = TfLiteXNNPackDelegateOptionsDefault();
  xopts.num_threads = num_threads;
  enc_xnnpack_      = TfLiteXNNPackDelegateCreate(&xopts);
  if (enc_xnnpack_ == nullptr) {
    last_error_ = "Failed to create XNNPACK delegate (encoder).";
    return false;
  }

  // CRITICAL: ModifyGraphWithDelegate MUST be called before AllocateTensors.
  if (enc_interp_->ModifyGraphWithDelegate(enc_xnnpack_) != kTfLiteOk) {
    last_error_ = "ModifyGraphWithDelegate failed for encoder/XNNPACK.";
    return false;
  }

  if (enc_interp_->AllocateTensors() != kTfLiteOk) {
    last_error_ = "Encoder AllocateTensors() failed.";
    return false;
  }
  return true;
}

bool SlmEngine::BuildDecoderInterpreter(const std::string& path,
                                        int num_threads) {
  dec_model_ = tflite::FlatBufferModel::BuildFromFile(path.c_str());
  if (dec_model_ == nullptr) {
    last_error_ = "Failed to load TFLite model: " + path;
    return false;
  }

  tflite::ops::builtin::BuiltinOpResolver resolver;
  tflite::InterpreterBuilder builder(*dec_model_, resolver);
  builder.SetNumThreads(num_threads);
  if (builder(&dec_interp_) != kTfLiteOk || dec_interp_ == nullptr) {
    last_error_ = "Failed to build decoder Interpreter from: " + path;
    return false;
  }

  TfLiteXNNPackDelegateOptions xopts = TfLiteXNNPackDelegateOptionsDefault();
  xopts.num_threads = num_threads;
  dec_xnnpack_      = TfLiteXNNPackDelegateCreate(&xopts);
  if (dec_xnnpack_ == nullptr) {
    last_error_ = "Failed to create XNNPACK delegate (decoder).";
    return false;
  }
  if (dec_interp_->ModifyGraphWithDelegate(dec_xnnpack_) != kTfLiteOk) {
    last_error_ = "ModifyGraphWithDelegate failed for decoder/XNNPACK.";
    return false;
  }
  if (dec_interp_->AllocateTensors() != kTfLiteOk) {
    last_error_ = "Decoder AllocateTensors() failed.";
    return false;
  }
  return true;
}


// ============================================================================
//  Tensor index discovery
//  --------------------------------------------------------------------------
//  Mirrors export_tflite._build_ep_index / _build_ds_index:
//   * Prefer name-based lookup ('source_ids', 'src_mask', 'token', 'step',
//     'self_kv', 'cross_kv', 'direction').
//   * Fall back to shape-based disambiguation when MAX_TGT_LEN == MAX_SRC_LEN
//     causes self_kv and cross_kv to share a shape.
// ============================================================================

namespace {

inline std::string LowerName(const TfLiteTensor* t) {
  if (!t || !t->name) return {};
  std::string s(t->name);
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return s;
}

}  // namespace

bool SlmEngine::CacheEncoderTensorIndices() {
  const auto& ins  = enc_interp_->inputs();
  const auto& outs = enc_interp_->outputs();

  // ---- Inputs: source_ids [1,128] int32 ; src_mask [1,128] int32 ----
  for (int idx : ins) {
    const TfLiteTensor* t = enc_interp_->tensor(idx);
    if (!t) continue;
    const std::string nm = LowerName(t);
    if (nm.find("source") != std::string::npos) enc_in_src_ids_idx_  = idx;
    else if (nm.find("src_mask") != std::string::npos ||
             nm.find("mask") != std::string::npos) enc_in_src_mask_idx_ = idx;
  }
  if (enc_in_src_ids_idx_ < 0 || enc_in_src_mask_idx_ < 0) {
    // Fallback: signature order.
    if (ins.size() >= 2) {
      enc_in_src_ids_idx_  = ins[0];
      enc_in_src_mask_idx_ = ins[1];
    } else {
      last_error_ = "encode_prefill: could not identify input tensors.";
      return false;
    }
  }

  // ---- Outputs: cross_kv [8,2,1,128,8,64] f32 ; direction_ids [1] i32 ----
  for (int idx : outs) {
    const TfLiteTensor* t = enc_interp_->tensor(idx);
    if (!t || !t->dims) continue;
    if (t->dims->size == 6) enc_out_cross_kv_idx_ = idx;
    else                    enc_out_dir_ids_idx_  = idx;
  }
  if (enc_out_cross_kv_idx_ < 0 || enc_out_dir_ids_idx_ < 0) {
    last_error_ = "encode_prefill: could not identify output tensors.";
    return false;
  }
  return true;
}

bool SlmEngine::CacheDecoderTensorIndices() {
  const auto& ins  = dec_interp_->inputs();
  const auto& outs = dec_interp_->outputs();

  std::vector<int> shape_1{};  // collected (1,)-shaped int32s: step + direction

  for (int idx : ins) {
    const TfLiteTensor* t = dec_interp_->tensor(idx);
    if (!t || !t->dims) continue;
    const std::string nm = LowerName(t);
    const TfLiteIntArray* sh = t->dims;

    if (sh->size == 2 && sh->data[0] == 1 && sh->data[1] == 1) {
      dec_in_token_idx_ = idx;
    } else if (sh->size == 6 && sh->data[0] == kNumLayers) {
      if (nm.find("cross") != std::string::npos) {
        dec_in_cross_kv_idx_ = idx;
      } else if (nm.find("self") != std::string::npos) {
        dec_in_self_kv_idx_ = idx;
      } else {
        // Disambiguation by sequence length axis 3.
        if (sh->data[3] == kMaxTgtLen && dec_in_self_kv_idx_ < 0) {
          dec_in_self_kv_idx_ = idx;
        } else if (dec_in_cross_kv_idx_ < 0) {
          dec_in_cross_kv_idx_ = idx;
        } else {
          dec_in_self_kv_idx_ = idx;
        }
      }
    } else if (sh->size == 2 && sh->data[0] == 1 && sh->data[1] == kMaxSrcLen) {
      dec_in_src_mask_idx_ = idx;
    } else if (sh->size == 1 && sh->data[0] == 1) {
      // step OR direction_ids; disambiguate by name first, fall back below.
      if (nm.find("direction") != std::string::npos) dec_in_dir_ids_idx_ = idx;
      else if (nm.find("step") != std::string::npos) dec_in_step_idx_    = idx;
      else                                            shape_1.push_back(idx);
    }
  }

  // Fallback: if neither name matched, the lower tensor index is `step`,
  // the higher is `direction_ids` (matches signature declaration order).
  if ((dec_in_step_idx_ < 0 || dec_in_dir_ids_idx_ < 0) &&
      shape_1.size() == 2) {
    std::sort(shape_1.begin(), shape_1.end());
    if (dec_in_step_idx_    < 0) dec_in_step_idx_    = shape_1[0];
    if (dec_in_dir_ids_idx_ < 0) dec_in_dir_ids_idx_ = shape_1[1];
  } else if (dec_in_step_idx_ < 0 && !shape_1.empty()) {
    dec_in_step_idx_ = shape_1.front();
  } else if (dec_in_dir_ids_idx_ < 0 && !shape_1.empty()) {
    dec_in_dir_ids_idx_ = shape_1.back();
  }

  // ---- Outputs ----
  for (int idx : outs) {
    const TfLiteTensor* t = dec_interp_->tensor(idx);
    if (!t || !t->dims) continue;
    if (t->dims->size == 2) dec_out_logits_idx_  = idx;
    else                    dec_out_self_kv_idx_ = idx;
  }

  if (dec_in_token_idx_   < 0 || dec_in_step_idx_     < 0 ||
      dec_in_self_kv_idx_ < 0 || dec_in_cross_kv_idx_ < 0 ||
      dec_in_src_mask_idx_< 0 || dec_in_dir_ids_idx_  < 0 ||
      dec_out_logits_idx_ < 0 || dec_out_self_kv_idx_ < 0) {
    last_error_ = "decode_step: failed to identify all I/O tensors.";
    return false;
  }
  return true;
}


// ============================================================================
//  RunEncoderPrefill
//  --------------------------------------------------------------------------
//  Encoder is invoked ONCE per source sentence; its output `cross_kv` lands
//  directly in cross_kv_buffer_ thanks to SetCustomAllocationForTensor.
// ============================================================================

bool SlmEngine::RunEncoderPrefill(const int32_t* src_ids,
                                  const int32_t* src_mask) {
  // 1. Stage src_mask into the shared buffer so the decoder graph sees it.
  std::memcpy(src_mask_buffer_.get(), src_mask, sizeof(int32_t) * kMaxSrcLen);

  // 2. Copy inputs into the encoder's typed tensors.
  int32_t* src_ids_tensor  =
      enc_interp_->typed_tensor<int32_t>(enc_in_src_ids_idx_);
  int32_t* src_mask_tensor =
      enc_interp_->typed_tensor<int32_t>(enc_in_src_mask_idx_);
  if (!src_ids_tensor || !src_mask_tensor) {
    last_error_ = "RunEncoderPrefill: typed_tensor() returned null.";
    return false;
  }
  std::memcpy(src_ids_tensor,  src_ids,  sizeof(int32_t) * kMaxSrcLen);
  std::memcpy(src_mask_tensor, src_mask, sizeof(int32_t) * kMaxSrcLen);

  // 3. Invoke.
  if (enc_interp_->Invoke() != kTfLiteOk) {
    last_error_ = "RunEncoderPrefill: Interpreter::Invoke() failed.";
    return false;
  }

  // 4. cross_kv_buffer_ is already populated in place.
  //    Pull direction_ids (single int32) out.
  const int32_t* dir_t =
      enc_interp_->typed_tensor<int32_t>(enc_out_dir_ids_idx_);
  direction_id_ = dir_t ? dir_t[0] : 0;
  return true;
}


// ============================================================================
//  RunDecoderStep
//  --------------------------------------------------------------------------
//  ZERO heap allocation. The hot loop performs:
//    1. Write token_id, step, direction_ids into typed_input_tensor<int32_t>.
//       (self_kv / cross_kv / src_mask are already custom-allocated, so they
//        don't need a copy here.)
//    2. Invoke.
//    3. Slide-window update: copy ONLY the freshly written K/V slot at
//       position `step` for each layer from the TFLite output arena back
//       into the canonical kv_cache_buffer_. That's
//          kNumLayers * 2 * kNumHeads * kHeadDim = 8 * 2 * 8 * 64 = 8192
//          floats == 32 KiB, regardless of how big the full cache is.
//    4. Return a pointer to logits[0, :] in the TFLite arena (valid until
//       the next Invoke()).
// ============================================================================

const float* SlmEngine::RunDecoderStep(int step, int32_t prev_token) {
  // ---- 1. Scalar inputs ----
  int32_t* tok_t = dec_interp_->typed_tensor<int32_t>(dec_in_token_idx_);
  int32_t* stp_t = dec_interp_->typed_tensor<int32_t>(dec_in_step_idx_);
  int32_t* dir_t = dec_interp_->typed_tensor<int32_t>(dec_in_dir_ids_idx_);
  if (!tok_t || !stp_t || !dir_t) {
    last_error_ = "RunDecoderStep: typed_tensor() returned null.";
    return nullptr;
  }
  tok_t[0] = prev_token;
  stp_t[0] = step;
  dir_t[0] = direction_id_;

  // ---- 2. Invoke ----
  if (dec_interp_->Invoke() != kTfLiteOk) {
    last_error_ = "RunDecoderStep: Interpreter::Invoke() failed.";
    return nullptr;
  }

  // ---- 3. Sliding-window KV write-back ----
  //
  // Layout of self_kv: [LAYERS, 2(K/V), 1(batch), TGT_LEN, HEADS, HEAD_DIM]
  // Innermost contiguous run for one (layer, kv, batch, step) tuple is
  // HEADS * HEAD_DIM = kStepSliceElemsPerKv floats.
  //
  // For each of the 8 layers, copy the K-slot and the V-slot at index `step`
  // from the output arena into the matching offset in kv_cache_buffer_.
  const float* self_kv_out =
      dec_interp_->typed_tensor<float>(dec_out_self_kv_idx_);
  if (!self_kv_out) {
    last_error_ = "RunDecoderStep: self_kv_next typed_tensor() returned null.";
    return nullptr;
  }

  constexpr std::size_t kv_axis_stride =
      static_cast<std::size_t>(kMaxTgtLen) * kNumHeads * kHeadDim;  // per K-or-V
  constexpr std::size_t layer_stride   = 2 * kv_axis_stride;        // K + V

  float*       dst_base = kv_cache_buffer_.get();
  const float* src_base = self_kv_out;
  const std::size_t step_off =
      static_cast<std::size_t>(step) * kStepSliceElemsPerKv;
  const std::size_t slice_bytes = kStepSliceElemsPerKv * sizeof(float);

  for (int l = 0; l < kNumLayers; ++l) {
    const std::size_t lo = static_cast<std::size_t>(l) * layer_stride;
    // K slot
    std::memcpy(dst_base + lo + step_off,
                src_base + lo + step_off,
                slice_bytes);
    // V slot (offset by one kv_axis_stride)
    std::memcpy(dst_base + lo + kv_axis_stride + step_off,
                src_base + lo + kv_axis_stride + step_off,
                slice_bytes);
  }

  // ---- 4. Return raw logits pointer ----
  return dec_interp_->typed_tensor<float>(dec_out_logits_idx_);
}


// ============================================================================
//  Generate
//  --------------------------------------------------------------------------
//  End-to-end driver. Mirrors the Python loop in benchmark_tflite.translate():
//    token = pad_id
//    for s in [0, max_new_tokens):
//        logits = decode_step(token, s, ...)
//        next   = sample(logits)        # greedy or top-k/top-p
//        out.append(next); token = next
//        if next == eos_id: break
// ============================================================================

int SlmEngine::Generate(const int32_t* src_ids,
                        const int32_t* src_mask,
                        const GenerationConfig& cfg,
                        int32_t* out_tokens,
                        int      out_tokens_capacity) {
  if (!RunEncoderPrefill(src_ids, src_mask)) return 0;

  // Reset self-KV cache for the new sentence.
  std::memset(kv_cache_buffer_.get(), 0, sizeof(float) * kSelfKvElems);

  if (cfg.sampling.method == SamplingOptions::kSample) {
    rng_.seed(cfg.sampling.seed);
  }

  const int32_t pad_id = cfg.pad_id;
  const int32_t eos_id = cfg.eos_id;
  const int     limit  = std::min(cfg.max_new_tokens,
                                  std::min(kMaxTgtLen, out_tokens_capacity));

  int32_t prev_tok  = pad_id;       // first decoder step receives PAD as BOS
  int     n_written = 0;

  for (int s = 0; s < limit; ++s) {
    const float* logits = RunDecoderStep(s, prev_tok);
    if (logits == nullptr) return n_written;

    int32_t next_tok;
    if (cfg.sampling.method == SamplingOptions::kGreedy) {
      next_tok = SampleGreedy(logits);
    } else {
      next_tok = SampleTopKTopP(logits,
                                cfg.sampling.temperature,
                                cfg.sampling.top_k,
                                cfg.sampling.top_p);
    }

    out_tokens[n_written++] = next_tok;
    if (next_tok == eos_id) break;
    prev_tok = next_tok;
  }
  return n_written;
}


// ============================================================================
//  Sampling
//  --------------------------------------------------------------------------
//  Greedy:  argmax (matches generate_fast_greedy_jitted).
//  Sample:  temperature -> Top-K -> Top-P (nucleus) -> categorical draw.
//           Matches generate_fast_sample_jitted in export/moe_inference.py:
//
//     next_logits = logits[:,0,:] / temperature
//     topk = top_k(next_logits, top_k)
//     next_logits[next_logits < topk[-1]] = -inf
//     probs       = softmax(next_logits)
//     sorted_p    = sort(probs, descending)
//     cumsum      = cumsum(sorted_p)
//     mask        = cumsum < top_p
//     mask        = concat([[1], mask[:-1]])           # always keep top-1
//     thresh      = min(sorted_p where mask, else 1.0)
//     next_logits[probs < thresh] = -inf
//     next_token  = categorical(next_logits)
// ============================================================================

int32_t SlmEngine::SampleGreedy(const float* logits) const {
  int   best_i = 0;
  float best_v = logits[0];
  for (int i = 1; i < vocab_size_; ++i) {
    const float v = logits[i];
    if (v > best_v) { best_v = v; best_i = i; }
  }
  return static_cast<int32_t>(best_i);
}

int32_t SlmEngine::SampleTopKTopP(const float* logits_in,
                                  float temperature,
                                  int   top_k,
                                  float top_p) {
  // We avoid allocating per-call by using a thread-local scratch sized to the
  // largest vocabulary we expect. 64 K covers every plausible SLM tokenizer.
  static constexpr int kMaxVocab = 65536;
  thread_local std::array<float, kMaxVocab> scaled_logits{};
  thread_local std::array<int,   kMaxVocab> indices{};
  thread_local std::array<float, kMaxVocab> probs{};

  const int V = vocab_size_;
  if (V <= 0 || V > kMaxVocab) {
    // Fall back to greedy if vocab is out of range (safety net).
    return SampleGreedy(logits_in);
  }
  const int k = std::min(std::max(top_k, 1), V);
  const float inv_T = (temperature > 0.f) ? (1.f / temperature) : 1.f;

  // ---- 1. Temperature scaling. ----
  for (int i = 0; i < V; ++i) {
    scaled_logits[i] = logits_in[i] * inv_T;
    indices[i]       = i;
  }

  // ---- 2. Top-K filter: partial_sort indices by descending logit. ----
  std::partial_sort(
      indices.begin(), indices.begin() + k, indices.begin() + V,
      [&](int a, int b) { return scaled_logits[a] > scaled_logits[b]; });

  const float kth_logit = scaled_logits[indices[k - 1]];

  // Mask everything below the K-th into -inf.
  const float neg_inf = -std::numeric_limits<float>::infinity();
  for (int i = 0; i < V; ++i) {
    if (scaled_logits[i] < kth_logit) scaled_logits[i] = neg_inf;
  }

  // ---- 3. Softmax over (filtered) logits. ----
  float max_l = neg_inf;
  for (int i = 0; i < V; ++i) {
    if (scaled_logits[i] > max_l) max_l = scaled_logits[i];
  }
  float sum_e = 0.f;
  for (int i = 0; i < V; ++i) {
    const float e = (scaled_logits[i] == neg_inf)
                        ? 0.f
                        : std::exp(scaled_logits[i] - max_l);
    probs[i] = e;
    sum_e   += e;
  }
  const float inv_sum = (sum_e > 0.f) ? (1.f / sum_e) : 0.f;
  for (int i = 0; i < V; ++i) probs[i] *= inv_sum;

  // ---- 4. Top-P (nucleus) filter on the surviving Top-K candidates. ----
  //
  // Sort the K survivors by probability (descending), compute cumulative
  // mass, find the smallest prefix whose mass >= top_p, and mask anything
  // strictly below that prefix's probability threshold.
  std::sort(indices.begin(), indices.begin() + k,
            [&](int a, int b) { return probs[a] > probs[b]; });

  float cum  = 0.f;
  float thresh_p = probs[indices[0]];           // always keep top-1
  for (int j = 0; j < k; ++j) {
    cum += probs[indices[j]];
    // The JAX code prepends [1] to `cumsum<top_p`, effectively keeping the
    // first token unconditionally. The cutoff sits at the smallest j such
    // that cum >= top_p; from there on tokens are dropped.
    if (cum >= top_p) {
      thresh_p = probs[indices[j]];
      break;
    }
  }

  // Re-normalise only over the surviving Top-K ∩ Top-P set.
  double surviving_sum = 0.0;
  for (int j = 0; j < k; ++j) {
    if (probs[indices[j]] < thresh_p) probs[indices[j]] = 0.f;
    surviving_sum += probs[indices[j]];
  }
  if (surviving_sum <= 0.0) {
    // Numerical degenerate case: just return top-1.
    return static_cast<int32_t>(indices[0]);
  }
  const float renorm = static_cast<float>(1.0 / surviving_sum);

  // ---- 5. Categorical draw via inverse-CDF on the surviving K candidates.
  std::uniform_real_distribution<float> u01(0.f, 1.f);
  const float r = u01(rng_);
  float acc = 0.f;
  for (int j = 0; j < k; ++j) {
    acc += probs[indices[j]] * renorm;
    if (r <= acc) return static_cast<int32_t>(indices[j]);
  }
  return static_cast<int32_t>(indices[k - 1]);
}

}  // namespace matmoe
