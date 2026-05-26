// =============================================================================
//  slm_engine.h
//
//  Production-ready C++ inference engine for the MatMoE Small Language Model
//  exported to TFLite as a two-graph KV-cache decoder:
//
//      encode_prefill.tflite : (source_ids, src_mask)
//                              -> (cross_kv, direction_ids)
//
//      decode_step.tflite    : (token_id, step, self_kv, cross_kv,
//                               src_mask, direction_ids)
//                              -> (logits, self_kv_next)
//
//  Target platform : Mobile CPU (Android / iOS)
//  Runtime         : LiteRT (TensorFlow Lite) C++ API + XNNPACK delegate
//  Threads         : 4
//
//  Design goals
//  ------------
//   * Zero heap allocation inside RunDecoderStep().
//     All KV-cache state lives in pre-allocated, 64-byte aligned, contiguous
//     buffers owned by the engine (kv_cache_buffer_ / cross_kv_buffer_).
//   * "Sliding window" pointer math: only the freshly produced K/V slot for
//     position `step` is copied back from the TFLite output arena into the
//     canonical kv_cache_buffer_ each step (~32 KB instead of ~4 MB).
//   * Sampling matches the JAX reference exactly: greedy = argmax, sample =
//     temperature scaling -> Top-K filter -> Top-P (nucleus) filter ->
//     categorical draw.
// =============================================================================

#ifndef MATMOE_INFERENCE_SLM_ENGINE_H_
#define MATMOE_INFERENCE_SLM_ENGINE_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <vector>

// We need the real TFLite types here because tflite::Interpreter and
// tflite::FlatBufferModel are aliases for types in the `tflite::impl::`
// namespace; a bare forward declaration like `namespace tflite { class
// Interpreter; }` produces a *different* type and breaks unique_ptr
// construction in the .cpp.
#include "tflite/interpreter.h"
#include "tflite/model_builder.h"

struct TfLiteDelegate;

namespace matmoe {

// ---- Model geometry (mirrors export/export_tflite.py) ----------------------
constexpr int kMaxSrcLen   = 128;
constexpr int kMaxTgtLen   = 128;
constexpr int kDModel      = 512;
constexpr int kNumLayers   = 8;
constexpr int kNumHeads    = 8;
constexpr int kHeadDim     = kDModel / kNumHeads;     // 64

// cross_kv : [NUM_LAYERS, 2, 1, MAX_SRC_LEN, NUM_HEADS, HEAD_DIM] f32
// self_kv  : [NUM_LAYERS, 2, 1, MAX_TGT_LEN, NUM_HEADS, HEAD_DIM] f32
constexpr std::size_t kCrossKvElems =
    static_cast<std::size_t>(kNumLayers) * 2 * 1 * kMaxSrcLen * kNumHeads * kHeadDim;
constexpr std::size_t kSelfKvElems =
    static_cast<std::size_t>(kNumLayers) * 2 * 1 * kMaxTgtLen * kNumHeads * kHeadDim;

// Number of floats that change per decode step (one K + one V slot per layer).
constexpr std::size_t kStepSliceElemsPerKv =
    static_cast<std::size_t>(kNumHeads) * kHeadDim;             // 512
constexpr std::size_t kStepSliceElemsPerLayer =
    2 * kStepSliceElemsPerKv;                                   // 1024 (K + V)


// ---- Sampling configuration ------------------------------------------------
struct SamplingOptions {
  enum Method { kGreedy, kSample };

  Method   method      = kGreedy;
  float    temperature = 0.7f;
  int      top_k       = 40;
  float    top_p       = 0.9f;
  uint64_t seed        = 42;
};


// ---- Generation configuration ----------------------------------------------
struct GenerationConfig {
  int   max_new_tokens = kMaxTgtLen;   // upper bound on decoded length
  int32_t pad_id       = 0;
  int32_t eos_id       = 1;
  SamplingOptions sampling{};
};


// ---- Engine ----------------------------------------------------------------
class SlmEngine {
 public:
  SlmEngine();
  ~SlmEngine();

  SlmEngine(const SlmEngine&)            = delete;
  SlmEngine& operator=(const SlmEngine&) = delete;

  // Load both TFLite graphs, wire up the XNNPACK delegate (num_threads),
  // cache I/O tensor indices, and pre-allocate every working buffer.
  //
  // Returns false on any failure; call LastError() for a human-readable reason.
  bool Init(const std::string& encode_prefill_tflite_path,
            const std::string& decode_step_tflite_path,
            int num_threads = 4);

  // End-to-end translation/generation.
  //
  //   src_ids   : kMaxSrcLen int32 token IDs, padded to kMaxSrcLen
  //   src_mask  : kMaxSrcLen int32 attention mask (1 = real token, 0 = pad)
  //   out_tokens: receives generated token IDs (NOT including the BOS pad and
  //               NOT including a trailing EOS).
  //
  // Returns the number of tokens written. No heap allocation occurs during
  // the decode loop.
  int Generate(const int32_t* src_ids,
               const int32_t* src_mask,
               const GenerationConfig& cfg,
               int32_t* out_tokens,
               int      out_tokens_capacity);

  // Lower-level entry points (exposed for benchmarking / streaming).
  bool   RunEncoderPrefill(const int32_t* src_ids,
                           const int32_t* src_mask);   // populates cross_kv_buffer_ + direction_id_
  // Runs ONE decode step. `step` is the 0-indexed position to write into the
  // self-KV cache. Returns the raw logits pointer (length = vocab_size_),
  // valid until the next call.
  const float* RunDecoderStep(int step, int32_t prev_token);

  // Inspect last error string (for logging).
  const std::string& LastError() const { return last_error_; }

  int  VocabSize() const { return vocab_size_; }

 private:
  // ----- Helpers -----
  bool BuildEncoderInterpreter(const std::string& path, int num_threads);
  bool BuildDecoderInterpreter(const std::string& path, int num_threads);
  bool CacheEncoderTensorIndices();
  bool CacheDecoderTensorIndices();

  // Sampling
  int32_t SampleGreedy(const float* logits) const;
  int32_t SampleTopKTopP(const float* logits,
                         float temperature,
                         int   top_k,
                         float top_p);

  // ----- TFLite state -----
  std::unique_ptr<tflite::FlatBufferModel> enc_model_;
  std::unique_ptr<tflite::FlatBufferModel> dec_model_;
  std::unique_ptr<tflite::Interpreter>     enc_interp_;
  std::unique_ptr<tflite::Interpreter>     dec_interp_;

  // Engine retains ownership of the XNNPACK delegates because
  // ModifyGraphWithDelegate (non-owning overload) does not.
  TfLiteDelegate* enc_xnnpack_ = nullptr;
  TfLiteDelegate* dec_xnnpack_ = nullptr;

  // ----- Cached tensor indices (Interpreter input/output array indices) -----
  // Encoder
  int enc_in_src_ids_idx_   = -1;
  int enc_in_src_mask_idx_  = -1;
  int enc_out_cross_kv_idx_ = -1;
  int enc_out_dir_ids_idx_  = -1;

  // Decoder
  int dec_in_token_idx_     = -1;
  int dec_in_step_idx_      = -1;
  int dec_in_self_kv_idx_   = -1;
  int dec_in_cross_kv_idx_  = -1;
  int dec_in_src_mask_idx_  = -1;
  int dec_in_dir_ids_idx_   = -1;
  int dec_out_logits_idx_   = -1;
  int dec_out_self_kv_idx_  = -1;

  // ----- Pre-allocated buffers (NO allocation in hot loop) -----
  // Canonical KV-cache state, owned by the engine. Used as a sliding window:
  // each step writes only the slot at position `step` per layer.
  std::unique_ptr<float[]>  kv_cache_buffer_;     // kSelfKvElems  floats
  std::unique_ptr<float[]>  cross_kv_buffer_;     // kCrossKvElems floats
  std::unique_ptr<int32_t[]> src_mask_buffer_;    // kMaxSrcLen    int32
  int32_t                   direction_id_ = 0;

  // ----- Runtime metadata -----
  int          vocab_size_  = 0;
  std::string  last_error_;
  std::mt19937_64 rng_;
};

}  // namespace matmoe

#endif  // MATMOE_INFERENCE_SLM_ENGINE_H_
