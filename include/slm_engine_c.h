// =============================================================================
//  slm_engine_c.h — C-only API around matmoe::SlmEngine for cross-language
//  bindings (Obj-C/Swift, JNI/Kotlin, etc.).
//
//  Unlike slm_engine.h this header has NO TFLite, STL, or other C++ types in
//  it, so it's safe to ship as the only header inside a packaged framework
//  (xcframework, AAR, etc.).
//
//  The engine itself stays in slm_engine.cpp; this is a thin extern "C" shim.
// =============================================================================

#ifndef MATMOE_INFERENCE_SLM_ENGINE_C_H_
#define MATMOE_INFERENCE_SLM_ENGINE_C_H_

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Model geometry constants — mirror the C++ constants in slm_engine.h so
/// callers can size their buffers without including the C++ header.
#define MATMOE_MAX_SRC_LEN 128
#define MATMOE_MAX_TGT_LEN 128

typedef enum {
  MATMOE_SAMPLING_GREEDY      = 0,
  MATMOE_SAMPLING_TOP_K_TOP_P = 1,
} MatMoeSamplingMethod;

/// Opaque handle. Create with matmoe_engine_create, free with destroy.
typedef struct MatMoeEngineOpaque* MatMoeEngine;

MatMoeEngine matmoe_engine_create(void);
void         matmoe_engine_destroy(MatMoeEngine eng);

/// Returns 1 on success, 0 on failure. On failure, matmoe_engine_last_error
/// returns a human-readable reason.
int matmoe_engine_load(MatMoeEngine eng,
                       const char* encoder_path,
                       const char* decoder_path,
                       int num_threads);

int         matmoe_engine_vocab_size(MatMoeEngine eng);
const char* matmoe_engine_last_error(MatMoeEngine eng);

/// One-shot translation/generation.
///
///   src_ids   : MATMOE_MAX_SRC_LEN int32 token IDs (padded with pad_id).
///   src_mask  : MATMOE_MAX_SRC_LEN int32 attention mask (1 = real, 0 = pad).
///   out_tokens: caller-owned buffer; capacity must be >= MATMOE_MAX_TGT_LEN.
///
/// Returns the number of tokens written, or < 0 on error.
int matmoe_engine_generate(MatMoeEngine eng,
                           const int32_t* src_ids,
                           const int32_t* src_mask,
                           int   max_new_tokens,
                           int32_t pad_id,
                           int32_t eos_id,
                           MatMoeSamplingMethod method,
                           float temperature,
                           int   top_k,
                           float top_p,
                           uint64_t seed,
                           int32_t* out_tokens,
                           int      out_tokens_capacity);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // MATMOE_INFERENCE_SLM_ENGINE_C_H_
