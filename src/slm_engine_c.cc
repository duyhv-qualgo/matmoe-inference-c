// =============================================================================
//  slm_engine_c.cc — implementation of the extern "C" wrapper declared in
//  slm_engine_c.h. Thin forwarders to matmoe::SlmEngine.
// =============================================================================

#include "slm_engine_c.h"

#include "slm_engine.h"

struct MatMoeEngineOpaque {
  matmoe::SlmEngine engine;
};

static_assert(matmoe::kMaxSrcLen == MATMOE_MAX_SRC_LEN,
              "MATMOE_MAX_SRC_LEN must mirror matmoe::kMaxSrcLen");
static_assert(matmoe::kMaxTgtLen == MATMOE_MAX_TGT_LEN,
              "MATMOE_MAX_TGT_LEN must mirror matmoe::kMaxTgtLen");

extern "C" MatMoeEngine matmoe_engine_create(void) {
  return new MatMoeEngineOpaque();
}

extern "C" void matmoe_engine_destroy(MatMoeEngine eng) {
  delete eng;
}

extern "C" int matmoe_engine_load(MatMoeEngine eng,
                                   const char* encoder_path,
                                   const char* decoder_path,
                                   int num_threads) {
  return eng->engine.Init(encoder_path, decoder_path, num_threads) ? 1 : 0;
}

extern "C" int matmoe_engine_vocab_size(MatMoeEngine eng) {
  return eng->engine.VocabSize();
}

extern "C" const char* matmoe_engine_last_error(MatMoeEngine eng) {
  return eng->engine.LastError().c_str();
}

extern "C" int matmoe_engine_generate(MatMoeEngine eng,
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
                                       int      out_tokens_capacity) {
  matmoe::GenerationConfig cfg;
  cfg.max_new_tokens   = max_new_tokens;
  cfg.pad_id           = pad_id;
  cfg.eos_id           = eos_id;
  cfg.sampling.method  = (method == MATMOE_SAMPLING_TOP_K_TOP_P)
                          ? matmoe::SamplingOptions::kSample
                          : matmoe::SamplingOptions::kGreedy;
  cfg.sampling.temperature = temperature;
  cfg.sampling.top_k       = top_k;
  cfg.sampling.top_p       = top_p;
  cfg.sampling.seed        = seed;
  return eng->engine.Generate(src_ids, src_mask, cfg,
                              out_tokens, out_tokens_capacity);
}
