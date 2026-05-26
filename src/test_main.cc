// =============================================================================
//  test_main.cc — driver / smoke test for the MatMoE SLM engine.
//
//  Two modes:
//    1) Dummy smoke test (default): feeds a fake single-token prompt and
//       runs a few greedy + sampled steps. Validates engine wiring only.
//    2) Real translation: --prompt-bin prompt.bin produced by
//       scripts/dump_prompt.py; writes generated IDs to --out-bin so
//       scripts/decode_output.py can detokenize them.
//
//  Usage:
//    Dummy:
//      test_slm <encode.tflite> <decode.tflite> [num_threads]
//    Real:
//      test_slm <encode.tflite> <decode.tflite> [num_threads] \
//               --prompt-bin prompt.bin --out-bin out_ids.bin \
//               [--max-new 60] [--method greedy|sample] \
//               [--temperature 0.7] [--top-k 40] [--top-p 0.9] [--seed 42] \
//               [--pad-id 0] [--eos-id 1]
// =============================================================================

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "slm_engine.h"

namespace {

struct Args {
  std::string enc_path;
  std::string dec_path;
  int         num_threads   = 4;

  std::string prompt_bin;     // empty -> dummy smoke test
  std::string out_bin        = "out_ids.bin";

  int     max_new            = 60;
  std::string method         = "greedy";
  float   temperature        = 0.7f;
  int     top_k              = 40;
  float   top_p              = 0.9f;
  uint64_t seed              = 42;
  int32_t pad_id             = 0;
  int32_t eos_id             = 1;
};

bool ParseArgs(int argc, char** argv, Args& a) {
  if (argc < 3) {
    std::fprintf(stderr,
        "Usage: %s <encode.tflite> <decode.tflite> [num_threads] [flags...]\n"
        "Flags:\n"
        "  --prompt-bin <path>   Tokenized prompt (256 int32) from dump_prompt.py\n"
        "  --out-bin    <path>   Write generated token IDs as int32 (default out_ids.bin)\n"
        "  --max-new    <n>      Decode steps (default 60)\n"
        "  --method     greedy|sample (default greedy)\n"
        "  --temperature <f>     (default 0.7)\n"
        "  --top-k      <n>      (default 40)\n"
        "  --top-p      <f>      (default 0.9)\n"
        "  --seed       <u64>    (default 42)\n"
        "  --pad-id     <n>      (default 0)\n"
        "  --eos-id     <n>      (default 1)\n",
        argv[0]);
    return false;
  }
  a.enc_path = argv[1];
  a.dec_path = argv[2];
  int i = 3;
  if (i < argc && argv[i][0] != '-') {
    a.num_threads = std::atoi(argv[i]);
    ++i;
  }
  for (; i < argc; ++i) {
    std::string k = argv[i];
    auto next = [&]() -> const char* {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "Missing value for %s\n", k.c_str());
        std::exit(1);
      }
      return argv[++i];
    };
    if      (k == "--prompt-bin")  a.prompt_bin = next();
    else if (k == "--out-bin")     a.out_bin    = next();
    else if (k == "--max-new")     a.max_new    = std::atoi(next());
    else if (k == "--method")      a.method     = next();
    else if (k == "--temperature") a.temperature= std::atof(next());
    else if (k == "--top-k")       a.top_k      = std::atoi(next());
    else if (k == "--top-p")       a.top_p      = std::atof(next());
    else if (k == "--seed")        a.seed       = std::strtoull(next(), nullptr, 10);
    else if (k == "--pad-id")      a.pad_id     = std::atoi(next());
    else if (k == "--eos-id")      a.eos_id     = std::atoi(next());
    else {
      std::fprintf(stderr, "Unknown flag: %s\n", k.c_str());
      return false;
    }
  }
  return true;
}

// Read a packed (src_ids, src_mask) binary file written by dump_prompt.py.
bool LoadPromptBin(const std::string& path,
                   std::vector<int32_t>& src_ids,
                   std::vector<int32_t>& src_mask) {
  std::FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) {
    std::fprintf(stderr, "Cannot open %s\n", path.c_str());
    return false;
  }
  src_ids .assign(matmoe::kMaxSrcLen, 0);
  src_mask.assign(matmoe::kMaxSrcLen, 0);
  const size_t n1 = std::fread(src_ids .data(), sizeof(int32_t),
                               matmoe::kMaxSrcLen, f);
  const size_t n2 = std::fread(src_mask.data(), sizeof(int32_t),
                               matmoe::kMaxSrcLen, f);
  std::fclose(f);
  if (n1 != static_cast<size_t>(matmoe::kMaxSrcLen) ||
      n2 != static_cast<size_t>(matmoe::kMaxSrcLen)) {
    std::fprintf(stderr,
        "Prompt file is short: read %zu ids + %zu mask, expected %d+%d\n",
        n1, n2, matmoe::kMaxSrcLen, matmoe::kMaxSrcLen);
    return false;
  }
  return true;
}

}  // namespace


int main(int argc, char** argv) {
  Args a;
  if (!ParseArgs(argc, argv, a)) return 1;

  matmoe::SlmEngine eng;
  std::printf("Loading models (threads=%d):\n  enc=%s\n  dec=%s\n",
              a.num_threads, a.enc_path.c_str(), a.dec_path.c_str());
  auto t0 = std::chrono::steady_clock::now();
  if (!eng.Init(a.enc_path, a.dec_path, a.num_threads)) {
    std::fprintf(stderr, "Init failed: %s\n", eng.LastError().c_str());
    return 2;
  }
  auto t_init = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - t0).count();
  std::printf("Init OK in %.2fs  | vocab=%d\n", t_init, eng.VocabSize());

  // ----- Prepare input -----
  std::vector<int32_t> src_ids, src_mask;
  if (a.prompt_bin.empty()) {
    src_ids .assign(matmoe::kMaxSrcLen, 0);
    src_mask.assign(matmoe::kMaxSrcLen, 0);
    src_ids [0] = 10;
    src_mask[0] = 1;
    std::printf("(dummy prompt: single token id=10)\n");
  } else if (!LoadPromptBin(a.prompt_bin, src_ids, src_mask)) {
    return 3;
  } else {
    int real = 0;
    for (int32_t m : src_mask) real += m;
    std::printf("Loaded %s  (real tokens: %d/%d)\n",
                a.prompt_bin.c_str(), real, matmoe::kMaxSrcLen);
  }

  // ----- Generation config -----
  matmoe::GenerationConfig cfg;
  cfg.max_new_tokens   = a.max_new;
  cfg.pad_id           = a.pad_id;
  cfg.eos_id           = a.eos_id;
  cfg.sampling.method  = (a.method == "sample")
                            ? matmoe::SamplingOptions::kSample
                            : matmoe::SamplingOptions::kGreedy;
  cfg.sampling.temperature = a.temperature;
  cfg.sampling.top_k       = a.top_k;
  cfg.sampling.top_p       = a.top_p;
  cfg.sampling.seed        = a.seed;

  std::vector<int32_t> out(matmoe::kMaxTgtLen, 0);

  // ----- Encoder timing -----
  t0 = std::chrono::steady_clock::now();
  if (!eng.RunEncoderPrefill(src_ids.data(), src_mask.data())) {
    std::fprintf(stderr, "RunEncoderPrefill failed: %s\n",
                 eng.LastError().c_str());
    return 4;
  }
  auto enc_ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - t0).count();
  std::printf("Encoder prefill: %.1f ms\n", enc_ms);

  // ----- Generate (uses Encoder again internally; that's fine for a one-shot
  //        test and keeps the public API simple) -----
  t0 = std::chrono::steady_clock::now();
  const int n = eng.Generate(src_ids.data(), src_mask.data(), cfg,
                             out.data(), static_cast<int>(out.size()));
  auto gen_ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - t0).count();
  std::printf("Generate (%s): %d tokens in %.1f ms (%.1f ms/tok)\n",
              a.method.c_str(), n, gen_ms, n > 0 ? gen_ms / n : 0.0);

  std::printf("Tokens:");
  for (int i = 0; i < n; ++i) std::printf(" %d", out[i]);
  std::printf("\n");

  // ----- Persist tokens for the Python detokenizer -----
  if (!a.out_bin.empty()) {
    std::FILE* f = std::fopen(a.out_bin.c_str(), "wb");
    if (!f) {
      std::fprintf(stderr, "Cannot write %s\n", a.out_bin.c_str());
      return 5;
    }
    std::fwrite(out.data(), sizeof(int32_t), n, f);
    std::fclose(f);
    std::printf("Wrote %d int32 tokens to %s\n", n, a.out_bin.c_str());
  }

  std::printf("OK\n");
  return 0;
}
