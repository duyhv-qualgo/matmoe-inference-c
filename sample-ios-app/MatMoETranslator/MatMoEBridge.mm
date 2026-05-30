#import "MatMoEBridge.h"

#include <memory>
#include <string>
#include <vector>

#include "slm_engine.h"

@implementation MatMoEBridge {
    std::unique_ptr<matmoe::SlmEngine> _eng;
    std::string _lastError;
}

- (instancetype)init {
    if ((self = [super init])) {
        _eng = std::make_unique<matmoe::SlmEngine>();
    }
    return self;
}

- (BOOL)loadEncoderPath:(NSString *)encoderPath
            decoderPath:(NSString *)decoderPath
                threads:(int)threads {
    const bool ok = _eng->Init([encoderPath UTF8String],
                               [decoderPath UTF8String],
                               threads);
    if (!ok) _lastError = _eng->LastError();
    return ok ? YES : NO;
}

- (NSArray<NSNumber *> *)generateSrcIds:(NSArray<NSNumber *> *)srcIds
                                srcMask:(NSArray<NSNumber *> *)srcMask
                           maxNewTokens:(int)maxNewTokens
                                  padId:(int32_t)padId
                                  eosId:(int32_t)eosId
                                 method:(MatMoESamplingMethod)method
                            temperature:(float)temperature
                                   topK:(int)topK
                                   topP:(float)topP
                                   seed:(uint64_t)seed {
    if (srcIds.count != matmoe::kMaxSrcLen ||
        srcMask.count != matmoe::kMaxSrcLen) {
        _lastError = "srcIds / srcMask must each be exactly 128 elements";
        return @[];
    }

    std::vector<int32_t> ids(matmoe::kMaxSrcLen);
    std::vector<int32_t> mask(matmoe::kMaxSrcLen);
    for (NSUInteger i = 0; i < matmoe::kMaxSrcLen; ++i) {
        ids[i]  = [srcIds[i] intValue];
        mask[i] = [srcMask[i] intValue];
    }

    matmoe::GenerationConfig cfg;
    cfg.max_new_tokens   = maxNewTokens;
    cfg.pad_id           = padId;
    cfg.eos_id           = eosId;
    cfg.sampling.method  = (method == MatMoESamplingTopKTopP)
                              ? matmoe::SamplingOptions::kSample
                              : matmoe::SamplingOptions::kGreedy;
    cfg.sampling.temperature = temperature;
    cfg.sampling.top_k       = topK;
    cfg.sampling.top_p       = topP;
    cfg.sampling.seed        = seed;

    std::vector<int32_t> out(matmoe::kMaxTgtLen, 0);
    const int n = _eng->Generate(ids.data(), mask.data(), cfg,
                                 out.data(), static_cast<int>(out.size()));
    if (n < 0) {
        _lastError = _eng->LastError();
        return @[];
    }

    // Strip a trailing EOS (the Python decoder skips special tokens anyway,
    // but stripping here keeps the public API closer to "just the words").
    int realN = n;
    if (realN > 0 && out[realN - 1] == eosId) --realN;

    NSMutableArray<NSNumber *> *result =
        [NSMutableArray arrayWithCapacity:realN];
    for (int i = 0; i < realN; ++i) {
        [result addObject:@(out[i])];
    }
    return result;
}

- (int)vocabSize {
    return _eng->VocabSize();
}

- (NSString *)lastError {
    if (!_lastError.empty()) {
        return [NSString stringWithUTF8String:_lastError.c_str()];
    }
    const std::string &eng_err = _eng->LastError();
    return [NSString stringWithUTF8String:eng_err.c_str()];
}

@end
