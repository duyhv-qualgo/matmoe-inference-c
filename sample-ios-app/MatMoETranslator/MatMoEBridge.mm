#import "MatMoEBridge.h"

#include <vector>

#include "slm_engine_c.h"

@implementation MatMoEBridge {
    MatMoeEngine _eng;
    NSString *_overrideError;
}

- (instancetype)init {
    if ((self = [super init])) {
        _eng = matmoe_engine_create();
    }
    return self;
}

- (void)dealloc {
    if (_eng) {
        matmoe_engine_destroy(_eng);
        _eng = NULL;
    }
}

- (BOOL)loadEncoderPath:(NSString *)encoderPath
            decoderPath:(NSString *)decoderPath
                threads:(int)threads {
    const int ok = matmoe_engine_load(_eng,
                                      [encoderPath UTF8String],
                                      [decoderPath UTF8String],
                                      threads);
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
    if (srcIds.count != MATMOE_MAX_SRC_LEN ||
        srcMask.count != MATMOE_MAX_SRC_LEN) {
        _overrideError = @"srcIds / srcMask must each be exactly 128 elements";
        return @[];
    }
    _overrideError = nil;

    std::vector<int32_t> ids(MATMOE_MAX_SRC_LEN);
    std::vector<int32_t> mask(MATMOE_MAX_SRC_LEN);
    for (NSUInteger i = 0; i < MATMOE_MAX_SRC_LEN; ++i) {
        ids[i]  = [srcIds[i] intValue];
        mask[i] = [srcMask[i] intValue];
    }

    const MatMoeSamplingMethod cMethod =
        (method == MatMoESamplingTopKTopP)
            ? MATMOE_SAMPLING_TOP_K_TOP_P
            : MATMOE_SAMPLING_GREEDY;

    std::vector<int32_t> out(MATMOE_MAX_TGT_LEN, 0);
    const int n = matmoe_engine_generate(_eng,
                                          ids.data(), mask.data(),
                                          maxNewTokens, padId, eosId,
                                          cMethod, temperature,
                                          topK, topP, seed,
                                          out.data(),
                                          static_cast<int>(out.size()));
    if (n < 0) return @[];

    // Strip a trailing EOS so callers get just the words.
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
    return matmoe_engine_vocab_size(_eng);
}

- (NSString *)lastError {
    if (_overrideError) return _overrideError;
    const char *msg = matmoe_engine_last_error(_eng);
    return msg ? [NSString stringWithUTF8String:msg] : @"";
}

@end
