#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

typedef NS_ENUM(NSInteger, MatMoESamplingMethod) {
    MatMoESamplingGreedy = 0,
    MatMoESamplingTopKTopP = 1,
};

/// Thin Obj-C wrapper around matmoe::SlmEngine so Swift can drive translation
/// without touching C++ types. The engine itself is tokenizer-free; pass
/// already-encoded int32 src_ids + src_mask (each exactly 128 long).
@interface MatMoEBridge : NSObject

- (instancetype)init;

/// Load both .tflite graphs and wire XNNPACK. Returns NO on failure; check
/// -lastError for a human-readable reason.
- (BOOL)loadEncoderPath:(NSString *)encoderPath
            decoderPath:(NSString *)decoderPath
                threads:(int)threads NS_SWIFT_NAME(load(encoderPath:decoderPath:threads:));

/// One-shot translate. `srcIds` and `srcMask` must each have exactly 128
/// NSNumbers (int32). Returns the generated token IDs, **without** the
/// trailing EOS if the model produced one.
- (NSArray<NSNumber *> *)generateSrcIds:(NSArray<NSNumber *> *)srcIds
                                srcMask:(NSArray<NSNumber *> *)srcMask
                           maxNewTokens:(int)maxNewTokens
                                  padId:(int32_t)padId
                                  eosId:(int32_t)eosId
                                 method:(MatMoESamplingMethod)method
                            temperature:(float)temperature
                                   topK:(int)topK
                                   topP:(float)topP
                                   seed:(uint64_t)seed
    NS_SWIFT_NAME(generate(srcIds:srcMask:maxNewTokens:padId:eosId:method:temperature:topK:topP:seed:));

@property (nonatomic, readonly) int vocabSize;
@property (nonatomic, readonly, copy) NSString *lastError;

@end

NS_ASSUME_NONNULL_END
