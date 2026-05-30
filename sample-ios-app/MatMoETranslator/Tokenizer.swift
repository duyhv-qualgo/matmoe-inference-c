import Foundation
import Tokenizers
import Hub

/// Wraps swift-transformers' `PreTrainedTokenizer` so the view model can
/// stay in plain Swift land. Loads `tokenizer.json` directly out of the
/// app bundle; synthesises a minimal `tokenizer_config.json` next to it in
/// the caches dir if one isn't provided alongside the model.
final class MatMoETokenizer {

    private let tokenizer: any Tokenizer

    /// Look up the tokenizer.json bundled into the app and load it.
    static func loadFromBundle() async throws -> MatMoETokenizer {
        guard let url = ResourceLookup.url("tokenizer", ext: "json") else {
            throw TokenizerLoadError.tokenizerJSONMissing
        }
        return try await load(tokenizerJSON: url)
    }

    static func load(tokenizerJSON: URL) async throws -> MatMoETokenizer {
        // swift-transformers' AutoTokenizer wants a folder containing
        // tokenizer.json + tokenizer_config.json. The bundle is read-only, so
        // mirror both into a writable cache dir and point the loader there.
        let cache = try cacheDir()
        let dstJSON = cache.appendingPathComponent("tokenizer.json")
        if FileManager.default.fileExists(atPath: dstJSON.path) {
            try FileManager.default.removeItem(at: dstJSON)
        }
        try FileManager.default.copyItem(at: tokenizerJSON, to: dstJSON)

        let dstCfg = cache.appendingPathComponent("tokenizer_config.json")
        if !FileManager.default.fileExists(atPath: dstCfg.path) {
            let minimal = #"{"tokenizer_class":"PreTrainedTokenizerFast"}"#
            try minimal.write(to: dstCfg, atomically: true, encoding: .utf8)
        }

        let tok = try await AutoTokenizer.from(modelFolder: cache)
        return MatMoETokenizer(tokenizer: tok)
    }

    private init(tokenizer: any Tokenizer) { self.tokenizer = tokenizer }

    /// Encode `text` into the (src_ids, src_mask) pair the C++ engine expects:
    /// both arrays exactly `maxLen` long, padded with 0s.
    func encodePadded(_ text: String, maxLen: Int = 128, padId: Int32 = 0) -> (ids: [Int32], mask: [Int32]) {
        var ids = tokenizer.encode(text: text).map { Int32($0) }
        if ids.count > maxLen { ids = Array(ids.prefix(maxLen)) }
        let real = ids.count
        if real < maxLen {
            ids.append(contentsOf: Array(repeating: padId, count: maxLen - real))
        }
        var mask = Array(repeating: Int32(0), count: maxLen)
        for i in 0..<min(real, maxLen) { mask[i] = 1 }
        return (ids, mask)
    }

    func decode(_ ids: [Int32]) -> String {
        tokenizer.decode(tokens: ids.map { Int($0) }, skipSpecialTokens: true)
    }

    private static func cacheDir() throws -> URL {
        let base = try FileManager.default.url(
            for: .cachesDirectory, in: .userDomainMask,
            appropriateFor: nil, create: true)
        let dir = base.appendingPathComponent("matmoe-tokenizer", isDirectory: true)
        try FileManager.default.createDirectory(
            at: dir, withIntermediateDirectories: true)
        return dir
    }
}

enum TokenizerLoadError: LocalizedError {
    case tokenizerJSONMissing
    var errorDescription: String? {
        switch self {
        case .tokenizerJSONMissing:
            return "tokenizer.json not found in the app bundle. Drop it into the Resources/ folder before building."
        }
    }
}
