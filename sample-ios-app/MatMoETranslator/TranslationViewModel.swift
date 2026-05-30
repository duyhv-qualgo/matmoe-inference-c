import Foundation
import SwiftUI

@MainActor
final class TranslationViewModel: ObservableObject {

    @Published var direction: TranslationDirection = .enToVi
    @Published var output: String = ""
    @Published var isRunning: Bool = false
    @Published var isReady: Bool = false
    @Published var errorMessage: String?
    @Published var timing: String?

    private var engine: MatMoEBridge?
    private var tokenizer: MatMoETokenizer?

    private let maxNewTokens: Int32 = 96
    private let padId: Int32 = 0
    private let eosId: Int32 = 1
    private let threads: Int32 = 4

    func warmUpIfNeeded() async {
        if isReady { return }
        do {
            let (enc, dec) = try bundleModelPaths()
            let threadsLocal = threads
            let t0 = Date()
            let bridge: MatMoEBridge = try await Task.detached(priority: .userInitiated) {
                let b = MatMoEBridge()
                let ok = b.load(encoderPath: enc, decoderPath: dec, threads: threadsLocal)
                if !ok {
                    throw NSError(domain: "MatMoE", code: 1,
                                  userInfo: [NSLocalizedDescriptionKey: b.lastError])
                }
                return b
            }.value
            let tok = try await MatMoETokenizer.loadFromBundle()
            self.engine = bridge
            self.tokenizer = tok
            self.isReady = true
            let ms = Int(Date().timeIntervalSince(t0) * 1000)
            self.timing = "Loaded in \(ms) ms · vocab \(bridge.vocabSize)"
        } catch {
            self.errorMessage = error.localizedDescription
        }
    }

    func translate(_ text: String) async {
        guard let engine, let tokenizer else {
            errorMessage = "Engine not ready"
            return
        }
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }

        isRunning = true
        errorMessage = nil
        output = ""
        defer { isRunning = false }

        let prompt = "\(direction.promptPrefix) \(trimmed)"
        let (ids, mask) = tokenizer.encodePadded(prompt, padId: padId)
        let nsIds = ids.map(NSNumber.init(value:))
        let nsMask = mask.map(NSNumber.init(value:))

        // Capture into locals so the detached task doesn't need to read
        // @MainActor state.
        let maxNew = maxNewTokens
        let pad = padId
        let eos = eosId
        let t0 = Date()
        let outIdsBoxed: [NSNumber] = await Task.detached(priority: .userInitiated) {
            engine.generate(srcIds: nsIds, srcMask: nsMask,
                            maxNewTokens: maxNew,
                            padId: pad, eosId: eos,
                            method: .greedy,
                            temperature: 0.7, topK: 40, topP: 0.9, seed: 42)
        }.value
        let elapsed = Date().timeIntervalSince(t0) * 1000

        let outIds = outIdsBoxed.map { $0.int32Value }
        let decoded = tokenizer.decode(outIds)
        self.output = decoded
        self.timing = String(format: "%d tokens · %.0f ms · %.1f ms/tok",
                             outIds.count, elapsed,
                             outIds.isEmpty ? 0 : elapsed / Double(outIds.count))
    }

    private func bundleModelPaths() throws -> (String, String) {
        guard let enc = ResourceLookup.path("encode_prefill", ext: "tflite") else {
            throw NSError(domain: "MatMoE", code: 2,
                          userInfo: [NSLocalizedDescriptionKey:
                            "encode_prefill.tflite not in bundle — drop into Resources/."])
        }
        guard let dec = ResourceLookup.path("decode_step", ext: "tflite") else {
            throw NSError(domain: "MatMoE", code: 3,
                          userInfo: [NSLocalizedDescriptionKey:
                            "decode_step.tflite not in bundle — drop into Resources/."])
        }
        return (enc, dec)
    }
}

/// XcodeGen mounts `Resources/` as a folder reference, so the files end up
/// nested under "Resources/" inside the .app. If someone later flattens the
/// build to group references the files end up at the bundle root. Try both.
enum ResourceLookup {
    static func path(_ name: String, ext: String) -> String? {
        if let p = Bundle.main.path(forResource: name, ofType: ext) { return p }
        return Bundle.main.path(forResource: name, ofType: ext, inDirectory: "Resources")
    }
    static func url(_ name: String, ext: String) -> URL? {
        if let u = Bundle.main.url(forResource: name, withExtension: ext) { return u }
        return Bundle.main.url(forResource: name, withExtension: ext, subdirectory: "Resources")
    }
}
