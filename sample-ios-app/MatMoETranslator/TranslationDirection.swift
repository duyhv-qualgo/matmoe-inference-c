import Foundation

enum TranslationDirection: String, CaseIterable, Identifiable {
    case enToVi
    case viToEn

    var id: String { rawValue }

    var label: String {
        switch self {
        case .enToVi: return "EN → VI"
        case .viToEn: return "VI → EN"
        }
    }

    var sourceLabel: String { self == .enToVi ? "English" : "Tiếng Việt" }
    var targetLabel: String { self == .enToVi ? "Tiếng Việt" : "English" }

    /// Special prefix the encoder expects; matches the Python reference
    /// (`scripts/dump_prompt.py` and `export/test_translation.py`).
    var promptPrefix: String {
        switch self {
        case .enToVi: return "<translate-en-vi>"
        case .viToEn: return "<translate-vi-en>"
        }
    }
}
