import SwiftUI

struct ContentView: View {
    @StateObject private var vm = TranslationViewModel()
    @State private var input: String = "Hello, how are you today?"

    var body: some View {
        NavigationStack {
            VStack(alignment: .leading, spacing: 12) {
                directionPicker
                inputCard
                translateButton
                outputCard
                Spacer()
                statusFooter
            }
            .padding()
            .navigationTitle("MatMoE Translator")
            .navigationBarTitleDisplayMode(.inline)
        }
        .task { await vm.warmUpIfNeeded() }
    }

    private var directionPicker: some View {
        Picker("Direction", selection: $vm.direction) {
            ForEach(TranslationDirection.allCases) { dir in
                Text(dir.label).tag(dir)
            }
        }
        .pickerStyle(.segmented)
    }

    private var inputCard: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(vm.direction.sourceLabel)
                .font(.caption).foregroundStyle(.secondary)
            TextEditor(text: $input)
                .frame(minHeight: 100)
                .padding(8)
                .background(Color(.secondarySystemBackground))
                .clipShape(RoundedRectangle(cornerRadius: 10))
        }
    }

    private var translateButton: some View {
        Button {
            Task { await vm.translate(input) }
        } label: {
            HStack {
                if vm.isRunning { ProgressView().controlSize(.small) }
                Text(vm.isRunning ? "Translating…" : "Translate")
                    .frame(maxWidth: .infinity)
            }
            .padding(.vertical, 10)
        }
        .buttonStyle(.borderedProminent)
        .disabled(vm.isRunning || input.trimmingCharacters(in: .whitespaces).isEmpty)
    }

    private var outputCard: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(vm.direction.targetLabel)
                .font(.caption).foregroundStyle(.secondary)
            ScrollView {
                Text(vm.output.isEmpty ? " " : vm.output)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(8)
                    .textSelection(.enabled)
            }
            .frame(minHeight: 120)
            .background(Color(.secondarySystemBackground))
            .clipShape(RoundedRectangle(cornerRadius: 10))
        }
    }

    private var statusFooter: some View {
        Group {
            if let err = vm.errorMessage {
                Text(err).font(.caption).foregroundStyle(.red)
            } else if let s = vm.timing {
                Text(s).font(.caption).foregroundStyle(.secondary)
            } else if !vm.isReady {
                Text("Loading model…").font(.caption).foregroundStyle(.secondary)
            }
        }
    }
}

#Preview {
    ContentView()
}
