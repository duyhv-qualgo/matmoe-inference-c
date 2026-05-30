#!/usr/bin/env bash
# =============================================================================
#  bootstrap.sh — one-shot setup for the sample iOS app.
#
#  Lives at matmoe-inference-c/sample-ios-app/scripts/bootstrap.sh, i.e. one
#  level inside the engine repo.
#
#  1. Builds matmoe-inference-c into SlmEngine.xcframework (if not already).
#  2. Copies it into sample-ios-app/Frameworks/.
#  3. Runs `xcodegen` to produce MatMoETranslator.xcodeproj.
#
#  Re-run any time you change project.yml or rebuild the engine.
#
#  Prereqs (macOS):
#    - Xcode 15+
#    - brew install xcodegen
# =============================================================================
set -euo pipefail

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "ERROR: This script must run on macOS (Xcode + iOS SDK are macOS-only)."
  exit 1
fi

cd "$(dirname "$0")/.."
APP_ROOT="$(pwd)"
# We sit at matmoe-inference-c/sample-ios-app, so the engine root is one up.
MATMOE="$(cd .. && pwd)"
DIST_XCFW="$MATMOE/dist/SlmEngine.xcframework"
APP_XCFW="$APP_ROOT/Frameworks/SlmEngine.xcframework"

if [[ ! -f "$MATMOE/CMakeLists.txt" ]]; then
  echo "ERROR: $MATMOE doesn't look like matmoe-inference-c (no CMakeLists.txt)."
  echo "       This script expects to live at matmoe-inference-c/sample-ios-app/scripts/."
  exit 1
fi

# ---- 1. Build (or rebuild) the xcframework -------------------------------
# Always invoke build_ios.sh; CMake handles up-to-date checks itself. The
# previous "skip if dist/ exists" shortcut silently shipped stale headers
# when slm_engine sources changed.
echo "=== Building (or refreshing) $DIST_XCFW..."
pushd "$MATMOE" >/dev/null
./scripts/build_ios.sh
popd >/dev/null

# ---- 2. Sync it into the app ----------------------------------------------
mkdir -p "$APP_ROOT/Frameworks"
rm -rf "$APP_XCFW"
cp -R "$DIST_XCFW" "$APP_XCFW"
echo "=== Copied xcframework -> $APP_XCFW"

# ---- 3. Resources sanity check --------------------------------------------
missing=()
for f in encode_prefill.tflite decode_step.tflite tokenizer.json; do
  if [[ ! -f "$APP_ROOT/Resources/$f" ]]; then
    missing+=("Resources/$f")
  fi
done
if (( ${#missing[@]} > 0 )); then
  echo
  echo "WARNING: missing resources (the app will crash on launch until these exist):"
  for m in "${missing[@]}"; do echo "  - $m"; done
  echo "See Resources/README.md for what to drop in."
fi

# ---- 4. Generate Xcode project --------------------------------------------
if ! command -v xcodegen >/dev/null; then
  echo "ERROR: xcodegen not found. Install with: brew install xcodegen"
  exit 1
fi
echo "=== Running xcodegen..."
xcodegen generate

echo
echo "================================================================="
echo "Done."
echo "  Open:  open MatMoETranslator.xcodeproj"
echo "  Run in simulator (one-liner):  scripts/run_sim.sh"
echo "================================================================="
