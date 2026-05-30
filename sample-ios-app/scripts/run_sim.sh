#!/usr/bin/env bash
# =============================================================================
#  run_sim.sh — build + install + launch the app on an iPhone simulator.
#
#  Picks the simulator named by SIM_NAME (default "iPhone 15"). Boots it if
#  needed. Uses xcodebuild for the build and xcrun simctl for install/launch.
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

SIM_NAME="${SIM_NAME:-iPhone 15}"
SCHEME="MatMoETranslator"
PROJECT="MatMoETranslator.xcodeproj"
BUNDLE_ID="com.example.matmoe.MatMoETranslator"
BUILD_DIR="build/sim"

if [[ ! -d "$PROJECT" ]]; then
  echo "ERROR: $PROJECT not found. Run scripts/bootstrap.sh first."
  exit 1
fi

# ---- 1. Boot the simulator -------------------------------------------------
UDID="$(xcrun simctl list devices available | awk -v n="$SIM_NAME" \
  '$0 ~ n { match($0, /\(([0-9A-Fa-f-]{36})\)/, a); print a[1]; exit }')"
if [[ -z "$UDID" ]]; then
  echo "ERROR: No simulator named '$SIM_NAME'. List with:  xcrun simctl list devices available"
  exit 1
fi
STATE="$(xcrun simctl list devices | awk -v u="$UDID" '$0 ~ u { match($0, /\(([A-Za-z]+)\)/, a); print a[1]; exit }')"
if [[ "$STATE" != "Booted" ]]; then
  echo "=== Booting simulator $SIM_NAME ($UDID)..."
  xcrun simctl boot "$UDID"
fi
open -a Simulator

# ---- 2. Build for the simulator -------------------------------------------
echo "=== Building $SCHEME for iphonesimulator..."
xcodebuild \
  -project "$PROJECT" \
  -scheme "$SCHEME" \
  -configuration Debug \
  -destination "platform=iOS Simulator,id=$UDID" \
  -derivedDataPath "$BUILD_DIR" \
  build | xcbeautify 2>/dev/null || true
# Re-run without xcbeautify so the exit code propagates if it failed.
xcodebuild \
  -project "$PROJECT" \
  -scheme "$SCHEME" \
  -configuration Debug \
  -destination "platform=iOS Simulator,id=$UDID" \
  -derivedDataPath "$BUILD_DIR" \
  build >/dev/null

APP_PATH="$(find "$BUILD_DIR/Build/Products/Debug-iphonesimulator" -name '*.app' -maxdepth 1 | head -1)"
if [[ -z "$APP_PATH" ]]; then
  echo "ERROR: Could not find built .app under $BUILD_DIR"
  exit 1
fi

# ---- 3. Install + launch --------------------------------------------------
echo "=== Installing $APP_PATH on $SIM_NAME"
xcrun simctl install "$UDID" "$APP_PATH"
echo "=== Launching $BUNDLE_ID"
xcrun simctl launch --console-pty "$UDID" "$BUNDLE_ID"
