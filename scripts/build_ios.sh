#!/usr/bin/env bash
# =============================================================================
#  build_ios.sh — produce SlmEngine.xcframework for iOS + iOS simulator.
#
#  Output:
#      dist/SlmEngine.xcframework        ← drop into your Xcode project
#      dist/include/slm_engine.h         ← public header
#
#  Slices produced:
#      iphoneos        / arm64           (physical devices: iPhone, iPad)
#      iphonesimulator / arm64           (Apple Silicon Mac simulators)
#      iphonesimulator / x86_64          (only when SIM_X86_64=1; Intel Mac simulators)
#
#  XNNPACK refuses multi-arch CMake configures, so the simulator slice is
#  built once per arch and (when x86_64 is enabled) the merged .a files are
#  fused with `lipo`. Set SIM_X86_64=1 if you also need Intel Mac sim
#  support; the default arm64-only path is faster and covers Apple Silicon
#  developers.
#
#  Re-builds are fast: TFLite is built once per slice (~10-25 min the first
#  time), incrementally afterwards (~10-30 s).
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")/.."

ROOT="$(pwd)"
DIST="$ROOT/dist"
TF_SRC="$ROOT/build/tflite_build/tensorflow-src"   # reuse macOS-fetched TF
IOS_DEPLOY_TARGET="${IOS_DEPLOY_TARGET:-14.0}"
JOBS="${JOBS:-$(sysctl -n hw.logicalcpu)}"

# Cross-compiling TFLite needs `flatc` (the FlatBuffers compiler) built for the
# host. Make sure a host build has run far enough to produce it; if not, do
# just enough to materialise it.
if [[ ! -d "$ROOT/build" || ! -d "$TF_SRC" ]]; then
  echo "=== Bootstrapping host build (fetches TF source, builds flatc only)…"
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
fi
HOST_FLATC="$(find "$ROOT/build" -name flatc -type f -perm -u+x 2>/dev/null | head -1)"
if [[ -z "$HOST_FLATC" ]]; then
  echo "=== Building host flatc…"
  cmake --build build -j "$JOBS" --target flatc
  HOST_FLATC="$(find "$ROOT/build" -name flatc -type f -perm -u+x | head -1)"
fi
if [[ -z "$HOST_FLATC" ]]; then
  echo "ERROR: Could not locate host flatc after build. Tried: $ROOT/build"
  exit 1
fi
HOST_TOOLS_DIR="$(dirname "$HOST_FLATC")"
echo "=== Using host flatc at: $HOST_FLATC"

if [[ ! -d "$TF_SRC" ]]; then
  echo "ERROR: TensorFlow source not found at $TF_SRC after host bootstrap."
  exit 1
fi

# -----------------------------------------------------------------------------
# Build one iOS slice.
#
# Args: <slice-name> <build-dir> <SYSROOT> <ARCHS>
# -----------------------------------------------------------------------------
build_slice () {
  local slice="$1" bld="$2" sysroot="$3" archs="$4"
  echo
  echo "=== $slice  ($archs)  ==============================================="
  cmake -S . -B "$bld" -G "Unix Makefiles" \
    -DCMAKE_SYSTEM_NAME=iOS \
    -DCMAKE_OSX_SYSROOT="$sysroot" \
    -DCMAKE_OSX_ARCHITECTURES="$archs" \
    -DCMAKE_OSX_DEPLOYMENT_TARGET="$IOS_DEPLOY_TARGET" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DBUILD_SHARED_LIBS=OFF \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    -DTENSORFLOW_SOURCE_DIR="$TF_SRC" \
    -DTFLITE_HOST_TOOLS_DIR="$HOST_TOOLS_DIR" \
    -DTFLITE_ENABLE_XNNPACK=ON \
    -DTFLITE_ENABLE_GPU=OFF \
    -DTFLITE_ENABLE_RUY=ON \
    -DFLATBUFFERS_BUILD_FLATC=OFF \
    -DFLATBUFFERS_INSTALL=OFF \
    -DFLATBUFFERS_BUILD_TESTS=OFF
  cmake --build "$bld" -j "$JOBS" --target slm_engine
}

build_slice "iphoneos"            build-ios           iphoneos        "arm64"
build_slice "iphonesimulator-arm64" build-ios-sim-arm64 iphonesimulator "arm64"
if [[ "${SIM_X86_64:-0}" == "1" ]]; then
  build_slice "iphonesimulator-x86_64" build-ios-sim-x86_64 iphonesimulator "x86_64"
fi

# -----------------------------------------------------------------------------
# Each slice has ~80 small static libs (TFLite + XNNPACK + absl + ...).
# Xcode needs a single library per slice in the xcframework, so we merge them
# with libtool -static.
# -----------------------------------------------------------------------------
merge_slice () {
  local bld="$1" out="$2"
  echo
  echo "=== Merging static libs from $bld -> $out ============================"
  local libs
  libs="$(find "$bld" -name '*.a' \
            -not -path '*/CMakeFiles/*' \
            -not -path '*/test*' | sort -u)"
  libtool -static -o "$out" $libs 2>/dev/null
  echo "  -> $out  ($(du -h "$out" | awk '{print $1}'))"
}

mkdir -p "$DIST/_merged"
merge_slice build-ios            "$DIST/_merged/libSlmEngine-iphoneos.a"
merge_slice build-ios-sim-arm64  "$DIST/_merged/libSlmEngine-iphonesim-arm64.a"
SIM_LIB="$DIST/_merged/libSlmEngine-iphonesim-arm64.a"
if [[ "${SIM_X86_64:-0}" == "1" ]]; then
  merge_slice build-ios-sim-x86_64 "$DIST/_merged/libSlmEngine-iphonesim-x86_64.a"
  echo "=== lipo: fusing sim arm64 + x86_64 ============================"
  lipo -create \
    "$DIST/_merged/libSlmEngine-iphonesim-arm64.a" \
    "$DIST/_merged/libSlmEngine-iphonesim-x86_64.a" \
    -output "$DIST/_merged/libSlmEngine-iphonesim.a"
  SIM_LIB="$DIST/_merged/libSlmEngine-iphonesim.a"
  lipo -info "$SIM_LIB"
fi

# -----------------------------------------------------------------------------
# Assemble the .xcframework.
# -----------------------------------------------------------------------------
rm -rf "$DIST/SlmEngine.xcframework"
mkdir -p "$DIST/include"
cp include/slm_engine.h "$DIST/include/"

xcodebuild -create-xcframework \
  -library "$DIST/_merged/libSlmEngine-iphoneos.a" \
    -headers "$DIST/include" \
  -library "$SIM_LIB" \
    -headers "$DIST/include" \
  -output  "$DIST/SlmEngine.xcframework"

echo
echo "================================================================="
echo "Done."
echo "  XCFramework : $DIST/SlmEngine.xcframework"
echo "  Header      : $DIST/include/slm_engine.h"
echo
echo "Drop SlmEngine.xcframework into your Xcode project's 'Frameworks,"
echo "Libraries, and Embedded Content' section. Bridge from Swift via an"
echo "Objective-C++ shim (.mm)."
echo "================================================================="
