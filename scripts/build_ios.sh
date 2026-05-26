#!/usr/bin/env bash
# =============================================================================
#  build_ios.sh — produce SlmEngine.xcframework for iOS + iOS simulator.
#
#  Output:
#      dist/SlmEngine.xcframework        ← drop into your Xcode project
#      dist/include/slm_engine.h         ← public header
#
#  Slices produced:
#      iphoneos  / arm64                 (physical devices: iPhone, iPad)
#      iphonesim / arm64 + x86_64        (Apple Silicon + Intel macOS simulators)
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

if [[ ! -d "$TF_SRC" ]]; then
  echo "ERROR: TensorFlow source not found at $TF_SRC"
  echo "       Run a desktop build first: cmake -S . -B build && cmake --build build -j"
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
    -DTENSORFLOW_SOURCE_DIR="$TF_SRC" \
    -DTFLITE_ENABLE_XNNPACK=ON \
    -DTFLITE_ENABLE_GPU=OFF \
    -DTFLITE_ENABLE_RUY=ON
  cmake --build "$bld" -j "$JOBS" --target slm_engine
}

build_slice "iphoneos"       build-ios          iphoneos          "arm64"
build_slice "iphonesimulator" build-ios-sim     iphonesimulator   "arm64;x86_64"

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
merge_slice build-ios     "$DIST/_merged/libSlmEngine-iphoneos.a"
merge_slice build-ios-sim "$DIST/_merged/libSlmEngine-iphonesim.a"

# -----------------------------------------------------------------------------
# Assemble the .xcframework.
# -----------------------------------------------------------------------------
rm -rf "$DIST/SlmEngine.xcframework"
mkdir -p "$DIST/include"
cp include/slm_engine.h "$DIST/include/"

xcodebuild -create-xcframework \
  -library "$DIST/_merged/libSlmEngine-iphoneos.a" \
    -headers "$DIST/include" \
  -library "$DIST/_merged/libSlmEngine-iphonesim.a" \
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
