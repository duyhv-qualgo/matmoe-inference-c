#!/usr/bin/env bash
# =============================================================================
#  build_android.sh — produce libslm_engine.so for one or more Android ABIs.
#
#  Output layout (matches the standard Android Studio jniLibs/ structure):
#      dist/android/jniLibs/arm64-v8a/libslm_engine.so
#      dist/android/jniLibs/x86_64/libslm_engine.so   (if --emu added)
#      dist/include/slm_engine.h
#
#  Default ABIs : arm64-v8a (covers ~all modern phones).
#  Add x86_64   : pass --with-emu  (for Android Studio emulator).
#  Add armv7    : pass --with-armv7 (only if you support very old devices).
#
#  Requires Android NDK r25+. Tells you what to install if missing.
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")/.."

ROOT="$(pwd)"
DIST="$ROOT/dist/android"
TF_SRC="$ROOT/build/tflite_build/tensorflow-src"
ANDROID_PLATFORM="${ANDROID_PLATFORM:-android-26}"   # API 26 = Android 8.0
JOBS="${JOBS:-$(sysctl -n hw.logicalcpu 2>/dev/null || nproc)}"

ABIS=("arm64-v8a")
for arg in "$@"; do
  case "$arg" in
    --with-emu)   ABIS+=("x86_64") ;;
    --with-armv7) ABIS+=("armeabi-v7a") ;;
    --abi=*)      ABIS=("${arg#--abi=}") ;;
    *) echo "Unknown flag: $arg"; exit 1 ;;
  esac
done

# -----------------------------------------------------------------------------
# Locate Android NDK. Tries (in order): $ANDROID_NDK_HOME, $ANDROID_NDK,
# ~/Library/Android/sdk/ndk/<latest>, /opt/homebrew/share/android-ndk.
# -----------------------------------------------------------------------------
find_ndk () {
  if [[ -n "${ANDROID_NDK_HOME:-}" && -d "$ANDROID_NDK_HOME" ]]; then
    echo "$ANDROID_NDK_HOME"; return; fi
  if [[ -n "${ANDROID_NDK:-}" && -d "$ANDROID_NDK" ]]; then
    echo "$ANDROID_NDK"; return; fi
  local p
  for p in "$HOME/Library/Android/sdk/ndk"/* \
           /opt/homebrew/share/android-ndk \
           /usr/local/share/android-ndk; do
    [[ -d "$p/build/cmake" ]] && echo "$p" && return
  done
  return 1
}

NDK="$(find_ndk || true)"
if [[ -z "${NDK:-}" ]]; then
  cat <<EOF
ERROR: Android NDK not found.

Install one of:
  * brew install --cask android-ndk
  * Android Studio → SDK Manager → SDK Tools → NDK (Side by side)
  * https://developer.android.com/ndk/downloads

Then set ANDROID_NDK_HOME (or re-run with it pointed at the install dir).
EOF
  exit 1
fi
echo "Using NDK: $NDK"

if [[ ! -d "$TF_SRC" ]]; then
  echo "ERROR: TensorFlow source not found at $TF_SRC"
  echo "       Run a desktop build first: cmake -S . -B build && cmake --build build -j"
  exit 1
fi

mkdir -p "$DIST/include" "$DIST/jniLibs"
cp include/slm_engine.h "$DIST/include/"

# -----------------------------------------------------------------------------
# Build one ABI.
# -----------------------------------------------------------------------------
build_abi () {
  local abi="$1"
  local bld="build-android-$abi"
  echo
  echo "=== Android $abi  (platform=$ANDROID_PLATFORM) ====================="
  cmake -S . -B "$bld" -G "Unix Makefiles" \
    -DCMAKE_TOOLCHAIN_FILE="$NDK/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI="$abi" \
    -DANDROID_PLATFORM="$ANDROID_PLATFORM" \
    -DANDROID_STL=c++_static \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DBUILD_SHARED_LIBS=OFF \
    -DTENSORFLOW_SOURCE_DIR="$TF_SRC" \
    -DTFLITE_ENABLE_XNNPACK=ON \
    -DTFLITE_ENABLE_GPU=OFF \
    -DTFLITE_ENABLE_RUY=ON
  cmake --build "$bld" -j "$JOBS" --target slm_engine

  # Produce a single shared library that bundles slm_engine + TFLite + XNNPACK
  # so the APK only carries one .so per ABI.
  mkdir -p "$DIST/jniLibs/$abi"
  local out="$DIST/jniLibs/$abi/libslm_engine.so"
  local sysroot_libs="$NDK/toolchains/llvm/prebuilt/darwin-x86_64/sysroot/usr/lib"

  local archs_dir
  case "$abi" in
    arm64-v8a)   archs_dir="aarch64-linux-android" ;;
    armeabi-v7a) archs_dir="arm-linux-androideabi" ;;
    x86_64)      archs_dir="x86_64-linux-android" ;;
    *) echo "Unknown ABI $abi"; exit 1 ;;
  esac

  # Collect every .a CMake produced for this ABI.
  local libs
  libs="$(find "$bld" -name '*.a' \
            -not -path '*/CMakeFiles/*' \
            -not -path '*/test*' | sort -u)"

  "$NDK/toolchains/llvm/prebuilt/darwin-x86_64/bin/clang++" \
    -shared -fPIC -O3 \
    --target=${archs_dir} \
    --sysroot="$NDK/toolchains/llvm/prebuilt/darwin-x86_64/sysroot" \
    -Wl,--whole-archive $libs -Wl,--no-whole-archive \
    -static-libstdc++ \
    -llog -landroid \
    -o "$out"
  echo "  -> $out  ($(du -h "$out" | awk '{print $1}'))"
}

for abi in "${ABIS[@]}"; do
  build_abi "$abi"
done

echo
echo "================================================================="
echo "Done."
echo "  jniLibs : $DIST/jniLibs/"
ls -la "$DIST/jniLibs"/*/libslm_engine.so 2>/dev/null || true
echo "  Header  : $DIST/include/slm_engine.h"
echo
echo "Wire into Android Studio:"
echo "  app/src/main/jniLibs/<abi>/libslm_engine.so"
echo "  app/src/main/cpp/slm_jni.cpp     (your JNI shim)"
echo "  android.defaultConfig.ndk.abiFilters 'arm64-v8a' /* etc */"
echo "================================================================="
