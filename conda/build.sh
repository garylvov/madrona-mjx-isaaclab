#!/bin/bash
set -euo pipefail

# Madrona bakes absolute paths into its binaries at configure time
# (MADRONA_RENDER_DATA_DIR for runtime HLSL/sky data, the NVRTC device-source
# path arrays, madrona_mjx's DATA_DIR). Building from a source tree that lives
# under $PREFIX makes every baked path contain the placeholder prefix, which
# rattler-build/conda rewrite to the real env prefix at install time.
INSTALL_ROOT="$PREFIX/share/madrona-mjx"

mkdir -p "$INSTALL_ROOT"
cp -a "$SRC_DIR"/. "$INSTALL_ROOT"/
rm -rf "$INSTALL_ROOT/conda" "$INSTALL_ROOT/.git"

BUILD_DIR="$SRC_DIR/build-conda"
cmake -S "$INSTALL_ROOT" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_SYSROOT="$BUILD_PREFIX/x86_64-conda-linux-gnu/sysroot" \
    -DCUDAToolkit_ROOT="$PREFIX"
cmake --build "$BUILD_DIR" -j"$(nproc)"

# Upstream's pyproject build backend only implements build_editable (a
# redirect shim of absolute paths), so install the package by hand.
SP_DIR="$("$PYTHON" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
PKG_DIR="$SP_DIR/madrona_mjx"
mkdir -p "$PKG_DIR/vk"

cp "$INSTALL_ROOT"/src/madrona_mjx/*.py "$PKG_DIR"/
cp "$BUILD_DIR"/_madrona_mjx_batch_renderer*.so "$PKG_DIR"/
cp "$BUILD_DIR"/_madrona_mjx_visualizer*.so "$PKG_DIR"/ 2>/dev/null || true
cp "$BUILD_DIR"/libmadmjx_mgr.so \
   "$BUILD_DIR"/libmadrona_std_mem.so \
   "$BUILD_DIR"/libmadrona_render_shader_compiler.so \
   "$BUILD_DIR"/libdxcompiler.so \
   "$PKG_DIR"/
cp -a "$BUILD_DIR"/libembree*.so* "$PKG_DIR"/
cp -a "$BUILD_DIR"/libglfw.so* "$PKG_DIR"/ 2>/dev/null || true
# the vulkan loader is dlopen'd by soname; keep the upstream vk/ layout
cp "$BUILD_DIR"/vk/libvulkan.so.1 "$PKG_DIR/vk/"

# $ORIGIN: sibling madrona libs; $ORIGIN/vk: vulkan loader;
# $ORIGIN/../../..: $PREFIX/lib for cuda-cudart/cuda-nvrtc
for so in "$PKG_DIR"/*.so* "$PKG_DIR"/vk/libvulkan.so.1; do
    [ -L "$so" ] && continue
    patchelf --set-rpath '$ORIGIN:$ORIGIN/vk:$ORIGIN/../../..' "$so"
done

DIST_INFO="$SP_DIR/madrona_mjx-$PKG_VERSION.dist-info"
mkdir -p "$DIST_INFO"
cat > "$DIST_INFO/METADATA" <<EOF
Metadata-Version: 2.1
Name: madrona-mjx
Version: $PKG_VERSION
EOF
printf 'conda\n' > "$DIST_INFO/INSTALLER"
printf 'madrona_mjx\n' > "$DIST_INFO/top_level.txt"

# Prune the installed source tree to the runtime subset. Needed at runtime:
#   external/madrona/src      runtime-NVRTC device sources + HLSL shaders + sky LUTs
#   external/madrona/include  -I'd into the runtime NVRTC compile
#   src, data                 madrona_mjx GPU sources and DATA_DIR assets
# Everything under external/madrona/external (vendored third-party deps,
# downloaded toolchain/deps bundles) is build-time only.
rm -rf "$INSTALL_ROOT/external/madrona/external"
rm -rf "$INSTALL_ROOT/docs" "$INSTALL_ROOT/scripts"
rm -rf "$INSTALL_ROOT/build"
