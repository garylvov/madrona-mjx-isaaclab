include("${CMAKE_CURRENT_LIST_DIR}/llvm-common.cmake")

set(LLVM_INCLUDE_TESTS OFF CACHE BOOL "")
set(LLVM_ENABLE_LTO OFF CACHE STRING "")

if (APPLE)
    # For mac builds, stage1 needs to be a cross compiler so stage2 can be
    # built for both arches
    set(LLVM_TARGETS_TO_BUILD "AArch64;X86" CACHE STRING "")
else()
    set(LLVM_TARGETS_TO_BUILD Native CACHE STRING "")
endif()

set(CLANG_BOOTSTRAP_PASSTHROUGH
    CMAKE_INSTALL_PREFIX
    CMAKE_BUILD_TYPE
    CMAKE_PREFIX_PATH
    CMAKE_OSX_ARCHITECTURES
    LLVM_DEFAULT_TARGET_TRIPLE
    ZLIB_USE_STATIC_LIBS

    CACHE STRING ""
)

set(CLANG_ENABLE_BOOTSTRAP ON CACHE BOOL "")
set(CLANG_BOOTSTRAP_CMAKE_ARGS
    -C ${CMAKE_CURRENT_LIST_DIR}/llvm-stage2.cmake
    CACHE STRING ""
)

set(BOOTSTRAP_LLVM_ENABLE_LTO THIN CACHE STRING "")
set(BOOTSTRAP_LLVM_ENABLE_LLD ON CACHE BOOL "")
set(BOOTSTRAP_LLVM_ENABLE_LIBCXX ON CACHE BOOL "")

set(XCODE_TOOLCHAIN_TARGETS)
if (APPLE)
    list(APPEND XCODE_TOOLCHAIN_TARGETS
        install-xcode-toolchain
    )
endif()

set(CLANG_BOOTSTRAP_TARGETS
    check-all
    check-clang
    check-lld
    check-llvm
    llvm-config
    clang-test-depends
    lld-test-depends
    llvm-test-depends
    test-suite
    test-depends
    distribution
    install-distribution
    install-distribution-stripped
    install-distribution-toolchain
    clang
    ${XCODE_TOOLCHAIN_TARGETS}

    CACHE STRING ""
)

set(CLANG_BOOTSTRAP_EXTRA_DEPS
    builtins
    runtimes
    # Critical for macOS otherwise system lipo is used, breaking stage2 LTO
    lipo
    libtool

    CACHE STRING ""
)
