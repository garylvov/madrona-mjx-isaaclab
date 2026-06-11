cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
cmake_policy(VERSION 3.18)
get_filename_component(PARENT_DIR ${CMAKE_SOURCE_DIR} DIRECTORY)
set(NAME madrona-toolchain-compile)

if (APPLE)
    set(MADRONA_TOOLCHAIN_MACOS_ARCH "" CACHE STRING "macOS compiler host architecture")
    if (NOT MADRONA_TOOLCHAIN_MACOS_ARCH)
        message(FATAL_ERROR "Must specify macOS host architecture")
    endif()

    set(CMAKE_OSX_ARCHITECTURES "${MADRONA_TOOLCHAIN_MACOS_ARCH}" CACHE STRING "")
    list(JOIN CMAKE_OSX_ARCHITECTURES "$<SEMICOLON>" EXT_OSX_ARCHITECTURES)
    set(CMAKE_OSX_DEPLOYMENT_TARGET "12.0" CACHE STRING "")
endif()

project(${NAME})

include(${PARENT_DIR}/cmake/sys-detect.cmake)

include(FetchContent)
include(ExternalProject)

find_package(Python 3.6 COMPONENTS Interpreter)
find_package(Patch REQUIRED)

include(versions.cmake)

set(NCURSES_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/ncurses")
set(NCURSES_SRC "${NCURSES_PREFIX}/sources")
set(NCURSES_INSTALL "${NCURSES_PREFIX}/install")

set(NCURSES_CFLAGS "-O3 -fPIC")
set(NCURSES_CXXFLAGS "-O3 -fPIC")
set(NCURSES_CPPFLAGS)
if (APPLE)
    set(NCURSES_CFLAGS "${NCURSES_CFLAGS} -arch ${MADRONA_TOOLCHAIN_MACOS_ARCH} -isysroot${CMAKE_OSX_SYSROOT} -mmacosx-version-min=${CMAKE_OSX_DEPLOYMENT_TARGET}")
    set(NCURSES_CXXFLAGS "${NCURSES_CXXFLAGS} -arch ${MADRONA_TOOLCHAIN_MACOS_ARCH} -isysroot${CMAKE_OSX_SYSROOT} -mmacosx-version-min=${CMAKE_OSX_DEPLOYMENT_TARGET}")
    set(NCURSES_CPPFLAGS "${NCURSES_CPPFLAGS} -isysroot${CMAKE_OSX_SYSROOT}")
endif()

ExternalProject_Add(ncurses
    URL "https://ftp.gnu.org/gnu/ncurses/ncurses-${NCURSES_VERSION}.tar.gz"
    URL_HASH SHA256=${NCURSES_HASH}
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    PREFIX "${NCURSES_PREFIX}"
    SOURCE_DIR "${NCURSES_SRC}"
    INSTALL_DIR "${NCURSES_INSTALL}"
    CONFIGURE_COMMAND
        ${CMAKE_COMMAND} -E env CC=${CMAKE_C_COMPILER} CXX=${CMAKE_CXX_COMPILER} CFLAGS=${NCURSES_CFLAGS} CPPFLAGS=${NCURSES_CPPFLAGS} CXXFLAGS=${NCURSES_CXXFLAGS} "${NCURSES_SRC}/configure" --prefix=/usr --with-termlib --enable-widec --without-debug --without-cxx-binding --without-ada
    BUILD_COMMAND
        make
    INSTALL_COMMAND
        make DESTDIR=${NCURSES_INSTALL} install
)

ExternalProject_Add_Step(ncurses copy_tinfo
    COMMAND 
        ${CMAKE_COMMAND} -E copy ${NCURSES_INSTALL}/usr/lib/libtinfow.a ${NCURSES_INSTALL}/usr/lib/libtinfo.a
    DEPENDEES
        install
    BYPRODUCTS
        ${NCURSES_INSTALL}/lib/libtinfo.a
)

set(ZSTD_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/zstd")
set(ZSTD_INSTALL "${ZSTD_PREFIX}/install")

ExternalProject_Add(zstd
    GIT_REPOSITORY https://github.com/facebook/zstd.git
    GIT_TAG "${ZSTD_VERSION}"
    GIT_SHALLOW ON
    PREFIX "${ZSTD_PREFIX}"
    INSTALL_DIR "${ZSTD_INSTALL}"
    CMAKE_ARGS
        "-DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}"
        "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
        "-DCMAKE_INSTALL_PREFIX=${ZSTD_INSTALL}"
        "-DCMAKE_BUILD_TYPE=Release"
        "-DCMAKE_POSITION_INDEPENDENT_CODE=ON"
        "-DCMAKE_OSX_ARCHITECTURES=${EXT_OSX_ARCHITECTURES}"
        "-DCMAKE_OSX_DEPLOYMENT_TARGET=${CMAKE_OSX_DEPLOYMENT_TARGET}"
        "-DZSTD_BUILD_SHARED=OFF"
        "-DZSTD_BUILD_STATIC=ON"
        "-DZSTD_MULTITHREAD_SUPPORT=ON"
        "-DZSTD_BUILD_PROGRAMS=OFF"
    SOURCE_SUBDIR build/cmake
)

set(ZLIB_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/zlib")
set(ZLIB_INSTALL "${ZLIB_PREFIX}/install")

list(APPEND ZLIB_CMAKE_ARGS
    "-DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}"
    "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
    "-DCMAKE_INSTALL_PREFIX=${ZLIB_INSTALL}"
    "-DCMAKE_BUILD_TYPE=Release"
    "-DCMAKE_POSITION_INDEPENDENT_CODE=ON"
    "-DCMAKE_OSX_ARCHITECTURES=${EXT_OSX_ARCHITECTURES}"
    "-DCMAKE_OSX_DEPLOYMENT_TARGET=${CMAKE_OSX_DEPLOYMENT_TARGET}"
)

if (MADRONA_LINUX)
    # https://github.com/madler/zlib/issues/856
    list(APPEND ZLIB_CMAKE_ARGS
        "-DCMAKE_SHARED_LINKER_FLAGS=-Wl,--undefined-version"
    )
endif()

ExternalProject_Add(zlib
    URL "https://www.zlib.net/zlib-${ZLIB_VERSION}.tar.xz"
    URL_HASH SHA256=${ZLIB_HASH}
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    PREFIX "${ZLIB_PREFIX}"
    INSTALL_DIR "${ZLIB_INSTALL}"
    CMAKE_ARGS ${ZLIB_CMAKE_ARGS}
)

list(APPEND LLVM_COMMON_ARGS
    "-DCMAKE_BUILD_TYPE=Release"
)

set(LLVM_C_FLAGS "-fno-stack-protector")
set(LLVM_CXX_FLAGS "-fno-stack-protector")

FetchContent_Declare(llvm_project
    URL https://github.com/llvm/llvm-project/releases/download/llvmorg-${LLVM_VERSION}/llvm-project-${LLVM_VERSION}.src.tar.xz
    URL_HASH SHA256=${LLVM_HASH}
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
)
FetchContent_MakeAvailable(llvm_project)

set(LLVM_TOOLCHAIN_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/toolchain")
set(LLVM_TOOLCHAIN_BUILD "${LLVM_TOOLCHAIN_PREFIX}/build")
set(LLVM_TOOLCHAIN_INSTALL "${LLVM_TOOLCHAIN_PREFIX}/install")

# libcxx is built without exception & rtti support, so these flags are
# required during stage2 otherwise the stage2 binaries will fail to link
set(LLVM_STAGE2_C_FLAGS "${LLVM_C_FLAGS}")
set(LLVM_STAGE2_CXX_FLAGS "${LLVM_CXX_FLAGS} -fno-exceptions -fno-rtti")

if (MADRONA_X64)
    set(LLVM_STAGE2_C_FLAGS "${LLVM_C_FLAGS} -march=x86-64-v3")
    set(LLVM_STAGE2_CXX_FLAGS "${LLVM_CXX_FLAGS} -march=x86-64-v3")
endif()

set(LLVM_STAGE2_LINKER_FLAGS "${LLVM_STAGE2_CXX_FLAGS}")

if (APPLE)
    # On Darwin, the custom libc++'s path isn't in the default search path, but
    # headers are. This results in a weird situation with the custom libc++ ABI
    # namespace where the stage1 compiler is broken due to building libraries
    # with the custom headers but linking system libc++. This combines with the
    # fact that clang doesn't set rpath to a custom libc++ location by default
    # and the Darwin clang driver not supporting -static-libstdc++, resulting
    # in none of the stage2 binaries working even with the correct linking
    # path. For now, disabling the standard libc++ link and manually passing in
    # the static library works. Known LLVM bug for the header / search path
    # mismatch on Darwin: https://reviews.llvm.org/D45639
    set(LLVM_STAGE2_LINKER_FLAGS "${LLVM_STAGE2_LINKER_FLAGS} -nostdlib++ ${LLVM_TOOLCHAIN_BUILD}/lib/libc++.a")
endif()

list(APPEND LLVM_TOOLCHAIN_ARGS
    "-DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}"
    "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
    "-DCMAKE_C_FLAGS=${LLVM_C_FLAGS}"
    "-DCMAKE_CXX_FLAGS=${LLVM_CXX_FLAGS}"
    "-DCMAKE_INSTALL_PREFIX=${LLVM_TOOLCHAIN_INSTALL}"
    "-DCMAKE_PREFIX_PATH=${NCURSES_INSTALL}/usr/$<SEMICOLON>${ZSTD_INSTALL}$<SEMICOLON>${ZLIB_INSTALL}"
    "-DZLIB_USE_STATIC_LIBS=ON"
    "-DCMAKE_OSX_ARCHITECTURES=${EXT_OSX_ARCHITECTURES}"
    "-DCMAKE_OSX_DEPLOYMENT_TARGET=${CMAKE_OSX_DEPLOYMENT_TARGET}"
    "-DBOOTSTRAP_CMAKE_C_FLAGS=${LLVM_STAGE2_C_FLAGS}"
    "-DBOOTSTRAP_CMAKE_CXX_FLAGS=${LLVM_STAGE2_CXX_FLAGS}"
    "-DBOOTSTRAP_CMAKE_EXE_LINKER_FLAGS=${LLVM_STAGE2_LINKER_FLAGS}"
    "-DBOOTSTRAP_CMAKE_SHARED_LINKER_FLAGS=${LLVM_STAGE2_LINKER_FLAGS}"
    "-DBOOTSTRAP_CMAKE_MODULE_LINKER_FLAGS=${LLVM_STAGE2_LINKER_FLAGS}"
    "-C ${CMAKE_CURRENT_LIST_DIR}/llvm-stage1.cmake"
)

if (MADRONA_X64 AND MADRONA_LINUX)
    # Force to pc-linux as opposed to unknown-linux
    LIST(APPEND LLVM_TOOLCHAIN_ARGS 
        "-DLLVM_DEFAULT_TARGET_TRIPLE=x86_64-pc-linux-gnu"
    )
endif()

if (APPLE)
    list(APPEND LLVM_INSTALL_TARGETS 
        stage2-install-xcode-toolchain
        stage2-install-distribution-toolchain
    )

    # Clang bootstrap builds on darwin are written to assume that tools like
    # libtool are the system version. To avoid LLVM bitcode incompatibilities
    # that confuse the system cc tools, we want to make sure that the bootstrap
    # uses the new LLVM lipo & libtool. Unfortunately, this runs into issues
    # with the clang bootstrap tooling hardcoding DYLD_LIBRARY_PATH (assuming
    # the system tools are in use), breaking the stage 1 compiled libtool due
    # to linkage against the wrong libc++. This script is a fake libtool that
    # unsets DYLD_LIBRARY_PATH to avoid these issues
    set(LIBTOOL_HACK_PATH "${LLVM_TOOLCHAIN_BUILD}/bin/libtool")
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/apple_libtool_hack.sh.in"
        "${LLVM_TOOLCHAIN_BUILD}/apple_libtool_hack.sh"
        @ONLY
    )

    list(APPEND LLVM_TOOLCHAIN_ARGS
        "-DBOOTSTRAP_CMAKE_LIBTOOL=${LLVM_TOOLCHAIN_BUILD}/apple_libtool_hack.sh"
    )
else()
    list(APPEND LLVM_INSTALL_TARGETS 
        stage2-install-distribution-stripped
    )

    # On non-apple, we can do a normal static link against our new libc++ 
    # in the bootstrap build. This is handled with custom linker flags above
    # for mac builds.
    list(APPEND LLVM_TOOLCHAIN_ARGS
        "-DBOOTSTRAP_LLVM_STATIC_LINK_CXX_STDLIB=ON"
    )
endif()

# There seems to be a race condition in the clang bootstrap process due to
# improper dependencies in LLVM's internal use of ExternalProject_Add. 
# Forcing the generator to Ninja  fixes it because Ninja can't work on
# multiple ExternalProjects in parallel
ExternalProject_Add(llvm_toolchain
    DEPENDS zstd zlib ncurses
    SOURCE_DIR "${llvm_project_SOURCE_DIR}"
    PREFIX "${LLVM_TOOLCHAIN_PREFIX}"
    BINARY_DIR "${LLVM_TOOLCHAIN_BUILD}"
    INSTALL_DIR "${LLVM_TOOLCHAIN_INSTALL}"
    CMAKE_GENERATOR Ninja
    CMAKE_ARGS
        ${LLVM_COMMON_ARGS}
        ${LLVM_TOOLCHAIN_ARGS}
    SOURCE_SUBDIR llvm
    BUILD_COMMAND
        ninja -C "${LLVM_TOOLCHAIN_BUILD}" stage2-distribution
    INSTALL_COMMAND 
        ninja -C "${LLVM_TOOLCHAIN_BUILD}" ${LLVM_INSTALL_TARGETS}
)

install(DIRECTORY ${LLVM_TOOLCHAIN_INSTALL}/
    DESTINATION toolchain/
    USE_SOURCE_PERMISSIONS
)

if (APPLE)
    set(TOOLCHAIN_PATH "${LLVM_TOOLCHAIN_INSTALL}/Toolchains/LLVM${LLVM_VERSION}.xctoolchain/usr")
else()
    set(TOOLCHAIN_PATH "${LLVM_TOOLCHAIN_INSTALL}")
endif()

set(NEW_C_COMPILER "${TOOLCHAIN_PATH}/bin/clang")
set(NEW_CXX_COMPILER "${TOOLCHAIN_PATH}/bin/clang++")

set(LIBCXX_COMMON_ARGS ${LLVM_COMMON_ARGS})
list(APPEND LIBCXX_COMMON_ARGS
    "-DCMAKE_OSX_ARCHITECTURES=${EXT_OSX_ARCHITECTURES}"
    "-DCMAKE_OSX_DEPLOYMENT_TARGET=${CMAKE_OSX_DEPLOYMENT_TARGET}"
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON
    -DLLVM_ENABLE_RUNTIMES=libcxx$<SEMICOLON>libcxxabi$<SEMICOLON>libunwind
    "-DCMAKE_C_COMPILER=${NEW_C_COMPILER}"
    "-DCMAKE_CXX_COMPILER=${NEW_CXX_COMPILER}"
    "-DCMAKE_C_FLAGS=${LLVM_C_FLAGS} -nostdlib++ -nostdinc++"
    "-DCMAKE_CXX_FLAGS=${LLVM_CXX_FLAGS} -nostdlib++ -nostdinc++"
    "-DCMAKE_EXE_LINKER_FLAGS=-nostdlib++"
    "-DCMAKE_SHARED_LINKER_FLAGS=-nostdlib++"
    "-DCMAKE_MODULE_LINKER_FLAGS=-nostdlib++"
    -C "${CMAKE_CURRENT_LIST_DIR}/libcxx-common.cmake"
)

set(LIBCXX_MADRONA_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/libcxx-madrona")
set(LIBCXX_MADRONA_BUILD "${LIBCXX_MADRONA_PREFIX}/build")
set(LIBCXX_MADRONA_INSTALL "${LIBCXX_MADRONA_PREFIX}/install")

ExternalProject_Add(libcxx-madrona
    DEPENDS llvm_toolchain
    SOURCE_DIR "${llvm_project_SOURCE_DIR}"
    PREFIX "${LIBCXX_MADRONA_PREFIX}"
    BINARY_DIR "${LIBCXX_MADRONA_BUILD}"
    INSTALL_DIR "${LIBCXX_MADRONA_INSTALL}"
    CMAKE_GENERATOR Ninja
    CMAKE_ARGS
        ${LIBCXX_COMMON_ARGS}
        "-DCMAKE_INSTALL_PREFIX=${LIBCXX_MADRONA_INSTALL}"
        -DLIBCXX_ABI_NAMESPACE=__mad1
        -DLIBCXXABI_HERMETIC_STATIC_LIBRARY=ON
        -DLIBCXX_HERMETIC_STATIC_LIBRARY=ON
        -DLIBCXX_ENABLE_SHARED=OFF
        -DLIBCXX_ENABLE_EXCEPTIONS=OFF
        -DLIBCXX_ENABLE_RTTI=OFF
        -DLIBCXX_SHARED_OUTPUT_NAME=c++-madrona
        -DLIBCXX_STATIC_OUTPUT_NAME=c++-madrona
        -DLIBCXX_EXTRA_SITE_DEFINES=_LIBCPP_DISABLE_VISIBILITY_ANNOTATIONS
        -DLIBCXX_ENABLE_NEW_DELETE_DEFINITIONS=OFF
        -DLIBCXXABI_ENABLE_NEW_DELETE_DEFINITIONS=OFF
        -DLIBUNWIND_INSTALL_LIBRARY=OFF
        -DLIBUNWIND_INSTALL_HEADERS=OFF
    SOURCE_SUBDIR runtimes
    BUILD_COMMAND ${CMAKE_COMMAND} --build "${LIBCXX_BUILD}" --target cxx --target cxx_experimental --target cxxabi --target unwind
)

install(DIRECTORY ${LIBCXX_MADRONA_INSTALL}/include
    DESTINATION libcxx-madrona/
    USE_SOURCE_PERMISSIONS
)

install(DIRECTORY ${LIBCXX_MADRONA_INSTALL}/lib
    DESTINATION libcxx-madrona/
    USE_SOURCE_PERMISSIONS
    FILES_MATCHING
        PATTERN *
        PATTERN *experimental* EXCLUDE
)

set(LIBCXX_HERMETIC_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/libcxx-hermetic")
set(LIBCXX_HERMETIC_BUILD "${LIBCXX_HERMETIC_PREFIX}/build")
set(LIBCXX_HERMETIC_INSTALL "${LIBCXX_HERMETIC_PREFIX}/install")

ExternalProject_Add(libcxx-hermetic
    DEPENDS llvm_toolchain
    SOURCE_DIR "${llvm_project_SOURCE_DIR}"
    PREFIX "${LIBCXX_HERMETIC_PREFIX}"
    BINARY_DIR "${LIBCXX_HERMETIC_BUILD}"
    INSTALL_DIR "${LIBCXX_HERMETIC_INSTALL}"
    CMAKE_GENERATOR Ninja
    CMAKE_ARGS
        ${LIBCXX_COMMON_ARGS}
        "-DCMAKE_INSTALL_PREFIX=${LIBCXX_HERMETIC_INSTALL}"
        -DLIBCXX_ABI_NAMESPACE=__herm1
        -DLIBCXXABI_HERMETIC_STATIC_LIBRARY=ON
        -DLIBCXX_HERMETIC_STATIC_LIBRARY=ON
        -DLIBCXX_ENABLE_SHARED=OFF
        -DLIBCXX_ENABLE_EXCEPTIONS=ON
        -DLIBCXX_ENABLE_RTTI=ON
        -DLIBCXX_SHARED_OUTPUT_NAME=c++-hermetic
        -DLIBCXX_STATIC_OUTPUT_NAME=c++-hermetic
        -DLIBUNWIND_INSTALL_LIBRARY=OFF
        -DLIBUNWIND_INSTALL_HEADERS=OFF
    SOURCE_SUBDIR runtimes
    BUILD_COMMAND ${CMAKE_COMMAND} --build "${LIBCXX_BUILD}" --target cxx --target cxx_experimental --target cxxabi --target unwind
)

install(DIRECTORY ${LIBCXX_HERMETIC_INSTALL}/include
    DESTINATION libcxx-hermetic/
    USE_SOURCE_PERMISSIONS
)

install(DIRECTORY ${LIBCXX_HERMETIC_INSTALL}/lib
    DESTINATION libcxx-hermetic/
    USE_SOURCE_PERMISSIONS
    FILES_MATCHING
        PATTERN *
        PATTERN *experimental* EXCLUDE
)
