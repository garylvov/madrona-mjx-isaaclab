cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
cmake_policy(VERSION 3.18)
set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
set(CMAKE_CONFIGURATION_TYPES "Release" CACHE STRING "" FORCE)

include(FetchContent)
include(ExternalProject)

get_filename_component(PARENT_DIR ${CMAKE_SOURCE_DIR} DIRECTORY)

include(versions.cmake)

FetchContent_Declare(madrona-toolchain-cmake
    GIT_REPOSITORY https://github.com/shacklettbp/madrona-toolchain.git
    GIT_TAG ${TOOLCHAIN_VERSION}
)

FetchContent_GetProperties(madrona-toolchain-cmake)
FetchContent_Populate(madrona-toolchain-cmake)

if (NOT WIN32)
    include("${madrona-toolchain-cmake_SOURCE_DIR}/cmake/set_toolchain.cmake")
endif()

if (APPLE)
    option(MADRONA_DEPS_MACOS_ARCH "macOS target architecture")
    if (NOT MADRONA_DEPS_MACOS_ARCH)
        message(FATAL_ERROR "Must specify macOS target architecture")
    endif()

    set(CMAKE_OSX_ARCHITECTURES "${MADRONA_DEPS_MACOS_ARCH}" CACHE STRING "")
    list(JOIN CMAKE_OSX_ARCHITECTURES "$<SEMICOLON>" EXT_OSX_ARCHITECTURES)
    set(CMAKE_OSX_DEPLOYMENT_TARGET "12.0" CACHE STRING "")
endif()

set(NAME madrona-deps-compile)
project(${NAME} LANGUAGES C CXX)
include("${PARENT_DIR}/cmake/sys-detect.cmake")

find_package(Python 3.6 COMPONENTS Interpreter)

if (NOT MADRONA_WINDOWS) #FIX
    FetchContent_GetProperties(MadronaBundledToolchain)

    set(MADRONA_LIBCXX_PATH "${madronabundledtoolchain_SOURCE_DIR}/libcxx-madrona")
    set(HERMETIC_LIBCXX_PATH "${madronabundledtoolchain_SOURCE_DIR}/libcxx-hermetic")
    set(OP_NEWDEL_PATH "${madronabundledtoolchain_SOURCE_DIR}/toolchain-standalone-op-newdel/lib/libmadrona_toolchain_standalone_op_newdel.a")

    set(MADRONA_LIBCXX_INC_FLAGS "-nostdinc++ -isystem ${MADRONA_LIBCXX_PATH}/include/c++/v1")
    set(MADRONA_LIBCXX_LINKER_FLAGS
        "-nostdlib++ ${MADRONA_LIBCXX_PATH}/lib/libc++-madrona.a ${OP_NEWDEL_PATH}")
    set(NO_LIBCXX_LINKER_FLAGS "-nostdlib++")

    set(HERMETIC_LIBCXX_INC_FLAGS "-nostdinc++ -isystem ${HERMETIC_LIBCXX_PATH}/include/c++/v1")
    set(HERMETIC_LIBCXX_LINKER_FLAGS
        "-nostdlib++ ${HERMETIC_LIBCXX_PATH}/lib/libc++-hermetic.a ${OP_NEWDEL_PATH}")
else ()
    set(MADRONA_LIBCXX_INC_FLAGS "")
    set(MADRONA_LIBCXX_LINKER_FLAGS "")
    set(NO_LIBCXX_LINKER_FLAGS "")
    set(HERMETIC_LIBCXX_INC_FLAGS "")
    set(HERMETIC_LIBCXX_LINKER_FLAGS "")
endif()

if (MADRONA_LINUX)
    set (SDL_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/SDL")
    set (SDL_INSTALL "${SDL_PREFIX}/install")
    
    list(APPEND SDL_ARGS
        "-DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}"
        "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
        "-DCMAKE_INSTALL_PREFIX=${SDL_INSTALL}"
        "-DCMAKE_SYSTEM_VERSION=${CMAKE_SYSTEM_VERSION}"
        "-DCMAKE_BUILD_TYPE=Release"
        "-DSDL_SHARED=ON"
        "-DSDL_STATIC=OFF"
        "-DSDL_ASSERTIONS=release"
        "-DSDL_WAYLAND=OFF"
        "-DSDL_KMSDRM=OFF"
        "-DSDL_OPENGLES=OFF"
        "-DSDL_DISKAUDIO=OFF"
        "-DSDL_X11=ON"
        "-DSDL_OPENGL=ON"
        "-DSDL_VULKAN=ON"
    )

    ExternalProject_Add(SDL
        GIT_REPOSITORY https://github.com/libsdl-org/SDL.git
        GIT_TAG "${SDL_VERSION}"
        GIT_SHALLOW ON
        PREFIX "${SDL_PREFIX}"
        INSTALL_DIR "${SDL_INSTALL}"
        CMAKE_ARGS ${SDL_ARGS}
        CMAKE_GENERATOR Ninja
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )

    install(DIRECTORY "${SDL_INSTALL}/include/SDL2"
        DESTINATION SDL/include
    )
    
    install(DIRECTORY ${SDL_INSTALL}/lib/
        DESTINATION SDL/lib
        FILES_MATCHING
            PATTERN *
            PATTERN *pkgconfig* EXCLUDE
    )
endif()

set(EMBREE_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/embree")
set(EMBREE_INSTALL "${EMBREE_PREFIX}/install")

list(APPEND EMBREE_ARGS
    "-DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}"
    "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
    "-DCMAKE_CXX_FLAGS=${HERMETIC_LIBCXX_INC_FLAGS}"
    "-DCMAKE_EXE_LINKER_FLAGS=${HERMETIC_LIBCXX_LINKER_FLAGS}"
    "-DCMAKE_SHARED_LINKER_FLAGS=${HERMETIC_LIBCXX_LINKER_FLAGS}"
    "-DCMAKE_MODULE_LINKER_FLAGS=${HERMETIC_LIBCXX_LINKER_FLAGS}"
    "-DCMAKE_SYSTEM_VERSION=${CMAKE_SYSTEM_VERSION}"
    "-DCMAKE_INSTALL_PREFIX=${EMBREE_INSTALL}"
    "-DCMAKE_CONFIGURATION_TYPES=Release"
    "-DCMAKE_BUILD_TYPE=Release"
    "-DEMBREE_TUTORIALS=OFF"
    "-DEMBREE_MAX_ISA=NONE"
    "-DEMBREE_TASKING_SYSTEM=INTERNAL"
    "-DEMBREE_IGNORE_CMAKE_CXX_FLAGS=OFF"
)

if (MADRONA_MACOS OR MADRONA_IOS)
    list(APPEND EMBREE_ARGS 
        "-DCMAKE_OSX_ARCHITECTURES=${EXT_OSX_ARCHITECTURES}"
        "-DCMAKE_OSX_DEPLOYMENT_TARGET=${CMAKE_OSX_DEPLOYMENT_TARGET}"
    )
endif()

if (MADRONA_X64)
    list(APPEND EMBREE_ARGS
        "-DEMBREE_ISA_SSE2=ON"
        "-DEMBREE_ISA_SSE42=ON"
        "-DEMBREE_ISA_AVX=ON"
        "-DEMBREE_ISA_AVX2=ON"
        "-DEMBREE_ISA_AVX512=OFF"
        "-DEMBREE_ISA_NEON=OFF"
        "-DEMBREE_ISA_NEON2X=OFF"
    )
elseif (MADRONA_ARM)
    list(APPEND EMBREE_ARGS
        "-DEMBREE_ISA_SSE2=OFF"
        "-DEMBREE_ISA_SSE42=OFF"
        "-DEMBREE_ISA_AVX=OFF"
        "-DEMBREE_ISA_AVX2=OFF"
        "-DEMBREE_ISA_AVX512=OFF"
        # Double pumped NEON
        "-DEMBREE_ISA_NEON=OFF"
        "-DEMBREE_ISA_NEON2X=ON"
    )
endif()

ExternalProject_Add(embree
    GIT_REPOSITORY https://github.com/RenderKit/embree.git
    GIT_TAG "${EMBREE_VERSION}"
    GIT_SHALLOW ON
    PREFIX "${EMBREE_PREFIX}"
    INSTALL_DIR "${EMBREE_INSTALL}"
    CMAKE_ARGS ${EMBREE_ARGS}
    CMAKE_GENERATOR Ninja
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)

install(DIRECTORY "${EMBREE_INSTALL}/include/embree4"
    DESTINATION embree/include
)

install(DIRECTORY ${EMBREE_INSTALL}/lib/
    DESTINATION embree/lib
    FILES_MATCHING
        PATTERN *
        PATTERN *pkgconfig* EXCLUDE
)

if (MADRONA_WINDOWS)
    install(DIRECTORY ${EMBREE_INSTALL}/bin/
        DESTINATION embree/bin
        FILES_MATCHING
            PATTERN *dll
    )
endif()

set (VK_HDRS_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/vk_hdrs")
set (VK_HDRS_INSTALL "${VK_HDRS_PREFIX}/install")

list(APPEND VK_HDRS_ARGS
    "-DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}"
    "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
    "-DCMAKE_INSTALL_PREFIX=${VK_HDRS_INSTALL}"
    "-DCMAKE_BUILD_TYPE=Release"
    "-DCMAKE_CONFIGURATION_TYPES=Release"
    "-DCMAKE_SYSTEM_VERSION=${CMAKE_SYSTEM_VERSION}"
)

if (MADRONA_MACOS OR MADRONA_IOS)
    list(APPEND VK_HDRS_ARGS
        -DCMAKE_OSX_ARCHITECTURES=${EXT_OSX_ARCHITECTURES}
        -DCMAKE_OSX_DEPLOYMENT_TARGET=${CMAKE_OSX_DEPLOYMENT_TARGET}
    )
endif()

ExternalProject_Add(VulkanHeaders
    GIT_REPOSITORY https://github.com/KhronosGroup/Vulkan-Headers.git
    GIT_TAG vulkan-sdk-${VULKAN_HEADER_VERSION}
    PREFIX "${VK_HDRS_PREFIX}"
    INSTALL_DIR "${VK_HDRS_INSTALL}"
    CMAKE_ARGS ${VK_HDRS_ARGS}
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)

install(DIRECTORY ${VK_HDRS_INSTALL}/include/
    DESTINATION vk_hdrs/include/
)

set(VK_LOADER_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/vk_loader")
set(VK_LOADER_INSTALL "${VK_LOADER_PREFIX}/install")

list(APPEND VK_LOADER_ARGS
    "-DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}"
    "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
    "-DCMAKE_INSTALL_PREFIX=${VK_LOADER_INSTALL}"
    "-DCMAKE_BUILD_TYPE=Release"
    "-DCMAKE_CONFIGURATION_TYPES=Release"
    "-DCMAKE_SYSTEM_VERSION=${CMAKE_SYSTEM_VERSION}"
    "-DUPDATE_DEPS=ON"
)

if (MADRONA_MACOS OR MADRONA_IOS)
    list(APPEND VK_LOADER_ARGS
        -DCMAKE_OSX_ARCHITECTURES=${EXT_OSX_ARCHITECTURES}
        -DCMAKE_OSX_DEPLOYMENT_TARGET=${CMAKE_OSX_DEPLOYMENT_TARGET}
    )
endif()

ExternalProject_Add(VulkanLoader
    GIT_REPOSITORY https://github.com/KhronosGroup/Vulkan-Loader.git
    GIT_TAG vulkan-sdk-${VULKAN_LOADER_VERSION}
    PREFIX "${VK_LOADER_PREFIX}"
    INSTALL_DIR "${VK_LOADER_INSTALL}"
    CMAKE_ARGS ${VK_LOADER_ARGS}
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)

if (MADRONA_WINDOWS)
    install(DIRECTORY ${VK_LOADER_INSTALL}/bin/
        DESTINATION vk_loader/lib
        FILES_MATCHING
            PATTERN *dll
    )
else()
    install(DIRECTORY ${VK_LOADER_INSTALL}/lib/
        DESTINATION vk_loader/lib
        FILES_MATCHING
            PATTERN *
            PATTERN *pkgconfig* EXCLUDE
    )
endif()

if (MADRONA_MACOS)
    set (MOLTENVK_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/moltenvk")

    ExternalProject_Add(MoltenVK
        GIT_REPOSITORY https://github.com/KhronosGroup/MoltenVK.git
        GIT_TAG ${MOLTENVK_VERSION}
        PREFIX ${MOLTENVK_PREFIX}
        STAMP_DIR ${MOLTENVK_PREFIX}/stamps
        SOURCE_DIR ${MOLTENVK_PREFIX}/repo
        BINARY_DIR ${MOLTENVK_PREFIX}/repo
        CONFIGURE_COMMAND ""
        BUILD_COMMAND
            ./fetchDependencies --macos
            COMMAND make MVK_CONFIG_LOG_LEVEL=1 MVK_HIDE_VULKAN_SYMBOLS=1 macos
        INSTALL_COMMAND ""
    )

    install(DIRECTORY ${MOLTENVK_PREFIX}/repo/Package/Release/MoltenVK/dynamic/dylib/macOS/
        DESTINATION moltenvk/lib/
    )

    install(DIRECTORY ${MOLTENVK_PREFIX}/repo/Package/Release/MoltenVK/include/MoltenVK
        DESTINATION moltenvk/include/
    )

    FetchContent_Declare(metal-cpp
        URL "https://developer.apple.com/metal/cpp/files/metal-cpp_macOS13_iOS16.zip"
        URL_HASH SHA256=6f741894229e9c750add1afc3797274fc008c7507e2ae726370c17c34b7c6a68
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    FetchContent_MakeAvailable(metal-cpp)

    install(DIRECTORY ${metal-cpp_SOURCE_DIR}/Foundation
        DESTINATION metal-cpp/include
    )
    install(DIRECTORY ${metal-cpp_SOURCE_DIR}/Metal
        DESTINATION metal-cpp/include
    )
    install(DIRECTORY ${metal-cpp_SOURCE_DIR}/QuartzCore
        DESTINATION metal-cpp/include
    )
    install(FILES ${metal-cpp_SOURCE_DIR}/LICENSE.txt
        DESTINATION metal-cpp
    )

    # Want metal-cpp-extensions from this example
    # (why is it not distributed separately??)
    FetchContent_Declare(metal-cpp-extensions
        URL "https://developer.apple.com/metal/LearnMetalCPP.zip"
        URL_HASH SHA256=a709f3c0b532d5b9e3a5db3da3a35f3b783af27eb50f23a711115c02f86a256d
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
    )
    FetchContent_MakeAvailable(metal-cpp-extensions)

    set(metal_cpp_ext_dir "${metal-cpp-extensions_SOURCE_DIR}/metal-cpp-extensions")
    install(DIRECTORY "${metal_cpp_ext_dir}/AppKit"
        DESTINATION metal-cpp-extensions/include
    )
    install(DIRECTORY "${metal_cpp_ext_dir}/MetalKit"
        DESTINATION metal-cpp-extensions/include
    )
endif ()

set(DXC_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/dxc")
set(DXC_SRC "${DXC_PREFIX}/src")
set(DXC_BUILD "${DXC_PREFIX}/build")
set(DXC_INSTALL "${DXC_PREFIX}/install")

list(APPEND DXC_CMAKE_ARGS
    "-DCMAKE_INSTALL_PREFIX=${DXC_INSTALL}"
    "-DCMAKE_SYSTEM_VERSION=${CMAKE_SYSTEM_VERSION}"
    "-DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}"
    "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
    "-DCMAKE_CONFIGURATION_TYPES=Release"
    "-DCMAKE_BUILD_TYPE=Release"
    "-DCMAKE_OSX_ARCHITECTURES=${EXT_OSX_ARCHITECTURES}"
    "-DCMAKE_OSX_DEPLOYMENT_TARGET=${CMAKE_OSX_DEPLOYMENT_TARGET}"
    "-DCMAKE_CXX_FLAGS=${HERMETIC_LIBCXX_INC_FLAGS}"
    "-DCMAKE_EXE_LINKER_FLAGS=${HERMETIC_LIBCXX_LINKER_FLAGS}"
    "-DCMAKE_SHARED_LINKER_FLAGS=${HERMETIC_LIBCXX_LINKER_FLAGS}"
    "-DCMAKE_MODULE_LINKER_FLAGS=${HERMETIC_LIBCXX_LINKER_FLAGS}"
    "-DLLVM_ENABLE_ZLIB=OFF"
    "-DENABLE_SPIRV_CODEGEN=ON"
    "-C ${DXC_SRC}/cmake/caches/PredefinedParams.cmake"
)

if (MADRONA_MACOS)
    cmake_path(GET CMAKE_CXX_COMPILER PARENT_PATH TOOLCHAIN_BIN_DIR)

    # There is an annoying issue here: llvm-ar does not work with universal
    # binaries, because of this CMake uses system ar by default on macos.
    # However, it still tries to use llvm-ranlib by default, which then
    # breaks universal binaries. One option is to force system ranlib on macos
    # The better option is to do what LLVM itself does, which is to just use
    # libtool on macos for building static libraries since it is more
    # optimized anyway: https://reviews.llvm.org/D19611. llvm-libtool
    # correctly handles universal binaries
    list(APPEND DXC_CMAKE_ARGS
        "-DCMAKE_CXX_CREATE_STATIC_LIBRARY=\"${TOOLCHAIN_BIN_DIR}/llvm-libtool-darwin\" -static -no_warning_for_no_symbols -o <TARGET> <LINK_FLAGS> <OBJECTS>"
        "-DCMAKE_RANLIB=\"\""
    )
endif()

ExternalProject_Add(dxc
    GIT_REPOSITORY https://github.com/microsoft/DirectXShaderCompiler.git
    GIT_TAG "${DXC_VERSION}"
    PREFIX "${DXC_PREFIX}"
    SOURCE_DIR "${DXC_SRC}"
    STAMP_DIR "${DXC_PREFIX}/stamps"
    BINARY_DIR "${DXC_BUILD}"
    INSTALL_DIR "${DXC_INSTALL}"
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    CMAKE_GENERATOR Ninja
    INSTALL_COMMAND ninja -C "${DXC_BUILD}" install-distribution
    CMAKE_ARGS ${DXC_CMAKE_ARGS}
)

if (MADRONA_WINDOWS)
    find_package(Python 3.9 COMPONENTS Interpreter REQUIRED)

    ExternalProject_Add_Step(dxc download_taef
        COMMAND ${CMAKE_COMMAND} -E env HLSL_SRC_DIR=${DXC_SRC} ${Python_EXECUTABLE} "${DXC_SRC}/utils/hct/hctgettaef.py"
        DEPENDEES download update patch
        DEPENDERS configure
    )

    # For some reason dxcompiler.lib is left out of install-distribution
    install(FILES ${DXC_BUILD}/lib/dxcompiler.lib
        DESTINATION dxc/lib/
    )
else ()
    install(DIRECTORY ${DXC_INSTALL}/lib
        DESTINATION dxc/
        USE_SOURCE_PERMISSIONS
    )
endif ()

install(DIRECTORY ${DXC_INSTALL}/include
    DESTINATION dxc/
    USE_SOURCE_PERMISSIONS
)

install(DIRECTORY ${DXC_INSTALL}/bin
    DESTINATION dxc/
    USE_SOURCE_PERMISSIONS
)
