include(FetchContent)

option(MADRONA_USE_TOOLCHAIN "Use prebuilt toolchain" ON)
if (NOT MADRONA_USE_TOOLCHAIN)
    return()
endif()

function(madrona_setup_toolchain)
    cmake_path(GET CMAKE_CURRENT_FUNCTION_LIST_DIR PARENT_PATH TOOLCHAIN_REPO)

    include("${TOOLCHAIN_REPO}/cmake/current-hashes.cmake")
    include("${TOOLCHAIN_REPO}/cmake/sys-detect.cmake")

    if (NOT DEFINED MADRONA_TOOLCHAIN_VERSION)
        # This tree is vendored (no per-directory git checkout), so the
        # upstream `git rev-parse --short HEAD` version probe cannot work.
        # Pin to the madrona-toolchain commit this tree was vendored at.
        set(MADRONA_TOOLCHAIN_VERSION "8c0b55b")
    endif()

    if (MADRONA_LINUX)
        set(TOOLCHAIN_OS_NAME "linux")
        if (NOT DEFINED MADRONA_TOOLCHAIN_HASH)
            set(MADRONA_TOOLCHAIN_HASH "${MADRONA_TOOLCHAIN_LINUX_HASH}")
        endif()

        execute_process(COMMAND uname -m
            OUTPUT_VARIABLE TOOLCHAIN_ARCH
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
    elseif (MADRONA_MACOS)
        set(TOOLCHAIN_OS_NAME "macos")
        if (NOT DEFINED MADRONA_TOOLCHAIN_HASH)
            set(MADRONA_TOOLCHAIN_HASH "${MADRONA_TOOLCHAIN_MACOS_HASH}")
        endif()

        execute_process(COMMAND uname -m
            OUTPUT_VARIABLE TOOLCHAIN_ARCH
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
    endif()
    
    set(DEPS_URL "https://github.com/shacklettbp/madrona-toolchain/releases/download/${MADRONA_TOOLCHAIN_VERSION}/madrona-toolchain-${MADRONA_TOOLCHAIN_VERSION}-${TOOLCHAIN_OS_NAME}-${TOOLCHAIN_ARCH}.tar.xz")
    
    set(FETCHCONTENT_QUIET FALSE)
    set(FETCHCONTENT_BASE_DIR "${TOOLCHAIN_REPO}/cmake-tmp")
    FetchContent_Declare(MadronaBundledToolchain
        URL "${DEPS_URL}"
        URL_HASH SHA256=${MADRONA_TOOLCHAIN_HASH}
        SOURCE_DIR "${TOOLCHAIN_REPO}/bundled-toolchain"
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE 
    )
    
    FetchContent_MakeAvailable(MadronaBundledToolchain)
    
    set(CMAKE_TOOLCHAIN_FILE "${TOOLCHAIN_REPO}/cmake/toolchain.cmake" PARENT_SCOPE)
endfunction()

madrona_setup_toolchain()
unset(madrona_setup_toolchain)
