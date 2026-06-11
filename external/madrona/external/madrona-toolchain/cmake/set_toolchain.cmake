include(FetchContent)

option(MADRONA_USE_TOOLCHAIN "Use prebuilt toolchain" ON)
if (NOT MADRONA_USE_TOOLCHAIN)
    return()
endif()

function(madrona_setup_toolchain)
    cmake_path(GET CMAKE_CURRENT_FUNCTION_LIST_DIR PARENT_PATH TOOLCHAIN_REPO)

    include("${TOOLCHAIN_REPO}/cmake/current-hashes.cmake")
    include("${TOOLCHAIN_REPO}/cmake/sys-detect.cmake")

    find_package(Git QUIET)
    if (NOT DEFINED MADRONA_TOOLCHAIN_VERSION)
        if (NOT Git_FOUND)
            message(FATAL_ERROR "git not found, you must set MADRONA_TOOLCHAIN_VERSION to the short hash of the toolchain commit")
        endif()
    
        execute_process(
            COMMAND "${GIT_EXECUTABLE}" rev-parse --short HEAD
            WORKING_DIRECTORY "${TOOLCHAIN_REPO}"
            OUTPUT_VARIABLE MADRONA_TOOLCHAIN_VERSION
            OUTPUT_STRIP_TRAILING_WHITESPACE
            COMMAND_ERROR_IS_FATAL ANY
        )
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
