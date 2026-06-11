function(madrona_cfg_toolchain)
    cmake_path(GET CMAKE_CURRENT_FUNCTION_LIST_DIR PARENT_PATH TOOLCHAIN_REPO)
    
    if (MADRONA_TOOLCHAIN_ROOT_OVERRIDE)
        set(TOOLCHAIN_ROOT "${MADRONA_TOOLCHAIN_ROOT_OVERRIDE}")
    else()
        set(TOOLCHAIN_ROOT "${TOOLCHAIN_REPO}/bundled-toolchain")
    endif()
    
    set(TOOLCHAIN_SYSROOT "${TOOLCHAIN_ROOT}/toolchain")
    
    if (APPLE)
        file(GLOB TOOLCHAIN_SYSROOT "${TOOLCHAIN_SYSROOT}/Toolchains/LLVM*.xctoolchain/usr")
    endif()
    
    set(CMAKE_C_COMPILER "${TOOLCHAIN_SYSROOT}/bin/clang" CACHE STRING "")
    set(CMAKE_CXX_COMPILER "${TOOLCHAIN_SYSROOT}/bin/clang++" CACHE STRING "")
    
    # On macos, universal builds with this toolchain will be broken due to
    # llvm-ranlib not working with univeral libraries. /usr/bin/ar is picked
    # by default by cmake, but the compiler ranlib is still used, breaking
    # static libraries. 
    # One option is to force /usr/bin/ranlib on macos.
    # The better option is to do what LLVM itself does, which is to just use
    # libtool on macos for building static libraries since it is more
    # optimized anyway: https://reviews.llvm.org/D19611. llvm-libtool
    # correctly handles universal binaries
    if (APPLE)
        set(CMAKE_CXX_CREATE_STATIC_LIBRARY "\"${TOOLCHAIN_SYSROOT}/bin/llvm-libtool-darwin\" -static -no_warning_for_no_symbols -o <TARGET> <LINK_FLAGS> <OBJECTS>" CACHE STRING "")
        # need to disable ranlib or it will run after libtool
        set(CMAKE_RANLIB "" CACHE STRING "")
    endif ()
endfunction()

madrona_cfg_toolchain()
unset(madrona_cfg_toolchain)
