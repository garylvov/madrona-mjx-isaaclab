if (MADRONA_TOOLCHAIN_ROOT_OVERRIDE)
    set(TOOLCHAIN_ROOT "${MADRONA_TOOLCHAIN_ROOT_OVERRIDE}")
else()
    set(TOOLCHAIN_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/bundled-toolchain")
endif()

set(TOOLCHAIN_SYSROOT "${TOOLCHAIN_ROOT}/toolchain")

if (APPLE)
    file(GLOB TOOLCHAIN_SYSROOT "${TOOLCHAIN_SYSROOT}/Toolchains/LLVM*.xctoolchain/usr")
endif()

add_library(madrona_libcxx INTERFACE)

target_compile_options(madrona_libcxx INTERFACE
    -nostdinc++ -nostdlib++
)

target_include_directories(madrona_libcxx SYSTEM INTERFACE
    $<BUILD_INTERFACE:${TOOLCHAIN_ROOT}/libcxx-madrona/include/c++/v1>
)

target_link_libraries(madrona_libcxx INTERFACE
    "${TOOLCHAIN_ROOT}/libcxx-madrona/lib/libc++-madrona.a"
)

target_link_options(madrona_libcxx INTERFACE
    -nostdlib++
)

add_library(madrona_libcxx_hdrs INTERFACE)

target_compile_options(madrona_libcxx_hdrs INTERFACE
    -nostdinc++ -nostdlib++
)

target_include_directories(madrona_libcxx_hdrs SYSTEM INTERFACE
    $<BUILD_INTERFACE:${TOOLCHAIN_ROOT}/libcxx-madrona/include/c++/v1>
)

install(TARGETS madrona_libcxx EXPORT madronaLibcxx)
install(EXPORT madronaLibcxx DESTINATION ${CMAKE_INSTALL_PREFIX})
