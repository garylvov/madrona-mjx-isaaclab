include(cmake/current-hashes.cmake)
include(FetchContent)

option(MADRONA_USE_BUNDLED "Use prebuilt dependency binaries" ON)

find_package(Git QUIET)

if (NOT DEFINED MADRONA_DEPS_VERSION)
    if (NOT Git_FOUND)
        message(FATAL_ERROR "Git not found, you must set explicit madrona-deps version to use (the short hash)")
    endif()

    execute_process(
        COMMAND "${GIT_EXECUTABLE}" rev-parse --short HEAD
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        OUTPUT_VARIABLE MADRONA_DEPS_VERSION
        OUTPUT_STRIP_TRAILING_WHITESPACE
        COMMAND_ERROR_IS_FATAL ANY
    )
endif()

get_property(MULTI_CFG GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if (MULTI_CFG)
    set(DEP_LIB_OUT_DIR "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/$<CONFIG>")
else ()
    set(DEP_LIB_OUT_DIR "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
endif ()

if (MADRONA_USE_BUNDLED)
    if (MADRONA_LINUX)
        set(DEPS_OS_NAME "linux")
        if (NOT DEFINED MADRONA_DEPS_HASH)
            set(MADRONA_DEPS_HASH "${MADRONA_DEPS_LINUX_HASH}")
        endif()
    elseif (MADRONA_MACOS)
        set(DEPS_OS_NAME "macos")
        if (NOT DEFINED MADRONA_DEPS_HASH)
            set(MADRONA_DEPS_HASH "${MADRONA_DEPS_MACOS_HASH}")
        endif()
    elseif (MADRONA_WINDOWS)
        set(DEPS_OS_NAME "windows")
        if (NOT DEFINED MADRONA_DEPS_HASH)
            set(MADRONA_DEPS_HASH "${MADRONA_DEPS_WINDOWS_HASH}")
        endif()
    endif()

    set(DEPS_URL "https://github.com/shacklettbp/madrona-deps/releases/download/${MADRONA_DEPS_VERSION}/madrona-deps-${MADRONA_DEPS_VERSION}-${DEPS_OS_NAME}.tar.xz")

    set(FETCHCONTENT_QUIET FALSE)
    set(FETCHCONTENT_BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/cmake-tmp")
    FetchContent_Declare(MadronaPrecompiledDeps
        URL ${DEPS_URL}
        URL_HASH SHA256=${MADRONA_DEPS_HASH}
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/bundled-deps"
    )

    FetchContent_MakeAvailable(MadronaPrecompiledDeps)

    set(DEP_ROOT ${madronaprecompileddeps_SOURCE_DIR})

    if (MADRONA_LINUX)
        find_package(SDL2
            PATHS "${DEP_ROOT}/SDL"
            QUIET
            REQUIRED
            NO_DEFAULT_PATH
        )
    endif()

    # Setup Embree
    
    find_package(embree
        PATHS "${DEP_ROOT}/embree"
        QUIET
        REQUIRED
        NO_DEFAULT_PATH
    )

    if (MADRONA_WINDOWS)
        find_file(vk_loader_src_loc
            NAMES vulkan-1.dll
            PATHS "${DEP_ROOT}/vk_loader/lib"
            REQUIRED
            NO_DEFAULT_PATH
        )
    else()
        find_library(vk_loader_src_loc
            NAMES libvulkan.so.1 libvulkan.1.dylib
            PATHS "${DEP_ROOT}/vk_loader/lib"
            REQUIRED
            NO_DEFAULT_PATH
        )
    endif()

    cmake_path(GET vk_loader_src_loc FILENAME vk_loader_name)
    set(MADRONA_BUNDLED_VK_LOADER "${DEP_LIB_OUT_DIR}/vk/${vk_loader_name}")

    add_custom_command(
        OUTPUT ${MADRONA_BUNDLED_VK_LOADER}
        DEPENDS ${vk_loader_src_loc}
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${vk_loader_src_loc}  ${MADRONA_BUNDLED_VK_LOADER}
    )

    add_custom_target(madrona_vk_loader
        DEPENDS ${MADRONA_BUNDLED_VK_LOADER}
    )

    add_library(madrona_vk_hdrs INTERFACE)
    target_include_directories(madrona_vk_hdrs
        INTERFACE "${DEP_ROOT}/vk_hdrs/include"
    )

    add_dependencies(madrona_vk_hdrs madrona_vk_loader)

    # Setup DXC
    
    if (MADRONA_WINDOWS)
        find_file(MADRONA_BUNDLED_LIBDXC dxcompiler.dll
            PATHS "${DEP_ROOT}/dxc/bin"
            REQUIRED
            NO_DEFAULT_PATH
        )

        find_library(MADRONA_BUNDLED_IMPLIBDXC dxcompiler.lib
            PATHS "${DEP_ROOT}/dxc/lib"
            REQUIRED
            NO_DEFAULT_PATH
        )
    else()
        find_library(MADRONA_BUNDLED_LIBDXC libdxcompiler.so libdxcompiler.dylib
            PATHS "${DEP_ROOT}/dxc/lib"
            REQUIRED
            NO_DEFAULT_PATH
        )
    endif()
    
    cmake_path(GET MADRONA_BUNDLED_LIBDXC FILENAME MADRONA_LIBDXC_NAME)

    set(MADRONA_LIBDXC_BUILDLOC "${DEP_LIB_OUT_DIR}/${MADRONA_LIBDXC_NAME}")
    
    add_custom_command(
        OUTPUT ${MADRONA_LIBDXC_BUILDLOC}
        DEPENDS ${MADRONA_BUNDLED_LIBDXC}
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${MADRONA_BUNDLED_LIBDXC}  ${MADRONA_LIBDXC_BUILDLOC}
    )
    
    add_custom_target(madrona_libdxc_shlib
        DEPENDS ${MADRONA_LIBDXC_BUILDLOC}
    )
    
    add_library(madrona_dxcompiler SHARED IMPORTED GLOBAL)
    add_dependencies(madrona_dxcompiler madrona_libdxc_shlib)
    set_target_properties(madrona_dxcompiler PROPERTIES IMPORTED_LOCATION
        ${MADRONA_LIBDXC_BUILDLOC})

    if (MADRONA_WINDOWS)
        set_target_properties(madrona_dxcompiler PROPERTIES IMPORTED_IMPLIB
            ${MADRONA_BUNDLED_IMPLIBDXC})
    endif()

    target_include_directories(madrona_dxcompiler SYSTEM INTERFACE
        $<BUILD_INTERFACE:${DEP_ROOT}/dxc/include/>
    )

    if (MADRONA_MACOS OR MADRONA_IOS)
        set(MOLTENVK_DYLIB_NAME "libMoltenVK.dylib")
        set(MOLTENVK_SRC_LOC "${DEP_ROOT}/moltenvk/lib/${MOLTENVK_DYLIB_NAME}")
        set(MOLTENVK_BUNDLED_LIB "${DEP_LIB_OUT_DIR}/vk/${MOLTENVK_DYLIB_NAME}")
        add_custom_command(
            OUTPUT "${MOLTENVK_BUNDLED_LIB}"
            DEPENDS "${MOLTENVK_SRC_LOC}"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different "${MOLTENVK_SRC_LOC}" "${MOLTENVK_BUNDLED_LIB}"
        )

        set(MOLTENVK_ICD_NAME "MoltenVK_icd.json")
        set(MOLTENVK_ICD_SRC "${DEP_ROOT}/moltenvk/lib/${MOLTENVK_ICD_NAME}")
        set(MOLTENVK_ICD_DST "${DEP_LIB_OUT_DIR}/vk/${MOLTENVK_ICD_NAME}")
        add_custom_command(
            OUTPUT "${MOLTENVK_ICD_DST}"
            DEPENDS "${MOLTENVK_ICD_SRC}"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different "${MOLTENVK_ICD_SRC}" "${MOLTENVK_ICD_DST}"
        )

        add_custom_target(madrona_moltenvk_lib DEPENDS
            "${MOLTENVK_BUNDLED_LIB}"
            "${MOLTENVK_ICD_DST}"
        )

        add_library(madrona_moltenvk INTERFACE)
        target_include_directories(madrona_moltenvk SYSTEM INTERFACE
            "${DEP_ROOT}/moltenvk/include")

        add_dependencies(madrona_moltenvk madrona_moltenvk_lib)

        add_library(madrona_metal_cpp INTERFACE)
        target_include_directories(madrona_metal_cpp SYSTEM INTERFACE
            "${DEP_ROOT}/metal-cpp/include"
            "${DEP_ROOT}/metal-cpp-extensions/include"
        )
    endif ()
else() # FIXME: The non-bundled path is not currently maintained, will not work.
    if (MADRONA_LINUX)
        find_package(SDL2 REQUIRED)
        find_package(embree REQUIRED)
    endif()

    if (MADRONA_LINUX OR MADRONA_WINDOWS)
        find_package(Vulkan REQUIRED)
        add_library(madrona_vk_hdrs INTERFACE)
        target_include_directories(madrona_vk_hdrs
            INTERFACE "${Vulkan_INCLUDE_DIRS}"
        )
    endif()

    if (MADRONA_LINUX)
        add_library(madrona_vk_loader INTERFACE)
        target_link_libraries(madrona_vk_loader INTERFACE
            ${Vulkan_LIBRARIES}
        )
    endif()

    if (MADRONA_MACOS OR MADRONA_IOS)
        if (NOT DEFINED MADRONA_METAL_CPP_DIR)
            message(FATAL_ERROR "Must define MADRONA_METAL_CPP_DIR on Mac when not using bundled dependencies")
        endif()

        if (NOT DEFINED MADRONA_METAL_CPP_EXT_DIR)
            message(FATAL_ERROR "Must define MADRONA_METAL_CPP_EXT_DIR (from LearnMetalCPP example) on Mac when not using bundled dependencies")
        endif()

        add_library(madrona_metal_cpp INTERFACE)
        target_include_directories(madrona_metal_cpp INTERFACE
            ${MADRONA_METAL_CPP_DIR}
            ${MADRONA_METAL_CPP_EXT_DIR}
        )
    endif()

    add_custom_target(madrona_vk_loader_prepare)
endif()

if (MADRONA_LINUX)
    add_library(madrona_sdl2 INTERFACE)
    target_link_libraries(madrona_sdl2
        INTERFACE SDL2::SDL2
    )
endif()

# Copy embree to build directory
get_target_property(EMBREE_SRC_LOC embree IMPORTED_LOCATION_RELEASE)
cmake_path(GET EMBREE_SRC_LOC FILENAME EMBREE_LIB_NAME)

set(EMBREE_OUT_LOC "${DEP_LIB_OUT_DIR}/${EMBREE_LIB_NAME}")

add_custom_command(
    OUTPUT ${EMBREE_OUT_LOC}
    DEPENDS ${EMBREE_SRC_LOC}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${EMBREE_SRC_LOC} ${EMBREE_OUT_LOC}
)

add_custom_target(madrona_embree_lib
    DEPENDS ${EMBREE_OUT_LOC}
)

set_target_properties(embree PROPERTIES
    IMPORTED_LOCATION_RELEASE ${EMBREE_OUT_LOC}
)

add_library(madrona_embree INTERFACE)
target_link_libraries(madrona_embree
    INTERFACE embree
)
add_dependencies(madrona_embree madrona_embree_lib)

if (MADRONA_MACOS)
    find_library(foundation_framework
        Foundation
        REQUIRED
    )

    find_library(quartzcore_framework
        QuartzCore
        REQUIRED
    )

    find_library(metal_framework
        Metal
        REQUIRED
    )

    find_library(appkit_framework
        AppKit
        REQUIRED
    )

    find_library(metalkit_framework
        MetalKit
        REQUIRED
    )

    add_library(madrona_metal_libs INTERFACE)
    target_link_libraries(madrona_metal_libs INTERFACE
        "${foundation_framework}"
        "${quartzcore_framework}"
        "${metal_framework}"
        "${appkit_framework}"
        "${metalkit_framework}"
    )
endif ()
