cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
cmake_policy(VERSION 3.18)
include("${CMAKE_CURRENT_SOURCE_DIR}/external/madrona/cmake/madrona_init.cmake")

project(MadronaMJX LANGUAGES C CXX)

# Package relocation (conda prefix replacement) rewrites baked path literals
# in place, shifting their bytes left within the original slot. Linker string
# tail-merging (SHF_MERGE) stores small literals (e.g. "/") as pointers into
# the tails of longer literals; after relocation those pointers land in the
# NUL padding and the literals silently read as empty strings, corrupting
# every runtime path/flag built by concatenation. Disable constant and
# linker string merging so each literal owns its slot.
add_compile_options(-fno-merge-all-constants)
add_link_options(-Wl,-O0)

include(setup)
include(dependencies)

add_subdirectory(external)
add_subdirectory(src)
