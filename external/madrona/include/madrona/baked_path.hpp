#pragma once

#include <string>

namespace madrona {

// Paths baked into the binary at configure time (shader/data dirs, NVRTC
// device source dirs) may be rewritten in place by package relocation
// (e.g. conda prefix replacement), which shortens the C string and pads the
// remainder with NULs. A std::string or std::filesystem::path constructed
// directly from the literal lets the compiler constant-fold the ORIGINAL
// (pre-relocation) length, capturing the NUL padding inside the string; any
// component appended afterwards lands after an embedded NUL and is invisible
// through c_str(). The asm barrier launders the pointer so the length is
// computed at runtime from the relocated bytes.
inline std::string bakedPath(const char *baked)
{
    const char *p = baked;
#if defined(__GNUC__) || defined(__clang__)
    __asm__ volatile("" : "+r"(p));
#endif
    return std::string(p);
}

}
