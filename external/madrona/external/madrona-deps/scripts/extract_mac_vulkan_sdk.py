#!/usr/bin/env python3

import sys
import struct
import subprocess
import os

if len(sys.argv) < 4:
    print(f"{sys.argv[0]} DMG_FILE DMG_HASH OUT_DIR", file=sys.stderr)
    sys.exit(1)

os.chdir(sys.argv[3])
os.makedirs("tmp", exist_ok=True)

try:
    with open("tmp/hash", "r") as f:
        cached_hash = f.read()
        hash_match = cached_hash == sys.argv[2]
except:
    hash_match = False

if hash_match:
    sys.exit(0)

subprocess.run(["hdiutil", "attach", "-quiet", "-mountpoint", "tmp", sys.argv[1]],
        check=True, capture_output=True)

dat_path = "tmp/InstallVulkan.app/Contents/Resources/installer.dat"

with open(dat_path, 'rb') as f:
    data = f.read()

subprocess.run(["hdiutil", "detach", "tmp"], check=True, capture_output=True)

# Search for every occurrence of the 7z header in the data file.
# This is fragile, because technically the header can repeat inside the
# 7z archive itself. Should technically parse the QT binary format or the
# 7z stream info to figure out the size of each 7z in the data file.
hdr_7z = b'7z' + bytes([0xBC, 0xAF, 0x27, 0x1C])

offsets = []
prev_offset = 0
while True:
    offset = data.find(hdr_7z, prev_offset)
    if offset == -1:
        break
    prev_offset = offset + 4
    offsets.append(offset)

for i, offset in enumerate(offsets):
    next_offset = offsets[i + 1] if i < len(offsets) - 1 else len(data)
    with open(f"tmp/dat{i}.7z", "wb") as f:
        f.write(data[offset:next_offset])

for i in range(len(offsets)):
    try:
        subprocess.run(["/usr/bin/env", "tar", "-x", "-f", f"tmp/dat{i}.7z", "macOS"],
            check=True, capture_output=True)

        break
    except:
        pass

with open("tmp/hash", "w") as f:
    f.write(sys.argv[2])
