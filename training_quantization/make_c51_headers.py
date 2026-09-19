"""
make_c51_headers.py
====================
Converts the ARM-ready headers from Day 1 (model_Nunits.h,
test_vectors_Nunits.h) into 8051/Keil-C51-ready versions.

WHY THIS IS NEEDED
-------------------
The AT89C51 has only 128 bytes of internal RAM. Your weight/bias/
test-vector arrays (the `static const` arrays in the Day 1 headers)
would try to fit into that tiny RAM by default under Keil C51's
normal rules -- and for the 16-unit model alone the weights already
take ~370+ bytes. This script inserts the `code` keyword before every
array declaration, which tells the C51 compiler to place that data in
FLASH (program memory) instead, where it belongs (it's read-only
data, and flash is comparatively large). This does not change a
single number in your model -- only where the constants physically
live in the chip's memory.

USAGE
------
Copy this script into the SAME folder as your Day 1 `output/` folder
(the one containing model_4units.h, test_vectors_4units.h, etc. --
these should be the REAL versions you generated in Colab, not the
synthetic-data test versions). Then run:

    python make_c51_headers.py

It writes model_Nunits_c51.h and test_vectors_Nunits_c51.h alongside
your originals -- your ARM headers are left untouched, so Day 2's
project keeps working exactly as before.
"""
import re
import os

OUTPUT_DIR = "./output"
SIZES = [4, 8, 16]


def convert_to_c51(text):
    """
    1. Remove the `#include <stdint.h>` line -- headers must NOT
       define int8_t/int32_t themselves; only c51_types.h (included
       once, from inference_template_8051.c) does that, to avoid
       C231 redefinition errors.
    2. Rewrite `static const TYPE NAME[...]` into the standard,
       unambiguous Keil C51 idiom for ROM arrays:
           static TYPE code NAME[...]
       (no `const` at all -- `code` already places the data in flash,
       and mixing `const` with `code` is what produced the C141
       syntax errors).
    """
    text = re.sub(r'(?m)^#include <stdint\.h>\n?', '', text)
    text = re.sub(
        r'(?m)^static const (\w+)\s+(\w+\[)',
        r'static \1 code \2',
        text,
    )
    return text


def convert_file(src_name, dst_name):
    src_path = os.path.join(OUTPUT_DIR, src_name)
    dst_path = os.path.join(OUTPUT_DIR, dst_name)
    if not os.path.exists(src_path):
        print(f"SKIPPED (not found): {src_path}")
        return
    with open(src_path, "r") as f:
        content = f.read()
    converted = convert_to_c51(content)
    with open(dst_path, "w") as f:
        f.write(converted)
    print(f"Wrote {dst_path}")


if __name__ == "__main__":
    for n in SIZES:
        convert_file(f"model_{n}units.h", f"model_{n}units_c51.h")
        convert_file(f"test_vectors_{n}units.h", f"test_vectors_{n}units_c51.h")
    print("\nDone. Use the *_c51.h versions in your Keil C51 (8051) project.")
