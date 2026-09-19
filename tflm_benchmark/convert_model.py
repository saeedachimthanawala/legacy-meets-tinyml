"""
Converts a .tflite model file into a C array (model_data.cc / model_data.h)
that can be compiled directly into firmware -- no filesystem needed on the
target.

Usage (from within your tflite-micro repo root, in the MSYS2 UCRT64 terminal):
    python convert_model.py tensorflow/lite/micro/examples/hello_world/models/hello_world_float.tflite
"""
import sys

def main():
    if len(sys.argv) != 2:
        print("Usage: python convert_model.py <path-to-.tflite>")
        sys.exit(1)

    model_path = sys.argv[1]
    with open(model_path, "rb") as f:
        data = f.read()

    with open("model_data.h", "w") as f:
        f.write("#ifndef MODEL_DATA_H_\n#define MODEL_DATA_H_\n\n")
        f.write("extern const unsigned char g_model_data[];\n")
        f.write("extern const int g_model_data_len;\n\n")
        f.write("#endif  // MODEL_DATA_H_\n")

    with open("model_data.cc", "w") as f:
        f.write('#include "model_data.h"\n\n')
        f.write("alignas(8) const unsigned char g_model_data[] = {\n")
        for i, b in enumerate(data):
            f.write("0x%02x, " % b)
            if (i + 1) % 12 == 0:
                f.write("\n")
        f.write("\n};\n\n")
        f.write("const int g_model_data_len = %d;\n" % len(data))

    print(f"Wrote model_data.h and model_data.cc ({len(data)} bytes)")

if __name__ == "__main__":
    main()
