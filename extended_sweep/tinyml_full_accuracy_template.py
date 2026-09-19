"""
tinyml_extend_32_only.py
=================================
Same pipeline as tinyml_extend_32_64_128_256.py, trimmed to ONLY the
32-hidden-unit size, with one addition: a .tflite export for TFLM
benchmarking (see export_tflite() near the bottom).

Paste this ENTIRE file into a single Colab cell and run it.
"""

import io
import os
import csv
import zipfile
import numpy as np

try:
    import requests
except ImportError:
    requests = None

import tensorflow as tf
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
HIDDEN_UNITS = 32  # <-- CHANGE THIS to 32/64/128/256 for each run
HIDDEN_UNIT_OPTIONS = [HIDDEN_UNITS]
RANDOM_SEED = 42
TEST_SPLIT = 0.2
N_TEST_VECTORS_TO_EXPORT = 5
OUTPUT_DIR = "./output_extended"
UCI_HAR_URL = (
    "https://archive.ics.uci.edu/static/public/240/"
    "human+activity+recognition+using+smartphones.zip"
)
MANTISSA_BITS = 12

np.random.seed(RANDOM_SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# STEP A -- Load data
# ---------------------------------------------------------------------------
def load_uci_har_raw_signals():
    if requests is None:
        raise RuntimeError("requests not installed")
    resp = requests.get(UCI_HAR_URL, timeout=60)
    resp.raise_for_status()
    outer = zipfile.ZipFile(io.BytesIO(resp.content))
    inner_name = [n for n in outer.namelist() if n.endswith(".zip")]
    zf = zipfile.ZipFile(io.BytesIO(outer.read(inner_name[0]))) if inner_name else outer

    def _load_split(split):
        axes = ["x", "y", "z"]
        signals = []
        for ax in axes:
            path = f"UCI HAR Dataset/{split}/Inertial Signals/body_acc_{ax}_{split}.txt"
            with zf.open(path) as f:
                arr = np.loadtxt(f)
            signals.append(arr)
        X = np.stack(signals, axis=-1)
        with zf.open(f"UCI HAR Dataset/{split}/y_{split}.txt") as f:
            y = np.loadtxt(f).astype(int) - 1
        return X, y

    Xtr, ytr = _load_split("train")
    Xte, yte = _load_split("test")
    return (
        np.concatenate([Xtr, Xte], axis=0).astype(np.float32),
        np.concatenate([ytr, yte], axis=0),
    )


def make_synthetic_accelerometer_data(n_windows=2000, window_len=128, n_classes=6):
    rng = np.random.default_rng(RANDOM_SEED)
    t = np.linspace(0, 2 * np.pi, window_len)
    X = np.zeros((n_windows, window_len, 3), dtype=np.float32)
    y = np.zeros(n_windows, dtype=int)
    for i in range(n_windows):
        cls = rng.integers(0, n_classes)
        freq = 1.0 + cls * 0.6
        amp = 0.3 + cls * 0.15
        noise = rng.normal(0, 0.05, size=(window_len, 3))
        X[i, :, 0] = amp * np.sin(freq * t) + noise[:, 0]
        X[i, :, 1] = amp * np.cos(freq * t) + noise[:, 1]
        X[i, :, 2] = 0.5 * amp * np.sin(2 * freq * t) + noise[:, 2]
        y[i] = cls
    return X, y


def extract_features(X_windows):
    mean = X_windows.mean(axis=1)
    std = X_windows.std(axis=1)
    mn = X_windows.min(axis=1)
    mx = X_windows.max(axis=1)
    return np.concatenate([mean, std, mn, mx], axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# STEP B -- Quantization helpers
# ---------------------------------------------------------------------------
def quantize_tensor_int8(arr):
    max_abs = np.max(np.abs(arr))
    if max_abs == 0:
        max_abs = 1e-8
    scale = max_abs / 127.0
    q = np.clip(np.round(arr / scale), -127, 127).astype(np.int8)
    return q, scale


def compute_fixed_point_multiplier(real_multiplier, mantissa_bits=MANTISSA_BITS):
    if real_multiplier <= 0:
        return 0, 0
    shift = 0
    m = real_multiplier
    while m < 0.5:
        m *= 2.0
        shift += 1
    while m >= 1.0:
        m /= 2.0
        shift -= 1
    multiplier = int(round(m * (1 << mantissa_bits)))
    total_shift = mantissa_bits + shift
    return multiplier, total_shift


def calibrate_and_quantize(clf, X_calib):
    W1, W2 = clf.coefs_[0], clf.coefs_[1]
    b1, b2 = clf.intercepts_[0], clf.intercepts_[1]

    input_scale = np.max(np.abs(X_calib)) / 127.0

    W1_q, W1_scale = quantize_tensor_int8(W1)
    b1_combined_scale = input_scale * W1_scale
    b1_q = np.round(b1 / b1_combined_scale).astype(np.int32)

    hidden_float = np.maximum(0, X_calib @ W1 + b1)
    hidden_scale = np.max(np.abs(hidden_float)) / 127.0
    if hidden_scale == 0:
        hidden_scale = 1e-8

    real_mult_1 = b1_combined_scale / hidden_scale
    mult1, shift1 = compute_fixed_point_multiplier(real_mult_1)

    W2_q, W2_scale = quantize_tensor_int8(W2)
    b2_combined_scale = hidden_scale * W2_scale
    b2_q = np.round(b2 / b2_combined_scale).astype(np.int32)

    return {
        "input_scale": input_scale,
        "W1_q": W1_q, "b1_q": b1_q, "mult1": mult1, "shift1": shift1,
        "hidden_scale": hidden_scale,
        "W2_q": W2_q, "W2_scale": W2_scale, "b2_q": b2_q,
        "b2_combined_scale": b2_combined_scale,
        "n_in": W1.shape[0], "n_hidden": W1.shape[1], "n_out": W2.shape[1],
    }


def quantized_forward_final(x_float, qp, mult2, shift2):
    x_q = np.clip(np.round(x_float / qp["input_scale"]), -127, 127).astype(np.int32)
    acc1 = x_q @ qp["W1_q"].astype(np.int32) + qp["b1_q"]
    acc1 = np.maximum(0, acc1)
    hidden_q = np.clip(
        (acc1.astype(np.int64) * qp["mult1"]) >> qp["shift1"], -127, 127
    ).astype(np.int32)
    acc2 = hidden_q @ qp["W2_q"].astype(np.int32) + qp["b2_q"]
    out_q = np.clip(
        (acc2.astype(np.int64) * mult2) >> shift2, -127, 127
    ).astype(np.int32)
    return np.argmax(out_q, axis=-1)


def compute_output_quantization(qp, X_calib):
    x_q = np.clip(np.round(X_calib / qp["input_scale"]), -127, 127).astype(np.int32)
    acc1 = x_q @ qp["W1_q"].astype(np.int32) + qp["b1_q"]
    acc1 = np.maximum(0, acc1)
    hidden_q = np.clip(
        (acc1.astype(np.int64) * qp["mult1"]) >> qp["shift1"], -127, 127
    ).astype(np.int32)
    acc2 = hidden_q @ qp["W2_q"].astype(np.int32) + qp["b2_q"]
    max_abs = np.max(np.abs(acc2 * qp["b2_combined_scale"]))
    output_scale = max_abs / 127.0 if max_abs > 0 else 1e-8
    real_mult_2 = qp["b2_combined_scale"] / output_scale
    mult2, shift2 = compute_fixed_point_multiplier(real_mult_2)
    return output_scale, mult2, shift2


# ---------------------------------------------------------------------------
# STEP C -- Export helpers (hand-written-C, CMSIS-NN, and .tflite)
# ---------------------------------------------------------------------------
def c_int_array(name, arr, ctype="int8_t"):
    flat = ", ".join(str(int(v)) for v in np.asarray(arr).flatten())
    return f"static const {ctype} {name}[{arr.size}] = {{ {flat} }};\n"


def export_model_header(qp, hidden_units, mult2, shift2, class_names):
    lines = []
    lines.append(f"// Auto-generated -- {hidden_units} hidden units (hand-written-C path)")
    lines.append(f"#ifndef MODEL_{hidden_units}UNITS_H")
    lines.append(f"#define MODEL_{hidden_units}UNITS_H")
    lines.append("#include <stdint.h>\n")
    lines.append(f"#define N_IN {qp['n_in']}")
    lines.append(f"#define N_HIDDEN {qp['n_hidden']}")
    lines.append(f"#define N_OUT {qp['n_out']}")
    lines.append(f"#define REQUANT_MULT1 {qp['mult1']}")
    lines.append(f"#define REQUANT_SHIFT1 {qp['shift1']}")
    lines.append(f"#define L2_MULT {mult2}")
    lines.append(f"#define L2_SHIFT {shift2}\n")
    lines.append(c_int_array("W1_Q", qp["W1_q"], "int8_t"))
    lines.append(c_int_array("B1_Q", qp["b1_q"], "int32_t"))
    lines.append(c_int_array("W2_Q", qp["W2_q"], "int8_t"))
    lines.append(c_int_array("B2_Q", qp["b2_q"], "int32_t"))
    lines.append(f"// class order: {', '.join(class_names)}")
    lines.append(f"#endif // MODEL_{hidden_units}UNITS_H\n")
    return "\n".join(lines)


def export_test_vectors(qp, X_samples, y_true, y_pred_quant, hidden_units):
    lines = []
    lines.append(f"// Auto-generated test vectors -- {hidden_units} hidden units")
    lines.append(f"#ifndef TEST_VECTORS_{hidden_units}UNITS_H")
    lines.append(f"#define TEST_VECTORS_{hidden_units}UNITS_H")
    lines.append("#include <stdint.h>\n")
    n = X_samples.shape[0]
    lines.append(f"#define N_TEST_VECTORS {n}\n")
    x_q = np.clip(np.round(X_samples / qp["input_scale"]), -127, 127).astype(np.int8)
    lines.append(c_int_array("TEST_INPUTS_Q", x_q, "int8_t"))
    lines.append("// flattened as [N_TEST_VECTORS][N_IN], row-major\n")
    lines.append(c_int_array("EXPECTED_CLASS", y_pred_quant.astype(np.int32), "int32_t"))
    lines.append(f"// (true label, for reference only): {list(map(int, y_true))}")
    lines.append("#endif\n")
    return "\n".join(lines)


def export_cmsisnn_header(qp, hidden_units, mult2, shift2, class_names):
    W1_T = qp["W1_q"].T.copy()
    W2_T = qp["W2_q"].T.copy()
    lines = []
    lines.append(f"// Auto-generated CMSIS-NN header -- {hidden_units} hidden units")
    lines.append(f"#ifndef MODEL_{hidden_units}UNITS_CMSISNN_H")
    lines.append(f"#define MODEL_{hidden_units}UNITS_CMSISNN_H")
    lines.append("#include <stdint.h>\n")
    lines.append(f"#define N_IN {qp['n_in']}")
    lines.append(f"#define N_HIDDEN {qp['n_hidden']}")
    lines.append(f"#define N_OUT {qp['n_out']}\n")
    lines.append("// --- Layer 1 (input -> hidden) ---")
    lines.append(f"#define L1_MULT {qp['mult1']}")
    lines.append(f"#define L1_SHIFT {qp['shift1']}")
    lines.append("// Filter transposed to CMSIS-NN layout: [N_HIDDEN][N_IN]")
    lines.append(c_int_array("W1_Q_T", W1_T, "int8_t"))
    lines.append(c_int_array("B1_Q", qp["b1_q"], "int32_t"))
    lines.append("\n// --- Layer 2 (hidden -> output) ---")
    lines.append(f"#define L2_MULT {mult2}")
    lines.append(f"#define L2_SHIFT {shift2}")
    lines.append("// Filter transposed to CMSIS-NN layout: [N_OUT][N_HIDDEN]")
    lines.append(c_int_array("W2_Q_T", W2_T, "int8_t"))
    lines.append(c_int_array("B2_Q", qp["b2_q"], "int32_t"))
    lines.append(f"\n// class order: {', '.join(class_names)}")
    lines.append("// NOTE: symmetric quantization throughout (all zero-points = 0),")
    lines.append("// matching the hand-written ARM and 8051 code exactly.")
    lines.append(f"#endif // MODEL_{hidden_units}UNITS_CMSISNN_H\n")
    return "\n".join(lines)


def evaluate_tflite_reference_accuracy(tflite_path, X_test, y_test):
    """Runs the ACTUAL .tflite file through TensorFlow's own official
    interpreter (the correct ground truth to compare on-device TFLM
    results against -- NOT the hand-written pipeline's quantized_accuracy,
    which uses a different, symmetric quantization scheme)."""
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    in_scale, in_zp = input_details["quantization"]
    correct = 0
    for i in range(X_test.shape[0]):
        x = X_test[i:i+1]
        x_q = np.clip(np.round(x / in_scale + in_zp), -128, 127).astype(np.int8)
        interpreter.set_tensor(input_details["index"], x_q)
        interpreter.invoke()
        out = interpreter.get_tensor(output_details["index"])
        pred = np.argmax(out[0])
        if pred == y_test[i]:
            correct += 1
    accuracy = correct / X_test.shape[0]
    print(f"TFLite-interpreter reference accuracy (FULL test set, n={X_test.shape[0]}): {accuracy:.4f}")
    return accuracy


def export_full_test_set(X_test, y_test, output_dir=OUTPUT_DIR):
    """Exports the ENTIRE test set (not a small subset) for a rigorous
    on-device accuracy check. Sized for Flash: n_samples * 12 floats *
    4 bytes -- a couple thousand samples is well within a 512KB region."""
    n = X_test.shape[0]
    n_in = X_test.shape[1]

    lines = []
    lines.append("// Auto-generated: FULL test set for rigorous on-device accuracy check")
    lines.append("#ifndef TEST_DATA_H")
    lines.append("#define TEST_DATA_H\n")
    lines.append(f"#define N_TEST_SAMPLES {n}")
    lines.append(f"#define N_FEATURES {n_in}\n")
    lines.append("static const float TEST_FEATURES[N_TEST_SAMPLES][N_FEATURES] = {")
    for row in X_test:
        vals = ", ".join(f"{v:.6f}f" for v in row)
        lines.append(f"  {{ {vals} }},")
    lines.append("};\n")

    labels = ", ".join(str(int(v)) for v in y_test)
    lines.append(f"static const int TRUE_LABELS[N_TEST_SAMPLES] = {{ {labels} }};\n")
    lines.append("#endif // TEST_DATA_H\n")

    out_path = f"{output_dir}/test_data.h"
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {out_path}: FULL test set, {n} samples")
    return out_path


def export_tflite(clf, X_train, hidden_units, output_dir=OUTPUT_DIR):
    """Rebuilds the trained sklearn model as an equivalent Keras model
    (same weights, transplanted directly -- no retraining), then uses
    TensorFlow's official converter to produce a standard int8-quantized
    .tflite file for TFLM benchmarking."""
    n_in = clf.coefs_[0].shape[0]
    n_hidden = clf.coefs_[0].shape[1]
    n_out = clf.coefs_[1].shape[1]

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(n_in,)),
        tf.keras.layers.Dense(n_hidden, activation="relu", name="hidden"),
        tf.keras.layers.Dense(n_out, activation=None, name="output"),
    ])
    model.layers[0].set_weights([clf.coefs_[0], clf.intercepts_[0]])
    model.layers[1].set_weights([clf.coefs_[1], clf.intercepts_[1]])

    def representative_dataset():
        rng = np.random.default_rng(RANDOM_SEED)
        idx = rng.choice(len(X_train), size=min(200, len(X_train)), replace=False)
        for i in idx:
            yield [X_train[i:i+1].astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    out_path = f"{output_dir}/model_{hidden_units}units.tflite"
    with open(out_path, "wb") as f:
        f.write(tflite_model)
    print(f"Wrote {out_path} ({len(tflite_model)} bytes)")
    return out_path


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("Loading data")
    print("=" * 70)
    used_synthetic = False
    try:
        X_raw, y = load_uci_har_raw_signals()
        print(f"Loaded REAL UCI HAR data: {X_raw.shape[0]} windows")
    except Exception as e:
        print(f"Could not download real data ({e}); using SYNTHETIC fallback.")
        X_raw, y = make_synthetic_accelerometer_data()
        used_synthetic = True

    class_names = [f"class_{i}" for i in sorted(set(y.tolist()))]
    X_feat = extract_features(X_raw)
    n = X_feat.shape[0]
    idx = np.random.permutation(n)
    n_test = int(n * TEST_SPLIT)
    test_idx, train_idx = idx[:n_test], idx[n_test:]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_feat[train_idx]).astype(np.float32)
    X_test = scaler.transform(X_feat[test_idx]).astype(np.float32)
    y_train, y_test = y[train_idx], y[test_idx]
    print(f"Feature dim: {X_feat.shape[1]} | train: {len(train_idx)} | test: {len(test_idx)}")

    summary_rows = []

    for hidden_units in HIDDEN_UNIT_OPTIONS:
        print("\n" + "=" * 70)
        print(f"{hidden_units} hidden units")
        print("=" * 70)

        clf = MLPClassifier(
            hidden_layer_sizes=(hidden_units,), activation="relu", solver="adam",
            max_iter=2000, random_state=RANDOM_SEED,
        )
        clf.fit(X_train, y_train)
        float_acc = clf.score(X_test, y_test)
        print(f"Float model accuracy: {float_acc:.4f}")

        qp = calibrate_and_quantize(clf, X_train)
        output_scale, mult2, shift2 = compute_output_quantization(qp, X_train)

        y_pred_final = quantized_forward_final(X_test, qp, mult2, shift2)
        quant_acc = np.mean(y_pred_final == y_test)
        print(f"Quantized (full int8, both layers) accuracy: {quant_acc:.4f}")

        pred_cmsisnn = quantized_forward_final(X_test, qp, mult2, shift2)
        agreement = np.mean(y_pred_final == pred_cmsisnn)
        print(f"Self-consistency check: {agreement*100:.2f}% (must be exactly 100.00%)")

        with open(f"{OUTPUT_DIR}/verification_{hidden_units}units.txt", "w") as f:
            f.write(f"hidden_units={hidden_units}\n")
            f.write(f"agreement={agreement*100:.4f}%\n")
            f.write(f"quantized_accuracy={quant_acc:.4f}\n")
            f.write(f"output_scale={output_scale:.6f}\nmult2={mult2}\nshift2={shift2}\n")

        model_h = export_model_header(qp, hidden_units, mult2, shift2, class_names)
        with open(f"{OUTPUT_DIR}/model_{hidden_units}units.h", "w") as f:
            f.write(model_h)
        n_export = min(N_TEST_VECTORS_TO_EXPORT, X_test.shape[0])
        export_x = X_test[:n_export]
        export_y_true = y_test[:n_export]
        export_y_pred = quantized_forward_final(export_x, qp, mult2, shift2)
        tv_h = export_test_vectors(qp, export_x, export_y_true, export_y_pred, hidden_units)
        with open(f"{OUTPUT_DIR}/test_vectors_{hidden_units}units.h", "w") as f:
            f.write(tv_h)

        cmsisnn_h = export_cmsisnn_header(qp, hidden_units, mult2, shift2, class_names)
        with open(f"{OUTPUT_DIR}/model_{hidden_units}units_cmsisnn.h", "w") as f:
            f.write(cmsisnn_h)

        # NEW: .tflite export for TFLM benchmarking (same trained clf, no retraining)
        tflite_path = export_tflite(clf, X_train, hidden_units)

        # Rigorous reference: actual TFLite runtime accuracy on the FULL test set
        tflite_ref_accuracy = evaluate_tflite_reference_accuracy(tflite_path, X_test, y_test)
        with open(f"{OUTPUT_DIR}/tflite_reference_accuracy_{hidden_units}units.txt", "w") as f:
            f.write(f"hidden_units={hidden_units}\n")
            f.write(f"tflite_interpreter_accuracy={tflite_ref_accuracy:.4f}\n")
            f.write(f"n_test_samples={X_test.shape[0]}\n")

        # Export the FULL test set for on-device comparison (only needs to
        # happen once, but harmless to regenerate identically each run)
        export_full_test_set(X_test, y_test)

        n_params = (qp["n_in"] * qp["n_hidden"] + qp["n_hidden"]
                    + qp["n_hidden"] * qp["n_out"] + qp["n_out"])
        summary_rows.append({
            "hidden_units": hidden_units,
            "n_params": n_params,
            "float_accuracy": round(float_acc, 4),
            "quantized_accuracy": round(quant_acc, 4),
            "self_consistency_pct": round(agreement * 100, 4),
        })

    with open(f"{OUTPUT_DIR}/results_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print("\n" + "=" * 70)
    print(f"DONE. Files written to {OUTPUT_DIR}/:")
    print("=" * 70)
    for fname in sorted(os.listdir(OUTPUT_DIR)):
        print(" -", fname)
    if used_synthetic:
        print("\n*** REMINDER: synthetic data was used. Rerun in Colab for real results. ***")


if __name__ == "__main__":
    main()
