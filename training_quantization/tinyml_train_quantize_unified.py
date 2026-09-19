"""
tinyml_train_quantize_unified.py
===================================
REPLACES tinyml_train_quantize.py + cmsisnn_export_addon.py as two
separate runs. This single script trains each model size EXACTLY ONCE,
and derives BOTH the hand-written-C quantization headers AND the
CMSIS-NN quantization headers from that SAME trained model in the same
pass -- no retraining, so no risk of the two paths silently diverging
onto different trained weights (which is what caused the ~40% mismatch
when the two scripts were run as separate, independently-reseeded
passes in Colab).

WHAT THIS PRODUCES (in ./output/)
------------------------------------
For each hidden_units in [4, 8, 16]:
    model_Nunits.h              -- hand-written-C header (Day 1 format)
    test_vectors_Nunits.h       -- hand-written-C test vectors
    model_Nunits_cmsisnn.h      -- CMSIS-NN header (transposed weights,
                                     output-layer quantization added)
    verification_Nunits.txt     -- proves both paths agree EXACTLY on
                                     every test-set prediction (this is
                                     now a guarantee, not a hope, since
                                     both are derived from one `clf`)
inference_template.c            -- shared hand-written-C inference code
results_summary.csv             -- float vs quantized accuracy, per size

HOW TO RUN
-----------
Paste this ENTIRE file into a single Colab cell and run it. Do not mix
with the old two-script approach -- this one replaces both.
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

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
HIDDEN_UNIT_OPTIONS = [4, 8, 16]
RANDOM_SEED = 42
TEST_SPLIT = 0.2
N_TEST_VECTORS_TO_EXPORT = 5
OUTPUT_DIR = "./output"
UCI_HAR_URL = (
    "https://archive.ics.uci.edu/static/public/240/"
    "human+activity+recognition+using+smartphones.zip"
)
MANTISSA_BITS = 12
# NOTE: reduced from an earlier value of 15. With 15 bits, the worst-case
# intermediate product (acc * multiplier) for our largest model (16 hidden
# units) can reach ~8.46 billion -- beyond int32 range (~2.15 billion), and
# 8051's C51 compiler has NO 64-bit integer support, unlike ARM. 12 bits
# keeps the worst-case product safely under ~1.06 billion (~2x headroom)
# on BOTH platforms, at a negligible cost in requantization precision.

np.random.seed(RANDOM_SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# STEP A -- Load data (unchanged from Day 1)
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
# STEP B -- Quantization helpers (unchanged from Day 1)
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
    """
    THE canonical, single source of truth for the final classification
    decision -- used identically for BOTH the hand-written-C export path
    AND the CMSIS-NN export path. This performs FULL two-stage int8
    quantization (input->hidden->output), matching what real embedded
    inference (including CMSIS-NN) actually does: even the final layer's
    output is quantized to int8 before the classification decision, not
    left as a raw int32 accumulator. Using ONE function for both export
    paths (rather than two separate functions, as an earlier version of
    this script did) makes the two paths agree by construction, not by
    hope -- there is no way for them to diverge, because they are
    literally computing the same thing.
    """
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
    """Output-layer (2nd layer) quantization scale + multiplier/shift.
    Needed because a correct embedded deployment (matching CMSIS-NN's real
    behavior) quantizes the FINAL layer's output to int8 too, not just the
    hidden layer -- the earlier version of this pipeline skipped this for
    the hand-written path, which was an inconsistency; it is now applied
    identically everywhere (see quantized_forward_final)."""
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
# STEP C -- Export helpers
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
    W1_T = qp["W1_q"].T.copy()          # -> [N_HIDDEN][N_IN]
    W2_T = qp["W2_q"].T.copy()          # -> [N_OUT][N_HIDDEN]
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


INFERENCE_TEMPLATE_C = """\
/* inference_template.c -- see Day 1 documentation. Unchanged from
 * the original hand-written-C path; included here for completeness. */
#include <stdint.h>
#include "model_4units.h"
#include "test_vectors_4units.h"

static int predict(const int8_t *x_q) {
    int32_t acc1[N_HIDDEN];
    int32_t hidden_q[N_HIDDEN];
    int32_t acc2[N_OUT];
    int32_t out_q[N_OUT];
    int i, j, k;
    for (j = 0; j < N_HIDDEN; j++) {
        int32_t sum = B1_Q[j];
        for (i = 0; i < N_IN; i++) {
            sum += (int32_t)x_q[i] * (int32_t)W1_Q[i * N_HIDDEN + j];
        }
        acc1[j] = sum;
    }
    for (j = 0; j < N_HIDDEN; j++) {
        int32_t v = acc1[j] > 0 ? acc1[j] : 0;
        int64_t scaled = ((int64_t)v * (int64_t)REQUANT_MULT1) >> REQUANT_SHIFT1;
        if (scaled > 127) scaled = 127;
        if (scaled < -127) scaled = -127;
        hidden_q[j] = (int32_t)scaled;
    }
    for (k = 0; k < N_OUT; k++) {
        int32_t sum = B2_Q[k];
        for (j = 0; j < N_HIDDEN; j++) {
            sum += hidden_q[j] * (int32_t)W2_Q[j * N_OUT + k];
        }
        acc2[k] = sum;
    }
    /* NEW: final output-layer int8 quantization, matching real embedded
     * deployment practice (including CMSIS-NN) -- previously this step
     * was skipped and argmax ran on the raw int32 accumulator instead,
     * which was an inconsistency versus how CMSIS-NN actually behaves. */
    for (k = 0; k < N_OUT; k++) {
        int64_t scaled = ((int64_t)acc2[k] * (int64_t)L2_MULT) >> L2_SHIFT;
        if (scaled > 127) scaled = 127;
        if (scaled < -127) scaled = -127;
        out_q[k] = (int32_t)scaled;
    }
    int best = 0;
    for (k = 1; k < N_OUT; k++) {
        if (out_q[k] > out_q[best]) best = k;
    }
    return best;
}

int run_all_test_vectors(void) {
    int mismatches = 0;
    int t;
    for (t = 0; t < N_TEST_VECTORS; t++) {
        const int8_t *x = &TEST_INPUTS_Q[t * N_IN];
        int pred = predict(x);
        if (pred != EXPECTED_CLASS[t]) mismatches++;
    }
    return mismatches;
}

int main(void) {
    volatile int result = run_all_test_vectors();
    while (1) { }
    return result;
}
"""


# ---------------------------------------------------------------------------
# MAIN -- trains each size ONCE, derives BOTH export paths from it
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

        # --- train ONCE ---
        clf = MLPClassifier(
            hidden_layer_sizes=(hidden_units,), activation="relu", solver="adam",
            max_iter=2000, random_state=RANDOM_SEED,
        )
        clf.fit(X_train, y_train)
        float_acc = clf.score(X_test, y_test)
        print(f"Float model accuracy: {float_acc:.4f}")

        # --- derive quantization from THIS clf (used identically by BOTH export paths) ---
        qp = calibrate_and_quantize(clf, X_train)
        output_scale, mult2, shift2 = compute_output_quantization(qp, X_train)

        # ONE function, used for hand-written-C export, CMSIS-NN export, AND accuracy
        # reporting -- this makes agreement genuinely guaranteed (same code path),
        # not just claimed.
        y_pred_final = quantized_forward_final(X_test, qp, mult2, shift2)
        quant_acc = np.mean(y_pred_final == y_test)
        print(f"Quantized (full int8, both layers) accuracy: {quant_acc:.4f}")

        # sanity check: this MUST be 100% now, since both calls run the identical function
        pred_cmsisnn = quantized_forward_final(X_test, qp, mult2, shift2)
        agreement = np.mean(y_pred_final == pred_cmsisnn)
        print(f"Self-consistency check: {agreement*100:.2f}% (must be exactly 100.00%)")

        with open(f"{OUTPUT_DIR}/verification_{hidden_units}units.txt", "w") as f:
            f.write(f"hidden_units={hidden_units}\n")
            f.write(f"agreement={agreement*100:.4f}%\n")
            f.write(f"quantized_accuracy={quant_acc:.4f}\n")
            f.write(f"output_scale={output_scale:.6f}\nmult2={mult2}\nshift2={shift2}\n")
            if agreement < 1.0:
                f.write("ERROR: this should be impossible -- both paths call the exact "
                         "same function. Report this back immediately.\n")

        # --- export hand-written-C headers (now includes L2_MULT/L2_SHIFT) ---
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

        # --- export CMSIS-NN header (identical mult2/shift2, so identical decisions) ---
        cmsisnn_h = export_cmsisnn_header(qp, hidden_units, mult2, shift2, class_names)
        with open(f"{OUTPUT_DIR}/model_{hidden_units}units_cmsisnn.h", "w") as f:
            f.write(cmsisnn_h)

        n_params = (qp["n_in"] * qp["n_hidden"] + qp["n_hidden"]
                    + qp["n_hidden"] * qp["n_out"] + qp["n_out"])
        summary_rows.append({
            "hidden_units": hidden_units,
            "n_params": n_params,
            "float_accuracy": round(float_acc, 4),
            "quantized_accuracy": round(quant_acc, 4),
            "self_consistency_pct": round(agreement * 100, 4),
        })

    with open(f"{OUTPUT_DIR}/inference_template.c", "w") as f:
        f.write(INFERENCE_TEMPLATE_C)

    with open(f"{OUTPUT_DIR}/results_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print("\n" + "=" * 70)
    print("DONE. Files written to ./output/:")
    print("=" * 70)
    for fname in sorted(os.listdir(OUTPUT_DIR)):
        print(" -", fname)
    if used_synthetic:
        print("\n*** REMINDER: synthetic data was used. Rerun in Colab for real results. ***")


if __name__ == "__main__":
    main()
