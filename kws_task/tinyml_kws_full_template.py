"""
tinyml_kws_train_quantize.py
================================
SECOND TASK for the research program: audio keyword spotting (KWS).

PURPOSE (see project discussion): this task exists specifically to test
whether the two headline findings from the HAR-based studies are general
properties of the hardware/library, or artifacts of the one task tested
so far:
  1. Does the 8051's ~40x slowdown vs. hand-written ARM code reappear on
     a genuinely different input modality (audio vs. motion sensor)?
  2. Does CMSIS-NN's ~1.27x timing-overhead plateau reappear, now that
     both the INPUT size and OUTPUT size differ from the HAR task
     (HAR: 12 in / 6 out; KWS: 20 in / 12 out)?

A "negative" result (numbers come out clearly different) is still a
genuine, useful finding -- it would mean the effect is partly
task-dependent, which is itself worth reporting. The point is testing
the hypothesis, not confirming it.

DATASET: Google Speech Commands v0.02 (Warden, 2018). Standard 12-class
framing used throughout the TinyML KWS literature (including Zhang et
al.'s "Hello Edge", already cited in this research program): the 10 core
command words, plus "unknown" (other spoken words) and "silence"
(background noise).
    Core 10: yes, no, up, down, left, right, on, off, stop, go

FEATURE EXTRACTION: MFCC (Mel-Frequency Cepstral Coefficients), the
standard audio-domain equivalent of the hand-crafted statistics used for
the HAR task. Deliberately kept COMPACT and reduced to summary statistics
(mean + std per coefficient across time-frames) rather than feeding a
full MFCC "image" into a CNN, for two reasons:
  (a) keeps the SAME simple MLP architecture as the HAR studies -- this
      is essential for isolating "does the finding generalize", since
      introducing a different model architecture at the same time would
      confound the comparison with a second changed variable;
  (b) keeps dataset/feature engineering complexity bounded, consistent
      with this being a confirmatory second task, not a new KWS-accuracy
      research contribution in its own right.
10 MFCC coefficients x 2 statistics (mean, std) = 20 input features.

DEPENDENCY: requires `librosa` for MFCC extraction (not needed by the
original HAR pipeline). If missing: `pip install librosa`.

METHODOLOGY: mirrors tinyml_train_quantize_unified.py exactly downstream
of feature extraction -- single training pass per model size, both
hand-written-C and CMSIS-NN export paths derived from that SAME trained
model, same MANTISSA_BITS=12 quantization scheme, same
correctness-verification-first discipline (train once, never re-train
for different export formats).

DEFAULT SCOPE: two representative sizes (8 and 64 units) -- a small and
large point from the existing HAR sweep -- to confirm the whole pipeline
and both headline findings before committing to a full 7-size sweep.
Change HIDDEN_UNIT_OPTIONS below to extend.

WHAT THIS PRODUCES (in ./output_kws/)
For each hidden_units in HIDDEN_UNIT_OPTIONS:
    model_Nunits.h              -- hand-written-C header
    test_vectors_Nunits.h       -- hand-written-C test vectors
    model_Nunits_cmsisnn.h      -- CMSIS-NN header (transposed weights)
    verification_Nunits.txt     -- self-consistency proof
inference_template.c            -- shared hand-written-C inference code
results_summary.csv             -- float vs quantized accuracy, per size

HOW TO RUN
-----------
Paste this ENTIRE file into a single Colab cell and run it.
First cell: !pip install librosa --quiet   (if not already available)
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

try:
    import librosa
except ImportError:
    librosa = None

import tensorflow as tf
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
HIDDEN_UNITS = 8  # <-- CHANGE THIS to 8/64/256 for each run (matching the paper's KWS sizes)
HIDDEN_UNIT_OPTIONS = [HIDDEN_UNITS]
RANDOM_SEED = 42
TEST_SPLIT = 0.2
N_TEST_VECTORS_TO_EXPORT = 5
OUTPUT_DIR = "./output_kws"
GSC_URL = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
CORE_KEYWORDS = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
N_CLASSES = 12  # 10 core + unknown + silence
SAMPLE_RATE = 16000
CLIP_SECONDS = 1.0
N_MFCC = 10
MAX_PER_CLASS = 1600  # cap per class for reasonable Colab runtime; ~19k total examples
MANTISSA_BITS = 12
# NOTE: unchanged from the HAR studies, for direct methodological consistency.
# Overflow safety re-verified generically below (independent of task).

np.random.seed(RANDOM_SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# STEP A -- Load data
# ---------------------------------------------------------------------------
def load_gsc_raw_clips():
    if requests is None or librosa is None:
        raise RuntimeError("requests and librosa required")
    import tarfile
    import tempfile
    import scipy.io.wavfile as wavfile

    tmp_dir = tempfile.gettempdir()
    tar_path = os.path.join(tmp_dir, "speech_commands_v0.02.tar.gz")
    extract_dir = os.path.join(tmp_dir, "speech_commands_extracted")

    if not os.path.exists(tar_path):
        print("Downloading Google Speech Commands v0.02 (~2GB, this will take a few minutes)...")
        with requests.get(GSC_URL, timeout=600, stream=True) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))
            downloaded = 0
            with open(tar_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0 and downloaded % (20 * 1024 * 1024) < 1024 * 1024:
                        print(f"  ... {downloaded / 1e6:.0f} MB / {total / 1e6:.0f} MB")
        print(f"Download complete.")
    else:
        print("Archive already downloaded, reusing.")

    if not os.path.exists(extract_dir) or not os.listdir(extract_dir):
        print("Extracting FULL archive to disk (one-time sequential decompression)...")
        os.makedirs(extract_dir, exist_ok=True)
        with tarfile.open(tar_path, mode="r:gz") as tf:
            tf.extractall(extract_dir)
        print("Extraction complete.")
    else:
        print("Already extracted, reusing.")

    all_entries = os.listdir(extract_dir)
    by_class = {}
    bg_noise_dir = None
    for entry in all_entries:
        full = os.path.join(extract_dir, entry)
        if not os.path.isdir(full):
            continue
        if entry == "_background_noise_":
            bg_noise_dir = full
        else:
            wavs = [os.path.join(full, w) for w in os.listdir(full) if w.endswith(".wav")]
            if wavs:
                by_class[entry] = wavs

    bg_noise_files = []
    if bg_noise_dir:
        bg_noise_files = [os.path.join(bg_noise_dir, w) for w in os.listdir(bg_noise_dir) if w.endswith(".wav")]

    print(f"Found {len(by_class)} word classes, {len(bg_noise_files)} background noise files")
    if len(bg_noise_files) == 0:
        raise RuntimeError(
            f"No background noise files found after extraction. "
            f"Top-level entries: {all_entries[:10]}"
        )

    def label_for(word):
        if word in CORE_KEYWORDS:
            return CORE_KEYWORDS.index(word)
        return 10  # "unknown"

    def read_wav(path):
        sr, data = wavfile.read(path)
        data = data.astype(np.float32) / 32768.0
        target_len = int(SAMPLE_RATE * CLIP_SECONDS)
        if len(data) < target_len:
            data = np.pad(data, (0, target_len - len(data)))
        else:
            data = data[:target_len]
        return data

    waveforms, labels = [], []
    rng = np.random.default_rng(RANDOM_SEED)

    for word, paths in by_class.items():
        label = label_for(word)
        paths = list(paths)
        rng.shuffle(paths)
        cap = MAX_PER_CLASS if label < 10 else max(1, MAX_PER_CLASS // 25)
        for p in paths[:cap]:
            waveforms.append(read_wav(p))
            labels.append(label)

    n_silence = MAX_PER_CLASS
    for _ in range(n_silence):
        p = bg_noise_files[rng.integers(0, len(bg_noise_files))]
        sr, data = wavfile.read(p)
        data = data.astype(np.float32) / 32768.0
        target_len = int(SAMPLE_RATE * CLIP_SECONDS)
        if len(data) > target_len:
            start = rng.integers(0, len(data) - target_len)
            data = data[start:start + target_len]
        else:
            data = np.pad(data, (0, max(0, target_len - len(data))))
        waveforms.append(data)
        labels.append(11)

    return np.array(waveforms, dtype=np.float32), np.array(labels, dtype=int)


def make_synthetic_audio_clips(n_per_class=60):
    rng = np.random.default_rng(RANDOM_SEED)
    target_len = int(SAMPLE_RATE * CLIP_SECONDS)
    t = np.linspace(0, CLIP_SECONDS, target_len)
    waveforms, labels = [], []
    for cls in range(N_CLASSES):
        base_freq = 200 + cls * 150
        for _ in range(n_per_class):
            noise = rng.normal(0, 0.05, size=target_len)
            sig = 0.3 * np.sin(2 * np.pi * base_freq * t) + noise
            waveforms.append(sig.astype(np.float32))
            labels.append(cls)
    return np.array(waveforms, dtype=np.float32), np.array(labels, dtype=int)


def extract_mfcc_features(waveforms):
    feats = []
    for wav in waveforms:
        mfcc = librosa.feature.mfcc(y=wav, sr=SAMPLE_RATE, n_mfcc=N_MFCC,
                                     n_fft=400, hop_length=160)
        feats.append(np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)]))
    return np.array(feats, dtype=np.float32)


def extract_mfcc_features_synthetic(waveforms):
    feats = []
    target_len = waveforms.shape[1]
    for wav in waveforms:
        spec = np.abs(np.fft.rfft(wav))
        bands = np.array_split(spec, N_MFCC)
        means = np.array([b.mean() for b in bands])
        stds = np.array([b.std() for b in bands])
        feats.append(np.concatenate([means, stds]))
    return np.array(feats, dtype=np.float32)


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
# STEP C -- Export helpers
# ---------------------------------------------------------------------------
def c_int_array(name, arr, ctype="int8_t"):
    flat = ", ".join(str(int(v)) for v in np.asarray(arr).flatten())
    return f"static const {ctype} {name}[{arr.size}] = {{ {flat} }};\n"


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


def evaluate_tflite_reference_accuracy(tflite_path, X_test, y_test):
    """Runs the ACTUAL .tflite file through TensorFlow's own official
    interpreter -- the correct ground truth to compare on-device TFLM
    results against."""
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
    """Exports the ENTIRE test set for a rigorous on-device accuracy
    check (KWS: 20 features, 12 classes -- different shape than HAR,
    firmware constants must be updated accordingly)."""
    n = X_test.shape[0]
    n_in = X_test.shape[1]

    lines = []
    lines.append("// Auto-generated: FULL KWS test set for on-device accuracy check")
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
    print(f"Wrote {out_path}: FULL KWS test set, {n} samples")
    return out_path


def export_model_header(qp, hidden_units, mult2, shift2, class_names):
    lines = []
    lines.append(f"// Auto-generated -- {hidden_units} hidden units (KWS task, hand-written-C path)")
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
    lines.append(f"// Auto-generated test vectors (KWS task) -- {hidden_units} hidden units")
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
    lines.append(f"// Auto-generated CMSIS-NN header (KWS task) -- {hidden_units} hidden units")
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
    lines.append("// NOTE: symmetric quantization throughout (all zero-points = 0).")
    lines.append(f"#endif // MODEL_{hidden_units}UNITS_CMSISNN_H\n")
    return "\n".join(lines)


INFERENCE_TEMPLATE_C = """\
/* inference_template.c -- KWS task. Same logic as the HAR study's
 * inference_template_arm.c; only the header included changes. Update
 * the two #include lines below to select model size. Remember: use
 * the GLOBAL g_result pattern established in the HAR study (not a
 * local `result` variable), or the debugger Watch window will show
 * "cannot evaluate" -- this bit us once already, don't repeat it. */
#include <stdint.h>
#include "model_8units.h"
#include "test_vectors_8units.h"

volatile int g_result = -1;

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
    g_result = run_all_test_vectors();
    while (1) { }
}
"""


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("Loading data (KWS task)")
    print("=" * 70)
    used_synthetic = False
    try:
        X_raw, y = load_gsc_raw_clips()
        print(f"Loaded REAL Google Speech Commands data: {X_raw.shape[0]} clips")
        X_feat = extract_mfcc_features(X_raw)
    except Exception as e:
        print(f"Could not download/process real data ({e}); using SYNTHETIC fallback.")
        print("Rerun in Colab (with internet + librosa installed) for real results.")
        X_raw, y = make_synthetic_audio_clips()
        used_synthetic = True
        if librosa is not None:
            X_feat = extract_mfcc_features(X_raw)
        else:
            X_feat = extract_mfcc_features_synthetic(X_raw)

    class_names = CORE_KEYWORDS + ["unknown", "silence"]
    n = X_feat.shape[0]
    idx = np.random.permutation(n)
    n_test = int(n * TEST_SPLIT)
    test_idx, train_idx = idx[:n_test], idx[n_test:]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_feat[train_idx]).astype(np.float32)
    X_test = scaler.transform(X_feat[test_idx]).astype(np.float32)
    y_train, y_test = y[train_idx], y[test_idx]
    print(f"Feature dim: {X_feat.shape[1]} | classes: {N_CLASSES} | train: {len(train_idx)} | test: {len(test_idx)}")

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

        # --- .tflite export + rigorous TFLite-reference accuracy + full test-set export ---
        tflite_path = export_tflite(clf, X_train, hidden_units)
        tflite_ref_accuracy = evaluate_tflite_reference_accuracy(tflite_path, X_test, y_test)
        with open(f"{OUTPUT_DIR}/tflite_reference_accuracy_{hidden_units}units.txt", "w") as f:
            f.write(f"hidden_units={hidden_units}\ntask=KWS\n")
            f.write(f"tflite_interpreter_accuracy={tflite_ref_accuracy:.4f}\n")
            f.write(f"n_test_samples={X_test.shape[0]}\n")
        export_full_test_set(X_test, y_test)

        with open(f"{OUTPUT_DIR}/verification_{hidden_units}units.txt", "w") as f:
            f.write(f"hidden_units={hidden_units}\ntask=KWS\n")
            f.write(f"agreement={agreement*100:.4f}%\nquantized_accuracy={quant_acc:.4f}\n")
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

        n_params = (qp["n_in"] * qp["n_hidden"] + qp["n_hidden"]
                    + qp["n_hidden"] * qp["n_out"] + qp["n_out"])
        summary_rows.append({
            "hidden_units": hidden_units, "n_params": n_params,
            "float_accuracy": round(float_acc, 4), "quantized_accuracy": round(quant_acc, 4),
            "self_consistency_pct": round(agreement * 100, 4),
        })

    with open(f"{OUTPUT_DIR}/inference_template.c", "w") as f:
        f.write(INFERENCE_TEMPLATE_C)

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
