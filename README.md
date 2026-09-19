# Legacy Meets TinyML

Code and data accompanying **"Legacy Meets TinyML: An Extended, Cross-Task Study..."**,
an extended journal version of a conference paper originally presented at ISED-2026 (NIT Warangal).

This repository benchmarks quantized (int8) neural network inference across four
implementations — hand-written 8051 (AT89C51), hand-written ARM Cortex-M, ARM +
CMSIS-NN, and TensorFlow Lite Micro (TFLM) — on two tasks: Human Activity
Recognition (HAR, from the UCI HAR dataset) and Keyword Spotting (KWS, from
Google Speech Commands v0.02).

## Repository structure

### `training_quantization/`
The core pipeline: trains a single-hidden-layer MLP, quantizes it to integer-only
int8 (symmetric, multiply-then-shift requantization), and exports C headers for
both the hand-written and CMSIS-NN paths from one trained model (so the two paths
can never silently diverge onto different weights). Covers the original three
model sizes (4/8/16 hidden units).

- `tinyml_train_quantize_unified.py` — the training/quantization/export pipeline
- `inference_template_8051.c`, `c51_types.h` — the 8051/Keil C51 port
- `make_c51_headers.py` — converts ARM-format headers into 8051-compatible (flash-resident) ones

### `extended_sweep/`
Extends the original 4/8/16-unit HAR sweep to seven sizes (4/8/16/32/64/128/256),
to locate whether/where CMSIS-NN's overhead crosses over to parity with hand-written
code as model size grows. 8051 is intentionally excluded from this extension (see
script docstring for the architectural reasoning).

- `tinyml_full_accuracy_template.py` — set `HIDDEN_UNITS` at the top and rerun once
  per size (32/64/128/256) to reproduce. Exports hand-written-C headers, CMSIS-NN
  headers, a `.tflite` file for TFLM, and a full-test-set accuracy-verification header.

### `kws_task/`
A second task (audio keyword spotting, 12-class: 10 keywords + unknown + silence),
run with the identical MLP architecture and quantization scheme as HAR, to test
whether the HAR-based findings are general properties of the platforms/library or
artifacts of the one task tested first.

- `tinyml_kws_full_template.py` — set `HIDDEN_UNITS` at the top and rerun once per
  size (8/64/256) to reproduce.

### `tflm_benchmark/`
On-device firmware for benchmarking the TensorFlow Lite Micro implementation
(reference kernels vs. CMSIS-NN backend) on Keil MDK + Arm Virtual Hardware
(Cortex-M7 FVP).

- `convert_model.py` — converts a `.tflite` file into a C array for firmware linking
- `keil_har32_main.cpp` / `keil_har64_main.cpp` / `keil_har128_main.cpp` /
  `keil_har256_main.cpp` — per-size HAR timing harnesses (DWT cycle-counter based)
- `keil_har_accuracy_main.cpp`, `keil_kws_accuracy_main.cpp` — accuracy-verification
  harnesses (compare on-device predictions against known true labels)

### `toolchain_setup/`
Bring-up and infrastructure code used to validate the QEMU/Keil + TFLM toolchain
before real benchmarking began. Not part of the reported results.

- `main.c`, `dwt_diag_main.c` — QEMU boot/cycle-counter smoke tests
- `tflm_main.cpp`, `keil_tflm_main (1).cpp` — TFLM "hello_world" validation (QEMU and Keil)
- `startup.c`, `heap_stub.c`, `syscalls.c` — linker/runtime infrastructure required
  to build and run any of the above firmware

### `figures/`
- `generate_all_figures.py` — generates all reported figures (latency scaling, the
  CMSIS-NN overhead plateau, flash footprint scaling, the HAR/KWS generalization
  comparison, and the TFLM results) from hard-coded, manuscript-matching values, at
  300 DPI. Run standalone; no other script needs to run first.

### `data/`
- `Complete_Experimental_Dataset.xlsx` — the master workbook of all raw measurements
  and derived ratios reported in the paper, with a README sheet and a
  Notes/Provenance sheet documenting known caveats (e.g. the TFLM Flash footprint
  correction for test-data overhead, in `HAR_Flash_Corrected` vs. `TFLM_Flash_Original`).

## Reproducing the results

1. Run `training_quantization/tinyml_train_quantize_unified.py` in Google Colab
   (needs internet access to download the UCI HAR dataset) to reproduce the
   original 4/8/16-unit results.
2. Run `extended_sweep/tinyml_full_accuracy_template.py` four times (changing
   `HIDDEN_UNITS` to 32, 64, 128, 256 each time) to reproduce the extended sweep.
3. Run `kws_task/tinyml_kws_full_template.py` three times (changing `HIDDEN_UNITS`
   to 8, 64, 256 each time; needs `librosa` and internet access to download Google
   Speech Commands v0.02) to reproduce the KWS results.
4. Build and flash the corresponding firmware in `tflm_benchmark/` (via Keil MDK +
   Arm Virtual Hardware Cortex-M7 FVP) for on-device timing and accuracy figures.
5. Run `figures/generate_all_figures.py` to regenerate all reported figures.

All quantization uses `MANTISSA_BITS = 12`, chosen to keep the worst-case
requantization product safely within `int32` range on 8051 (which has no 64-bit
integer support) across all tested model sizes — see the comment in
`tinyml_train_quantize_unified.py` for the derivation.

## Data and code availability

This repository is the code and data referenced in the manuscript's Code and Data
Availability Statement.

## License

Released under the MIT License (see `LICENSE`).

## Citation

If you use this code or data, please cite:

```
[citation to be added upon publication]
```
