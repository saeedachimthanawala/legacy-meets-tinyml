# ================================================================
# ALL FIGURES
# Legacy Meets TinyML
#
# Generates Figures 1–10
# 300 DPI — suitable for LaTeX / IEEE manuscript preparation
# Colab-compatible
# ================================================================

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------
# OUTPUT DIRECTORY
# ------------------------------------------------

OUTDIR = Path.cwd() / "generated_figures"
OUTDIR.mkdir(parents=True, exist_ok=True)

DPI = 300

print("Output directory:")
print(OUTDIR)


# ================================================================
# AUTHORITATIVE DATA
# ================================================================

# ------------------------------------------------
# HAR — Direct ARM / 8051 benchmark
# ------------------------------------------------

har_sizes = np.array([4, 8, 16, 32, 64, 128, 256])

arm_hw_time = np.array([
    0.301,
    0.561,
    1.079,
    2.242,
    4.389,
    8.687,
    17.278
])

cmsis_time = np.array([
    0.574,
    0.914,
    1.590,
    2.942,
    5.644,
    11.059,
    21.878
])

# 8051 only available for 4–16
sizes_8051 = np.array([4, 8, 16])

arm_hw_8051 = np.array([
    0.301,
    0.561,
    1.079
])

cmsis_8051 = np.array([
    0.574,
    0.914,
    1.590
])

mcs51_time = np.array([
    11.61,
    22.61,
    44.39
])


# ------------------------------------------------
# Direct CMSIS-NN timing ratios
# ------------------------------------------------

har_time_ratio = cmsis_time / arm_hw_time

# Expected:
# 1.91, 1.63, 1.47, 1.31, 1.29, 1.27, 1.27


# ------------------------------------------------
# KWS — direct ARM / CMSIS-NN benchmark
# ------------------------------------------------

kws_sizes = np.array([8, 64, 256])

kws_arm_time = np.array([
    1.053,
    7.192,
    28.244
])

kws_cmsis_time = np.array([
    1.449,
    8.791,
    33.954
])

kws_time_ratio = kws_cmsis_time / kws_arm_time

# Expected:
# 1.38, 1.22, 1.20


# ------------------------------------------------
# HAR FLASH
# ------------------------------------------------

arm_hw_flash = np.array([
    1216,
    1304,
    1480,
    2080,
    2784,
    4200,
    7024
])

cmsis_flash = np.array([
    6320,
    6408,
    6584,
    6936,
    7640,
    9048,
    11872
])

flash_ratio_har = cmsis_flash / arm_hw_flash


# ------------------------------------------------
# HAR APPLICATION / MODEL FOOTPRINT
# Used in Figure 3
# ------------------------------------------------

arm_hw_application_flash = np.array([
    740,
    828,
    1004
])

cmsis_application_flash = np.array([
    6320,
    6408,
    6584
])

mcs51_flash = np.array([
    1313,
    1401,
    1577
])


# ------------------------------------------------
# HAR RAM
# Not plotted in the current figures
# Included here for consistency / future use
# ------------------------------------------------

arm_hw_ram = np.array([
    100,
    100,
    100,
    252,
    380,
    636,
    1148
])

cmsis_ram = np.array([
    560,
    576,
    612,
    784,
    912,
    1168,
    1680
])

ram_ratio_har = cmsis_ram / arm_hw_ram


# ------------------------------------------------
# KWS FLASH
# ------------------------------------------------

kws_arm_flash = np.array([
    1728,
    3744,
    10672
])

kws_cmsis_flash = np.array([
    6576,
    8592,
    15512
])

kws_flash_ratio = kws_cmsis_flash / kws_arm_flash


# ================================================================
# TFLM DATA
# ================================================================

# ------------------------------------------------
# TFLM HAR latency
# Units: CPU cycles
# ------------------------------------------------

tflm_har_sizes = np.array([32, 64, 128, 256])

tflm_reference_cycles = np.array([
    983,
    1759,
    3311,
    6415
])

tflm_cmsis_cycles = np.array([
    525,
    877,
    1581,
    2989
])

tflm_har_speedup = (
    tflm_reference_cycles /
    tflm_cmsis_cycles
)


# ------------------------------------------------
# TFLM KWS latency
# Units: CPU cycles
# ------------------------------------------------

tflm_kws_sizes = np.array([8, 64, 256])

tflm_kws_reference_cycles = np.array([
    549,
    2446,
    8950
])

tflm_kws_cmsis_cycles = np.array([
    330,
    1122,
    3845
])

tflm_kws_speedup = (
    tflm_kws_reference_cycles /
    tflm_kws_cmsis_cycles
)


# ------------------------------------------------
# TFLM HAR flash footprint
# Corrected: test-data overhead removed
# ------------------------------------------------

tflm_ref_flash = np.array([
    54904,
    56376,
    59320,
    65208
])

tflm_cmsis_flash = np.array([
    71756,
    73228,
    76172,
    82060
])

tflm_flash_ratio = (
    tflm_cmsis_flash /
    tflm_ref_flash
)


# ================================================================
# HELPER FUNCTIONS
# ================================================================

def save_fig(filename):
    path = OUTDIR / filename
    plt.savefig(
        path,
        dpi=DPI,
        bbox_inches="tight"
    )
    plt.close()
    print(f"Saved: {path}")


# ================================================================
# FIGURE 1
# ================================================================

plt.figure(figsize=(8, 5.5))

plt.plot(
    sizes_8051,
    mcs51_time,
    marker="s",
    linewidth=2,
    markersize=7,
    label="8051"
)

plt.plot(
    har_sizes,
    arm_hw_time,
    marker="o",
    linewidth=2,
    markersize=7,
    label="ARM Hand-Written"
)

plt.plot(
    har_sizes,
    cmsis_time,
    marker="^",
    linewidth=2,
    markersize=7,
    label="ARM CMSIS-NN"
)

# Data labels
for x, y in zip(sizes_8051, mcs51_time):
    plt.annotate(
        f"{y:.3f}",
        (x, y),
        xytext=(0, 8),
        textcoords="offset points",
        ha="center",
        fontsize=8
    )

for x, y in zip(har_sizes, arm_hw_time):
    plt.annotate(
        f"{y:.3f}",
        (x, y),
        xytext=(0, 8),
        textcoords="offset points",
        ha="center",
        fontsize=8
    )

for x, y in zip(har_sizes, cmsis_time):
    plt.annotate(
        f"{y:.3f}",
        (x, y),
        xytext=(0, -15),
        textcoords="offset points",
        ha="center",
        fontsize=8
    )

plt.yscale("log")

plt.xlabel("Hidden-layer size (units)")
plt.ylabel("Inference latency (ms)")

plt.title(
    "Inference Latency: 8051 vs. ARM Hand-Written vs. ARM CMSIS-NN"
)

plt.xticks(har_sizes)
plt.grid(True, which="both", alpha=0.25)
plt.legend(loc="lower right")

save_fig("figure1_inference_time_scaling.png")


# ================================================================
# FIGURE 2
# CMSIS-NN TIMING OVERHEAD PLATEAU
# ================================================================

plt.figure(figsize=(8, 5.2))

bars = plt.bar(
    har_sizes,
    har_time_ratio,
    width=12
)

plt.axhline(
    1.0,
    linestyle="--",
    linewidth=1.5,
    label="Parity (1.0×)"
)

# Plateau region
plt.axvspan(
    64,
    256,
    alpha=0.12
)

for bar, value in zip(bars, har_time_ratio):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.05,
        f"{value:.2f}×",
        ha="center",
        va="bottom",
        fontsize=8
    )

plt.xlabel("Hidden-layer size (units)")
plt.ylabel("CMSIS-NN / Hand-Written latency")

plt.title(
    "CMSIS-NN Overhead Plateaus — No Crossover Within Tested Range"
)

plt.xticks(har_sizes)
plt.ylim(0, 2.1)
plt.grid(axis="y", alpha=0.25)
plt.legend()

save_fig("figureE2_ratio_plateau.png")


# ================================================================
# FIGURE 3
# ORIGINAL 4–16 FLASH COMPARISON
# ================================================================

small_sizes = np.array([4, 8, 16])

x = np.arange(len(small_sizes))
width = 0.25

plt.figure(figsize=(8, 5.2))

plt.bar(
    x - width,
    arm_hw_application_flash,
    width,
    label="ARM Hand-Written"
)

plt.bar(
    x,
    cmsis_application_flash,
    width,
    label="ARM CMSIS-NN"
)

plt.bar(
    x + width,
    mcs51_flash,
    width,
    label="8051"
)

# Data labels
for i, v in enumerate(arm_hw_application_flash):
    plt.text(
        i - width,
        v + 100,
        f"{v}",
        ha="center",
        fontsize=8
    )

for i, v in enumerate(cmsis_application_flash):
    plt.text(
        i,
        v + 100,
        f"{v}",
        ha="center",
        fontsize=8
    )

for i, v in enumerate(mcs51_flash):
    plt.text(
        i + width,
        v + 100,
        f"{v}",
        ha="center",
        fontsize=8
    )

plt.xlabel("Hidden-layer size (units)")
plt.ylabel("Flash footprint (bytes)")

plt.title(
    "Flash Memory Footprint vs. Model Size"
)

plt.xticks(x, small_sizes)
plt.grid(axis="y", alpha=0.25)
plt.legend()

save_fig("figure3_flash_footprint_scaling.png")


# ================================================================
# FIGURE 4
# FULL ARM FLASH RANGE
# ================================================================

plt.figure(figsize=(8, 5.2))

plt.plot(
    har_sizes,
    arm_hw_flash,
    marker="o",
    linewidth=2,
    markersize=7,
    label="ARM Hand-Written"
)

plt.plot(
    har_sizes,
    cmsis_flash,
    marker="^",
    linewidth=2,
    markersize=7,
    label="ARM CMSIS-NN"
)

plt.xlabel("Hidden-layer size (units)")
plt.ylabel("Full-program RO / Flash (bytes)")

plt.title(
    "Flash Footprint vs. Model Size (4–256 units)"
)

plt.xticks(har_sizes)
plt.grid(True, alpha=0.25)
plt.legend()

save_fig("figureE3_flash_full_range.png")


# ================================================================
# FIGURE 5
# TIMING RATIO — HAR VS KWS
# ================================================================

plt.figure(figsize=(8, 5.2))

plt.plot(
    har_sizes,
    har_time_ratio,
    marker="o",
    linewidth=2,
    markersize=6,
    label="HAR"
)

plt.plot(
    kws_sizes,
    kws_time_ratio,
    marker="s",
    linewidth=2,
    markersize=7,
    label="KWS"
)

plt.axhline(
    1.0,
    linestyle="--",
    linewidth=1.5,
    label="Parity (1.0×)"
)

plt.xlabel("Hidden-layer size (units)")
plt.ylabel("CMSIS-NN / Hand-Written latency")

plt.title(
    "The Plateau Shape Generalizes Across Tasks\n"
    "(magnitude differs, shape does not)"
)

plt.xticks(har_sizes)
plt.grid(True, alpha=0.25)
plt.legend()

save_fig("figureG1_timing_ratio_both_tasks.png")


# ================================================================
# FIGURE 6
# MATCHED-SIZE TIMING AND FLASH OVERHEAD
# ================================================================

matched_sizes = np.array([8, 64, 256])

# HAR matched values
har_timing_matched = np.array([
    1.63,
    1.29,
    1.27
])

kws_timing_matched = np.array([
    1.38,
    1.22,
    1.20
])

har_flash_matched = np.array([
    4.91,
    2.74,
    1.69
])

kws_flash_matched = np.array([
    3.81,
    2.29,
    1.45
])


fig, axes = plt.subplots(
    1,
    2,
    figsize=(11, 4.8)
)

# ------------------------------------------------
# Timing
# ------------------------------------------------

x = np.arange(len(matched_sizes))
width = 0.18

axes[0].bar(
    x - width / 2,
    har_timing_matched,
    width,
    label="HAR"
)

axes[0].bar(
    x + width / 2,
    kws_timing_matched,
    width,
    label="KWS"
)

axes[0].set_xlabel("Hidden-layer size (units)")
axes[0].set_ylabel("CMSIS-NN / Hand-Written latency")
axes[0].set_title("Timing Overhead Ratio")
axes[0].set_xticks(x)
axes[0].set_xticklabels(matched_sizes)
axes[0].grid(axis="y", alpha=0.25)
axes[0].legend()

# ------------------------------------------------
# Flash
# ------------------------------------------------

axes[1].bar(
    x - width / 2,
    har_flash_matched,
    width,
    label="HAR"
)

axes[1].bar(
    x + width / 2,
    kws_flash_matched,
    width,
    label="KWS"
)

axes[1].set_xlabel("Hidden-layer size (units)")
axes[1].set_ylabel("CMSIS-NN / Hand-Written flash")
axes[1].set_title("Flash Overhead Ratio")
axes[1].set_xticks(x)
axes[1].set_xticklabels(matched_sizes)
axes[1].grid(axis="y", alpha=0.25)
axes[1].legend()

fig.suptitle(
    "Matched-Size Comparison Across HAR and KWS",
    fontsize=13
)

fig.tight_layout()

fig.savefig(
    OUTDIR / "figureG3_matched_size_bars.png",
    dpi=DPI,
    bbox_inches="tight"
)

plt.close(fig)

print(
    f"Saved: {OUTDIR / 'figureG3_matched_size_bars.png'}"
)


# ================================================================
# FIGURE 7
# TFLM LATENCY SCALING — HAR
# ================================================================

plt.figure(figsize=(8, 5.2))

plt.plot(
    tflm_har_sizes,
    tflm_reference_cycles,
    marker="o",
    linewidth=2,
    markersize=7,
    label="TFLM Reference"
)

plt.plot(
    tflm_har_sizes,
    tflm_cmsis_cycles,
    marker="^",
    linewidth=2,
    markersize=7,
    label="TFLM + CMSIS-NN"
)

plt.yscale("log")

plt.xlabel("Hidden-layer size (units)")
plt.ylabel("CPU cycles")

plt.title(
    "TFLM Latency Scaling — HAR"
)

plt.xticks(tflm_har_sizes)
plt.grid(True, which="both", alpha=0.25)
plt.legend()

save_fig("figureT1_tflm_latency_scaling.png")


# ================================================================
# FIGURE 8
# TFLM SPEEDUP — HAR
# ================================================================

plt.figure(figsize=(8, 5.2))

plt.plot(
    tflm_har_sizes,
    tflm_har_speedup,
    marker="o",
    linewidth=2,
    markersize=7
)

for x, y in zip(
    tflm_har_sizes,
    tflm_har_speedup
):
    plt.annotate(
        f"{y:.2f}×",
        (x, y),
        xytext=(0, 8),
        textcoords="offset points",
        ha="center",
        fontsize=8
    )

plt.xlabel("Hidden-layer size (units)")
plt.ylabel("TFLM Reference / TFLM + CMSIS-NN")

plt.title(
    "TFLM CMSIS-NN-Backend Speedup Over Reference — HAR"
)

plt.xticks(tflm_har_sizes)
plt.grid(True, alpha=0.25)

save_fig("figureT2_tflm_ratio_growth.png")


# ================================================================
# FIGURE 9
# TFLM SPEEDUP — BOTH TASKS
# ================================================================

plt.figure(figsize=(8, 5.2))

plt.plot(
    tflm_har_sizes,
    tflm_har_speedup,
    marker="o",
    linewidth=2,
    markersize=7,
    label="HAR"
)

plt.plot(
    tflm_kws_sizes,
    tflm_kws_speedup,
    marker="s",
    linewidth=2,
    markersize=7,
    label="KWS"
)

plt.xlabel("Hidden-layer size (units)")
plt.ylabel("TFLM Reference / TFLM + CMSIS-NN")

plt.title(
    "TFLM CMSIS-NN-Backend Speedup, Both Tasks"
)

plt.xticks(
    np.array([8, 32, 64, 128, 256])
)

plt.grid(True, alpha=0.25)
plt.legend()

save_fig("figureT3_tflm_ratio_both_tasks.png")


# ================================================================
# FIGURE 10
# DIRECT CMSIS-NN OVERHEAD VS TFLM BACKEND SPEEDUP
# ================================================================

fig, ax = plt.subplots(
    figsize=(8, 5.5)
)

# Direct CMSIS-NN ratio
ax.plot(
    har_sizes,
    har_time_ratio,
    marker="o",
    linewidth=2,
    markersize=6,
    label="Direct CMSIS-NN / Hand-Written"
)

# TFLM backend speedup
ax.plot(
    tflm_har_sizes,
    tflm_har_speedup,
    marker="^",
    linewidth=2,
    markersize=6,
    label="TFLM Reference / CMSIS-NN Backend"
)

ax.axhline(
    1.0,
    linestyle="--",
    linewidth=1.2,
    label="Parity (1.0×)"
)

ax.set_xlabel("Hidden-layer size (units)")
ax.set_ylabel("Ratio")

ax.set_title(
    "Contrasting Direct Kernel Overhead with "
    "TFLM Backend Speedup"
)

ax.set_xticks(
    np.array([4, 8, 16, 32, 64, 128, 256])
)

ax.grid(True, alpha=0.25)
ax.legend()

fig.tight_layout()

fig.savefig(
    OUTDIR / "figureT4_contrast_plateau_vs_growth.png",
    dpi=DPI,
    bbox_inches="tight"
)

plt.close(fig)

print(
    f"Saved: {OUTDIR / 'figureT4_contrast_plateau_vs_growth.png'}"
)


# ================================================================
# FINAL CHECK
# ================================================================

print("\n" + "=" * 65)
print("FIGURE GENERATION COMPLETE")
print("=" * 65)

figure_files = sorted(
    OUTDIR.glob("*.png")
)

for f in figure_files:
    print(f.name)

print(f"\nTotal figures generated: {len(figure_files)}")
print(f"Location: {OUTDIR}")
