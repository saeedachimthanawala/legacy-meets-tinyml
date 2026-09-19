/*
 * inference_template_8051.c
 * ===========================
 * Keil C51 (8051 / AT89C51) port of the ARM inference_template.c
 * used on Day 2. SAME algorithm, SAME weights, SAME quantization --
 * the only changes are memory-space keywords required because the
 * 8051 has just 128 bytes of internal RAM and separate CODE (flash),
 * DATA (internal RAM), and XDATA (external RAM) address spaces that
 * must be assigned explicitly, unlike ARM's single unified address
 * space.
 *
 * MEMORY PLACEMENT DECISIONS (read this before building):
 *   - Weight/bias/test-vector arrays -> `code` (flash). Handled by
 *     using the *_c51.h headers (see make_c51_headers.py) instead of
 *     the plain ARM headers. This uses ZERO bytes of the 128-byte
 *     internal RAM budget.
 *   - Working buffers (acc1, hidden_q, acc2) -> `xdata` (external
 *     RAM). These do NOT fit in 128 bytes of internal RAM once you
 *     add them up for the 16-unit model (16*4 + 16*4 + 6*4 = 152
 *     bytes > 128 available). `xdata` is used UNIFORMLY across all
 *     three model sizes (even where a smaller model's buffers WOULD
 *     technically fit in internal RAM) so that memory-access speed
 *     is held constant across the sweep -- this keeps the timing
 *     differences you measure attributable to the amount of compute
 *     each model size requires, not to a change in memory placement
 *     strategy between sizes. Document this as a deliberate
 *     methodological choice in your paper.
 *   - Keil's SIMULATOR provides simulated external RAM by default,
 *     even though a real AT89C51 chip would need a physical external
 *     RAM IC wired up for this to work. Fine for a simulation-only
 *     study -- note it explicitly as a limitation/assumption in your
 *     methodology section.
 *
 * HOW TO USE
 * -----------
 * 1. Add this file plus a matching pair of *_c51.h headers (produced
 *    by make_c51_headers.py) to your Keil C51 project.
 * 2. Change the two #include lines below to select model size.
 * 3. Build, run in the simulator, check g_result == 0 (see steps),
 *    then use the Performance Analyzer exactly as you did for ARM.
 */

#include "c51_types.h"

/* ---- CHANGE THESE TWO INCLUDES TO SWITCH MODEL SIZE ---- */
#include "model_4units_c51.h"
#include "test_vectors_4units_c51.h"
/* ---------------------------------------------------------- */

static unsigned char predict(int8_t code *x_q) {
    int32_t xdata acc1[N_HIDDEN];
    int32_t xdata hidden_q[N_HIDDEN];
    int32_t xdata acc2[N_OUT];
    int32_t xdata out_q[N_OUT];
    unsigned char i, j, k, best;
    int32_t sum, v, scaled;

    /* --- Layer 1: dense + bias --- */
    for (j = 0; j < N_HIDDEN; j++) {
        sum = B1_Q[j];
        for (i = 0; i < N_IN; i++) {
            sum += (int32_t)x_q[i] * (int32_t)W1_Q[(unsigned int)i * N_HIDDEN + j];
        }
        acc1[j] = sum;
    }

    /* --- ReLU + integer requantization (multiply-then-shift) ---
     * NOTE: 32-bit-only multiply (no int64 -- Keil C51 has no 64-bit
     * integer support). Safe ONLY because MANTISSA_BITS was reduced in
     * the Python pipeline to keep the worst-case product within int32
     * range with margin. Do not raise precision here without re-checking
     * that bound. */
    for (j = 0; j < N_HIDDEN; j++) {
        v = acc1[j] > 0 ? acc1[j] : 0;
        scaled = ((int32_t)v * (int32_t)REQUANT_MULT1) >> REQUANT_SHIFT1;
        if (scaled > 127) scaled = 127;
        if (scaled < -127) scaled = -127;
        hidden_q[j] = scaled;
    }

    /* --- Layer 2: dense + bias --- */
    for (k = 0; k < N_OUT; k++) {
        sum = B2_Q[k];
        for (j = 0; j < N_HIDDEN; j++) {
            sum += hidden_q[j] * (int32_t)W2_Q[(unsigned int)j * N_OUT + k];
        }
        acc2[k] = sum;
    }

    /* --- NEW: final output-layer int8 requantization --------------------
     * Matches the ARM and CMSIS-NN paths: previously this step was skipped
     * and argmax ran on the raw int32 accumulator (acc2) directly, which
     * was an inconsistency versus how real deployments (including
     * CMSIS-NN) actually work. Same 32-bit-only safety caveat as above.
     * --------------------------------------------------------------- */
    for (k = 0; k < N_OUT; k++) {
        scaled = ((int32_t)acc2[k] * (int32_t)L2_MULT) >> L2_SHIFT;
        if (scaled > 127) scaled = 127;
        if (scaled < -127) scaled = -127;
        out_q[k] = scaled;
    }

    /* --- Argmax (same scale across all classes, so raw compare is valid) --- */
    best = 0;
    for (k = 1; k < N_OUT; k++) {
        if (out_q[k] > out_q[best]) best = k;
    }
    return best;
}

int xdata g_result; /* global, always visible in the Watch window */

int run_all_test_vectors(void) {
    int mismatches = 0;
    unsigned char t;
    for (t = 0; t < N_TEST_VECTORS; t++) {
        int8_t code *x = &TEST_INPUTS_Q[(unsigned int)t * N_IN];
        unsigned char pred = predict(x);
        if (pred != (unsigned char)EXPECTED_CLASS[t]) {
            mismatches++;
        }
    }
    return mismatches;
}

/*
 * NOTE: on 8051/C51, main() is declared `void main(void)` and must
 * never return -- there is no OS underneath it to return to. This
 * differs from the ARM version's `int main(void)`.
 */
void main(void) {
    g_result = run_all_test_vectors();
    while (1) { /* halt here so the simulator lets you inspect g_result */ }
}
