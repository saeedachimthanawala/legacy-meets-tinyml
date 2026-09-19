#include <stdint.h>
#include "ARMCM7.h"

#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

extern "C" {
#include "tensorflow/lite/micro/cortex_m_generic/debug_log_callback.h"
}

#include "model_data.h"

extern "C" void itm_debug_log(const char *s) {
    while (*s) {
        ITM_SendChar((uint32_t)*s++);
    }
}

static void uart_puts(const char *s) {
    itm_debug_log(s);
}

static void print_float(float f) {
    char buf[32];
    int i = 0;
    if (f < 0) {
        buf[i++] = '-';
        f = -f;
    }
    int whole = (int)f;
    float frac = f - (float)whole;
    int frac_i = (int)(frac * 10000.0f + 0.5f);
    char tmp[16];
    int t = 0;
    if (whole == 0) {
        tmp[t++] = '0';
    }
    while (whole > 0) {
        tmp[t++] = '0' + (whole % 10);
        whole /= 10;
    }
    while (t > 0) {
        buf[i++] = tmp[--t];
    }
    buf[i++] = '.';
    char frac_buf[5];
    for (int k = 3; k >= 0; k--) {
        frac_buf[k] = '0' + (frac_i % 10);
        frac_i /= 10;
    }
    frac_buf[4] = '\0';
    for (int k = 0; k < 4; k++) {
        buf[i++] = frac_buf[k];
    }
    buf[i] = '\0';
    uart_puts(buf);
}

static void print_int(int v) {
    char tmp[12];
    int t = 0;
    if (v < 0) {
        uart_puts("-");
        v = -v;
    }
    if (v == 0) {
        uart_puts("0");
        return;
    }
    while (v > 0) {
        tmp[t++] = '0' + (v % 10);
        v /= 10;
    }
    char buf[12];
    int i = 0;
    while (t > 0) {
        buf[i++] = tmp[--t];
    }
    buf[i] = '\0';
    uart_puts(buf);
}

static void print_uint32(uint32_t v) {
    char tmp[12];
    int t = 0;
    if (v == 0) {
        uart_puts("0");
        return;
    }
    while (v > 0) {
        tmp[t++] = '0' + (v % 10);
        v /= 10;
    }
    char buf[12];
    int i = 0;
    while (t > 0) {
        buf[i++] = tmp[--t];
    }
    buf[i] = '\0';
    uart_puts(buf);
}

namespace {
using OpResolver = tflite::MicroMutableOpResolver<1>;

TfLiteStatus RegisterOps(OpResolver &op_resolver) {
    TF_LITE_ENSURE_STATUS(op_resolver.AddFullyConnected());
    return kTfLiteOk;
}
}  // namespace

static void dwt_init(void) {
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    __DSB();
    __ISB();
    DWT->LAR = 0xC5ACCE55;  /* Cortex-M7-specific DWT unlock */
    __DSB();
    DWT->CYCCNT = 0;
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
}

static void itm_init(void) {
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    ITM->TCR |= ITM_TCR_ITMENA_Msk;
    ITM->TER |= (1UL << 0);
}

#define N_IN 12
#define N_OUT 6
#define N_TEST_VECTORS 4

/* Representative feature vectors for TIMING purposes -- these are NOT
 * drawn from your real test set (the hand-written pipeline's exported
 * TEST_INPUTS_Q values use a different, symmetric quantization scheme
 * and can't be reused directly here). Values are plausible post-
 * StandardScaler magnitudes (roughly -3..+3). Swap in real float
 * feature rows from your test set if/when you want to also verify
 * classification correctness, not just measure cycles. */
static const float test_inputs[N_TEST_VECTORS][N_IN] = {
    {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
    {1.2f, -0.5f, 0.8f, -1.1f, 0.3f, 2.0f, -0.7f, 1.5f, -2.1f, 0.6f, -0.4f, 1.0f},
    {-1.5f, 0.9f, -0.3f, 1.8f, -2.4f, 0.1f, 1.1f, -0.9f, 0.5f, -1.3f, 2.2f, -0.6f},
    {2.5f, 2.5f, -2.5f, -2.5f, 1.0f, -1.0f, 0.0f, 0.5f, -0.5f, 1.5f, -1.5f, 0.2f},
};

volatile uint32_t g_cycles[N_TEST_VECTORS];
volatile int g_predicted_class[N_TEST_VECTORS];
volatile int g_done = 0;

int main(void) {
    itm_init();
    RegisterDebugLogCallback(itm_debug_log);

    uart_puts("Booting TFLM HAR 64-unit inference...\n");

    const tflite::Model *model = ::tflite::GetModel(g_model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        uart_puts("Model schema version mismatch!\n");
        while (1) {
        }
    }

    OpResolver op_resolver;
    if (RegisterOps(op_resolver) != kTfLiteOk) {
        uart_puts("Failed to register ops\n");
        while (1) {
        }
    }

    constexpr int kTensorArenaSize = 8192;
    static uint8_t tensor_arena[kTensorArenaSize];

    tflite::MicroInterpreter interpreter(model, op_resolver, tensor_arena,
                                          kTensorArenaSize);

    if (interpreter.AllocateTensors() != kTfLiteOk) {
        uart_puts("AllocateTensors failed\n");
        while (1) {
        }
    }

    TfLiteTensor *input = interpreter.input(0);
    TfLiteTensor *output = interpreter.output(0);

    /* Read quantization params directly from the model rather than
     * hardcoding -- these are specific to THIS .tflite file's own
     * (asymmetric) quantization scheme, chosen by TFLite's converter. */
    float input_scale = input->params.scale;
    int input_zero_point = input->params.zero_point;
    float output_scale = output->params.scale;
    int output_zero_point = output->params.zero_point;

    uart_puts("Tensors allocated OK. input_scale=");
    print_float(input_scale);
    uart_puts(" input_zp=");
    print_int(input_zero_point);
    uart_puts(" output_scale=");
    print_float(output_scale);
    uart_puts(" output_zp=");
    print_int(output_zero_point);
    uart_puts("\n");

    dwt_init();

    for (int t = 0; t < N_TEST_VECTORS; t++) {
        for (int i = 0; i < N_IN; i++) {
            int32_t q = (int32_t)(test_inputs[t][i] / input_scale + 0.5f) + input_zero_point;
            if (q > 127) q = 127;
            if (q < -128) q = -128;
            input->data.int8[i] = (int8_t)q;
        }

        uint32_t cycles_start = DWT->CYCCNT;
        TfLiteStatus invoke_status = interpreter.Invoke();
        uint32_t cycles_end = DWT->CYCCNT;

        if (invoke_status != kTfLiteOk) {
            uart_puts("Invoke failed\n");
            while (1) {
            }
        }

        int best_class = 0;
        int8_t best_val = output->data.int8[0];
        for (int k = 1; k < N_OUT; k++) {
            if (output->data.int8[k] > best_val) {
                best_val = output->data.int8[k];
                best_class = k;
            }
        }

        g_cycles[t] = cycles_end - cycles_start;
        g_predicted_class[t] = best_class;

        uart_puts("  vector=");
        print_int(t);
        uart_puts("  predicted_class=");
        print_int(best_class);
        uart_puts("  cycles=");
        print_uint32(g_cycles[t]);
        uart_puts("\n");
    }

    uart_puts("~~~DONE~~~\n");
    g_done = 1;  /* breakpoint here to inspect g_cycles/g_predicted_class */

    while (1) {
    }

    return 0;
}
