#include <stdint.h>
#include "ARMCM7.h"

#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

extern "C" {
#include "tensorflow/lite/micro/cortex_m_generic/debug_log_callback.h"
}

#include "model_data.h"   /* per-size, swapped between runs */
#include "test_data.h"    /* shared across all sizes -- real samples + true labels */

extern "C" void itm_debug_log(const char *s) {
    while (*s) {
        ITM_SendChar((uint32_t)*s++);
    }
}

static void uart_puts(const char *s) {
    itm_debug_log(s);
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
    DWT->LAR = 0xC5ACCE55;
    __DSB();
    DWT->CYCCNT = 0;
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
}

static void itm_init(void) {
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    ITM->TCR |= ITM_TCR_ITMENA_Msk;
    ITM->TER |= (1UL << 0);
}

/* Sized generously -- adjust upward if a larger model size needs more.
 * 256 units needed ~32KB in earlier testing; this covers that with
 * headroom for the (unchanged) op/allocator overhead. */
#define TENSOR_ARENA_SIZE 32768

volatile uint32_t g_cycles[N_TEST_SAMPLES];
volatile int g_predicted[N_TEST_SAMPLES];
volatile int g_correct_count = 0;
volatile int g_done = 0;

int main(void) {
    itm_init();
    RegisterDebugLogCallback(itm_debug_log);

    uart_puts("Booting TFLM accuracy+timing harness...\n");
    uart_puts("N_TEST_SAMPLES=");
    print_int(N_TEST_SAMPLES);
    uart_puts("\n");

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

    static uint8_t tensor_arena[TENSOR_ARENA_SIZE];
    tflite::MicroInterpreter interpreter(model, op_resolver, tensor_arena,
                                          TENSOR_ARENA_SIZE);

    if (interpreter.AllocateTensors() != kTfLiteOk) {
        uart_puts("AllocateTensors failed\n");
        while (1) {
        }
    }

    TfLiteTensor *input = interpreter.input(0);
    TfLiteTensor *output = interpreter.output(0);
    int n_out = output->dims->data[output->dims->size - 1];

    float input_scale = input->params.scale;
    int input_zero_point = input->params.zero_point;

    uart_puts("Tensors allocated OK. Running samples:\n");

    dwt_init();

    int correct = 0;

    for (int s = 0; s < N_TEST_SAMPLES; s++) {
        for (int i = 0; i < N_FEATURES; i++) {
            float scaled = TEST_FEATURES[s][i] / input_scale;
            int32_t q = (int32_t)(scaled >= 0.0f ? scaled + 0.5f : scaled - 0.5f) + input_zero_point;
            if (q > 127) q = 127;
            if (q < -128) q = -128;
            input->data.int8[i] = (int8_t)q;
        }

        uint32_t cycles_start = DWT->CYCCNT;
        TfLiteStatus invoke_status = interpreter.Invoke();
        uint32_t cycles_end = DWT->CYCCNT;

        if (invoke_status != kTfLiteOk) {
            uart_puts("Invoke failed at sample ");
            print_int(s);
            uart_puts("\n");
            while (1) {
            }
        }

        int best_class = 0;
        int8_t best_val = output->data.int8[0];
        for (int k = 1; k < n_out; k++) {
            if (output->data.int8[k] > best_val) {
                best_val = output->data.int8[k];
                best_class = k;
            }
        }

        g_cycles[s] = cycles_end - cycles_start;
        g_predicted[s] = best_class;

        if (best_class == TRUE_LABELS[s]) {
            correct++;
        }

        uart_puts("  sample=");
        print_int(s);
        uart_puts(" predicted=");
        print_int(best_class);
        uart_puts(" true=");
        print_int(TRUE_LABELS[s]);
        uart_puts(" cycles=");
        print_uint32(g_cycles[s]);
        uart_puts(best_class == TRUE_LABELS[s] ? "  OK\n" : "  MISMATCH\n");
    }

    g_correct_count = correct;

    uart_puts("~~~DONE~~~ accuracy=");
    print_int(correct);
    uart_puts("/");
    print_int(N_TEST_SAMPLES);
    uart_puts("\n");

    g_done = 1;  /* breakpoint here: inspect g_correct_count, g_cycles[], g_predicted[] */

    while (1) {
    }

    return 0;
}
