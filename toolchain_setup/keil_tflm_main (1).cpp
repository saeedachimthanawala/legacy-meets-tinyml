#include <stdint.h>
#include "ARMCM7.h"  /* Provided by the ARM::Cortex_DFP pack for your device --
                       * gives proper DWT/CoreDebug struct definitions. */

#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

extern "C" {
#include "tensorflow/lite/micro/cortex_m_generic/debug_log_callback.h"
}

#include "model_data.h"

/* Output via ITM (Instrumentation Trace Macrocell) -- this is what Keil's
 * Debug (printf) Viewer reads from. Requires ITM enabled in Debug >
 * Settings > Trace, with Stimulus Port 0 checked (steps given separately). */
extern "C" void itm_debug_log(const char *s) {
    while (*s) {
        ITM_SendChar((uint32_t)*s++);
    }
}

static void uart_puts(const char *s) {  /* kept the same name as the QEMU
                                          * version so the rest of the logic
                                          * below is unchanged */
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
using HelloWorldOpResolver = tflite::MicroMutableOpResolver<1>;

TfLiteStatus RegisterOps(HelloWorldOpResolver &op_resolver) {
    TF_LITE_ENSURE_STATUS(op_resolver.AddFullyConnected());
    return kTfLiteOk;
}
}  // namespace

static void dwt_init(void) {
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    __DSB();
    __ISB();
    DWT->LAR = 0xC5ACCE55;  /* Cortex-M7-specific: DWT is locked by default,
                              * unlike M3/M4. This unlock is required before
                              * DWT->CTRL writes take effect. */
    __DSB();
    DWT->CYCCNT = 0;
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
}

static void itm_init(void) {
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;  /* enable trace subsystem */
    ITM->TCR |= ITM_TCR_ITMENA_Msk;                   /* enable ITM itself */
    ITM->TER |= (1UL << 0);                            /* enable stimulus port 0 */
}

volatile float g_results[4];
volatile uint32_t g_cycles[4];
volatile int g_done = 0;
volatile void *g_input_tensor_ptr = 0;
volatile void *g_output_tensor_ptr = 0;
volatile void *g_input_data_ptr = 0;
volatile void *g_output_data_ptr = 0;

int main(void) {
    itm_init();  /* harmless to leave in, but no longer load-bearing */

    RegisterDebugLogCallback(itm_debug_log);

    uart_puts("Booting TFLM hello_world inference (Keil/armclang)...\n");

    const tflite::Model *model = ::tflite::GetModel(g_model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        uart_puts("Model schema version mismatch!\n");
        while (1) {
        }
    }

    HelloWorldOpResolver op_resolver;
    if (RegisterOps(op_resolver) != kTfLiteOk) {
        uart_puts("Failed to register ops\n");
        while (1) {
        }
    }

    constexpr int kTensorArenaSize = 3000;
    static uint8_t tensor_arena[kTensorArenaSize];

    tflite::MicroInterpreter interpreter(model, op_resolver, tensor_arena,
                                          kTensorArenaSize);

    if (interpreter.AllocateTensors() != kTfLiteOk) {
        uart_puts("AllocateTensors failed\n");
        while (1) {
        }
    }

    uart_puts("Tensors allocated OK. Running inference:\n");

    g_input_tensor_ptr = (void *)interpreter.input(0);
    g_output_tensor_ptr = (void *)interpreter.output(0);
    g_input_data_ptr = (void *)interpreter.input(0)->data.f;
    g_output_data_ptr = (void *)interpreter.output(0)->data.f;

    dwt_init();

    float test_inputs[4] = {0.0f, 1.0f, 3.0f, 5.0f};

    for (int i = 0; i < 4; i++) {
        interpreter.input(0)->data.f[0] = test_inputs[i];

        uint32_t cycles_start = DWT->CYCCNT;
        TfLiteStatus invoke_status = interpreter.Invoke();
        uint32_t cycles_end = DWT->CYCCNT;

        if (invoke_status != kTfLiteOk) {
            uart_puts("Invoke failed\n");
            while (1) {
            }
        }
        float y = interpreter.output(0)->data.f[0];

        g_results[i] = y;
        g_cycles[i] = cycles_end - cycles_start;

        uart_puts("  input=");
        print_float(test_inputs[i]);
        uart_puts("  predicted_sin=");
        print_float(y);
        uart_puts("  cycles=");
        print_uint32(cycles_end - cycles_start);
        uart_puts("\n");
    }

    uart_puts("~~~DONE~~~\n");

    g_done = 1;  /* set a breakpoint on this line to inspect g_results/g_cycles */

    while (1) {
    }

    return 0;
}
