#include <stdint.h>

#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

extern "C" {
#include "tensorflow/lite/micro/cortex_m_generic/debug_log_callback.h"
}

#include "model_data.h"

/* Same CMSDK UART0 register interface as main.c from the boot smoke test. */
#define UART0_BASE  0x40004000UL
#define UART0_DATA  (*(volatile uint32_t *)(UART0_BASE + 0x00))
#define UART0_STATE (*(volatile uint32_t *)(UART0_BASE + 0x04))
#define UART0_CTRL  (*(volatile uint32_t *)(UART0_BASE + 0x08))
#define UART_STATE_TXFULL (1u << 0)
#define UART_CTRL_TX_EN   (1u << 0)

static void uart_putc(char c) {
    while (UART0_STATE & UART_STATE_TXFULL) {
    }
    UART0_DATA = (uint32_t)c;
}

/* This is the callback TFLM's DebugLog()/MicroPrintf() will invoke --
 * registered below, before we touch the interpreter. */
extern "C" void uart_debug_log(const char *s) {
    while (*s) {
        if (*s == '\n') {
            uart_putc('\r');
        }
        uart_putc(*s++);
    }
}

static void uart_puts(const char *s) {
    uart_debug_log(s);
}

/* Minimal float-to-string for printing results without pulling in a full
 * printf float implementation (newlib-nano's snprintf often doesn't
 * support %f by default). Good enough for eyeballing results here. */
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

namespace {
using HelloWorldOpResolver = tflite::MicroMutableOpResolver<1>;

TfLiteStatus RegisterOps(HelloWorldOpResolver &op_resolver) {
    TF_LITE_ENSURE_STATUS(op_resolver.AddFullyConnected());
    return kTfLiteOk;
}
}  // namespace

int main(void) {
    UART0_CTRL = UART_CTRL_TX_EN;

    RegisterDebugLogCallback(uart_debug_log);

    uart_puts("Booting TFLM hello_world inference...\n");

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

    float test_inputs[4] = {0.0f, 1.0f, 3.0f, 5.0f};

    for (int i = 0; i < 4; i++) {
        interpreter.input(0)->data.f[0] = test_inputs[i];
        if (interpreter.Invoke() != kTfLiteOk) {
            uart_puts("Invoke failed\n");
            while (1) {
            }
        }
        float y = interpreter.output(0)->data.f[0];

        uart_puts("  input=");
        print_float(test_inputs[i]);
        uart_puts("  predicted_sin=");
        print_float(y);
        uart_puts("\n");
    }

    uart_puts("~~~DONE~~~\n");

    while (1) {
    }

    return 0;
}
