#include <stdint.h>

/* CMSDK APB UART0, as implemented on the mps2-an500 (Cortex-M7) QEMU
 * machine -- same peripheral layout as AN385/AN386, per Arm's CMSDK.
 * Reference: Arm Application Note AN385 (DAI 0385), section 3 (APB
 * memory map) and QEMU's hw/char/cmsdk-apb-uart.c. */
#define UART0_BASE  0x40004000UL

#define UART0_DATA  (*(volatile uint32_t *)(UART0_BASE + 0x00))
#define UART0_STATE (*(volatile uint32_t *)(UART0_BASE + 0x04))
#define UART0_CTRL  (*(volatile uint32_t *)(UART0_BASE + 0x08))

#define UART_STATE_TXFULL (1u << 0)
#define UART_CTRL_TX_EN   (1u << 0)

static void uart_putc(char c) {
    while (UART0_STATE & UART_STATE_TXFULL) {
        /* wait for space in the transmit buffer */
    }
    UART0_DATA = (uint32_t)c;
}

static void uart_puts(const char *s) {
    while (*s) {
        if (*s == '\n') {
            uart_putc('\r'); /* most terminals want CRLF, not just LF */
        }
        uart_putc(*s++);
    }
}

int main(void) {
    UART0_CTRL = UART_CTRL_TX_EN;

    uart_puts("Hello from Cortex-M7 on QEMU mps2-an500!\n");
    uart_puts("Boot smoke test OK.\n");

    while (1) {
        /* Nothing else to do yet -- this is just confirming boot + UART
         * output work before linking in TFLM and running real inference. */
    }

    return 0;
}
