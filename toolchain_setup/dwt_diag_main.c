#include <stdint.h>

#define UART0_BASE  0x40004000UL
#define UART0_DATA  (*(volatile uint32_t *)(UART0_BASE + 0x00))
#define UART0_STATE (*(volatile uint32_t *)(UART0_BASE + 0x04))
#define UART0_CTRL  (*(volatile uint32_t *)(UART0_BASE + 0x08))
#define UART_STATE_TXFULL (1u << 0)
#define UART_CTRL_TX_EN   (1u << 0)

#define DEMCR       (*(volatile uint32_t *)0xE000EDFCUL)
#define DWT_CTRL    (*(volatile uint32_t *)0xE0001000UL)
#define DWT_CYCCNT  (*(volatile uint32_t *)0xE0001004UL)
#define DEMCR_TRCENA        (1u << 24)
#define DWT_CTRL_CYCCNTENA  (1u << 0)

static void uart_putc(char c) {
    while (UART0_STATE & UART_STATE_TXFULL) {
    }
    UART0_DATA = (uint32_t)c;
}

static void uart_puts(const char *s) {
    while (*s) {
        if (*s == '\n') uart_putc('\r');
        uart_putc(*s++);
    }
}

static void print_uint32(uint32_t v) {
    char tmp[12];
    int t = 0;
    if (v == 0) { uart_puts("0"); return; }
    while (v > 0) { tmp[t++] = '0' + (v % 10); v /= 10; }
    char buf[12];
    int i = 0;
    while (t > 0) buf[i++] = tmp[--t];
    buf[i] = '\0';
    uart_puts(buf);
}

/* A fixed, simple, easily-reasoned-about workload: a busy loop of a known
 * iteration count, doing a trivial volatile increment each time. On real
 * hardware or a cycle-accurate simulator this should take a very
 * predictable, small number of cycles per iteration. */
volatile uint32_t dummy = 0;

static void busy_loop(uint32_t iterations) {
    for (uint32_t i = 0; i < iterations; i++) {
        dummy++;
    }
}

int main(void) {
    UART0_CTRL = UART_CTRL_TX_EN;

    DEMCR |= DEMCR_TRCENA;
    DWT_CYCCNT = 0;
    DWT_CTRL |= DWT_CTRL_CYCCNTENA;

    uart_puts("DWT cycle counter diagnostic\n");

    uint32_t counts[3] = {1000, 10000, 100000};

    for (int i = 0; i < 3; i++) {
        uint32_t start = DWT_CYCCNT;
        busy_loop(counts[i]);
        uint32_t end = DWT_CYCCNT;

        uart_puts("  iterations=");
        print_uint32(counts[i]);
        uart_puts("  cycles=");
        print_uint32(end - start);
        uart_puts("  cycles_per_iter=");
        print_uint32((end - start) / counts[i]);
        uart_puts("\n");
    }

    uart_puts("Done.\n");

    while (1) {
    }
    return 0;
}
