#include <rt_misc.h>

/* TFLM does a small amount of dynamic allocation internally. Rather than
 * rely on Keil's auto-generated scatter file to correctly guess a heap
 * region (which is what's failing -- see L6915E), we give the C library
 * an explicit, fixed heap buffer directly in code. 16KB is generous for
 * this model size; increase HEAP_SIZE later if a larger model needs more. */
#define HEAP_SIZE 0x4000

static unsigned char heap_mem[HEAP_SIZE] __attribute__((aligned(8)));

__value_in_regs struct __initial_stackheap __user_initial_stackheap(
    unsigned R0, unsigned SP, unsigned R2, unsigned SL) {
    struct __initial_stackheap config;
    config.heap_base = (unsigned)heap_mem;
    config.heap_limit = (unsigned)(heap_mem + HEAP_SIZE);
    return config;
}
