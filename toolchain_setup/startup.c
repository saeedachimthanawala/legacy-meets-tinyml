#include <stdint.h>

/* Symbols defined by the linker script (mps2_an500.ld) */
extern uint32_t _estack;
extern uint32_t _sidata;
extern uint32_t _sdata;
extern uint32_t _edata;
extern uint32_t _sbss;
extern uint32_t _ebss;

extern int main(void);

void Reset_Handler(void);
static void Default_Handler(void);

/* Weak aliases: any handler not explicitly overridden falls back to
 * Default_Handler (an infinite loop), so a stray/unused interrupt
 * doesn't crash into garbage memory. */
void NMI_Handler(void)        __attribute__((weak, alias("Default_Handler")));
void HardFault_Handler(void)  __attribute__((weak, alias("Default_Handler")));
void MemManage_Handler(void)  __attribute__((weak, alias("Default_Handler")));
void BusFault_Handler(void)   __attribute__((weak, alias("Default_Handler")));
void UsageFault_Handler(void) __attribute__((weak, alias("Default_Handler")));
void SVC_Handler(void)        __attribute__((weak, alias("Default_Handler")));
void DebugMon_Handler(void)   __attribute__((weak, alias("Default_Handler")));
void PendSV_Handler(void)     __attribute__((weak, alias("Default_Handler")));
void SysTick_Handler(void)    __attribute__((weak, alias("Default_Handler")));

/* The ARMv7-M architecture requires the very first word in memory to be
 * the initial stack pointer, and the second word to be the reset handler
 * address. The CPU loads these automatically on power-up/reset -- no
 * assembly startup file is required for this part. */
__attribute__((section(".isr_vector")))
void (* const vector_table[])(void) = {
    (void (*)(void))&_estack,   /* 0: initial MSP value        */
    Reset_Handler,               /* 1: Reset                    */
    NMI_Handler,                  /* 2: NMI                      */
    HardFault_Handler,            /* 3: Hard fault                */
    MemManage_Handler,            /* 4: MPU fault                 */
    BusFault_Handler,             /* 5: Bus fault                 */
    UsageFault_Handler,           /* 6: Usage fault                */
    0, 0, 0, 0,                  /* 7-10: Reserved                */
    SVC_Handler,                  /* 11: SVCall                    */
    DebugMon_Handler,             /* 12: Debug monitor             */
    0,                             /* 13: Reserved                 */
    PendSV_Handler,                /* 14: PendSV                   */
    SysTick_Handler,               /* 15: SysTick                  */
    /* Vectors 16+ are external (peripheral) interrupts. None are used
     * by this smoke test, so they're omitted -- the linker script only
     * reserves this array, it doesn't require a specific length. */
};

extern void __libc_init_array(void);

void Reset_Handler(void) {
    /* Copy initialized .data out of Flash (where it's stored) into RAM
     * (where it actually lives and gets read/written at runtime). */
    uint32_t *src = &_sidata;
    uint32_t *dst = &_sdata;
    while (dst < &_edata) {
        *dst++ = *src++;
    }

    /* Zero-fill .bss (uninitialized globals/statics). C assumes these
     * start at zero; nothing does that for us on bare metal. */
    dst = &_sbss;
    while (dst < &_ebss) {
        *dst++ = 0;
    }

    /* Run C++ static constructors (global objects, etc.) before main().
     * Not needed for the plain-C smoke test, but TFLM is C++ and relies
     * on this having run. */
    __libc_init_array();

    main();

    /* main() should never return on bare metal -- if it does, park here
     * rather than falling off into undefined memory. */
    while (1) {
    }
}

static void Default_Handler(void) {
    while (1) {
    }
}
