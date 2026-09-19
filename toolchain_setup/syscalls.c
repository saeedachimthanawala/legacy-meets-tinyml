#include <sys/stat.h>
#include <errno.h>
#include <stdint.h>

/* These are the syscalls newlib expects an embedded target to provide.
 * None of them are meaningful without an OS, so each just reports
 * failure/no-op in the simplest safe way. This is a standard, widely
 * used minimal stub set for bare-metal newlib linking. */

extern uint8_t _ebss; /* from the linker script: end of .bss = start of heap */
extern uint8_t _estack;

static uint8_t *heap_end = 0;

void *_sbrk(int incr) {
    uint8_t *prev_heap_end;

    if (heap_end == 0) {
        heap_end = &_ebss;
    }
    prev_heap_end = heap_end;

    /* Very small guard: don't let the heap run into the stack. This
     * firmware barely uses the heap, so this is generous, not tight. */
    if (heap_end + incr > (uint8_t *)&_estack - 4096) {
        errno = ENOMEM;
        return (void *)-1;
    }

    heap_end += incr;
    return (void *)prev_heap_end;
}

int _close(int file) {
    (void)file;
    return -1;
}

int _fstat(int file, struct stat *st) {
    (void)file;
    st->st_mode = S_IFCHR;
    return 0;
}

int _isatty(int file) {
    (void)file;
    return 1;
}

int _lseek(int file, int ptr, int dir) {
    (void)file;
    (void)ptr;
    (void)dir;
    return 0;
}

int _read(int file, char *ptr, int len) {
    (void)file;
    (void)ptr;
    (void)len;
    return 0;
}

int _write(int file, char *ptr, int len) {
    (void)file;
    (void)ptr;
    (void)len;
    return len;
}

void _exit(int status) {
    (void)status;
    while (1) {
    }
}

int _kill(int pid, int sig) {
    (void)pid;
    (void)sig;
    errno = EINVAL;
    return -1;
}

int _getpid(void) {
    return 1;
}

/* Normally supplied by crti.o/crtn.o, which -nostartfiles excludes.
 * __libc_init_array() calls _init() before running C++ static
 * constructors; we handle constructors via .init_array directly, so
 * this just needs to exist, not do anything. */
void _init(void) {
}

void _fini(void) {
}
