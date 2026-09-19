/*
 * c51_types.h
 * ============
 * Minimal stand-in for <stdint.h>, since this Keil C51 installation
 * doesn't ship it. Included by inference_template_8051.c ONLY --
 * do NOT include this (or <stdint.h>) from any model/test-vector
 * header, or you will get redefinition errors.
 */
#ifndef C51_TYPES_H
#define C51_TYPES_H

typedef signed char int8_t;
typedef long int32_t;

#endif
