/*
 * The GPU renderer's replay side has two backends behind the gpu_* functions of gpu.h: Vulkan
 * through SDL3's GPU API (gpu_vk.c, the default) and OpenGL 4.5 core (gpu_gl.c, SSTHEH_GS=gl).
 * gpu_backend.c picks one when the program starts and forwards to it; the recording side (gs.c)
 * does not know which one is in use.
 */
#pragma once
#include "gpu.h"

typedef struct {
    const char *name;
    u64 (*window_flags)(void);      /* SDL window flags the backend needs; may set GL attributes */
    int (*init)(void *sdl_window);
    void (*upload)(GpuFrame *f);
    void (*render)(GpuFrame *f, const GpuVertex *verts);
    void (*present)(void);
    void (*set_vsync)(int on);
    int (*vsync)(void);
    int (*read_display)(u32 *out, int max_w, int max_h, int *w, int *h);
    void (*set_overlay)(const u32 *px, int w, int h);
} GpuBackend;

extern const GpuBackend gpu_backend_vk, gpu_backend_gl;
