/*
 * The public gpu_* functions of gpu.h, forwarding to the replay backend chosen by SSTHEH_GS:
 *   (unset)  Vulkan through SDL3's GPU API (gpu_vk.c)
 *   gl       OpenGL 4.5 core (gpu_gl.c): for GPUs without a Vulkan driver
 *   sw       no GPU renderer at all (main.c: gpu_on = 0, the software GS in gs.c)
 * If the chosen backend cannot start, gpu_init fails and display.c falls back to the software GS;
 * it never silently swaps one GPU backend for the other.
 */
#include <stdlib.h>
#include <string.h>

#include "gpu_backend.h"

static const GpuBackend *be;

static const GpuBackend *backend(void) {
    if (!be) {
        const char *g = getenv("SSTHEH_GS");
        be = g && !strcmp(g, "gl") ? &gpu_backend_gl : &gpu_backend_vk;
    }
    return be;
}

u64 gpu_window_flags(void) { return backend()->window_flags(); }
int gpu_init(void *sdl_window) { return backend()->init(sdl_window); }
void gpu_upload(GpuFrame *f) { backend()->upload(f); }
void gpu_render(GpuFrame *f, const GpuVertex *verts) { backend()->render(f, verts); }
void gpu_present(void) { backend()->present(); }
void gpu_set_vsync(int on) { backend()->set_vsync(on); }
int gpu_vsync(void) { return backend()->vsync(); }
int gpu_read_display(u32 *out, int max_w, int max_h, int *w, int *h) { return backend()->read_display(out, max_w, max_h, w, h); }
void gpu_set_overlay(const u32 *px, int w, int h) { backend()->set_overlay(px, w, h); }
