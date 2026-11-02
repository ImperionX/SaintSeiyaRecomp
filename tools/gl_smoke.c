/* Smoke test of the OpenGL backend (port/gs/gpu_gl.c): synthetic frames replayed on a real GL
 * context, pixels checked (orientation, texture, blend, scissor, depth, render to texture, the window).
 * Put it in tools/ and, from the repo root:
 *   gcc -O1 -g -w -Iport/runtime -Iport/kernel -Iport/iop -Iport/gs tools/gl_smoke.c -o /tmp/gl_smoke -lSDL3 -lm
 *   /tmp/gl_smoke                (on a desktop; headless: xvfb-run -a /tmp/gl_smoke)
 * It prints "N checks, 0 failed" when every pixel is where it should be. */
/* Smoke test of gpu_gl.c: synthetic frames replayed on a real GL context, pixels checked.
 * Includes the backend's source so that its statics can be reached. */
/* Smoke test of gpu_gl.c: synthetic frames replayed on a real GL context, pixels checked.
 * Includes the backend's source so that its statics can be reached. */
#define GPU_GL_TEST 1
#include "gpu_gl.c"
#include <stdarg.h>

void rt_log(const char *a, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "[%s] ", a);
    vfprintf(stderr, fmt, ap);
    fputc('\n', stderr);
    va_end(ap);
}
_Noreturn void rt_fatal(Ctx *c, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    exit(2);
}
int kernel_ntsc(void) { return 1; }
u64 gs_priv_read(u32 o) { return 0; }
u32 gpu_tex_stats(u64 *b) { *b = 0; return 0; }
int gpu_on;

static GpuVertex verts[512];
static GpuDraw draws[32];
static GpuTexUpload ups[4];
static int nv, nd;

static GpuDraw *begin(void) {
    GpuDraw *d = &draws[nd++];
    memset(d, 0, sizeof *d);
    d->fbw = 10; d->zbp = 0x100; d->zpsm = 0x30; d->zwrite = 1; d->ztst = 2; d->fst = 1; d->tex = -1;
    d->region[0] = d->region[1] = d->region[2] = d->region[3] = -1.0f;
    d->scissor[0] = 0; d->scissor[1] = 639; d->scissor[2] = 0; d->scissor[3] = 447;
    d->first = nv;
    return d;
}
static void vtx(float x, float y, float z, float r, float g, float b, float a, float s, float t) {
    verts[nv++] = (GpuVertex){x, y, z, 0, r, g, b, a, s, t, 1, 0};
}
static GpuDraw *quad(float x0, float y0, float x1, float y1, float z, float r, float g, float b, float a, float u0, float v0, float u1, float v1) {
    GpuDraw *d = begin();
    vtx(x0, y0, z, r, g, b, a, u0, v0); vtx(x1, y0, z, r, g, b, a, u1, v0); vtx(x0, y1, z, r, g, b, a, u0, v1);
    vtx(x1, y0, z, r, g, b, a, u1, v0); vtx(x0, y1, z, r, g, b, a, u0, v1); vtx(x1, y1, z, r, g, b, a, u1, v1);
    d->count = 6;
    return d;
}

static int fails, checks;
static void expect(const char *what, u32 got, u32 want) {
    int ok = 1;
    for (int s = 0; s < 24; s += 8) {
        int a = (got >> s) & 255, b = (want >> s) & 255;
        if (abs(a - b) > 3) ok = 0;
    }
    checks++;
    if (!ok) fails++;
    printf("%s %-46s got %02X%02X%02X want %02X%02X%02X\n", ok ? "ok  " : "FAIL", what, got & 255, (got >> 8) & 255, (got >> 16) & 255,
           want & 255, (want >> 8) & 255, (want >> 16) & 255);
}
#define RGB(r, g, b) ((u32)(r) | (u32)(g) << 8 | (u32)(b) << 16)

static u32 shot[640 * 480];
static u32 backbuf[640 * 480];
static int have_back;
static void read_back(void) {
    typedef void (*RP)(int, int, int, int, unsigned, unsigned, void *);
    RP rp = (RP)SDL_GL_GetProcAddress("glReadPixels");
    rp(0, 0, 640, 480, GL_RGBA, GL_UNSIGNED_BYTE, backbuf);
    have_back = 1;
}
/* window pixel counted from the top */
static u32 win_px(int x, int y_top) { return backbuf[(479 - y_top) * 640 + x] & 0xFFFFFF; }

int main(void) {
    setenv("SSTHEH_SCALE", "2", 1);
    if (!SDL_Init(SDL_INIT_VIDEO)) { fprintf(stderr, "SDL: %s\n", SDL_GetError()); return 1; }
    SDL_Window *w = SDL_CreateWindow("t", 640, 480, gl_window_flags());
    if (!w) { fprintf(stderr, "window: %s\n", SDL_GetError()); return 1; }
    if (gl_init(w)) { fprintf(stderr, "gl_init failed\n"); return 1; }

    /* 0: red triangle at the top left (GS y grows down) */
    GpuDraw *d = begin();
    vtx(10, 10, 0.5f, 255, 0, 0, 128, 0, 0); vtx(110, 10, 0.5f, 255, 0, 0, 128, 0, 0); vtx(10, 110, 0.5f, 255, 0, 0, 128, 0, 0);
    d->count = 3;
    /* 1: 4 x 4 texture quad, nearest, decal */
    static u32 texels[16];
    for (int i = 0; i < 16; i++) texels[i] = RGB(128, 128, 128) | 0x80000000u;
    texels[0] = RGB(255, 0, 0) | 0x80000000u;  texels[3] = RGB(0, 255, 0) | 0x80000000u;
    texels[12] = RGB(0, 0, 255) | 0x80000000u; texels[15] = RGB(255, 255, 255) | 0x80000000u;
    ups[0] = (GpuTexUpload){1, 4, 4, texels};
    d = quad(200, 200, 264, 264, 0.5f, 128, 128, 128, 128, 0, 0, 4, 4);
    d->textured = 1; d->tfx = 1; d->tcc = 1; d->tex = 1; d->tw = 4; d->th = 4; d->wrap_u = d->wrap_v = 1;
    /* 2: half transparent green over the red: (Cs - Cd) * As + Cd */
    d = quad(20, 20, 40, 40, 0.5f, 0, 255, 0, 64, 0, 0, 0, 0);
    d->abe = 1; d->alpha = 0x44;
    /* 3: scissor */
    d = quad(280, 280, 360, 360, 0.5f, 0, 0, 255, 128, 0, 0, 0, 0);
    d->scissor[0] = 300; d->scissor[1] = 339; d->scissor[2] = 300; d->scissor[3] = 339;
    /* 4: depth, GEQUAL: yellow at .8, cyan at .3 is rejected, magenta at .9 on the right half wins */
    d = quad(400, 100, 450, 150, 0.8f, 255, 255, 0, 128, 0, 0, 0, 0); d->ztst = 3;
    d = quad(400, 100, 450, 150, 0.3f, 0, 255, 255, 128, 0, 0, 0, 0); d->ztst = 3;
    d = quad(425, 100, 450, 150, 0.9f, 255, 0, 255, 128, 0, 0, 0, 0); d->ztst = 3;
    /* 5: render to texture: green into target 0x20, then sampled into target 0 */
    d = quad(0, 0, 64, 64, 0.5f, 0, 255, 0, 128, 0, 0, 0, 0); d->fbp = 0x20;
    d = quad(500, 300, 564, 364, 0.5f, 128, 128, 128, 128, 0, 0, 64, 64);
    d->textured = 1; d->tfx = 1; d->tex = -2 - 0x20;
    /* 6: a target read by a draw into itself (copy first): the red of the triangle, moved */
    d = quad(500, 100, 530, 130, 0.5f, 128, 128, 128, 128, 45, 12, 75, 42);
    d->textured = 1; d->tfx = 1; d->tex = -2;

    /* 7: depth beyond the clip planes: the GS does not clip in z (Vulkan backend: depth clamp), so both are drawn */
    d = quad(200, 380, 250, 430, 1.5f, 255, 128, 0, 128, 0, 0, 0, 0); d->ztst = 3;
    d = quad(300, 380, 350, 430, -0.2f, 0, 128, 255, 128, 0, 0, 0, 0); d->ztst = 3;

    GpuFrame f = {0};
    f.draws = draws; f.ndraws = nd; f.verts = verts; f.nverts = nv; f.uploads = ups; f.nuploads = 1;
    f.priv[0] = 0; f.priv[1] = 0; f.priv[2] = ((u64)(640 - 1) << 32) | ((u64)(448 - 1) << 44);

    gl_upload(&f);
    gl_render(&f, NULL);
    gl_render(&f, NULL);            /* a second replay: fences, state cache, targets already there */
    int pw, ph;
    gl_read_display(shot, 1024, 1024, &pw, &ph);
    printf("display %d x %d\n", pw, ph);
#define P(x, y) (shot[(y) * pw + (x)] & 0xFFFFFF)
    expect("triangle, near the top left (y not flipped)", P(60, 15), RGB(255, 0, 0));
    expect("below the triangle's hypotenuse", P(100, 100), RGB(0, 0, 0));
    expect("bottom left is empty", P(30, 400), RGB(0, 0, 0));
    expect("top right is empty", P(400, 30), RGB(0, 0, 0));
    expect("blend: green 50% over red", P(30, 30), RGB(127, 127, 0));
    expect("texture (0,0) red, top left of the quad", P(205, 205), RGB(255, 0, 0));
    expect("texture (3,0) green, top right", P(260, 205), RGB(0, 255, 0));
    expect("texture (0,3) blue, bottom left", P(205, 260), RGB(0, 0, 255));
    expect("texture (3,3) white, bottom right", P(260, 260), RGB(255, 255, 255));
    expect("texture (1,1) grey", P(222, 222), RGB(128, 128, 128));
    expect("scissor: inside", P(310, 310), RGB(0, 0, 255));
    expect("scissor: left of it", P(290, 310), RGB(0, 0, 0));
    expect("scissor: right of it", P(350, 310), RGB(0, 0, 0));
    expect("scissor: above it", P(310, 290), RGB(0, 0, 0));
    expect("scissor: below it", P(310, 345), RGB(0, 0, 0));
    expect("depth: farther cyan rejected", P(410, 120), RGB(255, 255, 0));
    expect("depth: nearer magenta wins", P(440, 120), RGB(255, 0, 255));
    expect("render to texture", P(530, 330), RGB(0, 255, 0));
    expect("target read by a draw into itself", P(515, 115), RGB(255, 0, 0));
    expect("self-read draw's far corner is black too", P(500, 140), RGB(0, 0, 0));
    expect("z beyond the far plane is not clipped", P(225, 405), RGB(255, 128, 0));
    expect("z below the near plane is not clipped", P(325, 405), RGB(0, 128, 255));

    /* the window: picture 640 x 448 stretched to 4:3 in 640 x 480, y up, and the overlay at the top left */
    static u32 ov[16 * 16];
    for (int y = 0; y < 16; y++)
        for (int x = 0; x < 16; x++) ov[y * 16 + x] = y < 8 ? 0xFFFFFFFFu : 0xFF000000u;
    gl_set_overlay(ov, 16, 16);
    gpu_gl_before_swap = read_back;
    gl_present();
    if (have_back) {
        expect("window: triangle at the top (y up)", win_px(60, 24), RGB(255, 0, 0));
        expect("window: bottom is empty", win_px(60, 460), RGB(0, 0, 0));
        expect("window: overlay top half white", win_px(10, 6), RGB(255, 255, 255));
        expect("window: overlay bottom half black", win_px(10, 17), RGB(0, 0, 0));
        expect("window: texture quad, red corner", win_px(205, 215), RGB(255, 0, 0));
    }
    gl_set_vsync(1);
    printf("vsync %d\n", gl_vsync());
    printf("%d checks, %d failed\n", checks, fails);
    return fails != 0;
}
