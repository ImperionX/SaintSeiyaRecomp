/*
 * GPU renderer, replay side: OpenGL 4.5 (core profile), on the window's thread. It does the job of
 * gpu_vk.c (read its header for the model: one colour texture per GS frame buffer address, one
 * depth texture per depth buffer address, textures kept by id, the GS blend as blend factors,
 * alpha stored so that 0x80 is 1.0, the displayed buffer scaled to the window) with the same
 * limits. Chosen with SSTHEH_GS=gl, for GPUs that have no Vulkan driver.
 *
 * Conventions, so that the recorded frames replay as they do in Vulkan:
 *   - row r of a render target is GS row r: the vertex shader does not flip y (with OpenGL's
 *     lower-left origin NDC y = -1 is row 0), and textures are uploaded row 0 first;
 *   - clip depth is 0..1 as in Vulkan (glClipControl), so the depth values the recorder gives
 *     are stored as they are; depth is cleared to 0 and compared with GEQUAL like there;
 *   - the default framebuffer (window) is only touched by gpu_present, which draws the picture
 *     with y up as OpenGL has it.
 *
 * Nothing but SDL3 is linked: the functions are loaded with SDL_GL_GetProcAddress, and the types
 * and constants used are declared here.
 */
#include <SDL3/SDL.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "r5900.h"
#include "runtime.h"
#include "gs.h"
#include "gpu.h"
#include "gpu_backend.h"

int kernel_ntsc(void);

/* ------------------------------------------------------------------------------- OpenGL */

typedef unsigned int GLenum, GLuint, GLbitfield;
typedef int GLint, GLsizei;
typedef unsigned char GLboolean, GLubyte;
typedef float GLfloat;
typedef char GLchar;
typedef ptrdiff_t GLsizeiptr, GLintptr;
typedef unsigned long long GLuint64;
typedef struct __GLsync *GLsync;

#define GL_FALSE 0
#define GL_TRUE 1
#define GL_ZERO 0
#define GL_ONE 1
#define GL_NEVER 0x0200
#define GL_GREATER 0x0204
#define GL_GEQUAL 0x0206
#define GL_ALWAYS 0x0207
#define GL_SRC_ALPHA 0x0302
#define GL_ONE_MINUS_SRC_ALPHA 0x0303
#define GL_DST_ALPHA 0x0304
#define GL_ONE_MINUS_DST_ALPHA 0x0305
#define GL_CONSTANT_COLOR 0x8001
#define GL_ONE_MINUS_CONSTANT_COLOR 0x8002
#define GL_FUNC_ADD 0x8006
#define GL_FUNC_SUBTRACT 0x800A
#define GL_FUNC_REVERSE_SUBTRACT 0x800B
#define GL_DITHER 0x0BD0
#define GL_DEPTH_CLAMP 0x864F
#define GL_BLEND 0x0BE2
#define GL_DEPTH_TEST 0x0B71
#define GL_SCISSOR_TEST 0x0C11
#define GL_MAX_TEXTURE_SIZE 0x0D33
#define GL_VENDOR 0x1F00
#define GL_RENDERER 0x1F01
#define GL_VERSION 0x1F02
#define GL_COLOR_BUFFER_BIT 0x4000
#define GL_TRIANGLES 4
#define GL_UNSIGNED_BYTE 0x1401
#define GL_FLOAT 0x1406
#define GL_DEPTH_COMPONENT 0x1902
#define GL_RGBA 0x1908
#define GL_NEAREST 0x2600
#define GL_LINEAR 0x2601
#define GL_TEXTURE_MAG_FILTER 0x2800
#define GL_TEXTURE_MIN_FILTER 0x2801
#define GL_TEXTURE_WRAP_S 0x2802
#define GL_TEXTURE_WRAP_T 0x2803
#define GL_REPEAT 0x2901
#define GL_CLAMP_TO_EDGE 0x812F
#define GL_TEXTURE_WRAP_R 0x8072
#define GL_TEXTURE_2D 0x0DE1
#define GL_RGBA8 0x8058
#define GL_DEPTH_COMPONENT32F 0x8CAC
#define GL_FRAMEBUFFER 0x8D40
#define GL_COLOR_ATTACHMENT0 0x8CE0
#define GL_DEPTH_ATTACHMENT 0x8D00
#define GL_FRAMEBUFFER_COMPLETE 0x8CD5
#define GL_STREAM_DRAW 0x88E0
#define GL_FRAGMENT_SHADER 0x8B30
#define GL_VERTEX_SHADER 0x8B31
#define GL_COMPILE_STATUS 0x8B81
#define GL_LINK_STATUS 0x8B82
#define GL_LOWER_LEFT 0x8CA1
#define GL_ZERO_TO_ONE 0x935F
#define GL_TEXTURE_MAX_ANISOTROPY 0x84FE
#define GL_MAX_TEXTURE_MAX_ANISOTROPY 0x84FF
#define GL_SYNC_GPU_COMMANDS_COMPLETE 0x9117
#define GL_SYNC_FLUSH_COMMANDS_BIT 1
#define GL_TIMEOUT_EXPIRED 0x911B

#define GL_FUNCS(X) \
    X(const GLubyte *, glGetString, (GLenum)) \
    X(void, glGetIntegerv, (GLenum, GLint *)) \
    X(void, glGetFloatv, (GLenum, GLfloat *)) \
    X(GLenum, glGetError, (void)) \
    X(void, glEnable, (GLenum)) \
    X(void, glDisable, (GLenum)) \
    X(void, glViewport, (GLint, GLint, GLsizei, GLsizei)) \
    X(void, glScissor, (GLint, GLint, GLsizei, GLsizei)) \
    X(void, glColorMask, (GLboolean, GLboolean, GLboolean, GLboolean)) \
    X(void, glDepthMask, (GLboolean)) \
    X(void, glDepthFunc, (GLenum)) \
    X(void, glClearColor, (GLfloat, GLfloat, GLfloat, GLfloat)) \
    X(void, glClear, (GLbitfield)) \
    X(void, glBlendFuncSeparate, (GLenum, GLenum, GLenum, GLenum)) \
    X(void, glBlendEquationSeparate, (GLenum, GLenum)) \
    X(void, glBlendColor, (GLfloat, GLfloat, GLfloat, GLfloat)) \
    X(void, glClipControl, (GLenum, GLenum)) \
    X(void, glDrawArrays, (GLenum, GLint, GLsizei)) \
    X(void, glFlush, (void)) \
    X(void, glCreateTextures, (GLenum, GLsizei, GLuint *)) \
    X(void, glTextureStorage2D, (GLuint, GLsizei, GLenum, GLsizei, GLsizei)) \
    X(void, glTextureSubImage2D, (GLuint, GLint, GLint, GLint, GLsizei, GLsizei, GLenum, GLenum, const void *)) \
    X(void, glDeleteTextures, (GLsizei, const GLuint *)) \
    X(void, glBindTextureUnit, (GLuint, GLuint)) \
    X(void, glClearTexImage, (GLuint, GLint, GLenum, GLenum, const void *)) \
    X(void, glCopyImageSubData, (GLuint, GLenum, GLint, GLint, GLint, GLint, GLuint, GLenum, GLint, GLint, GLint, GLint, GLsizei, GLsizei, GLsizei)) \
    X(void, glGetTextureSubImage, (GLuint, GLint, GLint, GLint, GLint, GLsizei, GLsizei, GLsizei, GLenum, GLenum, GLsizei, void *)) \
    X(void, glCreateSamplers, (GLsizei, GLuint *)) \
    X(void, glSamplerParameteri, (GLuint, GLenum, GLint)) \
    X(void, glSamplerParameterf, (GLuint, GLenum, GLfloat)) \
    X(void, glBindSampler, (GLuint, GLuint)) \
    X(void, glCreateFramebuffers, (GLsizei, GLuint *)) \
    X(void, glNamedFramebufferTexture, (GLuint, GLenum, GLuint, GLint)) \
    X(GLenum, glCheckNamedFramebufferStatus, (GLuint, GLenum)) \
    X(void, glBindFramebuffer, (GLenum, GLuint)) \
    X(void, glDeleteFramebuffers, (GLsizei, const GLuint *)) \
    X(void, glCreateBuffers, (GLsizei, GLuint *)) \
    X(void, glNamedBufferData, (GLuint, GLsizeiptr, const void *, GLenum)) \
    X(void, glCreateVertexArrays, (GLsizei, GLuint *)) \
    X(void, glVertexArrayVertexBuffer, (GLuint, GLuint, GLuint, GLintptr, GLsizei)) \
    X(void, glEnableVertexArrayAttrib, (GLuint, GLuint)) \
    X(void, glVertexArrayAttribFormat, (GLuint, GLuint, GLint, GLenum, GLboolean, GLuint)) \
    X(void, glVertexArrayAttribBinding, (GLuint, GLuint, GLuint)) \
    X(void, glBindVertexArray, (GLuint)) \
    X(GLuint, glCreateShader, (GLenum)) \
    X(void, glShaderSource, (GLuint, GLsizei, const GLchar *const *, const GLint *)) \
    X(void, glCompileShader, (GLuint)) \
    X(void, glGetShaderiv, (GLuint, GLenum, GLint *)) \
    X(void, glGetShaderInfoLog, (GLuint, GLsizei, GLsizei *, GLchar *)) \
    X(GLuint, glCreateProgram, (void)) \
    X(void, glAttachShader, (GLuint, GLuint)) \
    X(void, glLinkProgram, (GLuint)) \
    X(void, glGetProgramiv, (GLuint, GLenum, GLint *)) \
    X(void, glGetProgramInfoLog, (GLuint, GLsizei, GLsizei *, GLchar *)) \
    X(void, glUseProgram, (GLuint)) \
    X(void, glDeleteShader, (GLuint)) \
    X(void, glUniform4f, (GLint, GLfloat, GLfloat, GLfloat, GLfloat)) \
    X(void, glUniform4i, (GLint, GLint, GLint, GLint, GLint)) \
    X(GLsync, glFenceSync, (GLenum, GLbitfield)) \
    X(GLenum, glClientWaitSync, (GLsync, GLbitfield, GLuint64)) \
    X(void, glDeleteSync, (GLsync))

#define X(ret, name, args) static ret (*name) args;
GL_FUNCS(X)
#undef X

static int load_gl(void) {
#define X(ret, name, args) \
    if (!(*(void **)&name = (void *)SDL_GL_GetProcAddress(#name))) { rt_log("gpu", "OpenGL: %s is missing", #name); return -1; }
    GL_FUNCS(X)
#undef X
    return 0;
}

/* ------------------------------------------------------------------------------- shaders */

static const char *gs_vert_src =
    "#version 450 core\n"
    "layout(location = 0) in vec4 in_pos;\n"
    "layout(location = 1) in vec4 in_color;\n"
    "layout(location = 2) in vec4 in_tex;\n"
    "layout(location = 0) out vec4 v_color;\n"
    "layout(location = 1) out vec3 v_tex;\n"
    "layout(location = 0) uniform vec4 size;\n"
    "void main() {\n"
    "    vec2 p = in_pos.xy * size.zw * 2.0 - 1.0;\n"
    "    gl_Position = vec4(p, in_pos.z, 1.0);\n"
    "    v_color = in_color / 255.0;\n"
    "    v_tex = in_tex.xyz;\n"
    "}\n";

static const char *gs_frag_src =
    "#version 450 core\n"
    "layout(location = 0) in vec4 v_color;\n"
    "layout(location = 1) in vec3 v_tex;\n"
    "layout(location = 0) out vec4 out_color;\n"
    "layout(binding = 0) uniform sampler2D tex;\n"
    "layout(location = 1) uniform ivec4 mode;\n"
    "layout(location = 2) uniform ivec4 atest;\n"
    "layout(location = 3) uniform vec4 texsize;\n"
    "layout(location = 4) uniform vec4 region;\n"
    "void main() {\n"
    "    vec4 vc = v_color * 255.0;\n"
    "    vec4 c = vc;\n"
    "    if (mode.x != 0) {\n"
    "        vec2 uv = mode.w != 0 ? v_tex.xy : v_tex.xy / v_tex.z * texsize.xy;\n"
    "        if (region.x >= 0.0) uv = clamp(uv, region.xy, region.zw + 0.999);\n"
    "        vec4 tc = texture(tex, uv * texsize.zw) * 255.0;\n"
    "        vec3 rgb;\n"
    "        if (mode.y == 0) rgb = tc.rgb * vc.rgb / 128.0;\n"
    "        else if (mode.y == 1) rgb = tc.rgb;\n"
    "        else rgb = tc.rgb * vc.rgb / 128.0 + vc.a;\n"
    "        float a;\n"
    "        if (mode.z == 0) a = vc.a;\n"
    "        else if (mode.y == 0) a = tc.a * vc.a / 128.0;\n"
    "        else if (mode.y == 2) a = tc.a + vc.a;\n"
    "        else a = tc.a;\n"
    "        c = vec4(min(rgb, vec3(255.0)), min(a, 255.0));\n"
    "    }\n"
    "    if (atest.x != 0) {\n"
    "        float a = floor(c.a + 0.5), r = float(atest.z);\n"
    "        bool pass;\n"
    "        switch (atest.y) {\n"
    "        case 0: pass = false; break;\n"
    "        case 1: pass = true; break;\n"
    "        case 2: pass = a < r; break;\n"
    "        case 3: pass = a <= r; break;\n"
    "        case 4: pass = a == r; break;\n"
    "        case 5: pass = a >= r; break;\n"
    "        case 6: pass = a > r; break;\n"
    "        default: pass = a != r; break;\n"
    "        }\n"
    "        if (!pass && (atest.w == 0 || atest.w == 2)) discard;\n"
    "    }\n"
    "    out_color = vec4(c.rgb / 255.0, c.a / 128.0);\n"
    "}\n";

static const char *present_vert_src =
    "#version 450 core\n"
    "layout(location = 0) out vec2 v_uv;\n"
    "layout(location = 0) uniform vec4 rect;\n"
    "void main() {\n"
    "    vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);\n"
    "    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);\n"
    "    v_uv = vec2(mix(rect.x, rect.z, p.x), mix(rect.w, rect.y, p.y));\n"
    "}\n";

static const char *present_frag_src =
    "#version 450 core\n"
    "layout(location = 0) in vec2 v_uv;\n"
    "layout(location = 0) out vec4 out_color;\n"
    "layout(binding = 0) uniform sampler2D src;\n"
    "layout(location = 1) uniform vec4 down;\n"
    "void main() {\n"
    "    int n = int(down.z);\n"
    "    if (n <= 1) { out_color = vec4(texture(src, v_uv).rgb, 1.0); return; }\n"
    "    vec3 sum = vec3(0.0);\n"
    "    for (int j = 0; j < n; j++)\n"
    "        for (int i = 0; i < n; i++) {\n"
    "            vec2 off = (vec2(i, j) + 0.5) / float(n) - 0.5;\n"
    "            sum += texture(src, v_uv + off * down.xy).rgb;\n"
    "        }\n"
    "    out_color = vec4(sum / float(n * n), 1.0);\n"
    "}\n";

/* ------------------------------------------------------------------------------- state */

#define RT_SIZE 1024
#define MAX_RT 64
#define MAX_TEX GPU_MAX_TEX

static int scale = 2;
static SDL_Window *win;
static int own_win;
static SDL_GLContext ctx;
static GLuint gs_prog, present_prog, vao, vbo;
static GLuint white, scratch;
static GLuint samplers[8];

typedef struct { u32 key; GLuint tex; } Target;
static Target colors[MAX_RT], depths[MAX_RT];
static int ncolors, ndepths;

typedef struct { GLuint color, depth, fbo; } Fbo;
static Fbo fbos[256];
static int nfbos;

typedef struct { u32 id; GLuint tex; u32 last_used; u32 w, h; } Tex;
static Tex texs[MAX_TEX];
static u32 frame_no;

static GLuint overlay;
static int overlay_w, overlay_h, overlay_on;
static int vsync_on = 1;
static int debug_flags = -1;
static u64 up_bytes, up_count;

#ifdef GPU_GL_TEST
void (*gpu_gl_before_swap)(void);
#endif

/* ------------------------------------------------------------------------------- setup */

static GLuint compile(GLenum type, const char *src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, NULL);
    glCompileShader(s);
    GLint ok = 0;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[2048] = "";
        glGetShaderInfoLog(s, sizeof log, NULL, log);
        rt_log("gpu", "OpenGL shader: %s", log);
        return 0;
    }
    return s;
}

static GLuint program(const char *vsrc, const char *fsrc) {
    GLuint v = compile(GL_VERTEX_SHADER, vsrc), f = compile(GL_FRAGMENT_SHADER, fsrc);
    if (!v || !f) return 0;
    GLuint p = glCreateProgram();
    glAttachShader(p, v);
    glAttachShader(p, f);
    glLinkProgram(p);
    glDeleteShader(v);
    glDeleteShader(f);
    GLint ok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[2048] = "";
        glGetProgramInfoLog(p, sizeof log, NULL, log);
        rt_log("gpu", "OpenGL program: %s", log);
        return 0;
    }
    return p;
}

static GLuint make_tex(u32 w, u32 h, GLenum fmt) {
    GLuint t;
    glCreateTextures(GL_TEXTURE_2D, 1, &t);
    glTextureStorage2D(t, 1, fmt, (GLsizei)w, (GLsizei)h);
    return t;
}

static u64 gl_window_flags(void) {
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 5);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
    return SDL_WINDOW_OPENGL;
}

static void gl_set_vsync(int on) {
    if (!ctx) return;
    if (!SDL_GL_SetSwapInterval(on ? 1 : 0)) rt_log("gpu", "swap interval: %s", SDL_GetError());
    else vsync_on = on;
}

static int gl_vsync(void) { return vsync_on; }

static int gl_init(void *sdl_window) {
    const char *sc = getenv("SSTHEH_SCALE");
    win = sdl_window;
    if (!win) {
        u64 fl = gl_window_flags();
        win = SDL_CreateWindow("sstheh", 64, 64, SDL_WINDOW_HIDDEN | fl);
        if (!win) { rt_log("gpu", "OpenGL window: %s", SDL_GetError()); return -1; }
        own_win = 1;
    }
    ctx = SDL_GL_CreateContext(win);
    if (!ctx) { rt_log("gpu", "no OpenGL 4.5 context: %s", SDL_GetError()); return -1; }
    SDL_GL_MakeCurrent(win, ctx);
    if (load_gl()) return -1;
    GLint maxtex = 0;
    glGetIntegerv(GL_MAX_TEXTURE_SIZE, &maxtex);
    if (sc && atoi(sc) >= 1 && atoi(sc) <= 8) scale = atoi(sc);
    else if (!own_win) {
        const SDL_DisplayMode *m = SDL_GetCurrentDisplayMode(SDL_GetDisplayForWindow(win));
        int h = m ? m->h : 1080;
        scale = (h * 3 / 2 + 447) / 448;
        if (scale < 2) scale = 2;
        if (scale > 4) scale = 4;
    }
    while (scale > 1 && RT_SIZE * scale > maxtex) scale--;
    rt_log("gpu", "OpenGL %s on %s, scale %d", (const char *)glGetString(GL_VERSION), (const char *)glGetString(GL_RENDERER), scale);

    glClipControl(GL_LOWER_LEFT, GL_ZERO_TO_ONE);
    glDisable(GL_DITHER);
    glEnable(GL_DEPTH_CLAMP);
    gs_prog = program(gs_vert_src, gs_frag_src);
    present_prog = program(present_vert_src, present_frag_src);
    if (!gs_prog || !present_prog) return -1;

    glCreateVertexArrays(1, &vao);
    glCreateBuffers(1, &vbo);
    glVertexArrayVertexBuffer(vao, 0, vbo, 0, (GLsizei)sizeof(GpuVertex));
    for (GLuint i = 0; i < 3; i++) {
        glEnableVertexArrayAttrib(vao, i);
        glVertexArrayAttribFormat(vao, i, 4, GL_FLOAT, GL_FALSE, i * 16);
        glVertexArrayAttribBinding(vao, i, 0);
    }

    float aniso = 0;
    if (SDL_GL_ExtensionSupported("GL_EXT_texture_filter_anisotropic")) glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY, &aniso);
    glCreateSamplers(8, samplers);
    for (int i = 0; i < 8; i++) {
        glSamplerParameteri(samplers[i], GL_TEXTURE_MIN_FILTER, (i & 1) ? GL_LINEAR : GL_NEAREST);
        glSamplerParameteri(samplers[i], GL_TEXTURE_MAG_FILTER, (i & 1) ? GL_LINEAR : GL_NEAREST);
        glSamplerParameteri(samplers[i], GL_TEXTURE_WRAP_S, (i & 2) ? GL_CLAMP_TO_EDGE : GL_REPEAT);
        glSamplerParameteri(samplers[i], GL_TEXTURE_WRAP_T, (i & 4) ? GL_CLAMP_TO_EDGE : GL_REPEAT);
        glSamplerParameteri(samplers[i], GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
        if ((i & 1) && aniso >= 1.0f) glSamplerParameterf(samplers[i], GL_TEXTURE_MAX_ANISOTROPY, aniso < 16.0f ? aniso : 16.0f);
    }
    white = make_tex(1, 1, GL_RGBA8);
    u32 w = 0xFFFFFFFFu;
    glTextureSubImage2D(white, 0, 0, 0, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, &w);
    if (!own_win) gl_set_vsync(getenv("SSTHEH_VSYNC") && atoi(getenv("SSTHEH_VSYNC")));
    return 0;
}

/* ------------------------------------------------------------------------------- targets, textures */

static GLuint target(Target *list, int *n, u32 key, int depth) {
    for (int i = 0; i < *n; i++)
        if (list[i].key == key) return list[i].tex;
    if (*n == MAX_RT) return list[0].tex;
    GLuint t = depth ? make_tex(RT_SIZE * scale, RT_SIZE * scale, GL_DEPTH_COMPONENT32F)
                     : make_tex(RT_SIZE * scale, RT_SIZE * scale, GL_RGBA8);
    list[*n] = (Target){key, t};
    (*n)++;
    rt_log("gpu", "new %s target %X (%d of them)", depth ? "depth" : "colour", key, *n);
    if (depth) {
        float z = 0.0f;
        glClearTexImage(t, 0, GL_DEPTH_COMPONENT, GL_FLOAT, &z);
    } else {
        float c[4] = {0, 0, 0, 0};
        if (getenv("SSTHEH_GPU_RED")) { c[0] = 1; c[3] = 1; }
        glClearTexImage(t, 0, GL_RGBA, GL_FLOAT, c);
    }
    return t;
}

static GLuint find_color(u32 fbp) {
    for (int i = 0; i < ncolors; i++)
        if (colors[i].key == fbp) return colors[i].tex;
    return 0;
}

static GLuint fbo_for(GLuint color, GLuint depth) {
    for (int i = 0; i < nfbos; i++)
        if (fbos[i].color == color && fbos[i].depth == depth) return fbos[i].fbo;
    if (nfbos == (int)(sizeof fbos / sizeof *fbos)) {
        for (int i = 0; i < nfbos; i++) glDeleteFramebuffers(1, &fbos[i].fbo);
        nfbos = 0;
    }
    GLuint f;
    glCreateFramebuffers(1, &f);
    glNamedFramebufferTexture(f, GL_COLOR_ATTACHMENT0, color, 0);
    glNamedFramebufferTexture(f, GL_DEPTH_ATTACHMENT, depth, 0);
    if (glCheckNamedFramebufferStatus(f, GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        rt_log("gpu", "framebuffer %u is not complete", f);
    fbos[nfbos++] = (Fbo){color, depth, f};
    return f;
}

static Tex *tex_slot(u32 id) { return &texs[id % MAX_TEX]; }

/* ------------------------------------------------------------------------------- state */

static GLenum factor_c(int c, int one_minus) {
    switch (c) {
    case 0: return one_minus ? GL_ONE_MINUS_SRC_ALPHA : GL_SRC_ALPHA;
    case 1: return one_minus ? GL_ONE_MINUS_DST_ALPHA : GL_DST_ALPHA;
    default: return one_minus ? GL_ONE_MINUS_CONSTANT_COLOR : GL_CONSTANT_COLOR;
    }
}

typedef struct { GLenum s, d, op; } Blend;

static void blend_state(u32 al, Blend *b) {
    int A = al & 3, B = (al >> 2) & 3, C = (al >> 4) & 3, D = (al >> 6) & 3;
    if (A == 3) A = 2;
    if (B == 3) B = 2;
    if (D == 3) D = 2;
    if (C == 3) C = 2;
    GLenum s = GL_ONE, d = GL_ZERO, op = GL_FUNC_ADD;
    if (A == B) {
        s = D == 0 ? GL_ONE : GL_ZERO;
        d = D == 1 ? GL_ONE : GL_ZERO;
    } else if (A == 0 && B == 1 && D == 1) { s = factor_c(C, 0); d = factor_c(C, 1); }
    else if (A == 0 && B == 2 && D == 1) { s = factor_c(C, 0); d = GL_ONE; }
    else if (A == 0 && B == 2 && D == 2) { s = factor_c(C, 0); d = GL_ZERO; }
    else if (A == 0 && B == 2 && D == 0) { s = factor_c(C, 0); d = GL_ZERO; }
    else if (A == 1 && B == 0 && D == 0) { s = factor_c(C, 1); d = factor_c(C, 0); }
    else if (A == 1 && B == 2 && D == 0) { s = GL_ONE; d = factor_c(C, 0); }
    else if (A == 1 && B == 2 && D == 2) { s = GL_ZERO; d = factor_c(C, 0); }
    else if (A == 2 && B == 0 && D == 1) { s = factor_c(C, 0); d = GL_ONE; op = GL_FUNC_REVERSE_SUBTRACT; }
    else if (A == 2 && B == 1 && D == 0) { s = GL_ONE; d = factor_c(C, 0); op = GL_FUNC_SUBTRACT; }
    else if (A == 0 && B == 1 && D == 2) { s = factor_c(C, 0); d = factor_c(C, 0); op = GL_FUNC_SUBTRACT; }
    else if (A == 1 && B == 0 && D == 2) { s = factor_c(C, 0); d = factor_c(C, 0); op = GL_FUNC_REVERSE_SUBTRACT; }
    else if (A == 2 && B == 0 && D == 2) { s = d = GL_ZERO; }
    else if (A == 2 && B == 1 && D == 2) { s = d = GL_ZERO; }
    else { s = factor_c(C, 0); d = factor_c(C, 1); }
    b->s = s;
    b->d = d;
    b->op = op;
}

static void atest_masks(const GpuDraw *d, u32 *wmask, int *zwrite, int *shader_test) {
    *shader_test = d->atest[0];
    if (!d->atest[0] || d->atest[1] == 1) { *shader_test = 0; return; }
    int never = d->atest[1] == 0;
    switch (d->atest[3]) {
    case 0: break;
    case 1: *zwrite = 0; if (never) *shader_test = 0; break;
    case 2: if (never) { *wmask = 0; *shader_test = 0; } break;
    default: *zwrite = 0; *wmask &= 7; if (never) *shader_test = 0; break;
    }
}

static u64 cur_key = ~0ull;

static void apply_state(const GpuDraw *d) {
    if (debug_flags < 0) debug_flags = getenv("SSTHEH_GPU_DEBUG") ? atoi(getenv("SSTHEH_GPU_DEBUG")) : 0;
    u32 wmask = 0xF;
    if ((d->fbmsk & 0x000000FF) == 0x000000FF) wmask &= ~1u;
    if ((d->fbmsk & 0x0000FF00) == 0x0000FF00) wmask &= ~2u;
    if ((d->fbmsk & 0x00FF0000) == 0x00FF0000) wmask &= ~4u;
    if ((d->fbmsk & 0xFF000000) == 0xFF000000 || d->fpsm == 1) wmask &= ~8u;
    int zwrite = d->zwrite, shader_test;
    atest_masks(d, &wmask, &zwrite, &shader_test);
    u64 key = (d->abe ? (d->alpha & 0xFF) | 0x100 : 0) | (u64)d->ztst << 9 | (u64)zwrite << 12 | (u64)wmask << 13;
    if (key == cur_key) return;
    cur_key = key;
    if (d->abe && !(debug_flags & 2)) {
        Blend b;
        blend_state((u32)d->alpha, &b);
        glEnable(GL_BLEND);
        glBlendFuncSeparate(b.s, b.d, GL_ONE, GL_ZERO);
        glBlendEquationSeparate(b.op, GL_FUNC_ADD);
    } else glDisable(GL_BLEND);
    glColorMask(wmask & 1, (wmask >> 1) & 1, (wmask >> 2) & 1, (wmask >> 3) & 1);
    static const GLenum ops[5] = {GL_ALWAYS, GL_NEVER, GL_ALWAYS, GL_GEQUAL, GL_GREATER};
    glDepthFunc(ops[d->ztst < 5 ? d->ztst : 0]);
    glDepthMask(zwrite ? GL_TRUE : GL_FALSE);
}

/* ------------------------------------------------------------------------------- replay */

typedef struct {
    s32 mode[4], atest[4];
    float texsize[4], region[4];
} DrawUniforms;

static void gl_upload(GpuFrame *f) {
    if (!ctx || !f) return;
    for (u32 i = 0; i < f->nfrees; i++) {
        Tex *t = tex_slot(f->frees[i]);
        if (t->tex) glDeleteTextures(1, &t->tex);
        t->tex = 0;
        t->id = 0;
    }
    for (u32 i = 0; i < f->nuploads; i++) {
        GpuTexUpload *u = &f->uploads[i];
        Tex *t = tex_slot(u->id);
        if (t->tex && (t->w != u->w || t->h != u->h)) { glDeleteTextures(1, &t->tex); t->tex = 0; }
        t->id = u->id;
        t->w = u->w;
        t->h = u->h;
        if (!t->tex) t->tex = make_tex(u->w, u->h, GL_RGBA8);
        glTextureSubImage2D(t->tex, 0, 0, 0, (GLsizei)u->w, (GLsizei)u->h, GL_RGBA, GL_UNSIGNED_BYTE, u->px);
        up_bytes += (u64)u->w * u->h * 4;
        up_count++;
    }
}

static u64 shown_priv[5];
static int have_shown_priv;
static GLsync inflight[2];
static int inflight_at;

static void gl_render(GpuFrame *f, const GpuVertex *verts) {
    if (!ctx || !f || !f->ndraws) return;
    if (inflight[inflight_at]) {
        glClientWaitSync(inflight[inflight_at], GL_SYNC_FLUSH_COMMANDS_BIT, 1000000000ull);
        glDeleteSync(inflight[inflight_at]);
        inflight[inflight_at] = NULL;
    }
    memcpy(shown_priv, f->priv, sizeof shown_priv);
    have_shown_priv = 1;
    frame_no++;
    if (debug_flags < 0) debug_flags = getenv("SSTHEH_GPU_DEBUG") ? atoi(getenv("SSTHEH_GPU_DEBUG")) : 0;

    u32 vbytes = f->nverts * (u32)sizeof(GpuVertex);
    glNamedBufferData(vbo, vbytes, verts ? verts : f->verts, GL_STREAM_DRAW);
    glBindVertexArray(vao);
    glUseProgram(gs_prog);
    glEnable(GL_SCISSOR_TEST);
    if (debug_flags & 1) glDisable(GL_DEPTH_TEST); else glEnable(GL_DEPTH_TEST);
    cur_key = ~0ull;

    u32 cur_fbp = ~0u, cur_zbp = ~0u;
    GLuint last_tex = 0, last_smp = 0;
    u32 last_sc[4] = {~0u, ~0u, ~0u, ~0u};
    float last_fix = -1.0f;
    DrawUniforms last_u;
    int have_u = 0;
    for (u32 i = 0; i < f->ndraws; i++) {
        GpuDraw *d = &f->draws[i];
        GLuint color = target(colors, &ncolors, d->fbp, 0);
        GLuint depth = target(depths, &ndepths, d->zbp, 1);
        GLuint tex = white;
        float tw = (float)d->tw, th = (float)d->th;
        if (d->textured && d->tex >= 0) {
            Tex *t = tex_slot((u32)d->tex);
            if (t->id == (u32)d->tex && t->tex) { tex = t->tex; t->last_used = frame_no; }
        } else if (d->textured && d->tex < -1) {
            u32 src_fbp = (u32)(-2 - d->tex);
            GLuint src = find_color(src_fbp);
            tw = th = RT_SIZE;
            if (src && src == color) {
                if (!scratch) scratch = make_tex(RT_SIZE * scale, RT_SIZE * scale, GL_RGBA8);
                glCopyImageSubData(src, GL_TEXTURE_2D, 0, 0, 0, 0, scratch, GL_TEXTURE_2D, 0, 0, 0, 0,
                                   RT_SIZE * scale, RT_SIZE * scale, 1);
                tex = scratch;
            } else if (src) tex = src;
        }
        if (d->fbp != cur_fbp || d->zbp != cur_zbp) {
            glBindFramebuffer(GL_FRAMEBUFFER, fbo_for(color, depth));
            glViewport(0, 0, RT_SIZE * scale, RT_SIZE * scale);
            glUniform4f(0, RT_SIZE, RT_SIZE, 1.0f / RT_SIZE, 1.0f / RT_SIZE);
            cur_fbp = d->fbp;
            cur_zbp = d->zbp;
            last_tex = 0;
            last_smp = 0;
            last_sc[0] = ~0u;
            last_fix = -1.0f;
            have_u = 0;
        }
        apply_state(d);
        u32 sc[4] = {d->scissor[0] * scale, d->scissor[2] * scale,
                     (d->scissor[1] - d->scissor[0] + 1) * scale, (d->scissor[3] - d->scissor[2] + 1) * scale};
        if (memcmp(sc, last_sc, sizeof sc)) {
            glScissor((GLint)sc[0], (GLint)sc[1], (GLsizei)sc[2], (GLsizei)sc[3]);
            memcpy(last_sc, sc, sizeof sc);
        }
        float fix = (float)((d->alpha >> 32) & 0xFF) / 128.0f;
        if (fix != last_fix) { glBlendColor(fix, fix, fix, fix); last_fix = fix; }
        u32 wm = 0xF;
        int zw = d->zwrite, st;
        atest_masks(d, &wm, &zw, &st);
        DrawUniforms u = {{(s32)d->textured, (s32)d->tfx, (s32)d->tcc, (s32)d->fst},
                          {st, (s32)d->atest[1], (s32)d->atest[2], (s32)d->atest[3]},
                          {tw, th, 1.0f / tw, 1.0f / th},
                          {d->region[0], d->region[1], d->region[2], d->region[3]}};
        if (!have_u || memcmp(&u, &last_u, sizeof u)) {
            glUniform4i(1, u.mode[0], u.mode[1], u.mode[2], u.mode[3]);
            glUniform4i(2, u.atest[0], u.atest[1], u.atest[2], u.atest[3]);
            glUniform4f(3, u.texsize[0], u.texsize[1], u.texsize[2], u.texsize[3]);
            glUniform4f(4, u.region[0], u.region[1], u.region[2], u.region[3]);
            last_u = u;
            have_u = 1;
        }
        int smp = (d->filter ? 1 : 0) | (d->wrap_u ? 2 : 0) | (d->wrap_v ? 4 : 0);
        if (d->tex < -1) smp |= 6;
        if (tex != last_tex) { glBindTextureUnit(0, tex); last_tex = tex; }
        if (samplers[smp] != last_smp) { glBindSampler(0, samplers[smp]); last_smp = samplers[smp]; }
        glDrawArrays(GL_TRIANGLES, (GLint)d->first, (GLsizei)d->count);
    }
    inflight[inflight_at] = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
    inflight_at ^= 1;
    glFlush();
}

/* ------------------------------------------------------------------------------- display */

static GLuint displayed(float rect[4], int *pw, int *ph) {
    u64 pmode = have_shown_priv ? shown_priv[0] : gs_priv_read(0x0000);
    int circuit2 = (pmode & 2) != 0;
    u64 dispfb = have_shown_priv ? shown_priv[circuit2 ? 3 : 1] : gs_priv_read(circuit2 ? 0x0090 : 0x0070);
    u64 display = have_shown_priv ? shown_priv[circuit2 ? 4 : 2] : gs_priv_read(circuit2 ? 0x00A0 : 0x0080);
    u32 fbp = dispfb & 0x1FF, dbx = (dispfb >> 32) & 0x7FF, dby = (dispfb >> 43) & 0x7FF;
    u32 magh = ((display >> 23) & 15) + 1, dw = ((display >> 32) & 0xFFF) + 1, dh = ((display >> 44) & 0x7FF) + 1;
    int w = (int)(dw / magh), h = (int)dh;
    if (w <= 0 || h <= 0 || w > 1024 || h > 1024) { w = 640; h = 448; }
    if (kernel_ntsc() && h > 448) h = 448;
    *pw = w;
    *ph = h;
    rect[0] = (float)dbx / RT_SIZE;
    rect[1] = (float)dby / RT_SIZE;
    rect[2] = (float)(dbx + w) / RT_SIZE;
    rect[3] = (float)(dby + h) / RT_SIZE;
    return find_color(fbp);
}

static void gl_set_overlay(const u32 *px, int w, int h) {
    if (!ctx) return;
    if (!px) { overlay_on = 0; return; }
    if (!overlay || w != overlay_w || h != overlay_h) {
        if (overlay) glDeleteTextures(1, &overlay);
        overlay = make_tex((u32)w, (u32)h, GL_RGBA8);
        overlay_w = w;
        overlay_h = h;
    }
    glTextureSubImage2D(overlay, 0, 0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, px);
    overlay_on = 1;
}

static void gl_present(void) {
    if (!ctx || !win) return;
    int sw, sh;
    SDL_GetWindowSizeInPixels(win, &sw, &sh);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDisable(GL_SCISSOR_TEST);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_BLEND);
    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    glViewport(0, 0, sw, sh);
    glClearColor(0, 0, 0, 0);
    glClear(GL_COLOR_BUFFER_BIT);
    float rect[4];
    int w, h;
    GLuint src = displayed(rect, &w, &h);
    glBindVertexArray(vao);
    glUseProgram(present_prog);
    if (src) {
        float aspect = 4.0f / 3.0f, ww = (float)sw, wh = (float)sh;
        float vw = ww, vh = ww / aspect;
        if (vh > wh) { vh = wh; vw = wh * aspect; }
        glViewport((GLint)((ww - vw) / 2), (GLint)((wh - vh) / 2), (GLsizei)vw, (GLsizei)vh);
        glUniform4f(0, rect[0], rect[1], rect[2], rect[3]);
        float ratio = (float)(w * scale) / vw;
        int taps = ratio <= 1.0f ? 1 : (int)ceilf(ratio);
        if (taps > 4) taps = 4;
        glUniform4f(1, (rect[2] - rect[0]) / vw, (rect[3] - rect[1]) / vh, (float)taps, 0);
        glBindTextureUnit(0, src);
        glBindSampler(0, samplers[1 | 6]);
        glDrawArrays(GL_TRIANGLES, 0, 3);
    }
    if (overlay_on && overlay) {
        glViewport(4, sh - 4 - overlay_h, overlay_w, overlay_h);
        glEnable(GL_BLEND);
        glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ZERO);
        glBlendEquationSeparate(GL_FUNC_ADD, GL_FUNC_ADD);
        glUniform4f(0, 0, 0, 1, 1);
        glUniform4f(1, 0, 0, 1, 0);
        glBindTextureUnit(0, overlay);
        glBindSampler(0, samplers[6]);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        glDisable(GL_BLEND);
    }
#ifdef GPU_GL_TEST
    if (gpu_gl_before_swap) gpu_gl_before_swap();
#endif
    SDL_GL_SwapWindow(win);
    cur_key = ~0ull;
}

static int gl_read_display(u32 *out, int max_w, int max_h, int *pw, int *ph) {
    float rect[4];
    int w, h;
    GLuint src = displayed(rect, &w, &h);
    if (w > max_w) w = max_w;
    if (h > max_h) h = max_h;
    *pw = w;
    *ph = h;
    if (!src) { memset(out, 0, (size_t)w * h * 4); return 0; }
    u32 x0 = (u32)(rect[0] * RT_SIZE) * scale, y0 = (u32)(rect[1] * RT_SIZE) * scale;
    u32 bw = (u32)w * scale, bh = (u32)h * scale;
    u32 *px = malloc((size_t)bw * bh * 4);
    if (!px) { memset(out, 0, (size_t)w * h * 4); return 0; }
    glGetTextureSubImage(src, 0, (GLint)x0, (GLint)y0, 0, (GLsizei)bw, (GLsizei)bh, 1, GL_RGBA, GL_UNSIGNED_BYTE,
                         (GLsizei)((size_t)bw * bh * 4), px);
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++) out[y * w + x] = px[(u32)y * scale * bw + (u32)x * scale] | 0xFF000000u;
    free(px);
    return 0;
}

const GpuBackend gpu_backend_gl = {"opengl", gl_window_flags, gl_init, gl_upload, gl_render, gl_present,
                                   gl_set_vsync, gl_vsync, gl_read_display, gl_set_overlay};
