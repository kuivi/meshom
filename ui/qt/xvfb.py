"""A virtual X display for running the Qt prototype headless on Levante (tests and measurements only).

conda-forge's Xvfb (`xorg-x11-server-xvfb-conda-x86_64`, a CentOS-7 build) links libcrypto.so.10 for SHA1 only;
Levante (RHEL 8) has libcrypto.so.1.1, so a three-function shim forwards SHA1_* to it (built once with gcc into
.tmp/ui/xlib). Rendering inside is Mesa llvmpipe (software OpenGL on the CPU).
"""
import os
import subprocess
import sys
import time
from pathlib import Path

SHIM_C = r"""
#include <dlfcn.h>
#include <stddef.h>
typedef int (*fi)(void *); typedef int (*fu)(void *, const void *, size_t); typedef int (*ff)(unsigned char *, void *);
static void *h;
static void *sym(const char *n) { if (!h) h = dlopen("libcrypto.so.1.1", RTLD_NOW | RTLD_LOCAL); return dlsym(h, n); }
int SHA1_Init(void *c) { return ((fi)sym("SHA1_Init"))(c); }
int SHA1_Update(void *c, const void *d, size_t n) { return ((fu)sym("SHA1_Update"))(c, d, n); }
int SHA1_Final(unsigned char *md, void *c) { return ((ff)sym("SHA1_Final"))(md, c); }
"""
XVFB = Path(sys.prefix) / "x86_64-conda-linux-gnu" / "sysroot" / "usr" / "bin" / "Xvfb"
LIB = Path(".tmp/ui/xlib")


def available() -> bool:
    return XVFB.exists()


def shim() -> Path:
    so = LIB / "libcrypto.so.10"
    if not so.exists():
        LIB.mkdir(parents=True, exist_ok=True)
        (LIB / "shim.c").write_text(SHIM_C)
        (LIB / "shim.map").write_text("libcrypto.so.10 { global: SHA1_Init; SHA1_Update; SHA1_Final; local: *; };\n")
        subprocess.run(["gcc", "-shared", "-fPIC", "-o", str(so), str(LIB / "shim.c"),
                        f"-Wl,--version-script={LIB / 'shim.map'}", "-Wl,-soname,libcrypto.so.10", "-ldl"], check=True)
    return so


def start(display: int = 95, size: str = "1600x1000x24") -> subprocess.Popen:
    """Start Xvfb on :display and point DISPLAY / QT_QPA_PLATFORM at it. With -terminate the server exits when its
    last client disconnects: Qt closes its connection only at interpreter exit, and killing Xvfb earlier makes Qt
    abort with "X11 connection broke" and exit code 1."""
    env = dict(os.environ, LD_LIBRARY_PATH=f"{shim().parent.resolve()}:{os.environ.get('LD_LIBRARY_PATH', '')}")
    p = subprocess.Popen([str(XVFB), f":{display}", "-screen", "0", size, "-nolisten", "tcp", "+extension", "GLX",
                          "-terminate"],
                         env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(50):
        if Path(f"/tmp/.X11-unix/X{display}").exists():
            break
        time.sleep(0.1)
    os.environ["DISPLAY"] = f":{display}"
    os.environ["QT_QPA_PLATFORM"] = "xcb"
    return p
