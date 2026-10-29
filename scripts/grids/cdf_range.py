"""Read chosen variables of a remote classic netCDF (CDF-1/2) file with HTTP range requests.
python cdf_range.py URL OUT.npz var1 var2 ...   (no var: list the header)"""
import struct, sys, urllib.request
import numpy as np

URL = sys.argv[1]
def rng(a, n):
    req = urllib.request.Request(URL, headers={"Range": f"bytes={a}-{a + n - 1}"})
    return urllib.request.urlopen(req, timeout=300).read()

H = rng(0, 1 << 20)
pos = 0
def i32():
    global pos; v = struct.unpack(">i", H[pos:pos + 4])[0]; pos += 4; return v
def i64():
    global pos; v = struct.unpack(">q", H[pos:pos + 8])[0]; pos += 8; return v
def name():
    global pos; n = i32(); s = H[pos:pos + n].decode(); pos += (n + 3) // 4 * 4; return s
SIZE = {1: 1, 2: 1, 3: 2, 4: 4, 5: 4, 6: 8}
DT = {1: "i1", 2: "S1", 3: ">i2", 4: ">i4", 5: ">f4", 6: ">f8"}
def atts():
    global pos
    tag, n = i32(), i32()
    for _ in range(n):
        name(); t = i32(); m = i32(); pos += (m * SIZE[t] + 3) // 4 * 4
assert H[:3] == b"CDF"; ver = H[3]; pos = 4; i32()
tag, n = i32(), i32(); dims = [(name(), i32()) for _ in range(n)]
atts()
tag, n = i32(), i32(); var = {}
for _ in range(n):
    nm = name(); k = i32(); ids = [i32() for _ in range(k)]; atts(); t = i32(); vs = i32()
    begin = i64() if ver == 2 else i32()
    var[nm] = ([dims[d] for d in ids], t, begin)
if len(sys.argv) < 4:
    for k, v in var.items(): print(k, v)
    sys.exit()
out = {}
for nm in sys.argv[3:]:
    shape = [max(d[1], 1) for d in var[nm][0]]; t = var[nm][1]
    nbytes = int(np.prod(shape)) * SIZE[t]
    out[nm] = np.frombuffer(rng(var[nm][2], nbytes), dtype=DT[t]).reshape(shape)
    print(nm, shape, out[nm].dtype)
np.savez_compressed(sys.argv[2], **out)
