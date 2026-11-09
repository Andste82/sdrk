"""The float32 arctangent ``atan2f`` of FreeBSD's msun (``e_atan2f.c`` and ``s_atanf.c``), the
algorithm of musl and of the Rust ``libm`` crate that SDRk's ``costas`` kernel calls for its QAM
phase detector (M44 section 3).

Every operation is one rounded float32 operation of ``f32model`` in the order of the C source (no
fused multiply-add), the branches compare the IEEE bits as the C source does, so the result equals
the bits of the C function compiled without FMA. Nothing here comes from SDRk's code: this is an
independent transcription of the published C source.

The C source carries this notice:

    Conversion to float by Ian Lance Taylor, Cygnus Support, ian@cygnus.com.

    Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.

    Developed at SunPro, a Sun Microsystems, Inc. business.
    Permission to use, copy, modify, and distribute this
    software is freely granted, provided that this notice
    is preserved.
"""

import struct

import numpy as np

from . import f32model as fm


def _bits(x):
    return struct.unpack("<I", struct.pack("<f", float(np.float32(x))))[0]


def _from_bits(b):
    return np.float32(struct.unpack("<f", struct.pack("<I", b & 0xFFFFFFFF))[0])


PI = _from_bits(0x40490FDB)
PI_LO = _from_bits(0xB3BBBD2E)
ATAN_HI = [_from_bits(b) for b in (0x3EED6338, 0x3F490FDA, 0x3F7B985E, 0x3FC90FDA)]
ATAN_LO = [_from_bits(b) for b in (0x31AC3769, 0x33222168, 0x33140FB4, 0x33A22168)]
A_T = [
    fm.f32(3.3333328366e-01),
    fm.f32(-1.9999158382e-01),
    fm.f32(1.4253635705e-01),
    fm.f32(-1.0648017377e-01),
    fm.f32(6.1687607318e-02),
]
_TINY = _from_bits(0x03800000)  # 0x1p-120
ONE = fm.f32(1.0)
TWO = fm.f32(2.0)
ONE_HALF = fm.f32(1.5)


def atanf(x):
    """``atanf`` of ``s_atanf.c`` on a float32 scalar."""
    x = fm.f32(x)
    ix = _bits(x)
    sign = ix >> 31
    ix &= 0x7FFFFFFF
    if ix >= 0x4C800000:  # |x| >= 2^26
        if np.isnan(x):
            return x
        z = fm.add(ATAN_HI[3], _TINY)
        return -z if sign else z
    if ix < 0x3EE00000:  # |x| < 0.4375
        if ix < 0x39800000:  # |x| < 2^-12
            return x
        ident = -1
    else:
        x = abs(x)
        if ix < 0x3F980000:  # |x| < 1.1875
            if ix < 0x3F300000:  # 7/16 <= |x| < 11/16
                ident = 0
                x = fm.div(fm.sub(fm.mul(TWO, x), ONE), fm.add(TWO, x))
            else:  # 11/16 <= |x| < 19/16
                ident = 1
                x = fm.div(fm.sub(x, ONE), fm.add(x, ONE))
        elif ix < 0x401C0000:  # |x| < 2.4375
            ident = 2
            x = fm.div(fm.sub(x, ONE_HALF), fm.add(ONE, fm.mul(ONE_HALF, x)))
        else:  # 2.4375 <= |x| < 2^26
            ident = 3
            x = fm.div(fm.f32(-1.0), x)
    z = fm.mul(x, x)
    w = fm.mul(z, z)
    s1 = fm.mul(z, fm.add(A_T[0], fm.mul(w, fm.add(A_T[2], fm.mul(w, A_T[4])))))
    s2 = fm.mul(w, fm.add(A_T[1], fm.mul(w, A_T[3])))
    if ident < 0:
        return fm.sub(x, fm.mul(x, fm.add(s1, s2)))
    z = fm.sub(ATAN_HI[ident], fm.sub(fm.sub(fm.mul(x, fm.add(s1, s2)), ATAN_LO[ident]), x))
    return -z if sign else z


def atan2f(y, x):
    """``atan2f(y, x)`` of ``e_atan2f.c`` on float32 scalars."""
    y, x = fm.f32(y), fm.f32(x)
    if np.isnan(x) or np.isnan(y):
        return fm.add(x, y)
    ix, iy = _bits(x), _bits(y)
    if ix == 0x3F800000:  # x = 1.0
        return atanf(y)
    m = ((iy >> 31) & 1) | ((ix >> 30) & 2)  # 2 sign(x) + sign(y)
    ix &= 0x7FFFFFFF
    iy &= 0x7FFFFFFF
    half_pi = fm.div(PI, TWO)
    if iy == 0:
        return (y, y, PI, -PI)[m]
    if ix == 0:
        return -half_pi if m & 1 else half_pi
    if ix == 0x7F800000:
        if iy == 0x7F800000:
            q = fm.div(PI, fm.f32(4.0))
            tq = fm.div(fm.mul(fm.f32(3.0), PI), fm.f32(4.0))
            return (q, -q, tq, -tq)[m]
        return (fm.f32(0.0), fm.f32(-0.0), PI, -PI)[m]
    if ix + (26 << 23) < iy or iy == 0x7F800000:  # |y / x| > 2^26
        return -half_pi if m & 1 else half_pi
    if (m & 2) and iy + (26 << 23) < ix:  # |y / x| < 2^-26, x < 0
        z = fm.f32(0.0)
    else:
        z = atanf(abs(fm.div(y, x)))
    if m == 0:
        return z
    if m == 1:
        return -z
    if m == 2:
        return fm.sub(PI, fm.sub(z, PI_LO))
    return fm.sub(fm.sub(z, PI_LO), PI)
