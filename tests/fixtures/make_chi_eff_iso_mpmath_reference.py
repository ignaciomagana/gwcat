"""Generate ``chi_eff_iso_mpmath_reference.json`` (GW-40i).

Independent high-precision reference for the isotropic uniform-magnitude
chi_eff prior.  It shares NO code with :mod:`gwcat.chi_eff_exact`: it evaluates
the defining convolution

    p(chi) = 1/(4 A1 A2) INT ln(A1/|x|) ln(A2/|chi - x|) dx,
    A_i = amax_i m_i / (m1 + m2),   |x| <= A1,  |chi - x| <= A2,

with mpmath's tanh-sinh quadrature.  Every piece between consecutive points of
{-A1, chi - A2, 0, chi, A1, chi + A2} is split at its midpoint, and each half is
parametrised by the distance t from its anchoring end so that the distances to
the log singularities are exact (no x - chi cancellation).  The working
precision starts at 40 digits and is raised by 30 until the quadrature's own
error estimate is below 1e-22 relative AND two successive precisions agree to
1e-22.  The inputs are the float64 values stored in the JSON, converted exactly.

Run (needs mpmath; ~1 min on 32 cores)::

    python tests/fixtures/make_chi_eff_iso_mpmath_reference.py
"""
import json
import os
import sys
from multiprocessing import Pool

import mpmath as mp
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "chi_eff_iso_mpmath_reference.json")
RTOL = mp.mpf("1e-22")


def _pieces(chi, A1, A2):
    lo = max(-A1, chi - A2)
    hi = min(A1, chi + A2)
    if not hi > lo:
        return None
    pts = sorted({lo, hi} | {p for p in (mp.mpf(0), chi) if lo < p < hi})
    return pts


def _integral(chi, A1, A2):
    pts = _pieces(chi, A1, A2)
    if pts is None:
        return mp.mpf(0), mp.mpf(0)
    tot = mp.mpf(0)
    err = mp.mpf(0)
    for p, pn in zip(pts[:-1], pts[1:]):
        half = (pn - p) / 2

        def fl(t, u0=p, v0=chi - p):       # x = p + t
            ax, ay = abs(u0 + t), abs(v0 - t)
            if ax == 0 or ay == 0:
                return mp.mpf(0)
            return mp.log(A1 / ax) * mp.log(A2 / ay)

        def fr(t, u1=pn, v1=chi - pn):     # x = pn - t
            ax, ay = abs(u1 - t), abs(v1 + t)
            if ax == 0 or ay == 0:
                return mp.mpf(0)
            return mp.log(A1 / ax) * mp.log(A2 / ay)
        for f in (fl, fr):
            v, e = mp.quad(f, [0, half], error=True, maxdegree=14)
            tot += v
            err += e
    return tot / (4 * A1 * A2), err / (4 * A1 * A2)


def reference(case):
    chi, m1, m2, a1, a2 = case
    dps, prev = 40, None
    while True:
        with mp.workdps(dps):
            M1, M2 = mp.mpf(m1), mp.mpf(m2)
            am1 = mp.mpf(a1)
            am2 = am1 if a2 is None else mp.mpf(a2)
            A1 = am1 * M1 / (M1 + M2)
            A2 = am2 * M2 / (M1 + M2)
            v, e = _integral(abs(mp.mpf(chi)), A1, A2)
            if v == 0:
                return "0"
            if (abs(e / v) < RTOL and prev is not None
                    and abs((v - prev) / v) < RTOL):
                return mp.nstr(v, 25)
            prev = v
        dps += 30
        if dps > 400:
            raise RuntimeError(f"no convergence for {case}")


def cases():
    out = []
    # (1) the edge grid: m1 = 1, m2 = q (so w1 = 1/(1+q) exactly as the q-API)
    for amax in (0.99, 0.998, 0.4, 0.05, 1.0):
        for q in (1.0, 1 - 1e-15, 1 - 1e-12, 1 - 1e-6, 0.999, 0.9, 0.7, 0.5,
                  0.3, 1 / 3.0, 0.1, 0.01, 1e-3, 1e-5, 1e-8):
            A1 = amax / (1 + q)
            A2 = amax * q / (1 + q)
            chis = [0.0, 1e-300, 1e-15, 1e-10, 1e-6, 1e-3, 0.5 * amax,
                    0.7 * amax]
            for k in (A2, A1, abs(A1 - A2)):
                for e in (0, 1e-14, 1e-12, 1e-9, 1e-6, 1e-3):
                    chis += [k * (1 + e), k * (1 - e)]
            for e in (1e-1, 1e-2, 1e-3, 1e-6, 1e-9, 1e-12, 1e-14):
                chis.append(amax * (1 - e))
            chis.append(float(np.nextafter(amax, 0)))
            for c in chis:
                if 0 <= c < amax:
                    out.append((float(c), 1.0, float(q), float(amax), None))
    # (2) random physical points: m1 in [3, 300], q log-uniform in [1e-3, 1]
    rng = np.random.default_rng(20260930)
    for _ in range(1200):
        amax = float(rng.choice([0.99, 0.998, 0.5]))
        m1 = float(rng.uniform(3.0, 300.0))
        q = float(10 ** rng.uniform(-3, 0))
        m2 = m1 * q
        if rng.uniform() < 0.3:          # masses in either order
            m1, m2 = m2, m1
        top = amax
        chi = float(rng.uniform(-top, top))
        out.append((chi, m1, m2, amax, None))
    # (3) different per-body ceilings (restricted secondary, and the reverse)
    for _ in range(300):
        a1 = 0.99
        a2 = float(rng.choice([0.05, 0.5, 0.998]))
        m1 = float(rng.uniform(5.0, 100.0))
        m2 = float(m1 * 10 ** rng.uniform(-2, 0))
        if rng.uniform() < 0.3:
            m1, m2 = m2, m1
        top = (a1 * m1 + a2 * m2) / (m1 + m2)
        chi = float(rng.uniform(-top, top))
        out.append((chi, m1, m2, a1, a2))
    return out


def main():
    cs = cases()
    nproc = int(os.environ.get("NPROC", os.cpu_count() or 1))
    with Pool(nproc) as pool:
        refs = pool.map(reference, cs, chunksize=4)
    rows = [{"chi_eff": c[0], "m1": c[1], "m2": c[2], "amax_1": c[3],
             "amax_2": c[4], "p": r} for c, r in zip(cs, refs)]
    doc = {
        "description": ("mpmath reference values of the isotropic "
                        "uniform-magnitude chi_eff prior p(chi_eff | m1, m2, "
                        "amax_1, amax_2); amax_2 null means amax_2 = amax_1. "
                        "Generated by make_chi_eff_iso_mpmath_reference.py."),
        "mpmath_version": mp.__version__,
        "rtol_reference": str(RTOL),
        "rows": rows,
    }
    with open(OUT, "w") as f:
        json.dump(doc, f, indent=0)
        f.write("\n")
    print(f"wrote {len(rows)} rows to {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
