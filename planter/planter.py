#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""おしゃれな植木鉢の 3D モデルをプログラムで生成する（純 Python / 依存パッケージなし）。

Generates a watertight, 3D-printable planter (and a matching saucer) from a
handful of parameters, then writes STL / OBJ files and an optional SVG preview.

    python3 planter.py --style fluted --preview preview.svg

The solid is assembled ring by ring so the result is guaranteed manifold:

    rim (annulus)
    ├── outer wall   z: 0 -> H       radius R(theta, z)
    ├── inner wall   z: H -> floor   radius R(theta, z) - wall_offset
    ├── floor top    (annulus around the drain hole)
    ├── drain hole   (cylindrical wall)
    └── bottom       (annulus, z = 0)

All dimensions are millimetres.
"""

from __future__ import annotations

import argparse
import math
import os
import struct
import sys
from dataclasses import dataclass, replace, fields

# --------------------------------------------------------------------------
# parameters
# --------------------------------------------------------------------------


@dataclass
class Params:
    """Every knob that shapes the pot. Millimetres / degrees."""

    # overall silhouette
    height: float = 132.0
    bottom_radius: float = 39.0
    top_radius: float = 62.0
    taper_curve: float = 0.45   # 0 = straight cone, 1 = S-curve
    belly: float = 0.045        # bulge, as a fraction of top_radius
    lip: float = 2.6            # outward flare at the rim
    lip_frac: float = 0.09      # share of the height the flare occupies
    foot_inset: float = 3.0     # how far the base tucks in ("floating" foot)
    foot_frac: float = 0.11

    # surface decoration
    flute_count: int = 26       # vertical grooves
    flute_depth: float = 2.4
    flute_sharpness: float = 1.8  # 1 = sine, higher = crisper grooves
    twist: float = 26.0         # degrees of groove rotation over the height
    wave_count: float = 0.0     # horizontal ripples
    wave_depth: float = 0.0
    squircle: float = 2.0       # 2 = round, 4 = rounded square section
    fade_bottom: float = 0.09   # grooves fade out near the base ...
    fade_top: float = 0.16      # ... leaving a smooth collar under the rim

    # material
    wall: float = 2.2           # minimum wall thickness
    floor: float = 5.0          # floor thickness
    drain_radius: float = 7.0   # 0 = no drainage hole (used for the saucer)
    smooth_inside: bool = True  # keep the cavity smooth, grooves outside only

    # tessellation
    segments: int = 192
    rows: int = 120


STYLE_PRESETS = {
    # classic twisted flutes
    "fluted": {},
    # many fine vertical ribs, no twist -- the "concrete pot" look
    "ribbed": dict(flute_count=36, flute_depth=1.5, flute_sharpness=2.2,
                   twist=0.0, belly=0.02, lip=2.0, segments=288,
                   fade_top=0.10),
    # low-poly twisted prism
    "faceted": dict(segments=9, flute_count=0, flute_depth=0.0, twist=22.0,
                    lip=1.5, lip_frac=0.08, foot_inset=2.5, belly=0.0,
                    taper_curve=0.0, rows=140),
    # horizontal ripples
    "wave": dict(flute_count=0, flute_depth=0.0, twist=0.0, wave_count=7.0,
                 wave_depth=1.6, belly=0.02),
    # rounded-square section, undecorated
    "squircle": dict(squircle=4.0, flute_count=0, flute_depth=0.0, twist=0.0,
                     lip=2.5, belly=0.0, taper_curve=0.5),
    # plain surface of revolution
    "smooth": dict(flute_count=0, flute_depth=0.0, twist=0.0, belly=0.04),
}


# --------------------------------------------------------------------------
# profile maths
# --------------------------------------------------------------------------


def _smootherstep(x: float) -> float:
    x = min(1.0, max(0.0, x))
    return x * x * x * (x * (x * 6.0 - 15.0) + 10.0)


def _fade(p: Params, t: float) -> float:
    """0 at the very bottom and the very rim, 1 in between."""
    a = _smootherstep(t / p.fade_bottom) if p.fade_bottom > 0 else 1.0
    b = _smootherstep((1.0 - t) / p.fade_top) if p.fade_top > 0 else 1.0
    return a * b


def base_profile(p: Params, t: float) -> float:
    """Radius of the undecorated body of revolution at height fraction t."""
    t = min(1.0, max(0.0, t))
    k = (1.0 - p.taper_curve) * t + p.taper_curve * _smootherstep(t)
    r = p.bottom_radius + (p.top_radius - p.bottom_radius) * k
    # bump with zero slope at both ends, so the belly never steepens the base
    r += p.belly * p.top_radius * math.sin(math.pi * t) ** 1.4
    if p.lip and p.lip_frac > 0.0 and t > 1.0 - p.lip_frac:
        u = (t - (1.0 - p.lip_frac)) / p.lip_frac
        r += p.lip * _smootherstep(u)
    if p.foot_inset and p.foot_frac > 0.0 and t < p.foot_frac:
        r -= p.foot_inset * (1.0 - _smootherstep(t / p.foot_frac))
    if p.wave_depth and p.wave_count:
        r += p.wave_depth * math.sin(2.0 * math.pi * p.wave_count * t)
    return r


def section_mod(p: Params, theta: float) -> float:
    """Cross-section multiplier: 1 for a circle, bulging for a squircle."""
    n = p.squircle
    if abs(n - 2.0) < 1e-9:
        return 1.0
    c = abs(math.cos(theta))
    s = abs(math.sin(theta))
    return (c ** n + s ** n) ** (-1.0 / n)


def base_radius(p: Params, theta: float, t: float) -> float:
    return base_profile(p, t) * section_mod(p, theta)


def groove(p: Params, theta: float, t: float) -> float:
    """Depth removed by the flutes at (theta, t); always >= 0."""
    if not p.flute_count or p.flute_depth <= 0.0:
        return 0.0
    phase = p.flute_count * (theta - math.radians(p.twist) * t)
    v = 0.5 + 0.5 * math.cos(phase)
    return p.flute_depth * _fade(p, t) * (v ** p.flute_sharpness)


def outer_radius(p: Params, theta: float, t: float) -> float:
    return base_radius(p, theta, t) - groove(p, theta, t)


def _offset_factor(p: Params, theta: float, t: float) -> float:
    """Radial distance needed to move one unit along the surface normal.

    For a surface r = R(theta, z) the unit normal is grad(r - R)/|grad|, so a
    radial step d covers d/|grad| of true thickness.  Compensating keeps the
    wall thickness honest on tapered and rippled bodies.
    """
    h = 1e-3
    dt = 1e-3
    r = base_radius(p, theta, t)
    d_theta = (base_radius(p, theta + h, t) - base_radius(p, theta - h, t)) / (2.0 * h)
    t_lo, t_hi = max(0.0, t - dt), min(1.0, t + dt)
    dz = (t_hi - t_lo) * p.height
    d_z = (base_radius(p, theta, t_hi) - base_radius(p, theta, t_lo)) / dz if dz > 0 else 0.0
    grad = math.sqrt(1.0 + (d_theta / r) ** 2 + d_z ** 2)
    return min(2.5, grad)


def inner_radius(p: Params, theta: float, t: float) -> float:
    """Cavity radius.

    With ``smooth_inside`` the cavity ignores the grooves and the wall is
    thickened by the groove depth instead, so the thinnest spot (the bottom of
    a groove) still measures ``wall``.
    """
    if p.smooth_inside:
        need = p.wall + (p.flute_depth * _fade(p, t) if p.flute_count else 0.0)
        return base_radius(p, theta, t) - need * _offset_factor(p, theta, t)
    return outer_radius(p, theta, t) - p.wall * _offset_factor(p, theta, t)


# --------------------------------------------------------------------------
# mesh
# --------------------------------------------------------------------------


class Mesh:
    """Triangle soup with outward-facing winding."""

    def __init__(self, name: str = "mesh"):
        self.name = name
        self.v: list[tuple[float, float, float]] = []
        self.f: list[tuple[int, int, int]] = []

    # -- construction ------------------------------------------------------
    def add_vertex(self, x: float, y: float, z: float) -> int:
        self.v.append((x, y, z))
        return len(self.v) - 1

    def add_ring(self, pts) -> list[int]:
        start = len(self.v)
        self.v.extend(pts)
        return list(range(start, len(self.v)))

    def bridge(self, a: list[int], b: list[int]) -> None:
        """Sew ring ``a`` to ring ``b``.

        The normal comes out along tangent(+theta) x (a -> b), so the caller
        picks the orientation by choosing which ring goes first.
        """
        n = len(a)
        for i in range(n):
            j = (i + 1) % n
            self.f.append((a[i], a[j], b[i]))
            self.f.append((a[j], b[j], b[i]))

    def cap(self, ring: list[int], center: int, up: bool) -> None:
        n = len(ring)
        for i in range(n):
            j = (i + 1) % n
            if up:
                self.f.append((center, ring[i], ring[j]))
            else:
                self.f.append((center, ring[j], ring[i]))

    def translate(self, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0) -> None:
        self.v = [(x + dx, y + dy, z + dz) for (x, y, z) in self.v]

    # -- measurement -------------------------------------------------------
    def volume(self) -> float:
        """Signed volume in mm^3 (positive when the winding is outward)."""
        total = 0.0
        for a, b, c in self.f:
            ax, ay, az = self.v[a]
            bx, by, bz = self.v[b]
            cx, cy, cz = self.v[c]
            total += (ax * (by * cz - bz * cy)
                      - ay * (bx * cz - bz * cx)
                      + az * (bx * cy - by * cx))
        return total / 6.0

    def area(self) -> float:
        total = 0.0
        for a, b, c in self.f:
            ax, ay, az = self.v[a]
            bx, by, bz = self.v[b]
            cx, cy, cz = self.v[c]
            ux, uy, uz = bx - ax, by - ay, bz - az
            vx, vy, vz = cx - ax, cy - ay, cz - az
            nx, ny, nz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
            total += 0.5 * math.sqrt(nx * nx + ny * ny + nz * nz)
        return total

    def bounds(self):
        xs = [p[0] for p in self.v]
        ys = [p[1] for p in self.v]
        zs = [p[2] for p in self.v]
        return (min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs))

    def check(self) -> list[str]:
        """Manifold test: every directed edge must appear exactly once."""
        problems = []
        seen = {}
        for a, b, c in self.f:
            if a == b or b == c or a == c:
                problems.append("degenerate triangle")
                break
            for e in ((a, b), (b, c), (c, a)):
                if e in seen:
                    problems.append(f"edge {e} used twice with the same direction")
                    return problems
                seen[e] = True
        for (a, b) in seen:
            if (b, a) not in seen:
                problems.append(f"boundary edge {(a, b)} has no twin (not watertight)")
                break
        if self.volume() <= 0.0:
            problems.append("non-positive volume: normals point inward")
        return problems


# --------------------------------------------------------------------------
# planter construction
# --------------------------------------------------------------------------


def _validate(p: Params) -> None:
    if p.segments < 3:
        raise ValueError("--segments は 3 以上にしてください")
    if p.rows < 4:
        raise ValueError("--rows は 4 以上にしてください")
    if p.floor >= p.height * 0.5:
        raise ValueError("--floor が高さに対して厚すぎます")
    if p.wall <= 0.0:
        raise ValueError("--wall は正の値にしてください")


def build_planter(p: Params) -> tuple[Mesh, dict]:
    """Assemble the pot. Returns the mesh plus a few derived measurements."""
    _validate(p)
    n = p.segments
    thetas = [2.0 * math.pi * i / n for i in range(n)]
    t_floor = p.floor / p.height
    mesh = Mesh("planter")

    def outer_ring(t: float) -> list[int]:
        z = t * p.height
        return mesh.add_ring([
            (outer_radius(p, th, t) * math.cos(th),
             outer_radius(p, th, t) * math.sin(th), z) for th in thetas])

    def inner_ring(t: float) -> list[int]:
        z = t * p.height
        return mesh.add_ring([
            (inner_radius(p, th, t) * math.cos(th),
             inner_radius(p, th, t) * math.sin(th), z) for th in thetas])

    def flat_ring(radius: float, z: float) -> list[int]:
        return mesh.add_ring([(radius * math.cos(th), radius * math.sin(th), z)
                              for th in thetas])

    # sanity: the cavity must stay inside the body and still hold something
    probe_rows = [t_floor + (1.0 - t_floor) * i / 40.0 for i in range(41)]
    probe_th = [2.0 * math.pi * i / 180.0 for i in range(180)]
    worst = min(outer_radius(p, th, t) - inner_radius(p, th, t)
                for t in probe_rows for th in probe_th)
    narrowest = min(inner_radius(p, th, t) for t in probe_rows for th in probe_th)
    if narrowest < 1.0:
        raise ValueError(
            f"壁が厚すぎて内部空間が残りません (内半径 {narrowest:.1f}mm)。"
            "--wall を小さくするか鉢を大きくしてください。")
    if worst < 0.4:
        raise ValueError(
            f"肉厚を確保できません (最薄部 {worst:.2f}mm)。"
            "--flute-depth を浅くするか --wall を見直してください。")
    r_floor_in = min(inner_radius(p, th, t_floor) for th in thetas)
    if p.drain_radius > 0.0 and p.drain_radius > r_floor_in - 3.0:
        raise ValueError(
            f"排水穴が大きすぎます (床の内半径 {r_floor_in:.1f}mm)。"
            "--drain-diameter を小さくしてください。")

    # 1. outer wall, bottom -> rim
    rings = [outer_ring(i / p.rows) for i in range(p.rows + 1)]
    bottom_outer = rings[0]
    for lo, hi in zip(rings, rings[1:]):
        mesh.bridge(lo, hi)

    # 2. rim: outer top -> inner top (normal +z)
    top_inner = inner_ring(1.0)
    mesh.bridge(rings[-1], top_inner)

    # 3. inner wall, rim -> floor (walk downwards so normals face the axis)
    inner_rings = [top_inner]
    capacity = 0.0
    prev_area = None
    prev_z = None
    for i in range(1, p.rows + 1):
        t = 1.0 - (1.0 - t_floor) * i / p.rows
        inner_rings.append(inner_ring(t))
    for hi, lo in zip(inner_rings, inner_rings[1:]):
        mesh.bridge(hi, lo)

    # soil capacity: integrate the cavity cross-section over its height
    for ring in inner_rings:
        pts = [mesh.v[i] for i in ring]
        a = 0.0
        for i in range(n):
            x1, y1, _ = pts[i]
            x2, y2, _ = pts[(i + 1) % n]
            a += x1 * y2 - x2 * y1
        a = abs(a) * 0.5
        z = pts[0][2]
        if prev_area is not None:
            capacity += 0.5 * (a + prev_area) * abs(z - prev_z)
        prev_area, prev_z = a, z

    floor_ring = inner_rings[-1]

    if p.drain_radius > 0.0:
        # 4. floor top (annulus) / 5. drain wall / 6. bottom (annulus)
        drain_top = flat_ring(p.drain_radius, p.floor)
        drain_bottom = flat_ring(p.drain_radius, 0.0)
        mesh.bridge(floor_ring, drain_top)
        mesh.bridge(drain_top, drain_bottom)
        mesh.bridge(drain_bottom, bottom_outer)
    else:
        # closed floor: two discs
        mesh.cap(floor_ring, mesh.add_vertex(0.0, 0.0, p.floor), up=True)
        mesh.cap(bottom_outer, mesh.add_vertex(0.0, 0.0, 0.0), up=False)

    info = {
        "height": p.height,
        "max_diameter": 2.0 * max(math.hypot(x, y) for x, y, _ in mesh.v),
        "foot_diameter": 2.0 * max(math.hypot(mesh.v[i][0], mesh.v[i][1])
                                   for i in bottom_outer),
        "material_cm3": mesh.volume() / 1000.0,
        "capacity_ml": capacity / 1000.0,
        "min_wall": worst,
        "triangles": len(mesh.f),
    }
    return mesh, info


def make_saucer(p: Params, clearance: float = 3.0, height: float | None = None) -> Params:
    """Derive a matching saucer that the planter drops into."""
    height = height if height is not None else max(18.0, p.height * 0.17)
    thetas = [2.0 * math.pi * i / max(24, p.segments) for i in range(max(24, p.segments))]
    foot = max(outer_radius(p, th, 0.0) for th in thetas)

    s = replace(
        p,
        height=height,
        wall=max(1.8, p.wall * 0.85),
        floor=max(3.0, p.floor * 0.7),
        drain_radius=0.0,
        belly=0.0,
        lip=p.lip * 0.35,
        lip_frac=0.30,
        foot_inset=1.5,
        foot_frac=0.22,
        taper_curve=0.2,
        flute_depth=p.flute_depth * 0.65,
        twist=p.twist * 0.5,
        fade_bottom=0.18,
        fade_top=0.12,
        bottom_radius=foot + clearance + 3.0,
        top_radius=foot + clearance + 3.0 + height * 0.34,
    )
    # nudge the size until the pot's foot really fits inside the saucer
    need = foot + clearance
    for _ in range(8):
        t_floor = s.floor / s.height
        have = min(inner_radius(s, th, t_floor) for th in thetas)
        if have >= need:
            break
        grow = need - have + 0.2
        s = replace(s, bottom_radius=s.bottom_radius + grow,
                    top_radius=s.top_radius + grow)
    return s


# --------------------------------------------------------------------------
# exporters
# --------------------------------------------------------------------------


def _face_normal(mesh: Mesh, face):
    ax, ay, az = mesh.v[face[0]]
    bx, by, bz = mesh.v[face[1]]
    cx, cy, cz = mesh.v[face[2]]
    ux, uy, uz = bx - ax, by - ay, bz - az
    vx, vy, vz = cx - ax, cy - ay, cz - az
    nx, ny, nz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
    ln = math.sqrt(nx * nx + ny * ny + nz * nz)
    if ln == 0.0:
        return 0.0, 0.0, 0.0
    return nx / ln, ny / ln, nz / ln


def write_stl(mesh: Mesh, path: str) -> None:
    with open(path, "wb") as fh:
        fh.write(f"{mesh.name} - generated by planter.py".ljust(80, " ")[:80].encode())
        fh.write(struct.pack("<I", len(mesh.f)))
        for face in mesh.f:
            nx, ny, nz = _face_normal(mesh, face)
            a, b, c = (mesh.v[i] for i in face)
            fh.write(struct.pack("<12fH", nx, ny, nz,
                                 a[0], a[1], a[2], b[0], b[1], b[2],
                                 c[0], c[1], c[2], 0))


def write_stl_ascii(mesh: Mesh, path: str) -> None:
    with open(path, "w") as fh:
        fh.write(f"solid {mesh.name}\n")
        for face in mesh.f:
            nx, ny, nz = _face_normal(mesh, face)
            fh.write(f"  facet normal {nx:.6e} {ny:.6e} {nz:.6e}\n    outer loop\n")
            for i in face:
                x, y, z = mesh.v[i]
                fh.write(f"      vertex {x:.6e} {y:.6e} {z:.6e}\n")
            fh.write("    endloop\n  endfacet\n")
        fh.write(f"endsolid {mesh.name}\n")


def write_obj(mesh: Mesh, path: str) -> None:
    with open(path, "w") as fh:
        fh.write(f"# {mesh.name} - generated by planter.py\n")
        for x, y, z in mesh.v:
            fh.write(f"v {x:.4f} {y:.4f} {z:.4f}\n")
        for a, b, c in mesh.f:
            fh.write(f"f {a + 1} {b + 1} {c + 1}\n")


# --------------------------------------------------------------------------
# SVG preview (orthographic, painter's algorithm, flat Lambert shading)
# --------------------------------------------------------------------------


def _hex_to_rgb(s: str):
    s = s.lstrip("#")
    return tuple(int(s[i:i + 2], 16) for i in (0, 2, 4))


def _vertex_normals(mesh: Mesh, crease: float = 26.0):
    """Averaged vertex normals, ignoring contributions across sharp creases."""
    acc = [[0.0, 0.0, 0.0] for _ in mesh.v]
    face_normals = []
    for face in mesh.f:
        n = _face_normal(mesh, face)
        face_normals.append(n)
        for i in face:
            acc[i][0] += n[0]
            acc[i][1] += n[1]
            acc[i][2] += n[2]
    out = []
    for a in acc:
        ln = math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])
        out.append((a[0] / ln, a[1] / ln, a[2] / ln) if ln else (0.0, 0.0, 1.0))
    return out, face_normals, math.cos(math.radians(crease))


def render_svg(meshes, path: str, azimuth: float = 35.0,
               elevation: float = 20.0, width: int = 900) -> None:
    """Paint (mesh, colour) pairs into a flat-shaded SVG, far faces first."""
    az, el = math.radians(azimuth), math.radians(elevation)
    view = (math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el))
    right = (-math.sin(az), math.cos(az), 0.0)
    up = (view[1] * right[2] - view[2] * right[1],
          view[2] * right[0] - view[0] * right[2],
          view[0] * right[1] - view[1] * right[0])
    light = [-0.45 * right[i] + 0.55 * up[i] + 0.70 * view[i] for i in range(3)]
    ln = math.sqrt(sum(c * c for c in light))
    light = [c / ln for c in light]

    polys = []
    for mesh, base in meshes:
        r0, g0, b0 = _hex_to_rgb(base)
        vnormals, fnormals, crease_cos = _vertex_normals(mesh)
        for fi, face in enumerate(mesh.f):
            fx, fy, fz = fnormals[fi]
            if fx * view[0] + fy * view[1] + fz * view[2] <= 0.02:
                continue  # back-facing
            # smooth shading: average the vertex normals that share this face's
            # orientation, so curvature reads as a gradient instead of banding
            nx = ny = nz = 0.0
            for i in face:
                vx, vy, vz = vnormals[i]
                if vx * fx + vy * fy + vz * fz >= crease_cos:
                    nx, ny, nz = nx + vx, ny + vy, nz + vz
                else:
                    nx, ny, nz = nx + fx, ny + fy, nz + fz
            ln = math.sqrt(nx * nx + ny * ny + nz * nz)
            nx, ny, nz = (nx / ln, ny / ln, nz / ln) if ln else (fx, fy, fz)
            pts = [mesh.v[i] for i in face]
            sx = [p[0] * right[0] + p[1] * right[1] + p[2] * right[2] for p in pts]
            sy = [p[0] * up[0] + p[1] * up[1] + p[2] * up[2] for p in pts]
            depth = sum(p[0] * view[0] + p[1] * view[1] + p[2] * view[2] for p in pts) / 3.0
            lam = max(0.0, nx * light[0] + ny * light[1] + nz * light[2])
            shade = 0.30 + 0.62 * lam + 0.16 * lam ** 6
            col = "#%02x%02x%02x" % tuple(min(255, int(c * shade)) for c in (r0, g0, b0))
            polys.append((depth, sx, sy, col))

    if not polys:
        raise ValueError("プレビューに描画できる面がありません")
    polys.sort(key=lambda q: q[0])

    xs = [x for q in polys for x in q[1]]
    ys = [y for q in polys for y in q[2]]
    pad = 0.05 * (max(xs) - min(xs))
    x0, x1 = min(xs) - pad, max(xs) + pad
    y0, y1 = min(ys) - pad, max(ys) + pad
    scale = width / (x1 - x0)
    height = int((y1 - y0) * scale)

    def px(x):
        return (x - x0) * scale

    def py(y):
        return (y1 - y) * scale

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
           f'viewBox="0 0 {width} {height}">',
           '<defs><radialGradient id="bg" cx="50%" cy="35%" r="75%">'
           '<stop offset="0%" stop-color="#f7f4ef"/>'
           '<stop offset="100%" stop-color="#ddd7cd"/></radialGradient>'
           '<radialGradient id="sh" cx="50%" cy="50%" r="50%">'
           '<stop offset="0%" stop-color="#000" stop-opacity="0.30"/>'
           '<stop offset="100%" stop-color="#000" stop-opacity="0"/>'
           '</radialGradient></defs>',
           f'<rect width="{width}" height="{height}" fill="url(#bg)"/>']

    # a soft contact shadow under each object
    for mesh, _ in meshes:
        lo, hi = mesh.bounds()
        cx = (lo[0] + hi[0]) / 2.0
        cy = (lo[1] + hi[1]) / 2.0
        rad = max(hi[0] - lo[0], hi[1] - lo[1]) / 2.0 * 1.25
        gx = cx * right[0] + cy * right[1]
        gy = cx * up[0] + cy * up[1]
        out.append(f'<ellipse cx="{px(gx):.1f}" cy="{py(gy):.1f}" '
                   f'rx="{rad * scale:.1f}" ry="{rad * scale * math.sin(el):.1f}" '
                   f'fill="url(#sh)"/>')

    for _, sx, sy, col in polys:
        pts = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in zip(sx, sy))
        out.append(f'<polygon points="{pts}" fill="{col}" stroke="{col}" stroke-width="0.6"/>')
    out.append("</svg>")

    with open(path, "w") as fh:
        fh.write("\n".join(out))


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="おしゃれな植木鉢の 3D モデル (STL/OBJ) を生成します。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    g = ap.add_argument_group("スタイル")
    g.add_argument("--style", choices=sorted(STYLE_PRESETS), default="fluted",
                   help="デザインのプリセット")
    g.add_argument("--flutes", type=int, help="縦溝の本数 (0 で溝なし)")
    g.add_argument("--flute-depth", type=float, help="縦溝の深さ mm")
    g.add_argument("--flute-sharpness", type=float, help="溝の鋭さ (1=正弦波)")
    g.add_argument("--twist", type=float, help="上端までのねじれ角 (度)")
    g.add_argument("--wave-count", type=float, help="横方向の波の数")
    g.add_argument("--wave-depth", type=float, help="横方向の波の深さ mm")
    g.add_argument("--squircle", type=float, help="断面 (2=円, 4=角丸四角)")

    g = ap.add_argument_group("寸法 (mm)")
    g.add_argument("--height", type=float, help="高さ")
    g.add_argument("--top-diameter", type=float, help="上端の直径")
    g.add_argument("--bottom-diameter", type=float, help="底の直径")
    g.add_argument("--wall", type=float, help="最小肉厚")
    g.add_argument("--floor", type=float, help="底の厚み")
    g.add_argument("--drain-diameter", type=float, help="排水穴の直径 (0 で穴なし)")
    g.add_argument("--lip", type=float, help="口元の張り出し")
    g.add_argument("--belly", type=float, help="胴の膨らみ (上端半径に対する比)")
    g.add_argument("--foot-inset", type=float, help="底の絞り込み量")
    g.add_argument("--scale", type=float, default=1.0, help="全体の倍率")

    g = ap.add_argument_group("メッシュ")
    g.add_argument("--segments", type=int, help="円周方向の分割数")
    g.add_argument("--rows", type=int, help="高さ方向の分割数")
    g.add_argument("--fluted-inside", action="store_true",
                   help="内側にも溝をつける (材料節約)")

    g = ap.add_argument_group("出力")
    g.add_argument("--out-dir", default=".", help="出力先ディレクトリ")
    g.add_argument("--name", default="planter", help="ファイル名のプレフィックス")
    g.add_argument("--format", choices=["stl", "stl-ascii", "obj", "all"],
                   default="stl", help="出力フォーマット")
    g.add_argument("--no-saucer", action="store_true", help="受け皿を作らない")
    g.add_argument("--saucer-clearance", type=float, default=3.0,
                   help="受け皿と鉢底のすきま")
    g.add_argument("--preview", metavar="FILE.svg", help="SVG プレビューを書き出す")
    g.add_argument("--preview-color", default="#c8734f", help="プレビューの色")
    g.add_argument("--preview-segments", type=int, default=160,
                   help="プレビュー用の分割数 (軽量化)")
    g.add_argument("--azimuth", type=float, default=35.0, help="プレビューの水平角")
    g.add_argument("--elevation", type=float, default=20.0, help="プレビューの仰角")
    return ap


def params_from_args(args) -> Params:
    p = replace(Params(), **STYLE_PRESETS[args.style])
    mapping = {
        "flutes": "flute_count",
        "flute_depth": "flute_depth",
        "flute_sharpness": "flute_sharpness",
        "twist": "twist",
        "wave_count": "wave_count",
        "wave_depth": "wave_depth",
        "squircle": "squircle",
        "height": "height",
        "wall": "wall",
        "floor": "floor",
        "lip": "lip",
        "belly": "belly",
        "foot_inset": "foot_inset",
        "segments": "segments",
        "rows": "rows",
    }
    updates = {}
    for arg_name, field in mapping.items():
        value = getattr(args, arg_name, None)
        if value is not None:
            updates[field] = value
    if args.top_diameter is not None:
        updates["top_radius"] = args.top_diameter / 2.0
    if args.bottom_diameter is not None:
        updates["bottom_radius"] = args.bottom_diameter / 2.0
    if args.drain_diameter is not None:
        updates["drain_radius"] = args.drain_diameter / 2.0
    if args.fluted_inside:
        updates["smooth_inside"] = False
    p = replace(p, **updates)

    if args.scale != 1.0:
        s = args.scale
        lengths = ("height", "bottom_radius", "top_radius", "lip", "foot_inset",
                   "flute_depth", "wave_depth", "wall", "floor", "drain_radius")
        p = replace(p, **{f.name: getattr(p, f.name) * s
                          for f in fields(p) if f.name in lengths})
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    p = params_from_args(args)

    try:
        pot, info = build_planter(p)
    except ValueError as exc:
        print(f"エラー: {exc}", file=sys.stderr)
        return 2

    models = [(pot, info, p)]
    if not args.no_saucer:
        sp = make_saucer(p, clearance=args.saucer_clearance)
        saucer, sinfo = build_planter(sp)
        saucer.name = "saucer"
        models.append((saucer, sinfo, sp))

    os.makedirs(args.out_dir, exist_ok=True)
    writers = {"stl": (write_stl, ".stl"),
               "stl-ascii": (write_stl_ascii, ".ascii.stl"),
               "obj": (write_obj, ".obj")}
    chosen = list(writers) if args.format == "all" else [args.format]

    for mesh, info, mp in models:
        label = args.name if mesh.name == "planter" else f"{args.name}_saucer"
        problems = mesh.check()
        status = "OK (watertight)" if not problems else "; ".join(problems)
        print(f"[{mesh.name}] 高さ {info['height']:.1f}mm / 最大径 "
              f"{info['max_diameter']:.1f}mm / 底径 {info['foot_diameter']:.1f}mm")
        print(f"           三角形 {info['triangles']:,} / 材料 "
              f"{info['material_cm3']:.1f}cm3 / 容量 {info['capacity_ml']:.0f}mL "
              f"/ 最薄肉厚 {info['min_wall']:.2f}mm")
        print(f"           メッシュ検証: {status}")
        if problems:
            return 3
        for key in chosen:
            fn, ext = writers[key]
            path = os.path.join(args.out_dir, label + ext)
            fn(mesh, path)
            print(f"           -> {path} ({os.path.getsize(path) / 1024:.0f} KB)")

    if args.preview:
        # a coarser rebuild keeps the SVG a sane size
        preview_meshes = []
        for _, _, mp in models:
            low = replace(mp, segments=max(9, min(mp.segments, args.preview_segments)),
                          rows=max(24, mp.rows // 3))
            preview_meshes.append(build_planter(low)[0])
        if len(preview_meshes) > 1:
            lo_a, hi_a = preview_meshes[0].bounds()
            lo_b, hi_b = preview_meshes[1].bounds()
            gap = (hi_a[0] - lo_a[0]) / 2.0 + (hi_b[0] - lo_b[0]) / 2.0 + 18.0
            preview_meshes[1].translate(dx=-gap, dy=gap * 0.15)
        render_svg([(m, args.preview_color) for m in preview_meshes], args.preview,
                   azimuth=args.azimuth, elevation=args.elevation)
        print(f"[preview]  -> {args.preview} "
              f"({os.path.getsize(args.preview) / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
