#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""planter.py の検証テスト / geometry checks for the planter generator.

    python3 test_planter.py          (または python3 -m unittest discover)
"""

import contextlib
import io
import math
import os
import struct
import sys
import tempfile
import unittest
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import planter as P


def coarse(p: P.Params) -> P.Params:
    """Same shape, far fewer triangles -- keeps the tests quick."""
    return replace(p, segments=max(9, min(p.segments, 72)), rows=40)


class TestGeometry(unittest.TestCase):
    def test_every_style_is_watertight(self):
        for style in P.STYLE_PRESETS:
            with self.subTest(style=style):
                p = coarse(replace(P.Params(), **P.STYLE_PRESETS[style]))
                mesh, info = P.build_planter(p)
                self.assertEqual(mesh.check(), [])
                self.assertGreater(info["material_cm3"], 0.0)
                self.assertGreater(info["capacity_ml"], 100.0)

    def test_normals_point_outward(self):
        mesh, _ = P.build_planter(coarse(P.Params()))
        self.assertGreater(mesh.volume(), 0.0)

    def test_wall_thickness_is_respected(self):
        p = coarse(replace(P.Params(), wall=3.0))
        _, info = P.build_planter(p)
        self.assertGreaterEqual(info["min_wall"], 3.0 - 0.05)

    def test_dimensions_match_requested(self):
        p = coarse(replace(P.Params(), height=200.0, top_radius=50.0, lip=0.0,
                           belly=0.0, flute_count=0, flute_depth=0.0))
        mesh, info = P.build_planter(p)
        lo, hi = mesh.bounds()
        self.assertAlmostEqual(hi[2] - lo[2], 200.0, places=6)
        self.assertAlmostEqual(info["max_diameter"], 100.0, places=1)

    def test_drain_hole_pierces_the_floor(self):
        p = coarse(replace(P.Params(), drain_radius=6.0))
        mesh, _ = P.build_planter(p)
        # vertices must exist on the drain wall at both the floor top and z=0
        radii_at_zero = [math.hypot(x, y) for x, y, z in mesh.v if abs(z) < 1e-9]
        self.assertAlmostEqual(min(radii_at_zero), 6.0, places=6)
        top = [math.hypot(x, y) for x, y, z in mesh.v if abs(z - p.floor) < 1e-9]
        self.assertAlmostEqual(min(top), 6.0, places=6)

    def test_closed_floor_when_no_drain(self):
        p = coarse(replace(P.Params(), drain_radius=0.0))
        mesh, _ = P.build_planter(p)
        self.assertEqual(mesh.check(), [])
        self.assertTrue(any(abs(x) < 1e-9 and abs(y) < 1e-9 and abs(z) < 1e-9
                            for x, y, z in mesh.v), "底面の中心頂点がない")

    def test_saucer_accepts_the_pot_foot(self):
        p = coarse(P.Params())
        sp = coarse(P.make_saucer(p, clearance=3.0))
        P.build_planter(sp)  # must not raise
        thetas = [2.0 * math.pi * i / 120 for i in range(120)]
        foot = max(P.outer_radius(p, th, 0.0) for th in thetas)
        cavity = min(P.inner_radius(sp, th, sp.floor / sp.height) for th in thetas)
        self.assertGreaterEqual(cavity, foot + 3.0 - 0.25)

    def test_impossible_wall_is_rejected(self):
        p = coarse(replace(P.Params(), wall=40.0))
        with self.assertRaises(ValueError):
            P.build_planter(p)

    def test_oversized_drain_is_rejected(self):
        p = coarse(replace(P.Params(), drain_radius=60.0))
        with self.assertRaises(ValueError):
            P.build_planter(p)

    def test_scaling_scales_the_volume(self):
        small, info_s = P.build_planter(coarse(P.Params()))
        big_params = coarse(P.Params())
        lengths = ("height", "bottom_radius", "top_radius", "lip", "foot_inset",
                   "flute_depth", "wave_depth", "wall", "floor", "drain_radius")
        big_params = replace(big_params, **{k: getattr(big_params, k) * 2.0
                                            for k in lengths})
        _, info_b = P.build_planter(big_params)
        self.assertAlmostEqual(info_b["material_cm3"] / info_s["material_cm3"],
                               8.0, delta=0.15)


class TestExport(unittest.TestCase):
    def test_binary_stl_round_trip(self):
        mesh, _ = P.build_planter(coarse(P.Params()))
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "pot.stl")
            P.write_stl(mesh, path)
            with open(path, "rb") as fh:
                header = fh.read(80)
                count = struct.unpack("<I", fh.read(4))[0]
                body = fh.read()
            self.assertIn(b"planter.py", header)
            self.assertEqual(count, len(mesh.f))
            self.assertEqual(len(body), count * 50)
            self.assertEqual(os.path.getsize(path), 84 + count * 50)

    def test_obj_and_ascii_stl(self):
        mesh, _ = P.build_planter(coarse(P.Params()))
        with tempfile.TemporaryDirectory() as d:
            obj = os.path.join(d, "pot.obj")
            P.write_obj(mesh, obj)
            with open(obj) as fh:
                text = fh.read()
            self.assertEqual(text.count("\nv "), len(mesh.v))
            self.assertEqual(text.count("\nf "), len(mesh.f))

            ascii_stl = os.path.join(d, "pot.ascii.stl")
            P.write_stl_ascii(mesh, ascii_stl)
            with open(ascii_stl) as fh:
                text = fh.read()
            self.assertEqual(text.count("facet normal"), len(mesh.f))
            self.assertTrue(text.startswith("solid "))
            self.assertTrue(text.rstrip().endswith("endsolid planter"))

    def test_svg_preview(self):
        mesh, _ = P.build_planter(coarse(P.Params()))
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "p.svg")
            P.render_svg([(mesh, "#c8734f")], path)
            with open(path) as fh:
                svg = fh.read()
            self.assertTrue(svg.startswith("<svg"))
            self.assertTrue(svg.rstrip().endswith("</svg>"))
            self.assertGreater(svg.count("<polygon"), 500)


class TestCli(unittest.TestCase):
    def test_cli_writes_pot_and_saucer(self):
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            rc = P.main(["--style", "wave", "--segments", "48", "--rows", "30",
                         "--out-dir", d, "--name", "pot", "--format", "all",
                         "--preview", os.path.join(d, "p.svg"),
                         "--preview-segments", "36"])
            self.assertEqual(rc, 0)
            for name in ("pot.stl", "pot.obj", "pot.ascii.stl",
                         "pot_saucer.stl", "p.svg"):
                self.assertTrue(os.path.getsize(os.path.join(d, name)) > 0, name)

    def test_cli_reports_bad_parameters(self):
        err = io.StringIO()
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stderr(err):
            rc = P.main(["--wall", "50", "--segments", "24", "--rows", "20",
                         "--out-dir", d])
            self.assertEqual(rc, 2)
        self.assertIn("壁が厚すぎ", err.getvalue())

    def test_cli_scale(self):
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            rc = P.main(["--scale", "0.5", "--segments", "36", "--rows", "24",
                         "--no-saucer", "--out-dir", d])
            self.assertEqual(rc, 0)
        p = P.params_from_args(P.build_parser().parse_args(["--scale", "0.5"]))
        self.assertAlmostEqual(p.height, P.Params().height * 0.5)
        self.assertEqual(p.segments, P.Params().segments)  # counts stay put


if __name__ == "__main__":
    unittest.main(verbosity=2)
