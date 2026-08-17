#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DXF -> parametric box-girder section extractor.

Strategy
--------
A full arbitrary-DWG-to-mesh pipeline is brittle.  For research-grade bridges,
the great majority of concrete bridges can be represented by a multi-cell box
section.  This module therefore:

1. Reads DXF entities (LWPOLYLINE, LINE, ARC, CIRCLE, POLYLINE, INSERT/block refs).
2. Converts open linework into closed faces via shapely ``polygonize`` so that
   drawings made of plain LINE segments (not closed polylines) also work.
3. Extracts the outer bounding box of the concrete outline and counts interior
   cavities (cells).
4. Falls back to a parametric section if parsing is ambiguous.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import ezdxf
import numpy as np
from shapely.geometry import LineString, MultiPolygon, Point, Polygon
from shapely.ops import polygonize, unary_union

from fem_box_core import BoxSection

warnings.filterwarnings("ignore")


@dataclass
class DXFProfile:
    """Parsed profile from a DXF file."""
    outer: Polygon
    holes: List[Polygon]
    concrete: MultiPolygon
    width: float
    height: float
    n_cells: int
    centroid: Tuple[float, float]


def _iter_entities(msp):
    """Iterate over model-space entities, expanding INSERT (block reference) entities."""
    for e in msp:
        if e.dxftype() == "INSERT":
            try:
                for ve in e.virtual_entities():
                    yield ve
            except Exception:
                pass
        else:
            yield e


def _polygon_from_entity(entity) -> Optional[object]:
    """Convert a single DXF entity to shapely polygon/line (projected to XY)."""
    dxftype = entity.dxftype()
    try:
        if dxftype == "LWPOLYLINE":
            pts = [(p[0], p[1]) for p in entity.get_points()]
            if len(pts) < 2:
                return None
            if entity.closed or pts[0] == pts[-1]:
                return Polygon(pts)
            return LineString(pts)
        elif dxftype == "LINE":
            p0 = (float(entity.dxf.start[0]), float(entity.dxf.start[1]))
            p1 = (float(entity.dxf.end[0]), float(entity.dxf.end[1]))
            if np.hypot(p1[0] - p0[0], p1[1] - p0[1]) < 1e-9:
                return None
            return LineString([p0, p1])
        elif dxftype == "POLYLINE":
            pts = [(v.dxf.location[0], v.dxf.location[1]) for v in entity.vertices]
            if len(pts) < 2:
                return None
            if entity.is_closed or pts[0] == pts[-1]:
                return Polygon(pts)
            return LineString(pts)
        elif dxftype == "CIRCLE":
            c = entity.dxf.center
            r = entity.dxf.radius
            if r <= 0:
                return None
            return Point(c[0], c[1]).buffer(r, resolution=64)
        elif dxftype == "ARC":
            c = entity.dxf.center
            r = entity.dxf.radius
            if r <= 0:
                return None
            sa = np.radians(entity.dxf.start_angle)
            ea = np.radians(entity.dxf.end_angle)
            if ea < sa:
                ea += 2 * np.pi
            angles = np.linspace(sa, ea, 48)
            pts = [(c[0] + r * np.cos(a), c[1] + r * np.sin(a)) for a in angles]
            return LineString(pts)
    except Exception:
        return None
    return None


def parse_dxf(path: str) -> DXFProfile:
    """Parse DXF file and infer outer hull + holes (robust to open linework)."""
    doc = ezdxf.readfile(path)
    msp = doc.modelspace()

    faces = []      # closed polygons (from closed entities + polygonized faces)
    linework = []   # open line segments / arcs
    for e in _iter_entities(msp):
        g = _polygon_from_entity(e)
        if g is None:
            continue
        if isinstance(g, Polygon):
            faces.append(g)
        elif isinstance(g, LineString):
            linework.append(g)

    # Merge open linework into closed faces (drawings made of LINE segments)
    if linework:
        try:
            merged = unary_union(linework)
            faces.extend(list(polygonize(merged)))
        except Exception:
            pass

    # Drop invalid / degenerate polygons
    faces = [p for p in faces if p is not None and p.is_valid and p.area > 1e-6]
    if not faces:
        raise ValueError(
            "No closed polygons found. Make sure the cross-section is drawn as "
            "closed polylines/regions (or connected LINE segments forming a closed loop)."
        )

    # Largest face = outer boundary; interior faces = cavities
    faces.sort(key=lambda p: p.area, reverse=True)
    outer = faces[0]
    outer_area = outer.area
    holes = []
    for p in faces[1:]:
        if p.area < 1e-6 or p.area > 0.98 * outer_area:
            continue
        rep = p.representative_point()
        if outer.covers(rep):
            if p.area >= 0.005 * outer_area:  # drop slivers
                holes.append(p)

    concrete = outer.difference(unary_union(holes)) if holes else outer
    if isinstance(concrete, Polygon):
        concrete = MultiPolygon([concrete])

    minx, miny, maxx, maxy = concrete.bounds
    return DXFProfile(
        outer=outer,
        holes=holes,
        concrete=concrete,
        width=maxx - minx,
        height=maxy - miny,
        n_cells=len(holes),
        centroid=(float((minx + maxx) / 2.0), float((miny + maxy) / 2.0)),
    )


def profile_to_section(prof: DXFProfile) -> BoxSection:
    """Convert a parsed profile to a parametric BoxSection (with mm -> m rescale)."""
    width = prof.width
    height = prof.height
    if width <= 0 or height <= 0:
        raise ValueError("Invalid section dimensions from DXF.")

    # Bridge drawings are typically in millimetres; rescale to metres if necessary.
    if width > 10.0:
        width /= 1000.0
        height /= 1000.0

    n_cells = max(1, prof.n_cells)
    if n_cells > 6:  # guard against many small circles being counted as cells
        n_cells = 3

    # Typical box-girder proportions
    top_slab = max(0.20, min(0.60, height * 0.12))
    bottom_slab = max(0.20, min(0.60, height * 0.10))
    web_thick = max(0.30, min(0.80, width * 0.04))
    overhang = max(0.0, min(width * 0.25, 2.5))

    return BoxSection(
        width=width,
        height=height,
        top_slab=top_slab,
        bottom_slab=bottom_slab,
        web_thick=web_thick,
        n_cells=n_cells,
        overhang=overhang,
    )


def dxf_to_box_section(path: str) -> Tuple[Optional[BoxSection], Optional[str]]:
    """
    Convert DXF to a parametric BoxSection.

    Returns
    -------
    (section, None) on success, or (None, error_message) on failure. The caller
    can fall back to the manual/default parametric section.
    """
    try:
        prof = parse_dxf(path)
        sec = profile_to_section(prof)
        return sec, None
    except Exception as e:
        return None, str(e)


if __name__ == "__main__":
    # Smoke test: create a tiny DXF with a rectangle + two cavities + open lines
    doc = ezdxf.new("R2010")
    msp = doc.modelspace()
    msp.add_lwpolyline([(0, 0), (10, 0), (10, 3), (0, 3)], close=True)
    msp.add_lwpolyline([(3, 0.5), (4.5, 0.5), (4.5, 2.5), (3, 2.5)], close=True)
    msp.add_lwpolyline([(5.5, 0.5), (7, 0.5), (7, 2.5), (5.5, 2.5)], close=True)
    test_path = "_test_section.dxf"
    doc.saveas(test_path)

    sec, err = dxf_to_box_section(test_path)
    print("Parsed section:", sec, "| error:", err)
    Path(test_path).unlink(missing_ok=True)
