#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DWG 几何精确重绘工具（基于 ezdwg + matplotlib）
==================================================
功能
----
- 直接解析 AutoCAD DWG 二进制（含 AC1032 / AutoCAD 2018），无需安装 AutoCAD
- 只绘制几何实体：直线 / 圆弧 / 圆 / 多段线 / 椭圆 / 样条 / 面 / 填充边界等
- 自动跳过文字(TEXT/MTEXT)、尺寸标注(DIMENSION)、图块插入点(INSERT)等
- 等比例输出 PNG（位图预览）与 SVG（矢量，可无限缩放、二次编辑）

环境
----
    pip install ezdwg matplotlib

用法
----
    python draw_dwg.py "F:/path/to/Drawing.dwg" [-o 输出前缀] [--dpi 200] [--lw 1.4]

示例
----
    python draw_dwg.py "F:/Desktop/气象桥梁温度场/12梅溪河特大桥截面/Drawing3.dwg"
    python draw_dwg.py "Drawing3.dwg" -o out/drawing3
"""
import argparse
import math
import os
from collections import Counter

import matplotlib
matplotlib.use("Agg")  # 无界面后端，只负责保存图片文件
import matplotlib.pyplot as plt

import ezdwg


# 要绘制的几何类型（其余实体类型如 TEXT/MTEXT/DIMENSION/INSERT 会被自动忽略）
GEOMETRY_TYPES = {
    "LINE", "LWPOLYLINE", "POLYLINE_2D", "POLYLINE_3D", "POLYLINE_MESH",
    "ARC", "CIRCLE", "ELLIPSE", "SPLINE", "3DFACE", "SOLID", "TRACE",
    "SHAPE", "HATCH", "LEADER", "XLINE", "RAY", "MLINE", "POINT",
}


def gtype(entity) -> str:
    """取实体类型名（ezdwg 的 dxftype 可能是属性或方法）"""
    t = entity.dxftype
    return t() if callable(t) else t


def arc_points(center, radius, start_deg, end_deg, segments=128):
    """圆弧/整圆 -> 采样点序列（角度单位为度）"""
    a0 = math.radians(start_deg)
    a1 = math.radians(end_deg)
    if a1 < a0:
        a1 += 2 * math.pi
    angles = [a0 + (a1 - a0) * i / segments for i in range(segments + 1)]
    return [(center[0] + radius * math.cos(a),
             center[1] + radius * math.sin(a)) for a in angles]


def ellipse_points(center, major_axis, axis_ratio, start, end, segments=128):
    m = major_axis
    vx, vy = -m[1] * axis_ratio, m[0] * axis_ratio
    if end < start:
        end += 2 * math.pi
    ts = [start + (end - start) * i / segments for i in range(segments + 1)]
    return [(center[0] + m[0] * math.cos(t) + vx * math.sin(t),
             center[1] + m[1] * math.cos(t) + vy * math.sin(t)) for t in ts]


def entity_xy(entity, dxf):
    """把单个实体转成 2D 点序列；解析失败时返回空列表"""
    t = gtype(entity)
    pts = []
    try:
        if t == "LINE":
            pts = [dxf["start"], dxf["end"]]
        elif t == "ARC":
            pts = arc_points(dxf["center"], dxf["radius"],
                             dxf["start_angle"], dxf["end_angle"])
        elif t == "CIRCLE":
            pts = arc_points(dxf["center"], dxf["radius"], 0.0, 360.0)
        elif t in ("LWPOLYLINE", "POLYLINE_2D", "POLYLINE_3D", "SPLINE"):
            pts = list(dxf.get("points", [])) or list(dxf.get("interpolated_points", []))
            if dxf.get("closed", False) and pts and pts[0] != pts[-1]:
                pts = pts + [pts[0]]
        elif t == "ELLIPSE":
            pts = ellipse_points(dxf["center"], dxf["major_axis"],
                                 dxf["axis_ratio"],
                                 dxf["start_angle"], dxf["end_angle"])
        elif t in ("3DFACE", "SOLID", "TRACE"):
            pts = list(dxf.get("points", []))
            if len(pts) >= 2 and pts[0] != pts[-1]:
                pts = pts + [pts[0]]
        elif t == "HATCH":
            for path in dxf.get("paths", []):
                sub = path.get("points", []) if isinstance(path, dict) else []
                if path.get("closed", False) and sub and sub[0] != sub[-1]:
                    sub = sub + [sub[0]]
                pts.extend(sub)
        elif t in ("LEADER", "MLINE"):
            pts = list(dxf.get("points", []))
        elif t in ("XLINE", "RAY"):
            start = dxf.get("start", (0.0, 0.0, 0.0))
            u = dxf.get("unit_vector", (1.0, 0.0, 0.0))
            length = math.hypot(u[0], u[1]) or 1.0
            d = (u[0] / length, u[1] / length)
            s = 200.0
            end = (start[0] + d[0] * s, start[1] + d[1] * s)
            pts = [(start[0] - d[0] * s, start[1] - d[1] * s), end] if t == "XLINE" else [start, end]
        elif t == "POINT":
            pts = [dxf.get("location", (0, 0, 0))]
        else:
            pts = list(entity.to_points())  # 兜底：ezdwg 自带的通用取点
    except Exception:
        pts = []
    return [tuple(p[:2]) for p in pts if len(p) >= 2]


def main():
    ap = argparse.ArgumentParser(description="DWG 几何精确重绘（忽略标注）")
    ap.add_argument("dwg", help="输入 DWG 文件路径")
    ap.add_argument("-o", "--out", help="输出文件前缀（默认: 当前目录/<图名>_geom）")
    ap.add_argument("--dpi", type=int, default=200, help="PNG 分辨率")
    ap.add_argument("--lw", type=float, default=1.4, help="线宽")
    ap.add_argument("--drop-near-origin", type=float, default=0.0,
                    help=">0 时跳过所有点距原点均小于该值(单位同图)的实体，"
                         "用于剔除坐标原点附近的小标记/短斜线，如 --drop-near-origin 2")
    ap.add_argument("--mirror-fix", action="store_true",
                    help="检测并按X轴对称补全缺失的LINE（如右侧缺失的对称轮廓线），"
                         "补线用与源线一致的黑色实线")
    args = ap.parse_args()

    src = os.path.abspath(args.dwg)
    if args.out:
        prefix = os.path.abspath(args.out)
    else:
        prefix = os.path.join(os.getcwd(),
                              os.path.splitext(os.path.basename(src))[0] + "_geom")

    doc = ezdwg.read(src)
    layout = doc.modelspace()
    print(f"文件 : {src}")
    print(f"版本 : {doc.version} | 单位 : {doc.units}")

    # 第一遍：收集全部 LINE 线段（供 --mirror-fix 做对称缺失检测）
    line_segs = set()
    if args.mirror_fix:
        for entity in layout.iter_entities():
            if gtype(entity) != "LINE":
                continue
            dxf = entity.dxf
            s, en = dxf.get("start"), dxf.get("end")
            if s is not None and en is not None and len(s) >= 2 and len(en) >= 2:
                if args.drop_near_origin > 0 and (
                        math.hypot(s[0], s[1]) < args.drop_near_origin and
                        math.hypot(en[0], en[1]) < args.drop_near_origin):
                    continue
                a = (round(s[0], 3), round(s[1], 3))
                b = (round(en[0], 3), round(en[1], 3))
                if a != b:
                    line_segs.add(frozenset((a, b)))

    fig, ax = plt.subplots(figsize=(13, 10))
    counts = Counter()
    xs, ys = [], []
    for entity in layout.iter_entities():
        t = gtype(entity)
        if t not in GEOMETRY_TYPES:
            continue  # 忽略标注/图块等非几何实体
        pts = entity_xy(entity, entity.dxf)
        if not pts:
            continue
        if args.drop_near_origin > 0 and all(
                math.hypot(p[0], p[1]) < args.drop_near_origin for p in pts):
            print(f"  跳过原点附近实体: {t} (handle={entity.handle}, 点数={len(pts)})")
            continue
        counts[t] += 1
        if t == "POINT":
            for p in pts:
                ax.plot([p[0]], [p[1]], marker="o", markersize=3,
                        linewidth=0, color="black")
        else:
            ax.plot([p[0] for p in pts], [p[1] for p in pts],
                    color="black", linewidth=args.lw)
        xs.extend(p[0] for p in pts)
        ys.extend(p[1] for p in pts)

    if not xs:
        print("警告: 未找到可绘制的几何实体")
        return 1

    # 第二遍：按 X 轴对称补全缺失的 LINE（蓝色虚线标记补线）
    if args.mirror_fix:
        added = []
        for seg in line_segs:
            a, b = tuple(seg)
            ma, mb = (-a[0], a[1]), (-b[0], b[1])
            if frozenset((ma, mb)) not in line_segs:
                added.append((ma, mb))
                ax.plot([ma[0], mb[0]], [ma[1], mb[1]],
                        color="black", linewidth=args.lw)
                xs.extend([ma[0], mb[0]]); ys.extend([ma[1], mb[1]])
        if added:
            print(f"补全对称缺失线段: {len(added)} 根（黑色实线，与源线一致）")
            for a, b in added:
                print(f"  ({a[0]:.3f},{a[1]:.3f}) -> ({b[0]:.3f},{b[1]:.3f})")

    ax.set_aspect("equal")
    ax.autoscale()
    ax.margins(0.06)
    ax.set_axis_off()

    png, svg = prefix + ".png", prefix + ".svg"
    fig.savefig(png, dpi=args.dpi, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")

    print(f"几何实体 : {dict(counts)}")
    print(f"外接范围 : X {min(xs):.3f} ~ {max(xs):.3f} (跨度 {max(xs)-min(xs):.3f} mm)")
    print(f"         Y {min(ys):.3f} ~ {max(ys):.3f} (跨度 {max(ys)-min(ys):.3f} mm)")
    print(f"输出 PNG : {png}")
    print(f"输出 SVG : {svg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
