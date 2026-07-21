#!/usr/bin/env python3
"""
Generate static selector-frame PNGs for a range of aspect ratios.

Dependencies:
    pip install numpy pillow

Run:
    python static_selector_frames.py
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image

# =============================================================================
# CONFIG
# =============================================================================

WIDTH = 357

# Ratios are interpreted as width:height.
ASPECT_RATIOS = [
    (1, 1), (1, 2), (2, 3), (3, 4), (3, 5), (4, 3), (5, 7),
    (99, 168), (105, 170), (112, 67), (135, 172), (143, 207),
    (145, 205), (243, 340), (257, 229), (373, 436), (752, 1440),
]

BORDER_WIDTH = 8
MARGIN = 0
CORNER_RADIUS = 0

INNER_BORDER_ENABLED = True
INNER_BORDER_WIDTH = 8
INNER_BORDER_GAP = 0
INNER_BORDER_COLOUR = (255, 255, 255, 255)  # RGBA

# "gradient" for Switch 2-style, or "solid" for Switch 1-style.
OUTER_BORDER_MODE = "solid"

SOLID_BORDER_COLOUR = (0, 195, 227, 255)  # RGBA ; dark - 0 186 218 ; light - 0 195 227

GRADIENT_STOPS = [
    (0.00, (35, 105, 255)),
    (0.25, (35, 235, 255)),
    (0.50, (125, 65, 255)),
    (0.75, (255, 75, 185)),
    (1.00, (35, 105, 255)),
]
GRADIENT_ROTATION = 0.125

SUPERSAMPLE = 4
OUTPUT_DIR = Path("static_selector_frames")
FILE_PREFIX = "selector"


def rounded_rectangle_sdf(x, y, half_width, half_height, radius):
    half_width = max(0.0, half_width)
    half_height = max(0.0, half_height)
    radius = max(0.0, min(radius, half_width, half_height))

    qx = np.abs(x) - (half_width - radius)
    qy = np.abs(y) - (half_height - radius)
    ox = np.maximum(qx, 0.0)
    oy = np.maximum(qy, 0.0)
    outside = np.hypot(ox, oy)
    inside = np.minimum(np.maximum(qx, qy), 0.0)
    return outside + inside - radius


def smooth_alpha(distance, edge_width):
    return np.clip(0.5 - distance / max(edge_width, 1e-9), 0.0, 1.0)


def make_ring_alpha(x, y, hw, hh, radius, ring_width, aa):
    if ring_width <= 0:
        return np.zeros_like(x, dtype=np.float32)

    outer = rounded_rectangle_sdf(x, y, hw, hh, radius)
    inner = rounded_rectangle_sdf(
        x, y,
        max(0.0, hw - ring_width),
        max(0.0, hh - ring_width),
        max(0.0, radius - ring_width),
    )
    return np.clip(smooth_alpha(outer, aa) - smooth_alpha(inner, aa), 0.0, 1.0)


def smootherstep(t):
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def sample_gradient(position):
    position = np.mod(position, 1.0)
    result = np.zeros(position.shape + (3,), dtype=np.float32)

    for i in range(len(GRADIENT_STOPS) - 1):
        p0, c0 = GRADIENT_STOPS[i]
        p1, c1 = GRADIENT_STOPS[i + 1]
        mask = (position >= p0) & (position <= p1 if i == len(GRADIENT_STOPS) - 2 else position < p1)
        if not np.any(mask):
            continue
        t = smootherstep((position[mask] - p0) / max(p1 - p0, 1e-9))
        c0 = np.asarray(c0, dtype=np.float32)
        c1 = np.asarray(c1, dtype=np.float32)
        result[mask] = c0 + (c1 - c0) * t[:, None]

    return result


def make_outer_rgb(x, y, hw, hh):
    mode = OUTER_BORDER_MODE.strip().lower()

    if mode == "solid":
        colour = np.asarray(SOLID_BORDER_COLOUR[:3], dtype=np.float32)
        rgb = np.broadcast_to(colour, x.shape + (3,)).copy()
        opacity = SOLID_BORDER_COLOUR[3] / 255.0 if len(SOLID_BORDER_COLOUR) >= 4 else 1.0
        return rgb, opacity

    if mode != "gradient":
        raise ValueError('OUTER_BORDER_MODE must be "gradient" or "solid".')

    nx = x / max(hw, 1e-9)
    ny = y / max(hh, 1e-9)
    position = np.mod(np.arctan2(ny, nx) / (2.0 * math.pi) + GRADIENT_ROTATION, 1.0)
    return sample_gradient(position), 1.0


def composite_solid(dst_rgb, dst_alpha, source_colour, source_alpha):
    colour = np.asarray(source_colour, dtype=np.float32)
    combined_alpha = source_alpha + dst_alpha * (1.0 - source_alpha)
    numerator = (
        colour[None, None, :] * source_alpha[:, :, None]
        + dst_rgb * dst_alpha[:, :, None] * (1.0 - source_alpha[:, :, None])
    )
    combined_rgb = np.divide(
        numerator,
        combined_alpha[:, :, None],
        out=np.zeros_like(numerator),
        where=combined_alpha[:, :, None] > 0.0,
    )
    return combined_rgb, combined_alpha


def calculate_height(width, ratio_width, ratio_height):
    return max(1, round(width * ratio_height / ratio_width))


def render_selector(width, height):
    ss = max(1, int(SUPERSAMPLE))
    rw, rh = width * ss, height * ss

    xs = (np.arange(rw, dtype=np.float32) + 0.5) / ss - width / 2.0
    ys = (np.arange(rh, dtype=np.float32) + 0.5) / ss - height / 2.0
    x, y = np.meshgrid(xs, ys)

    aa = 1.0 / ss
    hw = max(0.0, width / 2.0 - MARGIN)
    hh = max(0.0, height / 2.0 - MARGIN)
    radius = min(CORNER_RADIUS, hw, hh)

    outer_alpha = make_ring_alpha(x, y, hw, hh, radius, BORDER_WIDTH, aa)
    outer_rgb, outer_opacity = make_outer_rgb(x, y, hw, hh)
    output_rgb = outer_rgb
    output_alpha = outer_alpha * outer_opacity

    if INNER_BORDER_ENABLED and INNER_BORDER_WIDTH > 0:
        inset = BORDER_WIDTH + INNER_BORDER_GAP
        inner_hw = max(0.0, hw - inset)
        inner_hh = max(0.0, hh - inset)
        inner_radius = max(0.0, radius - inset)

        inner_alpha = make_ring_alpha(
            x, y, inner_hw, inner_hh, inner_radius,
            INNER_BORDER_WIDTH, aa,
        )
        inner_opacity = INNER_BORDER_COLOUR[3] / 255.0 if len(INNER_BORDER_COLOUR) >= 4 else 1.0
        inner_alpha *= inner_opacity

        output_rgb, output_alpha = composite_solid(
            output_rgb,
            output_alpha,
            INNER_BORDER_COLOUR[:3],
            inner_alpha,
        )

    rgba = np.dstack((output_rgb, output_alpha * 255.0))
    image = Image.fromarray(np.clip(rgba, 0, 255).astype(np.uint8), mode="RGBA")

    if ss > 1:
        image = image.resize((width, height), Image.Resampling.LANCZOS)

    return image


def main():
    if WIDTH <= 0:
        raise ValueError("WIDTH must be greater than zero.")
    if BORDER_WIDTH < 0 or INNER_BORDER_WIDTH < 0 or INNER_BORDER_GAP < 0:
        raise ValueError("Border widths and gap cannot be negative.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Generating {len(ASPECT_RATIOS)} static selector frames")
    print(f"Width: {WIDTH}px")
    print(f"Outer mode: {OUTER_BORDER_MODE}\n")

    for ratio_width, ratio_height in ASPECT_RATIOS:
        height = calculate_height(WIDTH, ratio_width, ratio_height)
        image = render_selector(WIDTH, height)

        ratio_name = f"{ratio_width}-{ratio_height}"
        filename = f"{FILE_PREFIX}_{ratio_name}_{WIDTH}x{height}_{OUTER_BORDER_MODE.lower()}.png"
        output_path = OUTPUT_DIR / filename
        image.save(output_path, optimize=True)

        print(f"{ratio_width}:{ratio_height} -> {WIDTH}x{height} -> {output_path.name}")

    print(f"\nFinished. Output folder: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
