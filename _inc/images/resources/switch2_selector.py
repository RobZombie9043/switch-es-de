#!/usr/bin/env python3
"""
Simple Switch 2-style animated selector renderer.

Creates:
- PNG frame sequence
- Animated GIF
- Animated WebP
- ES-DE-compatible Lottie JSON with embedded PNG frames

Dependencies:
    pip install numpy pillow

Run:
    python switch2_selector.py
"""

from __future__ import annotations

import base64
import io
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

# CONFIG
WIDTH = 397
HEIGHT = 397
BORDER_WIDTH = 8
CORNER_RADIUS = 30
MARGIN = 0
INNER_BORDER_ENABLED = True
INNER_BORDER_WIDTH = BORDER_WIDTH
INNER_BORDER_GAP = 0
INNER_BORDER_COLOUR = (255, 255, 255, 255)      # white (255, 255, 255, 255) black (0, 0, 0, 255)
FPS = 60
DURATION_SECONDS = 6.0
SUPERSAMPLE = 3
OUTPUT_DIR = Path("switch2_selector_output")
FILE_PREFIX = "switch2_selector"
EXPORT_PNG_SEQUENCE = True
EXPORT_GIF = True
EXPORT_WEBP = True
EXPORT_LOTTIE = True
COLOUR_STOPS = [
    (0.00, (35, 105, 255)),
    (0.23, (35, 235, 255)),
    (0.50, (125, 65, 255)),
    (0.76, (255, 75, 185)),
    (1.00, (35, 105, 255)),
]
FLOW_AMOUNT = 0.035
FLOW_FREQUENCY = 2.0
GIF_COLOURS = 256
GIF_DITHER = True


def smootherstep(t: np.ndarray) -> np.ndarray:
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def sample_colour_wheel(position: np.ndarray) -> np.ndarray:
    position = np.mod(position, 1.0)
    result = np.zeros(position.shape + (3,), dtype=np.float32)
    for index in range(len(COLOUR_STOPS) - 1):
        p0, c0 = COLOUR_STOPS[index]
        p1, c1 = COLOUR_STOPS[index + 1]
        mask = (position >= p0) & (position <= p1 if index == len(COLOUR_STOPS)-2 else position < p1)
        if not np.any(mask):
            continue
        local = (position[mask] - p0) / max(p1 - p0, 1e-9)
        local = smootherstep(local)
        start = np.asarray(c0, dtype=np.float32)
        end = np.asarray(c1, dtype=np.float32)
        result[mask] = start + (end - start) * local[:, None]
    return result


def rounded_rectangle_sdf(x, y, half_width, half_height, radius):
    radius = max(0.0, min(radius, half_width, half_height))
    qx = np.abs(x) - (half_width - radius)
    qy = np.abs(y) - (half_height - radius)
    ox = np.maximum(qx, 0.0)
    oy = np.maximum(qy, 0.0)
    return np.hypot(ox, oy) + np.minimum(np.maximum(qx, qy), 0.0) - radius


def smooth_alpha(distance: np.ndarray, edge_width: float) -> np.ndarray:
    return np.clip(0.5 - distance / max(edge_width, 1e-9), 0.0, 1.0)


def render_frame(progress: float) -> Image.Image:
    ss = max(1, int(SUPERSAMPLE))
    rw, rh = WIDTH * ss, HEIGHT * ss

    xs = (np.arange(rw, dtype=np.float32) + 0.5) / ss - WIDTH / 2.0
    ys = (np.arange(rh, dtype=np.float32) + 0.5) / ss - HEIGHT / 2.0
    x, y = np.meshgrid(xs, ys)

    hw = WIDTH / 2.0 - MARGIN
    hh = HEIGHT / 2.0 - MARGIN

    aa = 1.0 / ss

    # -------------------------------------------------------------------------
    # Animated outer border
    # -------------------------------------------------------------------------

    outer = rounded_rectangle_sdf(
        x,
        y,
        hw,
        hh,
        CORNER_RADIUS,
    )

    animated_inner_hw = max(0.0, hw - BORDER_WIDTH)
    animated_inner_hh = max(0.0, hh - BORDER_WIDTH)
    animated_inner_radius = max(0.0, CORNER_RADIUS - BORDER_WIDTH)

    animated_inner = rounded_rectangle_sdf(
        x,
        y,
        animated_inner_hw,
        animated_inner_hh,
        animated_inner_radius,
    )

    animated_alpha = np.clip(
        smooth_alpha(outer, aa) - smooth_alpha(animated_inner, aa),
        0.0,
        1.0,
    )

    # -------------------------------------------------------------------------
    # Animated colour field
    # -------------------------------------------------------------------------

    nx = x / max(hw, 1e-9)
    ny = y / max(hh, 1e-9)

    angle = np.mod(
        np.arctan2(ny, nx) / (2.0 * math.pi) - progress,
        1.0,
    )

    radial = np.sqrt(nx * nx + ny * ny)

    flow = FLOW_AMOUNT * np.sin(
        2.0
        * math.pi
        * (
            FLOW_FREQUENCY * angle
            + 0.35 * radial
            - progress
        )
    )

    animated_rgb = sample_colour_wheel(
        np.mod(angle + flow, 1.0)
    )

    # Start with the animated outer border.
    output_rgb = animated_rgb.copy()
    output_alpha = animated_alpha.copy()

    # -------------------------------------------------------------------------
    # Solid inner border
    # -------------------------------------------------------------------------

    if INNER_BORDER_ENABLED:
        inner_outer_inset = BORDER_WIDTH + INNER_BORDER_GAP
        inner_inner_inset = inner_outer_inset + INNER_BORDER_WIDTH

        solid_outer_hw = max(0.0, hw - inner_outer_inset)
        solid_outer_hh = max(0.0, hh - inner_outer_inset)
        solid_outer_radius = max(
            0.0,
            CORNER_RADIUS - inner_outer_inset,
        )

        solid_inner_hw = max(0.0, hw - inner_inner_inset)
        solid_inner_hh = max(0.0, hh - inner_inner_inset)
        solid_inner_radius = max(
            0.0,
            CORNER_RADIUS - inner_inner_inset,
        )

        solid_outer = rounded_rectangle_sdf(
            x,
            y,
            solid_outer_hw,
            solid_outer_hh,
            solid_outer_radius,
        )

        solid_inner = rounded_rectangle_sdf(
            x,
            y,
            solid_inner_hw,
            solid_inner_hh,
            solid_inner_radius,
        )

        solid_alpha = np.clip(
            smooth_alpha(solid_outer, aa)
            - smooth_alpha(solid_inner, aa),
            0.0,
            1.0,
        )

        solid_colour = np.asarray(
            INNER_BORDER_COLOUR[:3],
            dtype=np.float32,
        )

        solid_opacity = (
            INNER_BORDER_COLOUR[3] / 255.0
            if len(INNER_BORDER_COLOUR) >= 4
            else 1.0
        )

        solid_alpha *= solid_opacity

        # Alpha composite the solid line over the animated line.
        combined_alpha = (
            solid_alpha
            + output_alpha * (1.0 - solid_alpha)
        )

        numerator = (
            solid_colour[None, None, :] * solid_alpha[:, :, None]
            + output_rgb
            * output_alpha[:, :, None]
            * (1.0 - solid_alpha[:, :, None])
        )

        output_rgb = np.divide(
            numerator,
            combined_alpha[:, :, None],
            out=np.zeros_like(numerator),
            where=combined_alpha[:, :, None] > 0.0,
        )

        output_alpha = combined_alpha

    rgba = np.dstack(
        (
            output_rgb,
            output_alpha * 255.0,
        )
    )

    image = Image.fromarray(
        np.clip(rgba, 0, 255).astype(np.uint8),
        "RGBA",
    )

    if ss > 1:
        image = image.resize(
            (WIDTH, HEIGHT),
            Image.Resampling.LANCZOS,
        )

    return image


def generate_frames() -> list[Image.Image]:
    frame_count = max(1, round(FPS * DURATION_SECONDS))
    frames = []
    for index in range(frame_count):
        frames.append(render_frame(index / frame_count))
        if index == 0 or (index + 1) % max(1, FPS) == 0:
            print(f"Rendered frame {index + 1}/{frame_count}")
    return frames


def save_png_sequence(frames):
    frame_dir = OUTPUT_DIR / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    padding = max(4, len(str(len(frames))))
    for i, frame in enumerate(frames):
        frame.save(frame_dir / f"{FILE_PREFIX}_{i:0{padding}d}.png", optimize=True)
    print(f"Saved PNG sequence: {frame_dir.resolve()}")


def save_gif(frames):
    output_path = OUTPUT_DIR / f"{FILE_PREFIX}.gif"
    duration_ms = round(1000 / FPS)
    dither = Image.Dither.FLOYDSTEINBERG if GIF_DITHER else Image.Dither.NONE
    gif_frames = []
    for frame in frames:
        rgb = Image.new("RGB", frame.size, (0, 0, 0))
        rgb.paste(frame, mask=frame.getchannel("A"))
        q = rgb.quantize(colors=GIF_COLOURS-1, method=Image.Quantize.MEDIANCUT, dither=dither)
        data = np.asarray(q).copy()
        data[np.asarray(frame.getchannel("A")) < 128] = 255
        converted = Image.fromarray(data.astype(np.uint8), "P")
        palette = q.getpalette() or []
        palette += [0] * (768 - len(palette))
        converted.putpalette(palette)
        converted.info["transparency"] = 255
        converted.info["disposal"] = 2
        gif_frames.append(converted)
    gif_frames[0].save(output_path, save_all=True, append_images=gif_frames[1:], duration=duration_ms,
                       loop=0, disposal=2, transparency=255, optimize=False)
    print(f"Saved GIF: {output_path.resolve()}")


def save_webp(frames):
    output_path = OUTPUT_DIR / f"{FILE_PREFIX}.webp"
    frames[0].save(output_path, save_all=True, append_images=frames[1:], duration=round(1000/FPS),
                   loop=0, lossless=True, method=6, exact=True)
    print(f"Saved WebP: {output_path.resolve()}")


def image_to_data_uri(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def save_lottie(frames):
    output_path = OUTPUT_DIR / f"{FILE_PREFIX}.json"
    assets, layers = [], []
    for i, frame in enumerate(frames):
        asset_id = f"frame_{i}"
        assets.append({"id": asset_id, "w": WIDTH, "h": HEIGHT, "u": "", "p": image_to_data_uri(frame), "e": 1})
        layers.append({
            "ddd": 0, "ind": i+1, "ty": 2, "nm": asset_id, "refId": asset_id, "sr": 1,
            "ks": {
                "o": {"a": 0, "k": 100}, "r": {"a": 0, "k": 0},
                "p": {"a": 0, "k": [WIDTH/2, HEIGHT/2, 0]},
                "a": {"a": 0, "k": [WIDTH/2, HEIGHT/2, 0]},
                "s": {"a": 0, "k": [100, 100, 100]},
            },
            "ao": 0, "ip": i, "op": i+1, "st": i, "bm": 0,
        })
    payload = {"v": "5.7.4", "fr": FPS, "ip": 0, "op": len(frames), "w": WIDTH, "h": HEIGHT,
               "nm": FILE_PREFIX, "ddd": 0, "assets": assets, "layers": layers, "markers": []}
    output_path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    print(f"Saved Lottie JSON: {output_path.resolve()}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Generating {round(FPS * DURATION_SECONDS)} frames at {WIDTH}x{HEIGHT}...")
    frames = generate_frames()
    preview_path = OUTPUT_DIR / f"{FILE_PREFIX}_preview.png"
    frames[0].save(preview_path, optimize=True)
    print(f"Saved preview: {preview_path.resolve()}")
    if EXPORT_PNG_SEQUENCE:
        save_png_sequence(frames)
    if EXPORT_GIF:
        save_gif(frames)
    if EXPORT_WEBP:
        save_webp(frames)
    if EXPORT_LOTTIE:
        save_lottie(frames)
    print("Finished.")


if __name__ == "__main__":
    main()
