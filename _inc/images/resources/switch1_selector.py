#!/usr/bin/env python3
"""
Simple Nintendo Switch 1-style breathing selector renderer.

The outer border uses a static colour that gently brightens, dims, and expands.
An optional solid inner border can be enabled with its own configurable colour.

Creates:
- Preview PNG
- PNG frame sequence
- Animated GIF
- Animated WebP
- Optional ES-DE-compatible Lottie JSON with embedded PNG frames

Dependencies:
    pip install numpy pillow

Run:
    python switch1_breathing_selector.py
"""

from __future__ import annotations

import base64
import io
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image


# =============================================================================
# CONFIG
# =============================================================================

WIDTH = 397
HEIGHT = 397

# Animated outer border
BORDER_WIDTH = 8
CORNER_RADIUS = 0
MARGIN = 0
BORDER_COLOUR = (0, 195, 227)  # RGB dark 0, 195, 227 light 0, 240, 255

# Breathing effect
DURATION_SECONDS = 1
BREATH_MIN_BRIGHTNESS = 0.72
BREATH_MAX_BRIGHTNESS = 1.18
BREATH_SATURATION_BOOST = 0.08

# Set to 0 to disable the slight expansion/contraction.
BREATH_SIZE_PIXELS = 0

# Optional solid line inside the animated border
INNER_BORDER_ENABLED = True
INNER_BORDER_WIDTH = BORDER_WIDTH
INNER_BORDER_GAP = 0
INNER_BORDER_COLOUR = (0, 0, 0, 255)  # RGBA

FPS = 60
SUPERSAMPLE = 3

OUTPUT_DIR = Path("switch1_breathing_selector_output")
FILE_PREFIX = "switch1_breathing_selector"

EXPORT_PNG_SEQUENCE = True
EXPORT_GIF = True
EXPORT_WEBP = True
EXPORT_LOTTIE = True

GIF_COLOURS = 256
GIF_DITHER = True


# =============================================================================
# GEOMETRY HELPERS
# =============================================================================

def rounded_rectangle_sdf(
    x: np.ndarray,
    y: np.ndarray,
    half_width: float,
    half_height: float,
    radius: float,
) -> np.ndarray:
    """
    Signed distance to a centred rounded rectangle.

    Negative values are inside, zero is the edge, and positive values are outside.
    """
    half_width = max(0.0, half_width)
    half_height = max(0.0, half_height)
    radius = max(0.0, min(radius, half_width, half_height))

    qx = np.abs(x) - (half_width - radius)
    qy = np.abs(y) - (half_height - radius)

    outside_x = np.maximum(qx, 0.0)
    outside_y = np.maximum(qy, 0.0)

    outside_distance = np.hypot(outside_x, outside_y)
    inside_distance = np.minimum(np.maximum(qx, qy), 0.0)

    return outside_distance + inside_distance - radius


def smooth_alpha(distance: np.ndarray, edge_width: float) -> np.ndarray:
    """Convert a signed distance field into an anti-aliased fill mask."""
    return np.clip(
        0.5 - distance / max(edge_width, 1e-9),
        0.0,
        1.0,
    )


def make_ring_alpha(
    x: np.ndarray,
    y: np.ndarray,
    outer_half_width: float,
    outer_half_height: float,
    outer_radius: float,
    ring_width: float,
    antialias_width: float,
) -> np.ndarray:
    """Create an anti-aliased rounded-rectangle ring."""
    outer_distance = rounded_rectangle_sdf(
        x,
        y,
        outer_half_width,
        outer_half_height,
        outer_radius,
    )

    inner_distance = rounded_rectangle_sdf(
        x,
        y,
        max(0.0, outer_half_width - ring_width),
        max(0.0, outer_half_height - ring_width),
        max(0.0, outer_radius - ring_width),
    )

    outer_fill = smooth_alpha(outer_distance, antialias_width)
    inner_fill = smooth_alpha(inner_distance, antialias_width)

    return np.clip(outer_fill - inner_fill, 0.0, 1.0)


# =============================================================================
# COMPOSITING
# =============================================================================

def composite_solid(
    destination_rgb: np.ndarray,
    destination_alpha: np.ndarray,
    source_colour: tuple[int, int, int],
    source_alpha: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Alpha-composite a solid-colour layer over an RGB/alpha destination."""
    colour = np.asarray(source_colour, dtype=np.float32)

    combined_alpha = source_alpha + destination_alpha * (1.0 - source_alpha)

    numerator = (
        colour[None, None, :] * source_alpha[:, :, None]
        + destination_rgb
        * destination_alpha[:, :, None]
        * (1.0 - source_alpha[:, :, None])
    )

    combined_rgb = np.divide(
        numerator,
        combined_alpha[:, :, None],
        out=np.zeros_like(numerator),
        where=combined_alpha[:, :, None] > 0.0,
    )

    return combined_rgb, combined_alpha


# =============================================================================
# BREATHING COLOUR
# =============================================================================

def breathing_colour(progress: float) -> tuple[np.ndarray, float]:
    """
    Return the current outer-border RGB colour and normalized pulse amount.

    A cosine wave starts at the dimmest point, rises smoothly to maximum
    brightness halfway through, then returns to the dimmest point.
    """
    pulse = 0.5 - 0.5 * math.cos(progress * 2.0 * math.pi)

    brightness = (
        BREATH_MIN_BRIGHTNESS
        + pulse * (BREATH_MAX_BRIGHTNESS - BREATH_MIN_BRIGHTNESS)
    )

    saturation = 1.0 + BREATH_SATURATION_BOOST * pulse

    base = np.asarray(BORDER_COLOUR, dtype=np.float32)

    # Standard luminance produces a more natural saturation adjustment than mean().
    luminance = (
        base[0] * 0.2126
        + base[1] * 0.7152
        + base[2] * 0.0722
    )

    colour = luminance + (base - luminance) * saturation
    colour *= brightness

    return np.clip(colour, 0.0, 255.0), pulse


# =============================================================================
# FRAME RENDERER
# =============================================================================

def render_frame(progress: float) -> Image.Image:
    """Render one transparent RGBA animation frame."""
    supersample = max(1, int(SUPERSAMPLE))
    render_width = WIDTH * supersample
    render_height = HEIGHT * supersample

    # Pixel-centred coordinates measured in final output pixels.
    xs = (
        (np.arange(render_width, dtype=np.float32) + 0.5)
        / supersample
        - WIDTH / 2.0
    )
    ys = (
        (np.arange(render_height, dtype=np.float32) + 0.5)
        / supersample
        - HEIGHT / 2.0
    )
    x, y = np.meshgrid(xs, ys)

    antialias_width = 1.0 / supersample

    animated_colour, pulse = breathing_colour(progress)

    # Expand from 0 at minimum brightness to BREATH_SIZE_PIXELS at maximum.
    size_offset = BREATH_SIZE_PIXELS * pulse

    outer_half_width = WIDTH / 2.0 - MARGIN + size_offset
    outer_half_height = HEIGHT / 2.0 - MARGIN + size_offset
    outer_radius = CORNER_RADIUS + size_offset

    animated_alpha = make_ring_alpha(
        x,
        y,
        outer_half_width,
        outer_half_height,
        outer_radius,
        BORDER_WIDTH,
        antialias_width,
    )

    output_rgb = np.broadcast_to(
        animated_colour,
        x.shape + (3,),
    ).copy()

    output_alpha = animated_alpha.copy()

    # -------------------------------------------------------------------------
    # Optional solid inner border
    # -------------------------------------------------------------------------

    if INNER_BORDER_ENABLED:
        inner_outer_inset = BORDER_WIDTH + INNER_BORDER_GAP

        # Keep the solid inner line fixed while the outer line breathes.
        base_half_width = WIDTH / 2.0 - MARGIN
        base_half_height = HEIGHT / 2.0 - MARGIN

        inner_outer_half_width = max(
            0.0,
            base_half_width - inner_outer_inset,
        )
        inner_outer_half_height = max(
            0.0,
            base_half_height - inner_outer_inset,
        )
        inner_outer_radius = max(
            0.0,
            CORNER_RADIUS - inner_outer_inset,
        )

        inner_alpha = make_ring_alpha(
            x,
            y,
            inner_outer_half_width,
            inner_outer_half_height,
            inner_outer_radius,
            INNER_BORDER_WIDTH,
            antialias_width,
        )

        inner_opacity = (
            INNER_BORDER_COLOUR[3] / 255.0
            if len(INNER_BORDER_COLOUR) >= 4
            else 1.0
        )
        inner_alpha *= inner_opacity

        output_rgb, output_alpha = composite_solid(
            output_rgb,
            output_alpha,
            INNER_BORDER_COLOUR[:3],
            inner_alpha,
        )

    rgba = np.dstack(
        (
            output_rgb,
            output_alpha * 255.0,
        )
    )

    image = Image.fromarray(
        np.clip(rgba, 0.0, 255.0).astype(np.uint8),
        mode="RGBA",
    )

    if supersample > 1:
        image = image.resize(
            (WIDTH, HEIGHT),
            resample=Image.Resampling.LANCZOS,
        )

    return image


# =============================================================================
# EXPORTERS
# =============================================================================

def generate_frames() -> list[Image.Image]:
    frame_count = max(1, round(FPS * DURATION_SECONDS))
    frames: list[Image.Image] = []

    for index in range(frame_count):
        # Do not render progress=1.0 because it duplicates the first frame.
        progress = index / frame_count
        frames.append(render_frame(progress))

        if index == 0 or (index + 1) % max(1, FPS) == 0:
            print(f"Rendered frame {index + 1}/{frame_count}")

    return frames


def save_png_sequence(frames: list[Image.Image]) -> None:
    frame_directory = OUTPUT_DIR / "frames"
    frame_directory.mkdir(parents=True, exist_ok=True)

    padding = max(4, len(str(len(frames))))

    for index, frame in enumerate(frames):
        filename = f"{FILE_PREFIX}_{index:0{padding}d}.png"
        frame.save(frame_directory / filename, optimize=True)

    print(f"Saved PNG sequence: {frame_directory.resolve()}")


def save_gif(frames: list[Image.Image]) -> None:
    output_path = OUTPUT_DIR / f"{FILE_PREFIX}.gif"
    duration_ms = round(1000 / FPS)

    dither = (
        Image.Dither.FLOYDSTEINBERG
        if GIF_DITHER
        else Image.Dither.NONE
    )

    # Reserve index 255 for transparency by limiting visible colours to 255.
    palette_source = frames[0].convert("RGB").quantize(
        colors=min(GIF_COLOURS - 1, 255),
        method=Image.Quantize.MEDIANCUT,
        dither=dither,
    )

    gif_frames: list[Image.Image] = []
    transparent_index = 255

    for frame in frames:
        quantized = frame.convert("RGB").quantize(
            palette=palette_source,
            dither=dither,
        )

        palette = quantized.getpalette()
        pixel_data = np.asarray(quantized).copy()
        alpha_data = np.asarray(frame.getchannel("A"))

        pixel_data[alpha_data < 128] = transparent_index

        converted = Image.fromarray(
            pixel_data.astype(np.uint8),
            mode="P",
        )
        converted.putpalette(palette)
        converted.info["transparency"] = transparent_index
        converted.info["disposal"] = 2

        gif_frames.append(converted)

    gif_frames[0].save(
        output_path,
        save_all=True,
        append_images=gif_frames[1:],
        duration=duration_ms,
        loop=0,
        disposal=2,
        transparency=transparent_index,
        optimize=False,
    )

    print(f"Saved GIF: {output_path.resolve()}")


def save_webp(frames: list[Image.Image]) -> None:
    output_path = OUTPUT_DIR / f"{FILE_PREFIX}.webp"
    duration_ms = round(1000 / FPS)

    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        lossless=True,
        method=6,
        exact=True,
    )

    print(f"Saved WebP: {output_path.resolve()}")


def image_to_data_uri(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def save_lottie(frames: list[Image.Image]) -> None:
    """
    Export an embedded PNG image-sequence Lottie file.

    It is larger than a true vector Lottie, but reliably preserves the rendered
    appearance in ES-DE/rlottie implementations that support embedded images.
    """
    output_path = OUTPUT_DIR / f"{FILE_PREFIX}.json"
    frame_count = len(frames)

    assets = []
    layers = []

    for index, frame in enumerate(frames):
        asset_id = f"frame_{index}"

        assets.append({
            "id": asset_id,
            "w": WIDTH,
            "h": HEIGHT,
            "u": "",
            "p": image_to_data_uri(frame),
            "e": 1,
        })

        layers.append({
            "ddd": 0,
            "ind": index + 1,
            "ty": 2,
            "nm": asset_id,
            "refId": asset_id,
            "sr": 1,
            "ks": {
                "o": {"a": 0, "k": 100},
                "r": {"a": 0, "k": 0},
                "p": {"a": 0, "k": [WIDTH / 2, HEIGHT / 2, 0]},
                "a": {"a": 0, "k": [WIDTH / 2, HEIGHT / 2, 0]},
                "s": {"a": 0, "k": [100, 100, 100]},
            },
            "ao": 0,
            "ip": index,
            "op": index + 1,
            "st": index,
            "bm": 0,
        })

    lottie = {
        "v": "5.7.4",
        "fr": FPS,
        "ip": 0,
        "op": frame_count,
        "w": WIDTH,
        "h": HEIGHT,
        "nm": FILE_PREFIX,
        "ddd": 0,
        "assets": assets,
        "layers": layers,
        "markers": [],
    }

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(lottie, file, separators=(",", ":"))

    print(f"Saved Lottie JSON: {output_path.resolve()}")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    frame_count = max(1, round(FPS * DURATION_SECONDS))

    print("Generating Switch 1-style breathing selector")
    print(f"Size: {WIDTH}x{HEIGHT}")
    print(f"Frames: {frame_count}")
    print(f"FPS: {FPS}")
    print(f"Duration: {DURATION_SECONDS:g} seconds")
    print()

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

    print()
    print("Finished.")


if __name__ == "__main__":
    main()
