from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw


def generate_portrait_fixture(width: int = 64, height: int = 64) -> Image.Image:
    """Generate a deterministic portrait-like grayscale/skin-tone synthetic image."""
    img = Image.new("RGB", (width, height), (240, 240, 240))
    draw = ImageDraw.Draw(img)

    # Background gradient
    for y in range(height):
        val = int(200 - 50 * (y / height))
        draw.line([(0, y), (width, y)], fill=(val, val, val))

    # Face oval box
    face_box = [
        int(width * 0.25),
        int(height * 0.2),
        int(width * 0.75),
        int(height * 0.85),
    ]
    # Base skin-tone gray color
    draw.ellipse(face_box, fill=(180, 180, 180))

    # Left-shifted center for radial shading (modulating facial volume)
    cx, cy = int(width * 0.45), int(height * 0.45)
    y_indices, x_indices = np.mgrid[0:height, 0:width]
    dist = np.sqrt((x_indices - cx) ** 2 + (y_indices - cy) ** 2)
    max_d = np.sqrt(width**2 + height**2) / 2
    shading = np.clip(1.0 - (dist / max_d) * 0.5, 0.0, 1.0)

    # Convert to array and mask the shading
    arr = np.array(img, dtype=np.float32)
    mask_img = Image.new("L", (width, height), 0)
    mask_draw = ImageDraw.Draw(mask_img)
    mask_draw.ellipse(face_box, fill=255)
    mask = np.array(mask_img, dtype=np.float32) / 255.0

    for c in range(3):
        arr[:, :, c] = arr[:, :, c] * (1.0 - mask) + (arr[:, :, c] * shading) * mask

    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(img)

    # Draw facial features: Eyes
    eye_y = int(height * 0.45)
    draw.ellipse(
        [int(width * 0.38), eye_y - 2, int(width * 0.46), eye_y + 2],
        fill=(50, 50, 50),
    )
    draw.ellipse(
        [int(width * 0.54), eye_y - 2, int(width * 0.62), eye_y + 2],
        fill=(50, 50, 50),
    )

    # Nose
    draw.polygon(
        [
            (int(width * 0.5), int(height * 0.48)),
            (int(width * 0.47), int(height * 0.6)),
            (int(width * 0.53), int(height * 0.6)),
        ],
        fill=(120, 120, 120),
    )

    # Mouth arc
    draw.arc(
        [int(width * 0.4), int(height * 0.65), int(width * 0.6), int(height * 0.72)],
        start=0,
        end=180,
        fill=(80, 80, 80),
        width=2,
    )

    return img


def generate_landscape_fixture(width: int = 64, height: int = 64) -> Image.Image:
    """Generate a deterministic landscape-like gradients/edges image."""
    img = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(img)

    # Sky gradient
    for y in range(height):
        val = int(230 - (230 - 100) * (y / height))
        draw.line([(0, y), (width, y)], fill=(val, val, val))

    # Sun
    draw.ellipse(
        [int(width * 0.7), int(height * 0.15), int(width * 0.85), int(height * 0.3)],
        fill=(255, 255, 255),
    )

    # Distant mountain ridge (smooth sine wave)
    mountain1_pts = [(0, height)]
    for x in range(0, width + 1):
        y = int(height * 0.55 + 6 * np.sin(x * 0.12) + 2 * np.sin(x * 0.36))
        mountain1_pts.append((x, y))
    mountain1_pts.append((width, height))
    draw.polygon(mountain1_pts, fill=(140, 140, 140))

    # Near mountain ridge (rougher seed-based random walk)
    mountain2_pts = [(0, height)]
    rng = np.random.default_rng(12345)
    curr_y = int(height * 0.75)
    for x in range(0, width + 1, 4):
        curr_y += int(rng.integers(-3, 3))
        curr_y = np.clip(curr_y, int(height * 0.65), int(height * 0.9))
        mountain2_pts.append((x, curr_y))
    mountain2_pts.append((width, curr_y))
    mountain2_pts.append((width, height))
    draw.polygon(mountain2_pts, fill=(70, 70, 70))

    return img


def generate_line_art_fixture(width: int = 64, height: int = 64) -> Image.Image:
    """Generate a deterministic line art image with overlapping vector contours."""
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Concentric circles
    for r in range(6, 28, 5):
        draw.ellipse(
            [width // 2 - r, height // 2 - r, width // 2 + r, height // 2 + r],
            outline=(0, 0, 0),
            width=1,
        )

    # Radial lines
    for angle in range(0, 360, 30):
        rad = np.radians(angle)
        x2 = int(width // 2 + 28 * np.cos(rad))
        y2 = int(height // 2 + 28 * np.sin(rad))
        draw.line(
            [(width // 2, height // 2), (x2, y2)],
            fill=(0, 0, 0),
            width=1 if angle % 60 == 0 else 2,
        )

    # Intersecting arcs
    draw.arc([4, 4, width - 4, height - 4], start=45, end=135, fill=(0, 0, 0), width=3)
    draw.arc([8, 8, width - 8, height - 8], start=225, end=315, fill=(0, 0, 0), width=1)

    return img


def generate_noisy_scan_fixture(width: int = 64, height: int = 64) -> Image.Image:
    """Generate a deterministic noisy scan/document image with simulated dust/dirt."""
    img = Image.new("RGB", (width, height), (245, 245, 240))
    draw = ImageDraw.Draw(img)

    # Simulated text lines
    rng = np.random.default_rng(999)
    for y in range(8, height - 8, 8):
        margin_left = rng.integers(5, 12)
        margin_right = rng.integers(5, 12)
        x = margin_left
        while x < width - margin_right:
            word_len = rng.integers(4, 10)
            if x + word_len > width - margin_right:
                word_len = width - margin_right - x
            if word_len > 2:
                draw.rectangle([x, y, x + word_len - 1, y + 3], fill=(45, 45, 50))
            x += word_len

    # Add scan sensor noise
    arr = np.array(img, dtype=np.float32)
    noise = rng.normal(0, 15, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)

    # Add dark specks/dust
    for _ in range(8):
        sx = rng.integers(0, width)
        sy = rng.integers(0, height)
        sz = rng.integers(1, 2)
        arr[sy:sy + sz, sx:sx + sz, :] = rng.integers(30, 90)

    return Image.fromarray(arr)


def generate_high_contrast_fixture(width: int = 64, height: int = 64) -> Image.Image:
    """Generate a deterministic high-contrast graphic with sharp geometric boundaries."""
    img = Image.new("RGB", (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Border frame
    draw.rectangle(
        [int(width * 0.08), int(height * 0.08), int(width * 0.92), int(height * 0.92)],
        outline=(255, 255, 255),
        width=2,
    )

    # Center fill
    draw.rectangle(
        [int(width * 0.25), int(height * 0.25), int(width * 0.75), int(height * 0.75)],
        fill=(255, 255, 255),
    )

    # Core negative triangle
    draw.polygon(
        [
            (width // 2, int(height * 0.32)),
            (int(width * 0.32), int(height * 0.68)),
            (int(width * 0.68), int(height * 0.68)),
        ],
        fill=(0, 0, 0),
    )

    # Checkerboard in the corner
    chk_size = 8
    for y in range(0, height, chk_size):
        for x in range(0, width, chk_size):
            if ((x // chk_size) + (y // chk_size)) % 2 == 0:
                if x < width // 3 and y > 2 * height // 3:
                    draw.rectangle([x, y, x + chk_size, y + chk_size], fill=(255, 255, 255))

    return img


def generate_low_contrast_fixture(width: int = 64, height: int = 64) -> Image.Image:
    """Generate a deterministic low-contrast photo-like image (fog/clouds)."""
    img = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(img)

    # Low-contrast background wave
    for y in range(height):
        val = int(115 + 25 * np.sin(y * np.pi / height))
        draw.line([(0, y), (width, y)], fill=(val, val, val))

    # Add soft modulated textures
    arr = np.array(img, dtype=np.float32)
    y_indices, x_indices = np.mgrid[0:height, 0:width]
    dist = np.sqrt((x_indices - width // 2) ** 2 + (y_indices - height // 2) ** 2)
    mod = 12 * np.cos(dist * 0.16)

    arr[:, :, 0] += mod
    arr[:, :, 1] += mod
    arr[:, :, 2] += mod

    return Image.fromarray(np.clip(arr, 95, 145).astype(np.uint8))


def generate_icon_fixture(width: int = 48, height: int = 48) -> Image.Image:
    """Generate a deterministic small icon/illustration image (e.g., stylized gear)."""
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    cx, cy = width // 2, height // 2
    r_outer = int(width * 0.4)
    r_inner = int(width * 0.22)

    # Draw gear teeth
    for angle in range(0, 360, 45):
        rad = np.radians(angle)
        x1 = int(cx + r_outer * np.cos(rad))
        y1 = int(cy + r_outer * np.sin(rad))
        draw.line([(cx, cy), (x1, y1)], fill=(210, 110, 60), width=4)

    # Draw body ring & center hole
    draw.ellipse(
        [cx - r_inner, cy - r_inner, cx + r_inner, cy + r_inner],
        fill=(100, 160, 210),
    )
    draw.ellipse(
        [cx - r_inner // 2, cy - r_inner // 2, cx + r_inner // 2, cy + r_inner // 2],
        fill=(255, 255, 255),
    )

    return img


FIXTURES = {
    "portrait": generate_portrait_fixture,
    "landscape": generate_landscape_fixture,
    "line_art": generate_line_art_fixture,
    "noisy_scan": generate_noisy_scan_fixture,
    "high_contrast": generate_high_contrast_fixture,
    "low_contrast": generate_low_contrast_fixture,
    "icon": generate_icon_fixture,
}
