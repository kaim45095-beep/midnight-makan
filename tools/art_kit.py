"""
art_kit.py
----------
Shared drawing helpers for my asset generator scripts. Every image in
Midnight Makan is drawn with Pillow (PIL) code instead of being downloaded,
so the whole art style is consistent and I own all of it.

Main ideas used (from the lectures):
  * Supersampling: shapes are drawn 4x bigger, then shrunk with LANCZOS
    resampling (an interpolation filter). This removes jagged edges
    (aliasing, a common digital artefact).
  * Colour models: colours are RGB tuples; shade() and tint() move a colour
    towards black or white, and hsv() builds colours from hue/sat/value.
  * Image processing: GaussianBlur for soft shadows, glows and bokeh.

Coordinates in PIL start at the TOP-LEFT corner (y grows downwards),
which is different from Arcade (bottom-left). All helpers here use PIL
coordinates in the FINAL (small) pixel size; the helpers multiply by SS.
"""
import colorsys
import math
import os
import random

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

SS = 4  # supersampling factor

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(TOOLS_DIR)
IMAGE_DIR = os.path.join(PROJECT_DIR, "assets", "images")
FONT_DIR = os.path.join(PROJECT_DIR, "assets", "fonts")

# ----------------------------------------------------------------------
# Palette - use these names so every asset matches
# ----------------------------------------------------------------------
PALETTE = {
    # night & sky
    "night_deep": (14, 16, 38),
    "night": (26, 29, 66),
    "night_blue": (44, 52, 110),
    "dusk_purple": (86, 58, 120),
    "moon": (255, 243, 214),
    # warm lights
    "lantern_red": (214, 48, 49),
    "lantern_orange": (255, 140, 50),
    "gold": (255, 200, 87),
    "warm_light": (255, 225, 160),
    "cream": (255, 244, 224),
    # materials
    "wood_dark": (92, 58, 38),
    "wood": (140, 92, 58),
    "wood_light": (196, 144, 96),
    "steel": (168, 176, 186),
    "steel_dark": (92, 100, 112),
    "tile_green": (72, 160, 120),
    "tile_white": (236, 240, 232),
    "brick": (150, 70, 60),
    # skin tones
    "skin_1": (246, 208, 170),
    "skin_2": (222, 170, 128),
    "skin_3": (176, 118, 80),
    "skin_4": (128, 84, 56),
    # hair
    "hair_black": (34, 30, 40),
    "hair_grey": (190, 190, 200),
    "hair_brown": (92, 58, 40),
    # outline for characters, food and props (dark warm, never pure black)
    "outline": (44, 30, 38),
    # food
    "cucumber": (98, 170, 70),
    "pineapple": (250, 200, 60),
    "mango": (255, 170, 40),
    "jicama": (240, 232, 210),
    "jambu": (230, 70, 90),
    "chili": (220, 30, 30),
    "noodle": (240, 225, 190),
    "prawn": (250, 130, 90),
    "egg": (255, 215, 90),
    "sprout": (240, 240, 210),
    "tea": (120, 70, 35),
    "milk_tea": (200, 145, 95),
    "kopi": (95, 60, 35),
    "milo": (110, 70, 45),
    "rice": (250, 250, 245),
    "sambal": (200, 50, 30),
    # ui
    "ui_panel": (24, 26, 52),
    "ui_panel_light": (48, 52, 96),
    "good_green": (110, 220, 130),
    "bad_red": (240, 84, 84),
    "teal": (46, 196, 182),
    "lavender": (190, 150, 230),
    "pink": (236, 96, 170),
    "sky_blue": (120, 190, 255),
}


def color(name, alpha=255):
    """Palette colour with alpha, e.g. color("gold", 128)."""
    r, g, b = PALETTE[name]
    return (r, g, b, alpha)


def shade(rgb, amount=0.22, alpha=255):
    """Darker version of a colour (moves each channel towards 0)."""
    r, g, b = rgb[:3]
    k = 1.0 - amount
    return (int(r * k), int(g * k), int(b * k), alpha)


def tint(rgb, amount=0.3, alpha=255):
    """Lighter version of a colour (moves each channel towards 255)."""
    r, g, b = rgb[:3]
    return (int(r + (255 - r) * amount), int(g + (255 - g) * amount),
            int(b + (255 - b) * amount), alpha)


def mix(c1, c2, t):
    """Linear interpolation between two colours (t = 0..1)."""
    out = []
    for i in range(3):
        out.append(int(c1[i] + (c2[i] - c1[i]) * t))
    a1 = c1[3] if len(c1) > 3 else 255
    a2 = c2[3] if len(c2) > 3 else 255
    out.append(int(a1 + (a2 - a1) * t))
    return tuple(out)


def hsv(h, s, v, alpha=255):
    """Colour from the HSV model (h, s, v all 0..1)."""
    r, g, b = colorsys.hsv_to_rgb(h % 1.0, s, v)
    return (int(r * 255), int(g * 255), int(b * 255), alpha)


# ----------------------------------------------------------------------
# Canvas helpers (supersampled)
# ----------------------------------------------------------------------
def newCanvas(width, height, background=(0, 0, 0, 0)):
    """Create a big RGBA canvas (width*SS x height*SS) and its drawer."""
    img = Image.new("RGBA", (width * SS, height * SS), background)
    return img, ImageDraw.Draw(img)


def finish(img, width, height):
    """Shrink a supersampled canvas to the final size (anti-aliasing)."""
    return img.resize((width, height), Image.LANCZOS)


def S(value):
    """Scale a final-size number (or tuple / list of numbers) to canvas size."""
    if isinstance(value, (list, tuple)):
        return type(value)(S(v) for v in value)
    return value * SS


def box(x0, y0, x1, y1):
    return [x0 * SS, y0 * SS, x1 * SS, y1 * SS]


def pts(points):
    """Scale a list of (x, y) points to canvas size."""
    return [(x * SS, y * SS) for x, y in points]


# ----------------------------------------------------------------------
# Shapes with outline + shading (the house style)
# ----------------------------------------------------------------------
def ellipse(draw, x0, y0, x1, y1, fill, outline=None, width=3):
    draw.ellipse(box(x0, y0, x1, y1), fill=fill,
                 outline=outline, width=int(width * SS) if outline else 0)


def polygon(draw, points, fill, outline=None, width=3):
    draw.polygon(pts(points), fill=fill)
    if outline:
        closed = list(points) + [points[0]]
        draw.line(pts(closed), fill=outline, width=int(width * SS), joint="curve")


def roundedRect(draw, x0, y0, x1, y1, radius, fill, outline=None, width=3):
    draw.rounded_rectangle(box(x0, y0, x1, y1), radius=radius * SS, fill=fill,
                           outline=outline, width=int(width * SS) if outline else 0)


def line(draw, points, fill, width=3):
    draw.line(pts(points), fill=fill, width=int(width * SS), joint="curve")
    # round caps so thick lines look like limbs / strokes
    r = width * SS / 2
    for x, y in (points[0], points[-1]):
        draw.ellipse([x * SS - r, y * SS - r, x * SS + r, y * SS + r], fill=fill)


def bezier(p0, p1, p2, p3=None, steps=24):
    """Points along a quadratic (3 points) or cubic (4 points) Bezier curve."""
    out = []
    for i in range(steps + 1):
        t = i / steps
        if p3 is None:
            x = (1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t * t * p2[0]
            y = (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t * t * p2[1]
        else:
            x = ((1 - t) ** 3 * p0[0] + 3 * (1 - t) ** 2 * t * p1[0]
                 + 3 * (1 - t) * t * t * p2[0] + t ** 3 * p3[0])
            y = ((1 - t) ** 3 * p0[1] + 3 * (1 - t) ** 2 * t * p1[1]
                 + 3 * (1 - t) * t * t * p2[1] + t ** 3 * p3[1])
        out.append((x, y))
    return out


def circlePoints(cx, cy, rx, ry, count=48, start=0.0, end=360.0):
    out = []
    for i in range(count + 1):
        a = math.radians(start + (end - start) * i / count)
        out.append((cx + rx * math.cos(a), cy + ry * math.sin(a)))
    return out


# ----------------------------------------------------------------------
# Gradients, glows and shadows (image processing)
# ----------------------------------------------------------------------
def linearGradient(width, height, top, bottom):
    """Vertical gradient image (final size, not supersampled)."""
    img = Image.new("RGBA", (width, height))
    draw = ImageDraw.Draw(img)
    for y in range(height):
        draw.line([(0, y), (width, y)], fill=mix(top, bottom, y / max(1, height - 1)))
    return img


def radialGradient(size, inner, outer, power=1.6):
    """Square image with a round gradient from the centre (inner) to the edge (outer)."""
    img = Image.new("RGBA", (size, size))
    px = img.load()
    c = (size - 1) / 2
    for y in range(size):
        for x in range(size):
            d = math.hypot(x - c, y - c) / c
            t = min(1.0, d) ** power
            px[x, y] = mix(inner, outer, t)
    return img


def softGlow(size, rgb=(255, 255, 255), power=2.0):
    """White (or coloured) round glow that fades to transparent - for lights."""
    return radialGradient(size, (rgb[0], rgb[1], rgb[2], 255), (rgb[0], rgb[1], rgb[2], 0), power)


def blur(img, radius):
    return img.filter(ImageFilter.GaussianBlur(radius))


def glowFromAlpha(img, radius, rgb, strength=1.0):
    """A blurred coloured silhouette of an image - used behind glowing things."""
    alpha = img.getchannel("A").filter(ImageFilter.GaussianBlur(radius))
    if strength != 1.0:
        alpha = alpha.point(lambda a: min(255, int(a * strength)))
    glow = Image.new("RGBA", img.size, (rgb[0], rgb[1], rgb[2], 0))
    glow.putalpha(alpha)
    return glow


def dropShadow(img, offset=(0, 6), radius=6, opacity=110):
    """Return a new image = soft shadow under the original image."""
    pad = radius * 3 + max(abs(offset[0]), abs(offset[1]))
    w, h = img.size
    out = Image.new("RGBA", (w + pad * 2, h + pad * 2), (0, 0, 0, 0))
    shadowAlpha = img.getchannel("A").point(lambda a: int(a * opacity / 255))
    shadow = Image.new("RGBA", img.size, (0, 0, 0, 0))
    shadow.putalpha(shadowAlpha)
    out.alpha_composite(shadow, (pad + offset[0], pad + offset[1]))
    out = out.filter(ImageFilter.GaussianBlur(radius))
    out.alpha_composite(img, (pad, pad))
    return out


def addNoise(img, amount=10, seed=1):
    """Add subtle film grain so big flat areas do not look too digital."""
    rnd = random.Random(seed)
    noise = Image.new("L", img.size)
    noise.putdata([128 + rnd.randint(-amount, amount) for _ in range(img.size[0] * img.size[1])])
    rgb = img.convert("RGB")
    grain = Image.merge("RGB", (noise, noise, noise))
    mixed = ImageChops.overlay(rgb, grain)
    mixed = mixed.convert("RGBA")
    mixed.putalpha(img.getchannel("A"))
    return mixed


def bokeh(width, height, count, colors, minR, maxR, blurRadius, seed=7, alphaRange=(60, 170), area=None):
    """Blurred round light spots (camera bokeh), e.g. for night-market lights."""
    rnd = random.Random(seed)
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    x0, y0, x1, y1 = area if area else (0, 0, width, height)
    for _ in range(count):
        r = rnd.uniform(minR, maxR)
        x = rnd.uniform(x0, x1)
        y = rnd.uniform(y0, y1)
        c = rnd.choice(colors)
        a = rnd.randint(alphaRange[0], alphaRange[1])
        draw.ellipse([x - r, y - r, x + r, y + r], fill=(c[0], c[1], c[2], a))
    return img.filter(ImageFilter.GaussianBlur(blurRadius))


# ----------------------------------------------------------------------
# Text (fonts come from assets/fonts so the tool works on any computer)
# ----------------------------------------------------------------------
FONT_FILES = {
    "future": "Kenney_Future.ttf",
    "narrow": "Kenney_Future_Narrow.ttf",
    "pixel": "Kenney_Pixel.ttf",
    "sans": "Liberation_Sans_Regular.ttf",
    "sans_bold": "Liberation_Sans_Bold.ttf",
    "sans_italic": "Liberation_Sans_Italic.ttf",
}


def font(name, size):
    """Load a bundled font at a FINAL pixel size (multiplied by SS for canvases)."""
    return ImageFont.truetype(os.path.join(FONT_DIR, FONT_FILES[name]), int(size))


def drawText(draw, x, y, text, fontName, size, fill, anchor="la", stroke=0, strokeFill=None):
    """Draw text on a supersampled canvas using final-size coordinates."""
    f = font(fontName, size * SS)
    draw.text((x * SS, y * SS), text, font=f, fill=fill, anchor=anchor,
              stroke_width=int(stroke * SS), stroke_fill=strokeFill)


# ----------------------------------------------------------------------
# Saving and checking
# ----------------------------------------------------------------------
def save(img, relativePath):
    """Save into assets/images/<relativePath> and print the size."""
    path = os.path.join(IMAGE_DIR, relativePath)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path, optimize=True)
    print(f"saved {relativePath:48s} {img.size[0]}x{img.size[1]}")
    return path


def contactSheet(relativePaths, outPath, cell=220, columns=6, background=(40, 40, 60, 255)):
    """Put many images on one sheet (checkerboard behind) so they can be reviewed at once."""
    rows = (len(relativePaths) + columns - 1) // columns
    sheet = Image.new("RGBA", (columns * cell, rows * (cell + 20)), background)
    draw = ImageDraw.Draw(sheet)
    small = ImageFont.truetype(os.path.join(FONT_DIR, FONT_FILES["sans"]), 12)
    for i, rel in enumerate(relativePaths):
        path = os.path.join(IMAGE_DIR, rel)
        cx = (i % columns) * cell
        cy = (i // columns) * (cell + 20)
        # checkerboard shows transparency
        for yy in range(0, cell, 16):
            for xx in range(0, cell, 16):
                if (xx // 16 + yy // 16) % 2 == 0:
                    draw.rectangle([cx + xx, cy + yy, cx + xx + 15, cy + yy + 15], fill=(70, 70, 90, 255))
        if os.path.exists(path):
            im = Image.open(path).convert("RGBA")
            im.thumbnail((cell - 8, cell - 8), Image.LANCZOS)
            sheet.alpha_composite(im, (cx + (cell - im.size[0]) // 2, cy + (cell - im.size[1]) // 2))
        else:
            draw.text((cx + 8, cy + 8), "MISSING", fill=(255, 80, 80, 255), font=small)
        draw.text((cx + 4, cy + cell + 2), os.path.basename(rel)[:34], fill=(230, 230, 230, 255), font=small)
    os.makedirs(os.path.dirname(os.path.abspath(outPath)), exist_ok=True)
    sheet.save(outPath)
    return outPath


def spriteSheet(frames, columns):
    """Join same-size frames into one sprite sheet (left-to-right, top-to-bottom)."""
    fw, fh = frames[0].size
    rows = (len(frames) + columns - 1) // columns
    sheet = Image.new("RGBA", (fw * columns, fh * rows), (0, 0, 0, 0))
    for i, fr in enumerate(frames):
        sheet.alpha_composite(fr, ((i % columns) * fw, (i // columns) * fh))
    return sheet
