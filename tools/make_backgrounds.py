"""
make_backgrounds.py
-------------------
Draws every full-screen background and parallax layer of Midnight Makan
with Pillow + NumPy code (no downloaded art). Run from the project root:

    python tools/make_backgrounds.py              (all backgrounds)
    python tools/make_backgrounds.py street_main  (only some, for testing)

How one background is made - a tiny "renderer" built from lecture ideas:
  1. Shapes are drawn 4x bigger and shrunk with LANCZOS resampling
     (supersampling = anti-aliasing, see art_kit.py).
  2. The COLOUR layer is painted in normal "daytime" colours.
  3. Night lighting (Week 8 lighting): every pixel is multiplied by a dim
     blue AMBIENT colour and warm POINT LIGHTS are added where lamps are:
         lit colour = colour * (ambient + sum of point lights)
  4. Glowing things (bulbs, windows, neon, lanterns) are painted on a
     second GLOW layer that is not darkened. A blurred copy of it is added
     on top (bloom), so lights look bright and soft.
  5. Atmospheric perspective: far layers are mixed towards a blue haze
     colour (farther = bluer and lower contrast).
  6. Film grain (random noise) hides colour banding in big gradients.

PIL coordinates start at the TOP-LEFT (y grows downwards). The game uses
Arcade's bottom-left origin, so Arcade y = 720 - PIL y.
"""
import math
import os
import random
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TOOLS_DIR)
import art_kit  # noqa: E402  shared helpers (supersampling, palette, fonts, save)

SS = art_kit.SS
P = art_kit.PALETTE
box = art_kit.box
shade = art_kit.shade
tint = art_kit.tint

# ----------------------------------------------------------------------
# Layout numbers shared with the game code (see DESIGN_SPEC section 4)
# ----------------------------------------------------------------------
MOON_X, MOON_Y, MOON_R = 1030, 150, 62   # moon on title_sky.png (PIL coords)
STALL_XS = [620, 1320, 2020, 2720]      # settings.STALL_X (stall centres)
STALL_BASE_Y = 588                      # where the stall fronts stand on the pavement
FEET_Y = 602                            # Aiman's feet: Arcade y 118 -> PIL 720 - 118
OYEN_CRATE_X = 1650                     # Oyen sits on this crate in the street
OYEN_CRATE_TOP = 522                    # PIL y of the crate lid (Arcade y 198)

WARM_WINDOWS = [(255, 214, 150), (255, 196, 120), (255, 228, 176), (250, 182, 112)]
COOL_WINDOWS = [(176, 204, 255), (206, 224, 255)]
GOLD = P["gold"]
BLOOM = ((3, 0.55), (10, 0.45), (32, 0.35))


# ======================================================================
# 1. Small helpers (colours, drawing on a supersampled canvas)
# ======================================================================
def rgba(rgb, alpha=255):
    return (int(rgb[0]), int(rgb[1]), int(rgb[2]), int(alpha))


def jitter(rgb, rnd, amount):
    """Slightly random version of a colour so repeated shapes do not look copied."""
    out = []
    for c in rgb[:3]:
        out.append(max(0, min(255, int(c) + rnd.randint(-amount, amount))))
    return (out[0], out[1], out[2], 255)


def rect(draw, x0, y0, x1, y1, fill):
    draw.rectangle(box(x0, y0, x1, y1), fill=fill)


def ellipse(draw, cx, cy, rx, ry, fill):
    draw.ellipse(box(cx - rx, cy - ry, cx + rx, cy + ry), fill=fill)


def ring(draw, cx, cy, rx, ry, fill, width):
    draw.ellipse(box(cx - rx, cy - ry, cx + rx, cy + ry), outline=fill, width=max(1, int(width * SS)))


def poly(draw, points, fill):
    draw.polygon(art_kit.pts(points), fill=fill)


def rrect(draw, x0, y0, x1, y1, radius, fill):
    draw.rounded_rectangle(box(x0, y0, x1, y1), radius=radius * SS, fill=fill)


def stroke(draw, points, fill, width):
    art_kit.line(draw, points, fill, width)


def halfDisc(draw, cx, cy, r, fill, top=True):
    """Half a circle (top half = arch, bottom half = scallop)."""
    if top:
        draw.pieslice(box(cx - r, cy - r, cx + r, cy + r), 180, 360, fill=fill)
    else:
        draw.pieslice(box(cx - r, cy - r, cx + r, cy + r), 0, 180, fill=fill)


def sag(x0, y0, x1, y1, drop, steps=40):
    """Points of a hanging wire (a parabola is a good fake catenary)."""
    out = []
    for i in range(steps + 1):
        t = i / steps
        out.append((x0 + (x1 - x0) * t, y0 + (y1 - y0) * t + drop * 4 * t * (1 - t)))
    return out


def leafPoints(x, y, length, width, angle, steps=10):
    """A pointed leaf shape starting at (x, y) and growing in direction 'angle' (degrees)."""
    a = math.radians(angle)
    ca, sa = math.cos(a), math.sin(a)
    outline = []
    for i in range(steps + 1):
        t = i / steps
        outline.append((t * length, math.sin(t * math.pi) * width / 2))
    for i in range(steps, -1, -1):
        t = i / steps
        outline.append((t * length, -math.sin(t * math.pi) * width / 2 * 0.8))
    result = []
    for (u, v) in outline:
        # 2D rotation matrix, then translate (Week 5-6 transformations)
        result.append((x + u * ca - v * sa, y + u * sa + v * ca))
    return result


def fitSize(words, fontName, maxWidth, maxSize, minSize=6.0):
    """Biggest font size (final pixels) that makes the text fit into maxWidth."""
    size = maxSize
    while size > minSize:
        f = art_kit.font(fontName, size * SS)
        left, top, right, bottom = f.getbbox(words)
        if (right - left) / SS <= maxWidth:
            return size
        size -= 0.5
    return minSize


def signFont(words, fontName, maxSize):
    """Kenney Future looks great, but when it is small its K looks like an H and its R
    like an A ("PAK" reads "PAH"). Text that must be spelled right then uses Liberation
    Sans Bold instead."""
    if fontName not in ("future", "narrow"):
        return fontName
    if "K" in words.upper() or maxSize < 13 or ("R" in words.upper() and maxSize < 24):
        return "sans_bold"
    return fontName


def label(draw, cx, cy, words, fontName, maxWidth, maxSize, fill, shadow=None, shadowOffset=(0, 1.5)):
    """Centred text that shrinks itself to fit. Optional darker drop copy for depth."""
    fontName = signFont(words, fontName, maxSize)
    size = fitSize(words, fontName, maxWidth, maxSize)
    if shadow is not None:
        art_kit.drawText(draw, cx + shadowOffset[0], cy + shadowOffset[1], words, fontName, size,
                         shadow, anchor="mm")
    art_kit.drawText(draw, cx, cy, words, fontName, size, fill, anchor="mm")
    return size


# ======================================================================
# 2. NumPy image helpers (fast - no Python loops over pixels)
# ======================================================================
def toArray(img):
    return np.array(img.convert("RGBA"), dtype=np.float32)


def toImage(arr):
    return Image.fromarray(np.clip(arr + 0.5, 0, 255).astype(np.uint8), "RGBA")


def grid(width, height):
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
    return xs, ys


def distance(width, height, cx, cy, sx=1.0, sy=1.0):
    xs, ys = grid(width, height)
    return np.sqrt(((xs - cx) / sx) ** 2 + ((ys - cy) / sy) ** 2)


def gradientArray(width, height, stops):
    """Vertical gradient with several colour stops [(position 0..1, (r, g, b)), ...]."""
    t = np.linspace(0.0, 1.0, height, dtype=np.float32)
    arr = np.zeros((height, width, 4), np.float32)
    positions = [s[0] for s in stops]
    for ch in range(3):
        column = np.interp(t, positions, [s[1][ch] for s in stops])
        arr[:, :, ch] = column[:, None]
    arr[:, :, 3] = 255
    return arr


def smoothNoise(width, height, cell, seed):
    """Soft random pattern 0..1: a small random grid enlarged with BICUBIC interpolation."""
    rnd = np.random.default_rng(seed)
    gw = width // cell + 4
    gh = height // cell + 4
    small = Image.fromarray(rnd.random((gh, gw)).astype(np.float32), "F")
    big = small.resize((gw * cell, gh * cell), Image.BICUBIC)
    arr = np.array(big, np.float32)[cell:cell + height, cell:cell + width]
    return np.clip(arr, 0.0, 1.0)


def addGrain(arr, amount, seed):
    """Film grain: small random brightness changes (hides banding in gradients)."""
    rnd = np.random.default_rng(seed)
    noise = rnd.normal(0.0, amount, arr.shape[:2]).astype(np.float32)
    arr[:, :, :3] += noise[:, :, None]


def applyVignette(arr, strength, power=2.4):
    h, w = arr.shape[:2]
    d = distance(w, h, w / 2, h / 2, w / 2, h / 2) / math.sqrt(2)
    arr[:, :, :3] *= (1.0 - strength * np.clip(d, 0, 1) ** power)[:, :, None]


def applySaturation(arr, amount):
    lum = (arr[:, :, 0] * 0.299 + arr[:, :, 1] * 0.587 + arr[:, :, 2] * 0.114)[:, :, None]
    arr[:, :, :3] = lum + (arr[:, :, :3] - lum) * amount


def applyHaze(img, hazeRgb, topAmount, bottomAmount):
    """Atmospheric perspective: mix colours towards a haze colour (more at the bottom)."""
    arr = toArray(img)
    t = np.linspace(topAmount, bottomAmount, arr.shape[0], dtype=np.float32)[:, None, None]
    arr[:, :, :3] = arr[:, :, :3] * (1 - t) + np.array(hazeRgb, np.float32) * t
    return toImage(arr)


def lightMap(width, height, lights):
    """Sum of soft round point lights. light = (x, y, radius, rgb, strength, squash)."""
    total = np.zeros((height, width, 3), np.float32)
    for (x, y, radius, rgb, strength, squash) in lights:
        ry = radius * squash
        x0, x1 = max(0, int(x - radius)), min(width, int(x + radius) + 1)
        y0, y1 = max(0, int(y - ry)), min(height, int(y + ry) + 1)
        if x0 >= x1 or y0 >= y1:
            continue
        ys, xs = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        d = np.sqrt(((xs - x) / radius) ** 2 + ((ys - y) / ry) ** 2)
        t = np.clip(1.0 - d, 0.0, 1.0)
        falloff = t * t * (3.0 - 2.0 * t) * (0.35 + 0.65 * t)   # smooth edge, brighter centre
        colour = np.array(rgb[:3], np.float32) / 255.0 * strength
        total[y0:y1, x0:x1] += falloff[:, :, None] * colour
    return total


def premultiplied(img):
    arr = np.array(img, np.float32)
    return np.clip(arr[:, :, :3] * (arr[:, :, 3:4] / 255.0), 0, 255).astype(np.uint8)


def blurEnergy(premArr, radius):
    blurred = Image.fromarray(premArr, "RGB").filter(ImageFilter.GaussianBlur(radius))
    return np.array(blurred, np.float32)


def addEnergy(arr, energy):
    """Add light 'energy' (additive blending). Works for opaque AND transparent images:
    on transparent pixels the glow creates its own soft alpha."""
    a = arr[:, :, 3:4] / 255.0
    prem = arr[:, :, :3] * a + energy
    glowA = np.clip(energy.max(axis=2, keepdims=True) / 255.0, 0, 1)
    newA = np.maximum(a, glowA)
    arr[:, :, :3] = prem / np.maximum(newA, 1e-4)
    arr[:, :, 3:4] = newA * 255.0


def bokehEnergy(width, height, spots):
    """Out-of-focus light discs (camera bokeh): soft edge, slightly brighter rim."""
    energy = np.zeros((height, width, 3), np.float32)
    for (x, y, r, rgb, strength) in spots:
        x0, x1 = max(0, int(x - r - 3)), min(width, int(x + r + 4))
        y0, y1 = max(0, int(y - r - 3)), min(height, int(y + r + 4))
        if x0 >= x1 or y0 >= y1:
            continue
        ys, xs = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        d = np.sqrt((xs - x) ** 2 + (ys - y) ** 2)
        soft = 1.2 + r * 0.12
        disc = np.clip((r - d) / soft + 0.5, 0, 1)
        rim = 0.7 + 0.3 * np.clip(d / r, 0, 1) ** 4
        energy[y0:y1, x0:x1] += (disc * rim * strength)[:, :, None] * np.array(rgb, np.float32)
    return energy


def pasteOver(base, img, left, top):
    """alpha_composite that also works when the image hangs over the edge."""
    left, top = int(round(left)), int(round(top))
    sx, sy = max(0, -left), max(0, -top)
    dx, dy = max(0, left), max(0, top)
    w = min(img.width - sx, base.width - dx)
    h = min(img.height - sy, base.height - dy)
    if w > 0 and h > 0:
        base.alpha_composite(img, (dx, dy), (sx, sy, sx + w, sy + h))


def softShape(width, height, shapes, rgb, opacity, radius):
    """Blurred shapes as an RGBA overlay - soft shadows, grease stains, haze patches.
    shapes = [("ellipse" | "rect", x0, y0, x1, y1), ...]"""
    mask = Image.new("L", (width, height), 0)
    md = ImageDraw.Draw(mask)
    for (kind, x0, y0, x1, y1) in shapes:
        if kind == "ellipse":
            md.ellipse([x0, y0, x1, y1], fill=255)
        else:
            md.rectangle([x0, y0, x1, y1], fill=255)
    if radius > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius))
    alpha = (np.array(mask, np.float32) * (opacity / 255.0)).astype(np.uint8)
    out = Image.new("RGBA", (width, height), rgba(rgb, 0))
    out.putalpha(Image.fromarray(alpha, "L"))
    return out


def verticalFade(width, height, rgb, alphaTop, alphaBottom):
    """RGBA layer with one colour whose alpha fades from top to bottom (haze bands)."""
    arr = np.zeros((height, width, 4), np.float32)
    arr[:, :, :3] = np.array(rgb, np.float32)
    arr[:, :, 3] = np.linspace(alphaTop, alphaBottom, height, dtype=np.float32)[:, None]
    return toImage(arr)


# ======================================================================
# 3. Piece and Picture - my two drawing surfaces
# ======================================================================
class Piece:
    """One supersampled drawing (a stall, a building, a lamp...) with two layers:
    'draw' = colour layer (lit later by the night lighting) and
    'glow' = glowing parts (never darkened, they also create bloom)."""

    def __init__(self, width, height):
        self.width = int(math.ceil(width))
        self.height = int(math.ceil(height))
        self.img, self.draw = art_kit.newCanvas(self.width, self.height)
        self.glowImg, self.glow = art_kit.newCanvas(self.width, self.height)
        self.lights = []

    def addLight(self, x, y, radius, rgb, strength, squash=1.0):
        self.lights.append((x, y, radius, rgb, strength, squash))

    def finish(self):
        colour = art_kit.finish(self.img, self.width, self.height)
        glow = None
        if self.glowImg.getbbox() is not None:
            glow = art_kit.finish(self.glowImg, self.width, self.height)
        return colour, glow


class Picture:
    """A full background at final size. Pieces are placed into it, then render() lights it."""

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.colour = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        self.glow = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        self.lights = []
        self.energy = np.zeros((height, width, 3), np.float32)   # extra light (bokeh, halos)

    def place(self, piece, left, top, haze=None):
        colour, glow = piece.finish()
        if haze is not None:
            colour = applyHaze(colour, haze[0], haze[1], haze[2])
        pasteOver(self.colour, colour, left, top)
        if glow is not None:
            pasteOver(self.glow, glow, left, top)
        for (x, y, r, rgb, s, q) in piece.lights:
            self.lights.append((x + left, y + top, r, rgb, s, q))

    def overlay(self, img, left=0, top=0):
        pasteOver(self.colour, img, left, top)

    def addLight(self, x, y, radius, rgb, strength, squash=1.0):
        self.lights.append((x, y, radius, rgb, strength, squash))

    def render(self, ambient, background=None, bloom=BLOOM, grain=2.0, seed=1, vignette=0.0,
               saturation=1.0, glowAmount=1.0):
        # step 3: colour * (ambient + point lights)
        arr = toArray(self.colour)
        light = lightMap(self.width, self.height, self.lights) + np.array(ambient, np.float32)
        arr[:, :, :3] *= light
        lit = toImage(arr)
        # step 4: glowing parts on top (not darkened)
        lit.alpha_composite(self.glow)
        if background is not None:
            base = background.copy()
            base.alpha_composite(lit)
            lit = base
        arr = toArray(lit)
        energy = self.energy.copy()
        if self.glow.getbbox() is not None:
            prem = premultiplied(self.glow)
            for radius, strength in bloom:
                energy += blurEnergy(prem, radius) * (strength * glowAmount)
        addEnergy(arr, energy)
        if saturation != 1.0:
            applySaturation(arr, saturation)
        if vignette > 0:
            applyVignette(arr, vignette)
        if grain > 0:
            addGrain(arr, grain, seed)
        return toImage(arr)


# ======================================================================
# 4. Reusable pieces of scenery
# ======================================================================
def drawBulb(p, x, y, r, rgb=(255, 214, 150), lightRadius=220, strength=0.7, socket=True):
    if socket:
        rect(p.draw, x - r * 0.45, y - r * 1.9, x + r * 0.45, y - r * 0.8, (58, 54, 56))
    ellipse(p.glow, x, y, r, r * 1.12, rgba(rgb))
    ellipse(p.glow, x - r * 0.25, y - r * 0.3, r * 0.45, r * 0.45, (255, 252, 236, 255))
    if lightRadius > 0:
        p.addLight(x, y, lightRadius, rgb, strength)


def drawRedLantern(p, x, top, w, h, lit=True, light=0.55):
    """Chinese red lantern hanging at (x, top): gold caps, ribs and a tassel."""
    target = p.glow if lit else p.draw
    gold = (236, 184, 80, 255)
    goldDark = (170, 118, 50, 255)
    cy = top + h * 0.5
    rx, ry = w / 2, h * 0.40
    rect(p.draw, x - 0.8, top - 16, x + 0.8, top + 2, (40, 30, 30))
    body = (222, 52, 42, 255) if lit else (150, 36, 36, 255)
    ellipse(target, x, cy, rx, ry, body)
    ellipse(target, x - rx * 0.12, cy - ry * 0.08, rx * 0.66, ry * 0.72,
            (255, 112, 64, 255) if lit else (176, 50, 44, 255))
    ellipse(target, x - rx * 0.22, cy - ry * 0.2, rx * 0.28, ry * 0.34,
            (255, 196, 120, 255) if lit else (196, 70, 60, 255))
    for k in (0.38, 0.72):
        ring(target, x, cy, rx * k, ry, (170, 34, 34, 255) if lit else (110, 26, 26, 255), 0.9)
    rect(target, x - rx * 0.5, cy - ry - h * 0.08, x + rx * 0.5, cy - ry + h * 0.03, gold)
    rect(target, x - rx * 0.5, cy - ry + h * 0.01, x + rx * 0.5, cy - ry + h * 0.03, goldDark)
    rect(target, x - rx * 0.5, cy + ry - h * 0.03, x + rx * 0.5, cy + ry + h * 0.07, gold)
    stroke(target, [(x, cy + ry + h * 0.07), (x, top + h * 1.12)], (200, 40, 36, 255), max(1.2, w * 0.06))
    poly(target, [(x - w * 0.09, top + h * 1.02), (x + w * 0.09, top + h * 1.02),
                  (x + w * 0.05, top + h * 1.2), (x - w * 0.05, top + h * 1.2)], (230, 60, 44, 255))
    if lit and light > 0:
        p.addLight(x, cy, w * 5.5, (255, 110, 70), light)


def drawCrate(d, x0, y0, x1, y1, rgb=(176, 124, 74)):
    """Wooden fruit crate seen from the front: slats with dark gaps."""
    rect(d, x0, y0, x1, y1, shade(rgb, 0.55))
    slats = 3
    h = (y1 - y0) / slats
    for i in range(slats):
        sy0 = y0 + i * h + 1.5
        sy1 = y0 + (i + 1) * h - 1.5
        rect(d, x0, sy0, x1, sy1, rgb if i % 2 == 0 else shade(rgb, 0.08))
        rect(d, x0, sy0, x1, sy0 + 1.2, tint(rgb, 0.25))
    rect(d, x0, y0, x0 + 5, y1, shade(rgb, 0.2))
    rect(d, x1 - 5, y0, x1, y1, shade(rgb, 0.3))


def drawFruitPile(d, rnd, x0, x1, baseY, kind, scale=1.0):
    """A heap of fruit resting on baseY between x0 and x1."""
    width = x1 - x0
    rows = [(0.0, 1.0), (0.12, 0.88), (0.3, 0.7)]
    for level, (fa, fb) in enumerate(rows):
        y = baseY - level * 9 * scale
        count = max(1, int((fb - fa) * width / (14 * scale)))
        for i in range(count):
            x = x0 + width * (fa + (fb - fa) * (i + 0.5) / count) + rnd.uniform(-2, 2)
            drawFruit(d, rnd, x, y - rnd.uniform(0, 2), kind, scale)


def drawFruit(d, rnd, x, y, kind, s=1.0):
    """One small fruit (for crates, shelves and baskets). (x, y) = bottom centre."""
    if kind == "mango":
        c = jitter((250, 180, 50), rnd, 10)
        ellipse(d, x, y - 7 * s, 9 * s, 7 * s, c)
        ellipse(d, x + 2 * s, y - 5.5 * s, 7 * s, 5 * s, shade(c, 0.14))
        ellipse(d, x - 3 * s, y - 9 * s, 4 * s, 3 * s, (240, 110, 60, 255))
        ellipse(d, x - 4 * s, y - 10 * s, 2 * s, 1.5 * s, tint(c, 0.5))
    elif kind == "orange":
        c = jitter((250, 150, 40), rnd, 8)
        ellipse(d, x, y - 6 * s, 6.5 * s, 6.2 * s, c)
        ellipse(d, x + 1.5 * s, y - 5 * s, 5 * s, 4.6 * s, shade(c, 0.12))
        ellipse(d, x - 2 * s, y - 8.5 * s, 1.8 * s, 1.5 * s, tint(c, 0.55))
    elif kind == "jambu":
        c = jitter((226, 60, 86), rnd, 10)
        poly(d, [(x - 4 * s, y - 14 * s), (x + 4 * s, y - 14 * s), (x + 8 * s, y - 3 * s),
                 (x, y), (x - 8 * s, y - 3 * s)], c)
        ellipse(d, x, y - 4 * s, 8 * s, 4 * s, c)
        ellipse(d, x + 2 * s, y - 5 * s, 4 * s, 5 * s, shade(c, 0.15))
        ellipse(d, x - 3 * s, y - 10 * s, 1.6 * s, 3 * s, tint(c, 0.55))
    elif kind == "cucumber":
        c = jitter((80, 150, 64), rnd, 8)
        ang = rnd.uniform(-20, 20)
        pts = leafPoints(x - 13 * s, y - 5 * s, 26 * s, 11 * s, ang, 8)
        poly(d, pts, c)
        stroke(d, [(x - 8 * s, y - 7 * s), (x + 8 * s, y - 7 * s + ang * 0.1)], tint(c, 0.3), 1.2 * s)
    elif kind == "pineapple":
        leafC = (70, 140, 60, 255)
        for ang in (-60, -30, 0, 30, 60):
            a = math.radians(ang - 90)
            poly(d, [(x - 2.5 * s, y - 20 * s), (x + math.cos(a) * 11 * s, y - 20 * s + math.sin(a) * 13 * s),
                     (x + 2.5 * s, y - 20 * s)], leafC if ang % 60 else (50, 110, 50, 255))
        c = jitter((210, 150, 50), rnd, 8)
        ellipse(d, x, y - 10 * s, 7.5 * s, 10.5 * s, c)
        ellipse(d, x + 2 * s, y - 9 * s, 5 * s, 8.5 * s, shade(c, 0.16))
        for row in range(4):
            for col in range(-1, 2):
                ellipse(d, x + col * 3.6 * s + (row % 2) * 1.8 * s, y - (4 + row * 4.2) * s,
                        0.9 * s, 0.9 * s, (130, 86, 30, 255))
    elif kind == "watermelon":
        c = jitter((46, 120, 56), rnd, 6)
        ellipse(d, x, y - 12 * s, 14 * s, 12 * s, c)
        for k in (-8, -3, 3, 8):
            d.arc(box(x + k * s - 5 * s, y - 24 * s, x + k * s + 5 * s, y), 250, 290 if k < 0 else 300,
                  fill=(26, 80, 36, 255), width=int(2 * s * SS))
        ellipse(d, x - 5 * s, y - 17 * s, 4 * s, 2.5 * s, tint(c, 0.3))
    elif kind == "rambutan":
        c = jitter((210, 40, 50), rnd, 12)
        ellipse(d, x, y - 5 * s, 5.5 * s, 5 * s, c)
        for k in range(6):
            a = k * math.pi / 3
            stroke(d, [(x + math.cos(a) * 4 * s, y - 5 * s + math.sin(a) * 4 * s),
                       (x + math.cos(a) * 7 * s, y - 5 * s + math.sin(a) * 7 * s)], (90, 150, 60, 255), 0.8)
    elif kind == "chili":
        c = jitter((220, 36, 30), rnd, 8)
        ang = rnd.uniform(-30, 30)
        poly(d, leafPoints(x - 9 * s, y - 3 * s, 18 * s, 4.5 * s, ang, 6), c)
        ellipse(d, x - 9 * s, y - 3 * s, 1.6 * s, 1.6 * s, (60, 130, 50, 255))
    else:  # starfruit / lime fallback
        c = jitter((200, 220, 80), rnd, 10)
        ellipse(d, x, y - 5 * s, 6 * s, 5 * s, c)


def drawBananaBunch(d, x, y, s=1.0):
    """Hanging bunch of bananas; (x, y) = hook point."""
    stroke(d, [(x, y - 14 * s), (x, y + 4 * s)], (90, 70, 40, 255), 2 * s)
    for row in range(3):
        for k in range(-2, 3):
            ang = 90 + k * 22
            bx = x + k * 3 * s
            by = y + 4 * s + row * 9 * s
            poly(d, leafPoints(bx, by, 22 * s, 7 * s, ang, 8), (238, 200, 64, 255) if (k + row) % 2 else (222, 178, 50, 255))
    ellipse(d, x, y + 2 * s, 3 * s, 3 * s, (80, 110, 40, 255))


def drawStripedAwning(d, topY, topX0, topX1, botY, botX0, botX1, stripes, c1, c2, scallop=True):
    """Striped canopy (awning). Each stripe is a 4-point polygon + a round scallop below."""
    for i in range(stripes):
        t0 = i / stripes
        t1 = (i + 1) / stripes
        xa = topX0 + (topX1 - topX0) * t0
        xb = topX0 + (topX1 - topX0) * t1
        xc = botX0 + (botX1 - botX0) * t1
        xd = botX0 + (botX1 - botX0) * t0
        col = c1 if i % 2 == 0 else c2
        poly(d, [(xa, topY), (xb, topY), (xc, botY), (xd, botY)], col)
        # shade on the upper third (the fold of the canopy)
        yMid = topY + (botY - topY) * 0.3
        xm0 = xa + (xd - xa) * 0.3
        xm1 = xb + (xc - xb) * 0.3
        poly(d, [(xa, topY), (xb, topY), (xm1, yMid), (xm0, yMid)], shade(col, 0.16))
        if scallop:
            r = (xc - xd) / 2
            halfDisc(d, (xc + xd) / 2, botY, r, col, top=False)
            halfDisc(d, (xc + xd) / 2, botY, r * 0.62, shade(col, 0.08), top=False)


def drawLampPost(p, x, groundY, height):
    """Old-town cast-iron street lamp with a lantern head."""
    d = p.draw
    pole = (44, 66, 62, 255)
    poleLight = (84, 112, 104, 255)
    top = groundY - height
    rrect(d, x - 10, groundY - 30, x + 10, groundY, 3, pole)
    rect(d, x - 12, groundY - 6, x + 12, groundY, shade(pole, 0.2))
    rect(d, x - 3.5, top + 34, x + 3.5, groundY - 28, pole)
    rect(d, x - 3.5, top + 34, x - 1.5, groundY - 28, poleLight)
    for yy in (groundY - 34, top + 120, top + 40):
        rrect(d, x - 6.5, yy - 3, x + 6.5, yy + 3, 2, poleLight)
    # arm + lantern head
    stroke(d, [(x, top + 40), (x + 10, top + 30), (x + 22, top + 30)], pole, 2.5)
    poly(d, [(x - 14, top + 6), (x + 14, top + 6), (x + 8, top - 6), (x - 8, top - 6)], pole)
    ellipse(d, x, top - 8, 3, 3, pole)
    poly(p.glow, [(x - 11, top + 7), (x + 11, top + 7), (x + 8, top + 30), (x - 8, top + 30)], (255, 214, 150, 255))
    poly(p.glow, [(x - 5, top + 9), (x + 5, top + 9), (x + 3, top + 27), (x - 3, top + 27)], (255, 246, 220, 255))
    rect(p.glow, x - 0.8, top + 7, x + 0.8, top + 30, (60, 70, 66, 255))
    rect(d, x - 12, top + 29, x + 12, top + 34, pole)
    p.addLight(x, top + 18, 360, (255, 196, 128), 0.75)
    p.addLight(x, groundY + 16, 190, (255, 196, 128), 0.55, squash=0.22)


def drawStreetSign(p, x, groundY):
    """Green street-name sign 'PELITA STREET' on a grey pole."""
    d = p.draw
    rect(d, x - 3, groundY - 214, x + 3, groundY, (120, 126, 132, 255))
    rect(d, x - 3, groundY - 214, x - 1, groundY, (170, 176, 182, 255))
    rect(d, x - 7, groundY - 6, x + 7, groundY, (90, 94, 100, 255))
    rrect(d, x - 92, groundY - 250, x + 92, groundY - 208, 6, (236, 240, 236, 255))
    rrect(d, x - 88, groundY - 246, x + 88, groundY - 212, 4, (22, 112, 78, 255))
    rect(d, x - 86, groundY - 229, x + 86, groundY - 212, (18, 98, 68, 255))
    label(d, x, groundY - 229, "PELITA STREET", "future", 160, 17, (250, 250, 244, 255),
          shadow=(10, 60, 40, 255), shadowOffset=(0, 1.2))
    # small sign bracket bolts
    ellipse(d, x - 80, groundY - 229, 1.5, 1.5, (200, 210, 200, 255))
    ellipse(d, x + 80, groundY - 229, 1.5, 1.5, (200, 210, 200, 255))


def drawPlasticStool(d, x, groundY, rgb, h=46):
    """Classic kopitiam plastic stool (side view)."""
    top = groundY - h
    poly(d, [(x - 19, top + 6), (x + 19, top + 6), (x + 24, groundY), (x + 14, groundY),
             (x + 10, top + 16), (x - 10, top + 16), (x - 14, groundY), (x - 24, groundY)], rgb)
    poly(d, [(x - 19, top + 6), (x - 12, top + 6), (x - 18, groundY), (x - 24, groundY)], tint(rgb, 0.18))
    ellipse(d, x, top + 4, 21, 5, tint(rgb, 0.12))
    ellipse(d, x, top + 4, 7, 1.8, shade(rgb, 0.35))
    ellipse(d, x, top + 26, 5, 5, shade(rgb, 0.45))


def drawTableSet(p, x, groundY, stoolColours):
    """Round steel hawker table with plastic stools."""
    d = p.draw
    drawPlasticStool(d, x - 70, groundY, stoolColours[0])
    steel = (180, 186, 192, 255)
    rect(d, x - 3, groundY - 62, x + 3, groundY - 4, shade(steel, 0.3))
    ellipse(d, x, groundY - 2, 24, 4, shade(steel, 0.35))
    ellipse(d, x, groundY - 66, 58, 10, shade(steel, 0.2))
    ellipse(d, x, groundY - 68, 58, 9, steel)
    ellipse(d, x - 12, groundY - 70, 30, 4, tint(steel, 0.4))
    # a teh tarik glass and a plate on the table
    rect(d, x + 16, groundY - 86, x + 26, groundY - 68, (206, 150, 100, 255))
    rect(d, x + 16, groundY - 86, x + 26, groundY - 82, (246, 232, 206, 255))
    ellipse(d, x - 18, groundY - 70, 14, 3, (240, 240, 236, 255))
    drawPlasticStool(d, x + 72, groundY, stoolColours[1])
    if len(stoolColours) > 2:
        drawPlasticStool(d, x + 6, groundY + 10, stoolColours[2], h=40)


def drawPottedPlant(p, x, groundY, kind, rnd, s=1.0):
    d = p.draw
    if kind == "banana":
        for ang, ln in ((-150, 74), (-118, 90), (-90, 100), (-62, 88), (-30, 70), (-135, 60)):
            c = jitter((70, 140, 70), rnd, 12)
            pts = leafPoints(x, groundY - 44 * s, ln * s, 26 * s, ang, 12)
            poly(d, pts, c)
            a = math.radians(ang)
            stroke(d, [(x, groundY - 44 * s), (x + math.cos(a) * ln * s * 0.95, groundY - 44 * s + math.sin(a) * ln * s * 0.95)],
                   tint(c, 0.25), 1.2)
        rect(d, x - 3 * s, groundY - 50 * s, x + 3 * s, groundY - 34 * s, (90, 120, 60, 255))
    elif kind == "bougainvillea":
        for i in range(26):
            bx = x + rnd.uniform(-26, 26) * s
            by = groundY - 40 * s - rnd.uniform(0, 44) * s
            ellipse(d, bx, by, rnd.uniform(6, 10) * s, rnd.uniform(5, 8) * s, jitter((60, 120, 60), rnd, 12))
        for i in range(30):
            bx = x + rnd.uniform(-26, 26) * s
            by = groundY - 44 * s - rnd.uniform(0, 44) * s
            ellipse(d, bx, by, 3 * s, 2.6 * s, jitter((226, 70, 150), rnd, 16))
    else:  # palm
        for ang in (-160, -135, -110, -90, -70, -45, -20):
            c = jitter((64, 132, 72), rnd, 10)
            poly(d, leafPoints(x, groundY - 40 * s, 70 * s, 16 * s, ang, 10), c)
    # pot (terracotta, glazed for variety)
    potC = (184, 96, 66, 255) if kind != "banana" else (60, 90, 150, 255)
    poly(d, [(x - 22 * s, groundY - 40 * s), (x + 22 * s, groundY - 40 * s), (x + 16 * s, groundY), (x - 16 * s, groundY)], potC)
    rect(d, x - 24 * s, groundY - 44 * s, x + 24 * s, groundY - 36 * s, tint(potC, 0.15))
    poly(d, [(x + 6 * s, groundY - 36 * s), (x + 22 * s, groundY - 36 * s), (x + 16 * s, groundY), (x + 6 * s, groundY)], shade(potC, 0.18))
    if kind == "banana":
        for k in range(3):
            ellipse(d, x - 8 * s + k * 8 * s, groundY - 20 * s, 2.5 * s, 2.5 * s, (236, 240, 250, 255))


# ======================================================================
# 5. City silhouettes (skyline pieces)
# ======================================================================
def drawWindows(p, rnd, x0, y0, x1, y1, cell, win, chance, colours, alphaRange, floorBias=0.3, dark=None):
    """Grid of windows. Whole floors tend to be lit together, like a real building at night."""
    y = y0
    while y + win[1] < y1:
        floorOn = rnd.random() < floorBias
        x = x0
        while x + win[0] < x1:
            p_lit = chance * (2.6 if floorOn else 0.45)
            if rnd.random() < p_lit:
                c = rnd.choice(colours)
                rect(p.glow, x, y, x + win[0], y + win[1], rgba(c, rnd.randint(alphaRange[0], alphaRange[1])))
            elif dark is not None:
                rect(p.draw, x, y, x + win[0], y + win[1], dark)
            x += cell[0]
        y += cell[1]


def drawCityRow(p, rnd, x0, x1, baseY, topRange, widthRange, rgb, windowChance, windowColours,
                windowAlpha, cell=(7, 9), win=(3, 4), avoid=None, rimAmount=0.08, redLights=0.25):
    """A row of city buildings with rooftop details and lit windows."""
    d = p.draw
    x = x0
    while x < x1:
        w = rnd.randint(widthRange[0], widthRange[1])
        top = rnd.randint(topRange[0], topRange[1])
        if avoid:
            for (ax0, ax1, minTop) in avoid:
                if x + w > ax0 and x < ax1:
                    top = max(top, minTop + rnd.randint(0, 30))
        body = jitter(rgb, rnd, 5)
        rect(d, x, top, x + w, baseY, body)
        kind = rnd.random()
        tipY = top
        if kind < 0.2:                                   # stepped top block
            inset = w * rnd.uniform(0.15, 0.3)
            h2 = rnd.randint(8, 26)
            rect(d, x + inset, top - h2, x + w - inset, top, body)
            tipY = top - h2
        elif kind < 0.32:                                # slanted roof
            poly(d, [(x, top), (x + w, top - rnd.randint(8, 22)), (x + w, top)], body)
        elif kind < 0.42:                                # pointed crown
            h2 = rnd.randint(14, 40)
            poly(d, [(x + w * 0.18, top), (x + w * 0.5, top - h2), (x + w * 0.82, top)], body)
            tipY = top - h2
        elif kind < 0.52:                                # rooftop water tank
            tx = x + w * rnd.uniform(0.15, 0.55)
            rect(d, tx, top - 11, tx + 15, top - 4, body)
            rect(d, tx + 2, top - 4, tx + 4, top, body)
            rect(d, tx + 11, top - 4, tx + 13, top, body)
        elif kind < 0.6:                                 # rounded top
            d.pieslice(box(x, top - w * 0.25, x + w, top + w * 0.25), 180, 360, fill=body)
            tipY = top - w * 0.25
        if rnd.random() < 0.35:                          # antenna with a red aviation light
            axx = x + w * rnd.uniform(0.3, 0.7)
            ah = rnd.randint(14, 36)
            rect(d, axx - 0.8, tipY - ah, axx + 0.8, tipY, body)
            if rnd.random() < redLights / 0.35:
                ellipse(p.glow, axx, tipY - ah, 1.6, 1.6, (255, 70, 60, 255))
        # lit left edge + darker right edge give each block some volume
        rect(d, x, top, x + max(1.5, w * 0.07), baseY, tint(body, rimAmount))
        rect(d, x + w - max(1.0, w * 0.05), top, x + w, baseY, shade(body, 0.12))
        drawWindows(p, rnd, x + 4, top + 6, x + w - 3, baseY, cell, win, windowChance,
                    windowColours, windowAlpha, dark=shade(body, 0.2))
        x += w + rnd.randint(-10, 2)


def drawPetronasTower(p, cx, baseY, height, width, body, lit, lightRgb, lightAlpha):
    """One Petronas-style tower: tall shaft, stepped tapering tiers and a thin spire."""
    d, g = p.draw, p.glow
    tiers = [(0.00, 0.60, 1.00), (0.60, 0.675, 0.88), (0.675, 0.74, 0.76), (0.74, 0.795, 0.64),
             (0.795, 0.84, 0.52), (0.84, 0.875, 0.40), (0.875, 0.905, 0.28)]
    for (f0, f1, wf) in tiers:
        y0 = baseY - height * f1
        y1 = baseY - height * f0
        half = width * wf / 2
        topHalf = half * 0.95
        poly(d, [(cx - topHalf, y0), (cx + topHalf, y0), (cx + half, y1), (cx - half, y1)], body)
        # flood-lit glass: a faint glow over the whole tier
        poly(g, [(cx - topHalf, y0), (cx + topHalf * 0.4, y0), (cx + half * 0.4, y1), (cx - half, y1)],
             rgba(lightRgb, int(lightAlpha * 0.1)))
        # facets: lit left face, dark right face
        poly(d, [(cx - topHalf, y0), (cx - topHalf * 0.35, y0), (cx - half * 0.35, y1), (cx - half, y1)], lit)
        poly(d, [(cx + topHalf * 0.55, y0), (cx + topHalf, y0), (cx + half, y1), (cx + half * 0.55, y1)], shade(body, 0.25))
        # vertical light lines (the famous steel fins glowing at night)
        for k in (-0.62, -0.3, 0.0, 0.3, 0.62):
            xx0 = cx + half * k
            xx1 = cx + topHalf * k
            poly(g, [(xx1 - 0.35, y0 + 1), (xx1 + 0.35, y0 + 1), (xx0 + 0.35, y1), (xx0 - 0.35, y1)],
                 rgba(lightRgb, int(lightAlpha * (0.45 + 0.55 * f1))))
        # bright ring at each set-back
        rect(g, cx - topHalf, y0, cx + topHalf, y0 + 1.2, rgba(lightRgb, min(255, int(lightAlpha * 1.3))))
    # pinnacle spire
    sy0 = baseY - height * 0.905
    sy1 = baseY - height
    poly(d, [(cx - width * 0.06, sy0), (cx + width * 0.06, sy0), (cx + width * 0.012, sy1), (cx - width * 0.012, sy1)], body)
    for f in (0.925, 0.95):
        yy = baseY - height * f
        ellipse(d, cx, yy, width * 0.07, width * 0.035, lit)
        ellipse(g, cx, yy, width * 0.05, width * 0.02, rgba(lightRgb, lightAlpha))
    stroke(g, [(cx, sy0), (cx, sy1 + 4)], rgba(lightRgb, int(lightAlpha * 0.8)), 0.8)


def drawTwinTowers(p, cx, baseY, height, width, body, lit, lightRgb, lightAlpha):
    gap = width * 1.25
    left = cx - gap / 2 - width / 2
    right = cx + gap / 2 + width / 2
    drawPetronasTower(p, left, baseY, height, width, body, lit, lightRgb, lightAlpha)
    drawPetronasTower(p, right, baseY, height * 0.995, width, body, lit, lightRgb, lightAlpha)
    # sky bridge (41st floor) with its V-shaped legs
    by = baseY - height * 0.37
    x0 = left + width / 2
    x1 = right - width / 2
    rect(p.draw, x0, by - 2.5, x1, by + 2.5, body)
    stroke(p.draw, [(cx, by + 2), (x0, by + height * 0.08)], body, 1.6)
    stroke(p.draw, [(cx, by + 2), (x1, by + height * 0.08)], body, 1.6)
    rect(p.glow, x0, by - 0.6, x1, by + 0.6, rgba(lightRgb, lightAlpha))


def drawKLTower(p, cx, baseY, podY, tipY, body, lit, lightRgb, lightAlpha, s=1.0):
    """Menara KL: tapering shaft, turnip-shaped pod and a tall antenna with a red light."""
    d, g = p.draw, p.glow
    poly(d, [(cx - 13 * s, baseY), (cx + 13 * s, baseY), (cx + 7 * s, podY + 34 * s), (cx - 7 * s, podY + 34 * s)], body)
    poly(d, [(cx - 13 * s, baseY), (cx - 6 * s, baseY), (cx - 3 * s, podY + 34 * s), (cx - 7 * s, podY + 34 * s)], lit)
    poly(d, [(cx - 7 * s, podY + 34 * s), (cx + 7 * s, podY + 34 * s), (cx + 26 * s, podY + 8 * s), (cx - 26 * s, podY + 8 * s)], body)
    rrect(d, cx - 31 * s, podY - 6 * s, cx + 31 * s, podY + 9 * s, 4 * s, body)
    poly(d, [(cx - 28 * s, podY - 6 * s), (cx + 28 * s, podY - 6 * s), (cx + 15 * s, podY - 22 * s), (cx - 15 * s, podY - 22 * s)], body)
    poly(d, [(cx - 28 * s, podY - 6 * s), (cx - 12 * s, podY - 6 * s), (cx - 6 * s, podY - 22 * s), (cx - 15 * s, podY - 22 * s)], lit)
    rect(d, cx - 6 * s, podY - 36 * s, cx + 6 * s, podY - 22 * s, body)
    poly(d, [(cx - 3 * s, podY - 36 * s), (cx + 3 * s, podY - 36 * s), (cx + 0.8 * s, tipY), (cx - 0.8 * s, tipY)], body)
    # lights: window band round the observation deck, rings, red tip
    rect(g, cx - 29 * s, podY - 1 * s, cx + 29 * s, podY + 3 * s, rgba(lightRgb, lightAlpha))
    rect(g, cx - 24 * s, podY - 12 * s, cx + 24 * s, podY - 10 * s, rgba(lightRgb, int(lightAlpha * 0.7)))
    for f in (0.3, 0.55):
        yy = podY - 36 * s + (tipY - (podY - 36 * s)) * f
        rect(g, cx - 2.5 * s, yy, cx + 2.5 * s, yy + 1.5 * s, rgba(lightRgb, int(lightAlpha * 0.7)))
    stroke(g, [(cx, baseY), (cx, podY + 34 * s)], rgba(lightRgb, int(lightAlpha * 0.35)), 0.8 * s)
    ellipse(g, cx, tipY, 2.2 * s, 2.2 * s, (255, 70, 60, 255))
    ellipse(g, cx, podY - 36 * s, 1.6 * s, 1.6 * s, (255, 70, 60, 255))


def drawMarketCanopies(p, rnd, width, groundY, peakRange=(26, 50)):
    """Night-market tents: dark striped roofs, a row of bulbs under the eaves and warm
    light glowing underneath, with the silhouettes of counters and people against it."""
    d, g = p.draw, p.glow
    roofColours = [((120, 36, 40), (196, 170, 150)), ((34, 90, 64), (190, 186, 160)),
                   ((40, 58, 116), (180, 180, 200)), ((150, 84, 36), (200, 170, 130))]
    x = -30
    while x < width + 30:
        w = rnd.randint(110, 170)
        peak = rnd.randint(peakRange[0], peakRange[1])
        eave = peak + rnd.randint(24, 34)
        val = eave + 9
        c1, c2 = rnd.choice(roofColours)
        # the roofs are in shadow: mix the stripe colours towards a dark night purple
        dark1 = art_kit.mix(c1, (26, 20, 40), 0.62)
        dark2 = art_kit.mix(c2, (26, 20, 40), 0.66)
        # warm interior light (glow layer): brighter under the bulbs, fading towards the ground
        steps = 8
        for k in range(steps):
            t0 = val + (groundY - val) * k / steps
            t1 = val + (groundY - val) * (k + 1) / steps + 0.5
            alpha = int(150 - 80 * k / steps)
            rect(g, x + 6, t0, x + w - 6, t1, (255, 158 + 4 * k, 80, alpha))
        # counters and people block the light (dark shapes painted on the glow layer)
        rect(g, x + 10, groundY - 26, x + w - 10, groundY, (26, 18, 30, 255))
        rect(g, x + 10, groundY - 26, x + w - 10, groundY - 24, (120, 70, 50, 255))
        for k in range(rnd.randint(1, 3)):
            px = x + rnd.uniform(20, w - 20)
            ph = rnd.uniform(26, 38)
            ellipse(g, px, groundY - ph, 5.5, 6, (22, 16, 28, 255))
            rrect(g, px - 8, groundY - ph + 5, px + 8, groundY, 4, (22, 16, 28, 255))
        # roof: triangular tent or round umbrella, with vertical stripes
        roof = Image.new("L", (p.img.width, p.img.height), 0)
        rd = ImageDraw.Draw(roof)
        if rnd.random() < 0.72:
            rd.polygon(art_kit.pts([(x - 4, eave), (x + w / 2, peak), (x + w + 4, eave)]), fill=255)
        else:
            rd.pieslice(box(x - 4, peak, x + w + 4, eave * 2 - peak), 180, 360, fill=255)
        rd.rectangle(box(x - 4, eave, x + w + 4, val), fill=255)
        stripes = Image.new("RGBA", roof.size, rgba(dark1))
        sd = ImageDraw.Draw(stripes)
        stripeW = w / 7
        for k in range(8):
            if k % 2 == 1:
                sx = x - 4 + k * stripeW
                sd.polygon(art_kit.pts([(x + w / 2, peak), (sx, val), (sx + stripeW, val)]), fill=rgba(dark2))
        p.img.paste(stripes, (0, 0), roof)
        # scalloped valance and a string of bulbs under the eaves
        for k in range(int(w / 16) + 1):
            halfDisc(d, x + 4 + k * 16, val, 7, shade(dark1, 0.15), top=False)
        for k in range(int(w / 22)):
            bx = x + 11 + k * 22
            ellipse(g, bx, val + 9, 1.8, 1.8, (255, 236, 190, 255))
        rect(g, x - 4, val + 6, x + w + 4, val + 7, (255, 190, 110, 90))
        # support poles (glow layer so they block the light)
        rect(g, x + 2, val, x + 5, groundY, (26, 18, 30, 255))
        rect(g, x + w - 5, val, x + w - 2, groundY, (26, 18, 30, 255))
        x += w - rnd.randint(0, 14)


def drawLanternString(p, rnd, x0, y0, x1, y1, drop, spacing, colours, size=1.0, wire=(40, 30, 40, 255), light=0.0):
    """A string of small lanterns/bulbs hanging on a sagging wire."""
    points = sag(x0, y0, x1, y1, drop, 60)
    stroke(p.draw, points, wire, 1.1 * size)
    length = abs(x1 - x0)
    count = int(length / spacing)
    for i in range(1, count):
        t = i / count
        x = x0 + (x1 - x0) * t
        y = y0 + (y1 - y0) * t + drop * 4 * t * (1 - t)
        c = rnd.choice(colours)
        if c == "lantern":
            drawRedLantern(p, x, y + 3 * size, 9 * size, 11 * size, lit=True, light=light)
        else:
            stroke(p.draw, [(x, y), (x, y + 4 * size)], wire, 0.8)
            ellipse(p.glow, x, y + 6 * size, 2.4 * size, 2.8 * size, rgba(c))
            ellipse(p.glow, x - 0.5 * size, y + 5.5 * size, 1.0 * size, 1.0 * size, (255, 250, 236, 255))
            if light > 0:
                p.addLight(x, y + 6 * size, 90 * size, c, light * 0.5)


# ======================================================================
# 6. Sky helpers
# ======================================================================
def stretchedNoise(width, height, cellX, cellY, seed):
    """Smooth random pattern 0..1 whose blobs are cellX wide and cellY tall.
    A small grid of random numbers is enlarged with BICUBIC interpolation."""
    rnd = np.random.default_rng(seed)
    cellX, cellY = max(2, int(cellX)), max(2, int(cellY))
    gw = width // cellX + 4
    gh = height // cellY + 4
    small = Image.fromarray(rnd.random((gh, gw)).astype(np.float32), "F")
    big = small.resize((gw * cellX, gh * cellY), Image.BICUBIC)
    arr = np.array(big, np.float32)[cellY:cellY + height, cellX:cellX + width]
    return np.clip(arr, 0.0, 1.0)


def fbmNoise(width, height, cellX, cellY, octaves, seed):
    """Fractal noise (fBm): several noise layers, each twice as fine and half as strong.
    Big shapes come from the first layer, wispy detail from the last ones. Result ~0..1."""
    total = np.zeros((height, width), np.float32)
    amp, norm = 1.0, 0.0
    for i in range(octaves):
        total += amp * stretchedNoise(width, height, cellX, cellY, seed + i * 31)
        norm += amp
        amp *= 0.5
        cellX, cellY = cellX / 2, cellY / 2
    total /= norm
    # stretch the contrast back out (adding layers pulls the values towards 0.5)
    return np.clip((total - 0.5) * 2.2 + 0.5, 0.0, 1.0)


def moonImage(radius, seed, rgb=None):
    """Full moon: smooth disc, limb darkening and soft grey-blue 'maria' patches."""
    rgb = rgb or P["moon"]
    size = radius * 2 + 6
    big = Image.new("L", (size * SS, size * SS), 0)
    ImageDraw.Draw(big).ellipse([3 * SS, 3 * SS, (size - 3) * SS, (size - 3) * SS], fill=255)
    alpha = np.array(big.resize((size, size), Image.LANCZOS), np.float32)
    c = (size - 1) / 2.0
    r = np.clip(distance(size, size, c, c) / radius, 0, 1)
    bright = 1.0 - 0.16 * r ** 3                                    # limb darkening
    # maria: big soft patches from fractal noise, a little more on the upper-left half
    n = fbmNoise(size, size, max(6, radius * 0.7), max(6, radius * 0.6), 3, seed)
    xs, ys = grid(size, size)
    bias = np.clip(1.0 - ((xs - c * 0.8) ** 2 + (ys - c * 0.85) ** 2) / (radius * 1.1) ** 2, 0, 1)
    maria = np.clip((n * 0.8 + bias * 0.45 - 0.62) / 0.2, 0, 1)
    mariaImg = Image.fromarray((maria * 255).astype(np.uint8), "L")
    maria = np.array(mariaImg.filter(ImageFilter.GaussianBlur(max(1, radius / 18))), np.float32) / 255.0
    bright *= 1.0 - 0.2 * maria
    arr = np.zeros((size, size, 4), np.float32)
    arr[:, :, :3] = np.array(rgb, np.float32) * bright[:, :, None]
    # maria are slightly blue-grey, the lower edge a little warmer (lantern-festival moon)
    arr[:, :, :3] -= (maria * 10)[:, :, None] * np.array([1.0, 0.6, 0.0], np.float32)
    warm = np.clip((ys - c) / radius, 0, 1)[:, :, None]
    arr[:, :, :3] *= 1.0 - warm * np.array([0.0, 0.04, 0.1], np.float32)
    arr[:, :, 3] = alpha
    return toImage(arr)


def moonHalo(arr, mx, my, mr, rgb, amount):
    """Glow around the moon: a tight bright corona plus wide soft scattering in the air."""
    h, w = arr.shape[:2]
    dm = distance(w, h, mx, my)
    outside = np.clip(dm - mr, 0, None)
    halo = (0.6 * np.exp(-outside / (mr * 0.35)) + 0.3 * np.exp(-outside / (mr * 1.6))
            + 0.14 * np.exp(-dm / (mr * 6.0)))
    arr[:, :, :3] += halo[:, :, None] * np.array(rgb, np.float32) * amount


def cloudLayer(width, height, bands, seed, cellX=300, cellY=46):
    """Cloud density 0..1. bands = [(centreY, thickness, x0, x1, strength), ...].
    Fractal noise decides the wispy shape, the bands decide where clouds may appear."""
    xs, ys = grid(width, height)
    profile = np.zeros((height, width), np.float32)
    for (cy, thick, x0, x1, strength) in bands:
        band = np.exp(-((ys - cy) / thick) ** 2)
        edge = np.clip((xs - x0) / 160.0, 0, 1) * np.clip((x1 - xs) / 160.0, 0, 1)
        profile = np.maximum(profile, band * edge * strength)
    n = fbmNoise(width, height, cellX, cellY, 5, seed)
    t = np.clip((n * 0.62 + profile * 0.62 - 0.6) / 0.3, 0, 1)
    return t * t * (3 - 2 * t)          # smoothstep = soft edges


def shifted(values, dy, dx):
    """Copy of a 2D array moved by (dy, dx) pixels; the edges are repeated (no wrap-around)."""
    h, w = values.shape
    padded = np.pad(values, ((abs(dy), abs(dy)), (abs(dx), abs(dx))), mode="edge")
    top = abs(dy) - dy
    left = abs(dx) - dx
    return padded[top:top + h, left:left + w]


def shadeClouds(arr, density, bodyRgb, opacity, litRgb, moonX, moonY, warmRgb=None, warmFrom=0.5):
    """Paint clouds into a sky array. The edge facing the moon gets a silver lining
    (density minus the density of the neighbour towards the moon), and the underside
    of low clouds is lit warm by the city below (light pollution)."""
    h, w = arr.shape[:2]
    near = np.exp(-distance(w, h, moonX, moonY) / 240.0)[:, :, None]
    # body colour: darker than the sky far from the moon, silvery close to it
    body = np.array(bodyRgb, np.float32) * (1 - near * 0.6) + np.array(litRgb, np.float32) * (near * 0.42)
    a = (density * opacity)[:, :, None]
    arr[:, :, :3] = arr[:, :, :3] * (1 - a) + body * a
    towardMoon = shifted(density, 7, -4)       # value of the neighbour up and to the right
    rim = np.clip(density - towardMoon, 0, 1)[:, :, None]
    arr[:, :, :3] += rim * (0.22 + 1.3 * near) * np.array(litRgb, np.float32) * 0.55
    # moonlight scattered inside thin cloud right next to the moon
    arr[:, :, :3] += density[:, :, None] * near * 0.16 * np.array(litRgb, np.float32)
    if warmRgb is not None:
        below = shifted(density, -6, 0)
        under = np.clip(density - below, 0, 1)
        xs, ys = grid(w, h)
        warmth = np.clip((ys / h - warmFrom) / (1 - warmFrom), 0, 1)
        arr[:, :, :3] += ((under * 0.9 + density * 0.18) * warmth)[:, :, None] * np.array(warmRgb, np.float32)


def starField(arr, rnd, count, yMax, sizeRange=(0.5, 1.4)):
    """Tiny baked stars (only for skies that the game does not animate)."""
    h, w = arr.shape[:2]
    img = Image.new("L", (w * 2, h * 2), 0)
    sd = ImageDraw.Draw(img)
    for i in range(count):
        x = rnd.uniform(0, w) * 2
        y = rnd.uniform(0, yMax) ** 1.2 / yMax ** 0.2 * 2
        r = rnd.uniform(sizeRange[0], sizeRange[1])
        sd.ellipse([x - r, y - r, x + r, y + r], fill=rnd.randint(90, 255))
    stars = np.array(img.resize((w, h), Image.LANCZOS), np.float32) / 255.0
    arr[:, :, :3] += stars[:, :, None] * np.array([235, 236, 255], np.float32)


# ======================================================================
# 7. TITLE SKY  (1280x720 opaque, also the street sky)
# ======================================================================
def makeTitleSky():
    width, height = 1280, 720
    # deep indigo at the top -> night blue -> purple dusk glow at the horizon
    arr = gradientArray(width, height, [
        (0.00, (6, 7, 24)), (0.22, (12, 14, 42)), (0.46, (24, 28, 74)), (0.66, (42, 44, 106)),
        (0.80, (74, 56, 124)), (0.92, (116, 70, 128)), (1.00, (150, 84, 124))])
    # very soft large-scale variation so the gradient does not look "computer perfect"
    noise = fbmNoise(width, height, 420, 260, 2, 11)
    arr[:, :, :3] *= (0.94 + 0.12 * noise)[:, :, None]
    # warm light pollution glowing up from the city
    glow = np.clip(1 - distance(width, height, 640, 860, 1150, 420), 0, 1) ** 2
    arr[:, :, :3] += glow[:, :, None] * np.array([84, 36, 40], np.float32)
    # moon halo, then the moon itself
    moonHalo(arr, MOON_X, MOON_Y, MOON_R, (232, 218, 204), 0.5)
    img = toImage(arr)
    moon = moonImage(MOON_R, seed=4)
    pasteOver(img, moon, MOON_X - moon.width // 2, MOON_Y - moon.height // 2)
    arr = toArray(img)
    # clouds: thin high wisps, one streak crossing the moon, a big low bank lit by the city.
    # The middle (behind the logo) is kept calm.
    bands = [(70, 26, -200, 700, 0.75), (218, 13, 760, 1500, 0.9), (300, 30, -200, 1500, 0.55),
             (470, 54, -200, 1500, 1.0), (560, 60, -200, 1500, 0.8)]
    density = cloudLayer(width, height, bands, seed=6)
    shadeClouds(arr, density, (46, 44, 96), 0.62, (226, 220, 240), MOON_X, MOON_Y,
                warmRgb=(150, 70, 80), warmFrom=0.45)
    applyVignette(arr, 0.28)
    addGrain(arr, 2.0, 12)
    return toImage(arr)


# ======================================================================
# 8. TITLE SKYLINE (1280x720 transparent)
# ======================================================================
def makeTitleSkyline():
    width, height = 1280, 720
    pic = Picture(width, height)
    rnd = random.Random(2024)
    dimWarm = [(220, 170, 130), (200, 160, 140), (170, 170, 200)]
    # far hazy city
    far = Piece(width, 290)
    drawCityRow(far, rnd, -20, width + 20, 290, (50, 110), (30, 80), (52, 48, 104), 0.05, dimWarm,
                (50, 110), cell=(6, 8), win=(2.5, 3.5), rimAmount=0.05)
    pic.place(far, 0, 430, haze=((104, 76, 138), 0.1, 0.6))
    # landmarks: twin towers (left) and KL Tower (right) frame the logo
    marks = Piece(width, height)
    drawTwinTowers(marks, 196, 720, 486, 44, (40, 40, 90, 255), (60, 60, 116, 255), (206, 214, 255), 150)
    drawKLTower(marks, 1150, 720, 384, 240, (38, 36, 84, 255), (58, 56, 112, 255), (255, 214, 160), 190)
    pic.place(marks, 0, 0, haze=((100, 74, 136), 0.0, 0.55))
    pic.overlay(verticalFade(width, 300, (110, 78, 140), 0, 120), 0, 420)
    # nearer city: darker, more lit windows
    mid = Piece(width, 250)
    drawCityRow(mid, rnd, -30, width + 30, 250, (40, 140), (36, 104), (28, 26, 62), 0.18,
                WARM_WINDOWS + COOL_WINDOWS[:1], (120, 230), cell=(7, 9), win=(3, 4),
                avoid=[(120, 290, 150)])
    pic.place(mid, 0, 470, haze=((70, 44, 92), 0.0, 0.5))
    pic.overlay(verticalFade(width, 140, (255, 150, 90), 0, 70), 0, 580)
    # night-market canopies + lantern strings along the very bottom
    fg = Piece(width, 130)
    drawMarketCanopies(fg, rnd, width, 130)
    for (y0, y1, drop) in ((22, 34, 20), (6, 18, 26)):
        drawLanternString(fg, rnd, -20, y0, width + 20, y1, drop, 34,
                          ["lantern", (255, 200, 120), (255, 160, 90), "lantern", (255, 226, 160)], size=1.0)
    pic.place(fg, 0, 590)
    return pic.render((1.0, 1.0, 1.0), grain=1.6, seed=21)


# ======================================================================
# 9. STREET FAR (1920x720 transparent, parallax 0.25)
# ======================================================================
def makeStreetFar():
    width, height = 1920, 720
    pic = Picture(width, height)
    rnd = random.Random(77)
    dim = [(200, 170, 150), (170, 180, 220), (210, 180, 140)]
    back = Piece(width, 480)
    drawCityRow(back, rnd, -20, width + 20, 480, (40, 170), (34, 90), (50, 54, 108), 0.05, dim,
                (40, 100), cell=(6, 8), win=(2.5, 3.5), rimAmount=0.05)
    pic.place(back, 0, 240, haze=((78, 70, 128), 0.25, 0.7))
    marks = Piece(width, height)
    drawTwinTowers(marks, 780, 720, 640, 46, (44, 48, 100, 255), (60, 64, 120, 255), (190, 204, 255), 120)
    drawKLTower(marks, 1470, 720, 262, 118, (42, 46, 98, 255), (58, 62, 116, 255), (255, 214, 170), 150, s=1.1)
    pic.place(marks, 0, 0, haze=((76, 70, 128), 0.12, 0.6))
    front = Piece(width, 560)
    drawCityRow(front, rnd, -30, width + 30, 560, (0, 220), (46, 120), (38, 42, 90), 0.08,
                dim + WARM_WINDOWS[:1], (50, 130), cell=(7, 9), win=(3, 4),
                avoid=[(690, 870, 170), (1420, 1520, 190)], rimAmount=0.06)
    pic.place(front, 0, 160, haze=((70, 62, 118), 0.15, 0.62))
    pic.overlay(verticalFade(width, 420, (84, 70, 130), 0, 150), 0, 300)
    return pic.render((1.0, 1.0, 1.0), grain=1.4, seed=31, glowAmount=0.8)


# ======================================================================
# 10. STREET MID (2560x720 transparent, parallax 0.6) - shophouse row
# ======================================================================
SHOP_NAMES = ["AH HOCK GROCERY", "SIN HUAT HARDWARE", "SALLEH BARBER SHOP",
              "TASTE OF HOME CAFE", "PELITA PHARMACY", "LIAN HENG GOLDSMITH", "JASMINE TAILOR",
              "HAPPY COFFEE SHOP", "AHMAD BICYCLE REPAIR", "CITY BOOKSHOP",
              "PEARL TEXTILES", "ROSE FLORIST", "CHINESE MEDICINE HALL"]
FACADES = [(150, 200, 180), (230, 208, 150), (226, 170, 168), (150, 178, 212), (226, 214, 190),
           (172, 192, 150), (232, 180, 140), (186, 172, 212)]
SIGN_STYLES = [((196, 40, 40), (255, 222, 96)), ((30, 64, 140), (255, 255, 255)),
               ((248, 208, 64), (176, 30, 30)), ((20, 110, 70), (255, 240, 200)),
               ((236, 236, 226), (30, 60, 150)), ((120, 30, 90), (255, 212, 124))]


def drawShophouseFloor(p, w, fy0, fy1, rnd, facade):
    d, g = p.draw, p.glow
    rect(d, 0, fy0, w, fy1, rgba(facade))
    rect(d, 0, fy0, 11, fy1, tint(facade, 0.12))
    rect(d, w - 11, fy0, w, fy1, shade(facade, 0.14))
    count = 3 if w > 228 else 2
    margin = 24
    spacing = (w - 2 * margin) / count
    ww = min(32, spacing * 0.46)
    wh = 62
    shutter = rnd.choice([(58, 120, 96), (66, 98, 150), (122, 82, 50), (150, 60, 50), (70, 120, 120)])
    arched = rnd.random() < 0.5
    for k in range(count):
        cx = margin + spacing * (k + 0.5)
        x0, x1 = cx - ww / 2, cx + ww / 2
        y0, y1 = fy0 + 26, fy0 + 26 + wh
        rect(d, x0 - 4, y0 - 3, x1 + 4, y1 + 3, tint(facade, 0.28))
        if arched:
            halfDisc(d, cx, y0, ww / 2 + 4, tint(facade, 0.28), top=True)
        lit = rnd.random() < 0.42
        if lit:
            c = rnd.choice(WARM_WINDOWS)
            target = g
            rect(g, x0, y0, x1, y1, rgba(c, 235))
            if arched:
                halfDisc(g, cx, y0, ww / 2, rgba(c, 235), top=True)
            # curtain half drawn + window bars (dark shapes on the glow layer)
            if rnd.random() < 0.6:
                rect(g, x0, y0 - (ww / 2 if arched else 0), x0 + ww * 0.3, y1, rgba(shade(c, 0.35), 255))
            p.addLight(cx, (y0 + y1) / 2, 70, c, 0.5)
        else:
            target = d
            glass = (40, 46, 64, 255)
            rect(d, x0, y0, x1, y1, glass)
            if arched:
                halfDisc(d, cx, y0, ww / 2, glass, top=True)
        bar = (70, 56, 50, 255)
        rect(target, cx - 0.7, y0 - (ww / 2 if arched else 0), cx + 0.7, y1, bar)
        rect(target, x0, y0 + wh * 0.42, x1, y0 + wh * 0.42 + 1.4, bar)
        # open louvred shutters on both sides
        sw = ww * 0.42
        for sx0 in (x0 - sw - 3, x1 + 3):
            rect(d, sx0, y0, sx0 + sw, y1, rgba(shutter))
            yy = y0 + 3
            while yy < y1 - 2:
                rect(d, sx0 + 1.5, yy, sx0 + sw - 1.5, yy + 1.2, shade(shutter, 0.3))
                yy += 4.5
        rect(d, x0 - 7, y1 + 3, x1 + 7, y1 + 7, tint(facade, 0.35))
        # extras under / on the window sill
        extra = rnd.random()
        if extra < 0.3:
            ax = cx - 13
            rect(d, ax, y1 + 11, ax + 26, y1 + 27, (186, 188, 184, 255))
            ellipse(d, ax + 17, y1 + 19, 6, 6, (120, 124, 124, 255))
            ellipse(d, ax + 17, y1 + 19, 2, 2, (80, 84, 84, 255))
            rect(d, ax + 4, y1 + 27, ax + 5, y1 + 40, (80, 84, 90, 255))
        elif extra < 0.6:
            for kk in range(3):
                px = x0 - 2 + kk * (ww + 4) / 2.4
                rect(d, px, y1 - 4, px + 8, y1 + 3, (180, 96, 64, 255))
                ellipse(d, px + 4, y1 - 7, 7, 5, jitter((70, 140, 70), rnd, 12))
                stroke(d, [(px + 2, y1), (px - 1, y1 + 16 + kk * 3)], (60, 124, 64, 255), 1.4)
    rect(d, 3, fy1 - 6, w - 3, fy1, tint(facade, 0.18))
    rect(d, 3, fy1, w - 3, fy1 + 2, shade(facade, 0.3))


def drawShophouse(p, w, storeys, facade, signStyle, name, rnd, state):
    """One 2- or 3-storey Malaysian shophouse (base at local y 440)."""
    d, g = p.draw, p.glow
    B = 440
    groundTop = B - 150
    floorH = 118
    top = groundTop - floorH * (storeys - 1)
    # parapet with a small gable
    parapetTop = top - 22
    rect(d, 0, parapetTop, w, top, tint(facade, 0.06))
    style = rnd.random()
    if style < 0.4:
        poly(d, [(w * 0.28, parapetTop + 1), (w * 0.5, parapetTop - 22), (w * 0.72, parapetTop + 1)], tint(facade, 0.06))
        ellipse(d, w * 0.5, parapetTop - 7, 5, 5, tint(facade, 0.35))
    elif style < 0.75:
        d.pieslice(box(w * 0.3, parapetTop - 18, w * 0.7, parapetTop + 18), 180, 360, fill=tint(facade, 0.06))
        halfDisc(d, w * 0.5, parapetTop, 7, tint(facade, 0.3), top=True)
    for k in range(int(w / 18)):
        rect(d, 8 + k * 18, parapetTop + 6, 14 + k * 18, top - 6, shade(facade, 0.18))
    rect(d, -1, parapetTop - 2, w + 1, parapetTop + 2, tint(facade, 0.25))
    rect(d, -2, top - 4, w + 2, top + 1, tint(facade, 0.3))
    for f in range(storeys - 1):
        fy0 = top + f * floorH
        drawShophouseFloor(p, w, fy0, fy0 + floorH, rnd, facade)
    # ground floor: signboard + five-foot-way arch
    rect(d, 0, groundTop, w, B, shade(facade, 0.1))
    board, textC = signStyle
    sy0, sy1 = groundTop + 4, groundTop + 32
    rect(d, 8, sy0, w - 8, sy1, shade(board, 0.35))
    rect(d, 10, sy0 + 2, w - 10, sy1 - 2, rgba(board))
    backlit = rnd.random() < 0.45
    label(g if backlit else d, w / 2, (sy0 + sy1) / 2 + 0.5, name, "future", w - 34, 12,
          rgba(textC), shadow=None if backlit else shade(board, 0.5))
    if backlit:
        p.addLight(w / 2, (sy0 + sy1) / 2, 90, textC, 0.25)
    archTop = sy1 + 8
    spring = archTop + 22
    x0, x1 = 16, w - 16
    recess = (34, 30, 40, 255)
    rect(d, x0, spring, x1, B, recess)
    d.pieslice(box(x0, archTop, x1, spring * 2 - archTop), 180, 360, fill=recess)
    rect(d, 0, sy1, x0, B, tint(facade, 0.05))
    rect(d, x1, sy1, w, B, shade(facade, 0.12))
    rect(d, -1, sy1 + 2, x0 + 3, sy1 + 8, tint(facade, 0.3))
    rect(d, x1 - 3, sy1 + 2, w + 1, sy1 + 8, tint(facade, 0.3))
    # shop front inside the five-foot way
    shutterC = (126, 130, 138, 255)
    inner0, inner1 = x0 + 14, x1 - 14
    if state == 0:                       # closed roller shutter
        rect(d, inner0, spring - 6, inner1, B, shutterC)
        yy = spring - 4
        while yy < B:
            rect(d, inner0, yy, inner1, yy + 1, shade(shutterC, 0.28))
            yy += 4
        rect(d, (inner0 + inner1) / 2 - 6, B - 12, (inner0 + inner1) / 2 + 6, B - 9, shade(shutterC, 0.5))
    else:
        openTop = spring + (48 if state == 1 else -4)
        light = rnd.choice(WARM_WINDOWS)
        rect(g, inner0, openTop, inner1, B, rgba(light, 225))
        # shelves / goods as dark shapes in the lit shop
        for k in range(2):
            yy = openTop + 16 + k * 22
            if yy < B - 18:
                rect(g, inner0 + 6, yy, inner1 - 6, yy + 2.5, rgba(shade(light, 0.55)))
                xx = inner0 + 8
                while xx < inner1 - 14:
                    bw = rnd.uniform(5, 11)
                    rect(g, xx, yy - rnd.uniform(6, 14), xx + bw, yy, rgba(shade(light, rnd.uniform(0.3, 0.6))))
                    xx += bw + rnd.uniform(2, 6)
        rect(g, inner0, B - 22, inner1, B, rgba(shade(light, 0.45)))
        if state == 1:
            rect(d, inner0, spring - 6, inner1, openTop, shutterC)
            yy = spring - 4
            while yy < openTop:
                rect(d, inner0, yy, inner1, yy + 1, shade(shutterC, 0.28))
                yy += 4
        p.addLight((inner0 + inner1) / 2, B - 30, 150, light, 0.55)
    # five-foot-way ceiling lamp
    ellipse(g, w / 2, archTop + 6, 3.5, 3, (255, 226, 170, 255))
    p.addLight(w / 2, archTop + 10, 120, (255, 206, 140), 0.35)


def drawAlley(p, w):
    d, g = p.draw, p.glow
    rect(d, 0, 250, w, 440, (22, 22, 38, 255))
    rect(d, 0, 250, 3, 440, (36, 36, 54, 255))
    rect(d, w * 0.55, 330, w * 0.55 + 4, 440, (30, 30, 46, 255))
    ellipse(g, w * 0.5, 360, 2.5, 2.5, (255, 214, 160, 255))
    p.addLight(w * 0.5, 380, 70, (255, 200, 140), 0.4)


def makeStreetMid():
    width, height = 2560, 720
    pic = Picture(width, height)
    rnd = random.Random(606)
    x = -12
    index = 0
    alleys = (4, 9)
    names = list(SHOP_NAMES)
    rnd.shuffle(names)
    lastFacade = None
    while x < width + 12:
        if index in alleys:
            w = 46
            piece = Piece(w, 440)
            drawAlley(piece, w)
            pic.place(piece, x, 160)
            x += w
            index += 1
            continue
        w = rnd.randint(200, 248)
        storeys = 3 if rnd.random() < 0.45 else 2
        facade = rnd.choice(FACADES)
        while facade == lastFacade:
            facade = rnd.choice(FACADES)
        lastFacade = facade
        piece = Piece(w, 440)
        drawShophouse(piece, w, storeys, facade, rnd.choice(SIGN_STYLES), names[index % len(names)],
                      rnd, rnd.choice([0, 1, 1, 2, 2]))
        pic.place(piece, x, 160)
        x += w
        index += 1
    # the pavement in front of the shophouses (hidden behind the main layer, avoids gaps)
    pave = Piece(width, 40)
    rect(pave.draw, 0, 0, width, 40, (70, 66, 72, 255))
    pic.place(pave, 0, 600)
    img = pic.render((0.34, 0.37, 0.58), grain=1.8, seed=41, saturation=0.78, glowAmount=0.85)
    return applyHaze(img, (36, 40, 84), 0.22, 0.3)


# ======================================================================
# 11. STREET MAIN (3200x720 transparent) - the walkable street + stalls
# ======================================================================
def drawGroundStrip(p, rnd, width):
    """Pavement (walkway) + curb + asphalt road. Local y 0 = picture y 570."""
    d = p.draw
    rect(d, 0, 0, width, 150, (52, 50, 58, 255))
    rows = [(4, 16), (16, 29), (29, 44), (44, 61), (61, 82)]
    for r, (y0, y1) in enumerate(rows):
        pw = 44 + r * 4
        x = -(r % 2) * pw / 2
        while x < width:
            c = jitter((138, 132, 128), rnd, 4)
            if rnd.random() < 0.07:
                c = jitter((146, 112, 100), rnd, 4)      # a few older, warmer tiles
            rect(d, x + 1, y0 + 1, x + pw - 1, y1 - 0.5, c)
            rect(d, x + 1, y0 + 1, x + pw - 1, y0 + 2, tint(c, 0.12))
            x += pw
    rect(d, 0, 0, width, 5, (40, 36, 44, 255))               # shadow line at the stall bases
    # curb: top face + painted black/white face (a Malaysian street detail)
    rect(d, 0, 82, width, 88, (184, 180, 172, 255))
    k = 0
    x = 0
    while x < width:
        c = (226, 224, 216, 255) if k % 2 == 0 else (54, 54, 60, 255)
        rect(d, x, 88, x + 60, 98, c)
        x += 60
        k += 1
    rect(d, 0, 98, width, 100, (30, 30, 36, 255))
    # asphalt road with a yellow edge line
    rect(d, 0, 100, width, 150, (60, 60, 70, 255))
    rect(d, 0, 107, width, 109, (214, 180, 70, 255))
    rect(d, 0, 112, width, 114, (214, 180, 70, 255))
    for i in range(90):
        x = rnd.uniform(0, width)
        y = rnd.uniform(118, 148)
        ellipse(d, x, y, rnd.uniform(2, 6), rnd.uniform(0.6, 1.5), (70, 70, 80, 255))


def drawPakDinStall(p, rnd):
    """Stall 1: UNCLE DIN'S FRESH FRUIT stall. Local 500x460, ground at y 452."""
    d = p.draw
    G = 452
    wood, woodDark, woodLight = (150, 100, 62, 255), (104, 66, 42, 255), (196, 142, 92, 255)
    # back wall under the canopy
    rect(d, 52, 150, 448, 336, (96, 66, 48, 255))
    for x in range(52, 448, 24):
        rect(d, x, 150, x + 1.5, 336, (78, 52, 36, 255))
    drawBananaBunch(d, 96, 176, 1.0)
    drawBananaBunch(d, 404, 180, 0.9)
    for bx in (186, 314):
        stroke(d, [(bx, 150), (bx, 184)], (40, 30, 30, 255), 1.4)
        drawBulb(p, bx, 192, 6.5, lightRadius=300, strength=0.75)
    # sloped display with piles of fruit
    poly(d, [(58, 292), (442, 292), (452, 338), (48, 338)], woodDark)
    drawFruitPile(d, rnd, 64, 160, 314, "pineapple", 1.25)
    drawFruitPile(d, rnd, 168, 330, 316, "watermelon", 1.1)
    drawFruitPile(d, rnd, 338, 438, 316, "mango", 1.2)
    drawFruitPile(d, rnd, 70, 430, 334, "rambutan", 1.0)
    # counter
    rect(d, 40, 336, 460, G, wood)
    for x in range(40, 460, 30):
        rect(d, x, 340, x + 2, G, woodDark)
    rect(d, 34, 328, 466, 342, woodLight)
    rect(d, 34, 340, 466, 344, woodDark)
    # crates in front, full of fruit
    drawFruitPile(d, rnd, 70, 176, 382, "watermelon", 1.2)
    drawCrate(d, 64, 380, 182, G)
    drawFruitPile(d, rnd, 196, 300, 404, "orange", 1.3)
    drawCrate(d, 190, 400, 306, G, (190, 140, 84, 255))
    drawFruitPile(d, rnd, 326, 432, 388, "jambu", 1.3)
    drawCrate(d, 318, 384, 440, G)
    # price card
    rect(d, 214, 382, 282, 402, (250, 246, 230, 255))
    label(d, 248, 392, "RM5", "future", 60, 12, (200, 40, 40, 255))
    # posts
    for px in (46, 454):
        rect(d, px - 7, 146, px + 7, G, woodDark)
        rect(d, px - 7, 146, px - 3, G, wood)
    # green-white canopy
    drawStripedAwning(d, 92, 34, 466, 152, 12, 488, 12, (54, 150, 90, 255), (240, 236, 222, 255))
    rect(d, 30, 86, 470, 94, (40, 110, 70, 255))
    # sign board on the canopy
    for sx in (120, 380):
        rect(d, sx - 3, 72, sx + 3, 92, woodDark)
    rrect(d, 50, 18, 450, 80, 8, (92, 58, 36, 255))
    rrect(d, 56, 24, 444, 74, 6, (34, 112, 72, 255))
    rrect(d, 60, 28, 440, 70, 5, (28, 96, 62, 255))
    label(d, 250, 38, "UNCLE DIN", "future", 200, 13, (255, 234, 150, 255), shadow=(16, 60, 36, 255))
    label(d, 250, 58, "FRESH FRUIT", "future", 320, 21, (255, 250, 236, 255), shadow=(16, 60, 36, 255))
    drawFruit(d, rnd, 84, 60, "pineapple", 1.1)
    drawFruit(d, rnd, 416, 60, "mango", 1.3)


def drawAhSengStall(p, rnd):
    """Stall 2: AH SENG CHAR KUEY TEOW hawker cart. Local 500x460, ground at y 452."""
    d, g = p.draw, p.glow
    G = 452
    steel, steelDark, steelLight = (176, 182, 190, 255), (118, 124, 134, 255), (222, 228, 234, 255)
    # tiled back wall
    rect(d, 50, 148, 450, 330, (170, 172, 168, 255))
    for yy in range(148, 330, 18):
        rect(d, 50, yy, 450, yy + 1, (132, 134, 132, 255))
    for xx in range(50, 450, 18):
        rect(d, xx, 148, xx + 1, 330, (132, 134, 132, 255))
    # small menu board (simple bars instead of text)
    rrect(d, 70, 176, 190, 250, 4, (40, 40, 44, 255))
    for k in range(4):
        rect(d, 80, 188 + k * 15, 150 - (k % 2) * 16, 192 + k * 15, (240, 220, 160, 255))
        rect(d, 160, 188 + k * 15, 180, 192 + k * 15, (255, 150, 110, 255))
    # hanging ladle and spatula
    stroke(d, [(220, 160), (220, 214)], steelDark, 2)
    ellipse(d, 220, 222, 10, 7, steel)
    stroke(d, [(244, 160), (244, 206)], (80, 50, 34, 255), 3)
    poly(d, [(236, 206), (252, 206), (256, 226), (232, 226)], steel)
    # glass display case on the cart
    rect(d, 70, 224, 330, 300, (60, 70, 80, 255))
    rect(d, 74, 228, 326, 296, (92, 110, 118, 255))
    for (fx, kind) in ((104, "noodle"), (160, "egg"), (214, "prawn"), (270, "sprout")):
        ellipse(d, fx, 286, 24, 6, (236, 236, 230, 255))
        if kind == "noodle":
            for k in range(6):
                stroke(d, [(fx - 18 + k * 6, 282), (fx - 14 + k * 6, 272)], (240, 222, 180, 255), 3)
        elif kind == "egg":
            for k in range(4):
                ellipse(d, fx - 12 + k * 8, 278, 5, 6.5, (250, 244, 230, 255))
        elif kind == "prawn":
            for k in range(4):
                ellipse(d, fx - 12 + k * 8, 279, 5, 4, (250, 130, 90, 255))
        else:
            for k in range(7):
                stroke(d, [(fx - 16 + k * 5, 282), (fx - 13 + k * 5, 274)], (244, 244, 214, 255), 1.4)
    rect(d, 74, 228, 326, 232, steelLight)
    rect(d, 90, 232, 96, 296, (180, 200, 210, 255))
    rect(d, 68, 222, 332, 226, steel)
    # stainless cart body
    rect(d, 58, 300, 342, G - 6, steel)
    rect(d, 58, 300, 342, 308, steelLight)
    for xx in (58, 152, 246):
        rect(d, xx, 308, xx + 2, G - 6, steelDark)
        rect(d, xx + 2, 308, xx + 8, G - 6, tint(steel, 0.25))
    rect(d, 58, G - 10, 342, G - 4, steelDark)
    ellipse(d, 92, G - 6, 12, 12, (40, 40, 46, 255))
    ellipse(d, 92, G - 6, 5, 5, steelDark)
    ellipse(d, 306, G - 6, 12, 12, (40, 40, 46, 255))
    ellipse(d, 306, G - 6, 5, 5, steelDark)
    # stove section with a glowing burner and a wok
    rect(d, 344, 300, 444, G - 6, (70, 70, 76, 255))
    rect(d, 344, 300, 444, 308, (104, 104, 112, 255))
    poly(g, [(356, 300), (432, 300), (420, 270), (368, 270)], (255, 150, 60, 120))
    for k in range(7):
        fx = 362 + k * 11
        poly(g, [(fx - 5, 300), (fx + 5, 300), (fx + 1, 276 - (k % 3) * 6)], (255, 190, 90, 235))
        poly(g, [(fx - 2.5, 300), (fx + 2.5, 300), (fx + 0.5, 286 - (k % 3) * 3)], (255, 244, 200, 255))
    d.chord(box(346, 250, 442, 296), 0, 180, fill=(40, 40, 44, 255))
    rect(d, 346, 268, 442, 272, (60, 60, 66, 255))
    stroke(d, [(438, 268), (474, 258)], (40, 40, 44, 255), 3)
    rect(g, 354, 304, 434, 308, (80, 150, 255, 200))
    p.addLight(394, 286, 330, (255, 140, 60), 1.0)
    # red gas tank beside the cart
    rrect(d, 452, 380, 486, G, 10, (190, 44, 40, 255))
    rect(d, 462, 370, 476, 382, (120, 120, 124, 255))
    rect(d, 452, 396, 486, 400, (150, 30, 30, 255))
    # poles
    for px in (44, 456):
        rect(d, px - 3.5, 140, px + 3.5, G, steelDark)
        rect(d, px - 3.5, 140, px - 1.5, G, steelLight)
    # red canopy with gold trim and small lanterns
    red, redDark = (206, 46, 42, 255), (160, 30, 30, 255)
    poly(d, [(28, 86), (472, 86), (494, 144), (6, 144)], red)
    for k in range(9):
        fx0 = 28 + k * 49.3
        fx1 = 6 + k * 54.2
        poly(d, [(fx0, 86), (fx0 + 10, 86), (fx1 + 12, 144), (fx1, 144)], redDark)
    rect(d, 6, 140, 494, 146, (240, 190, 80, 255))
    for k in range(20):
        halfDisc(d, 18 + k * 24.2, 146, 12, red, top=False)
        halfDisc(d, 18 + k * 24.2, 146, 7, redDark, top=False)
    drawRedLantern(p, 20, 164, 22, 28, light=0.45)
    drawRedLantern(p, 480, 164, 22, 28, light=0.45)
    # signboard (classic yellow board, red letters)
    rrect(d, 48, 12, 452, 82, 6, (150, 30, 30, 255))
    rrect(d, 54, 18, 446, 76, 4, (250, 214, 96, 255))
    rect(d, 54, 64, 446, 76, (234, 190, 70, 255))
    label(d, 250, 33, "AH SENG", "future", 180, 15, (170, 30, 30, 255), shadow=(250, 240, 180, 255), shadowOffset=(0, -1))
    label(d, 250, 58, "CHAR KUEY TEOW", "future", 360, 22, (196, 34, 30, 255), shadow=(120, 20, 20, 255))


def drawMamakStall(p, rnd):
    """Stall 3: SRI PELITA CAFE (24 HOURS) shop front. Local 500x460, ground at y 452."""
    d, g = p.draw, p.glow
    G = 452
    tileG = (70, 150, 110, 255)
    # interior back wall (tiles) - the light inside makes it bright at night
    rect(d, 60, 124, 440, G, (184, 198, 190, 255))
    for yy in range(124, G, 20):
        rect(d, 60, yy, 440, yy + 1, (170, 180, 168, 255))
    for xx in range(60, 440, 20):
        rect(d, xx, 124, xx + 1, G, (170, 180, 168, 255))
    rect(d, 60, 300, 440, G, (78, 150, 112, 255))
    rect(d, 60, 298, 440, 302, (40, 96, 70, 255))
    # fluorescent tube + menu lightbox inside
    rrect(g, 150, 134, 350, 140, 3, (236, 248, 255, 255))
    p.addLight(250, 170, 330, (220, 240, 255), 0.45)
    rrect(d, 96, 160, 240, 228, 4, (40, 40, 44, 255))
    rrect(g, 100, 164, 236, 224, 3, (255, 236, 170, 200))
    for k in range(3):
        rect(g, 108, 172 + k * 17, 176, 178 + k * 17, (200, 60, 40, 255))
        rect(g, 196, 172 + k * 17, 228, 178 + k * 17, (40, 40, 40, 255))
    # counter with the tea urn and roti display
    rect(d, 70, 300, 430, 372, (184, 190, 198, 255))
    rect(d, 70, 300, 430, 308, (230, 236, 240, 255))
    rrect(d, 290, 222, 346, 300, 8, (190, 196, 204, 255))
    rect(d, 298, 226, 306, 296, (236, 240, 244, 255))
    ellipse(d, 318, 222, 24, 7, (160, 166, 176, 255))
    rect(d, 344, 272, 360, 278, (200, 150, 60, 255))
    rect(d, 120, 262, 250, 300, (120, 160, 176, 255))
    rect(d, 124, 266, 246, 296, (168, 200, 210, 255))
    for k in range(4):
        ellipse(d, 142 + k * 30, 288, 13, 5, (232, 196, 120, 255))
    # tiled pillars
    for (x0, x1) in ((18, 62), (438, 482)):
        rect(d, x0, 104, x1, G, tileG)
        for yy in range(104, G, 16):
            rect(d, x0, yy, x1, yy + 1, (48, 110, 80, 255))
        rect(d, x0, 104, x0 + 4, G, (110, 190, 150, 255))
    # rolled-up shutter box
    rect(d, 18, 102, 482, 124, (150, 156, 164, 255))
    rect(d, 18, 120, 482, 124, (96, 100, 110, 255))
    # table and plastic stools in front
    drawPlasticStool(d, 92, G, (60, 110, 200, 255), h=40)
    drawTableSet(p, 300, G, [(210, 60, 50, 255), (60, 110, 200, 255), (220, 180, 50, 255)])
    drawPlasticStool(d, 172, G, (210, 60, 50, 255), h=40)
    # big green-yellow signboard (lit box sign)
    rrect(d, 8, 8, 492, 104, 8, (26, 66, 44, 255))
    rrect(g, 14, 14, 486, 98, 6, (28, 128, 76, 245))
    rect(g, 14, 70, 486, 98, (232, 192, 58, 245))
    rrect(g, 18, 18, 482, 26, 3, (80, 180, 120, 230))
    label(g, 250, 44, "SRI PELITA CAFE", "future", 440, 29, (255, 226, 80, 255), shadow=(12, 60, 34, 255))
    rrect(g, 190, 74, 310, 95, 6, (206, 38, 40, 255))
    label(g, 250, 84.5, "24 HOURS", "future", 100, 15, (255, 255, 255, 255))
    p.addLight(250, 60, 300, (120, 230, 140), 0.35)


def drawDewanPelita(p, rnd):
    """Stall 4: PELITA HALL banquet hall entrance. Local 500x460, ground at y 452."""
    d, g = p.draw, p.glow
    G = 452
    red, redDark, redLight = (176, 36, 34, 255), (122, 22, 24, 255), (206, 60, 50, 255)
    gold, goldDark = (236, 186, 84, 255), (170, 122, 50, 255)
    # wall
    rect(d, 26, 100, 474, G, red)
    for yy in range(112, G, 26):
        rect(d, 26, yy, 474, yy + 1, redDark)
    # door frame + red double doors with gold studs
    rect(d, 142, 172, 358, G - 22, goldDark)
    rect(d, 148, 178, 352, G - 22, gold)
    rect(d, 156, 186, 344, G - 22, redDark)
    for (x0, x1) in ((158, 248), (252, 342)):
        rect(d, x0, 188, x1, G - 22, (192, 40, 36, 255))
        rect(d, x0, 188, x0 + 5, G - 22, redLight)
        for row in range(6):
            for col in range(4):
                sx = x0 + 14 + col * (x1 - x0 - 28) / 3
                sy = 206 + row * 36
                ellipse(d, sx, sy, 3.2, 3.2, goldDark)
                ellipse(d, sx - 0.8, sy - 0.8, 2.2, 2.2, gold)
    for kx in (236, 264):
        ring(d, kx, 318, 8, 8, gold, 2.2)
        ellipse(d, kx, 308, 5, 5, gold)
    rect(g, 248.5, 188, 251.5, G - 22, (255, 206, 130, 230))       # light through the door gap
    p.addLight(250, 330, 160, (255, 190, 120), 0.35)
    # columns
    for (x0, x1) in ((58, 98), (402, 442)):
        rect(d, x0, 104, x1, G - 12, redLight)
        rect(d, x0 + 26, 104, x1, G - 12, red)
        rect(d, x0 + 4, 104, x0 + 10, G - 12, (236, 100, 80, 255))
        for yy in (116, G - 40):
            rect(d, x0 - 4, yy, x1 + 4, yy + 10, gold)
            rect(d, x0 - 4, yy + 8, x1 + 4, yy + 10, goldDark)
    # steps
    rect(d, 112, G - 22, 388, G - 11, (150, 146, 150, 255))
    rect(d, 112, G - 22, 388, G - 20, (196, 194, 198, 255))
    rect(d, 96, G - 11, 404, G, (130, 126, 132, 255))
    rect(d, 96, G - 11, 404, G - 9, (180, 178, 184, 255))
    # plaque
    rrect(d, 142, 116, 358, 164, 5, goldDark)
    rrect(d, 146, 120, 354, 160, 4, (30, 22, 26, 255))
    rrect(d, 150, 124, 350, 156, 3, (44, 30, 34, 255))
    label(d, 250, 140.5, "PELITA HALL", "future", 186, 21, gold, shadow=(10, 6, 8, 255))
    # curved roof with upturned eaves
    roofC, roofDark = (96, 40, 40, 255), (64, 24, 28, 255)
    eave = []
    for i in range(41):
        t = i / 40
        x = -4 + 508 * t
        y = 96 - 22 * (abs(t - 0.5) * 2) ** 3.2
        eave.append((x, y))
    poly(d, [(40, 34), (460, 34)] + list(reversed(eave)), roofC)
    for k in range(18):
        x = 44 + k * 23.5
        stroke(d, [(x, 38), (x + (x - 250) * 0.08, 92 - 20 * (abs(x - 250) / 254) ** 3.2)], roofDark, 2.2)
    stroke(d, eave, gold, 3)
    stroke(d, [(e[0], e[1] + 4) for e in eave], redDark, 3)
    rect(d, 30, 24, 470, 36, roofDark)
    rect(d, 30, 24, 470, 27, gold)
    poly(d, [(30, 24), (14, 10), (26, 12), (40, 26)], gold)
    poly(d, [(470, 24), (486, 10), (474, 12), (460, 26)], gold)
    ellipse(d, 250, 18, 8, 8, gold)
    # big lanterns
    drawRedLantern(p, 110, 132, 40, 50, light=0.6)
    drawRedLantern(p, 390, 132, 40, 50, light=0.6)
    # mandarin orange plants by the steps
    for px in (72, 428):
        rect(d, px - 16, G - 30, px + 16, G, (60, 90, 160, 255))
        rect(d, px - 18, G - 32, px + 18, G - 26, (230, 236, 244, 255))
        for k in range(14):
            ellipse(d, px + rnd.uniform(-18, 18), G - 44 - rnd.uniform(0, 26), 8, 7, jitter((50, 110, 56), rnd, 10))
        for k in range(9):
            ellipse(d, px + rnd.uniform(-16, 16), G - 44 - rnd.uniform(0, 24), 3, 3, (250, 150, 40, 255))


def drawTrishaw(p, rnd, x, groundY):
    """A decorated trishaw (beca) parked by the road. (x, groundY) = centre bottom."""
    d, g = p.draw, p.glow
    tyre, metal = (34, 34, 40, 255), (170, 176, 184, 255)
    def wheel(cx, r):
        ring(d, cx, groundY - r, r, r, tyre, 4)
        ring(d, cx, groundY - r, r - 4, r - 4, metal, 1.2)
        for k in range(8):
            a = k * math.pi / 8
            stroke(d, [(cx - math.cos(a) * (r - 4), groundY - r - math.sin(a) * (r - 4)),
                       (cx + math.cos(a) * (r - 4), groundY - r + math.sin(a) * (r - 4))], metal, 0.7)
        ellipse(d, cx, groundY - r, 3, 3, metal)
    wheel(x - 58, 30)
    wheel(x + 78, 34)
    # frame
    frame = (40, 110, 150, 255)
    stroke(d, [(x + 78, groundY - 34), (x + 36, groundY - 70), (x - 6, groundY - 58)], frame, 3)
    stroke(d, [(x + 36, groundY - 70), (x + 44, groundY - 100)], frame, 3)
    stroke(d, [(x + 78, groundY - 34), (x + 62, groundY - 104)], frame, 3)
    stroke(d, [(x + 54, groundY - 108), (x + 74, groundY - 104)], (40, 40, 44, 255), 3)
    rrect(d, x + 34, groundY - 106, x + 54, groundY - 99, 3, (50, 40, 40, 255))
    # passenger carriage
    rrect(d, x - 108, groundY - 104, x + 8, groundY - 50, 10, (196, 40, 70, 255))
    rrect(d, x - 104, groundY - 100, x + 4, groundY - 88, 6, (236, 90, 110, 255))
    rect(d, x - 100, groundY - 62, x, groundY - 54, (160, 30, 56, 255))
    for k in range(5):
        ellipse(d, x - 92 + k * 20, groundY - 76, 5, 5, (250, 220, 90, 255))
    # canopy + flowers + fairy lights
    stroke(d, [(x - 104, groundY - 100), (x - 100, groundY - 170)], metal, 2)
    stroke(d, [(x + 2, groundY - 100), (x - 6, groundY - 170)], metal, 2)
    d.pieslice(box(x - 124, groundY - 206, x + 18, groundY - 134), 180, 360, fill=(250, 196, 70, 255))
    d.pieslice(box(x - 124, groundY - 206, x - 53, groundY - 134), 180, 360, fill=(240, 170, 50, 255))
    for k in range(9):
        halfDisc(d, x - 118 + k * 16.5, groundY - 170, 8, (236, 90, 110, 255), top=False)
    for k in range(10):
        ellipse(d, x - 118 + k * 15, groundY - 176 - 10 * math.sin(k / 9 * math.pi), 4.5, 4.5,
                (255, 120, 170, 255) if k % 2 else (255, 240, 240, 255))
    for k in range(12):
        lx = x - 120 + k * 12.5
        ellipse(g, lx, groundY - 158, 1.8, 1.8, rgba([(255, 230, 150), (255, 150, 200), (150, 220, 255)][k % 3]))
    p.addLight(x - 50, groundY - 150, 140, (255, 200, 170), 0.35)


def makeStreetMain():
    width, height = 3200, 720
    pic = Picture(width, height)
    rnd = random.Random(99)
    ground = Piece(width, 150)
    drawGroundStrip(ground, rnd, width)
    pic.place(ground, 0, 570)
    # soft shadows on the pavement under stalls and props
    shapes = []
    for sx in STALL_XS:
        shapes.append(("ellipse", sx - 240, STALL_BASE_Y - 10, sx + 240, STALL_BASE_Y + 26))
    for (sx, rx) in ((OYEN_CRATE_X, 60), (1566, 50), (975, 110), (2370, 130), (250, 30)):
        shapes.append(("ellipse", sx - rx, STALL_BASE_Y - 6, sx + rx, STALL_BASE_Y + 16))
    pic.overlay(softShape(width, height, shapes, (8, 4, 14), 150, 10))
    # the four stalls, centred exactly on settings.STALL_X
    stallDrawers = [drawPakDinStall, drawAhSengStall, drawMamakStall, drawDewanPelita]
    for sx, drawer in zip(STALL_XS, stallDrawers):
        piece = Piece(500, 460)
        drawer(piece, rnd)
        pic.place(piece, sx - 250, STALL_BASE_Y - 452)
    # props between the stalls
    props = Piece(width, 460)            # local y 452 = STALL_BASE_Y
    G = 452
    drawLampPost(props, 70, G, 300)
    drawStreetSign(props, 250, G)
    drawPottedPlant(props, 360, G, "banana", rnd)
    drawTableSet(props, 975, G, [(210, 60, 50, 255), (60, 110, 200, 255), (210, 60, 50, 255)])
    drawPottedPlant(props, 1076, G, "bougainvillea", rnd)
    # crates: a stack + the single crate Oyen sits on
    drawCrate(props.draw, 1530, G - 58, 1604, G, (170, 120, 70, 255))
    drawFruitPile(props.draw, rnd, 1540, 1594, G - 104, "orange", 1.2)
    drawCrate(props.draw, 1534, G - 104, 1600, G - 58, (186, 132, 80, 255))
    crateTop = OYEN_CRATE_TOP - (STALL_BASE_Y - G)
    drawCrate(props.draw, OYEN_CRATE_X - 46, crateTop, OYEN_CRATE_X + 46, G, (180, 128, 76, 255))
    rect(props.draw, OYEN_CRATE_X - 48, crateTop - 3, OYEN_CRATE_X + 48, crateTop + 3, (206, 156, 100, 255))
    drawLampPost(props, 1768, G, 300)
    drawTrishaw(props, rnd, 2370, G + 8)
    drawPottedPlant(props, 2980, G, "palm", rnd)
    drawLampPost(props, 3070, G, 300)
    drawPottedPlant(props, 3160, G, "bougainvillea", rnd)
    pic.place(props, 0, STALL_BASE_Y - G)
    # warm glow pools on the pavement in front of every stall
    for sx in STALL_XS:
        pic.addLight(sx, STALL_BASE_Y + 30, 300, (255, 186, 120), 0.45, squash=0.2)
    # the road is a little wet: bright things above it make long soft vertical
    # reflections (a stretched Gaussian blob drawn straight into the light energy)
    reflections = [(620, 150, (255, 230, 170)), (1320, 150, (255, 150, 90)), (2020, 190, (140, 255, 170)),
                   (2720, 150, (255, 90, 70)), (70, 40, (255, 220, 150)), (1768, 40, (255, 220, 150)),
                   (3070, 40, (255, 220, 150))]
    xs, ys = grid(width, 60)
    for (rx, spread, rgb) in reflections:
        blob = np.exp(-((xs - rx) / spread) ** 2 - ((ys - 34) / 22) ** 2)
        pic.energy[660:720] += (blob * 38)[:, :, None] * (np.array(rgb, np.float32) / 255.0)
    return pic.render((0.56, 0.57, 0.76), grain=1.8, seed=51, glowAmount=0.9)


# ======================================================================
# 12. SCENE 1: FRUIT STALL (1280x720 opaque)
# ======================================================================
def nightMarketBackdrop(width, height, seed, horizonY=380):
    """Blurry night market far behind the stall: dark stalls, strings of lights, bokeh."""
    rnd = random.Random(seed)
    arr = gradientArray(width, height, [(0.0, (16, 16, 42)), (horizonY / height * 0.6, (34, 26, 62)),
                                        (horizonY / height, (70, 40, 60)), (1.0, (40, 26, 36))])
    img = toImage(arr)
    far = Piece(width, height)
    x = -20
    while x < width:
        w = rnd.randint(90, 160)
        top = horizonY - rnd.randint(70, 130)
        body = jitter((36, 26, 50), rnd, 5)
        poly(far.draw, [(x, top + 26), (x + w / 2, top), (x + w, top + 26)], body)
        rect(far.draw, x + 4, top + 26, x + w - 4, height, shade(body, 0.1))
        rect(far.glow, x + 8, top + 34, x + w - 8, horizonY + 40, rgba(rnd.choice(WARM_WINDOWS), 110))
        x += w + rnd.randint(4, 30)
    for k in range(22):
        px = rnd.uniform(0, width)
        py = horizonY + rnd.uniform(-10, 40)
        ellipse(far.glow, px, py - 26, 9, 10, (26, 18, 34, 255))
        rrect(far.glow, px - 13, py - 18, px + 13, py + 60, 10, (26, 18, 34, 255))
    for (y0, drop) in ((150, 40), (210, 30)):
        drawLanternString(far, rnd, -20, y0, width + 20, y0 - 10, drop, 26,
                          [(255, 200, 120), (255, 150, 90), (255, 226, 160)], size=1.0)
    colour, glow = far.finish()
    img.alpha_composite(colour)
    img.alpha_composite(glow)
    img = img.filter(ImageFilter.GaussianBlur(3.5))     # depth of field: far things are soft
    arr = toArray(img)
    spots = []
    # mostly warm lantern colours, a few cooler ones for variety
    colours = [(255, 184, 100), (255, 150, 76), (255, 214, 150), (246, 124, 92), (255, 198, 116),
               (255, 184, 100), (150, 196, 200), (214, 146, 170)]
    for i in range(64):
        r = rnd.uniform(5, 26)
        c = rnd.choice(colours)
        spots.append((rnd.uniform(0, width), rnd.uniform(80, horizonY + 60), r, c, rnd.uniform(0.12, 0.32)))
    arr[:, :, :3] += bokehEnergy(width, height, spots)
    return toImage(arr)


def drawShelfRack(p, rnd, x0, x1, top, shelves, mirror=False):
    d = p.draw
    wood, woodDark = (132, 88, 56, 255), (92, 60, 38, 255)
    rect(d, x0, top, x0 + 12, shelves[-1] + 16, woodDark)
    rect(d, x1 - 12, top, x1, shelves[-1] + 16, woodDark)
    for y in shelves:
        rect(d, x0 - 4, y, x1 + 4, y + 12, wood)
        rect(d, x0 - 4, y, x1 + 4, y + 3, tint(wood, 0.25))
        rect(d, x0 - 4, y + 10, x1 + 4, y + 12, woodDark)


def drawCounterPlanks(p, rnd, width, top, bottom, rgb):
    """Wooden counter top seen from above: planks get wider towards the viewer (perspective)."""
    d = p.draw
    y = top
    h = 24
    while y < bottom:
        y1 = min(bottom, y + h)
        c = jitter(rgb, rnd, 7)
        rect(d, 0, y, width, y1, c)
        # wood grain: long thin darker and lighter streaks
        for k in range(int(width / 60)):
            gx = rnd.uniform(-40, width)
            gy = rnd.uniform(y + 2, y1 - 2)
            ln = rnd.uniform(80, 260)
            stroke(d, [(gx, gy), (gx + ln * 0.5, gy + rnd.uniform(-1.5, 1.5)), (gx + ln, gy)],
                   shade(c, 0.12) if k % 3 else tint(c, 0.08), rnd.uniform(0.8, 1.6))
        if rnd.random() < 0.5:
            kx = rnd.uniform(60, width - 60)
            ellipse(d, kx, (y + y1) / 2, 7, 3, shade(c, 0.25))
            ellipse(d, kx, (y + y1) / 2, 4, 1.6, shade(c, 0.4))
        # plank seam: dark gap + light bevel
        rect(d, 0, y1 - 1.6, width, y1, shade(c, 0.45))
        rect(d, 0, y1, width, y1 + 1, tint(c, 0.12))
        y = y1
        h = h * 1.16
        # butt joints between planks
        jx = rnd.uniform(120, width - 120)
        rect(d, jx, y - h / 1.16, jx + 1.6, y, shade(c, 0.45))


def makeFruitStallBg():
    width, height = 1280, 720
    rnd = random.Random(301)
    backdrop = nightMarketBackdrop(width, height, 302)
    pic = Picture(width, height)
    stall = Piece(width, height)
    d = stall.draw
    # side shelves full of fruit crates
    drawShelfRack(stall, rnd, 10, 280, 150, [286, 386])
    drawShelfRack(stall, rnd, 1000, 1270, 150, [286, 386])
    crates = [(18, 146, 286, "pineapple"), (150, 272, 286, "mango"), (18, 146, 386, "cucumber"),
              (150, 272, 386, "jambu"), (1008, 1136, 286, "watermelon"), (1140, 1262, 286, "pineapple"),
              (1008, 1136, 386, "mango"), (1140, 1262, 386, "orange")]
    for (x0, x1, shelfY, kind) in crates:
        drawFruitPile(d, rnd, x0 + 6, x1 - 6, shelfY - 36, kind, 1.9)
        drawCrate(d, x0, shelfY - 44, x1, shelfY)
    # hanging banana bunches and price tags
    drawBananaBunch(d, 300, 132, 1.8)
    drawBananaBunch(d, 980, 132, 1.7)
    for (tx, ty, words) in ((84, 216, "RM3"), (212, 216, "RM4"), (1072, 216, "RM8"), (1200, 316, "RM2")):
        rrect(d, tx - 22, ty - 11, tx + 22, ty + 11, 3, (250, 246, 232, 255))
        label(d, tx, ty, words, "future", 38, 13, (200, 40, 40, 255))
    # side posts
    for (x0, x1) in ((0, 30), (1250, 1280)):
        rect(d, x0, 60, x1, 400, (92, 60, 40, 255))
        rect(d, x0, 60, x0 + 8, 400, (132, 90, 58, 255))
    # counter top (warm planks) with a back rail
    drawCounterPlanks(stall, rnd, width, 392, height, (126, 84, 54))
    rect(d, 0, 382, width, 396, (92, 58, 38, 255))
    rect(d, 0, 382, width, 385, (150, 104, 66, 255))
    # low crate of chillies at the back-left corner (keeps the bowl area clear)
    drawFruitPile(d, rnd, 16, 128, 416, "chili", 2.0)
    drawCrate(d, 8, 404, 136, 452, (170, 120, 72, 255))
    # woven basket of fruit at the lower right (ingredients slide out of it)
    drawWovenBasket(stall, rnd, 1120, 532)
    # a few leaves on the counter
    for (lx, ly, ang) in ((930, 640, 20), (960, 660, -30), (360, 690, 160), (1040, 450, 200)):
        poly(d, leafPoints(lx, ly, 34, 12, ang), (70, 130, 60, 255))
    # canopy scallops + hanging bulbs
    drawStripedAwning(d, -30, -40, width + 40, 78, -60, width + 60, 20, (46, 136, 82, 255), (236, 232, 214, 255))
    rect(d, 0, -2, width, 22, (30, 86, 54, 255))
    for bx in (470, 810):
        stroke(d, [(bx, 100), (bx, 142)], (40, 30, 30, 255), 1.6)
        drawBulb(stall, bx, 152, 10, lightRadius=620, strength=0.72)
    # hanging sign FRUIT ROJAK RM6
    for sx in (86, 334):
        stroke(d, [(sx, 100), (sx, 132)], (60, 40, 30, 255), 1.6)
    rrect(d, 50, 126, 370, 192, 8, (110, 70, 42, 255))
    rrect(d, 56, 132, 364, 186, 6, (240, 228, 198, 255))
    label(d, 176, 160, "FRUIT ROJAK", "future", 210, 27, (190, 36, 36, 255), shadow=(120, 80, 60, 255))
    ellipse(d, 318, 159, 36, 24, (46, 136, 82, 255))
    label(d, 318, 160, "RM6", "future", 56, 21, (255, 250, 236, 255))
    pic.place(stall, 0, 0)
    # a soft darker shadow under the canopy edge and around the counter corners
    pic.overlay(verticalFade(width, 60, (10, 6, 16), 110, 0), 0, 100)
    # warm pool of light on the counter where the cutting board goes (Arcade (640, 235))
    pic.addLight(640, 500, 720, (255, 196, 140), 0.42, squash=0.42)
    return pic.render((0.50, 0.48, 0.60), background=backdrop, grain=2.0, seed=61, vignette=0.42,
                      saturation=0.9)


def drawWovenBasket(p, rnd, cx, cy):
    """Round rattan basket heaped with rojak fruit (3/4 view)."""
    d = p.draw
    rattan, rattanDark = (196, 150, 90, 255), (140, 96, 54, 255)
    # fruit heap first (it sits inside the basket)
    heap = [("pineapple", -60, -28), ("mango", -20, -34), ("jambu", 18, -30), ("cucumber", 56, -26),
            ("mango", -70, -8), ("jambu", -36, -10), ("pineapple", 4, -12), ("chili", 44, -10), ("jambu", 76, -8),
            ("cucumber", -10, 2), ("chili", 30, 4)]
    for (kind, dx, dy) in heap:
        drawFruit(d, rnd, cx + dx, cy + dy, kind, 2.1)
    # basket body
    poly(d, [(cx - 118, cy - 6), (cx + 118, cy - 6), (cx + 96, cy + 56), (cx - 96, cy + 56)], rattan)
    for k in range(-8, 9):
        x0 = cx + k * 13.5
        stroke(d, [(x0, cy - 4), (x0 * 0.82 + cx * 0.18, cy + 54)], rattanDark, 1.4)
    for row in range(5):
        y = cy + 4 + row * 10.5
        inset = 118 - (22 * (row + 1) / 5)
        for k in range(-9, 9):
            x = cx + k * 13 + (row % 2) * 6.5
            if abs(x - cx) < inset - 6:
                ellipse(d, x, y, 5.6, 3.2, tint(rattan, 0.12))
    ellipse(d, cx, cy - 6, 120, 13, rattanDark)
    ellipse(d, cx, cy - 8, 118, 10, tint(rattan, 0.18))
    d.chord(box(cx - 116, cy - 16, cx + 116, cy + 2), 180, 360, fill=shade(rattan, 0.4))
    for (kind, dx, dy) in (("mango", -40, -12), ("jambu", 10, -14), ("pineapple", 60, -12)):
        drawFruit(d, rnd, cx + dx, cy + dy, kind, 2.1)


# ======================================================================
# 13. SCENE 2: WOK KITCHEN (1280x720 opaque)
# ======================================================================
def drawTiles(p, rnd, x0, y0, x1, y1, size, rgb, groutRgb, variation=5, gap=1.6):
    d = p.draw
    rect(d, x0, y0, x1, y1, rgba(groutRgb))
    y = y0
    while y < y1:
        x = x0
        while x < x1:
            c = jitter(rgb, rnd, variation)
            rect(d, x + gap / 2, y + gap / 2, x + size - gap / 2, y + size - gap / 2, c)
            rect(d, x + gap / 2, y + gap / 2, x + size - gap / 2, y + gap / 2 + 1.4, tint(c, 0.18))
            rect(d, x + gap / 2, y + gap / 2, x + gap / 2 + 1.4, y + size - gap / 2, tint(c, 0.12))
            rect(d, x + gap / 2, y + size - gap / 2 - 1.2, x + size - gap / 2, y + size - gap / 2, shade(c, 0.1))
            x += size
        y += size


def cityNightView(width, height, seed):
    """Night view for small windows: sky gradient, far buildings, lit windows, blurred."""
    rnd = random.Random(seed)
    arr = gradientArray(width, height, [(0.0, (14, 16, 44)), (0.25, (30, 32, 80)), (0.4, (70, 50, 110)),
                                        (1.0, (40, 30, 60))])
    img = toImage(arr)
    city = Piece(width, height)
    drawCityRow(city, rnd, -10, width + 10, height, (120, 230), (20, 60), (26, 28, 60), 0.2,
                WARM_WINDOWS + COOL_WINDOWS, (140, 240), cell=(6, 8), win=(3, 4))
    colour, glow = city.finish()
    img.alpha_composite(colour)
    if glow is not None:
        img.alpha_composite(glow)
    return img.filter(ImageFilter.GaussianBlur(1.2))


def makeWokKitchenBg():
    width, height = 1280, 720
    rnd = random.Random(402)
    view = cityNightView(width, height, 403)
    vArr = toArray(view)
    moonHalo(vArr, 250, 78, 16, (230, 220, 210), 0.5)
    view = toImage(vArr)
    moon = moonImage(16, seed=9)
    pasteOver(view, moon, 250 - moon.width // 2, 78 - moon.height // 2)
    pic = Picture(width, height)
    wall = Piece(width, 470)
    d, g = wall.draw, wall.glow
    drawTiles(wall, rnd, 0, 0, width, 300, 40, (224, 226, 214), (160, 164, 152))
    rect(d, 0, 298, width, 312, (36, 106, 76, 255))
    rect(d, 0, 298, width, 300, (90, 170, 130, 255))
    drawTiles(wall, rnd, 0, 312, width, 470, 40, (74, 154, 114), (46, 96, 72))
    # window (hole in the wall) with frame and iron grille
    wx0, wy0, wx1, wy1 = 60, 40, 300, 190
    rect(d, wx0 - 12, wy0 - 12, wx1 + 12, wy1 + 16, (48, 96, 76, 255))
    rect(d, wx0 - 12, wy1 + 8, wx1 + 12, wy1 + 16, (130, 140, 132, 255))
    rect(d, wx0, wy0, wx1, wy1, (0, 0, 0, 0))
    grille = (30, 44, 40, 255)
    for k in range(1, 6):
        x = wx0 + k * (wx1 - wx0) / 6
        rect(d, x - 1.5, wy0, x + 1.5, wy1, grille)
    rect(d, wx0, (wy0 + wy1) / 2 - 1.5, wx1, (wy0 + wy1) / 2 + 1.5, grille)
    for k in range(3):
        cx = wx0 + (k + 0.5) * (wx1 - wx0) / 3
        ring(d, cx, (wy0 + wy1) / 2, 18, 18, grille, 2.4)
    rect(d, (wx0 + wx1) / 2 - 3, wy0, (wx0 + wx1) / 2 + 3, wy1, (70, 110, 94, 255))
    # extractor hood
    steel, steelDark, steelLight = (176, 182, 190, 255), (112, 118, 128, 255), (226, 230, 236, 255)
    poly(d, [(436, -4), (844, -4), (890, 122), (390, 122)], steel)
    poly(d, [(436, -4), (560, -4), (530, 122), (390, 122)], tint(steel, 0.18))
    poly(d, [(740, -4), (844, -4), (890, 122), (770, 122)], shade(steel, 0.14))
    for row in range(3):
        for k in range(10):
            vx = 500 + k * 29
            vy = 38 + row * 22
            rrect(d, vx, vy, vx + 20, vy + 5, 2.5, steelDark)
    rect(d, 384, 118, 896, 146, steelLight)
    rect(d, 384, 118, 896, 122, (244, 246, 250, 255))
    rect(d, 384, 140, 896, 146, steelDark)
    rect(d, 392, 146, 888, 154, (70, 72, 80, 255))
    rrect(g, 470, 147, 810, 153, 3, (255, 244, 214, 255))
    wall.addLight(640, 200, 600, (255, 236, 200), 0.46)
    for k in range(12):
        ellipse(d, 400 + k * 44, 132, 2, 2, steelDark)
    # shelf with jars and sauce bottles (left)
    rect(d, 20, 282, 340, 294, (120, 84, 54, 255))
    rect(d, 20, 282, 340, 285, (170, 124, 82, 255))
    for bx in (40, 310):
        poly(d, [(bx - 4, 294), (bx + 4, 294), (bx + 4, 326), (bx - 20 if bx < 100 else bx + 20, 294)], (80, 60, 44, 255))
    jars = [(46, 38, (200, 40, 30)), (92, 38, (244, 236, 214)), (138, 38, (236, 130, 60))]
    for (jx, jw, content) in jars:
        rrect(d, jx - jw / 2, 226, jx + jw / 2, 282, 6, (190, 206, 210, 255))
        rrect(d, jx - jw / 2 + 3, 240, jx + jw / 2 - 3, 279, 4, rgba(content))
        for k in range(6):
            ellipse(d, jx + rnd.uniform(-12, 12), rnd.uniform(246, 274), 2.5, 2, shade(content, 0.25))
        rect(d, jx - jw / 2 + 4, 230, jx - jw / 2 + 8, 276, (236, 244, 246, 255))
        rrect(d, jx - jw / 2 - 1, 218, jx + jw / 2 + 1, 230, 3, (200, 50, 40, 255))
    bottles = [(186, 16, 86, (40, 26, 20), (200, 40, 36)), (214, 16, 72, (90, 50, 26), (240, 200, 60)),
               (240, 14, 64, (210, 40, 30), (240, 240, 230))]
    for (bx, bw, bh, body, cap) in bottles:
        rrect(d, bx - bw / 2, 282 - bh, bx + bw / 2, 282, 5, rgba(body))
        rect(d, bx - 4, 282 - bh - 14, bx + 4, 282 - bh + 2, rgba(body))
        rect(d, bx - 4.5, 282 - bh - 20, bx + 4.5, 282 - bh - 13, rgba(cap))
        rect(d, bx - bw / 2 + 1, 282 - bh * 0.6, bx + bw / 2 - 1, 282 - bh * 0.25, (240, 230, 200, 255))
        rect(d, bx - bw / 2 + 2, 282 - bh + 4, bx - bw / 2 + 4, 278, tint(body, 0.4))
    for k in range(4):
        ellipse(d, 300, 276 - k * 8, 30, 6, (236, 240, 244, 255))
        ellipse(d, 300, 274 - k * 8, 26, 3.5, (60, 110, 170, 255))
    # hanging utensils (right)
    rect(d, 930, 168, 1250, 174, steelDark)
    rect(d, 930, 168, 1250, 170, steelLight)
    for bx in (944, 1236):
        rect(d, bx - 4, 158, bx + 4, 180, (90, 96, 104, 255))
    utensils = [(976, "spatula"), (1030, "ladle"), (1090, "strainer"), (1148, "ladle_small"), (1204, "cleaver")]
    for (ux, kind) in utensils:
        ring(d, ux, 180, 5, 5, steelDark, 1.6)
        if kind == "spatula":
            rect(d, ux - 2.5, 184, ux + 2.5, 280, steelDark)
            rect(d, ux - 3.5, 184, ux + 3.5, 230, (74, 46, 30, 255))
            poly(d, [(ux - 16, 280), (ux + 16, 280), (ux + 20, 318), (ux - 20, 318)], steel)
            poly(d, [(ux - 16, 280), (ux - 6, 280), (ux - 8, 318), (ux - 20, 318)], steelLight)
        elif kind in ("ladle", "ladle_small"):
            big = 1.0 if kind == "ladle" else 0.75
            rect(d, ux - 2, 184, ux + 2, 184 + 110 * big, steelDark)
            rect(d, ux - 3, 184, ux + 3, 214, (74, 46, 30, 255))
            ellipse(d, ux, 196 + 110 * big, 22 * big, 18 * big, steel)
            ellipse(d, ux - 6 * big, 190 + 110 * big, 9 * big, 7 * big, steelLight)
        elif kind == "strainer":
            rect(d, ux - 2, 184, ux + 2, 270, (150, 110, 60, 255))
            ring(d, ux, 296, 26, 26, (184, 150, 90, 255), 2.4)
            for k in range(-2, 3):
                stroke(d, [(ux + k * 9, 272), (ux + k * 9, 320)], (184, 150, 90, 255), 0.8)
                stroke(d, [(ux - 24, 296 + k * 9), (ux + 24, 296 + k * 9)], (184, 150, 90, 255), 0.8)
        else:
            rect(d, ux - 5, 184, ux + 5, 226, (74, 46, 30, 255))
            poly(d, [(ux - 20, 226), (ux + 22, 226), (ux + 22, 300), (ux - 20, 300)], steel)
            poly(d, [(ux - 20, 226), (ux + 22, 226), (ux + 22, 236), (ux - 20, 236)], steelLight)
            ellipse(d, ux + 12, 240, 3, 3, steelDark)
    # red wall calendar with a gold circle (keeps it simple - no Chinese glyphs in the fonts)
    rect(d, 1096, 330, 1180, 434, (206, 40, 40, 255))
    rect(d, 1100, 368, 1176, 430, (246, 242, 232, 255))
    ellipse(d, 1138, 350, 13, 13, (240, 190, 80, 255))
    for k in range(4):
        for j in range(5):
            rect(d, 1106 + j * 14, 376 + k * 13, 1114 + j * 14, 382 + k * 13, (180, 180, 176, 255))
    pic.place(wall, 0, 0)
    # grease stains and soot around the hood and the stove
    stains = []
    for i in range(26):
        sx = rnd.uniform(420, 860)
        sy = rnd.uniform(160, 450)
        rr = rnd.uniform(6, 30)
        stains.append(("ellipse", sx - rr, sy - rr * 0.7, sx + rr, sy + rr * 0.7))
    pic.overlay(softShape(width, height, stains, (110, 80, 36), 55, 7))
    drips = []
    for i in range(10):
        sx = rnd.uniform(400, 880)
        drips.append(("rect", sx, 156, sx + 2.5, 156 + rnd.uniform(20, 70)))
    pic.overlay(softShape(width, height, drips, (90, 64, 30), 60, 1.5))
    pic.overlay(softShape(width, height, [("ellipse", 470, 300, 810, 520)], (30, 20, 16), 90, 40))
    # stainless counter
    counter = Piece(width, 264)
    cd = counter.draw
    rect(cd, 0, 0, width, 264, (150, 156, 164, 255))
    rect(cd, 0, 0, width, 6, (230, 234, 238, 255))
    rect(cd, 0, 6, width, 16, (104, 110, 118, 255))
    for i in range(160):
        yy = rnd.uniform(18, 244)
        xx = rnd.uniform(-100, width)
        stroke(cd, [(xx, yy), (xx + rnd.uniform(80, 300), yy)],
               (176, 182, 190, 255) if i % 2 else (132, 138, 146, 255), rnd.uniform(0.6, 1.4))
    rect(cd, 0, 246, width, 250, (220, 226, 232, 255))
    rect(cd, 0, 250, width, 264, (96, 102, 110, 255))
    pic.place(counter, 0, 456)
    # reflection of the hood light on the steel + soot ring round the burner
    pic.overlay(softShape(width, height, [("ellipse", 420, 470, 860, 640)], (255, 236, 200), 60, 30))
    pic.overlay(softShape(width, height, [("ellipse", 410, 500, 870, 620)], (20, 16, 14), 150, 16))
    # gas burner with the blue pilot ring
    burner = Piece(460, 150)
    bd, bg = burner.draw, burner.glow
    ellipse(bd, 230, 64, 206, 54, (26, 26, 30, 255))
    ellipse(bd, 230, 62, 198, 48, (56, 56, 62, 255))
    ellipse(bd, 230, 60, 166, 38, (34, 34, 38, 255))
    ellipse(bd, 230, 66, 150, 32, (14, 14, 16, 255))
    for k in range(8):
        a = k * math.pi / 4
        stroke(bd, [(230 + math.cos(a) * 150, 64 + math.sin(a) * 34), (230 + math.cos(a) * 196, 62 + math.sin(a) * 46)],
               (70, 70, 76, 255), 5)
    ellipse(bd, 230, 72, 96, 22, (40, 40, 46, 255))
    ellipse(bd, 230, 72, 80, 17, (20, 20, 24, 255))
    for k in range(28):
        a = k * math.tau / 28
        fx = 230 + math.cos(a) * 88
        fy = 72 + math.sin(a) * 20
        poly(bg, [(fx - 3, fy), (fx + 3, fy), (fx, fy - 9)], (90, 150, 255, 235))
        poly(bg, [(fx - 1.5, fy), (fx + 1.5, fy), (fx, fy - 5)], (200, 230, 255, 255))
    burner.addLight(230, 72, 200, (80, 130, 255), 0.4)
    pic.place(burner, 410, 490)
    pic.addLight(640, 560, 460, (255, 150, 70), 0.35)
    pic.addLight(180, 115, 260, (140, 160, 255), 0.25)
    return pic.render((0.44, 0.43, 0.47), background=view, grain=2.0, seed=71, vignette=0.45,
                      saturation=0.88)


# ======================================================================
# 14. SCENE 3: MAMAK SHOP (1280x720 opaque)
# ======================================================================
def cylinderShade(d, x0, y0, x1, y1, rgb, strips=36, highlight=0.3):
    """Stainless cylinder: many thin vertical strips, bright near 'highlight' (0..1 across)."""
    w = (x1 - x0) / strips
    for i in range(strips):
        t = (i + 0.5) / strips
        light = 0.62 + 0.5 * math.exp(-((t - highlight) / 0.12) ** 2) - 0.25 * (abs(t - 0.5) * 2) ** 3
        c = (min(255, int(rgb[0] * light)), min(255, int(rgb[1] * light)), min(255, int(rgb[2] * light)), 255)
        rect(d, x0 + i * w, y0, x0 + (i + 1) * w + 0.5, y1, c)


def drawTeaUrn(p, rnd):
    d = p.draw
    steel = (196, 202, 210)
    # legs + base ring
    for lx in (104, 170, 236):
        rect(d, lx - 5, 512, lx + 5, 540, (110, 116, 124, 255))
    cylinderShade(d, 88, 300, 252, 516, steel)
    for yy in (300, 392, 506):
        rect(d, 86, yy, 254, yy + 7, (136, 142, 152, 255))
        rect(d, 86, yy, 254, yy + 2, (236, 240, 244, 255))
    # lid
    d.pieslice(box(90, 262, 250, 338), 180, 360, fill=(170, 176, 186, 255))
    d.pieslice(box(96, 268, 180, 332), 180, 270, fill=(214, 220, 228, 255))
    ellipse(d, 170, 258, 12, 7, (120, 126, 136, 255))
    ellipse(d, 166, 256, 5, 3, (230, 234, 240, 255))
    # handles
    for hx in (80, 260):
        ring(d, hx, 340, 10, 16, (120, 126, 136, 255), 3)
    # level gauge with tea inside
    rrect(d, 104, 332, 116, 492, 5, (220, 230, 236, 255))
    rrect(d, 106, 392, 114, 490, 4, (150, 86, 40, 255))
    # brass tap and drip tray
    brass, brassLight = (206, 154, 64, 255), (250, 214, 130, 255)
    rect(d, 250, 470, 286, 482, brass)
    rect(d, 250, 470, 286, 473, brassLight)
    rect(d, 278, 456, 286, 470, brass)
    rrect(d, 270, 448, 296, 458, 4, (40, 40, 44, 255))
    rect(d, 280, 482, 286, 496, brass)
    rect(d, 262, 526, 312, 534, (120, 126, 136, 255))
    rect(d, 262, 526, 312, 528, (220, 226, 232, 255))
    # label plate
    rrect(d, 142, 420, 208, 446, 4, (60, 60, 64, 255))
    label(d, 175, 433, "TEA", "future", 50, 13, (250, 214, 130, 255))


def drawTvPitch(p, rnd, x0, y0, x1, y1):
    """Football match on the TV: stands, striped pitch, lines, tiny players, score bug.
    Drawn on its own layer first, then pasted through a rectangle mask so the wide
    perspective pitch is clipped to the screen."""
    screen = Image.new("RGBA", p.glowImg.size, (0, 0, 0, 0))
    g = ImageDraw.Draw(screen)
    w, h = x1 - x0, y1 - y0
    horizon = y0 + h * 0.3
    rect(g, x0, y0, x1, horizon, (30, 36, 50, 255))
    for i in range(120):
        ellipse(g, rnd.uniform(x0, x1), rnd.uniform(y0 + 4, horizon - 2), 1.1, 1.1,
                rgba(rnd.choice([(200, 60, 60), (240, 220, 80), (220, 220, 230), (80, 90, 120)])))
    rect(g, x0, horizon - 3, x1, horizon, (150, 150, 160, 255))
    stripes = 10
    for k in range(stripes):
        t0, t1 = k / stripes, (k + 1) / stripes
        topA = x0 + w * (0.08 + 0.84 * t0)
        topB = x0 + w * (0.08 + 0.84 * t1)
        botA = x0 + w * (-0.3 + 1.6 * t0)
        botB = x0 + w * (-0.3 + 1.6 * t1)
        poly(g, [(topA, horizon), (topB, horizon), (botB, y1), (botA, y1)],
             (58, 156, 70, 255) if k % 2 == 0 else (72, 176, 82, 255))
    line = (236, 244, 236, 255)
    stroke(g, [(x0 + w * 0.08, horizon + 2), (x0 + w * 0.92, horizon + 2)], line, 1.1)
    stroke(g, [(x0 + w * 0.5, horizon + 2), (x0 + w * 0.5, y1)], line, 1.1)
    ring(g, x0 + w * 0.5, horizon + h * 0.36, w * 0.14, h * 0.13, line, 1.1)
    stroke(g, [(x0 + w * 0.84, horizon + 2), (x0 + w * 0.8, horizon + h * 0.3), (x0 + w * 1.0, horizon + h * 0.32)], line, 1.1)
    for i in range(10):
        px = x0 + w * rnd.uniform(0.15, 0.9)
        py = horizon + h * rnd.uniform(0.12, 0.6)
        rrect(g, px - 1.6, py - 4, px + 1.6, py + 1, 1, rgba((220, 40, 40) if i % 2 else (250, 230, 70)))
    ellipse(g, x0 + w * 0.56, horizon + h * 0.4, 1.3, 1.3, (255, 255, 255, 255))
    rrect(g, x0 + 6, y0 + 6, x0 + 78, y0 + 20, 3, (20, 24, 60, 235))
    art_kit.drawText(g, x0 + 42, y0 + 13, "MAS 2-1 THA", "sans_bold", 8.5, (255, 255, 255, 255), anchor="mm")
    mask = Image.new("L", p.glowImg.size, 0)
    ImageDraw.Draw(mask).rectangle(box(x0, y0, x1, y1), fill=255)
    p.glowImg.paste(screen, (0, 0), mask)


def makeMamakBg():
    width, height = 1280, 720
    rnd = random.Random(503)
    pic = Picture(width, height)
    wall = Piece(width, 540)
    d, g = wall.draw, wall.glow
    drawTiles(wall, rnd, 0, 0, width, 324, 44, (226, 224, 206), (168, 168, 150))
    # decorative border: small green/white checker tiles
    rect(d, 0, 322, width, 342, (40, 96, 68, 255))
    for k in range(int(width / 10) + 1):
        rect(d, k * 10, 326 + (k % 2) * 6, k * 10 + 10, 332 + (k % 2) * 6, (230, 236, 226, 255))
    drawTiles(wall, rnd, 0, 342, width, 540, 44, (62, 142, 104), (38, 92, 66))
    # ceiling edge + fan hint above the neon sign
    rect(d, 0, 0, width, 18, (60, 58, 60, 255))
    rect(d, 0, 16, width, 19, (120, 118, 120, 255))
    rect(d, 638, 0, 642, 14, (70, 70, 74, 255))
    # two long blades seen almost edge-on (the fan hangs just above the neon sign)
    poly(d, [(628, 15), (486, 13), (480, 18), (628, 21)], (84, 84, 92, 255))
    poly(d, [(652, 15), (794, 13), (800, 18), (652, 21)], (84, 84, 92, 255))
    poly(d, [(628, 15), (486, 13), (484, 14.5), (628, 16.5)], (130, 130, 138, 255))
    poly(d, [(652, 15), (794, 13), (796, 14.5), (652, 16.5)], (130, 130, 138, 255))
    ellipse(d, 640, 18, 16, 6, (96, 96, 104, 255))
    ellipse(d, 636, 16, 8, 2.5, (150, 150, 158, 255))
    # fluorescent tubes on the ceiling edge (cool mamak light)
    for (tx0, tx1) in ((140, 330), (950, 1140)):
        rect(d, tx0 - 6, 17, tx1 + 6, 25, (200, 204, 210, 255))
        rrect(g, tx0, 20, tx1, 26, 3, (236, 250, 255, 255))
        wall.addLight((tx0 + tx1) / 2, 60, 520, (220, 240, 255), 0.4)
    # dark backing board for the 24 HOURS neon sprite (so the neon pops)
    rrect(d, 448, 22, 832, 142, 12, (30, 44, 44, 255))
    rrect(d, 452, 26, 828, 138, 10, (22, 32, 34, 255))
    for (bx, by) in ((462, 36), (818, 36), (462, 128), (818, 128)):
        ellipse(d, bx, by, 3, 3, (120, 130, 130, 255))
    # menu lightbox (top-left)
    rrect(d, 36, 26, 434, 254, 8, (170, 176, 184, 255))
    rrect(d, 42, 32, 428, 248, 6, (20, 30, 28, 255))
    rrect(g, 46, 36, 424, 244, 5, (26, 42, 36, 200))
    label(g, 235, 62, "MENU", "future", 200, 28, (255, 206, 80, 255), shadow=(90, 60, 10, 255))
    rect(g, 70, 82, 400, 84, (255, 206, 80, 200))
    items = [("TEH TARIK", "2.50"), ("KOPI TARIK", "2.80"), ("MILO TARIK", "3.50"),
             ("ROTI CANAI", "1.80"), ("FRIED NOODLES", "6.00")]
    for i, (name, price) in enumerate(items):
        y = 104 + i * 30
        art_kit.drawText(g, 64, y, name, "sans_bold", 20, (250, 244, 226, 255), anchor="lm")
        art_kit.drawText(g, 406, y, price, "narrow", 20, (255, 214, 96, 255), anchor="rm")
        nameW = art_kit.font("sans_bold", 20 * SS).getbbox(name)[2] / SS
        x = 64 + nameW + 10
        while x < 340:
            ellipse(g, x, y + 5, 1.1, 1.1, (150, 150, 140, 255))
            x += 7
    wall.addLight(235, 140, 300, (255, 236, 190), 0.3)
    # wall-mounted TV (top-right) showing football
    rect(d, 1046, 18, 1054, 42, (60, 60, 64, 255))
    rrect(d, 896, 38, 1204, 204, 8, (24, 24, 28, 255))
    rrect(d, 900, 42, 1200, 200, 6, (40, 40, 46, 255))
    drawTvPitch(wall, rnd, 908, 50, 1192, 190)
    ellipse(d, 1190, 197, 2, 1.2, (255, 60, 60, 255))
    wall.addLight(1050, 130, 330, (140, 220, 160), 0.35)
    # tea urn on the left + stacked glasses
    drawTeaUrn(wall, rnd)
    for row in range(3):
        for k in range(3 - row):
            gx = 330 + k * 26 + row * 13
            gy = 530 - row * 30
            rect(d, gx - 10, gy - 28, gx + 10, gy, (170, 200, 206, 255))
            rect(d, gx - 8, gy - 26, gx - 5, gy - 2, (236, 246, 250, 255))
            rect(d, gx - 11, gy - 30, gx + 11, gy - 27, (210, 230, 236, 255))
    # framed picture (a teh tarik cup icon) left of centre - keeps the middle clear
    rrect(d, 380, 216, 470, 296, 3, (120, 80, 50, 255))
    rect(d, 386, 222, 464, 290, (240, 226, 196, 255))
    rect(d, 414, 244, 436, 280, (190, 130, 80, 255))
    rect(d, 414, 244, 436, 252, (250, 240, 220, 255))
    ring(d, 440, 262, 6, 8, (190, 130, 80, 255), 2)
    for k in range(3):
        stroke(d, [(418 + k * 7, 238), (421 + k * 7, 228), (418 + k * 7, 222)], (200, 200, 200, 255), 1.2)
    pic.place(wall, 0, 0)
    # steam above the urn
    steam = []
    for i in range(7):
        sx = 170 + rnd.uniform(-30, 30)
        sy = 240 - i * 22
        steam.append(("ellipse", sx - 14 - i * 2, sy - 10, sx + 14 + i * 2, sy + 10))
    pic.overlay(softShape(width, height, steam, (255, 255, 255), 40, 8))
    # stainless counter: thin top surface (glass stands here) + ribbed front panel
    counter = Piece(width, 190)
    cd = counter.draw
    rect(cd, 0, 0, width, 42, (172, 178, 186, 255))
    for i in range(70):
        yy = rnd.uniform(2, 38)
        xx = rnd.uniform(-100, width)
        stroke(cd, [(xx, yy), (xx + rnd.uniform(80, 260), yy)], (196, 202, 210, 255) if i % 2 else (150, 156, 164, 255), 0.8)
    rect(cd, 0, 0, width, 3, (120, 126, 134, 255))
    rect(cd, 0, 38, width, 44, (234, 238, 242, 255))
    rect(cd, 0, 44, width, 190, (126, 132, 142, 255))
    for x in range(0, width, 64):
        rect(cd, x, 44, x + 3, 190, (96, 100, 110, 255))
        rect(cd, x + 3, 44, x + 9, 190, (156, 162, 172, 255))
    rect(cd, 0, 44, width, 50, (88, 92, 102, 255))
    pic.place(counter, 0, 530)
    pic.overlay(softShape(width, height, [("ellipse", 440, 500, 840, 580)], (255, 240, 220), 50, 26))
    pic.addLight(640, 330, 520, (255, 220, 170), 0.22)
    return pic.render((0.42, 0.44, 0.48), grain=2.0, seed=81, vignette=0.42, saturation=0.86)


# ======================================================================
# 15. SCENE 4: BANQUET HALL (1280x720 opaque)
# ======================================================================
def drawKetupat(d, x, y, s=1.0):
    """Woven ketupat (palm-leaf rice pouch): diamond with a checked weave + ribbon tails."""
    green, yellow = (110, 170, 70, 255), (220, 200, 90, 255)
    poly(d, [(x, y - 11 * s), (x + 11 * s, y), (x, y + 11 * s), (x - 11 * s, y)], green)
    for k in range(-1, 2):
        stroke(d, [(x - 5.5 * s + k * 5.5 * s, y - 5.5 * s - k * 5.5 * s), (x + 5.5 * s + k * 5.5 * s, y + 5.5 * s - k * 5.5 * s)], yellow, 1.3 * s)
    stroke(d, [(x, y + 11 * s), (x - 4 * s, y + 24 * s)], green, 1.6 * s)
    stroke(d, [(x, y + 11 * s), (x + 5 * s, y + 22 * s)], yellow, 1.6 * s)


def drawLattice(d, x0, y0, x1, y1, rgb, cell=26):
    """Chinese window lattice: small squares joined to their neighbours by short bars,
    with a tiny diamond in every other square."""
    w = 1.6
    rect(d, x0, y0, x1, y0 + 3, rgb)
    rect(d, x0, y1 - 3, x1, y1, rgb)
    rect(d, x0, y0, x0 + 3, y1, rgb)
    rect(d, x1 - 3, y0, x1, y1, rgb)
    cols = int((x1 - x0) // cell)
    rows = int((y1 - y0) // cell)
    left = x0 + ((x1 - x0) - cols * cell) / 2
    top = y0 + ((y1 - y0) - rows * cell) / 2
    q = cell * 0.24
    for r in range(rows):
        for c in range(cols):
            cx = left + (c + 0.5) * cell
            cy = top + (r + 0.5) * cell
            stroke(d, [(cx - q, cy - q), (cx + q, cy - q), (cx + q, cy + q), (cx - q, cy + q), (cx - q, cy - q)], rgb, w)
            # bars reach the edge of the cell, so neighbouring cells join up (the frame catches the outer ones)
            barL = x0 if c == 0 else cx - cell / 2
            barR = x1 if c == cols - 1 else cx + cell / 2
            barT = y0 if r == 0 else cy - cell / 2
            barB = y1 if r == rows - 1 else cy + cell / 2
            stroke(d, [(barL, cy), (cx - q, cy)], rgb, w)
            stroke(d, [(cx + q, cy), (barR, cy)], rgb, w)
            stroke(d, [(cx, barT), (cx, cy - q)], rgb, w)
            stroke(d, [(cx, cy + q), (cx, barB)], rgb, w)
            if (r + c) % 2 == 0:
                s = q * 0.45
                poly(d, [(cx, cy - s), (cx + s, cy), (cx, cy + s), (cx - s, cy)], rgb)


def drawKolamMotif(d, x, y, size, squash, rgb):
    """One kolam flower (South Indian rice-flour art): four looped petals around a dot,
    squashed vertically because the rug lies flat on the floor (perspective)."""
    for k in range(4):
        petal = leafPoints(x, y, size, size * 0.62, 45 + k * 90)
        petal.append(petal[0])
        flat = []
        for (px, py) in petal:
            flat.append((px, y + (py - y) * squash))
        stroke(d, flat, rgb, 1.3)
    ellipse(d, x, y, 1.8, 1.8 * squash + 0.4, rgb)
    for k in range(4):
        a = math.radians(k * 90)
        ellipse(d, x + math.cos(a) * size * 0.95, y + math.sin(a) * size * 0.95 * squash, 1.4, 1.4 * squash + 0.3, rgb)


def makeBanquetBg():
    width, height = 1280, 720
    rnd = random.Random(604)
    gx, gy, gr = 640, 236, 168          # moon-gate window
    # night sky seen through the moon gate
    sky = gradientArray(width, height, [(0.0, (10, 12, 36)), (0.25, (22, 26, 70)), (0.45, (60, 50, 110)),
                                        (0.6, (110, 70, 120)), (1.0, (60, 40, 60))])
    starField(sky, rnd, 90, 330)
    moonHalo(sky, 560, 140, 30, (236, 226, 210), 0.55)
    skyImg = toImage(sky)
    moon = moonImage(30, seed=15)
    pasteOver(skyImg, moon, 560 - moon.width // 2, 140 - moon.height // 2)
    skyline = Piece(width, 200)
    drawCityRow(skyline, rnd, 440, 840, 200, (60, 140), (24, 60), (30, 28, 66), 0.2, WARM_WINDOWS, (120, 220))
    drawKLTower(skyline, 752, 200, 70, 4, (34, 32, 72, 255), (50, 48, 96, 255), (255, 214, 160), 180, s=0.6)
    colour, glow = skyline.finish()
    skyImg.alpha_composite(colour, (0, 220))
    skyImg.alpha_composite(glow, (0, 220))
    # floating sky lanterns far away in the window
    sArr = toArray(skyImg)
    lanterns = []
    for i in range(9):
        lanterns.append((rnd.uniform(520, 780), rnd.uniform(110, 300), rnd.uniform(2, 4.5), (255, 170, 80), 0.9))
    sArr[:, :, :3] += bokehEnergy(width, height, lanterns)
    background = toImage(sArr)

    pic = Picture(width, height)
    wall = Piece(width, 450)
    d = wall.draw
    red, redDark = (150, 30, 34, 255), (104, 18, 24, 255)
    gold, goldDark = (224, 172, 80, 255), (160, 112, 50, 255)
    rect(d, 0, 0, width, 450, red)
    # damask-like vertical stripes on the wall
    for x in range(0, width, 40):
        rect(d, x, 40, x + 18, 380, (142, 28, 32, 255))
    # gold lattice panels either side of the moon gate
    for (x0, x1) in ((80, 380), (900, 1200)):
        rect(d, x0 - 8, 120, x1 + 8, 372, redDark)
        drawLattice(d, x0, 128, x1, 364, goldDark, cell=30)
    # moon gate: open circle (hole) + thick gold / wood rings
    ellipse(d, gx, gy, gr + 26, gr + 26, (70, 30, 24, 255))
    ellipse(d, gx, gy, gr + 20, gr + 20, gold)
    ellipse(d, gx, gy, gr + 14, gr + 14, (96, 44, 30, 255))
    ellipse(d, gx, gy, gr + 6, gr + 6, goldDark)
    ellipse(d, gx, gy, gr, gr, (0, 0, 0, 0))
    for k in range(24):
        a = k * math.tau / 24
        ellipse(d, gx + math.cos(a) * (gr + 17), gy + math.sin(a) * (gr + 17), 2.2, 2.2, (255, 220, 140, 255))
    # ceiling beam
    rect(d, 0, 0, width, 38, (60, 30, 26, 255))
    rect(d, 0, 34, width, 40, gold)
    rect(d, 0, 40, width, 44, goldDark)
    # wainscot (dark wood) where the wall meets the floor
    rect(d, 0, 380, width, 450, (70, 36, 28, 255))
    rect(d, 0, 380, width, 384, gold)
    for x in range(0, width, 80):
        rect(d, x + 8, 392, x + 72, 440, (84, 44, 34, 255))
        rect(d, x + 8, 392, x + 72, 394, (110, 64, 46, 255))
    # side pillars
    for (x0, x1) in ((0, 46), (1234, 1280)):
        rect(d, x0, 38, x1, 450, (170, 36, 34, 255))
        rect(d, x0 + 6, 38, x0 + 14, 450, (206, 70, 60, 255))
        for yy in (60, 360):
            rect(d, x0 - 2, yy, x1 + 2, yy + 12, gold)
    # ketupat garlands on the side walls
    for (x0, x1) in ((40, 360), (920, 1240)):
        pts = sag(x0, 250, x1, 250, 34, 30)
        stroke(d, pts, (200, 170, 70, 255), 2)
        for k in range(1, 7):
            t = k / 7
            kx = x0 + (x1 - x0) * t
            ky = 250 + 34 * 4 * t * (1 - t)
            drawKetupat(d, kx, ky + 14, 1.1)
    pic.place(wall, 0, 0)
    # floor: dark wood planks
    floor = Piece(width, 270)
    fd = floor.draw
    y = 0
    h = 14
    while y < 270:
        c = jitter((78, 46, 32), rnd, 6)
        rect(fd, 0, y, width, y + h, c)
        rect(fd, 0, y + h - 1.4, width, y + h, shade(c, 0.4))
        jx = rnd.uniform(0, width)
        rect(fd, jx, y, jx + 1.4, y + h, shade(c, 0.4))
        y += h
        h *= 1.12
    # kolam rug (ellipse under the table): border ring of dots and loops
    cx, cy, rx, ry = 640, 150, 620, 128
    ellipse(fd, cx, cy, rx, ry, (170, 60, 40, 255))
    ellipse(fd, cx, cy, rx - 16, ry - 12, (236, 150, 60, 255))
    ellipse(fd, cx, cy, rx - 26, ry - 20, (120, 30, 50, 255))
    ellipse(fd, cx, cy, rx - 60, ry - 44, (40, 90, 110, 255))
    for k in range(80):
        a = k * math.tau / 80
        px = cx + math.cos(a) * (rx - 8)
        py = cy + math.sin(a) * (ry - 6)
        ellipse(fd, px, py, 2.2, 1.6, (255, 244, 224, 255))
    # ring of kolam flowers in cream "rice flour" lines, plus an inner dotted ring
    for k in range(34):
        a = (k + 0.5) * math.tau / 34
        px = cx + math.cos(a) * (rx - 43)
        py = cy + math.sin(a) * (ry - 32)
        drawKolamMotif(fd, px, py, 15, 0.42, (246, 232, 206, 255))
    for k in range(120):
        a = k * math.tau / 120
        ellipse(fd, cx + math.cos(a) * (rx - 76), cy + math.sin(a) * (ry - 56), 1.5, 1.0, (236, 150, 60, 255))
    pic.place(floor, 0, 450)
    # banner across the top
    banner = Piece(width, 70)
    bd = banner.draw
    poly(bd, [(150, 6), (1130, 6), (1150, 30), (1130, 56), (150, 56), (130, 30)], (120, 16, 22, 255))
    rect(bd, 160, 10, 1120, 52, (186, 30, 34, 255))
    rect(bd, 160, 12, 1120, 14, gold)
    rect(bd, 160, 48, 1120, 50, gold)
    for sx in (160, 1120):
        stroke(bd, [(sx, 8), (sx + (-50 if sx < 640 else 50), -30)], (60, 40, 30, 255), 1.6)
    label(bd, 640, 31, "PELITA HALL - LANTERN FESTIVAL", "future", 900, 26, (255, 214, 110, 255),
          shadow=(90, 10, 14, 255), shadowOffset=(0, 2))
    pic.place(banner, 0, 64)
    # rows of red lanterns
    lamps = Piece(width, 260)
    for (lx, ly, s) in ((70, 70, 1.0), (170, 50, 0.8), (270, 80, 1.1), (1010, 80, 1.1), (1110, 50, 0.8), (1210, 70, 1.0),
                        (120, 150, 1.3), (1160, 150, 1.3)):
        drawRedLantern(lamps, lx, ly, 34 * s, 42 * s, light=0.55)
    pic.place(lamps, 0, 0)
    # tall standing palace lanterns on the floor corners
    stand = Piece(width, 720)
    for sx in (64, 1216):
        sd = stand.draw
        rect(sd, sx - 3, 420, sx + 3, 640, (60, 34, 24, 255))
        rect(sd, sx - 18, 632, sx + 18, 644, (60, 34, 24, 255))
        poly(stand.glow, [(sx - 24, 360), (sx + 24, 360), (sx + 28, 400), (sx + 24, 440), (sx - 24, 440), (sx - 28, 400)],
             (240, 80, 50, 255))
        poly(stand.glow, [(sx - 14, 366), (sx + 14, 366), (sx + 16, 400), (sx + 14, 434), (sx - 14, 434), (sx - 16, 400)],
             (255, 190, 110, 255))
        rect(sd, sx - 28, 354, sx + 28, 362, gold)
        rect(sd, sx - 28, 438, sx + 28, 446, gold)
        stand.addLight(sx, 400, 260, (255, 120, 70), 0.6)
    pic.place(stand, 0, 0)
    pic.addLight(640, 300, 700, (255, 200, 150), 0.35, squash=0.6)
    pic.addLight(gx, gy, 300, (130, 150, 230), 0.25)
    return pic.render((0.46, 0.38, 0.43), background=background, grain=2.0, seed=91, vignette=0.45,
                      saturation=0.84)


# ======================================================================
# 16. FINALE: ROOFTOP (1280x720 opaque)
# ======================================================================
def makeRooftopBg():
    width, height = 1280, 720
    rnd = random.Random(705)
    sky = gradientArray(width, height, [(0.0, (7, 8, 26)), (0.3, (16, 18, 50)), (0.5, (32, 34, 86)),
                                        (0.62, (70, 54, 118)), (0.72, (122, 76, 124)), (1.0, (80, 50, 80))])
    noise = smoothNoise(width, height, 200, 17)
    sky[:, :, :3] *= (0.93 + 0.14 * noise)[:, :, None]
    starField(sky, rnd, 200, 380)
    moonHalo(sky, 180, 110, 36, (232, 220, 206), 0.55)
    skyImg = toImage(sky)
    moon = moonImage(36, seed=21)
    pasteOver(skyImg, moon, 180 - moon.width // 2, 110 - moon.height // 2)
    sArr = toArray(skyImg)
    # only a few thin clouds - the sky must stay open for the lanterns and fireworks
    density = cloudLayer(width, height, [(150, 16, -200, 560, 0.8), (300, 20, 640, 1500, 0.65)], seed=9)
    shadeClouds(sArr, density, (40, 40, 92), 0.5, (220, 214, 236), 180, 110, warmRgb=(170, 90, 90), warmFrom=0.3)
    skyImg = toImage(sArr)
    # KL skyline in the middle distance
    city = Picture(width, height)
    far = Piece(width, 260)
    drawCityRow(far, rnd, -20, width + 20, 260, (30, 110), (28, 80), (50, 46, 100), 0.06,
                [(210, 170, 140), (170, 180, 220)], (50, 110), cell=(6, 8), win=(2.5, 3.5),
                avoid=[(700, 900, 120)])
    city.place(far, 0, 320, haze=((110, 76, 136), 0.1, 0.5))
    marks = Piece(width, height)
    drawTwinTowers(marks, 800, 580, 440, 40, (62, 68, 122, 255), (92, 100, 158, 255), (220, 226, 250), 140)
    drawKLTower(marks, 330, 580, 290, 170, (40, 40, 88, 255), (60, 60, 112, 255), (255, 214, 160), 200, s=0.9)
    city.place(marks, 0, 0, haze=((110, 76, 136), 0.0, 0.3))
    near = Piece(width, 200)
    drawCityRow(near, rnd, -30, width + 30, 200, (40, 120), (40, 110), (28, 26, 60), 0.2,
                WARM_WINDOWS + COOL_WINDOWS[:1], (130, 230), cell=(7, 9), win=(3, 4),
                avoid=[(700, 900, 130), (290, 370, 110)])
    city.place(near, 0, 390, haze=((80, 50, 90), 0.0, 0.45))
    city.overlay(verticalFade(width, 120, (255, 150, 100), 0, 60), 0, 470)
    # soft glow around the twin towers (they are lit up for the festival)
    halo = np.zeros((height, width, 3), np.float32)
    d1 = distance(width, height, 800, 330, 90, 260)
    halo += (np.exp(-d1 * 1.6) * 0.45)[:, :, None] * np.array([200, 210, 255], np.float32)
    city.energy += halo
    skyline = city.render((1.0, 1.0, 1.0), grain=0, bloom=((2, 0.7), (8, 0.55), (26, 0.45)))
    skyImg.alpha_composite(skyline)
    # the rooftop itself
    pic = Picture(width, height)
    roof = Piece(width, 240)             # local y 0 = picture y 480
    d = roof.draw
    # terracotta floor tiles in one-point perspective: every tile edge that goes "into"
    # the picture points at the vanishing point, and rows get taller as they come closer
    vpX, vpY = 640, -170                 # vanishing point (local coords, eye level above the railing)
    rowsY = [110]
    h = 8
    while rowsY[-1] < 240:
        rowsY.append(rowsY[-1] + h)
        h *= 1.25
    rect(d, 0, 110, width, 240, (96, 56, 46, 255))              # grout colour underneath
    for r in range(len(rowsY) - 1):
        ya, yb = rowsY[r], rowsY[r + 1]
        fa = (ya - vpY) / (240 - vpY)
        fb = (yb - vpY) / (240 - vpY)
        for i in range(-14, 15):
            xb0 = vpX + (i - 0.5) * 92           # tile edges measured at the bottom of the picture
            xb1 = xb0 + 92
            c = jitter((168, 96, 72), rnd, 7)
            if (i + r) % 2 == 0:
                c = shade(c, 0.07)
            quad = [(vpX + (xb0 - vpX) * fa + 1, ya + 0.8), (vpX + (xb1 - vpX) * fa - 1, ya + 0.8),
                    (vpX + (xb1 - vpX) * fb - 1, yb - 0.8), (vpX + (xb0 - vpX) * fb + 1, yb - 0.8)]
            poly(d, quad, c)
            # soft highlight along the far edge of each tile (light from the sky)
            poly(d, [quad[0], quad[1], (quad[1][0], quad[1][1] + 1.2), (quad[0][0], quad[0][1] + 1.2)], tint(c, 0.12))
    # parapet wall
    rect(d, 0, 66, width, 112, (150, 146, 150, 255))
    rect(d, 0, 66, width, 74, (206, 204, 206, 255))
    rect(d, 0, 104, width, 112, (100, 96, 104, 255))
    for x in range(0, width, 160):
        rect(d, x, 74, x + 2, 104, (120, 116, 122, 255))
    # railing (top rail at picture y ~520)
    railC, railLight = (40, 44, 56, 255), (110, 116, 130, 255)
    rect(d, 0, 36, width, 44, railC)
    rect(d, 0, 36, width, 38, railLight)
    rect(d, 0, 52, width, 55, railC)
    for x in range(10, width, 22):
        rect(d, x, 44, x + 3, 66, railC)
    for x in range(40, width, 176):
        rect(d, x - 3, 30, x + 5, 66, railC)
        rect(d, x - 3, 30, x - 1, 66, railLight)
    pic.place(roof, 0, 480)
    # potted plants on the left, table with a white cloth on the right
    props = Piece(width, 720)
    pd, pg = props.draw, props.glow
    drawPottedPlant(props, 96, 660, "banana", rnd, s=1.9)
    drawPottedPlant(props, 236, 650, "bougainvillea", rnd, s=1.3)
    # table
    tx, ty = 1110, 560
    ellipse(pd, tx, ty + 70, 110, 14, (40, 26, 30, 255))
    poly(pd, [(tx - 104, ty), (tx + 104, ty), (tx + 118, ty + 86), (tx - 118, ty + 86)], (236, 236, 240, 255))
    for k in range(7):
        fx = tx - 100 + k * 34
        poly(pd, [(fx, ty + 4), (fx + 8, ty + 4), (fx + 12 + (fx - tx) * 0.1, ty + 86), (fx + 2 + (fx - tx) * 0.1, ty + 86)],
             (210, 212, 222, 255))
    ellipse(pd, tx, ty, 106, 18, (248, 248, 252, 255))
    # teapot, cups and a small lantern candle
    ellipse(pd, tx - 40, ty - 18, 20, 16, (230, 230, 236, 255))
    ellipse(pd, tx - 46, ty - 22, 8, 6, (255, 255, 255, 255))
    ellipse(pd, tx - 40, ty - 34, 6, 3, (200, 200, 210, 255))
    stroke(pd, [(tx - 22, ty - 18), (tx - 8, ty - 28)], (230, 230, 236, 255), 3)
    for cx in (tx + 10, tx + 34):
        rect(pd, cx - 7, ty - 14, cx + 7, ty - 2, (240, 240, 244, 255))
        ellipse(pd, cx, ty - 14, 7, 2, (170, 110, 60, 255))
    rect(pd, tx + 58, ty - 36, tx + 82, ty - 4, (40, 36, 40, 255))
    rect(pg, tx + 61, ty - 32, tx + 79, ty - 8, (255, 190, 100, 255))
    ellipse(pg, tx + 70, ty - 18, 3, 6, (255, 244, 200, 255))
    poly(pd, [(tx + 56, ty - 36), (tx + 84, ty - 36), (tx + 70, ty - 46)], (40, 36, 40, 255))
    props.addLight(tx + 70, ty - 20, 220, (255, 180, 100), 0.7)
    # chair
    rect(pd, 1222, 520, 1230, 650, (70, 50, 40, 255))
    rect(pd, 1222, 580, 1270, 590, (90, 64, 50, 255))
    rect(pd, 1262, 590, 1270, 650, (70, 50, 40, 255))
    # string-light poles and strands (kept low so the sky stays open)
    for (px, top) in ((20, 330), (1258, 330)):
        rect(pd, px - 4, top, px + 4, 700, (50, 44, 50, 255))
        rect(pd, px - 4, top, px - 2, 700, (100, 94, 100, 255))
    drawLanternString(props, rnd, 20, 334, 1258, 334, 84, 38,
                      [(255, 214, 150), (255, 190, 120), (255, 150, 170), (160, 220, 255), (255, 226, 160)],
                      size=1.6, light=0.35)
    pic.place(props, 0, 0)
    pic.addLight(640, 700, 700, (255, 180, 120), 0.3, squash=0.4)
    return pic.render((0.44, 0.44, 0.62), background=skyImg, grain=2.0, seed=101, vignette=0.35,
                      saturation=0.92)


# ======================================================================
# 17. Main
# ======================================================================
BACKGROUNDS = [
    ("title_sky", makeTitleSky, (1280, 720)),
    ("title_skyline", makeTitleSkyline, (1280, 720)),
    ("street_far", makeStreetFar, (1920, 720)),
    ("street_mid", makeStreetMid, (2560, 720)),
    ("street_main", makeStreetMain, (3200, 720)),
    ("fruit_stall_bg", makeFruitStallBg, (1280, 720)),
    ("wok_kitchen_bg", makeWokKitchenBg, (1280, 720)),
    ("mamak_bg", makeMamakBg, (1280, 720)),
    ("banquet_bg", makeBanquetBg, (1280, 720)),
    ("rooftop_bg", makeRooftopBg, (1280, 720)),
]


def main():
    wanted = sys.argv[1:]
    random.seed(2026)
    start = time.time()
    for name, maker, size in BACKGROUNDS:
        if wanted and name not in wanted:
            continue
        t0 = time.time()
        img = maker()
        if img.size != size:
            raise ValueError(f"{name} has size {img.size}, expected {size}")
        art_kit.save(img, f"backgrounds/{name}.png")
        print(f"    ({time.time() - t0:.1f} s)")
    print(f"done in {time.time() - start:.1f} s")


if __name__ == "__main__":
    main()
