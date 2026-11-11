"""
make_ui_particles.py
--------------------
Draws every user-interface picture (logo, icons, stars, cursors, the
thought bubble, the newspaper poster, the vignette and the Aura frame)
and every particle texture used by Midnight Makan.

Nothing is downloaded: every image is built from simple shapes with
Pillow, and NumPy is used for the smooth "maths" images (glows, the
vignette, smoke). Run it from the project root:

    python tools/make_ui_particles.py

Lecture ideas used in this file:
  * Supersampling anti-aliasing (Week 3-4, digital artefacts): shapes are
    drawn 4x bigger and shrunk with LANCZOS, so edges are smooth.
  * Masks and morphology (dilation / erosion) to build outlines, rim
    lights and shadows from one silhouette.
  * Image filtering: Gaussian blur for glows, drop shadows, soft particles.
  * Colour models: RGB gradients, the alpha channel for transparency.
  * Halftone printing (dot size follows brightness) for the newspaper photo.
  * Particle textures are pure white, so the game can tint them with
    sprite.color (colour multiplication).

The random generators are seeded, so every run gives the same images.
"""
import math
import os
import random
import sys

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TOOLS_DIR)
import art_kit as ak  # noqa: E402  (import after the path is set)

SS = ak.SS
OUTLINE = ak.PALETTE["outline"] + (255,)
GOLD = ak.PALETTE["gold"]
CREAM = ak.PALETTE["cream"]


# ======================================================================
# 1. Mask helpers
#    A "mask" is a greyscale (L) image at supersampled size:
#    255 = inside the shape, 0 = outside.
# ======================================================================
def blankMask(width, height):
    """Empty supersampled mask plus a drawer (width/height in final px)."""
    mask = Image.new("L", (width * SS, height * SS), 0)
    return mask, ImageDraw.Draw(mask)


def shiftMask(mask, dx, dy):
    """Move a mask by (dx, dy) FINAL pixels. Empty space becomes 0."""
    moved = Image.new("L", mask.size, 0)
    moved.paste(mask, (int(round(dx * SS)), int(round(dy * SS))))
    return moved


def growMask(mask, px):
    """Dilation (morphology): make a shape 'px' final pixels fatter.
    Done with a Gaussian blur + low threshold, which gives round corners
    and is much faster than a big MaxFilter."""
    if px <= 0:
        return mask.copy()
    blurred = mask.filter(ImageFilter.GaussianBlur(px * SS / 2.0))
    # Anything the blur reached (value > ~3) counts as inside.
    table = []
    for v in range(256):
        table.append(max(0, min(255, int((v - 3) * 255 / 6))))
    return blurred.point(table)


def shrinkMask(mask, px):
    """Erosion = dilation of the background."""
    return ImageChops.invert(growMask(ImageChops.invert(mask), px))


def smoothMask(mask, px):
    """Blur then threshold at 50%: rounds sharp corners of a polygon."""
    blurred = mask.filter(ImageFilter.GaussianBlur(px * SS))
    table = []
    for v in range(256):
        table.append(max(0, min(255, int((v - 118) * 255 / 20))))
    return blurred.point(table)


def scaleMask(mask, amount):
    """Multiply a mask by 0..1 (makes it more transparent)."""
    return mask.point(lambda v: int(v * amount))


def paintMask(mask, rgba):
    """Solid colour layer whose alpha comes from the mask."""
    r, g, b = rgba[:3]
    a = rgba[3] if len(rgba) > 3 else 255
    layer = Image.new("RGBA", mask.size, (r, g, b, 0))
    if a < 255:
        mask = scaleMask(mask, a / 255)
    layer.putalpha(mask)
    return layer


def clipLayer(layer, mask):
    """Keep only the part of an RGBA layer that is inside the mask."""
    alpha = ImageChops.multiply(layer.getchannel("A"), mask)
    out = layer.copy()
    out.putalpha(alpha)
    return out


def shadedLayer(mask, base, shadeRgb=None, lightRgb=None, outlineW=3.0,
                shadeShift=4.0, lightShift=2.0, lightInset=2.0, outline=OUTLINE,
                softLight=0.0):
    """The house style in one function:
       dark outline -> base colour -> shade on the lower-right -> rim light
       on the upper-left (the key light comes from the upper-left)."""
    layer = Image.new("RGBA", mask.size, (0, 0, 0, 0))
    if outlineW > 0:
        layer.alpha_composite(paintMask(growMask(mask, outlineW), outline))
    layer.alpha_composite(paintMask(mask, base))
    if shadeRgb is not None:
        # the part of the shape that is NOT covered when the shape moves up-left
        shade = ImageChops.subtract(mask, shiftMask(mask, -shadeShift, -shadeShift))
        layer.alpha_composite(paintMask(shade, shadeRgb))
    if lightRgb is not None:
        inner = shrinkMask(mask, lightInset)
        light = ImageChops.subtract(inner, shiftMask(inner, lightShift, lightShift))
        if softLight > 0:
            light = light.filter(ImageFilter.GaussianBlur(softLight * SS))
        layer.alpha_composite(paintMask(light, lightRgb))
    return layer


def downsample(img, width, height):
    """Supersampled canvas -> final size (anti-aliasing)."""
    return img.resize((width, height), Image.LANCZOS)


def addSoftShadow(img, offset=(1, 2), blurPx=1.4, opacity=0.45, rgb=(20, 12, 24)):
    """Soft drop shadow under a FINAL size image (same canvas size)."""
    alpha = img.getchannel("A")
    moved = Image.new("L", img.size, 0)
    moved.paste(alpha, offset)
    moved = moved.filter(ImageFilter.GaussianBlur(blurPx)).point(lambda v: int(v * opacity))
    shadow = Image.new("RGBA", img.size, rgb + (0,))
    shadow.putalpha(moved)
    shadow.alpha_composite(img)
    return shadow


def whiteParticle(alphaArray):
    """Turn a float array (0..1) into a PURE WHITE RGBA image.
    The shape lives only in the alpha channel so the game can tint it."""
    a = np.clip(alphaArray * 255.0 + 0.5, 0, 255).astype(np.uint8)
    h, w = a.shape
    rgba = np.full((h, w, 4), 255, dtype=np.uint8)
    rgba[:, :, 3] = a
    return Image.fromarray(rgba, "RGBA")


def whiteFromMask(maskImage):
    """Same as whiteParticle but from an L image."""
    img = Image.new("RGBA", maskImage.size, (255, 255, 255, 0))
    img.putalpha(maskImage)
    return img


def pixelGrid(width, height):
    """x and y coordinate arrays of pixel CENTRES (NumPy)."""
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
    return xs + 0.5, ys + 0.5


def valueNoise(width, height, cells, seed, octaves=4):
    """Fractal 'value noise': random grids blown up with bicubic
    interpolation and added together (big blobs + small details)."""
    rng = np.random.default_rng(seed)
    total = np.zeros((height, width), dtype=np.float32)
    weight = 1.0
    weightSum = 0.0
    for o in range(octaves):
        c = cells * (2 ** o)
        small = (rng.random((c + 1, c + 1)) * 255).astype(np.uint8)
        big = Image.fromarray(small, "L").resize((width, height), Image.BICUBIC)
        total += np.asarray(big, dtype=np.float32) / 255.0 * weight
        weightSum += weight
        weight *= 0.5
    return total / weightSum


def verticalGradient(size, stops, y0, y1):
    """RGB image with a multi-stop vertical gradient between rows y0..y1.
    stops = [(0.0, (r,g,b)), (0.5, (r,g,b)), (1.0, (r,g,b))]"""
    w, h = size
    ys = np.arange(h, dtype=np.float32)
    t = np.clip((ys - y0) / max(1.0, y1 - y0), 0.0, 1.0)
    positions = [s[0] for s in stops]
    columns = []
    for c in range(3):
        columns.append(np.interp(t, positions, [s[1][c] for s in stops]))
    rowColours = np.stack(columns, axis=1)
    arr = np.repeat(rowColours[:, None, :], w, axis=1).astype(np.uint8)
    return Image.fromarray(arr, "RGB").convert("RGBA")


def gradientInMask(mask, stops, y0, y1):
    """Gradient-filled layer clipped to a mask (y0/y1 in canvas px)."""
    layer = verticalGradient(mask.size, stops, y0, y1)
    layer.putalpha(mask)
    return layer


def rotatePoint(x, y, cx, cy, degrees):
    """2D rotation of a point around (cx, cy) (Week 5-6 rotation matrix).
    PIL y points down, so a positive angle turns clockwise on screen."""
    a = math.radians(degrees)
    dx, dy = x - cx, y - cy
    return (cx + dx * math.cos(a) - dy * math.sin(a),
            cy + dx * math.sin(a) + dy * math.cos(a))


def starPoints(cx, cy, outer, inner, points=5, rotation=-90.0):
    """Corner points of a star (alternating outer / inner radius)."""
    out = []
    for i in range(points * 2):
        r = outer if i % 2 == 0 else inner
        a = math.radians(rotation + i * 180.0 / points)
        out.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return out


def heartPoints(cx, cy, scale, count=120):
    """Parametric heart curve: x = 16 sin^3 t, y = 13cos t - 5cos 2t - 2cos 3t - cos 4t."""
    out = []
    for i in range(count):
        t = 2 * math.pi * i / count
        x = 16 * math.sin(t) ** 3
        y = 13 * math.cos(t) - 5 * math.cos(2 * t) - 2 * math.cos(3 * t) - math.cos(4 * t)
        out.append((cx + x * scale, cy - y * scale))
    return out


def sparkleLayer(size, x, y, radius, rgb=(255, 250, 225), alpha=255, thin=0.16):
    """A 4-point twinkle star (drawn at supersampled size)."""
    layer = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    r = radius * SS
    w = radius * thin * SS
    cx, cy = x * SS, y * SS
    shape = [(cx, cy - r), (cx + w, cy - w), (cx + r, cy), (cx + w, cy + w),
             (cx, cy + r), (cx - w, cy + w), (cx - r, cy), (cx - w, cy - w)]
    draw.polygon(shape, fill=rgb + (alpha,))
    glow = layer.filter(ImageFilter.GaussianBlur(radius * SS * 0.35))
    glow.alpha_composite(layer)
    return glow


# ======================================================================
# 2. Particles (ALL WHITE - the game tints them)
# ======================================================================
def makeSoftCircle():
    size = 64
    xs, ys = pixelGrid(size, size)
    r = np.hypot(xs - 32, ys - 32) / 31.5
    alpha = np.clip(1.0 - r * r, 0, 1) ** 2.2
    return whiteParticle(alpha)


def makeHardCircle():
    mask, draw = blankMask(32, 32)
    ak.ellipse(draw, 1.5, 1.5, 30.5, 30.5, fill=255)
    return whiteFromMask(downsample(mask, 32, 32))


def makeDroplet():
    # Teardrop pointing UP: x = sin(t) * sin(t/2)^1.3, y = -cos(t)
    w, h = 24, 40
    mask, draw = blankMask(w, h)
    points = []
    for i in range(160):
        t = 2 * math.pi * i / 160
        x = math.sin(t) * (math.sin(t / 2) ** 1.3)
        y = -math.cos(t)
        points.append((12 + x * 11.4, 20.5 + y * 17.0))
    draw.polygon(ak.pts(points), fill=255)
    mask = smoothMask(mask, 0.6)
    return whiteFromMask(downsample(mask, w, h))


def makeSpark():
    size = 48
    xs, ys = pixelGrid(size, size)
    dx = xs - 24
    dy = ys - 24
    length = 22.5

    def ray(along, across):
        t = np.clip(1.0 - np.abs(along) / length, 0, 1)
        thickness = 0.35 + 2.6 * t          # rays taper to a point
        return np.exp(-(across / thickness) ** 2) * t ** 1.4

    rays = np.maximum(ray(dx, dy), ray(dy, dx))
    r = np.hypot(dx, dy)
    core = np.exp(-(r / 4.2) ** 2)
    halo = 0.35 * np.clip(1 - r / 16.0, 0, 1) ** 2
    alpha = 1 - (1 - rays) * (1 - core) * (1 - halo)   # "screen" blend of the parts
    return whiteParticle(alpha)


def makeStar5():
    mask, draw = blankMask(48, 48)
    draw.polygon(ak.pts(starPoints(24, 25.5, 22, 10.2)), fill=255)
    mask = smoothMask(mask, 1.1)      # rounded tips
    return whiteFromMask(downsample(mask, 48, 48))


def makeSmokePuff():
    size = 128
    rng = random.Random(41)
    xs, ys = pixelGrid(size, size)
    density = np.zeros((size, size), dtype=np.float32)
    # a cluster of soft blobs of different sizes = a lumpy cloud (like cumulus)
    blobs = [(62, 68, 22)]
    for i in range(8):
        a = i * 2 * math.pi / 8 + rng.uniform(-0.3, 0.3)
        d = rng.uniform(18, 30)
        blobs.append((64 + math.cos(a) * d, 66 + math.sin(a) * d * 0.8, rng.uniform(9, 17)))
    for bx, by, br in blobs:
        density += np.exp(-((xs - bx) ** 2 + (ys - by) ** 2) / (2 * br * br))
    alpha = 1 - np.exp(-1.3 * density)
    noise = valueNoise(size, size, 3, seed=77, octaves=3)
    alpha *= 0.40 + 0.75 * noise              # big soft wisps, no fine speckles
    r = np.hypot(xs - 64, ys - 64) / 63.0
    alpha *= np.clip(1 - r ** 3, 0, 1)        # make sure the border is empty
    alpha = np.clip(alpha * 0.9, 0, 0.82)     # smoke overlaps a lot, so keep each puff see-through
    return whiteParticle(alpha)


def makeFlame():
    w, h = 64, 96
    mask, draw = blankMask(w, h)
    points = []
    for i in range(200):
        t = 2 * math.pi * i / 200
        x = math.sin(t) * (math.sin(t / 2) ** 1.5)
        y = -math.cos(t)
        points.append((32 + x * 25.0, 51 + y * 39.0))
    draw.polygon(ak.pts(points), fill=255)
    small = downsample(mask, w, h)
    soft = np.asarray(small.filter(ImageFilter.GaussianBlur(4.2)), dtype=np.float32) / 255.0
    # a brighter core low in the flame (fire is hottest near the fuel)
    xs, ys = pixelGrid(w, h)
    core = np.exp(-(((xs - 32) / 12.0) ** 2 + ((ys - 64) / 20.0) ** 2))
    alpha = np.clip(soft * 1.3, 0, 1) ** 1.1
    alpha = np.clip(alpha * (0.70 + 0.30 * core) + 0.2 * core * soft, 0, 1)
    edge = np.minimum(np.minimum(xs, w - xs), np.minimum(ys, h - ys))
    alpha *= np.clip(edge / 4.0, 0, 1)
    return whiteParticle(alpha)


def makeConfetti():
    mask, draw = blankMask(20, 12)
    ak.roundedRect(draw, 1.5, 1.8, 18.5, 10.2, 2.2, fill=255)
    return whiteFromMask(downsample(mask, 20, 12))


def makeHeartParticle():
    mask, draw = blankMask(48, 48)
    draw.polygon(ak.pts(heartPoints(24, 23, 1.34)), fill=255)
    mask = smoothMask(mask, 0.7)
    return whiteFromMask(downsample(mask, 48, 48))


def makeRing():
    size = 128
    xs, ys = pixelGrid(size, size)
    r = np.hypot(xs - 64, ys - 64)
    ringR = 56.0
    outer = np.exp(-((r - ringR) / 2.2) ** 2)      # crisp outer edge
    inner = np.exp(-((ringR - r) / 6.5) ** 2)      # soft trailing inner edge
    alpha = np.where(r > ringR, outer, inner)
    alpha *= np.clip((63.5 - r) / 3.0, 0, 1)
    return whiteParticle(alpha)


def makeLightGlow():
    size = 256
    xs, ys = pixelGrid(size, size)
    r = np.clip(np.hypot(xs - 128, ys - 128) / 127.5, 0, 1)
    smooth = np.clip(1 - r * r, 0, 1) ** 2               # wide soft pool of light
    hot = 1.0 / (1.0 + (r / 0.16) ** 2)                  # inverse-square style hot core
    hot = (hot - hot.min()) / (1 - hot.min())
    alpha = 0.62 * smooth + 0.38 * hot * smooth
    rng = np.random.default_rng(5)
    alpha += (rng.random(alpha.shape) - 0.5) / 255.0     # dithering stops colour banding
    alpha *= smooth > 0
    return whiteParticle(np.clip(alpha, 0, 1))


def makeBubble():
    size = 32
    xs, ys = pixelGrid(size * SS, size * SS)
    xs /= SS
    ys /= SS
    r = np.hypot(xs - 16, ys - 16)
    rim = np.exp(-((r - 13.0) / 1.25) ** 2) * 0.9
    body = np.where(r < 13, 0.16 + 0.12 * (r / 13.0) ** 2, 0)
    # small shiny window reflections on the upper-left (key light)
    shine = np.exp(-(((xs - 10.5) / 2.6) ** 2 + ((ys - 10.0) / 1.9) ** 2))
    shine2 = 0.8 * np.exp(-(((xs - 21.5) / 1.3) ** 2 + ((ys - 22.0) / 1.3) ** 2))
    alpha = np.clip(np.maximum(np.maximum(rim, body), np.maximum(shine, shine2)), 0, 1)
    big = whiteParticle(alpha)
    small = big.resize((size, size), Image.LANCZOS)
    # resizing can nudge RGB values; force the colour back to pure white
    return whiteFromMask(small.getchannel("A"))


# ======================================================================
# 3. Stars and icons (64x64, one shared style)
#    3 px dark warm outline, base colour, lower-right shade, upper-left
#    rim light, soft drop shadow. Content fits inside ~6..58 px.
# ======================================================================
ICON = 64


def finishIcon(canvas, inner=58, left=3, top=2):
    """Shrink the drawing to 58x58 and centre it on the 64x64 icon, so the
    outline AND the soft shadow keep a clean transparent margin."""
    small = downsample(canvas, inner, inner)
    icon = Image.new("RGBA", (ICON, ICON), (0, 0, 0, 0))
    icon.alpha_composite(small, (left, top))
    return addSoftShadow(icon, offset=(1, 2), blurPx=1.3, opacity=0.5)


def makeStarFull():
    canvas = Image.new("RGBA", (ICON * SS, ICON * SS), (0, 0, 0, 0))
    mask, draw = blankMask(ICON, ICON)
    draw.polygon(ak.pts(starPoints(32, 34, 27, 13.2)), fill=255)
    mask = smoothMask(mask, 1.3)
    canvas.alpha_composite(shadedLayer(mask, GOLD + (255,), (236, 150, 40, 255),
                                       (255, 244, 190, 255), outlineW=3,
                                       shadeShift=5, lightShift=2.5, lightInset=2.2))
    # inner bevel: a smaller star with a lighter centre gives a 'minted' look
    inner, idraw = blankMask(ICON, ICON)
    idraw.polygon(ak.pts(starPoints(32, 34.5, 13, 6.6)), fill=255)
    inner = smoothMask(inner, 0.8).filter(ImageFilter.GaussianBlur(1.5 * SS))
    canvas.alpha_composite(paintMask(inner, (255, 236, 150, 170)))
    canvas.alpha_composite(sparkleLayer(canvas.size, 21, 21, 5.5, alpha=240))
    return finishIcon(canvas)


def makeStarEmpty():
    canvas = Image.new("RGBA", (ICON * SS, ICON * SS), (0, 0, 0, 0))
    mask, draw = blankMask(ICON, ICON)
    draw.polygon(ak.pts(starPoints(32, 34, 27, 13.2)), fill=255)
    mask = smoothMask(mask, 1.3)
    # an engraved slot: dark inside, light rim on the LOWER-right (opposite
    # to a raised object) so it looks pressed into the panel
    canvas.alpha_composite(paintMask(growMask(mask, 3), (104, 110, 160, 230)))
    canvas.alpha_composite(paintMask(growMask(mask, 1.2), (30, 22, 34, 255)))
    canvas.alpha_composite(paintMask(mask, (22, 24, 50, 235)))
    innerShadow = ImageChops.subtract(mask, shiftMask(mask, 3, 3))
    canvas.alpha_composite(paintMask(innerShadow.filter(ImageFilter.GaussianBlur(SS)), (6, 6, 16, 220)))
    inner = shrinkMask(mask, 1.5)
    rim = ImageChops.subtract(inner, shiftMask(inner, -2, -2))
    canvas.alpha_composite(paintMask(rim, (78, 84, 136, 200)))
    return downsample(canvas, ICON, ICON)


def makeIconHeart():
    canvas = Image.new("RGBA", (ICON * SS, ICON * SS), (0, 0, 0, 0))
    mask, draw = blankMask(ICON, ICON)
    draw.polygon(ak.pts(heartPoints(32, 30, 1.62)), fill=255)
    mask = smoothMask(mask, 0.6)
    canvas.alpha_composite(shadedLayer(mask, (238, 72, 96, 255), (190, 36, 72, 255),
                                       (255, 150, 160, 255), shadeShift=5, lightShift=2.5))
    # glossy highlight oval on the left lobe
    shine, sdraw = blankMask(ICON, ICON)
    ak.ellipse(sdraw, 15, 15, 24, 23, fill=255)
    canvas.alpha_composite(paintMask(shine.filter(ImageFilter.GaussianBlur(0.8 * SS)), (255, 238, 240, 235)))
    ak.ellipse(ImageDraw.Draw(canvas), 26, 16.5, 28.5, 19, fill=(255, 238, 240, 200))
    return finishIcon(canvas)


def makeIconClock():
    canvas = Image.new("RGBA", (ICON * SS, ICON * SS), (0, 0, 0, 0))
    # crown button + side button (drawn first so the body overlaps them)
    cx, cy = 32, 36.5            # centre of the stopwatch face
    parts, pdraw = blankMask(ICON, ICON)
    ak.roundedRect(pdraw, 27, 6.5, 37, 12, 2, fill=255)
    ak.roundedRect(pdraw, 29.5, 10, 34.5, 16, 1, fill=255)
    side = [rotatePoint(x, y, cx, cy, 45) for x, y in [(29, 9), (35, 9), (35, 15), (29, 15)]]
    pdraw.polygon(ak.pts(side), fill=255)
    canvas.alpha_composite(shadedLayer(parts, (176, 184, 196, 255), (120, 128, 142, 255),
                                       (236, 240, 246, 255), shadeShift=2, lightShift=1.2, lightInset=1))
    body, bdraw = blankMask(ICON, ICON)
    ak.ellipse(bdraw, cx - 21.5, cy - 21.5, cx + 21.5, cy + 21.5, fill=255)
    canvas.alpha_composite(shadedLayer(body, ak.PALETTE["lantern_orange"] + (255,), (206, 92, 36, 255),
                                       (255, 196, 120, 255), shadeShift=5, lightShift=2.5))
    face, fdraw = blankMask(ICON, ICON)
    ak.ellipse(fdraw, cx - 16, cy - 16, cx + 16, cy + 16, fill=255)
    canvas.alpha_composite(shadedLayer(face, (255, 248, 234, 255), (226, 214, 196, 255), None,
                                       outlineW=2, shadeShift=-3))
    draw = ImageDraw.Draw(canvas)
    # the 'time left' wedge (like a kitchen timer)
    wedge = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    ImageDraw.Draw(wedge).pieslice(ak.box(cx - 14, cy - 14, cx + 14, cy + 14), -90, 35, fill=(255, 176, 90, 150))
    canvas.alpha_composite(wedge)
    for i in range(12):
        a = math.radians(i * 30 - 90)
        r0 = 11.5 if i % 3 == 0 else 13
        ak.line(draw, [(cx + math.cos(a) * r0, cy + math.sin(a) * r0),
                       (cx + math.cos(a) * 14.5, cy + math.sin(a) * 14.5)],
                OUTLINE, width=2 if i % 3 == 0 else 1.2)
    a = math.radians(35)
    ak.line(draw, [(cx, cy), (cx + math.cos(a) * 11, cy + math.sin(a) * 11)], (214, 48, 49, 255), width=2.4)
    ak.line(draw, [(cx, cy), (cx, cy - 10.5)], OUTLINE, width=2.6)
    ak.ellipse(draw, cx - 2.5, cy - 2.5, cx + 2.5, cy + 2.5, fill=OUTLINE)
    return finishIcon(canvas)


def makeIconLadle():
    canvas = Image.new("RGBA", (ICON * SS, ICON * SS), (0, 0, 0, 0))
    # warm aura glow behind the ladle (it is the magic Aura Ladle)
    glow = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    ak.ellipse(ImageDraw.Draw(glow), 8, 24, 42, 54, fill=(255, 214, 120, 110))
    canvas.alpha_composite(glow.filter(ImageFilter.GaussianBlur(5 * SS)))
    # two little steam curls rising from the hot soup
    steam = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    stDraw = ImageDraw.Draw(steam)
    for sx, phase in ((18, 0.0), (28, 2.2)):
        points = []
        for j in range(16):
            t = j / 15
            points.append((sx + math.sin(t * 5.5 + phase) * 2.6, 33 - t * 17))
        for j in range(15):
            t = j / 15
            ak.line(stDraw, [points[j], points[j + 1]], (255, 250, 240, int(230 * (1 - t))), width=3.2 - 2 * t)
    canvas.alpha_composite(steam.filter(ImageFilter.GaussianBlur(0.5 * SS)))
    brass = (226, 168, 72, 255)
    brassShade = (170, 108, 44, 255)
    brassLight = (255, 232, 160, 255)
    handle, hdraw = blankMask(ICON, ICON)
    ak.line(hdraw, [(34, 36), (42, 24), (50, 13)], 255, width=5.5)
    ak.ellipse(hdraw, 46, 5, 58, 17, fill=255)        # round end knob
    canvas.alpha_composite(shadedLayer(handle, brass, brassShade, brassLight,
                                       shadeShift=2, lightShift=1.4, lightInset=1.2))
    hole, holeDraw = blankMask(ICON, ICON)
    ak.ellipse(holeDraw, 50, 9, 54, 13, fill=255)
    canvas.alpha_composite(paintMask(hole, OUTLINE))
    bowl, bdraw = blankMask(ICON, ICON)
    bdraw.pieslice(ak.box(7, 22, 41, 56), 0, 180, fill=255)
    ak.ellipse(bdraw, 7, 34, 41, 44, fill=255)
    canvas.alpha_composite(shadedLayer(bowl, brass, brassShade, brassLight, shadeShift=4, lightShift=2))
    # the glowing soup surface (3/4 view opening of the bowl)
    soup, sdraw = blankMask(ICON, ICON)
    ak.ellipse(sdraw, 10.5, 35.5, 37.5, 42.5, fill=255)
    canvas.alpha_composite(gradientInMask(soup, [(0, (255, 244, 190)), (1, (255, 170, 60))],
                                          35.5 * SS, 42.5 * SS))
    ak.ellipse(ImageDraw.Draw(canvas), 10.5, 35.5, 37.5, 42.5, fill=None, outline=OUTLINE, width=1.6)
    canvas.alpha_composite(sparkleLayer(canvas.size, 9, 21, 4.5, alpha=240))
    canvas.alpha_composite(sparkleLayer(canvas.size, 38, 22, 3.5, alpha=230))
    return finishIcon(canvas)


def makeIconLock():
    canvas = Image.new("RGBA", (ICON * SS, ICON * SS), (0, 0, 0, 0))
    shackle, sdraw = blankMask(ICON, ICON)
    arc = ak.circlePoints(32, 25, 11.5, 13, count=40, start=180, end=360)
    ak.line(sdraw, [(20.5, 34)] + arc + [(43.5, 34)], 255, width=6)
    canvas.alpha_composite(shadedLayer(shackle, ak.PALETTE["steel"] + (255,), ak.PALETTE["steel_dark"] + (255,),
                                       (238, 242, 248, 255), shadeShift=2, lightShift=1.4, lightInset=1.2))
    body, bdraw = blankMask(ICON, ICON)
    ak.roundedRect(bdraw, 11, 29, 53, 58, 7, fill=255)
    canvas.alpha_composite(shadedLayer(body, GOLD + (255,), (222, 146, 44, 255),
                                       (255, 240, 180, 255), shadeShift=5, lightShift=2.5))
    draw = ImageDraw.Draw(canvas)
    # decorative band + keyhole
    ak.line(draw, [(16, 35), (48, 35)], (236, 168, 60, 255), width=1.5)
    ak.ellipse(draw, 28, 39, 36, 47, fill=OUTLINE)
    draw.polygon(ak.pts([(30, 44), (34, 44), (35.5, 53), (28.5, 53)]), fill=OUTLINE)
    return finishIcon(canvas)


def makeIconCheck():
    canvas = Image.new("RGBA", (ICON * SS, ICON * SS), (0, 0, 0, 0))
    mask, draw = blankMask(ICON, ICON)
    ak.line(draw, [(13, 34), (26, 47), (51, 17)], 255, width=10)
    canvas.alpha_composite(shadedLayer(mask, ak.PALETTE["good_green"] + (255,), (54, 164, 92, 255),
                                       (206, 255, 214, 255), shadeShift=3.5, lightShift=2, lightInset=1.8))
    return finishIcon(canvas)


def makeIconCoin():
    canvas = Image.new("RGBA", (ICON * SS, ICON * SS), (0, 0, 0, 0))
    edge, edraw = blankMask(ICON, ICON)
    ak.ellipse(edraw, 9, 10.5, 55, 56.5, fill=255)    # coin thickness (seen below the face)
    canvas.alpha_composite(shadedLayer(edge, (206, 132, 36, 255), (160, 96, 30, 255), None))
    face, fdraw = blankMask(ICON, ICON)
    ak.ellipse(fdraw, 9, 6, 55, 52, fill=255)
    canvas.alpha_composite(shadedLayer(face, GOLD + (255,), (236, 156, 46, 255),
                                       (255, 246, 196, 255), outlineW=2.5, shadeShift=4, lightShift=2.2))
    draw = ImageDraw.Draw(canvas)
    ak.ellipse(draw, 15.5, 12.5, 48.5, 45.5, fill=None, outline=(214, 140, 40, 255), width=2)
    # embossed "RM" (Malaysian ringgit): dark copy moved down-right + light copy
    # moved up-left, then the face colour on top = a stamped metal look
    ak.drawText(draw, 32.9, 30.1, "RM", "sans_bold", 17, (190, 112, 28, 255), anchor="mm")
    ak.drawText(draw, 31.5, 28.7, "RM", "sans_bold", 17, (255, 248, 210, 255), anchor="mm")
    ak.drawText(draw, 32.2, 29.4, "RM", "sans_bold", 17, (236, 164, 50, 255), anchor="mm")
    canvas.alpha_composite(sparkleLayer(canvas.size, 46, 14, 5.5, alpha=245))
    return finishIcon(canvas)


def makeIconCamera():
    canvas = Image.new("RGBA", (ICON * SS, ICON * SS), (0, 0, 0, 0))
    top, tdraw = blankMask(ICON, ICON)
    ak.roundedRect(tdraw, 19, 12, 39, 22, 3, fill=255)     # viewfinder hump
    ak.roundedRect(tdraw, 43, 13, 51, 20, 2, fill=255)     # shutter button
    canvas.alpha_composite(shadedLayer(top, (78, 84, 104, 255), (52, 56, 74, 255), (140, 148, 170, 255),
                                       shadeShift=2, lightShift=1.2, lightInset=1))
    red, rdraw = blankMask(ICON, ICON)
    ak.roundedRect(rdraw, 44, 13, 50, 17, 1.5, fill=255)
    canvas.alpha_composite(paintMask(red, (236, 84, 84, 255)))
    body, bdraw = blankMask(ICON, ICON)
    ak.roundedRect(bdraw, 6, 18, 58, 54, 8, fill=255)
    canvas.alpha_composite(shadedLayer(body, ak.PALETTE["teal"] + (255,), (26, 140, 138, 255),
                                       (160, 240, 228, 255), shadeShift=4, lightShift=2.2))
    grip, gdraw = blankMask(ICON, ICON)
    gdraw.rectangle(ak.box(6, 30, 58, 44), fill=255)
    grip = ImageChops.multiply(grip, body)
    canvas.alpha_composite(paintMask(grip, (30, 110, 112, 150)))
    draw = ImageDraw.Draw(canvas)
    ak.roundedRect(draw, 11, 22, 19, 27, 1.5, fill=(255, 250, 220, 255), outline=OUTLINE, width=1.5)  # flash
    lens, ldraw = blankMask(ICON, ICON)
    ak.ellipse(ldraw, 20, 24, 44, 48, fill=255)
    canvas.alpha_composite(shadedLayer(lens, (60, 64, 84, 255), (36, 38, 54, 255), (150, 156, 180, 255),
                                       outlineW=2.5, shadeShift=3, lightShift=1.6))
    glass, gldraw = blankMask(ICON, ICON)
    ak.ellipse(gldraw, 25, 29, 39, 43, fill=255)
    canvas.alpha_composite(gradientInMask(glass, [(0, (120, 190, 255)), (1, (40, 70, 150))], 29 * SS, 43 * SS))
    ak.ellipse(draw, 27.5, 31, 31.5, 35, fill=(240, 250, 255, 240))
    ak.ellipse(draw, 34, 38, 35.8, 39.8, fill=(220, 240, 255, 180))
    return finishIcon(canvas)


def makeIconWebcam():
    canvas = Image.new("RGBA", (ICON * SS, ICON * SS), (0, 0, 0, 0))
    base, bdraw = blankMask(ICON, ICON)
    ak.roundedRect(bdraw, 29, 38, 35, 50, 2, fill=255)       # neck
    ak.ellipse(bdraw, 15, 46, 49, 57, fill=255)             # round foot
    canvas.alpha_composite(shadedLayer(base, (78, 84, 104, 255), (52, 56, 74, 255), (140, 148, 170, 255),
                                       shadeShift=2.5, lightShift=1.4, lightInset=1.1))
    head, hdraw = blankMask(ICON, ICON)
    ak.ellipse(hdraw, 10, 7, 54, 43, fill=255)
    canvas.alpha_composite(shadedLayer(head, (240, 243, 250, 255), (192, 198, 220, 255),
                                       (255, 255, 255, 255), shadeShift=5, lightShift=2.5))
    ring, rdraw = blankMask(ICON, ICON)
    ak.ellipse(rdraw, 19, 12, 45, 38, fill=255)
    canvas.alpha_composite(shadedLayer(ring, (60, 64, 84, 255), (36, 38, 54, 255), (150, 156, 180, 255),
                                       outlineW=2.2, shadeShift=3, lightShift=1.5))
    glass, gdraw = blankMask(ICON, ICON)
    ak.ellipse(gdraw, 24.5, 17.5, 39.5, 32.5, fill=255)
    canvas.alpha_composite(gradientInMask(glass, [(0, (130, 200, 255)), (1, (46, 60, 150))], 17.5 * SS, 32.5 * SS))
    draw = ImageDraw.Draw(canvas)
    ak.ellipse(draw, 26.5, 20, 31, 24.5, fill=(240, 250, 255, 240))
    # red "recording" LED with a little glow
    led = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    ak.ellipse(ImageDraw.Draw(led), 44, 13, 50, 19, fill=(255, 80, 80, 200))
    canvas.alpha_composite(led.filter(ImageFilter.GaussianBlur(1.5 * SS)))
    ak.ellipse(draw, 45.2, 14.2, 48.8, 17.8, fill=(240, 60, 60, 255), outline=OUTLINE, width=1)
    return finishIcon(canvas)


# ======================================================================
# 4. Cursors (48x48). Hotspot = fingertip near (15, 4) / grab knuckles near (15, 9)
# ======================================================================
GLOVE = (255, 248, 236, 255)
GLOVE_SHADE = (214, 206, 214, 255)
GLOVE_LIGHT = (255, 255, 255, 255)
CUFF = ak.PALETTE["lantern_orange"] + (255,)


CREASE = (140, 112, 124, 255)


def finishCursor(canvas):
    """Shrink to 44x44 inside the 48x48 image so the shadow is not cut off.
    After this the fingertip hotspot sits at about (15, 4)."""
    small = downsample(canvas, 44, 44)
    img = Image.new("RGBA", (48, 48), (0, 0, 0, 0))
    img.alpha_composite(small, (1, 1))
    return addSoftShadow(img, offset=(2, 2), blurPx=1.2, opacity=0.55)


def cuffLayer():
    cuff, cdraw = blankMask(48, 48)
    ak.roundedRect(cdraw, 15, 35.5, 36, 42.5, 2.5, fill=255)
    return shadedLayer(cuff, CUFF, (208, 96, 36, 255), (255, 196, 130, 255),
                       outlineW=2.2, shadeShift=2, lightShift=1.2, lightInset=1)


def makeCursorHand():
    """Pointing glove. Hotspot (the fingertip) ends up at about (15, 4)."""
    canvas = Image.new("RGBA", (48 * SS, 48 * SS), (0, 0, 0, 0))
    hand, hdraw = blankMask(48, 48)
    ak.roundedRect(hdraw, 11.5, 3.5, 19.5, 26, 4, fill=255)     # index finger (pointing up)
    ak.roundedRect(hdraw, 12, 16.5, 36, 37.5, 7, fill=255)      # back of the hand
    ak.roundedRect(hdraw, 18.5, 14, 25.5, 24, 3.5, fill=255)    # curled middle finger
    ak.roundedRect(hdraw, 24.5, 15.5, 31, 25, 3.2, fill=255)    # ring finger
    ak.roundedRect(hdraw, 30, 17.5, 36, 26, 3, fill=255)        # little finger
    ak.line(hdraw, [(15, 31), (7.5, 24.5)], 255, width=7)       # thumb (a capsule pointing up-left)
    hand = smoothMask(hand, 0.9)
    canvas.alpha_composite(cuffLayer())
    canvas.alpha_composite(shadedLayer(hand, GLOVE, GLOVE_SHADE, GLOVE_LIGHT, outlineW=2.4,
                                       shadeShift=3, lightShift=1.4, lightInset=1.2))
    draw = ImageDraw.Draw(canvas)
    # knuckle lines between the curled fingers and the thumb crease
    for x0, y0, x1, y1 in [(25, 17, 25, 22), (30.5, 19, 30.5, 24)]:
        ak.line(draw, [(x0, y0), (x1, y1)], CREASE, width=1.3)
    ak.line(draw, [(11.5, 26.5), (15.5, 29.5)], CREASE, width=1.2)
    ak.line(draw, [(14, 7), (14, 12)], (255, 255, 255, 230), width=1.4)   # shine on the fingertip
    return finishCursor(canvas)


def makeCursorGrab():
    """Closed grabbing glove, same size and cuff as the pointing hand."""
    canvas = Image.new("RGBA", (48 * SS, 48 * SS), (0, 0, 0, 0))
    hand, hdraw = blankMask(48, 48)
    ak.roundedRect(hdraw, 11, 13, 37, 37.5, 8, fill=255)       # fist
    for x0 in (11, 17.5, 24, 30.5):                             # four curled fingers (knuckles on top)
        ak.roundedRect(hdraw, x0, 7.5 + (x0 - 11) * 0.12, x0 + 7, 21, 3.4, fill=255)
    ak.line(hdraw, [(16, 29), (8, 23.5)], 255, width=7)         # thumb folded over the side
    hand = smoothMask(hand, 0.9)
    canvas.alpha_composite(cuffLayer())
    canvas.alpha_composite(shadedLayer(hand, GLOVE, GLOVE_SHADE, GLOVE_LIGHT, outlineW=2.4,
                                       shadeShift=3, lightShift=1.4, lightInset=1.2))
    draw = ImageDraw.Draw(canvas)
    for x in (17.5, 24, 30.5):
        ak.line(draw, [(x, 10 + (x - 11) * 0.12), (x, 18)], CREASE, width=1.3)
    ak.line(draw, [(12, 25.5), (17, 28.5)], CREASE, width=1.2)   # thumb crease
    return finishCursor(canvas)


# ======================================================================
# 5. Thought bubble (130x120) - the dish icon is drawn on it by the game
# ======================================================================
def makeThoughtBubble():
    w, h = 130, 120
    canvas = Image.new("RGBA", (w * SS, h * SS), (0, 0, 0, 0))
    cloud, cdraw = blankMask(w, h)
    cx, cy = 68, 49
    ak.ellipse(cdraw, cx - 46, cy - 32, cx + 46, cy + 32, fill=255)
    # round bumps placed on an ellipse around the centre make it a cloud
    bumpCount = 10
    for i in range(bumpCount):
        a = math.radians(-90 + i * 360 / bumpCount + 12)
        br = 15.5 if i % 2 == 0 else 14.0
        bx = cx + math.cos(a) * 38
        by = cy + math.sin(a) * 25
        ak.ellipse(cdraw, bx - br, by - br, bx + br, by + br, fill=255)
    tails, tdraw = blankMask(w, h)
    ak.ellipse(tdraw, 17, 93, 30, 105, fill=255)
    ak.ellipse(tdraw, 7, 105.5, 14.5, 112.5, fill=255)
    base = (252, 250, 246, 255)
    shade = (214, 216, 238, 255)
    light = (255, 255, 255, 255)
    canvas.alpha_composite(shadedLayer(tails, base, shade, light, shadeShift=2.5, lightShift=1.4))
    canvas.alpha_composite(shadedLayer(cloud, base, shade, light, shadeShift=6, lightShift=2.5,
                                       lightInset=2.5, softLight=0.6))
    img = downsample(canvas, w, h)
    return addSoftShadow(img, offset=(2, 3), blurPx=2.2, opacity=0.4)


# ======================================================================
# 6. Full-screen overlays (1280x720)
# ======================================================================
def makeVignette():
    w, h = 1280, 720
    xs, ys = pixelGrid(w, h)
    nx = (xs - w / 2) / (w / 2)
    ny = (ys - h / 2) / (h / 2)
    d = np.sqrt(nx * nx + ny * ny)          # 1.0 at the middle of each edge, 1.41 in a corner
    t = np.clip((d - 0.55) / (1.45 - 0.55), 0, 1)
    alpha = t ** 1.7 * 0.94
    rng = np.random.default_rng(12)
    alpha += (rng.random(alpha.shape) - 0.5) / 255.0   # dither to avoid banding
    alpha = np.clip(alpha, 0, 1)
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, 0] = 6
    rgba[:, :, 1] = 6
    rgba[:, :, 2] = 20
    rgba[:, :, 3] = (alpha * 255 + 0.5).astype(np.uint8)
    return Image.fromarray(rgba, "RGBA")


def makeAuraFrame():
    w, h = 1280, 720
    xs, ys = pixelGrid(w, h)
    # smooth distance to the nearest screen edge (a "soft minimum" rounds the corners)
    k = 18.0
    e = -k * np.log(np.exp(-xs / k) + np.exp(-(w - xs) / k) + np.exp(-ys / k) + np.exp(-(h - ys) / k))
    e = np.maximum(e, 0)
    glow = 0.74 * np.exp(-e / 22.0) + 0.20 * np.exp(-e / 70.0)
    # flame-like flicker along the border so the glow is not a flat band
    wave = 0.5 + 0.5 * np.sin(xs / 61.0 + np.sin(ys / 47.0) * 2.0) * np.sin(ys / 53.0 + xs / 131.0)
    noise = valueNoise(w, h, 10, seed=19, octaves=3)
    glow *= 0.66 + 0.22 * wave + 0.28 * noise
    alpha = np.clip(glow, 0, 1) * 0.80
    # colour: warm white right at the edge -> gold -> amber further in
    tEdge = np.clip(e / 50.0, 0, 1)[..., None]
    edgeCol = np.array([255, 240, 190], dtype=np.float32)
    goldCol = np.array(GOLD, dtype=np.float32)
    orangeCol = np.array([255, 170, 70], dtype=np.float32)
    col = np.where(tEdge < 0.5, edgeCol + (goldCol - edgeCol) * (tEdge * 2),
                   goldCol + (orangeCol - goldCol) * ((tEdge - 0.5) * 2))
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, :3] = np.clip(col, 0, 255).astype(np.uint8)
    rgba[:, :, 3] = (alpha * 255).astype(np.uint8)
    frame = Image.fromarray(rgba, "RGBA")

    # thin golden inner border with corner ornaments (drawn supersampled at 1/2 scale to save time)
    half = 2
    ornament = Image.new("RGBA", (w * half, h * half), (0, 0, 0, 0))
    od = ImageDraw.Draw(ornament)
    inset = 14
    od.rounded_rectangle([inset * half, inset * half, (w - inset) * half, (h - inset) * half],
                         radius=22 * half, outline=(255, 226, 150, 70), width=2 * half)
    for cx, cy, sx, sy in [(0, 0, 1, 1), (w, 0, -1, 1), (0, h, 1, -1), (w, h, -1, -1)]:
        p = lambda x, y: ((cx + sx * x) * half, (cy + sy * y) * half)  # noqa: E731 (tiny local helper)
        od.line([p(26, 70), p(26, 26), p(70, 26)], fill=(255, 226, 150, 150), width=3 * half, joint="curve")
        diamond = [p(26, 14), p(38, 26), p(26, 38), p(14, 26)]
        od.polygon(diamond, fill=(255, 236, 170, 190))
    ornament = ornament.resize((w, h), Image.LANCZOS)
    frame.alpha_composite(ornament.filter(ImageFilter.GaussianBlur(3)))
    frame.alpha_composite(ornament)

    # sparkle glints scattered along the edges
    rng = random.Random(8)
    glints = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    for i in range(34):
        side = i % 4
        if side == 0:
            x, y = rng.uniform(40, w - 40), rng.uniform(8, 60)
        elif side == 1:
            x, y = rng.uniform(40, w - 40), rng.uniform(h - 60, h - 8)
        elif side == 2:
            x, y = rng.uniform(8, 60), rng.uniform(40, h - 40)
        else:
            x, y = rng.uniform(w - 60, w - 8), rng.uniform(40, h - 40)
        size = rng.uniform(5, 15)
        tile = int(size * 2 + 8)
        star = sparkleLayer((tile * SS, tile * SS), tile / 2, tile / 2, size, rgb=(255, 250, 225),
                            alpha=rng.randint(170, 255), thin=0.14)
        star = downsample(star, tile, tile)
        glints.alpha_composite(star, (int(x - tile / 2), int(y - tile / 2)))
    frame.alpha_composite(glints)
    return frame


# ======================================================================
# 7. Logo (820x300) and logo glow
# ======================================================================
LOGO_W, LOGO_H = 820, 300


def customLetterK(glyphSize, originX, baselineY, fontSize):
    """Kenney Future's 'K' looks like an 'H' with a notch, so MAKAN read as
    'MAHAN' when small. This draws my own K in the same geometric style:
    same stem width (12.5% of the font size) and cap height (62.5%)."""
    stem = 0.125 * fontSize
    width = 0.63 * fontSize
    top = baselineY - 0.625 * fontSize
    middle = (top + baselineY) / 2
    img = Image.new("L", glyphSize, 0)
    draw = ImageDraw.Draw(img)
    draw.rectangle([originX, top, originX + stem, baselineY], fill=255)
    arms = Image.new("L", glyphSize, 0)
    adraw = ImageDraw.Draw(arms)
    startX = originX + stem * 0.55
    # each arm is a thick straight line pushed past the letter box, then cut
    # flat by the box (like the square ends of the real font)
    adraw.line([(startX, middle), (originX + width + 0.25 * fontSize, top - 0.16 * fontSize)],
               fill=255, width=int(stem * 1.12))
    adraw.line([(startX, middle), (originX + width + 0.25 * fontSize, baselineY + 0.16 * fontSize)],
               fill=255, width=int(stem * 1.12))
    letterBox = Image.new("L", glyphSize, 0)
    ImageDraw.Draw(letterBox).rectangle([originX, top, originX + width, baselineY], fill=255)
    img = ImageChops.lighter(img, ImageChops.multiply(arms, letterBox))
    # soften the sharp corners a little, like the rounded corners of the font
    blurred = img.filter(ImageFilter.GaussianBlur(fontSize * 0.012))
    return blurred.point(lambda v: 255 if v > 127 else int(v * 2))


def customLetterR(glyphSize, originX, baselineY, fontSize, fnt):
    """Kenney Future's 'R' has a straight leg, so it looks like an 'A'
    ("MAAKET"). My R = the font's own 'P' + a diagonal leg."""
    stem = 0.125 * fontSize
    width = 0.625 * fontSize
    capTop = baselineY - 0.625 * fontSize
    bowlBottom = capTop + 0.6 * 0.625 * fontSize
    img = Image.new("L", glyphSize, 0)
    ImageDraw.Draw(img).text((originX, baselineY), "P", font=fnt, fill=255, anchor="ls")
    leg = Image.new("L", glyphSize, 0)
    ImageDraw.Draw(leg).line([(originX + 0.26 * fontSize, bowlBottom - stem * 0.5),
                              (originX + width + 0.12 * fontSize, baselineY + 0.14 * fontSize)],
                             fill=255, width=int(stem * 1.1))
    letterBox = Image.new("L", glyphSize, 0)
    ImageDraw.Draw(letterBox).rectangle([originX, capTop, originX + width, baselineY], fill=255)
    img = ImageChops.lighter(img, ImageChops.multiply(leg, letterBox))
    blurred = img.filter(ImageFilter.GaussianBlur(fontSize * 0.01))
    return blurred.point(lambda v: 255 if v > 127 else int(v * 2))


def wordMasks(canvasSize, text, fontName, size, centerX, baselineY, embolden, strokePx,
              tracking=0.0, arcRadius=0.0):
    """Render a word letter by letter so every letter can sit on an arc.
    Returns (fillMask, outlineMask) at supersampled size.
    arcRadius > 0 = rainbow arch, < 0 = smile curve, 0 = straight."""
    fnt = ak.font(fontName, size * SS)
    fillWidth = int(embolden * SS)
    strokeWidth = int((embolden + strokePx) * SS)
    fillMask = Image.new("L", canvasSize, 0)
    outlineMask = Image.new("L", canvasSize, 0)
    advances = [fnt.getlength(ch) for ch in text]
    total = sum(advances) + tracking * SS * (len(text) - 1)
    x = centerX * SS - total / 2
    ascent, descent = fnt.getmetrics()
    pad = strokeWidth + 8
    for i, ch in enumerate(text):
        adv = advances[i]
        if ch == " ":
            x += adv + tracking * SS
            continue
        u = x + adv / 2 - centerX * SS          # horizontal distance from the word centre
        dy = 0.0
        angle = 0.0
        if arcRadius != 0:
            R = abs(arcRadius) * SS
            sag = R - math.sqrt(max(0.0, R * R - u * u))
            angle = math.degrees(math.asin(max(-1.0, min(1.0, u / R))))
            if arcRadius > 0:
                dy = sag
                angle = -angle
            else:
                dy = -sag
        gw = int(adv + pad * 2)
        gh = int(ascent + descent + pad * 2)
        if ch in ("K", "R"):
            if ch == "K":
                plain = customLetterK((gw, gh), pad, pad + ascent, size * SS)
            else:
                plain = customLetterR((gw, gh), pad, pad + ascent, size * SS, fnt)
            glyphFill = growMask(plain, embolden)
            glyphLine = growMask(plain, embolden + strokePx)
        else:
            glyphFill = Image.new("L", (gw, gh), 0)
            glyphLine = Image.new("L", (gw, gh), 0)
            ImageDraw.Draw(glyphFill).text((pad, pad + ascent), ch, font=fnt, fill=255, anchor="ls",
                                           stroke_width=fillWidth, stroke_fill=255)
            ImageDraw.Draw(glyphLine).text((pad, pad + ascent), ch, font=fnt, fill=255, anchor="ls",
                                           stroke_width=strokeWidth, stroke_fill=255)
        if angle != 0:
            glyphFill = glyphFill.rotate(angle, resample=Image.BICUBIC, expand=True)
            glyphLine = glyphLine.rotate(angle, resample=Image.BICUBIC, expand=True)
        centreX = x + adv / 2
        centreY = baselineY * SS + (descent - ascent) / 2 + dy
        px = int(centreX - glyphFill.width / 2)
        py = int(centreY - glyphFill.height / 2)
        fillMask.paste(255, (px, py), glyphFill)
        outlineMask.paste(255, (px, py), glyphLine)
        x += adv + tracking * SS
    return fillMask, outlineMask


def logoWordLayer(canvasSize, fillMask, outlineMask, stops, sideTop, sideBottom, depth, top, bottom):
    """Chunky 3D title word: shadow, extruded side, outline, gradient face,
    top highlight band and bottom shade band."""
    layer = Image.new("RGBA", canvasSize, (0, 0, 0, 0))
    # extrusion: stack copies of the outlined word moved downwards
    side = Image.new("L", canvasSize, 0)
    for k in range(1, int(depth * SS) + 1, 2):
        side = ImageChops.lighter(side, shiftMask(outlineMask, 0, k / SS))
    silhouette = ImageChops.lighter(side, outlineMask)
    shadow = paintMask(shiftMask(silhouette, 3, 7), (8, 6, 20, 150))
    layer.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(5 * SS)))
    layer.alpha_composite(paintMask(growMask(silhouette, 2.0), OUTLINE))
    layer.alpha_composite(gradientInMask(side, [(0, sideTop), (1, sideBottom)], top * SS, (bottom + depth) * SS))
    layer.alpha_composite(paintMask(outlineMask, OUTLINE))
    layer.alpha_composite(gradientInMask(fillMask, stops, top * SS, bottom * SS))
    # bottom shade band inside each letter
    lowBand = ImageChops.subtract(fillMask, shiftMask(fillMask, 0, -4))
    darker = ak.shade(stops[-1][1], 0.25)
    layer.alpha_composite(paintMask(lowBand, darker[:3] + (200,)))
    # top highlight band (inner highlight)
    topBand = ImageChops.subtract(fillMask, shiftMask(fillMask, 0, 3.2))
    topBand = shrinkMask(topBand, 0.4)
    layer.alpha_composite(paintMask(topBand, (255, 255, 246, 235)))
    return layer


def lanternLayer(w, h):
    """Small glowing red Chinese lantern (supersampled, w x h final px)."""
    canvas = Image.new("RGBA", (w * SS, h * SS), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    # tassel
    tx = w / 2
    for off in (-1.6, 0, 1.6):
        ak.line(draw, [(tx, h * 0.80), (tx + off, h * 0.97)], (236, 170, 60, 255), width=1.4)
    body, bdraw = blankMask(w, h)
    ak.ellipse(bdraw, w * 0.10, h * 0.20, w * 0.90, h * 0.84, fill=255)
    canvas.alpha_composite(shadedLayer(body, ak.PALETTE["lantern_red"] + (255,), (160, 28, 40, 255),
                                       (255, 132, 96, 255), outlineW=max(1.6, w / 20),
                                       shadeShift=w * 0.12, lightShift=w * 0.06, lightInset=w * 0.04))
    # inner candle glow
    glow, gdraw = blankMask(w, h)
    ak.ellipse(gdraw, w * 0.28, h * 0.34, w * 0.72, h * 0.70, fill=255)
    glow = ImageChops.multiply(glow.filter(ImageFilter.GaussianBlur(w * 0.12 * SS)), body)
    canvas.alpha_composite(paintMask(glow, (255, 206, 110, 190)))
    # ribs (curved vertical lines)
    ribs = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    rdraw = ImageDraw.Draw(ribs)
    for frac in (0.26, 0.52):
        ak.ellipse(rdraw, w * (0.5 - frac * 0.75), h * 0.20, w * (0.5 + frac * 0.75), h * 0.84,
                   fill=None, outline=(150, 24, 36, 200), width=max(1.0, w / 30))
    canvas.alpha_composite(clipLayer(ribs, shrinkMask(body, 0.5)))
    caps, cdraw = blankMask(w, h)
    ak.roundedRect(cdraw, w * 0.28, h * 0.10, w * 0.72, h * 0.25, 1.5, fill=255)
    ak.roundedRect(cdraw, w * 0.32, h * 0.78, w * 0.68, h * 0.88, 1.5, fill=255)
    canvas.alpha_composite(shadedLayer(caps, GOLD + (255,), (214, 140, 40, 255), (255, 240, 180, 255),
                                       outlineW=max(1.4, w / 24), shadeShift=1.2, lightShift=0.8, lightInset=0.5))
    return canvas


def ladleLayer(w, h):
    """The brass Aura Ladle with glowing soup and rising steam (logo motif)."""
    canvas = Image.new("RGBA", (w * SS, h * SS), (0, 0, 0, 0))
    brass = (228, 170, 74, 255)
    brassShade = (168, 106, 42, 255)
    brassLight = (255, 234, 166, 255)
    # golden aura behind the ladle bowl
    aura = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    ak.ellipse(ImageDraw.Draw(aura), 4, h * 0.46, w * 0.72, h * 0.98, fill=(255, 200, 87, 170))
    canvas.alpha_composite(aura.filter(ImageFilter.GaussianBlur(9 * SS)))
    # handle: gentle S-curve from the bowl up to the right, with a curled end
    handle, hdraw = blankMask(w, h)
    path = ak.bezier((w * 0.52, h * 0.70), (w * 0.66, h * 0.56), (w * 0.72, h * 0.34), (w * 0.84, h * 0.20), steps=30)
    ak.line(hdraw, path, 255, width=7)
    ak.ellipse(hdraw, w * 0.77, h * 0.07, w * 0.97, h * 0.23, fill=255)
    canvas.alpha_composite(shadedLayer(handle, brass, brassShade, brassLight, outlineW=3,
                                       shadeShift=2.2, lightShift=1.5, lightInset=1.2))
    draw = ImageDraw.Draw(canvas)
    ak.ellipse(draw, w * 0.835, h * 0.115, w * 0.905, h * 0.185, fill=OUTLINE)
    ak.line(draw, [(w * 0.63, h * 0.53), (w * 0.66, h * 0.47)], (255, 240, 190, 255), width=1.6)
    # bowl (half sphere) + opening seen in 3/4 view
    bowl, bdraw = blankMask(w, h)
    bdraw.pieslice(ak.box(w * 0.06, h * 0.52, w * 0.62, h * 0.97), 0, 180, fill=255)
    ak.ellipse(bdraw, w * 0.06, h * 0.68, w * 0.62, h * 0.82, fill=255)
    canvas.alpha_composite(shadedLayer(bowl, brass, brassShade, brassLight, outlineW=3,
                                       shadeShift=5, lightShift=2.4))
    soup, sdraw = blankMask(w, h)
    ak.ellipse(sdraw, w * 0.11, h * 0.695, w * 0.57, h * 0.805, fill=255)
    canvas.alpha_composite(gradientInMask(soup, [(0, (255, 248, 206)), (1, (255, 164, 56))],
                                          h * 0.695 * SS, h * 0.805 * SS))
    ak.ellipse(draw, w * 0.11, h * 0.695, w * 0.57, h * 0.805, fill=None, outline=OUTLINE, width=2)
    # steam: three wavy tapered wisps rising from the soup
    steam = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    sdraw2 = ImageDraw.Draw(steam)
    for i, x0 in enumerate((0.20, 0.34, 0.48)):
        points = []
        for j in range(30):
            t = j / 29
            y = h * (0.66 - t * (0.50 + 0.06 * (i % 2)))
            x = w * x0 + math.sin(t * 5.2 + i * 1.7) * w * 0.05 * (0.4 + t)
            points.append((x, y))
        for j in range(len(points) - 1):
            t = j / (len(points) - 1)
            width = 7.5 * (1 - t) + 1.5
            alpha = int(255 * (1 - t) ** 0.7)
            ak.line(sdraw2, [points[j], points[j + 1]], (255, 250, 240, alpha), width=width)
    steamGlow = steam.filter(ImageFilter.GaussianBlur(3 * SS))
    canvas.alpha_composite(steamGlow)
    canvas.alpha_composite(steam.filter(ImageFilter.GaussianBlur(0.6 * SS)))
    return canvas


def ribbonLayer(canvasSize, cx, cy, halfWidth, height, curve, text):
    """Red banner that bends in a smile curve, with folded tails and the subtitle."""
    layer = Image.new("RGBA", canvasSize, (0, 0, 0, 0))

    def centreY(x):
        u = (x - cx) / halfWidth
        return cy + curve * (1 - u * u)          # middle lower than the ends (smile)

    # tails (behind the band)
    for sgn in (-1, 1):
        xIn = cx + sgn * (halfWidth - 14)
        xOut = cx + sgn * (halfWidth + 46)
        yTop = centreY(cx + sgn * halfWidth) + 9
        tail = [(xIn, yTop), (xOut, yTop - 2), (xOut - sgn * 14, yTop + height / 2 + 1),
                (xOut, yTop + height + 4), (xIn, yTop + height + 2)]
        tMask, tDraw = blankMask(canvasSize[0] // SS, canvasSize[1] // SS)
        tDraw.polygon(ak.pts(tail), fill=255)
        layer.alpha_composite(shadedLayer(tMask, (170, 32, 40, 255), (126, 20, 34, 255),
                                          (214, 70, 64, 255), outlineW=3, shadeShift=3, lightShift=1.5))
        fold = [(cx + sgn * halfWidth, centreY(cx + sgn * halfWidth) + height / 2),
                (cx + sgn * halfWidth, centreY(cx + sgn * halfWidth) + height / 2 + 11),
                (xIn, yTop + height + 2)]
        fMask, fDraw = blankMask(canvasSize[0] // SS, canvasSize[1] // SS)
        fDraw.polygon(ak.pts(fold), fill=255)
        layer.alpha_composite(shadedLayer(fMask, (92, 14, 28, 255), None, None, outlineW=2))
    # main band
    band, bdraw = blankMask(canvasSize[0] // SS, canvasSize[1] // SS)
    topEdge = []
    bottomEdge = []
    for i in range(41):
        x = cx - halfWidth + 2 * halfWidth * i / 40
        topEdge.append((x, centreY(x) - height / 2))
        bottomEdge.append((x, centreY(x) + height / 2))
    bdraw.polygon(ak.pts(topEdge + bottomEdge[::-1]), fill=255)
    layer.alpha_composite(shadedLayer(band, (222, 54, 52, 255), (176, 34, 42, 255), (255, 128, 110, 255),
                                      outlineW=3, shadeShift=4, lightShift=1.8))
    draw = ImageDraw.Draw(layer)
    # gold trim lines
    inTop = [(x, y + 4.2) for x, y in topEdge[1:-1]]
    inBottom = [(x, y - 4.2) for x, y in bottomEdge[1:-1]]
    ak.line(draw, inTop, GOLD + (235,), width=1.6)
    ak.line(draw, inBottom, GOLD + (235,), width=1.6)
    # subtitle letters follow the same smile curve
    # (a circle with radius halfWidth^2 / (2 * curve) bends like the parabola above)
    capHeight = 17 * 0.63
    fill, outline = wordMasks(canvasSize, text, "future", 17, cx, cy + curve + capHeight / 2, 0.25, 2.0,
                              tracking=1.6, arcRadius=-(halfWidth * halfWidth) / (2 * curve))
    layer.alpha_composite(paintMask(outline, (92, 14, 28, 255)))
    layer.alpha_composite(paintMask(fill, CREAM + (255,)))
    # little gold diamonds at both ends of the text
    for sgn in (-1, 1):
        dx = cx + sgn * (halfWidth - 20)
        dy = centreY(dx)
        diamond = [(dx, dy - 5), (dx + 5, dy), (dx, dy + 5), (dx - 5, dy)]
        ak.polygon(draw, diamond, GOLD + (255,), outline=(92, 14, 28, 255), width=1.2)
    return layer


def catenary(x0, y0, x1, y1, sag, count=60):
    """Hanging string between two points (a parabola is close to a real catenary)."""
    out = []
    for i in range(count + 1):
        t = i / count
        out.append((x0 + (x1 - x0) * t, y0 + (y1 - y0) * t + sag * 4 * t * (1 - t)))
    return out


def makeLogo():
    size = (LOGO_W * SS, LOGO_H * SS)
    canvas = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)

    # --- lantern string across the top ---------------------------------
    # drawn on its own layer so it can fade out towards both ends
    # (a hard cut at the image border looks wrong when the logo floats)
    string = catenary(0, 8, LOGO_W, 8, 24)
    stringLayer = Image.new("RGBA", size, (0, 0, 0, 0))
    sdraw = ImageDraw.Draw(stringLayer)
    ak.line(sdraw, string, (70, 46, 56, 235), width=2)
    bulbs = Image.new("RGBA", size, (0, 0, 0, 0))
    bdraw = ImageDraw.Draw(bulbs)
    for i in range(2, 60, 4):
        x, y = string[i]
        if 170 < x < 650:
            continue
        ak.ellipse(bdraw, x - 3.2, y - 1, x + 3.2, y + 5.4, fill=(255, 214, 120, 255))
    stringLayer.alpha_composite(bulbs.filter(ImageFilter.GaussianBlur(4 * SS)))
    stringLayer.alpha_composite(bulbs)
    fade = np.clip(np.minimum(np.arange(size[0]), size[0] - np.arange(size[0])) / (70.0 * SS), 0, 1)
    fadeRow = Image.fromarray((np.tile(fade, (size[1], 1)) * 255).astype(np.uint8), "L")
    canvas.alpha_composite(clipLayer(stringLayer, fadeRow))

    def hangLantern(x, w, h, top):
        yString = string[int(round(x / LOGO_W * 60))][1]
        ak.line(draw, [(x, yString), (x, top + h * 0.12)], (70, 46, 56, 255), width=1.6)
        glow = Image.new("RGBA", size, (0, 0, 0, 0))
        ak.ellipse(ImageDraw.Draw(glow), x - w * 0.9, top - h * 0.1, x + w * 0.9, top + h * 1.05,
                   fill=(255, 150, 60, 120))
        canvas.alpha_composite(glow.filter(ImageFilter.GaussianBlur(w * 0.45 * SS)))
        lantern = lanternLayer(w, h)
        canvas.alpha_composite(lantern, (int((x - w / 2) * SS), int(top * SS)))
        return lantern

    bigLantern = hangLantern(62, 46, 62, 104)
    hangLantern(130, 24, 33, 28)
    hangLantern(698, 24, 33, 28)
    hangLantern(768, 20, 28, 20)

    # --- the Aura Ladle next to MAKAN (drawn before the words so they overlap it)
    ladle = ladleLayer(120, 150)
    canvas.alpha_composite(ladle, (int(676 * SS), int(56 * SS)))
    # the glow texture is built only from the big shapes (not the thin string)
    glowSource = Image.new("L", size, 0)
    glowSource.paste(bigLantern.getchannel("A"), (int((62 - 23) * SS), int(104 * SS)))
    glowSource = ImageChops.lighter(glowSource, shiftAlpha(ladle, size, 676, 56))

    # --- title words -----------------------------------------------------
    makanFill, makanLine = wordMasks(size, "MAKAN", "future", 118, 400, 184, embolden=3.2,
                                     strokePx=6.5, tracking=8)
    makanStops = [(0.0, (255, 250, 214)), (0.30, (255, 222, 104)), (0.52, (255, 190, 64)),
                  (0.53, (255, 176, 58)), (1.0, (240, 106, 40))]
    canvas.alpha_composite(logoWordLayer(size, makanFill, makanLine, makanStops,
                                         (168, 52, 42), (92, 22, 32), 8, 110, 186))
    midFill, midLine = wordMasks(size, "MIDNIGHT", "future", 72, 400, 84, embolden=2.4,
                                 strokePx=5.5, tracking=7, arcRadius=1500)
    midStops = [(0.0, (255, 255, 250)), (0.45, (255, 240, 206)), (0.46, (255, 230, 180)),
                (1.0, (255, 196, 96))]
    canvas.alpha_composite(logoWordLayer(size, midFill, midLine, midStops,
                                         (150, 72, 44), (90, 34, 30), 6, 36, 86))

    # --- sparkles on the letters ------------------------------------------
    for x, y, r in [(190, 38, 9), (612, 114, 11), (352, 104, 6), (538, 40, 7), (140, 200, 6)]:
        canvas.alpha_composite(sparkleLayer(size, x, y, r, alpha=250))

    # --- subtitle ribbon ----------------------------------------------------
    ribbon = ribbonLayer(size, 400, 232, 270, 32, 8, "A NIGHT-MARKET DINING ADVENTURE")
    canvas.alpha_composite(ribbon)

    for part in (makanLine, midLine, ribbon.getchannel("A")):
        glowSource = ImageChops.lighter(glowSource, part)
    return downsample(canvas, LOGO_W, LOGO_H), glowSource.resize((LOGO_W, LOGO_H), Image.LANCZOS)


def shiftAlpha(layer, canvasSize, left, top):
    """Alpha of a small layer placed on a big empty mask at (left, top) final px."""
    out = Image.new("L", canvasSize, 0)
    out.paste(layer.getchannel("A"), (int(left * SS), int(top * SS)))
    return out


def makeLogoGlow(glowSource):
    """Blurred warm copy of the logo's main shapes, drawn additively by the game."""
    wide = glowSource.filter(ImageFilter.GaussianBlur(16)).point(lambda v: min(255, int(v * 1.3)))
    tight = glowSource.filter(ImageFilter.GaussianBlur(5)).point(lambda v: int(v * 0.65))
    combined = ImageChops.lighter(wide, tight)
    arr = np.asarray(combined, dtype=np.float32) / 255.0
    xs, ys = pixelGrid(LOGO_W, LOGO_H)
    edge = np.minimum(np.minimum(xs, LOGO_W - xs), np.minimum(ys, LOGO_H - ys))
    arr *= np.clip(edge / 40.0, 0, 1) ** 1.5      # fade out well before the image border
    rgba = np.zeros((LOGO_H, LOGO_W, 4), dtype=np.uint8)
    # warm colour that is more golden in the middle rows, more orange at the bottom
    rows = np.linspace(0, 1, LOGO_H, dtype=np.float32)[:, None]
    rgba[:, :, 0] = 255
    rgba[:, :, 1] = (200 - 50 * rows).astype(np.uint8)
    rgba[:, :, 2] = (100 - 40 * rows).astype(np.uint8)
    rgba[:, :, 3] = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(rgba, "RGBA")


# ======================================================================
# 8. Newspaper poster (520x360) with a halftone photo of the critic
# ======================================================================
INK = (36, 32, 34, 255)


def criticPortraitSource(w, h):
    """Greyscale 'photo' of Dato' Leon Tan (0 = black ink, 255 = paper).
    Only the TONES are drawn here; the halftone step turns them into dots."""
    img = Image.new("L", (w * SS, h * SS), 0)
    grad = verticalGradient((w * SS, h * SS), [(0, (150, 150, 150)), (1, (112, 112, 112))], 0, h * SS)
    img.paste(grad.convert("L"))
    # light studio halo behind the head separates him from the backdrop
    halo = Image.new("L", img.size, 0)
    ak.ellipse(ImageDraw.Draw(halo), w * 0.5 - 60, 2, w * 0.5 + 60, 118, fill=255)
    halo = halo.filter(ImageFilter.GaussianBlur(18 * SS))
    img = Image.composite(Image.new("L", img.size, 205), img, halo)
    draw = ImageDraw.Draw(img)
    cx = w * 0.5
    # cream-white suit (light), lapels, shirt, shaded right side
    ak.ellipse(draw, cx - 80, 102, cx + 80, 210, fill=228)
    ak.ellipse(draw, cx + 42, 108, cx + 80, 210, fill=196)
    draw.polygon(ak.pts([(cx - 17, 104), (cx, 146), (cx + 17, 104)]), fill=252)
    draw.polygon(ak.pts([(cx - 30, 104), (cx - 6, 146), (cx - 20, 146), (cx - 46, 112)]), fill=198)
    draw.polygon(ak.pts([(cx + 30, 104), (cx + 6, 146), (cx + 20, 146), (cx + 46, 112)]), fill=180)
    # neck, ears, head (light skin with a shadow side)
    ak.roundedRect(draw, cx - 11, 82, cx + 11, 108, 5, fill=170)
    ak.ellipse(draw, cx - 33, 50, cx - 23, 68, fill=196)
    ak.ellipse(draw, cx + 23, 50, cx + 33, 68, fill=168)
    ak.ellipse(draw, cx - 27, 20, cx + 27, 94, fill=212)
    ak.ellipse(draw, cx + 6, 24, cx + 27, 92, fill=182)
    # slicked-back grey hair with shiny comb lines
    draw.chord(ak.box(cx - 29, 12, cx + 29, 66), 180, 360, fill=138)
    for i in range(5):
        x0 = cx - 20 + i * 9
        ak.line(draw, [(x0, 34), (x0 + 10, 16)], 215, width=1.6)
    # round glasses (bright glare on the lenses), eyes, eyebrows (one raised)
    for ex in (-12, 12):
        ak.ellipse(draw, cx + ex - 9, 49, cx + ex + 9, 67, fill=240)
    for ex in (-12, 12):
        ak.ellipse(draw, cx + ex - 2.6, 56, cx + ex + 2.6, 62, fill=30)
    ak.line(draw, [(cx - 20, 44), (cx - 5, 42)], 50, width=2.6)
    ak.line(draw, [(cx + 5, 38.5), (cx + 20, 34.5)], 50, width=2.6)
    ak.line(draw, [(cx - 11, 77), (cx - 2, 74.5), (cx + 2, 74.5), (cx + 11, 77)], 60, width=3)
    ak.line(draw, [(cx - 6, 84), (cx + 4, 84), (cx + 9, 81)], 60, width=1.8)
    # bow tie
    draw.polygon(ak.pts([(cx - 17, 104), (cx, 111), (cx - 17, 119)]), fill=78)
    draw.polygon(ak.pts([(cx + 17, 104), (cx, 111), (cx + 17, 119)]), fill=78)
    ak.ellipse(draw, cx - 4, 106, cx + 4, 116, fill=50)
    # the famous golden spoon, held up next to his face
    ak.line(draw, [(cx + 50, 146), (cx + 45, 76)], 236, width=4.5)
    ak.ellipse(draw, cx + 36, 44, cx + 54, 78, fill=246)
    ak.ellipse(draw, cx + 39, 116, cx + 61, 146, fill=196)
    return downsample(img, w, h)


def criticPortraitLines(draw, left, top, w):
    """Loose 'sketchy' ink contours drawn over the halftone photo.
    Each line is drawn twice with a tiny offset, like a quick pen sketch."""
    cx = left + w * 0.5
    t = top
    lineInk = (36, 32, 34, 210)
    for jitter in (0.0, 0.45):
        j = jitter
        ak.ellipse(draw, cx - 27 + j, t + 20, cx + 27 + j, t + 94 - j, fill=None, outline=lineInk, width=1.1)
        for ex in (-12, 12):
            ak.ellipse(draw, cx + ex - 9 + j, t + 49, cx + ex + 9, t + 67 + j, fill=None, outline=(30, 26, 28, 255), width=1.8)
        ak.line(draw, [(cx - 3, t + 57 + j), (cx + 3, t + 57 + j)], lineInk, width=1.4)
        ak.line(draw, [(cx - 30 + j, t + 104), (cx - 6, t + 146)], lineInk, width=1.0)
        ak.line(draw, [(cx + 30 + j, t + 104), (cx + 6, t + 146)], lineInk, width=1.0)
        ak.line(draw, [(cx - 30, t + 104 + j), (cx - 72, t + 128), (cx - 80, t + 146)], lineInk, width=1.1)
        ak.line(draw, [(cx + 30, t + 104 + j), (cx + 72, t + 128), (cx + 80, t + 146)], lineInk, width=1.1)
        ak.ellipse(draw, cx + 36 + j, t + 44, cx + 54, t + 78 + j, fill=None, outline=lineInk, width=1.2)
        ak.line(draw, [(cx + 45 + j, t + 78), (cx + 50, t + 118)], lineInk, width=1.0)
        ak.polygon(draw, [(cx - 17, t + 104 + j), (cx, t + 111), (cx - 17, t + 119)], None,
                   outline=lineInk, width=1.0)
        ak.polygon(draw, [(cx + 17, t + 104 + j), (cx, t + 111), (cx + 17, t + 119)], None,
                   outline=lineInk, width=1.0)
    # a little glare arc on each lens
    for ex in (-12, 12):
        draw.arc(ak.box(cx + ex - 6, t + 52, cx + ex + 6, t + 64), 200, 250,
                 fill=(255, 255, 255, 230), width=int(1.2 * SS))


def halftoneLayer(source, left, top, canvasSize, cell=3.1):
    """Newspaper halftone: one ink dot per grid cell, the dot AREA grows
    with how dark the photo is (digital image -> print simulation).
    left/top/cell are in final pixels; canvasSize is supersampled."""
    layer = Image.new("RGBA", canvasSize, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    w, h = source.size
    px = source.load()
    y = cell / 2
    row = 0
    while y < h:
        x = cell / 2 + (cell / 2 if row % 2 else 0)     # staggered rows like real print screens
        while x < w:
            darkness = 1.0 - px[min(w - 1, int(x)), min(h - 1, int(y))] / 255.0
            r = cell * 0.62 * math.sqrt(darkness)
            if r > 0.18:
                ak.ellipse(draw, left + x - r, top + y - r, left + x + r, top + y + r, fill=INK)
            x += cell
        y += cell * 0.87
        row += 1
    return layer


def makePoster():
    w, h = 520, 360
    rng = random.Random(360)
    # --- paper shape with a slightly rough edge ------------------------
    paper, pdraw = blankMask(w, h)
    edge = []
    for i in range(60):
        edge.append((6 + i * (w - 12) / 60, 7 + rng.uniform(-0.8, 0.8)))
    for i in range(40):
        edge.append((w - 6 + rng.uniform(-0.8, 0.8), 7 + i * (h - 14) / 40))
    for i in range(60):
        edge.append((w - 6 - i * (w - 12) / 60, h - 7 + rng.uniform(-0.8, 0.8)))
    for i in range(40):
        edge.append((6 + rng.uniform(-0.8, 0.8), h - 7 - i * (h - 14) / 40))
    pdraw.polygon(ak.pts(edge), fill=255)
    paperMask = downsample(paper, w, h)

    # newsprint colour + yellowed edges + grain (NumPy)
    xs, ys = pixelGrid(w, h)
    nx = (xs - w / 2) / (w / 2)
    ny = (ys - h / 2) / (h / 2)
    age = np.clip((np.sqrt(nx ** 2 + ny ** 2) - 0.6) / 0.8, 0, 1) ** 1.5
    fibres = valueNoise(w, h, 24, seed=3, octaves=3)
    base = np.array([240, 233, 214], dtype=np.float32)
    old = np.array([214, 194, 150], dtype=np.float32)
    colour = base + (old - base) * age[..., None]
    colour += ((fibres - 0.5) * 14)[..., None]
    rgb = np.clip(colour, 0, 255).astype(np.uint8)
    paperImg = Image.fromarray(rgb, "RGB").convert("RGBA")
    paperImg = ak.addNoise(paperImg, amount=7, seed=11)

    ink = Image.new("RGBA", (w * SS, h * SS), (0, 0, 0, 0))
    draw = ImageDraw.Draw(ink)
    # --- masthead -------------------------------------------------------
    ak.drawText(draw, w / 2, 40, "THE PELITA POST", "future", 31, INK, anchor="ms")
    ak.drawText(draw, 22, 28, "NIGHT", "sans_bold", 9, INK, anchor="ls")
    ak.drawText(draw, 22, 39, "EDITION", "sans_bold", 9, INK, anchor="ls")
    ak.drawText(draw, w - 22, 28, "RM1.50", "sans_bold", 9, INK, anchor="rs")
    ak.drawText(draw, w - 22, 39, "NO. 1986", "sans_bold", 9, INK, anchor="rs")
    draw.rectangle(ak.box(18, 48, w - 18, 50), fill=INK)
    draw.rectangle(ak.box(18, 52.5, w - 18, 53.2), fill=INK)
    ak.drawText(draw, w / 2, 64, "KUALA LUMPUR  -  FRIDAY NIGHT  -  LANTERN FESTIVAL EDITION",
                "sans_bold", 9, INK, anchor="ms")
    draw.rectangle(ak.box(18, 69, w - 18, 69.8), fill=INK)
    # --- headline (the words GOLDEN SPOON printed in a gold spot colour) --
    goldInk = (178, 124, 24, 255)
    parts = [("THE ", INK), ("GOLDEN SPOON", goldInk), (" VISITS", INK)]
    f = ak.font("sans_bold", 30)
    total = sum(f.getlength(t) for t, c in parts)
    x = w / 2 - total / 2
    for text, colour in parts:
        ak.drawText(draw, x, 100, text, "sans_bold", 30, colour, anchor="ls")
        x += f.getlength(text)
    ak.drawText(draw, w / 2, 132, "PELITA STREET TONIGHT!", "sans_bold", 30, INK, anchor="ms")
    ak.drawText(draw, w / 2, 150, "Famous food critic will choose the last stall for the Festival Food Hall",
                "sans_italic", 11.5, INK, anchor="ms")
    draw.rectangle(ak.box(18, 158, w - 18, 158.8), fill=INK)

    # --- halftone photo box ---------------------------------------------
    photoL, photoT, photoW, photoH = 20, 166, 176, 146
    source = criticPortraitSource(photoW, photoH)
    ink.alpha_composite(halftoneLayer(source, photoL, photoT, ink.size, cell=3.0))
    criticPortraitLines(draw, photoL, photoT, photoW)
    draw.rectangle(ak.box(photoL, photoT, photoL + photoW, photoT + photoH), outline=INK, width=int(1.4 * SS))
    # red 'EXCLUSIVE' tag in the corner of the photo
    ak.roundedRect(draw, photoL + 6, photoT + 6, photoL + 70, photoT + 20, 2, fill=(206, 44, 44, 255))
    ak.drawText(draw, photoL + 38, photoT + 16.5, "EXCLUSIVE", "sans_bold", 9, (255, 246, 230, 255), anchor="ms")
    ak.drawText(draw, photoL, photoT + photoH + 12, "Dato' Leon Tan, \"The Golden Spoon\".",
                "sans_italic", 10, INK, anchor="ls")
    ak.drawText(draw, photoL, photoT + photoH + 24, "He has never given five spoons.",
                "sans_italic", 10, (80, 74, 72, 255), anchor="ls")

    # --- two text columns (a few real words, then 'greeked' lines) ---------
    colX = [212, 366]
    colW = 142
    ak.drawText(draw, colX[0], 176, "WHO WILL WIN?", "sans_bold", 12, INK, anchor="ls")
    ak.drawText(draw, colX[0], 190, "Four stalls. One critic.", "sans", 10, INK, anchor="ls")
    ak.drawText(draw, colX[0], 202, "One spot in the Food Hall.", "sans", 10, INK, anchor="ls")
    lineRng = random.Random(77)
    greyInk = (70, 64, 64, 200)
    for c in range(2):
        y = 212 if c == 0 else 176
        bottom = 344 if c == 0 else 262
        while y < bottom:
            if lineRng.random() < 0.12 and y > 180:
                y += 5                                   # paragraph gap
                continue
            length = colW * lineRng.uniform(0.72, 1.0)
            if lineRng.random() < 0.15:
                length = colW * lineRng.uniform(0.3, 0.6)
            ak.roundedRect(draw, colX[c], y, colX[c] + length, y + 2.6, 1.2, fill=greyInk)
            y += 6.6
    # small advert box for Grandma Siti's stall (a story clue)
    adL, adT = colX[1], 270
    ak.roundedRect(draw, adL, adT, adL + colW, 344, 3, fill=None, outline=INK, width=1.4)
    ak.drawText(draw, adL + colW / 2, adT + 18, "GRANDMA SITI'S STALL", "sans_bold", 10, INK, anchor="ms")
    ak.drawText(draw, adL + colW / 2, adT + 32, "Since 1986 - Pelita Street", "sans_italic", 9.5, INK, anchor="ms")
    ak.drawText(draw, adL + colW / 2, adT + 48, "Nasi lemak - Rojak - Teh tarik", "sans", 9.5, INK, anchor="ms")
    ak.drawText(draw, adL + colW / 2, adT + 64, "Cooked with heart!", "sans_bold", 9.5, (206, 44, 44, 255), anchor="ms")
    draw.line(ak.pts([(360, 170), (360, 344)]), fill=(60, 56, 56, 160), width=int(0.8 * SS))

    inkSmall = downsample(ink, w, h)
    inkSmall.putalpha(inkSmall.getchannel("A").point(lambda v: int(v * 0.93)))  # ink soaks into paper
    paperImg.alpha_composite(inkSmall)

    # --- a teh tarik cup ring stain (story flavour) -----------------------
    stain = Image.new("RGBA", (w * 2, h * 2), (0, 0, 0, 0))
    sdraw = ImageDraw.Draw(stain)
    stainX, stainY = 462, 196
    for i in range(3):
        r = 27 - i * 1.4
        sdraw.arc([(stainX - r) * 2, (stainY - r) * 2 + i, (stainX + r) * 2, (stainY + r) * 2 + i],
                  200 + i * 30, 520 - i * 20, fill=(150, 96, 50, 95 - i * 20), width=3)
    stain = stain.resize((w, h), Image.LANCZOS).filter(ImageFilter.GaussianBlur(0.8))
    paperImg.alpha_composite(stain)

    # a soft horizontal fold crease
    crease = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    cdraw = ImageDraw.Draw(crease)
    cdraw.line([(8, 163), (w - 8, 163)], fill=(120, 100, 70, 45), width=2)
    cdraw.line([(8, 165), (w - 8, 165)], fill=(255, 255, 240, 60), width=1)
    paperImg.alpha_composite(crease.filter(ImageFilter.GaussianBlur(0.7)))

    paperImg.putalpha(paperMask)
    return addSoftShadow(paperImg, offset=(1, 2), blurPx=2.0, opacity=0.55, rgb=(10, 8, 16))


# ======================================================================
# 9. Main: build everything and save with the exact manifest names
# ======================================================================
def main():
    random.seed(2026)
    np.random.seed(2026)

    particles = [
        ("soft_circle.png", makeSoftCircle),
        ("hard_circle.png", makeHardCircle),
        ("droplet.png", makeDroplet),
        ("spark.png", makeSpark),
        ("star5.png", makeStar5),
        ("smoke_puff.png", makeSmokePuff),
        ("flame.png", makeFlame),
        ("confetti.png", makeConfetti),
        ("heart.png", makeHeartParticle),
        ("ring.png", makeRing),
        ("light_glow.png", makeLightGlow),
        ("bubble.png", makeBubble),
    ]
    for name, maker in particles:
        ak.save(maker(), "particles/" + name)

    ui = [
        ("star_full.png", makeStarFull),
        ("star_empty.png", makeStarEmpty),
        ("icon_heart.png", makeIconHeart),
        ("icon_clock.png", makeIconClock),
        ("icon_ladle.png", makeIconLadle),
        ("icon_lock.png", makeIconLock),
        ("icon_check.png", makeIconCheck),
        ("icon_coin.png", makeIconCoin),
        ("icon_camera.png", makeIconCamera),
        ("icon_webcam.png", makeIconWebcam),
        ("cursor_hand.png", makeCursorHand),
        ("cursor_grab.png", makeCursorGrab),
        ("thought_bubble.png", makeThoughtBubble),
        ("vignette.png", makeVignette),
        ("aura_frame.png", makeAuraFrame),
        ("poster_critic.png", makePoster),
    ]
    for name, maker in ui:
        ak.save(maker(), "ui/" + name)

    logo, glowSource = makeLogo()
    ak.save(logo, "ui/logo.png")
    ak.save(makeLogoGlow(glowSource), "ui/logo_glow.png")
    print("UI and particle images done.")


if __name__ == "__main__":
    main()
