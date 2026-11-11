"""
make_food_props.py
------------------
Draws every FOOD and PROP image of Midnight Makan with Pillow code
(no downloaded art). Run it from the project root:

    python tools/make_food_props.py            (makes everything)
    python tools/make_food_props.py wok chili  (only makers whose name contains a word)

Output: assets/images/food/*.png and assets/images/props/*.png

How the pictures are built (ideas from the lectures and tutorials):
  * Supersampling (anti-aliasing): every shape is drawn 4x bigger and
    shrunk with LANCZOS at the end, so edges are smooth, not jagged.
  * Masks: each part of an object (a chili body, a stem, a label...) is
    first drawn as a white-on-black "mask" (a greyscale L image). Masks can
    be joined (union), cut (subtract), grown (dilate) and blurred.
  * Fake 3D lighting: a mask is blurred into a "height map" (a soft hill).
    The slope of that hill gives a surface normal, and the Lambert (diffuse)
    rule  light = N . L  tells how much each pixel faces the lamp in the
    upper-left. The result is cut into flat bands (base / shade / highlight)
    = "toon shading", which gives the clean flat-vector look.
  * Colour models: RGB tuples, linear interpolation of colours (lerp) for
    gradients, and alpha (transparency) for glass and glows.
  * Transformations: circles swept along curves (tubes), shapes of
    revolution (bowls, jugs, glasses) built from stacked ellipses, and a
    perspective (QUAD) warp.
  * Image processing: GaussianBlur for soft shadows/glows, value noise for
    wood grain, paper and frosted glass.
  * Painter's algorithm (visibility / depth layering): food piles are drawn
    back-to-front, bowls paste their front lip back over the food, and the
    wok is drawn once then cut into wok_back.png + wok_front.png from the
    same pixels, so the game can draw food bits between the two layers.

The script is deterministic: every random number comes from a seeded
random.Random / numpy RandomState, so the images are the same each run.
"""
import math
import os
import random
import sys

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TOOLS_DIR)
import art_kit as ak  # noqa: E402  (shared helpers written for all image scripts)

SS = ak.SS
OUTLINE = ak.PALETTE["outline"]
WHITE = (255, 255, 255)

# Key light comes from the upper-left and a little towards the viewer.
# (x to the right, y DOWN like PIL, z towards the viewer)
_lx, _ly, _lz = -0.52, -0.62, 0.58
_len = math.sqrt(_lx * _lx + _ly * _ly + _lz * _lz)
LIGHT_DIR = (_lx / _len, _ly / _len, _lz / _len)


# ======================================================================
# Small colour helpers
# ======================================================================
def rgb(c):
    return (int(c[0]), int(c[1]), int(c[2]))


def rgba(c, a=255):
    return (int(c[0]), int(c[1]), int(c[2]), int(a))


def mixRgb(c1, c2, t):
    """Linear interpolation of two colours: c = c1 + (c2 - c1) * t."""
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))


def darkOf(c, amount=0.34):
    """Shadow colour: darker and pushed a little towards warm purple
    (shadows with a hue shift look richer than plain black)."""
    return mixRgb(mixRgb(c, (70, 30, 70), 0.28), (0, 0, 0), amount)


def lightOf(c, amount=0.38):
    """Highlight colour: towards a warm white."""
    return mixRgb(c, (255, 250, 232), amount)


def _band(col, target, t):
    """Blend a shading band into the colour array.
    target = colour tuple (ints)  -> mix towards that colour
    target = float or 3 floats    -> multiply (keeps painted patterns visible)"""
    if isinstance(target, float):
        target = (target, target, target)
    if isinstance(target[0], float):
        mul = np.array(target, np.float32)
        return col * (1.0 + (mul - 1.0) * t)
    return col + (np.array(target, np.float32) - col) * t


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


# ======================================================================
# Mask helpers (masks are "L" images at supersampled size)
# ======================================================================
def union(*masks):
    out = masks[0]
    for m in masks[1:]:
        out = ImageChops.lighter(out, m)
    return out


def minus(a, b):
    return ImageChops.subtract(a, b)


def inter(a, b):
    return ImageChops.darker(a, b)


def shift(mask, dx, dy):
    """Move a mask by (dx, dy) final pixels (empty space fills with 0)."""
    out = Image.new(mask.mode, mask.size, 0)
    out.paste(mask, (int(round(dx * SS)), int(round(dy * SS))))
    return out


def scaleAlpha(mask, factor):
    return mask.point(lambda v: int(v * factor))


def _paddedBox(mask, pad):
    bb = mask.getbbox()
    if bb is None:
        return None
    w, h = mask.size
    return (max(0, bb[0] - pad), max(0, bb[1] - pad), min(w, bb[2] + pad), min(h, bb[3] + pad))


def grow(mask, px):
    """Dilate (grow) a mask by about px final pixels.
    Trick: blur the mask, then keep every pixel that got even a little
    bit of white -> the shape becomes fatter with round corners."""
    if px <= 0:
        return mask
    pad = int(px * SS * 2) + 6
    box = _paddedBox(mask, pad)
    if box is None:
        return mask
    sigma = px * SS / 1.85
    sub = mask.crop(box).filter(ImageFilter.GaussianBlur(sigma))
    sub = sub.point(lambda v: 255 if v > 9 else 0)
    out = Image.new("L", mask.size, 0)
    out.paste(sub, box[:2])
    return union(out, mask)


def shrink(mask, px):
    """Erode (shrink) a mask by about px final pixels (the opposite of grow)."""
    if px <= 0:
        return mask
    pad = int(px * SS * 2) + 6
    box = _paddedBox(mask, pad)
    if box is None:
        return mask
    sigma = px * SS / 1.85
    sub = mask.crop(box).filter(ImageFilter.GaussianBlur(sigma))
    sub = sub.point(lambda v: 255 if v > 246 else 0)
    out = Image.new("L", mask.size, 0)
    out.paste(sub, box[:2])
    return inter(out, mask)


def soften(mask, px):
    return mask.filter(ImageFilter.GaussianBlur(px * SS))


def _boxBlurAxis(a, r, axis):
    """Moving average of width 2r+1 along one axis (using a running sum)."""
    if r < 1:
        return a
    pad = [(0, 0), (0, 0)]
    pad[axis] = (r + 1, r)
    c = np.cumsum(np.pad(a, pad, mode="constant"), axis=axis, dtype=np.float64)
    n = a.shape[axis]
    if axis == 0:
        out = c[2 * r + 1:2 * r + 1 + n] - c[0:n]
    else:
        out = c[:, 2 * r + 1:2 * r + 1 + n] - c[:, 0:n]
    return (out / (2 * r + 1)).astype(np.float32)


def floatBlur(a, sigma):
    """Gaussian-like blur in floating point (3 box blurs = almost a Gaussian).
    Pillow only blurs 8-bit images; the tiny 1/255 steps would show up as
    ripples in the lighting, so the height map is blurred with NumPy."""
    width = math.sqrt(4.0 * sigma * sigma + 1.0)
    r = max(1, int(round((width - 1) / 2)))
    for _ in range(3):
        a = _boxBlurAxis(a, r, 0)
        a = _boxBlurAxis(a, r, 1)
    return a


def valueNoise(width, height, cellW, cellH, seed, octaves=2):
    """Smooth random noise (0..1): a small grid of random numbers scaled up
    with bicubic interpolation. Used for wood grain, paper and frost."""
    rs = np.random.RandomState(seed)
    total = np.zeros((height, width), np.float32)
    amp, norm = 1.0, 0.0
    cw, ch = cellW, cellH
    for _ in range(octaves):
        gw = max(2, int(width / cw) + 2)
        gh = max(2, int(height / ch) + 2)
        grid = (rs.rand(gh, gw) * 255).astype(np.uint8)
        img = Image.fromarray(grid, "L").resize((width, height), Image.BICUBIC)
        total += np.asarray(img, np.float32) / 255.0 * amp
        norm += amp
        amp *= 0.5
        cw, ch = cw / 2.0, ch / 2.0
    return total / norm


# ======================================================================
# Sketch = one supersampled drawing (one output image)
# ======================================================================
class Sketch:
    def __init__(self, width, height):
        self.w = width
        self.h = height
        self.img = Image.new("RGBA", (width * SS, height * SS), (0, 0, 0, 0))
        self.size = self.img.size

    # ---------------- shape masks (coordinates in FINAL pixels) --------
    def newMask(self):
        m = Image.new("L", self.size, 0)
        return m, ImageDraw.Draw(m)

    def ellipse(self, x0, y0, x1, y1):
        m, d = self.newMask()
        d.ellipse(ak.box(x0, y0, x1, y1), fill=255)
        return m

    def circle(self, cx, cy, r):
        return self.ellipse(cx - r, cy - r, cx + r, cy + r)

    def poly(self, points):
        m, d = self.newMask()
        d.polygon(ak.pts(points), fill=255)
        return m

    def rrect(self, x0, y0, x1, y1, r):
        m, d = self.newMask()
        d.rounded_rectangle(ak.box(x0, y0, x1, y1), radius=r * SS, fill=255)
        return m

    def stroke(self, points, width):
        """A thick line with round caps."""
        m, d = self.newMask()
        ak.line(d, points, 255, width)
        return m

    def tube(self, points, r0, r1=None, radiusFn=None, step=0.35):
        """Sweep a circle along a path (like squeezing icing from a bag).
        The radius can change from r0 to r1, or follow radiusFn(t)."""
        if r1 is None:
            r1 = r0
        m, d = self.newMask()
        # cumulative length so the circles are evenly spaced
        lengths = [0.0]
        for i in range(1, len(points)):
            lengths.append(lengths[-1] + math.dist(points[i - 1], points[i]))
        total = max(lengths[-1], 0.001)
        count = max(2, int(total / step))
        j = 0
        for i in range(count + 1):
            s = total * i / count
            while j < len(points) - 2 and lengths[j + 1] < s:
                j += 1
            segLen = max(lengths[j + 1] - lengths[j], 0.0001)
            u = (s - lengths[j]) / segLen
            x = points[j][0] + (points[j + 1][0] - points[j][0]) * u
            y = points[j][1] + (points[j + 1][1] - points[j][1]) * u
            t = s / total
            r = radiusFn(t) if radiusFn else r0 + (r1 - r0) * t
            if r > 0:
                d.ellipse([(x - r) * SS, (y - r) * SS, (x + r) * SS, (y + r) * SS], fill=255)
        return m

    def revolve(self, cx, topY, height, radiusFn, tilt, step=0.25):
        """Shape of revolution (bowl, jug, glass): stack the horizontal
        circles of the object. Each circle, seen from above at an angle,
        is an ellipse with ry = r * tilt."""
        m, d = self.newMask()
        count = max(2, int(height / step))
        for i in range(count + 1):
            h = height * i / count
            r = radiusFn(h / height)
            y = topY + h
            d.ellipse([(cx - r) * SS, (y - r * tilt) * SS, (cx + r) * SS, (y + r * tilt) * SS], fill=255)
        return m

    def text(self, x, y, message, fontName, size, anchor="mm", strokeWidth=0):
        m, d = self.newMask()
        f = ak.font(fontName, size * SS)
        d.text((x * SS, y * SS), message, font=f, fill=255, anchor=anchor,
               stroke_width=int(strokeWidth * SS), stroke_fill=255)
        return m

    # ---------------- painting ----------------------------------------
    def layer(self):
        """A transparent RGBA canvas (same size) for drawing texture details."""
        lay = Image.new("RGBA", self.size, (0, 0, 0, 0))
        return lay, ImageDraw.Draw(lay)

    def paint(self, mask, color, alpha=255):
        """Fill the mask area with one colour."""
        box = _paddedBox(mask, 0)
        if box is None:
            return
        sub = mask.crop(box)
        if alpha < 255:
            sub = scaleAlpha(sub, alpha / 255.0)
        solid = Image.new("RGBA", sub.size, rgba(color, 0))
        solid.putalpha(sub)
        self.img.alpha_composite(solid, box[:2])

    def clip(self, layerImg, mask, alpha=1.0):
        """Composite a texture layer, but only inside the mask."""
        box = _paddedBox(mask, 0)
        if box is None:
            return
        lay = layerImg.crop(box)
        a = ImageChops.multiply(lay.getchannel("A"), mask.crop(box))
        if alpha < 1.0:
            a = scaleAlpha(a, alpha)
        lay.putalpha(a)
        self.img.alpha_composite(lay, box[:2])

    def over(self, layerImg):
        self.img.alpha_composite(layerImg)

    def shine(self, mask, clipTo, color=(255, 250, 235), alpha=200, blur=0.6):
        """Soft specular highlight, kept inside the object (clipTo)."""
        self.paint(inter(soften(mask, blur), clipTo), color, alpha)

    def outline(self, mask, width=3.0, color=OUTLINE, alpha=255):
        """Dark warm outline = the grown mask painted behind the part."""
        self.paint(grow(mask, width), color, alpha)

    def shadow(self, mask, blurPx=6, alpha=110, dx=0, dy=0, color=(20, 10, 20)):
        """Soft drop / contact shadow (blurred silhouette)."""
        m = shift(mask, dx, dy) if (dx or dy) else mask
        m = soften(m, blurPx)
        self.paint(m, color, alpha)

    def part(self, mask, base, dark=None, light=None, soft=5.0, outline=3.0,
             spec=None, darkAt=0.10, lightAt=0.13, specAt=0.34, bulge=5.0,
             smooth=0.25, deep=None, deepAt=0.55, edge=0.03, outlineColor=OUTLINE,
             alpha=255):
        """Draw one shaded part: outline behind, then a toon-shaded fill.

        base   = colour tuple or a full-size RGB(A) image (gradient)
        dark   = shade colour (lower-right), light = highlight (upper-left)
        spec   = optional glossy specular colour (small bright band)
        soft   = how round the part looks (blur radius of the height map)
        """
        if outline:
            self.outline(mask, outline, outlineColor)
        baseRgb = base if isinstance(base, Image.Image) else None
        baseColor = None if baseRgb is not None else base
        if dark is None:
            dark = darkOf(baseColor) if baseColor else 0.72
        if light is None:
            light = lightOf(baseColor) if baseColor else 1.12
        pad = int(soft * SS * 3) + 4
        box = _paddedBox(mask, pad)
        if box is None:
            return
        sub = mask.crop(box)
        d = self.lightArray(sub, soft, bulge)
        hh, ww = d.shape
        if baseRgb is not None:
            col = np.asarray(baseRgb.crop(box).convert("RGB"), np.float32).copy()
        else:
            col = np.empty((hh, ww, 3), np.float32)
            col[:] = baseColor
        # gentle continuous shading so big flat areas are not dead flat
        col *= (1.0 + smooth * np.clip(d, -0.8, 0.5))[..., None]
        t = smoothstep(darkAt - edge, darkAt + edge, -d)[..., None]
        col = _band(col, dark, t)
        if deep is not None:
            t = smoothstep(deepAt - edge, deepAt + edge, -d)[..., None]
            col = _band(col, deep, t)
        t = smoothstep(lightAt - edge, lightAt + edge, d)[..., None]
        col = _band(col, light, t)
        if spec is not None:
            t = smoothstep(specAt - edge, specAt + edge, d)[..., None]
            col = _band(col, spec, t)
        a = np.asarray(sub, np.float32) * (alpha / 255.0)
        out = np.dstack([np.clip(col, 0, 255), a]).astype(np.uint8)
        self.img.alpha_composite(Image.fromarray(out, "RGBA"), box[:2])

    @staticmethod
    def lightArray(maskImg, soft, bulge):
        """Lambert lighting from a blurred mask used as a height map.
        Returns (N.L - Lz): 0 on flat areas, + where the surface turns
        towards the light, - where it turns away."""
        s = max(1.0, soft * SS)
        h = floatBlur(np.asarray(maskImg, np.float32) / 255.0, s)
        gy, gx = np.gradient(h)
        k = s * bulge
        nx = -gx * k
        ny = -gy * k
        inv = 1.0 / np.sqrt(nx * nx + ny * ny + 1.0)
        lx, ly, lz = LIGHT_DIR
        return (nx * lx + ny * ly + lz) * inv - lz

    def gradient(self, stops, p0, p1):
        """Linear colour gradient image (full canvas) from point p0 to p1
        (final px). stops = [(t, colour), ...] with t from 0 to 1."""
        w, h = self.size
        xs = np.arange(w, dtype=np.float32)[None, :] / SS
        ys = np.arange(h, dtype=np.float32)[:, None] / SS
        dx, dy = p1[0] - p0[0], p1[1] - p0[1]
        length2 = dx * dx + dy * dy
        t = ((xs - p0[0]) * dx + (ys - p0[1]) * dy) / length2
        t = np.clip(t, 0.0, 1.0)
        return self._stopsToImage(t, stops)

    def radial(self, stops, cx, cy, rx, ry):
        w, h = self.size
        xs = (np.arange(w, dtype=np.float32)[None, :] / SS - cx) / rx
        ys = (np.arange(h, dtype=np.float32)[:, None] / SS - cy) / ry
        t = np.clip(np.sqrt(xs * xs + ys * ys), 0.0, 1.0)
        return self._stopsToImage(t, stops)

    @staticmethod
    def _stopsToImage(t, stops):
        out = np.zeros(t.shape + (3,), np.float32)
        for c in range(3):
            xp = [s[0] for s in stops]
            fp = [s[1][c] for s in stops]
            out[..., c] = np.interp(t, xp, fp)
        return Image.fromarray(out.astype(np.uint8), "RGB")

    # ---------------- output ------------------------------------------
    def result(self):
        return ak.finish(self.img, self.w, self.h)


def saveImage(img, relPath):
    ak.save(img, relPath)
    SAVED.append(relPath)


SAVED = []


def ellipsePoint(cx, cy, rx, ry, degrees):
    a = math.radians(degrees)
    return (cx + rx * math.cos(a), cy + ry * math.sin(a))


def arc(cx, cy, rx, ry, a0, a1, count=40):
    return ak.circlePoints(cx, cy, rx, ry, count, a0, a1)


def bez(*p, steps=40):
    return ak.bezier(*p, steps=steps)


def chain(*curves):
    """Join several point lists into one path (skip repeated join points)."""
    out = []
    for c in curves:
        if out and c and math.dist(out[-1], c[0]) < 0.01:
            out.extend(c[1:])
        else:
            out.extend(c)
    return out


# ======================================================================
# FOOD - chopping ingredients (horizontal side views)
# The game cuts these into vertical strips at fixed x fractions, so
# every strip must look good: details are spread along the whole length.
# ======================================================================
def makeCucumber():
    sk = Sketch(300, 90)
    rng = random.Random(101)

    def centreY(t):
        return 45 - 3.5 * math.sin(math.pi * t)

    def radius(t):
        return 20.5 + 7.5 * math.sin(math.pi * t) ** 0.55

    x0, x1 = 34, 266
    path = [(x0 + (x1 - x0) * i / 60, centreY(i / 60)) for i in range(61)]
    body = sk.tube(path, 0, radiusFn=radius)

    # small woody stem on the left end
    stem = sk.tube([(18, 43), (12, 41), (9, 38.5)], 4.0, 2.8)
    sk.part(stem, (120, 132, 60), soft=1.5, outline=2.2)

    base = sk.gradient([(0, (70, 140, 58)), (0.5, (52, 120, 50)), (1, (46, 108, 46))], (20, 20), (280, 70))
    sk.part(body, base, dark=(24, 70, 42), light=(112, 172, 82), soft=8, outline=3,
            darkAt=0.14, lightAt=0.16, bulge=4.0, deep=(18, 52, 36), deepAt=0.62)

    # pale stripes running along the length (they follow the round skin)
    lay, d = sk.layer()
    for v in (-0.62, -0.25, 0.12, 0.48, 0.8):
        pts = []
        for i in range(61):
            t = i / 60
            x = x0 - 12 + (x1 - x0 + 24) * t
            y = centreY(t) + radius(t) * math.sin(v + 0.08 * math.sin(t * 9 + v * 3))
            pts.append((x, y))
        width = 2.6 * math.cos(v) + 0.8
        ak.line(d, pts, (150, 196, 104, 150), width)
    sk.clip(lay.filter(ImageFilter.GaussianBlur(SS * 0.8)), shrink(body, 1.5), 0.9)

    # warty bumps: dark dot + light dot (tiny highlight from the upper-left)
    lay, d = sk.layer()
    for i in range(75):
        t = rng.uniform(0.04, 0.96)
        v = rng.uniform(-1.25, 1.2)
        x = x0 + (x1 - x0) * t
        y = centreY(t) + radius(t) * math.sin(v)
        r = rng.uniform(1.2, 2.0) * (0.6 + 0.4 * math.cos(v))
        ak.ellipse(d, x - r, y - r, x + r, y + r, (22, 66, 36, 200))
        ak.ellipse(d, x - r * 0.9 - 0.4, y - r * 0.9 - 0.5, x + r * 0.2 - 0.4, y + r * 0.2 - 0.5,
                   (190, 225, 150, 210))
    sk.clip(lay, shrink(body, 2))

    # long glossy highlight on the upper side
    hl = []
    for i in range(41):
        t = 0.1 + 0.72 * i / 40
        hl.append((x0 + (x1 - x0) * t, centreY(t) - radius(t) * 0.52))
    shine = sk.tube(hl, 0, radiusFn=lambda t: 2.6 * math.sin(math.pi * t) ** 0.7)
    sk.paint(soften(shine, 0.6), (225, 245, 200), 200)
    dot = sk.ellipse(52, 26, 64, 31)
    sk.paint(soften(dot, 0.5), (240, 250, 225), 190)

    # dried flower spot on the right tip
    tip = sk.circle(284, 42, 3.4)
    sk.part(tip, (214, 190, 110), soft=1, outline=1.6)
    saveImage(sk.result(), "food/cucumber.png")


def makePineapple():
    """Peeled pineapple log lying on its side (crown end on the left, cut
    base on the right). What makes it read as a pineapple (and not corn):
    long flat knife facets from peeling, brown 'eyes' sitting in a diagonal
    diamond lattice, a pale fibrous cut face with a core, and a bit of
    green-brown rind left at the ends."""
    sk = Sketch(300, 150)
    cy = 75
    xL, xF = 18, 268           # left (crown) cut end, centre of the right cut face

    def halfH(x):
        t = max(0.0, min(1.0, (x - xL) / float(xF - xL)))
        return 50 + 11 * math.sin(math.pi * (0.08 + 0.8 * t))

    def surfY(x, v):
        """Point on the log: v = angle around the log (-pi/2 top .. +pi/2 bottom)."""
        return cy + halfH(x) * math.sin(v)

    top = [(x, cy - halfH(x)) for x in range(xL + 8, xF + 1, 3)]
    bottom = [(x, cy + halfH(x)) for x in range(xF, xL + 7, -3)]
    barrel = sk.poly(top + bottom)
    leftEnd = sk.rrect(xL, cy - halfH(xL), xL + 26, cy + halfH(xL), 10)
    face = sk.ellipse(xF - 13, cy - halfH(xF), xF + 13, cy + halfH(xF))
    body = union(barrel, leftEnd, face)
    skin = minus(body, face)

    flesh = sk.gradient([(0, (255, 214, 70)), (0.5, (250, 192, 44)), (1, (234, 160, 30))], (40, 20), (250, 140))
    sk.part(body, flesh, dark=(214, 128, 26), light=(255, 234, 128), soft=9, outline=3,
            darkAt=0.13, lightAt=0.15, bulge=4.0, deep=(176, 96, 26), deepAt=0.64)

    # 1) long flat facets left by the knife when peeling (subtle light / dark planes)
    lay, d = sk.layer()
    edges = [-1.38, -0.98, -0.6, -0.22, 0.16, 0.55, 0.95, 1.38]
    for k in range(len(edges) - 1):
        v0, v1 = edges[k], edges[k + 1]
        xs = list(range(xL - 4, xF + 6, 4))
        upper = [(x, surfY(x, v0 + 0.03 * math.sin(x * 0.05 + k))) for x in xs]
        lower = [(x, surfY(x, v1 + 0.03 * math.sin(x * 0.05 + k + 1))) for x in reversed(xs)]
        tone = (255, 240, 150, 30) if k % 2 == 0 else (196, 120, 24, 24)
        d.polygon(ak.pts(upper + lower), fill=tone)
    sk.clip(lay.filter(ImageFilter.GaussianBlur(SS * 0.9)), shrink(skin, 2))

    # 2) diamond pattern: every brown eye sits in the middle of a small
    #    diamond-shaped bump. The bumps are separated by two families of
    #    diagonal grooves (the spirals of a real pineapple).
    #    The grid lives in surface coordinates (x along the log, v around it).
    step, dv = 32.0, 0.3
    eyes = []
    for j in range(10):
        v = -1.35 + j * dv
        x = xL - 8 + (j % 2) * step / 2
        while x < xF + 12:
            eyes.append((x, v))
            x += step

    def diamondEdge(a, b, n=6):
        return [(a[0] + (b[0] - a[0]) * i / n,
                 surfY(min(max(a[0] + (b[0] - a[0]) * i / n, xL), xF), a[1] + (b[1] - a[1]) * i / n))
                for i in range(n + 1)]

    lay, d = sk.layer()
    for x, v in eyes:
        left, topC, right, bottomC = (x - step / 2, v), (x, v - dv), (x + step / 2, v), (x, v + dv)
        # grooves on all four sides, light on the upper-left sides, shade on the lower-right
        for a, b in ((left, topC), (topC, right), (right, bottomC), (bottomC, left)):
            ak.line(d, diamondEdge(a, b), (178, 100, 22, 150), 1.3)
        ak.line(d, [(p[0] + 1.4, p[1] + 1.2) for p in diamondEdge(left, topC)], (255, 242, 170, 150), 1.6)
        ak.line(d, [(p[0] - 1.4, p[1] - 1.4) for p in diamondEdge(right, bottomC)], (200, 118, 24, 120), 2.2)
    sk.clip(lay.filter(ImageFilter.GaussianBlur(SS * 0.45)), shrink(skin, 2.5))

    lay, d = sk.layer()
    for x, v in eyes:
        if x < xL + 6 or x > xF - 10:
            continue
        y = surfY(x, v)
        fore = math.cos(v)                      # squash near the top / bottom (turning away)
        rx, ry = 4.6 * (0.7 + 0.3 * fore), 3.7 * fore + 0.6
        ak.ellipse(d, x - rx - 2.2, y - ry - 1.8, x + rx + 2.2, y + ry + 1.8, (214, 136, 34, 110))
        ak.ellipse(d, x - rx, y - ry, x + rx, y + ry, (146, 82, 32, 245))
        ak.ellipse(d, x - rx * 0.55, y - ry * 0.35, x + rx * 0.6, y + ry * 0.8, (70, 40, 22, 255))
        ak.line(d, [(x - rx * 0.2, y), (x + rx * 0.45, y - ry * 0.5)], (40, 24, 14, 255), 0.8)
        ak.ellipse(d, x - rx * 0.9, y - ry * 0.95, x - rx * 0.15, y - ry * 0.3, (255, 226, 150, 180))
    sk.clip(lay, shrink(skin, 2.5))

    # fibre texture (fine noise running along the length)
    noise = valueNoise(sk.size[0], sk.size[1], 90, 5, seed=7)
    tex = np.zeros(sk.size[::-1] + (4,), np.uint8)
    tex[..., 0], tex[..., 1], tex[..., 2] = 170, 100, 26
    tex[..., 3] = (np.clip(noise - 0.56, 0, 1) * 130).astype(np.uint8)
    sk.clip(Image.fromarray(tex, "RGBA"), skin, 0.7)

    # 3) green-brown rind left at the crown end and a few missed patches
    rng = random.Random(21)
    rindPts = [(xL - 4, cy - 70)]
    for i in range(15):
        yy = cy - 56 + i * 8
        rindPts.append((xL + 5 + rng.uniform(-1.5, 2.5) + (3 if i % 2 else 0), yy))
    rindPts += [(xL - 4, cy + 70)]
    rind = inter(sk.poly(rindPts), body)
    for (px, pv) in ((150, -1.42), (214, 1.42)):
        py = surfY(px, pv)
        patch = sk.poly([(px - 8, py - 3), (px - 2, py - 6), (px + 9, py - 3), (px + 7, py + 4), (px - 6, py + 4)])
        rind = union(rind, inter(patch, shrink(skin, 0.5)))
    sk.part(rind, (122, 112, 52), dark=(84, 72, 36), light=(160, 150, 80), soft=2, outline=0, bulge=3)
    sk.paint(minus(grow(rind, 1.0), rind), (160, 96, 30), 120)

    # 4) cut face: pale fibrous flesh, golden outer ring, radial fibres, woody core
    sk.part(face, (255, 238, 158), dark=(240, 204, 104), light=(255, 250, 210), soft=3,
            outline=2.4, darkAt=0.18, lightAt=0.12)
    faceRy = halfH(xF)
    lay, d = sk.layer()
    ak.ellipse(d, xF - 11, cy - faceRy + 5, xF + 11, cy + faceRy - 5, None, (246, 206, 96, 170), 2.2)
    for k in range(22):
        a = math.radians(k * (360 / 22) + 4)
        ak.line(d, [(xF + 3.2 * math.cos(a), cy + 13 * math.sin(a)),
                    (xF + 11.5 * math.cos(a), cy + (faceRy - 3) * math.sin(a))], (232, 186, 80, 140), 1.0)
    ak.ellipse(d, xF - 4.6, cy - 15, xF + 4.6, cy + 15, (255, 247, 212, 255))
    ak.ellipse(d, xF - 4.6, cy - 15, xF + 4.6, cy + 15, None, (222, 176, 84, 255), 1.2)
    sk.clip(lay, shrink(face, 1))
    sk.paint(minus(face, shrink(face, 2.4)), (140, 110, 46), 230)

    # glossy juicy highlights on the upper-left
    shine = sk.tube(bez((42, 34), (110, 20), (200, 22), steps=30), 0,
                    radiusFn=lambda t: 2.8 * math.sin(math.pi * t) ** 0.8)
    sk.paint(soften(shine, 0.7), (255, 250, 214), 175)
    sk.paint(soften(sk.ellipse(30, 44, 38, 60), 1.0), (255, 248, 210), 130)
    saveImage(sk.result(), "food/pineapple.png")


def makeMango():
    sk = Sketch(260, 150)
    outlinePts = chain(
        bez((16, 84), (14, 22), (126, 6), (200, 26)),
        bez((200, 26), (246, 38), (252, 78), (238, 104)),
        bez((238, 104), (224, 132), (178, 142), (126, 134)),
        bez((126, 134), (90, 128), (60, 142), (34, 128)),
        bez((34, 128), (20, 120), (15, 104), (16, 84)),
    )
    body = sk.poly(outlinePts)
    base = sk.gradient([(0, (226, 62, 52)), (0.3, (246, 110, 44)), (0.62, (255, 168, 42)),
                        (1, (246, 208, 70))], (30, 20), (230, 130))
    sk.part(body, base, dark=(206, 92, 34), light=(255, 212, 116), soft=12, outline=3,
            darkAt=0.2, lightAt=0.16, bulge=3.4, deep=(168, 56, 36), deepAt=0.72,
            spec=(255, 244, 214), specAt=0.36, edge=0.06, smooth=0.35)

    # blush patches and tiny lenticel specks (the pale dots on mango skin)
    rng = random.Random(33)
    lay, d = sk.layer()
    for i in range(70):
        x = rng.uniform(26, 236)
        y = rng.uniform(22, 132)
        r = rng.uniform(0.6, 1.1)
        ak.ellipse(d, x - r, y - r, x + r, y + r, (255, 236, 170, 150))
    sk.clip(lay, shrink(body, 4))
    blush = soften(sk.ellipse(40, 30, 130, 80), 10)
    sk.clip(Image.new("RGBA", sk.size, (214, 40, 60, 255)), inter(blush, body), 0.28)

    # glossy highlight on the upper-left
    sk.paint(soften(sk.ellipse(50, 30, 104, 44), 1.4), (255, 246, 222), 200)
    sk.paint(soften(sk.ellipse(38, 48, 48, 60), 0.8), (255, 246, 222), 180)

    # stem stub + small leaf at the shoulder
    leaf = sk.poly(chain(bez((34, 34), (22, 12), (4, 10), (4, 10)), bez((4, 10), (14, 28), (26, 36), (34, 34))))
    sk.part(leaf, (92, 150, 60), dark=(48, 96, 44), light=(150, 200, 96), soft=2.5, outline=2.2)
    vein = sk.stroke(bez((31, 32), (20, 22), (8, 12), steps=12), 0.9)
    sk.paint(vein, (60, 110, 46), 200)
    stem = sk.tube([(40, 36), (36, 28), (34, 22)], 3.2, 2.4)
    sk.part(stem, (110, 84, 44), soft=1.2, outline=2)
    saveImage(sk.result(), "food/mango.png")


def makeJicama():
    sk = Sketch(220, 170)
    # squat turnip shape: flat stem end on the left, short root tip on the right
    shapePts = chain(
        bez((20, 60), (24, 10), (96, 4), (140, 18)),
        bez((140, 18), (178, 30), (200, 62), (207, 85)),
        bez((207, 85), (200, 108), (178, 140), (140, 152)),
        bez((140, 152), (96, 166), (24, 160), (20, 110)),
        bez((20, 110), (15, 95), (15, 75), (20, 60)),
    )
    body = sk.poly(shapePts)
    base = sk.gradient([(0, (252, 246, 228)), (1, (238, 226, 198))], (30, 20), (200, 150))
    sk.part(body, base, dark=(212, 190, 150), light=(255, 254, 246), soft=13, outline=3,
            darkAt=0.13, lightAt=0.13, bulge=3.4, deep=(186, 160, 122), deepAt=0.66)

    # a few flat knife facets near the edges (straight light/dark planes)
    rng = random.Random(55)
    lay, d = sk.layer()
    for k in range(9):
        a = math.radians(-150 + k * 36 + rng.uniform(-8, 8))
        cx, cy2 = 118 + 92 * math.cos(a), 86 + 72 * math.sin(a)
        nx, ny = -math.sin(a), math.cos(a)
        ln = rng.uniform(16, 26)
        facet = [(cx + nx * ln, cy2 + ny * ln), (cx - nx * ln, cy2 - ny * ln),
                 (cx - nx * ln * 0.7 - math.cos(a) * 12, cy2 - ny * ln * 0.7 - math.sin(a) * 12),
                 (cx + nx * ln * 0.7 - math.cos(a) * 12, cy2 + ny * ln * 0.7 - math.sin(a) * 12)]
        tone = (255, 255, 252, 55) if math.cos(a) + math.sin(a) < 0 else (200, 178, 140, 40)
        d.polygon(ak.pts(facet), fill=tone)
    for i in range(46):
        x = rng.uniform(46, 188)
        y = rng.uniform(32, 138)
        ak.line(d, [(x, y), (x + rng.uniform(4, 10), y + rng.uniform(-1.2, 1.2))], (196, 172, 130, 90), 0.7)
    sk.clip(lay.filter(ImageFilter.GaussianBlur(SS * 1.2)), shrink(body, 2.5))

    # leftover strips of brown skin at both ends (clearly a peeled root)
    peelL = sk.poly([(10, 40), (40, 30), (44, 46), (38, 58), (46, 72), (40, 88), (47, 104), (38, 118),
                     (44, 132), (36, 146), (10, 140)])
    peelR = sk.poly([(220, 40), (182, 44), (188, 58), (178, 70), (186, 84), (176, 98), (185, 112),
                     (176, 126), (220, 140)])
    skin = inter(union(peelL, peelR), body)
    skinBase = sk.gradient([(0, (186, 140, 92)), (1, (150, 106, 66))], (0, 30), (0, 150))
    sk.part(skin, skinBase, dark=(116, 78, 50), light=(214, 176, 124), soft=5, outline=0,
            darkAt=0.14, lightAt=0.14, bulge=3.0)
    lay, d = sk.layer()
    for i in range(30):
        x = rng.choice((rng.uniform(16, 44), rng.uniform(186, 214)))
        y = rng.uniform(40, 140)
        ak.line(d, [(x, y), (x + rng.uniform(-2, 2), y + rng.uniform(3, 7))], (110, 74, 44, 150), 0.7)
    sk.clip(lay, skin)
    sk.paint(minus(grow(skin, 1.2), skin), (120, 84, 56), 150)

    # stem scar on the flat left side, root tip on the right
    scar = sk.ellipse(19, 77, 31, 93)
    sk.part(scar, (130, 94, 60), dark=(90, 60, 40), light=(170, 130, 90), soft=2, outline=1.6)
    root = sk.tube(bez((204, 85), (208, 86), (211, 89), (213, 94), steps=12), 2.6, 0.8)
    sk.part(root, (150, 106, 66), soft=0.8, outline=1.4)
    sk.shine(sk.ellipse(62, 28, 112, 42), body, (255, 255, 255), 200, 2)
    sk.shine(sk.ellipse(50, 48, 58, 60), body, (255, 255, 255), 150, 1.2)
    saveImage(sk.result(), "food/jicama.png")


def makeJambu():
    """Jambu air (rose apple) lying on its side: small round stem end on the
    left, bell body that flares to a broad, softly dimpled bottom on the
    right where the fleshy calyx lobes sit. Waxy, very glossy skin."""
    sk = Sketch(220, 150)
    outlinePts = chain(
        bez((14, 75), (14, 55), (30, 47), (52, 47)),
        bez((52, 47), (86, 47), (108, 18), (156, 14)),
        bez((156, 14), (194, 12), (208, 38), (205, 62)),
        bez((205, 62), (201, 69), (201, 81), (205, 88)),
        bez((205, 88), (208, 112), (194, 138), (156, 136)),
        bez((156, 136), (108, 132), (86, 103), (52, 103)),
        bez((52, 103), (30, 103), (14, 95), (14, 75)),
    )
    body = sk.poly(outlinePts)
    base = sk.gradient([(0, (250, 150, 162)), (0.25, (238, 84, 104)), (0.7, (214, 34, 68)),
                        (1, (190, 22, 58))], (14, 75), (206, 75))
    sk.part(body, base, dark=(150, 16, 52), light=(255, 140, 158), soft=11, outline=3,
            darkAt=0.13, lightAt=0.14, bulge=4.0, deep=(108, 10, 42), deepAt=0.62,
            spec=(255, 226, 234), specAt=0.34, smooth=0.3)

    # soft lobes running along the fruit: they follow the flare of the bell,
    # so the top / bottom of the silhouette is measured column by column
    bodyArr = np.asarray(body)
    lay, d = sk.layer()
    for v in (-0.56, 0.0, 0.56):
        pts = []
        for x in range(50, 198, 3):
            ys = np.nonzero(bodyArr[:, int(x * SS)])[0]
            topY, botY = ys.min() / SS, ys.max() / SS
            pts.append((x, (topY + botY) / 2 + (botY - topY) / 2 * v))
        ak.line(d, [(p[0], p[1] - 2.6) for p in pts], (255, 176, 190, 80), 3.4)
        ak.line(d, [(p[0], p[1] + 2.2) for p in pts], (140, 16, 50, 75), 2.4)
    sk.clip(lay.filter(ImageFilter.GaussianBlur(SS * 1.4)), shrink(body, 4))

    # pale blush near the stem end (jambu is lighter there)
    sk.clip(Image.new("RGBA", sk.size, (255, 196, 204, 255)),
            inter(soften(sk.ellipse(4, 52, 60, 98), 8), body), 0.45)

    # dimple at the broad end + calyx lobes poking out of it
    sk.paint(inter(soften(sk.ellipse(186, 56, 214, 94), 4), body), (120, 12, 44), 150)
    # calyx: three fleshy lobes around the dimple and a small brown tuft
    lobes = union(sk.circle(202, 68.5, 3.6), sk.circle(205.5, 75, 3.8), sk.circle(202, 81.5, 3.6))
    sk.part(lobes, (200, 74, 88), dark=(132, 30, 52), light=(240, 140, 150), soft=1.0, outline=1.5,
            darkAt=0.14, lightAt=0.1)
    lay, d = sk.layer()
    for k in range(5):
        a = math.radians(-28 + k * 14)
        ak.line(d, [(207, 75), (207 + 6 * math.cos(a), 75 + 6 * math.sin(a))], (108, 70, 40, 255), 1.0)
    sk.over(lay)

    # waxy highlights: long streak, crisp spot, small echoes and a lower rim light
    streak = sk.tube(bez((40, 58), (80, 44), (118, 26), (170, 22), steps=36), 0,
                     radiusFn=lambda t: 4.4 * math.sin(math.pi * t) ** 0.8)
    sk.shine(streak, shrink(body, 3), (255, 240, 244), 205, 0.9)
    sk.shine(sk.ellipse(170, 24, 184, 31), shrink(body, 2), (255, 255, 255), 235, 0.5)
    sk.shine(sk.ellipse(26, 60, 33, 67), shrink(body, 2), (255, 244, 246), 200, 0.5)
    sk.shine(sk.ellipse(120, 58, 126, 62), shrink(body, 2), (255, 236, 240), 130, 0.6)
    rim = sk.tube(bez((70, 100), (110, 116), (150, 130), (186, 122), steps=30), 0,
                  radiusFn=lambda t: 1.8 * math.sin(math.pi * t))
    sk.shine(rim, shrink(body, 2), (255, 150, 170), 150, 0.8)

    # little woody stem on the narrow end
    stem = sk.tube(bez((17, 73), (10, 71), (6, 67), steps=10), 2.6, 1.8)
    sk.part(stem, (118, 92, 52), soft=0.8, outline=1.6)
    saveImage(sk.result(), "food/jambu.png")


def makeChili():
    sk = Sketch(240, 60)
    path = bez((44, 29), (110, 44), (180, 48), (232, 28), steps=60)

    def radius(t):
        if t < 0.08:
            return 11.5 + 18 * t
        return max(1.3, 13.0 * (1.0 - ((t - 0.08) / 0.92) ** 1.7))

    body = sk.tube(path, 0, radiusFn=radius)
    sk.part(body, (222, 30, 34), dark=(140, 12, 30), light=(255, 98, 82), soft=5, outline=2.6,
            darkAt=0.12, lightAt=0.14, bulge=4.0, deep=(100, 8, 26), deepAt=0.6,
            spec=(255, 214, 200), specAt=0.33)
    shine = sk.tube(bez((56, 22), (100, 30), (160, 36), (208, 30), steps=40), 0,
                    radiusFn=lambda t: 2.1 * math.sin(math.pi * t) ** 0.6)
    sk.paint(soften(shine, 0.4), (255, 236, 228), 220)
    crease = sk.stroke(bez((58, 34), (72, 40), (86, 41), steps=10), 1.0)
    sk.paint(soften(crease, 0.3), (150, 14, 30), 150)

    # green calyx cap and curved stem
    cap = sk.poly(chain(bez((34, 16), (44, 12), (54, 18), (56, 24)),
                        [(56, 24), (50, 26), (54, 32), (48, 36), (52, 42)],
                        bez((52, 42), (44, 44), (36, 42), (32, 34)),
                        bez((32, 34), (28, 28), (28, 20), (34, 16))))
    sk.part(cap, (86, 150, 58), dark=(44, 96, 40), light=(146, 200, 96), soft=2.5, outline=2.2)
    stem = sk.tube(bez((34, 26), (22, 24), (14, 20), (8, 12), steps=20), 3.4, 2.4)
    sk.part(stem, (98, 156, 64), dark=(52, 100, 44), light=(160, 206, 110), soft=1.2, outline=2.0)
    sk.paint(sk.circle(8, 12, 2.0), (196, 214, 150), 255)
    saveImage(sk.result(), "food/chili.png")


# ======================================================================
# PROPS - Scene 1
# ======================================================================
def boardShape(sk, inset=0.0, dy=0.0):
    """Rounded trapezoid = the top of the board seen in 3/4 view."""
    tl = (46 + inset, 16 + inset + dy)
    tr = (654 - inset, 16 + inset + dy)
    br = (686 - inset, 196 - inset + dy)
    bl = (14 + inset, 196 - inset + dy)
    m = sk.poly([tl, tr, br, bl])
    r = max(4.0, 22 - inset)
    return shrink(grow(shrink(m, r * 0.5), r * 0.5), 0.1) if r > 0 else m


def makeCuttingBoard():
    sk = Sketch(700, 260)
    top = boardShape(sk)
    thickness = 30
    extruded = top
    for k in range(1, thickness + 1, 2):
        extruded = union(extruded, shift(top, 0, k))
    extruded = union(extruded, shift(top, 0, thickness))
    front = minus(extruded, top)

    # soft shadow on the counter
    sk.shadow(shift(extruded, 4, 8), blurPx=9, alpha=150)
    sk.outline(extruded, 4)

    # front edge (the board's thickness) - darker wood with grain lines
    frontBase = sk.gradient([(0, (178, 116, 64)), (1, (136, 84, 46))], (0, 190), (0, 230))
    sk.part(front, frontBase, dark=(112, 66, 38), light=(206, 146, 90), soft=3, outline=0,
            darkAt=0.25, lightAt=0.25, smooth=0.1)
    lay, d = sk.layer()
    rng = random.Random(21)
    for i in range(9):
        y = 202 + i * 3.2 + rng.uniform(-0.6, 0.6)
        ak.line(d, [(10, y), (240, y + rng.uniform(-1, 1)), (480, y + rng.uniform(-1, 1)), (690, y)],
                (96, 58, 32, 70), 0.9)
    sk.clip(lay, front)
    bevel = minus(shift(top, 0, 2.5), top)
    sk.paint(bevel, (228, 176, 118), 220)

    # top surface: glued strips of wood (butcher block)
    w, h = sk.size
    ys = np.arange(h, dtype=np.float32)[:, None] / SS
    xs = np.arange(w, dtype=np.float32)[None, :] / SS
    # perspective: strips get wider towards the front (bottom)
    tpos = np.clip((ys - 16) / 180.0, 0, 1)
    stripCoord = (tpos ** 0.85) * 7.0
    stripIndex = np.floor(stripCoord).astype(int)
    tones = np.array([[226, 172, 110], [214, 158, 98], [232, 182, 120], [218, 164, 104],
                      [210, 152, 92], [228, 176, 116], [220, 166, 106], [226, 172, 110]], np.float32)
    col = tones[np.clip(stripIndex, 0, 7)] * np.ones((1, w, 1), np.float32)
    # wood grain: stretched value noise pushed through a sine -> growth rings
    noise = valueNoise(w, h, 140 * SS, 7 * SS, seed=12, octaves=3)
    rings = np.sin((ys * 2.2 + noise * 14.0 + stripIndex * 1.7) * 1.3)
    grainDark = np.clip(rings, 0, 1) ** 3
    col *= (1.0 - 0.10 * grainDark)[..., None]
    col *= (0.96 + 0.08 * noise)[..., None]
    # thin dark seams between strips
    frac = stripCoord - stripIndex
    seam = ((frac < 0.035) & (stripIndex > 0)).astype(np.float32)
    col *= (1.0 - 0.2 * seam)[..., None]
    del xs
    topImg = Image.fromarray(np.clip(col, 0, 255).astype(np.uint8), "RGB")
    sk.part(top, topImg, dark=(178, 118, 66), light=(246, 206, 150), soft=6, outline=0,
            darkAt=0.16, lightAt=0.18, smooth=0.15, bulge=3)

    # juice groove around the edge
    grooveOuter = boardShape(sk, inset=14)
    grooveInner = boardShape(sk, inset=19)
    groove = minus(grooveOuter, grooveInner)
    sk.paint(groove, (164, 106, 58), 230)
    sk.paint(minus(shift(grooveInner, 0, 1.2), grooveInner), (246, 206, 150), 200)
    sk.paint(minus(grooveOuter, shift(grooveOuter, 0, 1.4)), (120, 74, 40), 120)

    # knife marks (mostly in the middle where the chopping happens)
    lay, d = sk.layer()
    for i in range(110):
        x = rng.gauss(350, 120)
        y = rng.uniform(46, 170)
        if not (70 < x < 630):
            continue
        ang = math.radians(rng.uniform(-30, 30) + (90 if rng.random() < 0.7 else 20))
        ln = rng.uniform(6, 22)
        dx, dy = math.cos(ang) * ln * 0.5, math.sin(ang) * ln * 0.35
        ak.line(d, [(x - dx + 0.7, y - dy + 0.7), (x + dx + 0.7, y + dy + 0.7)], (140, 90, 50, 80), 0.8)
        ak.line(d, [(x - dx, y - dy), (x + dx, y + dy)], (250, 222, 180, 130), 0.9)
    # an old knot in the wood
    for k in range(4):
        r = 4 + k * 3.2
        ak.ellipse(d, 548 - r * 1.8, 150 - r * 0.7, 548 + r * 1.8, 150 + r * 0.7, None, (150, 96, 54, 110 - k * 20), 1.0)
    ak.ellipse(d, 544, 148, 552, 152, (140, 88, 50, 170))
    sk.clip(lay, shrink(grooveInner, 1))
    # soft sheen on the upper-left of the board
    sk.clip(Image.new("RGBA", sk.size, (255, 238, 200, 255)),
            inter(soften(sk.ellipse(40, 10, 360, 90), 20), top), 0.25)
    saveImage(sk.result(), "props/cutting_board.png")


def makeCleaver():
    sk = Sketch(150, 230)
    # Blade: spine at the top, sharp edge at the bottom (flares a little,
    # like a real Chinese cleaver). Handle sits straight up in the middle so
    # the player can read exactly where the cut will land.
    bladePts = chain(bez((19, 112), (19, 106), (23, 102), (29, 102)),
                     [(121, 102)],
                     bez((121, 102), (127, 102), (131, 106), (131, 112)),
                     [(137, 208)],
                     bez((137, 208), (138, 219), (112, 223), (75, 224)),
                     bez((75, 224), (38, 223), (12, 219), (13, 208)))
    blade = sk.poly(bladePts)
    hole = sk.circle(36, 121, 6.2)
    blade = minus(blade, hole)
    steel = sk.gradient([(0, (170, 180, 196)), (0.4, (214, 222, 234)), (0.75, (186, 196, 212)),
                         (1, (140, 150, 168))], (14, 110), (136, 214))
    sk.part(blade, steel, dark=(110, 118, 138), light=(238, 242, 250), soft=3, outline=3.2,
            darkAt=0.16, lightAt=0.16, bulge=3.0, smooth=0.1)

    # dark forged band along the spine (hammered carbon steel look)
    spineBand = inter(blade, sk.poly([(0, 95), (150, 95), (150, 146), (0, 152)]))
    forged = sk.gradient([(0, (78, 84, 100)), (1, (54, 58, 72))], (0, 100), (0, 150))
    sk.part(spineBand, forged, dark=(38, 40, 52), light=(120, 128, 146), soft=2.5, outline=0,
            darkAt=0.15, lightAt=0.12, bulge=3.0)
    lay, d = sk.layer()
    rng = random.Random(8)
    for i in range(46):
        x = rng.uniform(20, 132)
        y = rng.uniform(106, 146)
        r = rng.uniform(1.6, 3.2)
        ak.ellipse(d, x - r, y - r * 0.7, x + r, y + r * 0.7, (140, 150, 170, 70))
        ak.ellipse(d, x - r * 0.6 + 0.6, y - r * 0.3 + 0.8, x + r + 0.6, y + r * 0.7 + 0.8, (30, 32, 44, 60))
    sk.clip(lay, shrink(spineBand, 1))
    # wavy border between forged and polished steel
    border = [(x, 149 - 3.0 * (x - 13) / 124 + 1.2 * math.sin(x * 0.35)) for x in range(10, 142, 2)]
    sk.paint(inter(sk.stroke(border, 1.6), blade), (40, 44, 56), 200)
    sk.paint(inter(sk.stroke([(p[0], p[1] + 1.6) for p in border], 1.0), blade), (246, 248, 252), 160)

    # polished edge bevel along the bottom (the sharp part catches light)
    bevelZone = inter(blade, sk.poly([(0, 202), (150, 202), (150, 230), (0, 230)]))
    sk.part(bevelZone, (232, 238, 246), dark=(176, 186, 202), light=(255, 255, 255),
            soft=1.5, outline=0, darkAt=0.2, lightAt=0.1)
    sk.paint(inter(sk.stroke([(14, 202), (136, 202)], 1.2), blade), (112, 120, 140), 210)
    sk.paint(inter(sk.stroke(bez((20, 219.5), (60, 222.5), (100, 222.5), (130, 219), steps=20), 0.9),
                   shrink(blade, 0.5)), (255, 255, 255), 230)

    # brushed-metal streaks and two diagonal reflections
    lay, d = sk.layer()
    for i in range(24):
        y = rng.uniform(154, 198)
        x0 = rng.uniform(18, 70)
        ak.line(d, [(x0, y), (x0 + rng.uniform(30, 70), y + rng.uniform(-0.5, 0.5))],
                (255, 255, 255, rng.randint(25, 60)), 0.6)
    d.polygon(ak.pts([(44, 150), (72, 150), (48, 202), (20, 202)]), fill=(255, 255, 255, 70))
    d.polygon(ak.pts([(80, 150), (90, 150), (66, 202), (56, 202)]), fill=(255, 255, 255, 95))
    sk.clip(lay, shrink(minus(blade, spineBand), 1.0))

    # hanging hole rim + maker's stamp on the polished part
    sk.paint(minus(sk.circle(36, 121, 8.4), sk.circle(36, 121, 6.2)), (28, 30, 40), 230)
    sk.paint(minus(sk.circle(36.9, 121.9, 7.0), sk.circle(36, 121, 6.2)), (170, 180, 200), 200)
    stampBox = sk.rrect(96, 158, 120, 184, 4)
    sk.paint(minus(stampBox, sk.rrect(98.3, 160.3, 117.7, 181.7, 3)), (96, 104, 122), 170)
    lay, d = sk.layer()
    for yy in (165, 171, 177):
        ak.line(d, [(101.5, yy), (114.5, yy)], (96, 104, 122, 255), 1.3)
    ak.line(d, [(108, 162.5), (108, 180)], (96, 104, 122, 255), 1.3)
    sk.clip(lay, stampBox, 0.7)

    # steel bolster (collar) joining handle and blade
    bolster = sk.rrect(54, 82, 96, 104, 7)
    sk.part(bolster, (176, 184, 198), dark=(100, 108, 126), light=(240, 244, 252), soft=2.5, outline=2.8,
            spec=(255, 255, 255), specAt=0.3)
    sk.paint(inter(sk.stroke([(56, 93), (94, 93)], 1.0), bolster), (110, 118, 136), 180)
    # wooden handle (rosewood) with grain and brass rivets
    handle = sk.rrect(59, 6, 91, 88, 14)
    wood = sk.gradient([(0, (150, 84, 44)), (0.45, (168, 98, 52)), (1, (112, 60, 34))], (59, 0), (91, 0))
    sk.part(handle, wood, dark=(82, 42, 28), light=(208, 138, 82), soft=4, outline=3.2,
            darkAt=0.12, lightAt=0.14, bulge=3.5)
    lay, d = sk.layer()
    for i, x in enumerate((64, 69, 74, 79.5, 85)):
        pts = [(x + 1.3 * math.sin(y * 0.09 + i * 1.7), y) for y in range(10, 86, 3)]
        ak.line(d, pts, (86, 44, 26, 120), 0.9)
    sk.clip(lay, shrink(handle, 1.5))
    for ry in (26, 62):
        rivet = sk.circle(75, ry, 4.4)
        sk.part(rivet, (232, 186, 92), dark=(160, 110, 46), light=(255, 236, 170), soft=1.5, outline=1.6,
                spec=(255, 255, 240), specAt=0.25)
    sk.shine(sk.rrect(63.5, 12, 68, 82, 2), handle, (236, 178, 122), 150, 0.6)
    saveImage(sk.result(), "props/cleaver.png")


# ======================================================================
# Shared helpers: bowls, chopped fruit pieces, noodles
# ======================================================================
def snapshot(sk, mask):
    """Copy of the drawing so far, only inside the mask. Used for the
    front lip of bowls: food is drawn first, then this copy is pasted
    back on top so the lip hides the food behind it (painter's algorithm)."""
    empty = Image.new("RGBA", sk.size, (0, 0, 0, 0))
    return Image.composite(sk.img, empty, mask)


def below(sk, y):
    """Mask of everything below the line y (final px)."""
    return sk.rrect(-10, y, sk.w + 10, sk.h + 10, 0)


def above(sk, y):
    return sk.rrect(-10, -10, sk.w + 10, y, 0)


def bowlRadius(t):
    """Profile of a bowl: radius factor from the rim (t=0) to the bottom (t=1)."""
    return 1.0 - 0.42 * t ** 1.7


def drawBowl(sk, cx, cy, rx, ry, depth, body=(244, 246, 250), bodyDark=(172, 186, 212),
             bodyLight=(255, 255, 255), inside=(250, 251, 253), insideDark=(190, 200, 222),
             footRx=None, footH=None, outline=3.0, wallPattern=None, insidePattern=None, shadow=True):
    """Round bowl in 3/4 view. (cx, cy) = centre of the rim ellipse.
    Draws everything except the front lip, which is returned as a layer:
    draw the food clipped to 'inside', then sk.over(frontLip)."""
    tilt = ry / float(rx)
    if footRx is None:
        footRx = rx * 0.46
    if footH is None:
        footH = max(3.0, depth * 0.1)
    wall = sk.revolve(cx, cy, depth, lambda t: rx * bowlRadius(t), tilt)
    foot = sk.revolve(cx, cy + depth - 1, footH + 1, lambda t: footRx, tilt)
    silhouette = union(wall, foot)
    bottomY = cy + depth + footH
    if shadow:
        sk.shadow(sk.ellipse(cx - footRx * 1.55, bottomY - footRx * tilt * 1.05,
                             cx + footRx * 1.55, bottomY + footRx * tilt * 1.1),
                  blurPx=2 + rx * 0.05, alpha=120)
    sk.part(silhouette, body, dark=bodyDark, light=bodyLight, soft=max(3.0, rx * 0.09), outline=outline,
            darkAt=0.1, lightAt=0.14, bulge=3.0, spec=(255, 255, 255), specAt=0.34)
    # the line where the foot ring meets the bowl
    footTop = sk.ellipse(cx - footRx, cy + depth - footRx * tilt, cx + footRx, cy + depth + footRx * tilt)
    sk.paint(inter(minus(grow(footTop, max(0.8, outline * 0.35)), footTop), below(sk, cy + depth)), bodyDark, 190)

    rimOuter = sk.ellipse(cx - rx, cy - ry, cx + rx, cy + ry)
    if wallPattern:
        wallPattern(minus(silhouette, rimOuter))

    lip = max(2.0, rx * 0.05)
    insideMask = sk.ellipse(cx - rx + lip, cy - ry + lip * 0.8, cx + rx - lip, cy + ry - lip * 0.8)
    sk.paint(rimOuter, bodyLight)
    sk.paint(inter(soften(rimOuter, 1), below(sk, cy + ry * 0.2)), body, 200)
    grad = sk.radial([(0, inside), (0.7, mixRgb(inside, insideDark, 0.3)), (1, insideDark)],
                     cx + rx * 0.08, cy + ry * 0.15, rx - lip, ry - lip)
    sk.clip(grad.convert("RGBA"), insideMask)
    # concave shading: the rim shades the upper-left inner wall, the lower-right wall catches light
    darkMoon = minus(insideMask, shift(insideMask, lip * 1.8, lip * 1.5))
    sk.paint(inter(soften(darkMoon, lip * 0.7), insideMask), insideDark, 170)
    lightMoon = minus(insideMask, shift(insideMask, -lip * 1.2, -lip * 1.8))
    sk.paint(inter(soften(lightMoon, lip * 0.5), insideMask), (255, 255, 255), 110)
    if insidePattern:
        insidePattern(insideMask)
    sk.paint(minus(grow(insideMask, max(0.7, lip * 0.25)), insideMask), bodyDark, 190)
    # bright highlight along the back-left of the rim
    sk.paint(soften(sk.stroke(arc(cx, cy, rx - lip * 0.5, ry - lip * 0.4, 196, 256), lip * 0.45), 0.3),
             (255, 255, 255), 235)
    frontLip = snapshot(sk, inter(minus(rimOuter, insideMask), below(sk, cy)))
    return insideMask, frontLip


def bluePorcelainWall(sk, cx, cy, rx, ry, depth, scale=1.0):
    """Returns a function that paints the blue-and-white porcelain pattern on a bowl wall."""
    tilt = ry / float(rx)
    blue, blueLight = (34, 74, 166), (116, 156, 224)

    def paintWall(wallMask):
        lay, d = sk.layer()
        # double band under the rim (follows the round wall = ellipses)
        for off, w in ((5.0 * scale, 1.8 * scale), (9.5 * scale, 1.0 * scale)):
            r = rx * bowlRadius(off / depth)
            ak.ellipse(d, cx - r, cy + off - r * tilt, cx + r, cy + off + r * tilt, None, blue + (255,), w)
        # flower motifs around the belly; squashed towards the sides (foreshortening)
        t = 0.47
        r = rx * bowlRadius(t)
        yc = cy + depth * t
        for a in range(12, 180, 24):
            rad = math.radians(a)
            px, py = cx + r * math.cos(rad), yc + r * tilt * math.sin(rad)
            fore = max(0.25, math.sin(rad))
            size = 6.2 * scale
            for k in range(5):
                pa = math.radians(k * 72 - 90)
                ex = px + size * 0.62 * math.cos(pa) * fore
                ey = py + size * 0.58 * math.sin(pa)
                ak.ellipse(d, ex - size * 0.42 * fore, ey - size * 0.4, ex + size * 0.42 * fore, ey + size * 0.4,
                           blueLight + (255,))
            ak.ellipse(d, px - size * 0.34 * fore, py - size * 0.34, px + size * 0.34 * fore, py + size * 0.34,
                       blue + (255,))
            # little leaf swirl between the flowers
            rad2 = math.radians(a + 12)
            qx, qy = cx + r * math.cos(rad2), yc + r * tilt * math.sin(rad2)
            ak.line(d, [(qx - 3 * scale * fore, qy + 2 * scale), (qx, qy - 1 * scale),
                        (qx + 3 * scale * fore, qy + 1.5 * scale)], blue + (255,), 1.0 * scale)
        # wavy band near the bottom
        t2 = 0.84
        r2 = rx * bowlRadius(t2)
        yc2 = cy + depth * t2
        pts = []
        for a in range(0, 181, 3):
            rad = math.radians(a)
            pts.append((cx + r2 * math.cos(rad), yc2 + r2 * tilt * math.sin(rad) + 1.4 * scale * math.sin(a * 0.3)))
        ak.line(d, pts, blue + (255,), 1.3 * scale)
        sk.clip(lay, shrink(wallMask, 0.8), 0.95)
    return paintWall


def bluePorcelainInside(sk, cx, cy, rx, ry, scale=1.0):
    tilt = ry / float(rx)
    blue, blueLight = (34, 74, 166), (116, 156, 224)

    def paintInside(insideMask):
        lay, d = sk.layer()
        lip = max(2.0, rx * 0.05) + 3 * scale
        ak.ellipse(d, cx - rx + lip, cy - ry + lip * tilt * 1.4, cx + rx - lip, cy + ry - lip * tilt * 1.4,
                   None, blue + (255,), 1.2 * scale)
        # medallion on the bottom of the bowl
        mx, my, mr = cx + rx * 0.02, cy + ry * 0.3, rx * 0.26
        ak.ellipse(d, mx - mr, my - mr * tilt, mx + mr, my + mr * tilt, None, blue + (255,), 1.4 * scale)
        for k in range(6):
            a = math.radians(k * 60)
            ex, ey = mx + mr * 0.5 * math.cos(a), my + mr * 0.5 * tilt * math.sin(a)
            ak.ellipse(d, ex - mr * 0.26, ey - mr * 0.26 * tilt, ex + mr * 0.26, ey + mr * 0.26 * tilt, blueLight + (255,))
        ak.ellipse(d, mx - mr * 0.2, my - mr * 0.2 * tilt, mx + mr * 0.2, my + mr * 0.2 * tilt, blue + (255,))
        sk.clip(lay.filter(ImageFilter.GaussianBlur(SS * 0.3)), insideMask, 0.8)
    return paintInside


def convexHull(points):
    """Smallest convex polygon around the points (monotone chain algorithm)."""
    pts = sorted(set(points))
    if len(pts) <= 2:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


# colours of chopped rojak pieces: (top face, side face, skin or None, corners)
PIECES = {
    "cucumber": ((196, 232, 150), (150, 200, 108), (58, 128, 56), 4),
    "pineapple": ((255, 222, 86), (238, 176, 40), None, 3),
    "mango": ((255, 186, 60), (236, 134, 30), None, 4),
    "jicama": ((252, 248, 234), (214, 204, 176), None, 4),
    "jambu": ((252, 238, 232), (236, 222, 214), (214, 34, 66), 3),
    "chili": ((236, 52, 40), (178, 22, 30), None, 5),
    "youtiao": ((226, 162, 80), (182, 112, 48), None, 4),
    "garlic": ((250, 234, 176), (218, 176, 96), None, 4),
    "ketupat": ((248, 246, 232), (222, 218, 198), (128, 170, 70), 4),
    "onion": ((242, 218, 236), (206, 150, 196), (150, 56, 118), 3),
}


def drawChunk(d, x, y, size, angle, kind, outlineW=1.5, squash=0.56, height=0.52, jitter=None):
    """One chopped piece in 3/4 view = a small prism (top face + visible side faces).
    Returns the top face points (used for sauce coating)."""
    top, side, skin, corners = PIECES[kind]
    r = size * 0.62
    topPts = []
    for k in range(corners):
        a = math.radians(angle + k * 360.0 / corners)
        rr = r * (1.0 if corners != 5 else 0.8)
        if jitter:                       # irregular knife cuts: every corner a bit different
            rr *= jitter[k][0]
            a += math.radians(jitter[k][1])
        topPts.append((x + rr * math.cos(a), y + rr * squash * math.sin(a)))
    h = size * height * (0.5 if kind == "chili" else 1.0)
    hull = convexHull(topPts + [(px, py + h) for px, py in topPts])
    d.polygon(ak.pts(hull), fill=rgba(side))
    # side faces that face the viewer get their own shade (Lambert-like: facing right = darker)
    n = len(topPts)
    skinDone = False
    for i in range(n):
        p, q = topPts[i], topPts[(i + 1) % n]
        ex, ey = q[0] - p[0], q[1] - p[1]
        nx, ny = ey, -ex                 # outward normal of a clockwise polygon (y down)
        if ny <= 0:
            continue
        length = math.hypot(nx, ny) or 1.0
        facing = nx / length
        col = mixRgb(side, darkOf(side, 0.2), 0.5 + 0.5 * facing)
        if skin is not None and not skinDone:
            col = mixRgb(skin, darkOf(skin, 0.25), 0.3 + 0.4 * facing)
            skinDone = True
        d.polygon(ak.pts([p, q, (q[0], q[1] + h), (p[0], p[1] + h)]), fill=rgba(col))
    d.polygon(ak.pts(topPts), fill=rgba(top))
    # highlight on the upper-left part of the top face
    hx = sum(p[0] for p in topPts) / n
    hy = sum(p[1] for p in topPts) / n
    inner = [(hx + (px - hx) * 0.55 - r * 0.12, hy + (py - hy) * 0.55 - r * 0.08) for px, py in topPts]
    d.polygon(ak.pts(inner), fill=rgba(lightOf(top, 0.4), 95))
    if kind == "cucumber":        # seeds in the middle of the pale flesh
        for k in range(3):
            sx = hx - r * 0.25 + k * r * 0.25
            ak.ellipse(d, sx - 0.9, hy - 0.6, sx + 0.9, hy + 0.6, (236, 246, 206, 255))
    if kind == "chili":           # pale seeds inside a chili ring
        ak.ellipse(d, hx - r * 0.35, hy - r * 0.2, hx + r * 0.35, hy + r * 0.2, (255, 220, 170, 255))
    d.line(ak.pts(hull + [hull[0]]), fill=rgba(OUTLINE), width=int(outlineW * SS), joint="curve")
    d.line(ak.pts(topPts + [topPts[0]]), fill=rgba(darkOf(side, 0.25), 200), width=max(1, int(outlineW * 0.5 * SS)))
    return topPts


def moundPieces(rng, count, cx, cy, rx, ry, moundH, sizeRange, kinds):
    """Random pieces on an elliptic surface, lifted into a mound
    (highest in the middle). Sorted back-to-front for the painter's algorithm."""
    out = []
    for i in range(count):
        a = rng.uniform(0, math.tau)
        rr = math.sqrt(rng.uniform(0.0, 1.0)) * 0.92
        px, py = cx + rx * rr * math.cos(a), cy + ry * rr * math.sin(a)
        lift = moundH * max(0.0, 1.0 - rr * rr)
        jitter = [(rng.uniform(0.8, 1.15), rng.uniform(-12, 12)) for k in range(6)]
        out.append((py - lift, px, rng.uniform(sizeRange[0], sizeRange[1]), rng.uniform(0, 360),
                    kinds[i % len(kinds)], jitter))
    out.sort()
    return out


def moundTop(cx, cy, rx, ry, moundH, u):
    """Highest point of a mound (same formula as moundPieces) above the
    horizontal position u = -1..1 across the mound."""
    vMax = math.sqrt(max(0.0, 1.0 - u * u))
    best = 0.0
    for i in range(13):
        v = vMax * i / 12.0
        best = max(best, ry * v + moundH * max(0.0, 1.0 - u * u - v * v))
    return cy - best


def fruitMass(d, cx, cy, rx, ry, moundH, color):
    """The juicy mass under the top pieces, so gaps never show the empty bowl."""
    pts = []
    for i in range(41):
        u = -1.0 + 2.0 * i / 40
        pts.append((cx + rx * u, moundTop(cx, cy, rx, ry, moundH, u)))
    for i in range(40, -1, -1):
        u = -1.0 + 2.0 * i / 40
        pts.append((cx + rx * u, cy + ry * math.sqrt(max(0.0, 1.0 - u * u))))
    d.polygon(ak.pts(pts), fill=rgba(color))


def drawRibbon(d, points, width, color, outlineW=1.2):
    """A flat noodle / ribbon: dark outline, body colour, a light centre stripe."""
    ak.line(d, points, rgba(OUTLINE, 235), width + outlineW * 2)
    ak.line(d, points, rgba(color), width)
    ak.line(d, [(p[0] - width * 0.12, p[1] - width * 0.18) for p in points], rgba(lightOf(color, 0.45), 200), width * 0.34)


# ======================================================================
# FOOD - Scene 1: rojak bowl stages and the finished rojak
# ======================================================================
ROJAK_KINDS = ["pineapple", "cucumber", "mango", "jambu", "jicama", "pineapple", "cucumber", "mango", "jambu", "chili"]
ROJAK_DISH_KINDS = ["pineapple", "youtiao", "cucumber", "mango", "jambu", "jicama", "youtiao", "pineapple", "cucumber", "jambu"]


def makeRojakBowls():
    """rojak_bowl_0..3: the same blue-and-white bowl, 0 = empty, 1-3 = more
    and more chopped fruit (layer k is always drawn the same, so the stages
    look like one bowl filling up)."""
    cx, cy, rx, ry, depth = 135, 57, 124, 42, 70
    layers = [
        # (surface depth below the rim, piece count, mound height, seed)
        (34, 22, 4, 11),
        (15, 44, 10, 12),
        (0, 84, 32, 13),
    ]
    for stage in range(4):
        sk = Sketch(270, 170)
        inside, frontLip = drawBowl(sk, cx, cy, rx, ry, depth, outline=3.0,
                                    wallPattern=bluePorcelainWall(sk, cx, cy, rx, ry, depth),
                                    insidePattern=bluePorcelainInside(sk, cx, cy, rx, ry))
        if stage > 0:
            lay, d = sk.layer()
            for k in range(stage):
                h, count, moundH, seed = layers[k]
                rng = random.Random(seed)
                r = rx * bowlRadius(h / float(depth)) - 6
                surfaceY = cy + h * 0.95
                if k > 0:
                    fruitMass(d, cx, surfaceY + 3, r * 0.9, r * 0.33 * 0.9, moundH * 0.85, (176, 138, 64))
                for yy, xx, size, ang, kind, jit in moundPieces(rng, count, cx, surfaceY, r * 0.9, r * 0.33 * 0.9,
                                                                moundH, (17, 24), ROJAK_KINDS):
                    drawChunk(d, xx, yy, size, ang, kind, 1.6, jitter=jit)
            food = union(inside, above(sk, cy))
            sk.shadow(inter(shift(lay.getchannel("A"), 1.5, 3), inside), 2, 70)
            sk.clip(lay, food)
            sk.over(frontLip)
        saveImage(sk.result(), f"food/rojak_bowl_{stage}.png")


def rojakHeap(sk, lay, d, rng, cx, cy, rx, ry, moundH, count, sizeRange, outlineW):
    """Chopped fruit and fried dough (youtiao) coated in thick rojak sauce.
    The pieces are drawn in 'lay'. Returns (sauce mask, glint spots).
    The sauce mask respects the drawing order: a front piece first erases
    the sauce of the pieces behind it, then adds its own coat."""
    sauce, sd = sk.newMask()
    glints = []
    fruitMass(d, cx, cy + 3, rx, ry, moundH * 0.85, (84, 40, 22))
    for yy, xx, size, ang, kind, jit in moundPieces(rng, count, cx, cy, rx, ry, moundH, sizeRange, ROJAK_DISH_KINDS):
        topPts = drawChunk(d, xx, yy, size, ang, kind, outlineW, jitter=jit)
        hull = convexHull(topPts + [(px, py + size * 0.52) for px, py in topPts])
        sd.polygon(ak.pts(hull), fill=0)
        # the sauce is thick like a paste: it covers the top and folds over the front edge
        hx = sum(p[0] for p in topPts) / len(topPts) + rng.uniform(-1.5, 1.5)
        hy = sum(p[1] for p in topPts) / len(topPts) + rng.uniform(-1, 1)
        cover = rng.uniform(0.82, 1.02)
        coat = []
        for px, py in topPts:
            x, y = hx + (px - hx) * cover, hy + (py - hy) * cover
            if y > hy:
                y += size * rng.uniform(0.1, 0.24)
            coat.append((x, y))
        sd.polygon(ak.pts(coat), fill=255)
        glints.append((hx - size * 0.24, hy - size * 0.14, size))
    return sauce, glints


def paintSauce(sk, sauceMask, clipTo, scale=1.0):
    sauceMask = inter(grow(sauceMask, 0.5 * scale), clipTo)
    sk.part(sauceMask, (96, 44, 24), dark=(56, 22, 14), light=(156, 84, 46), soft=1.6 * scale, outline=0,
            darkAt=0.1, lightAt=0.1, bulge=4.0, spec=(255, 214, 180), specAt=0.28, edge=0.04)
    sk.paint(minus(grow(sauceMask, 0.6 * scale), sauceMask), (54, 24, 16), 170)
    return sauceMask


def sprinklePeanuts(sk, d, rng, cx, cy, rx, ry, moundH, count, scale=1.0):
    """Crushed peanuts: small irregular tan crumbs, darker on the lower edge."""
    for i in range(count):
        a = rng.uniform(0, math.tau)
        rr = math.sqrt(rng.uniform(0, 1)) * 0.85
        x, y = cx + rx * rr * math.cos(a), cy + ry * rr * math.sin(a) - moundH * (1 - rr * rr) - 4 * scale
        s = rng.uniform(1.6, 3.0) * scale
        crumb = [(x + s * math.cos(math.radians(k * 72 + rng.uniform(-20, 20))) * rng.uniform(0.7, 1.2),
                  y + s * 0.8 * math.sin(math.radians(k * 72 + rng.uniform(-20, 20))) * rng.uniform(0.7, 1.2))
                 for k in range(5)]
        d.polygon(ak.pts([(p[0] + 0.3 * scale, p[1] + 0.5 * scale) for p in crumb]), fill=(120, 72, 36, 230))
        d.polygon(ak.pts(crumb), fill=rgba(rng.choice([(236, 196, 132), (222, 176, 110), (246, 214, 156)])))


def makeRojakDish():
    """The finished rojak: sauce-coated fruit and youtiao heaped in the same
    porcelain bowl, crushed peanuts and torch ginger on top, two bamboo picks."""
    sk = Sketch(320, 220)
    cx, cy, rx, ry, depth = 160, 100, 146, 50, 56
    inside, frontLip = drawBowl(sk, cx, cy, rx, ry, depth, outline=3.2,
                                wallPattern=bluePorcelainWall(sk, cx, cy, rx, ry, depth, 1.1),
                                insidePattern=bluePorcelainInside(sk, cx, cy, rx, ry, 1.1))
    food = union(inside, above(sk, cy))
    rng = random.Random(77)
    lay, d = sk.layer()
    sauce, glints = rojakHeap(sk, lay, d, rng, cx, cy + 4, rx * 0.74, ry * 0.66, 58, 104, (22, 29), 1.8)
    sk.shadow(inter(shift(lay.getchannel("A"), 2, 4), inside), 3, 80)
    sk.clip(lay, food)
    sauce = paintSauce(sk, inter(sauce, food), food, 1.2)
    paintGlints(sk, sauce, glints, 1.2)
    lay, d = sk.layer()
    sprinklePeanuts(sk, d, rng, cx, cy + 4, rx * 0.68, ry * 0.58, 58, 150, 1.25)
    # shredded torch ginger flower (bunga kantan): thin pink curls, the classic garnish
    for i in range(22):
        a = rng.uniform(0, math.tau)
        rr = math.sqrt(rng.uniform(0, 1)) * 0.6
        x = cx + rx * 0.7 * rr * math.cos(a)
        y = cy + 4 + ry * 0.6 * rr * math.sin(a) - 58 * (1 - rr * rr) - 6
        ang = rng.uniform(0, math.pi)
        curl = [(x + 5 * math.cos(ang) * u + 1.5 * math.sin(u * 3), y + 5 * math.sin(ang) * u * 0.6)
                for u in (-1, -0.5, 0, 0.5, 1)]
        ak.line(d, curl, (120, 30, 60, 220), 2.4)
        ak.line(d, curl, (246, 110, 150, 255), 1.3)
    sk.clip(lay, food)
    sk.over(frontLip)
    # bamboo picks stuck into the heap
    for (x0, y0, x1, y1) in ((196, 60, 236, 6), (122, 58, 104, 10)):
        pick = sk.stroke([(x0, y0), (x1, y1)], 3.0)
        sk.part(pick, (226, 196, 140), dark=(170, 136, 86), light=(250, 232, 190), soft=0.8, outline=1.6)
    saveImage(sk.result(), "food/rojak_dish.png")


# ======================================================================
# FOOD - Scene 2: bits tossed in the wok + the plated char kuey teow
# ======================================================================
def drawPrawn(sk, cx, cy, size, rotation=0.0, outlineW=1.8):
    """A cooked prawn curled into a C (seen from the side).
    size = overall width in px; rotation turns the C (degrees)."""
    R = size * 0.3
    path = []
    for i in range(31):
        a = math.radians(rotation - 150 + 265 * i / 30.0)
        path.append((cx + R * math.cos(a), cy + R * math.sin(a)))

    def thickness(t):
        return size * (0.2 - 0.1 * t) if t < 0.92 else size * 0.1 * (1.0 - (t - 0.92) / 0.08 * 0.5)
    body = sk.tube(path, 0, radiusFn=thickness)
    # tail fan at the end of the curl
    ex, ey = path[-1]
    px, py = path[-4]
    dirX, dirY = ex - px, ey - py
    length = math.hypot(dirX, dirY) or 1.0
    dirX, dirY = dirX / length, dirY / length
    nX, nY = -dirY, dirX
    tl = size * 0.2
    tail = sk.poly([(ex - dirX * 2, ey - dirY * 2), (ex + dirX * tl + nX * tl * 0.7, ey + dirY * tl + nY * tl * 0.7),
                    (ex + dirX * tl * 0.7, ey + dirY * tl * 0.7),
                    (ex + dirX * tl - nX * tl * 0.7, ey + dirY * tl - nY * tl * 0.7)])
    sk.part(tail, (236, 84, 52), dark=(170, 40, 34), light=(255, 150, 110), soft=1.0, outline=outlineW * 0.8)
    sk.part(body, (252, 136, 92), dark=(214, 76, 56), light=(255, 196, 160), soft=size * 0.05, outline=outlineW,
            darkAt=0.12, lightAt=0.12, bulge=4.0, spec=(255, 236, 220), specAt=0.34)
    # shell segments: light bands across the body + a red back line
    lay, d = sk.layer()
    for i in range(3, 28, 4):
        t = i / 30.0
        x0, y0 = path[i]
        x1, y1 = path[i + 1]
        tx, ty = x1 - x0, y1 - y0
        tlen = math.hypot(tx, ty) or 1.0
        nx, ny = -ty / tlen, tx / tlen
        w = thickness(t)
        ak.line(d, [(x0 + nx * w, y0 + ny * w), (x0 - nx * w, y0 - ny * w)], (255, 222, 200, 220), max(0.8, size * 0.035))
    back = [(cx + (R + thickness(i / 30.0) * 0.55) * math.cos(math.radians(rotation - 150 + 265 * i / 30.0)),
             cy + (R + thickness(i / 30.0) * 0.55) * math.sin(math.radians(rotation - 150 + 265 * i / 30.0)))
            for i in range(2, 28)]
    ak.line(d, back, (214, 60, 44, 200), max(0.8, size * 0.04))
    sk.clip(lay, shrink(body, 0.6))
    return body


def makeBitGarlic():
    sk = Sketch(30, 30)
    lay, d = sk.layer()
    jitter = [(1.1, 8), (0.85, -10), (1.05, 12), (0.9, -6), (1.0, 0), (1.0, 0)]
    for (x, y, s, a) in ((11, 10, 10, 20), (20, 12, 9, 60), (14, 17, 10, 35)):
        drawChunk(d, x, y, s, a, "garlic", 1.3, squash=0.7, height=0.45, jitter=jitter)
    sk.shadow(lay.getchannel("A"), 1, 60, 0.5, 1)
    sk.over(lay)
    # golden fried edges
    sk.paint(soften(sk.ellipse(7, 14, 22, 22), 1.5), (214, 150, 60), 60)
    saveImage(sk.result(), "food/bit_garlic.png")


def makeBitPrawn():
    sk = Sketch(50, 36)
    drawPrawn(sk, 25, 19.5, 32, rotation=10, outlineW=1.7)
    saveImage(sk.result(), "food/bit_prawn.png")


def makeBitNoodle():
    """One flat kuey teow strip with a twist and a few charred (wok hei) spots."""
    sk = Sketch(64, 22)
    path = [(9 + 46 * i / 30.0, 11 + 3.0 * math.sin(i / 30.0 * math.pi * 1.6 + 0.3)) for i in range(31)]
    strip = sk.tube(path, 0, radiusFn=lambda t: 3.8 - 1.4 * math.exp(-((t - 0.55) / 0.08) ** 2))
    sk.part(strip, (244, 226, 186), dark=(206, 170, 118), light=(255, 248, 226), soft=1.4, outline=1.6,
            darkAt=0.12, lightAt=0.12, spec=(255, 255, 245), specAt=0.3)
    lay, d = sk.layer()
    rng = random.Random(5)
    for i in range(7):
        x, y = path[rng.randint(2, 28)]
        r = rng.uniform(0.8, 1.6)
        ak.ellipse(d, x - r * 1.4, y - r + rng.uniform(-1, 1), x + r * 1.4, y + r, (150, 90, 44, 170))
    sk.clip(lay.filter(ImageFilter.GaussianBlur(SS * 0.4)), shrink(strip, 0.5))
    saveImage(sk.result(), "food/bit_noodle.png")


def makeBitSoy():
    """A small clump of noodles already coated in dark soy sauce."""
    sk = Sketch(40, 24)
    lay, d = sk.layer()
    strands = [
        bez((7, 14), (13, 5), (24, 17), (32, 8), steps=20),
        bez((8, 8), (16, 16), (24, 6), (32, 15), steps=20),
        bez((10, 17), (18, 11), (26, 19), (31, 12), steps=20),
    ]
    colours = [(128, 70, 34), (104, 56, 28), (146, 84, 40)]
    for pts, col in zip(strands, colours):
        drawRibbon(d, pts, 5.0, col, 1.2)
    sk.over(lay)
    sk.shine(sk.ellipse(12, 7, 17, 9), lay.getchannel("A"), (255, 226, 190), 200, 0.3)
    sk.shine(sk.ellipse(24, 11, 28, 13), lay.getchannel("A"), (255, 226, 190), 160, 0.3)
    saveImage(sk.result(), "food/bit_soy.png")


def makeBitEgg():
    """A piece of fluffy fried egg: yellow, some white, crispy brown edge."""
    sk = Sketch(44, 32)
    blob = union(sk.circle(16, 14, 8.6), sk.circle(26, 12.5, 8.2), sk.circle(30, 18.5, 6.8), sk.circle(19, 19.5, 7.2),
                 sk.circle(10, 17.5, 5.3))
    sk.part(blob, (255, 206, 72), dark=(226, 150, 40), light=(255, 236, 150), soft=3, outline=1.8,
            darkAt=0.12, lightAt=0.12, bulge=3.5)
    white = inter(union(sk.circle(25, 11, 4.2), sk.circle(13, 21, 3.4)), shrink(blob, 1))
    sk.paint(soften(white, 0.5), (255, 250, 236), 235)
    crisp = inter(minus(blob, shift(blob, -2, -2.5)), blob)
    sk.paint(soften(crisp, 0.5), (190, 112, 40), 170)
    sk.shine(sk.ellipse(12, 9, 18, 12), blob, (255, 250, 220), 200, 0.5)
    saveImage(sk.result(), "food/bit_egg.png")


def makeBitSprout():
    """Bean sprout: pale crunchy stem, yellow bean head, thin root tail."""
    sk = Sketch(46, 14)
    stem = sk.tube(bez((11, 6.5), (20, 8.4), (30, 8.6), (39, 6.8), steps=20), 2.2, 1.5)
    sk.part(stem, (246, 244, 222), dark=(206, 206, 176), light=(255, 255, 250), soft=1.0, outline=1.4,
            spec=(255, 255, 255), specAt=0.3)
    tail = sk.stroke(bez((39, 6.8), (41, 6), (42.5, 5), steps=6), 0.8)
    sk.paint(tail, (200, 190, 150), 230)
    head = sk.ellipse(4.5, 3, 13.5, 10)
    sk.part(head, (236, 222, 110), dark=(190, 170, 70), light=(252, 244, 170), soft=1.2, outline=1.4)
    sk.paint(soften(sk.ellipse(5.5, 4.5, 8.5, 7.5), 0.4), (150, 190, 90), 200)
    saveImage(sk.result(), "food/bit_sprout.png")


def drawOvalPlate(sk, cx, cy, rx, ry, thick=6.0, outline=3.0, band=(64, 150, 140)):
    """Oval hawker plate in 3/4 view: rim edge, flat rim with a coloured band,
    slightly sunken middle. Returns the mask of the middle (the food area)."""
    sk.shadow(sk.ellipse(cx - rx * 0.95, cy - ry * 0.8 + thick + 4, cx + rx * 0.95, cy + ry * 0.95 + thick + 4),
              blurPx=4, alpha=120)
    edge = sk.ellipse(cx - rx, cy - ry + thick, cx + rx, cy + ry + thick)
    top = sk.ellipse(cx - rx, cy - ry, cx + rx, cy + ry)
    sk.part(union(edge, top), (226, 230, 236), dark=(170, 176, 196), light=(250, 252, 255), soft=2,
            outline=outline, darkAt=0.1, lightAt=0.2)
    sk.paint(top, (250, 250, 252))
    sk.paint(minus(grow(top, 0.8), top), (190, 196, 212), 200)
    well = sk.ellipse(cx - rx * 0.76, cy - ry * 0.72, cx + rx * 0.76, cy + ry * 0.74)
    sk.paint(soften(minus(well, shift(well, 2.5, 2.5)), 1.2), (200, 206, 222), 200)
    sk.paint(soften(minus(well, shift(well, -2, -2)), 1.0), (255, 255, 255), 200)
    lay, d = sk.layer()
    for k, (f, w) in enumerate(((0.9, 2.2), (0.84, 1.0))):
        ak.ellipse(d, cx - rx * f, cy - ry * f, cx + rx * f, cy + ry * f, None, rgba(band), w)
    sk.clip(lay, minus(top, well))
    sk.paint(soften(sk.stroke(arc(cx, cy, rx * 0.95, ry * 0.95, 200, 250), 1.6), 0.4), (255, 255, 255), 230)
    return well


def drawCharKueyTeow(sk, cx, cy, rx, ry, moundH, seed, scale=1.0, prawns=4):
    """A heap of flat rice noodles fried dark in the wok, with prawns,
    Chinese sausage, egg, bean sprouts and chives."""
    rng = random.Random(seed)
    lay, d = sk.layer()
    fruitMass(d, cx, cy, rx * 0.92, ry * 0.92, moundH * 0.85, (150, 92, 46))
    noodleCols = [(210, 150, 84), (190, 124, 62), (226, 174, 106), (172, 108, 54), (216, 160, 94), (236, 196, 136)]
    ribbons = []
    for i in range(int(110 * max(0.45, scale))):
        a0 = rng.uniform(0, math.tau)
        r0 = math.sqrt(rng.uniform(0, 1)) * 0.85
        ang = rng.uniform(-0.7, 0.7) + (math.pi if rng.random() < 0.5 else 0)
        length = rng.uniform(0.35, 0.65)
        phase = rng.uniform(0, math.tau)
        pts = []
        for k in range(14):
            u = (k / 13.0 - 0.5) * length
            wave = 0.06 * math.sin(k * 0.9 + phase)
            ux = r0 * math.cos(a0) + u * math.cos(ang) - wave * math.sin(ang)
            uy = r0 * math.sin(a0) + u * math.sin(ang) + wave * math.cos(ang)
            rr2 = min(1.0, ux * ux + uy * uy)
            pts.append((cx + rx * ux, cy + ry * uy - moundH * (1.0 - rr2)))
        width = rng.uniform(7.0, 9.5) * scale
        widths = [width * (0.45 + 0.55 * abs(math.cos(k * 0.45 + phase))) for k in range(14)]
        ribbons.append((sum(p[1] for p in pts) / len(pts), pts, widths, rng.choice(noodleCols)))
    ribbons.sort(key=lambda r: r[0])                     # back to front (painter's algorithm)
    for _, pts, widths, col in ribbons:
        drawFlatRibbon(d, pts, widths, col, 1.1 * max(0.8, scale))
    noodles = lay.getchannel("A")
    sk.shadow(shift(noodles, 1.5 * scale, 3 * scale), 2 * scale, 90)
    sk.over(lay)
    # wok hei: charred brown spots, only on the noodles
    lay, d = sk.layer()
    for i in range(int(160 * scale)):
        x, y = cx + rng.uniform(-rx, rx), cy + rng.uniform(-ry - moundH, ry)
        r = rng.uniform(0.9, 2.0) * scale
        ak.ellipse(d, x - r * 1.5, y - r, x + r * 1.5, y + r, (96, 52, 24, 150))
    sk.clip(lay.filter(ImageFilter.GaussianBlur(SS * 0.4 * scale)), shrink(noodles, 1.0))

    def spot(spread=0.8):
        a = rng.uniform(0, math.tau)
        rr = math.sqrt(rng.uniform(0, 1)) * spread
        return cx + rx * rr * math.cos(a), cy + ry * rr * math.sin(a) - moundH * (1.0 - rr * rr) - 3 * scale

    lay, d = sk.layer()
    for i in range(int(12 * scale) + 3):                  # bean sprouts
        x, y = spot()
        ang = rng.uniform(-0.6, 0.6)
        l = 8 * scale
        pts = [(x - l * math.cos(ang), y - l * math.sin(ang)), (x, y + 2 * scale), (x + l * math.cos(ang), y + l * math.sin(ang))]
        ak.line(d, pts, rgba(OUTLINE, 230), 4.2 * scale)
        ak.line(d, pts, (250, 248, 230, 255), 2.6 * scale)
        ak.ellipse(d, pts[0][0] - 2.8 * scale, pts[0][1] - 2.0 * scale, pts[0][0] + 2.8 * scale, pts[0][1] + 2.0 * scale,
                   rgba(OUTLINE))
        ak.ellipse(d, pts[0][0] - 2.0 * scale, pts[0][1] - 1.3 * scale, pts[0][0] + 2.0 * scale, pts[0][1] + 1.3 * scale,
                   (240, 224, 110, 255))
    for i in range(int(8 * scale) + 2):                   # lap cheong (Chinese sausage) slices
        x, y = spot()
        r = 6.0 * scale
        ak.ellipse(d, x - r - 1.2 * scale, y - r * 0.6 - 1.2 * scale, x + r + 1.2 * scale, y + r * 0.6 + 1.2 * scale, rgba(OUTLINE))
        ak.ellipse(d, x - r, y - r * 0.6, x + r, y + r * 0.6, (204, 56, 62, 255))
        ak.ellipse(d, x - r * 0.7, y - r * 0.45, x + r * 0.1, y, (232, 110, 110, 255))
        for k in range(3):
            fx, fy = x + rng.uniform(-r * 0.6, r * 0.6), y + rng.uniform(-r * 0.3, r * 0.3)
            ak.ellipse(d, fx - 1.0 * scale, fy - 0.7 * scale, fx + 1.0 * scale, fy + 0.7 * scale, (250, 206, 200, 255))
    for i in range(int(7 * scale) + 2):                   # scrambled egg bits
        x, y = spot()
        r = rng.uniform(4.5, 6.0) * scale
        blob = [(x + r * math.cos(math.radians(k * 60 + rng.uniform(-15, 15))) * rng.uniform(0.7, 1.1),
                 y + r * 0.7 * math.sin(math.radians(k * 60 + rng.uniform(-15, 15))) * rng.uniform(0.7, 1.1))
                for k in range(6)]
        d.polygon(ak.pts(blob), fill=(255, 206, 70, 255), outline=rgba(OUTLINE), width=max(1, int(1.2 * scale * SS)))
        ak.ellipse(d, x - r * 0.45, y - r * 0.4, x + r * 0.1, y, (255, 240, 170, 255))
    for i in range(int(20 * scale) + 5):                  # garlic chives
        x, y = spot(0.85)
        ang = rng.uniform(0, math.pi)
        l = rng.uniform(5, 8) * scale
        pts = [(x - l * math.cos(ang), y - l * 0.6 * math.sin(ang)), (x + l * math.cos(ang), y + l * 0.6 * math.sin(ang))]
        ak.line(d, pts, (30, 64, 28, 255), 3.8 * scale)
        ak.line(d, pts, (92, 178, 66, 255), 2.3 * scale)
    sk.over(lay)
    # prawns sit on top of the heap, spread out (fixed spots so they never clump)
    spots = [(-0.5, -0.05, 200), (0.05, -0.4, 20), (0.5, 0.0, 150), (-0.08, 0.35, 300), (0.3, -0.62, 80)]
    for i in range(prawns):
        u, v, rot = spots[i]
        rr2 = min(1.0, u * u + v * v)
        drawPrawn(sk, cx + rx * u, cy + ry * v - moundH * (1.0 - rr2) - 2 * scale, 32 * scale,
                  rotation=rot, outlineW=1.7 * max(0.7, scale))


def makePlateCkt():
    sk = Sketch(320, 210)
    drawOvalPlate(sk, 160, 112, 150, 80, thick=7, outline=3.0)
    drawCharKueyTeow(sk, 160, 118, 108, 50, 44, seed=41, scale=1.0, prawns=5)
    saveImage(sk.result(), "food/plate_ckt.png")


def drawFlatRibbon(d, points, widths, color, outlineW=1.2):
    """Flat noodle whose width changes along its length (it twists), drawn
    as one polygon: the centre line is offset left and right along the normal."""
    left, right = [], []
    n = len(points)
    for i in range(n):
        p0 = points[max(0, i - 1)]
        p1 = points[min(n - 1, i + 1)]
        tx, ty = p1[0] - p0[0], p1[1] - p0[1]
        length = math.hypot(tx, ty) or 1.0
        nx, ny = -ty / length, tx / length
        w = widths[i] / 2.0
        left.append((points[i][0] + nx * w, points[i][1] + ny * w))
        right.append((points[i][0] - nx * w, points[i][1] - ny * w))
    poly = left + right[::-1]
    d.polygon(ak.pts(poly), fill=rgba(color), outline=rgba(OUTLINE, 235), width=max(1, int(outlineW * SS)))
    # light stripe on the upper edge = the flat noodle catching the light
    stripe = [(p[0] * 0.65 + q[0] * 0.35, p[1] * 0.65 + q[1] * 0.35) for p, q in zip(left, points)]
    ak.line(d, stripe[1:-1], rgba(lightOf(color, 0.4), 190), max(0.8, min(widths) * 0.3))


# ======================================================================
# FOOD - Scene 4: banquet dishes (150x110, 3/4 top view, readable when small)
# ======================================================================
def makeDishRojak():
    sk = Sketch(150, 110)
    cx, cy, rx, ry, depth = 75, 53, 60, 21, 33
    inside, frontLip = drawBowl(sk, cx, cy, rx, ry, depth, outline=2.2,
                                wallPattern=bluePorcelainWall(sk, cx, cy, rx, ry, depth, 0.55),
                                insidePattern=bluePorcelainInside(sk, cx, cy, rx, ry, 0.55))
    food = union(inside, above(sk, cy))
    rng = random.Random(90)
    lay, d = sk.layer()
    sauce, glints = rojakHeap(sk, lay, d, rng, cx, cy + 2, rx * 0.72, ry * 0.66, 24, 48, (12, 16), 1.2)
    sk.clip(lay, food)
    sauce = paintSauce(sk, inter(sauce, food), food, 0.7)
    paintGlints(sk, sauce, glints, 0.7)
    lay, d = sk.layer()
    sprinklePeanuts(sk, d, rng, cx, cy + 2, rx * 0.64, ry * 0.58, 24, 60, 0.8)
    sk.clip(lay, food)
    sk.over(frontLip)
    saveImage(sk.result(), "food/dish_rojak.png")


def makeDishCkt():
    sk = Sketch(150, 110)
    drawOvalPlate(sk, 75, 60, 68, 38, thick=4, outline=2.2)
    drawCharKueyTeow(sk, 75, 64, 48, 22, 20, seed=52, scale=0.55, prawns=3)
    saveImage(sk.result(), "food/dish_ckt.png")


def makeDishTehTarik():
    """A tall glass mug of frothy teh tarik on a small steel tray."""
    sk = Sketch(150, 110)
    # round steel tray
    trayEdge = sk.ellipse(18, 73, 132, 101)
    trayTop = sk.ellipse(18, 70, 132, 98)
    sk.shadow(shift(trayEdge, 0, 2), 2, 110)
    sk.part(union(trayEdge, trayTop), (176, 184, 196), dark=(110, 118, 134), light=(226, 232, 240), soft=2,
            outline=2.2, darkAt=0.1, lightAt=0.2)
    sk.paint(trayTop, (196, 204, 214))
    inner = sk.ellipse(25, 74, 125, 94)
    sk.paint(inner, (214, 220, 228))
    sk.paint(soften(minus(inner, shift(inner, 2, 2)), 0.8), (150, 158, 172), 200)
    sk.paint(soften(sk.stroke(arc(75, 84, 54, 12.5, 200, 250), 1.2), 0.3), (255, 255, 255), 220)
    # glass mug: liquid first, then the glass walls on top
    gx0, gx1, top, bottom = 56, 94, 22, 88
    glassBody = union(sk.rrect(gx0, top, gx1, bottom, 3), sk.ellipse(gx0, bottom - 5, gx1, bottom + 4))
    sk.shadow(shift(glassBody, 5, 2), 2.5, 90)
    handle = minus(sk.rrect(gx1 - 4, 38, gx1 + 16, 74, 9), sk.rrect(gx1 + 1, 45, gx1 + 9, 67, 5))
    sk.part(handle, (226, 236, 240), dark=(168, 184, 194), light=(255, 255, 255), soft=1.4, outline=2.0, alpha=235)
    sk.outline(glassBody, 2.2)
    tea = sk.gradient([(0, (214, 158, 104)), (0.55, (196, 136, 84)), (1, (160, 100, 58))], (0, 36), (0, 88))
    liquid = sk.rrect(gx0 + 2, 34, gx1 - 2, bottom - 3, 2)
    sk.clip(tea.convert("RGBA"), liquid)
    foam = union(sk.rrect(gx0 + 2, 26, gx1 - 2, 38, 2), sk.ellipse(gx0 + 2, 22, gx1 - 2, 30))
    sk.part(foam, (248, 232, 200), dark=(222, 196, 156), light=(255, 248, 232), soft=1.6, outline=0,
            darkAt=0.15, lightAt=0.12)
    lay, d = sk.layer()
    rng = random.Random(8)
    for i in range(10):                                  # tiny foam bubbles
        x, y = rng.uniform(gx0 + 5, gx1 - 5), rng.uniform(24, 34)
        r = rng.uniform(0.8, 1.6)
        ak.ellipse(d, x - r, y - r, x + r, y + r, None, (214, 184, 140, 200), 0.6)
    sk.clip(lay, foam)
    sk.paint(soften(sk.rrect(gx0 + 2, 36, gx1 - 2, 40, 1), 1.2), (230, 196, 150), 180)
    # glass: frosty rim, bright left edge streak, thick base
    sk.paint(minus(sk.ellipse(gx0, 19, gx1, 27), sk.ellipse(gx0 + 2, 20.5, gx1 - 2, 25.5)), (255, 255, 255), 200)
    sk.paint(soften(sk.rrect(gx0 + 4, 30, gx0 + 8, 82, 2), 0.5), (255, 255, 255), 150)
    sk.paint(soften(sk.rrect(gx1 - 7, 32, gx1 - 5, 80, 1), 0.4), (255, 255, 255), 110)
    sk.paint(soften(sk.ellipse(gx0 + 1, bottom - 7, gx1 - 1, bottom + 2), 0.8), (236, 244, 248), 150)
    # two soft steam curls
    for sx, ph in ((68, 0.0), (80, 1.8)):
        pts = [(sx + 3 * math.sin(ph + k * 0.7), 18 - k * 2.2) for k in range(7)]
        sk.paint(soften(sk.stroke(pts, 1.6), 0.8), (255, 255, 255), 110)
    saveImage(sk.result(), "food/dish_tehtarik.png")


def makeDishNasiLemak():
    """Nasi lemak on a banana leaf: coconut rice, sambal, boiled egg,
    cucumber, fried peanuts and ikan bilis (anchovies)."""
    sk = Sketch(150, 110)
    drawOvalPlate(sk, 75, 60, 66, 38, thick=4, outline=2.2, band=(196, 150, 70))
    # banana leaf with a midrib and parallel veins
    leafPts = chain(bez((16, 62), (22, 36), (70, 26), (126, 34)),
                    bez((126, 34), (140, 50), (136, 76), (118, 88)),
                    bez((118, 88), (80, 98), (34, 92), (16, 62)))
    leaf = sk.poly(leafPts)
    leafCol = sk.gradient([(0, (96, 172, 76)), (1, (54, 128, 56))], (20, 30), (130, 95))
    sk.part(leaf, leafCol, dark=(40, 100, 46), light=(140, 204, 110), soft=2, outline=1.8, darkAt=0.14, lightAt=0.14)
    lay, d = sk.layer()
    ak.line(d, bez((22, 80), (60, 60), (100, 46), (132, 40), steps=20), (170, 220, 140, 220), 1.4)
    for k in range(-6, 8):
        x = 30 + k * 10
        ak.line(d, [(x, 90), (x + 22, 26)], (70, 140, 60, 140), 0.8)
    sk.clip(lay, shrink(leaf, 1))
    # coconut rice dome
    rice = union(sk.ellipse(38, 30, 84, 74), sk.ellipse(34, 50, 88, 76))
    sk.shadow(shift(rice, 2, 3), 2, 110)
    sk.part(rice, (250, 250, 244), dark=(206, 210, 214), light=(255, 255, 255), soft=4, outline=2.0,
            darkAt=0.12, lightAt=0.16, bulge=3.0)
    lay, d = sk.layer()
    rng = random.Random(4)
    for i in range(60):
        x, y = rng.uniform(38, 86), rng.uniform(32, 74)
        ak.ellipse(d, x - 1.3, y - 0.6, x + 1.3, y + 0.6, (214, 216, 212, 170) if rng.random() < 0.5 else (255, 255, 255, 220))
    sk.clip(lay, shrink(rice, 1.5))
    # sambal on the right of the rice
    sambal = union(sk.ellipse(76, 36, 104, 56), sk.ellipse(86, 30, 106, 50))
    sk.part(sambal, (200, 42, 28), dark=(130, 20, 24), light=(242, 92, 60), soft=2.4, outline=2.0,
            spec=(255, 214, 190), specAt=0.3)
    lay, d = sk.layer()
    for i in range(8):
        x, y = rng.uniform(80, 104), rng.uniform(34, 54)
        ak.ellipse(d, x - 1.8, y - 0.8, x + 1.8, y + 0.8, (120, 22, 22, 200))
    sk.clip(lay, shrink(sambal, 1))
    # half boiled egg
    white = sk.ellipse(84, 56, 114, 78)
    sk.part(white, (252, 252, 246), dark=(214, 214, 206), light=(255, 255, 255), soft=2, outline=2.0)
    yolk = sk.ellipse(92, 60, 108, 72)
    sk.part(yolk, (255, 176, 36), dark=(230, 128, 24), light=(255, 214, 110), soft=1.6, outline=1.2)
    # cucumber slices
    for (x, y) in ((24, 64), (32, 70)):
        sl = sk.ellipse(x - 9, y - 5.5, x + 9, y + 5.5)
        sk.part(sl, (206, 236, 164), dark=(160, 200, 120), light=(232, 250, 206), soft=1, outline=1.6)
        sk.paint(minus(sl, shrink(sl, 1.6)), (64, 138, 58), 255)
        for k in range(4):
            a = math.radians(k * 90 + 45)
            sk.paint(sk.ellipse(x + 3 * math.cos(a) - 0.8, y + 2 * math.sin(a) - 0.5,
                                x + 3 * math.cos(a) + 0.8, y + 2 * math.sin(a) + 0.5), (238, 250, 210), 255)
    # fried peanuts and ikan bilis at the front
    lay, d = sk.layer()
    for i in range(12):
        x, y = rng.uniform(46, 76), rng.uniform(76, 86)
        ak.ellipse(d, x - 2.6, y - 1.8, x + 2.6, y + 1.8, rgba(OUTLINE))
        ak.ellipse(d, x - 2.0, y - 1.3, x + 2.0, y + 1.3, (212, 150, 84, 255))
        ak.ellipse(d, x - 1.4, y - 1.0, x, y - 0.1, (240, 196, 140, 255))
    for i in range(9):
        x, y = rng.uniform(70, 102), rng.uniform(78, 88)
        ang = rng.uniform(-0.6, 0.6)
        pts = [(x - 5 * math.cos(ang), y - 5 * math.sin(ang)), (x + 5 * math.cos(ang), y + 5 * math.sin(ang))]
        ak.line(d, pts, rgba(OUTLINE), 3.0)
        ak.line(d, pts, (214, 140, 64, 255), 1.8)
        ak.ellipse(d, pts[0][0] - 1.2, pts[0][1] - 1.2, pts[0][0] + 1.2, pts[0][1] + 1.2, (110, 64, 30, 255))
    sk.over(lay)
    sk.shine(sk.ellipse(48, 36, 60, 42), rice, (255, 255, 255), 230, 0.6)
    saveImage(sk.result(), "food/dish_nasilemak.png")


def drawSkewer(sk, d, x0, y0, x1, y1, rng, chunks=4, scale=1.0):
    """Bamboo stick with grilled, glazed meat chunks: irregular rounded
    pieces, caramel glaze on the lit side, charred spots on the shaded side."""
    ak.line(d, [(x0, y0), (x1, y1)], rgba(OUTLINE), 3.4 * scale)
    ak.line(d, [(x0, y0), (x1, y1)], (236, 210, 156, 255), 2.0 * scale)
    dx, dy = x1 - x0, y1 - y0
    length = math.hypot(dx, dy)
    ux, uy = dx / length, dy / length
    nx, ny = -uy, ux
    for k in range(chunks):
        t = 0.36 + 0.58 * k / max(1, chunks - 1)
        cx, cy = x0 + dx * t, y0 + dy * t
        pts = []
        for j in range(9):
            a = j * math.tau / 9
            r = rng.uniform(0.8, 1.15)
            px, py = math.cos(a) * 5.6 * r * scale, math.sin(a) * 6.4 * r * scale
            pts.append((cx + ux * px + nx * py, cy + uy * px + ny * py))
        d.polygon(ak.pts(pts), fill=(134, 58, 26, 255), outline=rgba(OUTLINE), width=max(1, int(1.4 * scale * SS)))
        inner = [(cx + (px - cx) * 0.6 - 1.2 * scale, cy + (py - cy) * 0.6 - 1.2 * scale) for px, py in pts]
        d.polygon(ak.pts(inner), fill=(184, 94, 38, 255))
        ak.line(d, [(cx - 3 * scale, cy - 2.6 * scale), (cx + 1 * scale, cy - 3.4 * scale)], (246, 170, 96, 255), 1.3 * scale)
        for j in range(3):
            sx = cx + rng.uniform(0.5, 3.5) * scale
            sy = cy + rng.uniform(0.5, 3.0) * scale
            ak.ellipse(d, sx - 1.3 * scale, sy - 0.9 * scale, sx + 1.3 * scale, sy + 0.9 * scale, (86, 34, 16, 255))


def makeDishSatay():
    """Satay: grilled skewers, a bowl of peanut sauce, ketupat rice cubes,
    cucumber and red onion."""
    sk = Sketch(150, 110)
    drawOvalPlate(sk, 75, 60, 68, 38, thick=4, outline=2.2)
    # peanut sauce bowl at the back right
    bx, by = 108, 40
    inside, frontLip = drawBowl(sk, bx, by, 22, 8.5, 13, outline=1.8, shadow=True)
    sauce = shrink(inside, 1.2)
    sk.part(sauce, (200, 112, 48), dark=(160, 80, 34), light=(232, 156, 80), soft=1.5, outline=0,
            spec=(255, 214, 160), specAt=0.3)
    lay, d = sk.layer()
    rng = random.Random(12)
    for i in range(14):
        x, y = bx + rng.uniform(-15, 15), by + rng.uniform(-4, 4)
        ak.ellipse(d, x - 1.2, y - 0.8, x + 1.2, y + 0.8, (150, 80, 36, 230))
    sk.clip(lay, sauce)
    sk.over(frontLip)
    # ketupat, cucumber and onion at the front right (drawn back to front)
    lay, d = sk.layer()
    for (x, y, a, kind, size) in ((110, 64, 10, "ketupat", 14), (126, 62, 40, "cucumber", 10), (96, 76, 30, "onion", 9),
                                  (122, 76, 25, "ketupat", 14), (108, 84, 60, "cucumber", 10)):
        drawChunk(d, x, y, size, a, kind, 1.4, squash=0.6, height=0.6)
    sk.shadow(shift(lay.getchannel("A"), 2, 3), 2, 100)
    sk.over(lay)
    # skewers fanned out, resting across the plate
    lay, d = sk.layer()
    for k in range(4):
        y0 = 90 - k * 9
        drawSkewer(sk, d, 12 + k * 4, y0, 92 + k * 3, y0 - 38 + k * 3, rng, 4, 1.0)
    sk.shadow(shift(lay.getchannel("A"), 2, 3), 2, 110)
    sk.over(lay)
    saveImage(sk.result(), "food/dish_satay.png")


def makeDishRotiCanai():
    """Roti canai: flaky folded flatbread with dhal and fish curry."""
    sk = Sketch(150, 110)
    drawOvalPlate(sk, 75, 60, 68, 38, thick=4, outline=2.2, band=(214, 96, 60))
    # two small curry bowls at the back
    for (bx, by, col, dark) in ((50, 38, (236, 178, 58), (196, 128, 36)), (98, 36, (206, 92, 42), (150, 54, 30))):
        inside, frontLip = drawBowl(sk, bx, by, 22, 8.5, 13, outline=1.8)
        curry = shrink(inside, 1.2)
        sk.part(curry, col, dark=dark, light=lightOf(col, 0.35), soft=1.5, outline=0, spec=(255, 236, 200), specAt=0.3)
        lay, d = sk.layer()
        rng = random.Random(bx)
        for i in range(6):
            x, y = bx + rng.uniform(-14, 14), by + rng.uniform(-3.5, 3.5)
            ak.ellipse(d, x - 1.5, y - 0.9, x + 1.5, y + 0.9, (190, 50, 20, 200))
        sk.clip(lay, curry)
        sk.over(frontLip)
    # two pieces of folded roti overlapping at the front
    rng = random.Random(31)
    for (cx, cy, rx, ry, rot) in ((58, 70, 34, 17, -8), (90, 74, 32, 16, 10)):
        pts = []
        for k in range(24):
            a = math.radians(k * 15)
            wob = 1.0 + 0.07 * math.sin(k * 2.3 + cx) + rng.uniform(-0.04, 0.04)
            x, y = rx * wob * math.cos(a), ry * wob * math.sin(a)
            r = math.radians(rot)
            pts.append((cx + x * math.cos(r) - y * math.sin(r), cy + x * math.sin(r) + y * math.cos(r)))
        roti = sk.poly(pts)
        edge = sk.poly([(p[0], p[1] + 3) for p in pts])
        sk.shadow(shift(roti, 2, 4), 2, 110)
        sk.part(union(roti, edge), (214, 154, 78), dark=(170, 104, 48), light=(236, 190, 118), soft=2, outline=2.0)
        sk.part(roti, (240, 200, 126), dark=(214, 162, 88), light=(255, 230, 170), soft=3.5, outline=0,
                darkAt=0.1, lightAt=0.12)
        lay, d = sk.layer()
        for k in range(10):                                # browned crispy patches
            x, y = cx + rng.uniform(-rx * 0.7, rx * 0.7), cy + rng.uniform(-ry * 0.6, ry * 0.6)
            r = rng.uniform(2, 4)
            ak.ellipse(d, x - r, y - r * 0.6, x + r, y + r * 0.6, (196, 128, 56, 150))
        for k in range(4):                                 # flaky layer lines (swirls)
            s = 0.3 + k * 0.17
            ak.line(d, arc(cx, cy, rx * s, ry * s, 190 + k * 20, 330 + k * 12, 16), (170, 108, 50, 170), 1.0)
            ak.line(d, arc(cx - 1, cy - 1.2, rx * s, ry * s, 190 + k * 20, 330 + k * 12, 16), (255, 236, 186, 200), 0.8)
        sk.clip(lay.filter(ImageFilter.GaussianBlur(SS * 0.4)), shrink(roti, 1))
        sk.shine(sk.ellipse(cx - rx * 0.6, cy - ry * 0.6, cx - rx * 0.2, cy - ry * 0.35), roti, (255, 246, 214), 200, 0.6)
    saveImage(sk.result(), "food/dish_roticanai.png")


# ======================================================================
# PROPS - Scene 2: the wok (one drawing, cut into a back and a front layer)
# ======================================================================
WOK_FRONT_TOP = 91      # wok_front.png starts 91 px below the top of wok_back.png
                        # (Arcade: back centre y=255, front centre y=214 ->
                        #  back top = 720-255-105 = 360, front top = 720-214-55 = 451)


def stopsColour(t, stops):
    """Colour at position t along a list of (t, colour) stops (linear interpolation)."""
    for i in range(len(stops) - 1):
        t0, c0 = stops[i]
        t1, c1 = stops[i + 1]
        if t <= t1:
            return mixRgb(c0, c1, (t - t0) / max(t1 - t0, 0.0001))
    return stops[-1][1]


def ringImage(sk, cx, cy, rx, tilt, depth, radiusFn, stops, backToFront):
    """Colours a bowl surface with rings (heat tint): one filled ellipse per
    depth t. Painted in order, every pixel ends up with the colour of the
    ring that is really visible there (painter's algorithm):
      inside wall  -> paint from the rim down (t = 0 -> 1)
      outside wall -> paint from the bottom up (t = 1 -> 0)"""
    img = Image.new("RGB", sk.size, (0, 0, 0))
    d = ImageDraw.Draw(img)
    n = 200
    order = range(n + 1) if backToFront else range(n, -1, -1)
    for i in order:
        t = i / float(n)
        r = radiusFn(t)
        y = cy + depth * t
        d.ellipse(ak.box(cx - r, y - r * tilt, cx + r, y + r * tilt), fill=stopsColour(t, stops))
    return img


def makeWok():
    """Carbon-steel wok in 3/4 view with blue-bronze heat tint.
    The full wok is drawn once. wok_back.png = the whole wok; wok_front.png =
    only the front lip and outer belly, cut from the SAME pixels, so the two
    layers line up exactly and food drawn between them sits inside the wok."""
    W, H = 460, 210
    sk = Sketch(W, H)
    cx, cy, rx, ry, depth = 230, 78, 196, 58, 92
    tilt = ry / float(rx)
    lip = 6.0

    def outerRadius(t):
        return rx * math.sqrt(max(0.0, 1.0 - 0.93 * t * t))

    def innerRadius(t):
        return (rx - lip) * math.sqrt(max(0.0, 1.0 - 0.93 * t * t))

    body = sk.revolve(cx, cy, depth, outerRadius, tilt)
    rimOuter = sk.ellipse(cx - rx, cy - ry, cx + rx, cy + ry)
    opening = sk.ellipse(cx - rx + lip, cy - ry + lip * 0.7, cx + rx - lip, cy + ry - lip * 0.7)

    # soft shadow on the stove under the wok
    sk.shadow(sk.ellipse(cx - 120, cy + depth - 6, cx + 120, cy + depth + 22), blurPx=8, alpha=120)

    # 1) outer belly: dark steel, blue-purple then bronze heat rings, sooty bottom
    # (the lower half of the belly is squeezed into a few pixels by the
    #  perspective, so the tint rings start fairly high up)
    outerStops = [(0.0, (74, 78, 90)), (0.18, (52, 54, 62)), (0.34, (50, 60, 100)), (0.46, (96, 76, 132)),
                  (0.58, (164, 116, 60)), (0.72, (110, 74, 42)), (0.86, (44, 36, 32)), (1.0, (30, 28, 26))]
    outerImg = ringImage(sk, cx, cy, rx, tilt, depth, outerRadius, outerStops, False)
    sk.part(union(body, rimOuter), outerImg, dark=0.6, light=1.32, soft=14, outline=4.0,
            darkAt=0.1, lightAt=0.12, bulge=3.0, spec=1.9, specAt=0.3)

    # 2) inside: seasoned black steel with the heat-tint rings of a well used wok
    innerStops = [(0.0, (70, 72, 84)), (0.2, (46, 46, 54)), (0.42, (40, 44, 66)), (0.53, (68, 70, 126)),
                  (0.62, (114, 84, 112)), (0.7, (152, 110, 58)), (0.8, (106, 74, 42)), (0.9, (52, 40, 32)),
                  (1.0, (60, 46, 36))]
    innerImg = ringImage(sk, cx, cy, rx - lip, tilt, depth - 4, innerRadius, innerStops, True)
    sk.clip(innerImg.convert("RGBA"), opening)
    # concave lighting: the left inner wall is turned away from the lamp,
    # the right inner wall faces it and shows an oily specular reflection
    sk.paint(inter(soften(sk.ellipse(cx - rx, cy - ry, cx - 30, cy + ry * 0.7), 26), opening), (8, 8, 14), 110)
    sk.paint(inter(soften(sk.ellipse(cx - rx * 0.3, cy - ry * 1.1, cx + rx * 0.5, cy - ry * 0.35), 16), opening),
             (8, 8, 14), 70)
    glint = sk.stroke(arc(cx + 8, cy + 6, rx * 0.8, ry * 0.72, 292, 338, 30), 7)
    sk.paint(inter(soften(glint, 4), opening), (196, 212, 236), 95)
    glint2 = sk.stroke(arc(cx + 8, cy + 6, rx * 0.8, ry * 0.72, 300, 322, 20), 2.2)
    sk.paint(inter(soften(glint2, 1), opening), (236, 244, 255), 150)
    sk.paint(inter(soften(sk.ellipse(cx - 40, cy + 30, cx + 70, cy + 52), 10), opening), (190, 170, 150), 45)

    # 3) the rim: a thin bright steel lip
    ring = minus(rimOuter, opening)
    sk.paint(ring, (126, 132, 146))
    sk.paint(inter(ring, below(sk, cy + 20)), (150, 156, 170), 255)
    sk.paint(soften(sk.stroke(arc(cx, cy, rx - lip * 0.5, ry - lip * 0.35, 186, 262, 40), 2.6), 0.6),
             (236, 242, 250), 235)
    sk.paint(soften(sk.stroke(arc(cx, cy, rx - lip * 0.5, ry - lip * 0.35, 60, 118, 30), 1.6), 0.5),
             (214, 222, 234), 200)
    sk.paint(minus(grow(opening, 1.4), opening), (26, 26, 32), 220)

    # 4) two side handles: steel loops with wooden grips and riveted plates
    for side in (-1, 1):
        ax = cx + side * (rx - 10)
        plate = sk.rrect(ax - 9, cy + 6, ax + 9, cy + 30, 5)
        sk.part(plate, (70, 74, 86), dark=(40, 42, 52), light=(130, 136, 150), soft=2, outline=2.4)
        for ry2 in (cy + 12, cy + 24):
            sk.part(sk.circle(ax, ry2, 2.4), (170, 176, 188), dark=(96, 100, 112), light=(230, 234, 242),
                    soft=0.8, outline=1.2)
        outX = cx + side * (rx + 8)
        loopPts = chain(bez((ax + side * 6, cy + 10), (ax + side * 18, cy + 6), (outX, cy + 6), steps=12),
                        bez((outX, cy + 6), (outX + side * 9, cy + 6), (outX + side * 9, cy + 26), (outX, cy + 26), steps=16),
                        bez((outX, cy + 26), (ax + side * 18, cy + 26), (ax + side * 6, cy + 24), steps=12))
        loop = sk.tube(loopPts, 3.2)
        sk.part(loop, (150, 156, 170), dark=(84, 88, 100), light=(222, 228, 238), soft=1.4, outline=2.4,
                spec=(255, 255, 255), specAt=0.3)
        gripPts = bez((outX - side * 4, cy + 6), (outX + side * 9, cy + 5), (outX + side * 9, cy + 27), (outX - side * 4, cy + 26), steps=20)
        grip = sk.tube(gripPts, 5.4)
        sk.part(grip, (136, 82, 44), dark=(86, 48, 28), light=(196, 132, 80), soft=2, outline=2.6,
                spec=(236, 186, 130), specAt=0.3)

    full = sk.result()
    saveImage(full, "props/wok_back.png")

    # front layer = rim lip + outer belly below the rim centre line (not the opening)
    # Only pixels that are fully opaque in wok_back are copied: a half
    # transparent pixel (soft shadow, anti-aliased outer edge) drawn twice
    # would look darker. The inner edge of the lip keeps its smooth edge
    # from the (downsized) mask, so food disappears smoothly behind it.
    frontMask = inter(minus(grow(union(body, rimOuter), 5), opening), below(sk, cy))
    frontMask = frontMask.resize((W, H), Image.LANCZOS)
    opaque = full.getchannel("A").point(lambda a: 255 if a >= 250 else 0)
    front = full.copy()
    front.putalpha(ImageChops.multiply(opaque, frontMask))
    saveImage(front.crop((0, WOK_FRONT_TOP, W, WOK_FRONT_TOP + 110)), "props/wok_front.png")


# ======================================================================
# PROPS - Scene 2: ingredient bowls on the counter + soy sauce bottle
# ======================================================================
def cylinderShade(sk, mask, x0, x1, dark=110, light=90, lightAt=0.3, darkColor=(20, 14, 30)):
    """Lighting of a standing cylinder (can, bottle, jug, glass): every
    column gets its own brightness. The surface normal turns from left to
    right across the cylinder, so Lambert's rule gives a bright band on the
    left (facing the lamp) and a dark side on the right."""
    w, h = sk.size
    xs = np.arange(w, dtype=np.float32) / SS
    u = np.clip((xs - x0) / max(1.0, (x1 - x0)), 0.0, 1.0)      # 0 = left edge, 1 = right edge
    angle = (u - 0.5) * math.pi                                   # surface normal angle
    lightAngle = (lightAt - 0.5) * math.pi
    lambert = np.cos(angle - lightAngle)
    darkA = np.clip((0.55 - lambert) / 0.55, 0, 1) ** 1.3 * dark
    lightA = np.clip((lambert - 0.8) / 0.2, 0, 1) ** 1.5 * light
    lay = np.zeros((h, w, 4), np.uint8)
    lay[..., 0], lay[..., 1], lay[..., 2] = darkColor
    lay[..., 3] = np.broadcast_to(darkA.astype(np.uint8)[None, :], (h, w))
    sk.clip(Image.fromarray(lay, "RGBA"), mask)
    lay = np.zeros((h, w, 4), np.uint8)
    lay[..., 0:3] = 255
    lay[..., 3] = np.broadcast_to(lightA.astype(np.uint8)[None, :], (h, w))
    sk.clip(Image.fromarray(lay, "RGBA"), mask)


def bandWall(sk, cx, cy, rx, ry, depth, color=(44, 150, 130)):
    """Simple two-line coloured band under the rim (hawker melamine bowl)."""
    tilt = ry / float(rx)

    def paintWall(wallMask):
        lay, d = sk.layer()
        for off, w in ((4.5, 2.2), (8.0, 1.0)):
            r = rx * bowlRadius(off / depth)
            ak.ellipse(d, cx - r, cy + off - r * tilt, cx + r, cy + off + r * tilt, None, rgba(color), w)
        sk.clip(lay, shrink(wallMask, 0.6))
    return paintWall


def prepBowl():
    """Every counter bowl uses the same white bowl with a teal band.
    Returns (sketch, food clip mask, front lip layer, bowl geometry)."""
    sk = Sketch(130, 100)
    cx, cy, rx, ry, depth = 65, 50, 56, 19, 30
    inside, frontLip = drawBowl(sk, cx, cy, rx, ry, depth, outline=2.4,
                                wallPattern=bandWall(sk, cx, cy, rx, ry, depth))
    food = union(inside, above(sk, cy))
    return sk, food, frontLip, (cx, cy, rx, ry)


def makeBowlGarlic():
    """Peeled garlic cloves heaped in the bowl with one whole bulb on top
    (whole garlic is much easier to recognise at a small size than mince)."""
    sk, food, frontLip, (cx, cy, rx, ry) = prepBowl()
    lay, d = sk.layer()
    fruitMass(d, cx, cy + 2, rx * 0.84, ry * 0.8, 16, (226, 204, 150))
    sk.clip(lay, food)
    cloves = [(38, 44, 70), (56, 46, -60), (76, 46, 100), (94, 42, -80), (46, 34, 20), (66, 36, -15),
              (86, 32, 40), (56, 26, -30), (74, 24, 60)]
    for (x, y, rot) in cloves:
        drawGarlicClove(sk, x, y, 20, rot)
    # whole bulb: round, segmented, purple streaks, little stem and roots
    bulb = union(sk.ellipse(78, 12, 108, 40), sk.poly([(89, 14), (93, 5), (97, 14)]))
    sk.part(bulb, (248, 242, 236), dark=(206, 192, 196), light=(255, 255, 255), soft=3, outline=1.8)
    lay, d = sk.layer()
    for k in (-8, -3, 2, 7):
        ak.line(d, bez((93 + k * 0.4, 14), (93 + k * 1.6, 24), (93 + k * 0.8, 39), steps=12), (176, 150, 170, 200), 0.9)
    for k in (-6, 5):
        ak.line(d, bez((93 + k * 0.5, 16), (93 + k * 1.5, 26), (93 + k * 0.9, 37), steps=12), (196, 120, 170, 170), 1.6)
    sk.clip(lay, shrink(bulb, 1))
    sk.over(frontLip)
    saveImage(sk.result(), "props/bowl_garlic.png")


def makeBowlPrawns():
    sk, food, frontLip, (cx, cy, rx, ry) = prepBowl()
    # heap of prawns drawn back to front (painter's algorithm)
    for (x, y, rot) in ((38, 42, 200), (64, 44, 250), (90, 42, 330), (50, 32, 120), (78, 32, 20),
                        (64, 20, 280), (44, 20, 170), (86, 18, 60)):
        drawPrawn(sk, x, y, 28, rotation=rot, outlineW=1.6)
    sk.over(frontLip)
    saveImage(sk.result(), "props/bowl_prawns.png")


def makeBowlNoodles():
    sk, food, frontLip, (cx, cy, rx, ry) = prepBowl()
    rng = random.Random(9)
    lay, d = sk.layer()
    fruitMass(d, cx, cy + 2, rx * 0.84, ry * 0.8, 32, (214, 202, 172))
    ribbons = []
    for i in range(84):
        a0 = rng.uniform(0, math.tau)
        r0 = math.sqrt(rng.uniform(0, 1)) * 0.8
        ang = rng.uniform(-0.6, 0.6)
        phase = rng.uniform(0, math.tau)
        pts = []
        for k in range(12):
            u = (k / 11.0 - 0.5) * 0.9
            ux = r0 * math.cos(a0) + u * math.cos(ang)
            uy = r0 * math.sin(a0) + u * math.sin(ang) + 0.07 * math.sin(k * 1.1 + phase)
            rr2 = min(1.0, ux * ux + uy * uy)
            pts.append((cx + rx * 0.8 * ux, cy + 2 + ry * 0.75 * uy - 34 * (1.0 - rr2)))
        widths = [9.0 * (0.5 + 0.5 * abs(math.cos(k * 0.5 + phase))) for k in range(12)]
        ribbons.append((sum(p[1] for p in pts) / 12.0, pts, widths,
                        rng.choice([(250, 242, 222), (240, 228, 200), (255, 250, 236)])))
    ribbons.sort(key=lambda r: r[0])
    for _, pts, widths, col in ribbons:
        drawFlatRibbon(d, pts, widths, col, 1.1)
    sk.clip(lay, food)
    sk.over(frontLip)
    saveImage(sk.result(), "props/bowl_noodles.png")


def makeBowlEgg():
    """Two cracked raw eggs (glossy whites + yolks) and the empty shells."""
    sk, food, frontLip, (cx, cy, rx, ry) = prepBowl()
    whites = union(sk.ellipse(cx - 44, cy - 13, cx + 44, cy + 15), sk.ellipse(cx - 30, cy - 17, cx + 20, cy + 6))
    sk.part(inter(whites, food), (246, 244, 236), dark=(206, 206, 196), light=(255, 255, 255), soft=3, outline=0,
            spec=(255, 255, 255), specAt=0.25)
    for (x, y) in ((50, 49), (79, 46)):
        yolk = sk.ellipse(x - 11, y - 9, x + 11, y + 7)
        sk.shadow(shift(yolk, 1.5, 2.5), 1.5, 90, color=(180, 150, 90))
        sk.part(yolk, (255, 168, 26), dark=(232, 118, 18), light=(255, 210, 90), soft=2.4, outline=1.6,
                spec=(255, 248, 220), specAt=0.3)
    sk.over(frontLip)
    # cracked shell halves leaning on the right rim
    for (x, y, rot) in ((104, 36, 30), (22, 38, -35)):
        pts = []
        for k in range(13):
            a = math.radians(k * 15)
            px, py = 11 * math.cos(a), 8 * math.sin(a) + 1.5
            if k in (0, 3, 6, 9, 12):
                py -= 2.0                                      # jagged cracked edge
            rr = math.radians(rot)
            pts.append((x + px * math.cos(rr) - py * math.sin(rr), y + px * math.sin(rr) + py * math.cos(rr)))
        shell = sk.poly(pts)
        sk.part(shell, (246, 226, 196), dark=(206, 176, 140), light=(255, 246, 228), soft=1.6, outline=1.6)
    saveImage(sk.result(), "props/bowl_egg.png")


def makeBowlSprouts():
    sk, food, frontLip, (cx, cy, rx, ry) = prepBowl()
    rng = random.Random(21)
    lay, d = sk.layer()
    fruitMass(d, cx, cy + 2, rx * 0.84, ry * 0.8, 26, (226, 228, 204))
    sprouts = []
    for i in range(62):
        a = rng.uniform(0, math.tau)
        rr = math.sqrt(rng.uniform(0, 1)) * 0.85
        x = cx + rx * 0.8 * rr * math.cos(a)
        y = cy + 2 + ry * 0.75 * rr * math.sin(a) - 26 * (1 - rr * rr)
        sprouts.append((y, x, rng.uniform(-0.8, 0.8), rng.random() < 0.2))
    sprouts.sort()
    for y, x, ang, chive in sprouts:
        l = 9
        pts = [(x - l * math.cos(ang), y - l * math.sin(ang)), (x, y + 2), (x + l * math.cos(ang), y + l * math.sin(ang))]
        if chive:
            ak.line(d, pts, (30, 64, 28, 255), 3.6)
            ak.line(d, pts, (92, 178, 66, 255), 2.2)
            continue
        ak.line(d, pts, rgba(OUTLINE, 235), 3.8)
        ak.line(d, pts, (250, 248, 232, 255), 2.4)
        ak.ellipse(d, pts[0][0] - 2.6, pts[0][1] - 1.9, pts[0][0] + 2.6, pts[0][1] + 1.9, rgba(OUTLINE))
        ak.ellipse(d, pts[0][0] - 1.8, pts[0][1] - 1.2, pts[0][0] + 1.8, pts[0][1] + 1.2, (238, 222, 104, 255))
    sk.clip(lay, food)
    sk.over(frontLip)
    saveImage(sk.result(), "props/bowl_sprouts.png")


def makeBottleSoy():
    """Dark soy sauce bottle: dark glass, red cap, cream label."""
    sk = Sketch(70, 150)
    cx = 35
    profile = chain([(cx - 7, 24), (cx - 7, 44)], bez((cx - 7, 44), (cx - 8, 56), (cx - 26, 58), (cx - 26, 72)),
                    [(cx - 26, 72), (cx - 26, 136)], bez((cx - 26, 136), (cx - 26, 143), (cx - 20, 144), (cx, 144)))
    right = [(2 * cx - x, y) for x, y in reversed(profile)]
    bottle = sk.poly(profile + right)
    sk.shadow(sk.ellipse(cx - 28, 134, cx + 28, 145), 2.5, 110)
    glass = sk.gradient([(0, (70, 40, 26)), (0.5, (44, 24, 16)), (1, (26, 14, 10))], (0, 30), (0, 144))
    sk.part(bottle, glass, dark=0.7, light=1.35, soft=3, outline=2.4, darkAt=0.1, lightAt=0.14)
    cylinderShade(sk, bottle, cx - 26, cx + 26, dark=90, light=70)
    # label with a red band and "DARK SOY / SAUCE"
    label = sk.rrect(cx - 26, 82, cx + 26, 124, 2)
    sk.paint(label, (244, 232, 204))
    sk.paint(sk.rrect(cx - 26, 84, cx + 26, 92, 0), (196, 36, 40))
    sk.paint(sk.rrect(cx - 26, 116, cx + 26, 121, 0), (196, 36, 40))
    sk.paint(sk.text(cx, 101.5, "SAUCE", "sans_bold", 10.5), (120, 24, 28))
    sk.paint(sk.text(cx, 88.3, "DARK SOY", "sans_bold", 5.5), (255, 236, 200))
    lay, d = sk.layer()
    for k in range(3):                                   # three little soybeans
        bx = cx - 8 + k * 8
        ak.ellipse(d, bx - 2.6, 110 - 2, bx + 2.6, 110 + 2, (170, 130, 60, 255))
    sk.over(lay)
    cylinderShade(sk, label, cx - 26, cx + 26, dark=80, light=40)
    # glass highlights
    sk.paint(soften(sk.rrect(cx - 19, 64, cx - 15, 134, 2), 0.5), (255, 236, 220), 130)
    sk.paint(soften(sk.rrect(cx - 5, 28, cx - 3, 44, 1), 0.4), (255, 236, 220), 110)
    # red screw cap
    cap = sk.rrect(cx - 9, 8, cx + 9, 26, 3)
    sk.part(cap, (214, 40, 40), dark=(150, 20, 30), light=(250, 110, 96), soft=1.6, outline=2.2)
    cylinderShade(sk, cap, cx - 9, cx + 9, dark=70, light=90)
    lay, d = sk.layer()
    for k in range(5):
        x = cx - 6 + k * 3
        ak.line(d, [(x, 12), (x, 24)], (150, 20, 30, 160), 0.9)
    sk.clip(lay, cap)
    saveImage(sk.result(), "props/bottle_soy.png")


def drawGarlicClove(sk, x, y, size, rot):
    """Peeled garlic clove: curved wedge with a pointed tip and a flat root end."""
    s = size / 20.0
    local = chain(bez((0, -10), (7, -6), (9, 4), (5, 10)),
                  bez((5, 10), (0, 11.5), (-5, 11), (-6, 9)),
                  bez((-6, 9), (-3, 2), (-3, -5), (0, -10)))
    rr = math.radians(rot)
    pts = [(x + (px * math.cos(rr) - py * math.sin(rr)) * s, y + (px * math.sin(rr) + py * math.cos(rr)) * s)
           for px, py in local]
    clove = sk.poly(pts)
    sk.part(clove, (250, 238, 198), dark=(214, 190, 136), light=(255, 252, 232), soft=1.8 * s, outline=1.5,
            darkAt=0.12, lightAt=0.12)
    root = inter(sk.circle(x + (-0.5 * math.cos(rr) - 11 * math.sin(rr)) * s,
                           y + (-0.5 * math.sin(rr) + 11 * math.cos(rr)) * s, 4.5 * s), clove)
    sk.paint(soften(root, 0.6), (196, 150, 96), 200)
    return clove


# ======================================================================
# PROPS - Scene 3: teh tarik jug, glass, condensed milk can, neon sign
# ======================================================================
def makeJug():
    """Stainless steel pulling mug: handle on the LEFT, pouring lip on the
    RIGHT of the rim. Spout tip = (142, 27) in image pixels."""
    sk = Sketch(150, 170)
    cx, x0, x1, top, bottom = 82, 42, 122, 34, 154
    rx, ry = 40.0, 10.0
    sk.shadow(sk.ellipse(x0 - 4, bottom - 4, x1 + 8, bottom + 8), 2.5, 110)
    # handle (behind the body): a flat steel strap bent into a loop
    handlePts = chain(bez((x0 + 4, 52), (x0 - 16, 50), (x0 - 28, 60), (x0 - 28, 80), steps=16),
                      bez((x0 - 28, 80), (x0 - 28, 104), (x0 - 20, 124), (x0 + 4, 132), steps=16))
    handle = sk.tube(handlePts, 5.2)
    sk.part(handle, (178, 186, 198), dark=(104, 112, 126), light=(236, 240, 246), soft=1.8, outline=2.6,
            spec=(255, 255, 255), specAt=0.3)
    # body: straight cylinder with a rounded bottom edge + the rim with the spout
    body = union(sk.rrect(x0, top, x1, bottom - ry, 2), sk.ellipse(x0, bottom - 2 * ry, x1, bottom))
    spout = sk.poly(chain(bez((x1 - 16, top - 9), (x1 + 4, top - 12), (x1 + 16, top - 8), (x1 + 20, top - 7)),
                          bez((x1 + 20, top - 7), (x1 + 12, top + 2), (x1 + 2, top + 8), (x1 - 8, top + 9))))
    rim = sk.ellipse(x0, top - ry, x1, top + ry)
    steel = sk.gradient([(0, (200, 206, 216)), (1, (150, 158, 172))], (0, top), (0, bottom))
    sk.part(union(body, rim, spout), steel, dark=0.78, light=1.1, soft=2.5, outline=3.0, darkAt=0.1, lightAt=0.16)
    cylinderShade(sk, body, x0, x1, dark=120, light=150, lightAt=0.28)
    # brushed steel: faint horizontal lines + a sharp environment reflection on the right
    lay, d = sk.layer()
    rng = random.Random(6)
    for i in range(26):
        y = rng.uniform(top + 6, bottom - 8)
        ak.line(d, [(x0 + 2, y), (x1 - 2, y + rng.uniform(-0.5, 0.5))], (255, 255, 255, rng.randint(20, 50)), 0.6)
    ak.line(d, [(x1 - 9, top + 12), (x1 - 9, bottom - 14)], (255, 255, 255, 150), 1.6)
    ak.line(d, [(x0 + 22, top + 12), (x0 + 22, bottom - 16)], (255, 255, 255, 200), 3.2)
    sk.clip(lay, shrink(body, 1))
    # rolled rim, dark inside, bright lip on the spout
    sk.paint(rim, (214, 220, 230))
    inner = sk.ellipse(x0 + 3.5, top - ry + 2.6, x1 - 3.5, top + ry - 2.6)
    innerCol = sk.gradient([(0, (66, 70, 82)), (1, (120, 126, 140))], (0, top - ry), (0, top + ry))
    sk.clip(innerCol.convert("RGBA"), inner)
    sk.paint(inter(soften(minus(inner, shift(inner, 3, 3)), 0.8), inner), (30, 32, 40), 160)
    sk.paint(soften(sk.stroke(arc(cx, top, rx - 1.5, ry - 1.2, 190, 260, 20), 1.2), 0.3), (255, 255, 255), 230)
    sk.paint(inter(spout, minus(below(sk, top - 12), rim)), (224, 230, 238), 255)
    sk.paint(soften(sk.stroke(bez((x1 - 6, top - 9), (x1 + 6, top - 10.5), (x1 + 18, top - 7.5), steps=12), 1.2), 0.3),
             (255, 255, 255), 230)
    # bottom rim line
    sk.paint(soften(sk.stroke(arc(cx, bottom - ry - 3, rx - 1, ry - 1, 15, 165, 30), 1.0), 0.3), (110, 118, 132), 180)
    saveImage(sk.result(), "props/jug.png")


def makeGlassFront():
    """Tall clear glass, drawn as the FRONT layer over the liquid. Almost
    everything is transparent: only light edges, highlight streaks, a thin
    rim and a frosted thick bottom. The liquid area (drawn by the game) is
    x 14..106, from 16 to 190 px above the bottom of the image."""
    sk = Sketch(120, 200)
    x0, x1, top, base = 8, 112, 10, 194
    innerL, innerR, innerBottom = 14, 106, 184     # 200 - 16 = 184 (PIL y of the liquid bottom)
    glass = union(sk.rrect(x0, top, x1, base - 5, 4), sk.ellipse(x0, base - 11, x1, base + 1))
    # faint dark outline so the glass reads on any background (style guide outline, but see-through)
    sk.paint(minus(grow(glass, 1.6), glass), OUTLINE, 150)
    # side walls: bright thin strips (light passes through more glass at the edges)
    walls = union(sk.rrect(x0, top, innerL, base - 4, 3), sk.rrect(innerR, top, x1, base - 4, 3))
    sk.paint(walls, (230, 242, 250), 95)
    sk.paint(sk.rrect(x0 + 0.8, top + 3, x0 + 2.4, base - 12, 1), (255, 255, 255), 200)
    sk.paint(sk.rrect(innerL - 1.2, top + 6, innerL, innerBottom, 0.5), (255, 255, 255), 120)
    sk.paint(sk.rrect(innerR, top + 6, innerR + 1.2, innerBottom, 0.5), (255, 255, 255), 90)
    sk.paint(sk.rrect(x1 - 2.4, top + 3, x1 - 0.8, base - 12, 1), (220, 234, 246), 150)
    # frosted thick bottom with a lens-like highlight
    bottomGlass = union(sk.rrect(x0, innerBottom, x1, base - 5, 1), sk.ellipse(x0, base - 11, x1, base + 1))
    frost = valueNoise(sk.size[0], sk.size[1], 12, 6, seed=3)
    tex = np.zeros(sk.size[::-1] + (4,), np.uint8)
    tex[..., 0:3] = 236
    tex[..., 3] = (110 + frost * 80).astype(np.uint8)
    sk.clip(Image.fromarray(tex, "RGBA"), bottomGlass)
    sk.paint(soften(sk.ellipse(x0 + 10, innerBottom + 2, x1 - 22, innerBottom + 7), 0.8), (255, 255, 255), 170)
    sk.paint(sk.rrect(innerL, innerBottom - 0.6, innerR, innerBottom + 0.8, 0.5), (255, 255, 255), 150)
    # long highlight streaks on the front (left = facing the lamp)
    sk.paint(soften(sk.rrect(20, top + 10, 27, innerBottom - 16, 3), 1.2), (255, 255, 255), 80)
    sk.paint(soften(sk.rrect(30, top + 26, 32, innerBottom - 40, 1), 0.5), (255, 255, 255), 70)
    sk.paint(soften(sk.rrect(92, top + 16, 97, innerBottom - 26, 2), 1.4), (255, 255, 255), 45)
    # rim: thin ellipse, the front edge brighter
    rimOuter = sk.ellipse(x0, top - 5, x1, top + 5)
    rimInner = sk.ellipse(x0 + 2.2, top - 3.4, x1 - 2.2, top + 3.4)
    sk.paint(minus(rimOuter, rimInner), (240, 248, 255), 190)
    sk.paint(inter(minus(rimOuter, rimInner), below(sk, top + 1)), (255, 255, 255), 235)
    sk.paint(minus(grow(rimOuter, 1.2), rimOuter), OUTLINE, 110)
    saveImage(sk.result(), "props/glass_front.png")


def makeMilkCan():
    """Tin of sweetened condensed milk, cream and blue label,
    a punched pouring hole in the lid (at about (58, 13))."""
    sk = Sketch(90, 120)
    cx, x0, x1, top, bottom = 45, 10, 80, 18, 106
    rx, ry = 35.0, 8.5
    sk.shadow(sk.ellipse(x0 - 2, bottom - 4, x1 + 6, bottom + 9), 3, 110)
    body = union(sk.rrect(x0, top, x1, bottom - ry, 0), sk.ellipse(x0, bottom - 2 * ry, x1, bottom))
    lid = sk.ellipse(x0, top - ry, x1, top + ry)
    sk.part(union(body, lid), (190, 196, 206), dark=0.8, light=1.1, soft=2, outline=2.6, darkAt=0.1, lightAt=0.2)
    # label: cream middle, blue bands
    label = sk.rrect(x0, 32, x1, 94, 0)
    labelCol = Image.new("RGBA", sk.size, (248, 240, 218, 255))
    sk.clip(labelCol, label)
    sk.paint(sk.rrect(x0, 32, x1, 42, 0), (40, 92, 176))
    sk.paint(sk.rrect(x0, 86, x1, 94, 0), (40, 92, 176))
    sk.paint(sk.rrect(x0, 43, x1, 44.5, 0), (214, 170, 60))
    sk.paint(sk.rrect(x0, 84, x1, 85.5, 0), (214, 170, 60))
    sk.paint(sk.text(cx, 37.2, "SWEETENED", "sans_bold", 6.5), (255, 244, 220))
    sk.paint(sk.text(cx, 53, "CONDENSED", "sans_bold", 9.5), (36, 80, 160))
    sk.paint(sk.text(cx, 66.5, "MILK", "sans_bold", 12), (200, 44, 48))
    # simple milk drop icon
    drop = sk.poly(chain(bez((cx, 74), (cx + 5, 79), (cx + 5, 82), (cx, 83)), bez((cx, 83), (cx - 5, 82), (cx - 5, 79), (cx, 74))))
    sk.paint(drop, (36, 80, 160))
    sk.paint(sk.ellipse(cx - 2, 78.5, cx, 80.5), (255, 255, 255))
    cylinderShade(sk, body, x0, x1, dark=120, light=120, lightAt=0.3)
    # tin rims: little ridges at the top and bottom
    lay, d = sk.layer()
    for y in (22, 26, 99):
        ak.line(d, arc(cx, y, rx, ry, 0, 180, 30), (120, 128, 142, 200), 1.0)
        ak.line(d, arc(cx, y - 1.2, rx, ry, 0, 180, 30), (240, 244, 250, 160), 0.8)
    sk.clip(lay, body)
    # lid: steel with a ring groove and a punched triangular hole
    sk.paint(lid, (206, 212, 222))
    sk.paint(minus(sk.ellipse(x0 + 4, top - ry + 2, x1 - 4, top + ry - 2), sk.ellipse(x0 + 5.5, top - ry + 3, x1 - 5.5, top + ry - 3)),
             (140, 148, 162), 220)
    hole = sk.poly([(54, 11), (64, 12), (58, 16.5)])
    sk.paint(grow(hole, 0.8), (150, 156, 170))
    sk.paint(hole, (40, 34, 30))
    sk.paint(sk.ellipse(59, 14.5, 63, 17.5), (250, 244, 226))           # a drop of milk at the hole
    sk.paint(soften(sk.stroke(arc(cx, top, rx - 1.5, ry - 1, 190, 250, 16), 1.2), 0.3), (255, 255, 255), 230)
    saveImage(sk.result(), "props/milk_can.png")


# neon glyph strokes, in a 40 x 64 letter box (x right, y down)
NEON_GLYPHS = {
    "2": [chain(bez((1, 17), (0, -3), (40, -4), (39, 18)), bez((39, 18), (38, 34), (6, 44), (1, 64)), [(1, 64), (41, 64)])],
    "4": [[(30, 64), (30, 1), (0, 43), (42, 43)]],
    "J": [[(12, 1), (40, 1)], chain([(32, 1), (32, 44)], bez((32, 44), (32, 68), (0, 68), (0, 46)))],
    "A": [[(0, 64), (20, 1), (40, 64)], [(8, 42), (32, 42)]],
    "M": [[(0, 64), (0, 1), (20, 38), (40, 1), (40, 64)]],
    "H": [[(0, 1), (0, 64)], [(40, 1), (40, 64)], [(0, 33), (40, 33)]],
    "O": [chain(bez((20, 1), (46, 1), (46, 64), (20, 64)), bez((20, 64), (-6, 64), (-6, 1), (20, 1)))],
    "U": [chain([(0, 1), (0, 44)], bez((0, 44), (0, 68), (40, 68), (40, 44)), [(40, 44), (40, 1)])],
    "R": [chain([(0, 64), (0, 1), (22, 1)], bez((22, 1), (44, 1), (44, 34), (22, 34)), [(22, 34), (0, 34)]),
          [(18, 34), (40, 64)]],
    "S": [chain(bez((38, 10), (32, -2), (4, -2), (2, 14)), bez((2, 14), (0, 30), (40, 32), (40, 48)),
                bez((40, 48), (40, 68), (4, 68), (0, 54)))],
}


def neonTubes(sk, x, y, text, color, scale=1.0):
    """Draw the tube strokes of a word. Returns the list of stroke paths."""
    paths = []
    for ch in text:
        if ch == " ":
            x += 24 * scale
            continue
        for stroke in NEON_GLYPHS[ch]:
            paths.append(([(x + px * scale, y + py * scale) for px, py in stroke], color))
        x += 54 * scale
    return paths


def makeNeonSign():
    """'24 HOURS' (open all night) in pink and teal neon tubes on a dark
    backing board. The glow image is a blurred, brighter copy of the tubes
    (the game fades it in and out to make the sign flicker)."""
    W, H = 360, 110
    sk = Sketch(W, H)
    pink, teal = (255, 70, 170), (40, 236, 214)
    board = sk.rrect(8, 8, W - 8, H - 8, 16)
    sk.part(board, (30, 24, 44), dark=(18, 14, 28), light=(52, 44, 72), soft=3, outline=3.0)
    sk.paint(minus(sk.rrect(12, 12, W - 12, H - 12, 13), sk.rrect(13.5, 13.5, W - 13.5, H - 13.5, 12)), (70, 60, 96), 200)
    # tube paths: border + "24" in pink, "HOURS" in teal
    paths = [(chain([(34, 20), (W - 34, 20)], arc(W - 34, 34, 14, 14, 270, 360, 10), [(W - 20, 34), (W - 20, H - 34)],
                    arc(W - 34, H - 34, 14, 14, 0, 90, 10), [(W - 34, H - 20), (34, H - 20)],
                    arc(34, H - 34, 14, 14, 90, 180, 10), [(20, H - 34), (20, 34)], arc(34, 34, 14, 14, 180, 270, 10)), teal)]
    paths += neonTubes(sk, 60, 35, "24", pink, 0.62)
    paths += neonTubes(sk, 142, 35, "HOURS", teal, 0.62)
    # tube shadow on the board, then tube (colour -> lighter -> white-hot core)
    lay, d = sk.layer()
    for pts, col in paths:
        ak.line(d, [(p[0] + 2.5, p[1] + 3) for p in pts], (0, 0, 0, 150), 5.0)
    sk.over(lay.filter(ImageFilter.GaussianBlur(SS * 1.2)))
    tubes, td = sk.layer()
    for pts, col in paths:
        ak.line(td, pts, rgba(mixRgb(col, (0, 0, 0), 0.25)), 5.6)
        ak.line(td, pts, rgba(col), 4.2)
        ak.line(td, pts, rgba(mixRgb(col, (255, 255, 255), 0.55)), 2.4)
        ak.line(td, pts, (255, 255, 255, 235), 0.9)
    sk.over(tubes)
    base = sk.result()
    saveImage(base, "props/neon_24jam.png")

    # glow: every tube colour gets a soft wide halo + a tight bloom
    # (blurred copies of the tube shapes = glowFromAlpha), brightest in the middle
    glow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    for col in (pink, teal):
        lay, d = sk.layer()
        for pts, c in paths:
            if c == col:
                ak.line(d, pts, (255, 255, 255, 255), 5.0)
        shape = ak.finish(lay, W, H)
        glow.alpha_composite(ak.glowFromAlpha(shape, 10, col, 1.6))
        glow.alpha_composite(ak.glowFromAlpha(shape, 4, mixRgb(col, (255, 255, 255), 0.3), 1.8))
        glow.alpha_composite(ak.glowFromAlpha(shape, 1.2, mixRgb(col, (255, 255, 255), 0.7), 1.2))
    saveImage(glow, "props/neon_24jam_glow.png")


# ======================================================================
# PROPS - Scene 4: banquet table and the glass lazy susan
# ======================================================================
def drawPlaceSetting(sk, x, y, scale):
    """Small plate + rice bowl, red chopsticks on a rest, porcelain spoon."""
    s = scale
    plate = sk.ellipse(x - 24 * s, y - 8.5 * s, x + 24 * s, y + 8.5 * s)
    sk.shadow(shift(plate, 0, 2.5 * s), 2 * s, 90)
    sk.part(plate, (250, 250, 252), dark=(200, 204, 218), light=(255, 255, 255), soft=2 * s, outline=2.0 * s,
            darkAt=0.14, lightAt=0.2)
    well = sk.ellipse(x - 15 * s, y - 5 * s, x + 15 * s, y + 5 * s)
    sk.paint(soften(minus(well, shift(well, 1.5 * s, 1.2 * s)), 0.6 * s), (200, 206, 222), 200)
    sk.paint(minus(sk.ellipse(x - 21 * s, y - 7.3 * s, x + 21 * s, y + 7.3 * s),
                   sk.ellipse(x - 19.6 * s, y - 6.6 * s, x + 19.6 * s, y + 6.6 * s)), (200, 60, 60), 220)
    # little rice bowl on the plate
    bowlTop = sk.ellipse(x - 10 * s, y - 7 * s, x + 10 * s, y - 2 * s)
    bowl = union(bowlTop, sk.revolve(x, y - 4.5 * s, 6 * s, lambda t: 10 * s * (1 - 0.35 * t ** 1.5), 0.25))
    sk.part(bowl, (248, 248, 250), dark=(196, 200, 214), light=(255, 255, 255), soft=1.5 * s, outline=1.6 * s)
    sk.paint(sk.ellipse(x - 8.6 * s, y - 6.4 * s, x + 8.6 * s, y - 2.6 * s), (214, 220, 232))
    sk.paint(soften(sk.stroke([(x - 8 * s, y - 1.8 * s), (x + 8 * s, y - 1.8 * s)], 0.8 * s), 0.3 * s), (200, 60, 60), 200)
    # chopsticks on a rest, to the right
    rest = sk.ellipse(x + 26 * s, y - 2 * s, x + 32 * s, y + 2 * s)
    sk.part(rest, (240, 240, 244), soft=0.8 * s, outline=1.2 * s)
    for k in (-1.4, 1.4):
        sticks = sk.stroke([(x + 22 * s, y + 7 * s + k * s), (x + 40 * s, y - 9 * s + k * s)], 1.8 * s)
        sk.part(sticks, (196, 34, 40), dark=(140, 20, 30), light=(240, 90, 90), soft=0.6 * s, outline=1.1 * s)
    sk.paint(sk.stroke([(x + 37 * s, y - 6.5 * s), (x + 40 * s, y - 9 * s)], 2.0 * s), (236, 190, 80), 255)
    # porcelain spoon on the left
    spoon = union(sk.ellipse(x - 36 * s, y - 3 * s, x - 26 * s, y + 3.5 * s),
                  sk.stroke([(x - 28 * s, y), (x - 20 * s, y - 5 * s)], 2.4 * s))
    sk.part(spoon, (248, 248, 250), dark=(200, 204, 216), light=(255, 255, 255), soft=1 * s, outline=1.3 * s)


def makeTableTop():
    """Round banquet table in perspective: white tablecloth top, red skirt
    with gold trim hanging at the front, place settings at the guests' seats."""
    W, H = 980, 360
    sk = Sketch(W, H)
    cx, cy, rx, ry = 490, 160, 478, 150
    drop = 38
    top = sk.ellipse(cx - rx, cy - ry, cx + rx, cy + ry)
    # the skirt is the table edge swept straight down (a short cylinder)
    skirt = inter(sk.revolve(cx, cy, drop, lambda t: rx, ry / float(rx)), below(sk, cy))
    sk.shadow(sk.ellipse(cx - rx + 30, cy + ry + drop - 30, cx + rx - 30, cy + ry + drop + 2), 4, 110)
    # red skirt with vertical folds (lighter / darker bands) and a gold hem
    skirtCol = sk.gradient([(0, (206, 44, 52)), (1, (150, 24, 36))], (0, cy), (0, cy + ry + drop))
    sk.part(skirt, skirtCol, dark=(120, 18, 30), light=(230, 76, 80), soft=6, outline=4.0, darkAt=0.1, lightAt=0.14)
    lay, d = sk.layer()
    for k in range(-24, 25):
        a = k / 24.0
        x = cx + rx * math.sin(a * math.pi / 2.0)
        yTop = cy + ry * math.cos(a * math.pi / 2.0)
        width = 7 + 9 * math.cos(a * math.pi / 2.0)
        ak.line(d, [(x, yTop), (x, yTop + drop + 8)], (110, 14, 28, 110) if k % 2 else (240, 110, 110, 70), width)
    sk.clip(lay.filter(ImageFilter.GaussianBlur(SS * 3)), skirt)
    hem = minus(skirt, shift(skirt, 0, -7))
    sk.paint(hem, (226, 172, 64))
    sk.paint(minus(hem, shift(hem, 0, -2.5)), (150, 100, 30), 200)
    sk.paint(inter(shift(minus(skirt, shift(skirt, 0, -2.2)), 0, -7), skirt), (255, 226, 140), 200)
    # white tablecloth top, slightly darker at the back (farther away)
    cloth = sk.gradient([(0, (214, 204, 196)), (0.5, (238, 230, 218)), (1, (246, 238, 226))], (0, cy - ry), (0, cy + ry))
    sk.part(top, cloth, dark=(200, 188, 180), light=(252, 248, 238), soft=10, outline=4.0, darkAt=0.14, lightAt=0.18)
    # scalloped lace edge of the white cloth hanging over the red skirt
    lace, ld = sk.newMask()
    for k in range(0, 361, 2):
        a = math.radians(k * 0.5)
        x, y = cx + rx * math.cos(a), cy + ry * math.sin(a)
        r = (2.5 + 2.5 * math.sin(a)) * (1.0 if k % 4 == 0 else 0.7)
        ld.ellipse(ak.box(x - r * 1.3, y - r * 0.4, x + r * 1.3, y + r * 1.4), fill=255)
    lace = minus(inter(lace, below(sk, cy + 4)), top)
    sk.part(lace, (240, 232, 220), dark=(206, 196, 186), light=(252, 248, 240), soft=1.2, outline=1.2)
    # a faint woven damask pattern + a soft shadow ring where the lazy susan sits
    lay, d = sk.layer()
    for k in range(1, 5):
        f = 0.3 + 0.15 * k
        ak.ellipse(d, cx - rx * f, cy - ry * f, cx + rx * f, cy + ry * f, None, (200, 190, 184, 60), 1.4)
    # embroidered red and gold border near the edge (ties the top to the red skirt)
    for f, col, w in ((0.955, (214, 166, 70, 230), 1.2), (0.94, (184, 34, 46, 235), 3.4), (0.925, (214, 166, 70, 230), 1.2)):
        ak.ellipse(d, cx - rx * f, cy - ry * f, cx + rx * f, cy + ry * f, None, col, w)
    sk.clip(lay, shrink(top, 6))
    sk.paint(inter(soften(sk.ellipse(cx - 318, cy - 96, cx + 318, cy + 124), 12), top), (120, 100, 100), 55)
    sk.paint(soften(sk.stroke(arc(cx, cy, rx - 8, ry - 5, 196, 244, 40), 3), 1.5), (255, 255, 255), 200)
    # place settings in front of the five guests (angles from the game, back half of the table)
    for angle in (160, 125, 90, 55, 20):
        a = math.radians(angle)
        x = cx + rx * 0.83 * math.cos(a)
        y = cy - ry * 0.79 * math.sin(a)
        depthScale = 1.25 + 0.3 * (1.0 - math.sin(a))
        drawPlaceSetting(sk, x, y, depthScale)
    saveImage(sk.result(), "props/table_top.png")


def makeLazySusan():
    """Glass turntable: see-through tinted glass (alpha about 130), a much
    clearer rim, the thick glass edge at the front, reflections and the
    metal bearing faintly visible in the middle."""
    W, H = 640, 230
    sk = Sketch(W, H)
    cx, cy, rx, ry, thick = 320, 110, 312, 102, 9
    topMask = sk.ellipse(cx - rx, cy - ry, cx + rx, cy + ry)
    edge = minus(sk.ellipse(cx - rx, cy - ry + thick, cx + rx, cy + ry + thick), topMask)
    sk.paint(minus(grow(union(topMask, edge), 2.4), union(topMask, edge)), OUTLINE, 190)
    # thick glass edge (greenish, like real glass seen edge-on)
    edgeCol = sk.gradient([(0, (150, 206, 196)), (0.5, (110, 176, 170)), (1, (150, 206, 196))], (cx - rx, 0), (cx + rx, 0))
    lay = edgeCol.convert("RGBA")
    sk.clip(lay, edge, 0.85)
    sk.paint(inter(shift(minus(topMask, shift(topMask, 0, -1.5)), 0, thick - 1), edge), (230, 255, 250), 170)
    # glass top: pale teal tint, see-through
    sk.paint(topMask, (206, 232, 236), 128)
    # bearing under the glass in the middle (seen through it)
    hub = sk.ellipse(cx - 70, cy - 23, cx + 70, cy + 23)
    sk.paint(soften(hub, 4), (110, 120, 130), 34)
    sk.paint(minus(sk.ellipse(cx - 70, cy - 23, cx + 70, cy + 23), sk.ellipse(cx - 65, cy - 21, cx + 65, cy + 21)),
             (170, 180, 190), 60)
    # rim: clearer (more opaque) ring with a bright bevel on the upper-left
    rim = minus(topMask, sk.ellipse(cx - rx + 12, cy - ry + 5, cx + rx - 12, cy + ry - 5))
    sk.paint(rim, (224, 246, 246), 110)
    sk.paint(soften(sk.stroke(arc(cx, cy, rx - 3, ry - 1.5, 188, 262, 50), 2.4), 0.6), (255, 255, 255), 235)
    sk.paint(soften(sk.stroke(arc(cx, cy, rx - 3, ry - 1.5, 30, 150, 60), 1.6), 0.5), (255, 255, 255), 170)
    sk.paint(minus(grow(sk.ellipse(cx - rx + 12, cy - ry + 5, cx + rx - 12, cy + ry - 5), 0.8),
                   sk.ellipse(cx - rx + 12, cy - ry + 5, cx + rx - 12, cy + ry - 5)), (255, 255, 255), 120)
    # diagonal reflection streaks (window lights) across the glass
    lay, d = sk.layer()
    for (x0, w, a) in ((130, 40, 60), (190, 14, 70), (420, 26, 40), (462, 8, 60)):
        d.polygon(ak.pts([(x0, -10), (x0 + w, -10), (x0 + w - 110, H + 10), (x0 - 110, H + 10)]), fill=(255, 255, 255, a))
    sk.clip(lay.filter(ImageFilter.GaussianBlur(SS * 3)), shrink(topMask, 3))
    for (x, y) in ((150, 40), (470, 170)):
        sk.paint(soften(sk.ellipse(x - 14, y - 2, x + 14, y + 2), 1), (255, 255, 255), 200)
    saveImage(sk.result(), "props/lazy_susan.png")


# ======================================================================
# PROPS - lanterns (street, finale) and the Aura Ladle
# ======================================================================
def makeLanternRed():
    """Small Chinese red lantern with gold caps, ribs and a red tassel."""
    sk = Sketch(52, 70)
    cx = 26
    string = sk.stroke([(cx, 0), (cx, 9)], 1.2)
    sk.paint(string, (60, 40, 30), 255)
    body = sk.ellipse(5, 12, 47, 54)
    glow = sk.radial([(0, (255, 110, 70)), (0.6, (226, 46, 44)), (1, (170, 24, 36))], cx - 3, 30, 22, 22)
    sk.part(body, glow, dark=(150, 20, 34), light=(255, 120, 96), soft=4, outline=2.0, darkAt=0.14, lightAt=0.16)
    lay, d = sk.layer()
    for k in (-2, -1, 0, 1, 2):                      # bamboo ribs bowing outwards
        ak.line(d, bez((cx + k * 3.5, 13), (cx + k * 12, 33), (cx + k * 3.5, 53), steps=16), (140, 16, 30, 190), 1.0)
    for y in (22, 33, 44):                           # thin gold bands
        w = 21 * math.sqrt(max(0.0, 1 - ((y - 33) / 21.0) ** 2))
        ak.line(d, [(cx - w + 1, y), (cx + w - 1, y)], (240, 190, 80, 140), 0.8)
    sk.clip(lay, shrink(body, 1))
    sk.shine(sk.ellipse(12, 19, 19, 31), body, (255, 200, 170), 170, 0.6)
    for (y0, y1) in ((8.5, 15), (51, 57.5)):
        cap = sk.rrect(cx - 9, y0, cx + 9, y1, 2)
        sk.part(cap, (240, 190, 70), dark=(176, 120, 40), light=(255, 236, 150), soft=1, outline=1.6)
    knot = sk.circle(cx, 60, 2.4)
    sk.part(knot, (240, 190, 70), soft=0.6, outline=1.2)
    lay, d = sk.layer()
    for k in range(-3, 4):
        ak.line(d, [(cx + k * 0.8, 61), (cx + k * 1.6, 66.5)], (150, 20, 30, 255), 1.6)
        ak.line(d, [(cx + k * 0.8, 61), (cx + k * 1.6, 66.5)], (220, 40, 44, 255), 0.9)
    sk.over(lay)
    saveImage(sk.result(), "props/lantern_red.png")


def skyLanternShape(sk):
    """Paper sky lantern: rounded top, sides tapering to an open bottom ring."""
    pts = chain(bez((24, 94), (18, 70), (12, 44), (14, 28)),
                bez((14, 28), (16, 6), (74, 6), (76, 28)),
                bez((76, 28), (78, 44), (72, 70), (66, 94)))
    body = union(sk.poly(pts), sk.ellipse(24, 90, 66, 98))
    return body, sk.ellipse(24, 89, 66, 99)


def makeSkyLanterns():
    """sky_lantern.png (unlit, cream paper) and sky_lantern_lit.png (glowing
    warm orange from the flame inside). Same shape so the game can swap them."""
    for lit in (False, True):
        sk = Sketch(90, 110)
        body, opening = skyLanternShape(sk)
        if lit:
            # soft halo around the lantern (kept inside the image)
            halo = soften(grow(body, 3), 5)
            sk.paint(halo, (255, 160, 60), 150)
            paper = sk.radial([(0, (255, 250, 214)), (0.45, (255, 204, 110)), (0.8, (255, 146, 56)), (1, (236, 96, 40))],
                              45, 92, 44, 90)
            sk.part(body, paper, dark=0.9, light=1.05, soft=5, outline=2.2, darkAt=0.16, lightAt=0.16,
                    outlineColor=(120, 50, 30))
        else:
            paper = sk.gradient([(0, (252, 240, 214)), (1, (232, 212, 176))], (0, 8), (0, 96))
            sk.part(body, paper, dark=(206, 180, 142), light=(255, 250, 236), soft=5, outline=2.2,
                    darkAt=0.12, lightAt=0.15)
        # paper seams (vertical panels) and a few wrinkles
        lay, d = sk.layer()
        seamCol = (230, 120, 50, 150) if lit else (190, 160, 120, 160)
        for k in (-2, -1, 0, 1, 2):
            ak.line(d, bez((45 + k * 6, 10), (45 + k * 16, 30), (45 + k * 14, 70), (45 + k * 8, 94), steps=24), seamCol, 0.9)
        rng = random.Random(4)
        for i in range(8):
            x, y = rng.uniform(22, 68), rng.uniform(26, 84)
            ak.line(d, [(x, y), (x + rng.uniform(-4, 4), y + rng.uniform(3, 7))], seamCol[:3] + (80,), 0.7)
        sk.clip(lay, shrink(body, 1))
        if not lit:
            sk.shine(sk.ellipse(22, 20, 32, 46), body, (255, 255, 246), 150, 1.4)
        # wire ring and the dark inside of the bottom opening
        sk.paint(opening, (120, 60, 30) if lit else (70, 56, 46))
        inner = sk.ellipse(27, 91, 63, 97.5)
        sk.paint(inner, (255, 214, 120) if lit else (42, 34, 30))
        sk.paint(minus(opening, inner), (96, 80, 66), 255)
        # cross wires holding the fuel cell, fuel cell and (when lit) the flame
        lay, d = sk.layer()
        ak.line(d, [(28, 95), (62, 93)], (90, 76, 62, 255), 0.9)
        ak.line(d, [(30, 92), (60, 96)], (90, 76, 62, 255), 0.9)
        sk.over(lay)
        cell = sk.rrect(41, 91, 49, 97, 1.5)
        sk.part(cell, (236, 226, 206), soft=0.6, outline=1.2)
        if lit:
            flame = sk.poly(chain(bez((45, 80), (49, 86), (49, 91), (45, 92)), bez((45, 92), (41, 91), (41, 86), (45, 80))))
            sk.paint(soften(grow(flame, 2), 2), (255, 180, 60), 220)
            sk.part(flame, (255, 214, 90), dark=(255, 150, 40), light=(255, 252, 220), soft=1, outline=0)
            saveImage(sk.result(), "props/sky_lantern_lit.png")
        else:
            saveImage(sk.result(), "props/sky_lantern.png")


def makeAuraLadle():
    """The Aura Ladle: a brass ladle with an ornate handle and a gentle gold
    sheen (used on the story panels and in the HUD)."""
    sk = Sketch(220, 220)
    brass = (222, 168, 64)
    brassDark = (150, 96, 36)
    brassLight = (255, 226, 140)
    # soft golden aura behind everything
    handlePath = bez((96, 128), (130, 100), (160, 62), (184, 36), steps=40)
    aura = union(sk.ellipse(14, 110, 126, 200), sk.tube(handlePath, 14), sk.circle(192, 28, 20))
    # the glow fades out before the image border, so it never shows a hard square edge
    fade = soften(sk.rrect(10, 10, 210, 210, 30), 5)
    sk.paint(inter(soften(aura, 9), fade), (255, 214, 110), 90)
    # handle: tapering brass rod with decorative collars
    handle = sk.tube(handlePath, 0, radiusFn=lambda t: 7.0 - 2.0 * t)
    sk.part(handle, brass, dark=brassDark, light=brassLight, soft=2, outline=3.0, spec=(255, 250, 220), specAt=0.3)
    lay, d = sk.layer()
    for k in range(4, 38, 5):                      # engraved diamonds along the handle
        x, y = handlePath[k]
        ak.ellipse(d, x - 1.8, y - 1.8, x + 1.8, y + 1.8, (160, 104, 40, 200))
    sk.clip(lay, shrink(handle, 1))
    for t0 in (0.12, 0.72):                        # collars
        i = int(t0 * 40)
        (x0, y0), (x1, y1) = handlePath[i], handlePath[i + 1]
        tx, ty = x1 - x0, y1 - y0
        length = math.hypot(tx, ty)
        nx, ny = -ty / length * 9, tx / length * 9
        collar = sk.stroke([(x0 + nx, y0 + ny), (x0 - nx, y0 - ny)], 5.0)
        sk.part(collar, (236, 186, 80), dark=brassDark, light=brassLight, soft=1, outline=2.2, spec=(255, 250, 220))
    # ornate end: a ring with a curl and a small red gem
    ring = minus(sk.circle(192, 28, 15), sk.circle(192, 28, 8.5))
    sk.part(ring, brass, dark=brassDark, light=brassLight, soft=1.8, outline=2.6, spec=(255, 250, 220), specAt=0.3)
    curl = sk.tube(bez((180, 44), (170, 50), (166, 42), (172, 38), steps=16), 2.6)
    sk.part(curl, brass, dark=brassDark, light=brassLight, soft=1, outline=2.0)
    gem = sk.circle(192, 28, 5.2)
    sk.part(gem, (220, 40, 60), dark=(140, 16, 40), light=(255, 130, 140), soft=1.2, outline=1.8,
            spec=(255, 240, 240), specAt=0.25)
    # bowl of the ladle (3/4 view), shiny inside
    bowlOuter = union(sk.ellipse(22, 118, 118, 170), sk.revolve(70, 144, 30, lambda t: 48 * math.sqrt(max(0.0, 1 - 0.8 * t * t)), 0.5))
    sk.part(bowlOuter, brass, dark=brassDark, light=brassLight, soft=5, outline=3.0, spec=(255, 250, 220), specAt=0.32)
    rimMask = sk.ellipse(22, 118, 118, 170)
    sk.paint(rimMask, (244, 206, 110))
    inside = sk.ellipse(27, 121, 113, 167)
    grad = sk.radial([(0, (255, 232, 150)), (0.6, (214, 152, 56)), (1, (140, 88, 34))], 78, 150, 46, 26)
    sk.clip(grad.convert("RGBA"), inside)
    sk.paint(inter(soften(minus(inside, shift(inside, 5, 4)), 2), inside), (120, 72, 26), 170)
    sk.shine(sk.ellipse(78, 146, 100, 156), inside, (255, 250, 220), 200, 1.4)
    sk.paint(soften(sk.stroke(arc(70, 144, 46, 24.5, 190, 260, 20), 1.8), 0.4), (255, 252, 230), 230)
    # handle joins the bowl rim with a small leaf-shaped plate
    joint = sk.poly(chain(bez((96, 122), (104, 116), (112, 118), (110, 126)), bez((110, 126), (104, 132), (96, 130), (96, 122))))
    sk.part(joint, (236, 186, 80), dark=brassDark, light=brassLight, soft=1.2, outline=2.2)
    # a few sparkles (4-point stars)
    for (x, y, r) in ((40, 112, 7), (150, 120, 5), (206, 64, 5), (118, 96, 4)):
        star = sk.poly([(x, y - r), (x + r * 0.25, y - r * 0.25), (x + r, y), (x + r * 0.25, y + r * 0.25),
                        (x, y + r), (x - r * 0.25, y + r * 0.25), (x - r, y), (x - r * 0.25, y - r * 0.25)])
        sk.paint(soften(star, 0.3), (255, 250, 220), 240)
    saveImage(sk.result(), "props/aura_ladle.png")




def paintGlints(sk, sauceMask, glints, scale=1.0):
    """Small wet highlights on the sauce (only where there really is sauce)."""
    arr = np.asarray(sauceMask)
    lay, d = sk.layer()
    for x, y, size in glints:
        px, py = int(x * SS), int(y * SS)
        if 0 <= py < arr.shape[0] and 0 <= px < arr.shape[1] and arr[py, px] > 128:
            w, h = size * 0.17, max(0.7, size * 0.06)
            ak.ellipse(d, x - w, y - h, x + w, y + h, (255, 226, 196, 150))
            ak.ellipse(d, x - w * 0.35, y - h * 0.7, x + w * 0.25, y + h * 0.7, (255, 255, 255, 220))
    sk.clip(lay.filter(ImageFilter.GaussianBlur(SS * 0.25 * scale)), sauceMask)


# ======================================================================
# main
# ======================================================================
MAKERS = [
    # food - Scene 1 (chopping) and the rojak bowl / dish
    ("cucumber", makeCucumber),
    ("pineapple", makePineapple),
    ("mango", makeMango),
    ("jicama", makeJicama),
    ("jambu", makeJambu),
    ("chili", makeChili),
    ("rojak_bowl", makeRojakBowls),
    ("rojak_dish", makeRojakDish),
    # food - Scene 2 (wok bits + plated char kuey teow)
    ("bit_garlic", makeBitGarlic),
    ("bit_prawn", makeBitPrawn),
    ("bit_noodle", makeBitNoodle),
    ("bit_soy", makeBitSoy),
    ("bit_egg", makeBitEgg),
    ("bit_sprout", makeBitSprout),
    ("plate_ckt", makePlateCkt),
    # food - Scene 4 (banquet dishes)
    ("dish_rojak", makeDishRojak),
    ("dish_ckt", makeDishCkt),
    ("dish_tehtarik", makeDishTehTarik),
    ("dish_nasilemak", makeDishNasiLemak),
    ("dish_satay", makeDishSatay),
    ("dish_roticanai", makeDishRotiCanai),
    # props
    ("cutting_board", makeCuttingBoard),
    ("cleaver", makeCleaver),
    ("wok", makeWok),
    ("bowl_garlic", makeBowlGarlic),
    ("bowl_prawns", makeBowlPrawns),
    ("bowl_noodles", makeBowlNoodles),
    ("bowl_egg", makeBowlEgg),
    ("bowl_sprouts", makeBowlSprouts),
    ("bottle_soy", makeBottleSoy),
    ("jug", makeJug),
    ("glass_front", makeGlassFront),
    ("milk_can", makeMilkCan),
    ("neon_24jam", makeNeonSign),
    ("table_top", makeTableTop),
    ("lazy_susan", makeLazySusan),
    ("lantern_red", makeLanternRed),
    ("sky_lantern", makeSkyLanterns),
    ("aura_ladle", makeAuraLadle),
]


def main():
    words = [w.lower() for w in sys.argv[1:]]
    for name, maker in MAKERS:
        if words and not any(w in name for w in words):
            continue
        maker()
    print(f"done: {len(SAVED)} images")


if __name__ == "__main__":
    main()
