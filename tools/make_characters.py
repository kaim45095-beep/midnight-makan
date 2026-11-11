"""
make_characters.py
------------------
Draws every CHARACTER image of Midnight Makan with Pillow code (no
downloaded art). Run it from the project root:

    python tools/make_characters.py

Output (assets/images/characters/):
  aiman_walk.png   8-frame walk cycle (4 x 2 frames of 128x160)
  aiman_idle.png   4-frame idle (breathing + blink)
  portrait_*.png   256x256 dialogue portraits (7 characters)
  guest_*.png      seated banquet guests, 4 frames of 180x220
                   (0 idle, 1 blink, 2 eating, 3 cheer)
  oyen_walk.png    Oyen the cat sneaking (4 frames of 128x96)
  oyen_sit.png     Oyen sitting, tail swishing, smug blink (4 frames)
  crowd_people.png 6 background walkers (silhouettes, 110x190 each)

How the pictures are built (ideas from the lectures and tutorials):
  * ONE DICTIONARY PER CHARACTER (skin, hair, outfit colours, accessories).
    The same drawing functions (drawHead, drawEye, drawArm ...) read that
    dictionary for every view, so a portrait, a walk frame and a seated
    guest of the same person always match.
  * MASKS: every body part is first drawn as a white-on-black greyscale
    mask. A part is then painted in layers - dark outline, base colour,
    a shade crescent on the lower-right and a highlight on the upper-left
    (key light from the upper-left). The crescents come from shifting the
    mask and subtracting it (simple image arithmetic = cel shading).
  * SUPERSAMPLING (anti-aliasing, Week 3-4 digital artefacts): shapes are
    drawn 4x bigger and shrunk with LANCZOS resampling at the end.
  * CURVES (Week 8 modelling): Catmull-Rom splines and Bezier curves give
    smooth hair, tails and fabric.
  * TRANSFORMATIONS (Week 5-6): limbs, shoes and phones are rotated with
    the 2D rotation matrix; walk-cycle angles come from sin/cos.
  * COLOUR MODEL: shadows are mixed towards a violet night tone and
    highlights towards warm lantern light (hue-shifted shading).
  * IMAGE PROCESSING: GaussianBlur for blush, glows and soft shadows.

The script is deterministic (all random numbers use seeded generators).
"""
import math
import os
import random
import sys

from PIL import Image, ImageChops, ImageDraw, ImageFilter

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TOOLS_DIR)
import art_kit  # noqa: E402  (shared helpers written for all image scripts)

SS = art_kit.SS
OUTLINE = art_kit.PALETTE["outline"]
WHITE = (255, 255, 255)
GOLD = art_kit.PALETTE["gold"]
BLUSH = (255, 118, 128)
MOUTH_DARK = (122, 36, 52)
TONGUE = (242, 118, 128)
SHADE_TONE = (70, 30, 70)       # shadows lean towards a warm purple (night)
LIGHT_TONE = (255, 246, 222)    # highlights lean towards warm lantern light


# ======================================================================
# Colour helpers
# ======================================================================
def rgb(c):
    return (int(c[0]), int(c[1]), int(c[2]))


def mixRgb(c1, c2, t):
    """Linear interpolation of two colours: c = c1 + (c2 - c1) * t."""
    out = []
    for i in range(3):
        out.append(int(c1[i] + (c2[i] - c1[i]) * t))
    return tuple(out)


def darkOf(c, amount=0.25):
    """Shade colour: darker and shifted towards violet (hue-shifted shading)."""
    return mixRgb(mixRgb(c, SHADE_TONE, 0.35), (0, 0, 0), amount)


def lightOf(c, amount=0.35):
    """Highlight colour: towards a warm white."""
    return mixRgb(c, LIGHT_TONE, amount)


def skinShadeOf(skin):
    """Skin shadows are warm and rosy, never grey."""
    return mixRgb(mixRgb(skin, (170, 60, 80), 0.30), (0, 0, 0), 0.08)


def rampTable(low, high):
    """Lookup table for Image.point(): 0 below low, 255 above high, a ramp between."""
    table = []
    for v in range(256):
        if v <= low:
            table.append(0)
        elif v >= high:
            table.append(255)
        else:
            table.append(int(255 * (v - low) / (high - low)))
    return table


def alphaTable(alpha):
    """Lookup table that scales a mask by alpha/255 (for see-through paint)."""
    table = []
    for v in range(256):
        table.append(v * alpha // 255)
    return table


GROW_TABLE = rampTable(3, 13)
SOLID_TABLE = rampTable(90, 160)


# ======================================================================
# Geometry helpers
# ======================================================================
def lerp(a, b, t):
    return a + (b - a) * t


def lerpPoint(p0, p1, t):
    return (p0[0] + (p1[0] - p0[0]) * t, p0[1] + (p1[1] - p0[1]) * t)


def rotatePoint(x, y, angle, cx=0.0, cy=0.0):
    """2D rotation matrix (Week 5-6) around (cx, cy), angle in degrees:
    x' = x cos(a) - y sin(a),  y' = x sin(a) + y cos(a)."""
    a = math.radians(angle)
    dx = x - cx
    dy = y - cy
    return (cx + dx * math.cos(a) - dy * math.sin(a),
            cy + dx * math.sin(a) + dy * math.cos(a))


def rotatePoints(points, angle, cx, cy):
    out = []
    for x, y in points:
        out.append(rotatePoint(x, y, angle, cx, cy))
    return out


def mirrorPoint(point, cx):
    """Reflection about the vertical line x = cx (mirroring transformation)."""
    return (2 * cx - point[0], point[1])


def ellipsePoints(cx, cy, rx, ry, count=48, start=0.0, end=360.0, angle=0.0):
    """Points on an ellipse (parametric form x = rx cos t, y = ry sin t)."""
    out = []
    for i in range(count + 1):
        a = math.radians(start + (end - start) * i / count)
        x = cx + rx * math.cos(a)
        y = cy + ry * math.sin(a)
        if angle != 0:
            x, y = rotatePoint(x, y, angle, cx, cy)
        out.append((x, y))
    return out


def catmullRom(points, closed=True, steps=8):
    """Smooth curve that passes THROUGH every control point (Catmull-Rom spline)."""
    out = []
    n = len(points)
    if closed:
        segments = n
    else:
        segments = n - 1
    for i in range(segments):
        if closed:
            p0 = points[(i - 1) % n]
            p1 = points[i]
            p2 = points[(i + 1) % n]
            p3 = points[(i + 2) % n]
        else:
            p0 = points[max(i - 1, 0)]
            p1 = points[i]
            p2 = points[i + 1]
            p3 = points[min(i + 2, n - 1)]
        for s in range(steps):
            t = s / steps
            t2 = t * t
            t3 = t2 * t
            xy = []
            for k in range(2):
                v = 0.5 * (2 * p1[k] + (-p0[k] + p2[k]) * t
                           + (2 * p0[k] - 5 * p1[k] + 4 * p2[k] - p3[k]) * t2
                           + (-p0[k] + 3 * p1[k] - 3 * p2[k] + p3[k]) * t3)
                xy.append(v)
            out.append((xy[0], xy[1]))
    if not closed:
        out.append(points[-1])
    return out


def ribbonPoints(centreLine, startWidth, endWidth):
    """Polygon around a centre line whose width changes from start to end
    (used for tails, bandana ends, towels and saree borders)."""
    left = []
    right = []
    n = len(centreLine)
    for i in range(n):
        x, y = centreLine[i]
        xa, ya = centreLine[max(i - 1, 0)]
        xb, yb = centreLine[min(i + 1, n - 1)]
        dx = xb - xa
        dy = yb - ya
        length = math.hypot(dx, dy)
        if length == 0:
            length = 1.0
        nx = -dy / length
        ny = dx / length
        half = lerp(startWidth, endWidth, i / max(1, n - 1)) / 2.0
        left.append((x + nx * half, y + ny * half))
        right.append((x - nx * half, y - ny * half))
    right.reverse()
    return left + right


def curveYAt(points, x):
    """y of a poly-line at a given x (linear interpolation between points)."""
    for i in range(len(points) - 1):
        x0, y0 = points[i]
        x1, y1 = points[i + 1]
        if min(x0, x1) <= x <= max(x0, x1):
            if x1 == x0:
                return y0
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    if x < points[0][0]:
        return points[0][1]
    return points[-1][1]


def starPoints(cx, cy, outer, inner, spikes=4, angle=0.0):
    """Points of a star (sparkle) - alternating outer and inner radius."""
    out = []
    for i in range(spikes * 2):
        r = outer if i % 2 == 0 else inner
        a = math.radians(angle - 90 + i * 180.0 / spikes)
        out.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return out


def roundRectPoints(cx, cy, w, h, r, angle=0.0):
    """Rounded rectangle as a polygon, then rotated (for phones, cards)."""
    out = []
    corners = [(cx + w / 2 - r, cy - h / 2 + r, 270), (cx + w / 2 - r, cy + h / 2 - r, 0),
               (cx - w / 2 + r, cy + h / 2 - r, 90), (cx - w / 2 + r, cy - h / 2 + r, 180)]
    for ccx, ccy, start in corners:
        for i in range(7):
            a = math.radians(start + 90 * i / 6)
            out.append((ccx + r * math.cos(a), ccy + r * math.sin(a)))
    if angle != 0:
        out = rotatePoints(out, angle, cx, cy)
    return out


# ======================================================================
# Mask arithmetic (masks are "L" images: 0 = empty, 255 = full)
# ======================================================================
def maskUnion(a, b):
    return ImageChops.lighter(a, b)


def maskIntersect(a, b):
    return ImageChops.multiply(a, b)


def maskSubtract(a, b):
    return ImageChops.subtract(a, b)


def maskShift(mask, dx, dy):
    """Move a mask by (dx, dy) final pixels (translation)."""
    out = Image.new("L", mask.size, 0)
    out.paste(mask, (int(round(dx * SS)), int(round(dy * SS))))
    return out


def maskGrow(mask, amount):
    """Dilate a mask by about 'amount' final pixels: blur it, then keep
    everything above a small threshold. Only the area around the shape is
    processed, which keeps it fast."""
    if amount <= 0:
        return mask
    bbox = mask.getbbox()
    if bbox is None:
        return mask
    margin = int(amount * SS * 2) + 6
    x0 = max(0, bbox[0] - margin)
    y0 = max(0, bbox[1] - margin)
    x1 = min(mask.size[0], bbox[2] + margin)
    y1 = min(mask.size[1], bbox[3] + margin)
    region = mask.crop((x0, y0, x1, y1))
    region = region.filter(ImageFilter.GaussianBlur(amount * SS / 2.0)).point(GROW_TABLE)
    out = Image.new("L", mask.size, 0)
    out.paste(region, (x0, y0))
    return out


def maskShrink(mask, amount):
    """Erode a mask (grow the empty space around it)."""
    return ImageChops.invert(maskGrow(ImageChops.invert(mask), amount))


# ======================================================================
# Painter: one supersampled canvas + the house painting style
# ======================================================================
class Painter:
    """Holds a 4x canvas. All coordinates are in FINAL pixels (PIL: y down)."""

    def __init__(self, width, height, line=3.0, shade=3.0, light=1.2):
        self.width = width
        self.height = height
        self.image = Image.new("RGBA", (width * SS, height * SS), (0, 0, 0, 0))
        self.under = Image.new("RGBA", self.image.size, (0, 0, 0, 0))  # soft shadows, glows
        self.over = Image.new("RGBA", self.image.size, (0, 0, 0, 0))   # whiskers, sparkles
        self.size = self.image.size
        self.line = line
        self.shadeDepth = shade
        self.lightDepth = light
        self.masks = {}

    # ---------------- masks ----------------
    def blank(self):
        mask = Image.new("L", self.size, 0)
        return mask, ImageDraw.Draw(mask)

    def ellipse(self, cx, cy, rx, ry, angle=0.0):
        mask, draw = self.blank()
        if angle == 0:
            draw.ellipse(art_kit.box(cx - rx, cy - ry, cx + rx, cy + ry), fill=255)
        else:
            draw.polygon(art_kit.pts(ellipsePoints(cx, cy, rx, ry, 64, 0, 360, angle)), fill=255)
        return mask

    def circle(self, cx, cy, r):
        return self.ellipse(cx, cy, r, r)

    def poly(self, points, smooth=False):
        if smooth:
            points = catmullRom(points, True)
        mask, draw = self.blank()
        draw.polygon(art_kit.pts(points), fill=255)
        return mask

    def rect(self, x0, y0, x1, y1, radius=0):
        mask, draw = self.blank()
        if radius > 0:
            draw.rounded_rectangle(art_kit.box(x0, y0, x1, y1), radius=radius * SS, fill=255)
        else:
            draw.rectangle(art_kit.box(x0, y0, x1, y1), fill=255)
        return mask

    def limb(self, points, width):
        """Thick line with round ends and joints (arms, legs, sticks)."""
        mask, draw = self.blank()
        if len(points) > 1:
            draw.line(art_kit.pts(points), fill=255, width=max(1, int(width * SS)), joint="curve")
        r = width / 2.0
        for x, y in points:
            draw.ellipse(art_kit.box(x - r, y - r, x + r, y + r), fill=255)
        return mask

    def taper(self, points, startWidth, endWidth):
        """Stroke that gets thinner along its path (strands, whiskers)."""
        mask, draw = self.blank()
        n = len(points)
        for i in range(n):
            r = lerp(startWidth, endWidth, i / max(1, n - 1)) / 2.0
            x, y = points[i]
            draw.ellipse(art_kit.box(x - r, y - r, x + r, y + r), fill=255)
            if i > 0:
                draw.line(art_kit.pts([points[i - 1], points[i]]), fill=255,
                          width=max(1, int(2 * r * SS)))
        return mask

    # ---------------- painting ----------------
    def fill(self, mask, colour, alpha=255, target=None):
        """Paint a colour through a mask (proper alpha compositing)."""
        if target is None:
            target = self.image
        bbox = mask.getbbox()
        if bbox is None:
            return
        region = mask.crop(bbox)
        if alpha < 255:
            region = region.point(alphaTable(alpha))
        layer = Image.new("RGBA", region.size, (colour[0], colour[1], colour[2], 0))
        layer.putalpha(region)
        target.alpha_composite(layer, dest=(bbox[0], bbox[1]))

    def gradient(self, mask, topColour, bottomColour, alpha=255, horizontal=False):
        """Linear colour interpolation across the mask's bounding box."""
        bbox = mask.getbbox()
        if bbox is None:
            return
        strip = Image.new("RGBA", (1, 256))
        data = []
        for i in range(256):
            c = mixRgb(topColour, bottomColour, i / 255.0)
            data.append((c[0], c[1], c[2], 255))
        strip.putdata(data)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        if horizontal:
            strip = strip.rotate(90, expand=True).transpose(Image.FLIP_LEFT_RIGHT)
        layer = strip.resize((w, h), Image.BILINEAR)
        region = mask.crop(bbox)
        if alpha < 255:
            region = region.point(alphaTable(alpha))
        layer.putalpha(region)
        self.image.alpha_composite(layer, dest=(bbox[0], bbox[1]))

    def part(self, mask, colour, line=None, shade=None, light=None, shadeColour=None,
             lightColour=None, lineColour=OUTLINE, lightAlpha=110):
        """The house style for one body part: outline -> base -> shade -> highlight."""
        if line is None:
            line = self.line
        if shade is None:
            shade = self.shadeDepth
        if light is None:
            light = self.lightDepth
        if line > 0:
            self.fill(maskGrow(mask, line), lineColour)
        self.fill(mask, colour)
        if shade > 0:
            # shade crescent = the part minus a copy moved towards the light
            crescent = maskSubtract(mask, maskShift(mask, -shade, -shade))
            if shadeColour is None:
                shadeColour = darkOf(colour)
            self.fill(crescent, shadeColour)
        if light > 0:
            band = maskSubtract(mask, maskShift(mask, light, light))
            if lightColour is None:
                lightColour = lightOf(colour)
            self.fill(band, lightColour, lightAlpha)
        return mask

    def stroke(self, points, colour, width, alpha=255, smooth=False, target=None):
        if smooth:
            points = catmullRom(points, False)
        self.fill(self.limb(points, width), colour, alpha, target)

    def softEllipse(self, cx, cy, rx, ry, colour, alpha, blurPx, clip=None, target=None):
        """Blurred ellipse (blush, glows, ground shadows)."""
        mask = self.ellipse(cx, cy, rx, ry).filter(ImageFilter.GaussianBlur(blurPx * SS))
        if clip is not None:
            mask = maskIntersect(mask, clip)
        self.fill(mask, colour, alpha, target)

    def castShadow(self, caster, receiver, dx, dy, colour, alpha=150):
        """Shadow of one part falling on another (shifted caster AND receiver)."""
        shadow = maskIntersect(maskShift(caster, dx, dy), receiver)
        shadow = maskSubtract(shadow, caster)
        self.fill(shadow, colour, alpha)

    def sparkle(self, cx, cy, size, colour=GOLD, target=None):
        star = self.poly(starPoints(cx, cy, size, size * 0.3, 4))
        if target is None:
            self.fill(maskGrow(star, 1.2), OUTLINE)
            self.fill(star, colour)
            self.fill(self.circle(cx, cy, size * 0.22), WHITE)
        else:
            self.fill(star, colour, 255, target)

    # ---------------- output ----------------
    def result(self, outer=0.9):
        """Add the outer silhouette line (thicker than inner lines = line
        weight variation), put the soft layers under/over, then shrink 4x."""
        img = self.image
        if outer > 0:
            solid = img.getchannel("A").point(SOLID_TABLE)
            ring = maskGrow(solid, outer)
            base = Image.new("RGBA", self.size, OUTLINE + (0,))
            base.putalpha(ring)
            base.alpha_composite(img)
            img = base
        final = self.under.copy()
        final.alpha_composite(img)
        final.alpha_composite(self.over)
        return final.resize((self.width, self.height), Image.LANCZOS)


# ======================================================================
# CHARACTER DICTIONARIES - the single source of truth for every view
# ======================================================================
AIMAN = {
    "id": "aiman", "skin": (226, 168, 122), "hair": (36, 30, 42), "brow": (36, 30, 42),
    "eyes": (104, 62, 36), "hairStyle": "bandana", "bandana": (255, 140, 50),
    "shirt": (250, 248, 242), "apron": (214, 48, 49), "pants": (54, 58, 92),
    "shoe": (248, 248, 244), "shoeStripe": (214, 48, 49), "sole": (206, 204, 216),
    "sleeve": "short", "lashes": False, "eyeScale": 1.0, "mouth": "grin",
}

NENEK = {
    "id": "nenek", "skin": (234, 186, 148), "brow": (120, 100, 96), "eyes": (92, 58, 40),
    "hairStyle": "tudung", "tudung": (214, 190, 242), "tudungPattern": "floral",
    "innerCap": (252, 250, 255), "flowerA": (255, 252, 255), "flowerB": (164, 112, 206),
    "outfit": (184, 150, 226), "sleeve": "long", "sleeveColour": (184, 150, 226),
    "cuff": (160, 124, 206), "glasses": (150, 92, 70), "glassStyle": "round",
    "lashes": True, "eyeScale": 0.86, "mouth": "smile", "wrinkles": True,
    "drapePointed": True,
}

LEON = {
    "id": "leon", "skin": (244, 206, 170), "hair": (200, 200, 210), "brow": (150, 150, 162),
    "eyes": (70, 46, 34), "hairStyle": "slick", "suit": (246, 238, 216),
    "shirt": (255, 255, 252), "bowtie": GOLD, "sleeve": "long", "sleeveColour": (246, 238, 216),
    "cuff": (255, 255, 252), "glasses": (232, 176, 60), "glassStyle": "round",
    "lashes": False, "eyeScale": 0.92, "mouth": "smirk", "moustache": True,
}

LIM = {
    "id": "lim", "skin": (238, 192, 150), "hair": (176, 176, 186), "brow": (120, 120, 132),
    "eyes": (80, 52, 38), "hairStyle": "bald", "singlet": (250, 250, 246),
    "towel": (252, 252, 250), "towelStripe": (70, 130, 210), "towelRed": (214, 48, 49),
    "sleeve": "none", "lashes": False, "eyeScale": 0.9, "mouth": "open", "wrinkles": True,
    "belly": True,
}

KAVITHA = {
    "id": "kavitha", "skin": (178, 116, 80), "hair": (32, 26, 36), "brow": (32, 26, 36),
    "eyes": (70, 40, 26), "hairStyle": "bun", "saree": (206, 38, 122), "sareeBorder": GOLD,
    "blouse": (244, 182, 62), "sleeve": "short", "sleeveColour": (244, 182, 62),
    "jasmine": (255, 252, 240), "bindi": (200, 22, 50), "lashes": True, "eyeScale": 1.0,
    "mouth": "smile", "earrings": True,
}

FARAH = {
    "id": "farah", "skin": (230, 180, 140), "brow": (50, 36, 40), "eyes": (84, 50, 34),
    "hairStyle": "tudung", "tudung": (36, 170, 162), "tudungPattern": "plain",
    "innerCap": (22, 110, 108), "top": (252, 244, 226), "cardigan": (240, 176, 64),
    "sleeve": "long", "sleeveColour": (240, 176, 64), "cuff": (252, 244, 226),
    "glasses": (70, 52, 96), "glassStyle": "square", "lanyard": (40, 52, 120),
    "phone": (242, 110, 110), "lashes": True, "eyeScale": 1.0, "mouth": "open",
    "drapePointed": False,
}

OYEN = {
    "id": "oyen", "fur": (244, 146, 54), "stripe": (206, 100, 36), "white": (255, 247, 236),
    "eyes": (128, 206, 70), "nose": (240, 128, 142), "innerEar": (248, 170, 172),
}


# ======================================================================
# HEAD - shared by every human character and every view
# ======================================================================
def makeHead(cx, cy, w, h, turn):
    """turn = 0 for a front view, ~0.3 for 3/4 portraits, ~0.55 for walking."""
    return {"cx": cx, "cy": cy, "w": w, "h": h, "turn": turn, "flutter": 0.0}


def headOutline(head, grow=0.0):
    """Chibi head: a wide round skull that narrows into soft cheeks and a
    small chin. The chin moves towards the facing direction (3/4 view)."""
    cx = head["cx"]
    cy = head["cy"]
    w = head["w"] + grow * 2
    h = head["h"] + grow * 2
    t = head["turn"]
    points = []
    for i in range(72):
        a = 2 * math.pi * i / 72
        c = math.cos(a)
        s = math.sin(a)
        x = c * w / 2
        y = s * h / 2
        if s > 0:
            x = x * (1.0 - 0.15 * s * s)
            x = x + t * w * 0.10 * s * s
            if c > 0:
                x = x + t * w * 0.04 * s
        points.append((cx + x, cy + y))
    return points


def faceGeometry(head, ch):
    """Where the eyes, nose, mouth and cheeks go (moves with head turn)."""
    cx = head["cx"]
    cy = head["cy"]
    w = head["w"]
    h = head["h"]
    t = head["turn"]
    scale = ch.get("eyeScale", 1.0)
    eyeY = cy + h * 0.10
    ew = w * 0.13 * scale
    eh = h * 0.21 * scale
    leftX = cx + w * (-0.20 + 0.15 * t)
    rightX = cx + w * (0.20 + 0.08 * t)
    return {
        "eyeY": eyeY, "leftX": leftX, "rightX": rightX,
        "ewL": ew * (1 + 0.05 * t), "ewR": ew * (1 - 0.30 * t), "eh": eh,
        "mouthX": cx + w * 0.12 * t, "mouthY": cy + h * 0.30,
        "noseX": cx + w * 0.16 * t, "noseY": cy + h * 0.21,
        "faceX": cx + w * 0.12 * t, "size": w,
    }


def drawEar(p, ch, x, y, w, h):
    ear = p.ellipse(x, y, w / 2, h / 2)
    p.part(ear, ch["skin"], shadeColour=skinShadeOf(ch["skin"]))
    inner = p.ellipse(x + w * 0.05, y + h * 0.02, w * 0.2, h * 0.26)
    p.fill(inner, skinShadeOf(ch["skin"]), 200)
    return ear


def drawEye(p, ch, x, y, ew, eh, state, outerSide, lid=0.0, isCat=False):
    """Big chibi eye. outerSide = -1 if the outer corner is on the left.
    state: open / closed (blink, content) / happy (^ shape) / smug."""
    lw = max(1.0, eh * 0.12)
    if state == "open":
        eye = p.ellipse(x, y, ew / 2, eh / 2)
        p.fill(maskGrow(eye, lw * 0.35), OUTLINE)
        if isCat:
            p.gradient(eye, darkOf(ch["eyes"], 0.35), lightOf(ch["eyes"], 0.25))
            pupil = p.ellipse(x + ew * 0.05 * outerSide * -1, y + eh * 0.02, ew * 0.17, eh * 0.36)
            p.fill(pupil, (24, 20, 26))
        else:
            p.gradient(eye, (28, 18, 28), ch["eyes"])
            low = maskIntersect(p.ellipse(x, y + eh * 0.28, ew * 0.34, eh * 0.18), eye)
            p.fill(low, lightOf(ch["eyes"], 0.5), 190)
        # two highlights (the key light is upper-left)
        p.fill(p.ellipse(x - ew * 0.13, y - eh * 0.16, ew * 0.21, eh * 0.15), WHITE)
        p.fill(p.circle(x + ew * 0.16, y + eh * 0.18, ew * 0.09), WHITE, 230)
        if lid > 0:
            # upper eyelid covers the top of the eye (relaxed / smug look)
            lidY = y - eh / 2 + eh * lid
            tilt = eh * 0.10 * outerSide if isCat else 0
            cover = p.poly([(x - ew, lidY - tilt - eh), (x + ew, lidY + tilt - eh),
                            (x + ew, lidY + tilt), (x - ew, lidY - tilt)])
            cover = maskIntersect(cover, maskGrow(eye, lw * 0.6))
            p.fill(cover, ch["skin"] if "skin" in ch else ch["fur"])
            p.stroke([(x - ew * 0.58, lidY - tilt), (x + ew * 0.58, lidY + tilt)], OUTLINE, lw * 1.3)
        else:
            arc = ellipsePoints(x, y, ew / 2 + lw * 0.1, eh / 2 + lw * 0.1, 20, 198, 342)
            p.stroke(arc, OUTLINE, lw * 1.35)
        if ch.get("lashes"):
            ox = x + outerSide * ew * 0.46
            p.stroke([(ox, y - eh * 0.2), (ox + outerSide * ew * 0.22, y - eh * 0.36)], OUTLINE, lw * 1.0)
    elif state == "closed":
        curve = art_kit.bezier((x - ew * 0.58, y), (x, y + eh * 0.34), (x + ew * 0.58, y), steps=16)
        p.stroke(curve, OUTLINE, lw * 1.3)
        if ch.get("lashes"):
            ox = x + outerSide * ew * 0.55
            p.stroke([(ox, y + eh * 0.01), (ox + outerSide * ew * 0.2, y - eh * 0.1)], OUTLINE, lw)
    elif state == "happy":
        curve = art_kit.bezier((x - ew * 0.6, y + eh * 0.12), (x, y - eh * 0.36),
                               (x + ew * 0.6, y + eh * 0.12), steps=16)
        p.stroke(curve, OUTLINE, lw * 1.35)
    elif state == "smug":
        curve = art_kit.bezier((x - ew * 0.6, y - eh * 0.05), (x, y + eh * 0.2),
                               (x + ew * 0.6, y - eh * 0.05), steps=16)
        p.stroke(curve, OUTLINE, lw * 1.4)
        p.stroke([(x - ew * 0.62, y - eh * 0.12), (x + ew * 0.62, y - eh * 0.12 + outerSide * eh * 0.06)],
                 OUTLINE, lw * 0.9, 200)


def drawBrow(p, ch, x, y, ew, lift=0.0, tilt=0.0, thick=1.0):
    """tilt > 0 lifts the inner end (worried), tilt < 0 lowers it (grumpy)."""
    w = ew * 0.62
    points = art_kit.bezier((x - w, y + tilt - lift * 0.5), (x, y - ew * 0.22 - lift),
                            (x + w, y - tilt - lift * 0.5), steps=12)
    p.stroke(points, ch["brow"], max(1.2, ew * 0.2 * thick))


def drawMouth(p, ch, x, y, size, kind):
    lw = max(1.0, size * 0.2)
    if kind == "smile":
        curve = art_kit.bezier((x - size, y - size * 0.1), (x, y + size * 0.7), (x + size, y - size * 0.1),
                               steps=16)
        p.stroke(curve, OUTLINE, lw)
    elif kind == "smirk":
        curve = art_kit.bezier((x - size * 0.9, y + size * 0.05), (x + size * 0.1, y + size * 0.5),
                               (x + size, y - size * 0.35), steps=16)
        p.stroke(curve, OUTLINE, lw)
    elif kind in ("open", "grin", "cheer"):
        wide = 1.0
        deep = 1.5
        if kind == "cheer":
            wide = 1.25
            deep = 2.0
        top = y - size * 0.2
        outline = art_kit.bezier((x + size * wide, top), (x, top + size * deep * 1.3),
                                 (x - size * wide, top), steps=20)
        mouth = p.poly(outline)
        p.fill(maskGrow(mouth, lw * 0.55), OUTLINE)
        p.fill(mouth, MOUTH_DARK)
        tongue = p.ellipse(x + size * 0.12, top + size * deep * 0.75, size * 0.55 * wide, size * 0.36 * deep)
        p.fill(maskIntersect(tongue, mouth), TONGUE)
        if kind == "grin":
            teeth = maskIntersect(p.rect(x - size * 2, top - 2, x + size * 2, top + size * 0.32), mouth)
            p.fill(teeth, WHITE)
    elif kind == "eat":
        mouth = p.ellipse(x, y + size * 0.2, size * 0.55, size * 0.62)
        p.fill(maskGrow(mouth, lw * 0.55), OUTLINE)
        p.fill(mouth, MOUTH_DARK)
        p.fill(maskIntersect(p.ellipse(x, y + size * 0.62, size * 0.4, size * 0.3), mouth), TONGUE)
    elif kind == "flat":
        p.stroke([(x - size * 0.7, y), (x + size * 0.7, y + size * 0.05)], OUTLINE, lw)


def drawFace(p, ch, head, expr):
    """Eyes, brows, cheeks, nose and mouth from an expression dictionary."""
    g = faceGeometry(head, ch)
    w = head["w"]
    h = head["h"]
    t = head["turn"]
    face = p.masks["face"]
    skinDark = skinShadeOf(ch["skin"])
    eyeY = g["eyeY"]

    # rosy cheeks (soft blurred ellipses, clipped to the face)
    for bx, rx in ((g["leftX"] - w * 0.04, w * 0.085), (g["rightX"] + w * 0.03 * (1 - t), w * 0.075 * (1 - 0.3 * t))):
        p.softEllipse(bx, eyeY + h * 0.17, rx, h * 0.045, BLUSH, expr.get("blush", 120), w * 0.018, clip=face)

    eyes = expr.get("eyes", "open")
    lid = expr.get("lid", 0.0)
    drawEye(p, ch, g["leftX"], eyeY, g["ewL"], g["eh"], eyes, -1, lid)
    drawEye(p, ch, g["rightX"], eyeY, g["ewR"], g["eh"], eyes, 1, lid)

    # eyebrows (hidden under a bandana or fringe for some characters)
    browY = eyeY - g["eh"] * 0.82 - h * 0.02
    lifts = expr.get("brows", (0.0, 0.0))
    tilts = expr.get("browTilt", (0.0, 0.0))
    thick = 1.35 if ch["id"] == "lim" else 1.0
    drawBrow(p, ch, g["leftX"] - w * 0.01, browY, g["ewL"] * 1.1, lifts[0] * h, tilts[0] * h, thick)
    drawBrow(p, ch, g["rightX"] + w * 0.01, browY, g["ewR"] * 1.1, lifts[1] * h, -tilts[1] * h, thick)

    # age details: smile lines at the outer eye corners and under the eyes
    if ch.get("wrinkles"):
        for ex, side, ew in ((g["leftX"], -1, g["ewL"]), (g["rightX"], 1, g["ewR"])):
            ox = ex + side * ew * 0.75
            p.stroke([(ox, eyeY - g["eh"] * 0.05), (ox + side * w * 0.03, eyeY + g["eh"] * 0.02)],
                     skinDark, w * 0.012)
            p.stroke(art_kit.bezier((ex - ew * 0.35, eyeY + g["eh"] * 0.62), (ex, eyeY + g["eh"] * 0.72),
                                    (ex + ew * 0.35, eyeY + g["eh"] * 0.62), steps=8), skinDark, w * 0.01, 150)

    # nose: a tiny soft hook
    nx = g["noseX"]
    ny = g["noseY"]
    if t > 0.1:
        p.stroke(art_kit.bezier((nx - w * 0.005, ny - h * 0.035), (nx + w * 0.035, ny + h * 0.005),
                                (nx - w * 0.01, ny + h * 0.02), steps=10), skinDark, w * 0.018)
    else:
        p.fill(p.ellipse(nx, ny, w * 0.022, h * 0.014), skinDark, 220)

    # moustache (Dato' Leon) sits above the mouth
    mouthKind = expr.get("mouth", ch.get("mouth", "smile"))
    size = w * 0.075
    if ch.get("moustache"):
        mx = g["mouthX"]
        my = g["mouthY"] - h * 0.075
        for side in (-1, 1):
            tip = (mx + side * w * 0.11, my + h * 0.03)
            shape = p.poly(catmullRom([(mx, my - h * 0.015), (mx + side * w * 0.06, my - h * 0.025),
                                       tip, (mx + side * w * 0.05, my + h * 0.02), (mx, my + h * 0.015)], True, 6))
            p.part(shape, ch["hair"], line=p.line * 0.55, shade=0.8, light=0)

    drawMouth(p, ch, g["mouthX"], g["mouthY"], size, mouthKind)

    # bindi: a small red dot in the middle of the forehead (Auntie Kavitha)
    if "bindi" in ch:
        bx = head["cx"] + w * 0.13 * t
        by = eyeY - h * 0.27
        dot = p.circle(bx, by, w * 0.024)
        p.fill(maskGrow(dot, 0.6), darkOf(ch["bindi"], 0.3))
        p.fill(dot, ch["bindi"])
        p.fill(p.circle(bx - w * 0.008, by - w * 0.008, w * 0.007), WHITE, 200)


def drawGlasses(p, ch, head):
    g = faceGeometry(head, ch)
    w = head["w"]
    t = head["turn"]
    frame = ch["glasses"]
    lw = w * 0.022
    lenses = []
    for ex, ew in ((g["leftX"], g["ewL"]), (g["rightX"], g["ewR"])):
        cy = g["eyeY"] + g["eh"] * 0.05
        if ch["glassStyle"] == "round":
            lens = p.ellipse(ex, cy, ew * 0.95, g["eh"] * 0.64)
        else:
            lens = p.rect(ex - ew * 0.98, cy - g["eh"] * 0.52, ex + ew * 0.98, cy + g["eh"] * 0.56,
                          radius=g["eh"] * 0.3)
        lenses.append((lens, ex, ew, cy))
    for lens, ex, ew, cy in lenses:
        p.fill(lens, (210, 236, 255), 40)
        glare = p.poly([(ex - ew * 0.9, cy + g["eh"] * 0.1), (ex - ew * 0.2, cy - g["eh"] * 0.7),
                        (ex + ew * 0.05, cy - g["eh"] * 0.7), (ex - ew * 0.65, cy + g["eh"] * 0.3)])
        p.fill(maskIntersect(glare, lens), WHITE, 120)
        ring = maskSubtract(maskGrow(lens, lw * 1.25), lens)
        p.fill(ring, OUTLINE)
        p.fill(maskSubtract(maskGrow(lens, lw * 0.75), lens), frame)
    # bridge across the nose and the arm towards the ear
    leftLens = lenses[0]
    rightLens = lenses[1]
    by = leftLens[3] - g["eh"] * 0.25
    bridge = [(leftLens[1] + leftLens[2] * 0.9, by), ((leftLens[1] + rightLens[1]) / 2, by - g["eh"] * 0.08),
              (rightLens[1] - rightLens[2] * 0.9, by)]
    p.stroke(bridge, OUTLINE, lw * 1.25, smooth=True)
    p.stroke(bridge, frame, lw * 0.7, smooth=True)
    if t > 0.1:
        arm = [(leftLens[1] - leftLens[2] * 0.95, by), (head["cx"] - w * (0.34 - 0.1 * t), by + g["eh"] * 0.05)]
        p.stroke(arm, OUTLINE, lw * 1.25)
        p.stroke(arm, frame, lw * 0.7)


# ----------------------------------------------------------------------
# Hair / headwear styles
# ----------------------------------------------------------------------
def bandanaEdge(head):
    """Lower edge of Aiman's bandana across the forehead (a smooth curve)."""
    cx = head["cx"]
    cy = head["cy"]
    w = head["w"]
    h = head["h"]
    t = head["turn"]
    control = [(cx - 0.62 * w, cy + 0.02 * h), (cx - 0.40 * w, cy - 0.10 * h),
               (cx - 0.05 * w + 0.12 * w * t, cy - 0.20 * h), (cx + 0.36 * w, cy - 0.19 * h),
               (cx + 0.64 * w, cy - 0.10 * h)]
    return catmullRom(control, False, 10)


def drawAimanHairBack(p, ch, head):
    cx = head["cx"]
    cy = head["cy"]
    w = head["w"]
    h = head["h"]
    # bandana tails flutter behind the knot (sin wave phase from the pose)
    kx = cx - w * 0.47
    ky = cy - h * 0.15
    flutter = head.get("flutter", 0.0)
    tails = [(w * 0.34, 150 + flutter * 12, w * 0.13), (w * 0.27, 118 + flutter * 16, w * 0.11)]
    for length, angle, width in tails:
        a = math.radians(angle)
        tip = (kx + length * math.cos(a), ky + length * math.sin(a))
        mid = lerpPoint((kx, ky), tip, 0.5)
        wave = w * 0.05 * math.sin(math.radians(flutter * 90 + angle))
        mid = (mid[0] + wave * math.sin(a), mid[1] - wave * math.cos(a))
        centre = catmullRom([(kx, ky), mid, tip], False, 8)
        tail = p.poly(ribbonPoints(centre, width, width * 0.72))
        p.part(tail, darkOf(ch["bandana"], 0.05), shade=p.shadeDepth * 0.6)
    # back hair: a short spiky mass behind the head and at the nape
    back = p.ellipse(cx - w * 0.04, cy - h * 0.02, w * 0.5, h * 0.5)
    spikes = p.poly([(cx - w * 0.40, cy - h * 0.10), (cx - w * 0.60, cy + h * 0.10), (cx - w * 0.45, cy + h * 0.10),
                     (cx - w * 0.52, cy + h * 0.30), (cx - w * 0.36, cy + h * 0.22), (cx - w * 0.33, cy + h * 0.38),
                     (cx - w * 0.18, cy + h * 0.22), (cx - w * 0.1, cy)])
    p.part(maskUnion(back, spikes), ch["hair"])


def drawAimanHairFront(p, ch, head):
    cx = head["cx"]
    cy = head["cy"]
    w = head["w"]
    h = head["h"]
    t = head["turn"]
    face = p.masks["face"]
    edge = bandanaEdge(head)
    skinDark = skinShadeOf(ch["skin"])

    # spiky fringe that pokes out under the bandana (swept to the right)
    fringePts = [(cx - 0.40 * w, curveYAt(edge, cx - 0.40 * w) - h * 0.06)]
    for fx in (-0.34, -0.14, 0.06, 0.25, 0.42):
        x = cx + fx * w + 0.04 * w * t
        ey = curveYAt(edge, x)
        fringePts.append((x - 0.09 * w, ey - h * 0.01))
        fringePts.append((x + 0.04 * w, ey + h * 0.13))
    fringePts.append((cx + 0.52 * w, curveYAt(edge, cx + 0.5 * w) - h * 0.02))
    fringePts.append((cx + 0.52 * w, curveYAt(edge, cx + 0.5 * w) - h * 0.1))
    fringe = maskIntersect(p.poly(fringePts), p.poly(headOutline(head, w * 0.03)))
    p.castShadow(fringe, face, 0.6, h * 0.05, skinDark, 150)
    p.part(fringe, ch["hair"], line=p.line * 0.8)
    # sideburn in front of the ear
    if t > 0.2:
        sx = cx - w * (0.34 - 0.12 * t)
        burn = p.poly([(sx - w * 0.05, cy - h * 0.06), (sx + w * 0.06, cy - h * 0.06), (sx + w * 0.01, cy + h * 0.16)])
        p.part(burn, ch["hair"], line=p.line * 0.7, shade=0)

    # the bandana cap: the head (slightly bigger) above the edge curve
    region = p.poly(edge + [(cx + w, cy - h), (cx - w, cy - h)])
    capHead = makeHead(cx, cy - h * 0.02, w * 1.06, h * 1.07, t)
    cap = maskIntersect(p.poly(headOutline(capHead)), region)
    p.castShadow(cap, face, 0.4, h * 0.035, skinDark, 120)
    p.part(cap, ch["bandana"])
    # little white dots pattern (paisley-like), clipped to the cap
    rnd = random.Random(19)
    dots, draw = p.blank()
    for row in range(5):
        for col in range(8):
            x = cx - w * 0.6 + col * w * 0.16 + (row % 2) * w * 0.08 + rnd.uniform(-1, 1) * w * 0.01
            y = cy - h * 0.62 + row * h * 0.12
            r = w * 0.016
            draw.ellipse(art_kit.box(x - r, y - r, x + r, y + r), fill=255)
    p.fill(maskIntersect(dots, maskShrink(cap, w * 0.02)), (255, 236, 200), 210)
    # folded hem along the lower edge
    hem = [(x, y - h * 0.035) for x, y in edge]
    p.fill(maskIntersect(p.limb(hem, h * 0.035), cap), darkOf(ch["bandana"], 0.1), 200)
    # gloss on the top of the cap
    shine = p.ellipse(cx - w * 0.12, cy - h * 0.40, w * 0.2, h * 0.07, -12)
    p.fill(maskIntersect(shine, cap), lightOf(ch["bandana"], 0.5), 160)
    # knot at the back of the head
    kx = cx - w * 0.47
    ky = cy - h * 0.15
    knot = p.ellipse(kx, ky, w * 0.075, h * 0.08)
    p.part(knot, ch["bandana"], line=p.line * 0.9)


def drawSlickHair(p, ch, head):
    """Dato' Leon: grey hair slicked back with comb lines and a side parting."""
    cx = head["cx"]
    cy = head["cy"]
    w = head["w"]
    h = head["h"]
    t = head["turn"]
    face = p.masks["face"]
    line = [(cx - 0.64 * w, cy + 0.10 * h), (cx - 0.47 * w, cy - 0.05 * h),
            (cx - 0.22 * w + 0.18 * w * t, cy - 0.26 * h), (cx + 0.16 * w + 0.12 * w * t, cy - 0.31 * h),
            (cx + 0.47 * w, cy - 0.17 * h), (cx + 0.64 * w, cy + 0.06 * h)]
    edge = catmullRom(line, False, 10)
    region = p.poly(edge + [(cx + w, cy - h), (cx - w, cy - h)])
    capHead = makeHead(cx, cy - h * 0.03, w * 1.07, h * 1.08, t)
    hair = maskIntersect(p.poly(headOutline(capHead)), region)
    quiff = p.ellipse(cx + w * 0.05 + w * 0.08 * t, cy - h * 0.43, w * 0.40, h * 0.14, -6)
    hair = maskUnion(hair, quiff)
    p.castShadow(hair, face, 0.5, h * 0.04, skinShadeOf(ch["skin"]), 130)
    p.part(hair, ch["hair"])
    # comb lines sweeping back (lighter and darker strands)
    for i in range(4):
        y0 = cy - h * (0.30 + i * 0.065)
        strand = art_kit.bezier((cx + w * (0.38 - i * 0.04), y0 + h * 0.08),
                                (cx - w * 0.05, y0 - h * 0.06), (cx - w * 0.46, y0 + h * 0.14), steps=14)
        p.fill(maskIntersect(p.limb(strand, w * 0.014), maskShrink(hair, w * 0.02)), darkOf(ch["hair"], 0.08), 190)
    shine = art_kit.bezier((cx + w * 0.26, cy - h * 0.36), (cx - w * 0.02, cy - h * 0.52),
                           (cx - w * 0.30, cy - h * 0.34), steps=14)
    p.fill(maskIntersect(p.limb(shine, w * 0.03), hair), WHITE, 150)
    # side parting
    px = cx - w * 0.18 + w * 0.15 * t
    p.stroke(art_kit.bezier((px, cy - h * 0.28), (px - w * 0.04, cy - h * 0.42), (px - w * 0.12, cy - h * 0.5),
                            steps=10), darkOf(ch["hair"], 0.3), w * 0.012)


def drawBaldHair(p, ch, head):
    """Uncle Lim: bald shiny top, grey tufts over the ears, three brave hairs."""
    cx = head["cx"]
    cy = head["cy"]
    w = head["w"]
    h = head["h"]
    t = head["turn"]
    tufts = []
    if t < 0.15:
        sides = [(-1, 1.0), (1, 1.0)]
    else:
        sides = [(-1, 1.25), (1, 0.7)]
    for side, size in sides:
        x = cx + side * w * (0.47 - 0.05 * t * side)
        y = cy - h * 0.06
        blob = p.circle(x, y, w * 0.07 * size)
        blob = maskUnion(blob, p.circle(x - side * w * 0.02, y + h * 0.1, w * 0.065 * size))
        blob = maskUnion(blob, p.circle(x + side * w * 0.01, y - h * 0.08, w * 0.05 * size))
        tufts.append(blob)
    for blob in tufts:
        p.part(blob, ch["hair"], line=p.line * 0.85)
    # the three famous hairs on top
    top = cy - h * 0.5
    for dx, curl in ((-0.07, -1), (0.0, 1), (0.07, 1)):
        x = cx + w * (dx + 0.06 * t)
        strand = art_kit.bezier((x, top + h * 0.03), (x + curl * w * 0.02, top - h * 0.1),
                                (x + curl * w * 0.07, top - h * 0.08), steps=12)
        p.fill(p.taper(strand, w * 0.018, w * 0.008), darkOf(ch["hair"], 0.2))
    # shine on the bald head
    p.fill(p.ellipse(cx - w * 0.17, cy - h * 0.33, w * 0.1, h * 0.055, -30), WHITE, 190)
    p.fill(p.circle(cx - w * 0.04, cy - h * 0.40, w * 0.018), WHITE, 190)
    # forehead lines
    dark = skinShadeOf(ch["skin"])
    for k in range(2):
        y = cy - h * (0.20 + 0.06 * k)
        x = cx + w * 0.1 * t
        p.stroke(art_kit.bezier((x - w * 0.12, y + h * 0.01), (x, y - h * 0.02), (x + w * 0.12, y + h * 0.01),
                                steps=10), dark, w * 0.012, 170)


def bunCentre(head):
    t = head["turn"]
    return (head["cx"] - head["w"] * 1.2 * t, head["cy"] - head["h"] * (0.52 - 0.72 * t))


def drawBunBack(p, ch, head):
    """Auntie Kavitha: hair bun decorated with a string of jasmine (malli poo)."""
    w = head["w"]
    bx, by = bunCentre(head)
    bun = p.circle(bx, by, w * 0.2)
    p.part(bun, ch["hair"])
    for k in range(3):
        a0 = 200 + k * 40
        arc = ellipsePoints(bx, by, w * 0.13, w * 0.12, 10, a0, a0 + 28)
        p.stroke(arc, lightOf(ch["hair"], 0.3), w * 0.012, 160)
    # jasmine buds around the lower half of the bun
    for i in range(9):
        a = math.radians(10 + i * 20)
        x = bx + w * 0.2 * math.cos(a)
        y = by + w * 0.17 * math.sin(a) + w * 0.02
        bud = p.ellipse(x, y, w * 0.035, w * 0.03, i * 25)
        p.part(bud, ch["jasmine"], line=p.line * 0.5, shade=0.8, light=0)
        if i % 3 == 1:
            leaf = p.ellipse(x + w * 0.03, y + w * 0.02, w * 0.025, w * 0.012, 40)
            p.part(leaf, (90, 160, 80), line=p.line * 0.4, shade=0, light=0)


def drawBunFront(p, ch, head):
    cx = head["cx"]
    cy = head["cy"]
    w = head["w"]
    h = head["h"]
    t = head["turn"]
    face = p.masks["face"]
    px = cx + w * 0.12 * t
    line = [(cx - 0.62 * w, cy + 0.12 * h), (cx - 0.48 * w, cy + 0.04 * h), (px - 0.30 * w, cy - 0.20 * h),
            (px, cy - 0.29 * h), (px + 0.30 * w * (1 - 0.3 * t), cy - 0.20 * h), (cx + 0.48 * w, cy + 0.02 * h),
            (cx + 0.62 * w, cy + 0.10 * h)]
    edge = catmullRom(line, False, 10)
    region = p.poly(edge + [(cx + w, cy - h), (cx - w, cy - h)])
    capHead = makeHead(cx, cy - h * 0.02, w * 1.05, h * 1.06, t)
    hair = maskIntersect(p.poly(headOutline(capHead)), region)
    p.castShadow(hair, face, 0.5, h * 0.04, skinShadeOf(ch["skin"]), 130)
    p.part(hair, ch["hair"])
    # middle parting and glossy bands
    p.stroke([(px, cy - h * 0.29), (px - w * 0.03, cy - h * 0.52)], lightOf(ch["hair"], 0.25), w * 0.012)
    for side in (-1, 1):
        shine = art_kit.bezier((px + side * w * 0.06, cy - h * 0.42), (px + side * w * 0.22, cy - h * 0.40),
                               (px + side * w * 0.36, cy - h * 0.20), steps=12)
        p.fill(maskIntersect(p.limb(shine, w * 0.025), hair), lightOf(ch["hair"], 0.35), 170)


def drawEarring(p, ch, x, y, w):
    """Jhumka: small gold bell-shaped earring."""
    p.stroke([(x, y), (x, y + w * 0.035)], OUTLINE, w * 0.012)
    bell = p.poly([(x - w * 0.035, y + w * 0.085), (x - w * 0.025, y + w * 0.05), (x, y + w * 0.035),
                   (x + w * 0.025, y + w * 0.05), (x + w * 0.035, y + w * 0.085)], smooth=False)
    p.part(bell, GOLD, line=p.line * 0.5, shade=0.8, light=0)
    for k in range(3):
        p.fill(p.circle(x - w * 0.025 + k * w * 0.025, y + w * 0.1, w * 0.01), darkOf(GOLD, 0.1))


def drawTudung(p, ch, head):
    """Tudung / hijab: fully covers the hair, ears and neck, and drapes over
    the shoulders and chest. Only the face opening shows skin."""
    cx = head["cx"]
    cy = head["cy"]
    w = head["w"]
    h = head["h"]
    t = head["turn"]
    sy = head["shoulderY"]
    drop = head["drapeBottom"]
    dw = head["drapeW"]
    fx = cx + w * 0.12 * t

    crown = p.ellipse(cx - w * 0.05 * t, cy - h * 0.05, w * 0.58, h * 0.57)
    jaw = p.poly([(cx - w * 0.57, cy - h * 0.08), (cx + w * 0.58, cy - h * 0.08), (cx + w * 0.54, cy + h * 0.36),
                  (fx + w * 0.16, cy + h * 0.66), (fx - w * 0.16, cy + h * 0.66), (cx - w * 0.56, cy + h * 0.36)],
                 smooth=True)
    tip = 12 if ch.get("drapePointed") else 3
    drape = p.poly([(cx - w * 0.50, cy + h * 0.28), (cx - dw, sy + 8), (cx - dw + 5, drop - 16),
                    (fx - dw * 0.45, drop - 3), (fx, drop + tip), (fx + dw * 0.45, drop - 3),
                    (cx + dw - 5, drop - 16), (cx + dw, sy + 8), (cx + w * 0.50, cy + h * 0.28)], smooth=True)
    scarf = maskUnion(maskUnion(crown, jaw), drape)
    p.part(scarf, ch["tudung"], shade=p.shadeDepth * 1.2)
    p.masks["scarf"] = scarf
    fabric = maskShrink(scarf, w * 0.015)

    # the face opening
    opening = p.ellipse(fx, cy + h * 0.10, w * 0.41, h * 0.44)
    faceShape = maskIntersect(opening, p.poly(headOutline(head)))

    # fabric folds (soft darker curves falling from under the chin)
    for k in (-1, 0, 1):
        fold = art_kit.bezier((fx + k * w * 0.18, cy + h * 0.62), (fx + k * w * 0.34, sy + 18),
                              (fx + k * w * 0.42, drop - 8), steps=16)
        p.fill(maskIntersect(p.limb(fold, w * 0.02), fabric), darkOf(ch["tudung"], 0.12), 140)
    side = art_kit.bezier((cx - w * 0.5, cy - h * 0.3), (cx - w * 0.56, cy + h * 0.1), (cx - w * 0.42, cy + h * 0.5),
                          steps=16)
    p.fill(maskIntersect(p.limb(side, w * 0.02), fabric), darkOf(ch["tudung"], 0.12), 120)

    # floral print for Nenek's lilac tudung
    if ch.get("tudungPattern") == "floral":
        rnd = random.Random(7)
        flowers, fdraw = p.blank()
        centres, cdraw = p.blank()
        bbox = scarf.getbbox()
        x0 = bbox[0] / SS
        y0 = bbox[1] / SS
        x1 = bbox[2] / SS
        y1 = bbox[3] / SS
        step = w * 0.17
        row = 0
        y = y0 + step * 0.4
        while y < y1:
            x = x0 + (row % 2) * step * 0.5
            while x < x1:
                px = x + rnd.uniform(-0.2, 0.2) * step
                py = y + rnd.uniform(-0.2, 0.2) * step
                r = w * rnd.uniform(0.024, 0.032)
                for k in range(5):
                    a = math.radians(k * 72 + row * 20)
                    qx = px + math.cos(a) * r
                    qy = py + math.sin(a) * r
                    fdraw.ellipse(art_kit.box(qx - r * 0.75, qy - r * 0.75, qx + r * 0.75, qy + r * 0.75), fill=255)
                cdraw.ellipse(art_kit.box(px - r * 0.45, py - r * 0.45, px + r * 0.45, py + r * 0.45), fill=255)
                x += step
            y += step * 0.8
            row += 1
        flowerArea = maskSubtract(fabric, maskGrow(opening, w * 0.03))
        half, hdraw = p.blank()
        hdraw.rectangle([0, 0, p.size[0] // 2 + int(w * 0.1 * SS), p.size[1]], fill=255)
        p.fill(maskIntersect(maskIntersect(flowers, flowerArea), half), ch["flowerB"], 200)
        p.fill(maskSubtract(maskIntersect(flowers, flowerArea), half), ch["flowerA"], 220)
        p.fill(maskIntersect(centres, flowerArea), (255, 214, 110), 230)

    # soft highlight on the crown
    shine = p.ellipse(cx - w * 0.2, cy - h * 0.36, w * 0.18, h * 0.08, -25)
    p.fill(maskIntersect(shine, scarf), lightOf(ch["tudung"], 0.5), 120)

    # face inside the opening (the outline doubles as the scarf edge)
    p.part(faceShape, ch["skin"], shade=w * 0.03, light=0, shadeColour=skinShadeOf(ch["skin"]))
    p.masks["face"] = faceShape
    # inner cap band (anak tudung) along the top of the face
    top = p.rect(cx - w, cy - h, cx + w, cy - h * 0.12)
    band = maskIntersect(maskSubtract(maskGrow(faceShape, w * 0.028), faceShape), top)
    band = maskIntersect(maskSubtract(maskShift(faceShape, 0, -h * 0.001), maskShift(faceShape, 0, h * 0.045)), top)
    p.fill(band, ch["innerCap"])
    p.castShadow(band, faceShape, 0.3, h * 0.03, skinShadeOf(ch["skin"]), 110)

    # kerongsang brooch (Nenek) pinned under the chin
    if ch["id"] == "nenek":
        bx = fx
        by = cy + h * 0.74
        brooch = p.circle(bx, by, w * 0.055)
        p.part(brooch, GOLD, line=p.line * 0.7, shade=w * 0.012, light=0)
        for k in range(8):
            a = math.radians(k * 45)
            p.fill(p.circle(bx + math.cos(a) * w * 0.045, by + math.sin(a) * w * 0.045, w * 0.01),
                   lightOf(GOLD, 0.5))
        p.fill(p.circle(bx, by, w * 0.026), (220, 70, 120))
        p.fill(p.circle(bx - w * 0.008, by - w * 0.008, w * 0.009), WHITE, 220)


def drawHead(p, ch, head, expr, which="all"):
    """Back layers (hair behind / bandana tails / bun) and then the head itself.
    which = "back" draws only the back layers, "front" only the rest."""
    style = ch["hairStyle"]
    cx = head["cx"]
    cy = head["cy"]
    w = head["w"]
    h = head["h"]
    t = head["turn"]
    if which in ("all", "back"):
        if style == "bandana":
            drawAimanHairBack(p, ch, head)
        elif style == "bun":
            drawBunBack(p, ch, head)
    if which == "back":
        return

    if style == "tudung":
        drawTudung(p, ch, head)
    else:
        # ears behind the head in front view
        if t < 0.15:
            for side in (-1, 1):
                drawEar(p, ch, cx + side * w * 0.47, cy + h * 0.12, w * 0.15, h * 0.22)
        face = p.poly(headOutline(head))
        p.part(face, ch["skin"], shade=w * 0.045, light=w * 0.012, shadeColour=skinShadeOf(ch["skin"]),
               lightAlpha=70)
        p.masks["face"] = face
        if t >= 0.15:
            earX = cx - w * (0.34 - 0.10 * t)
            ear = drawEar(p, ch, earX, cy + h * 0.10, w * 0.13, h * 0.2)
            p.masks["ear"] = ear

    drawFace(p, ch, head, expr)

    if style == "bandana":
        drawAimanHairFront(p, ch, head)
    elif style == "slick":
        drawSlickHair(p, ch, head)
    elif style == "bald":
        drawBaldHair(p, ch, head)
    elif style == "bun":
        drawBunFront(p, ch, head)

    if ch.get("earrings"):
        if t < 0.15:
            for side in (-1, 1):
                drawEarring(p, ch, cx + side * w * 0.47, cy + h * 0.22, w)
        else:
            drawEarring(p, ch, cx - w * (0.34 - 0.10 * t), cy + h * 0.2, w)

    if "glasses" in ch:
        drawGlasses(p, ch, head)


# ======================================================================
# Hands, arms and props held in hands
# ======================================================================
def drawHand(p, ch, x, y, r, thumbSide=1, colour=None):
    """Mitten hand: a round palm with a thumb bump (reads well when small)."""
    if colour is None:
        colour = ch["skin"]
    palm = p.ellipse(x, y, r, r * 0.92)
    thumb = p.ellipse(x + thumbSide * r * 0.75, y - r * 0.35, r * 0.42, r * 0.5, thumbSide * 30)
    hand = maskUnion(palm, thumb)
    p.part(hand, colour, line=p.line * 0.9, shade=r * 0.25, light=0, shadeColour=skinShadeOf(colour))
    p.stroke([(x - r * 0.35, y + r * 0.1), (x - r * 0.1, y + r * 0.55)], skinShadeOf(colour), r * 0.14, 150)
    return hand


def drawArm(p, ch, shoulder, elbow, hand, thumbSide=1, sleeveColour=None, bangles=False):
    """Front-view arm: shoulder -> elbow -> wrist, sleeve drawn on top."""
    aw = ch.get("armW", 19)
    skin = ch["skin"]
    wrist = lerpPoint(elbow, hand, 0.78)
    sleeve = ch.get("sleeve", "short")
    if sleeveColour is None:
        sleeveColour = ch.get("sleeveColour", WHITE)
    arm = p.limb([shoulder, elbow, wrist], aw)
    p.part(arm, skin, shadeColour=skinShadeOf(skin))
    if sleeve == "short":
        end = lerpPoint(shoulder, elbow, 0.62)
        cuffMask = p.limb([shoulder, end], aw + 5)
        p.part(cuffMask, sleeveColour)
        p.stroke([lerpPoint(shoulder, end, 0.9), end], darkOf(sleeveColour, 0.05), 1.0, 0)
    elif sleeve == "long":
        end = lerpPoint(elbow, wrist, 0.9)
        cuffStart = lerpPoint(elbow, wrist, 0.68)
        sleeveMask = p.limb([shoulder, elbow, end], aw + 3)
        p.part(sleeveMask, sleeveColour)
        cuffMask = p.limb([cuffStart, end], aw + 4)
        p.part(cuffMask, ch.get("cuff", sleeveColour), line=p.line * 0.8)
    if bangles:
        dx = hand[0] - elbow[0]
        dy = hand[1] - elbow[1]
        length = math.hypot(dx, dy)
        nx = -dy / length
        ny = dx / length
        for k, colour in enumerate((GOLD, (214, 40, 70), GOLD)):
            c = lerpPoint(elbow, hand, 0.62 + k * 0.06)
            ring = [(c[0] - nx * aw * 0.55, c[1] - ny * aw * 0.55), (c[0] + nx * aw * 0.55, c[1] + ny * aw * 0.55)]
            p.stroke(ring, OUTLINE, 3.2)
            p.stroke(ring, colour, 1.8)
    return drawHand(p, ch, hand[0], hand[1], aw * 0.56, thumbSide)


def drawSpoon(p, base, tip, colour, food=None, width=4.0):
    """Spoon from the hand (base) to the bowl (tip)."""
    handle = p.limb([base, lerpPoint(base, tip, 0.8)], width)
    p.part(handle, colour, line=p.line * 0.6, shade=1.0, light=0)
    angle = math.degrees(math.atan2(tip[1] - base[1], tip[0] - base[0]))
    bowl = p.ellipse(tip[0], tip[1], width * 2.3, width * 1.5, angle)
    p.part(bowl, colour, line=p.line * 0.7, shade=1.4, light=0)
    p.fill(p.ellipse(tip[0] - width * 0.6, tip[1] - width * 0.5, width * 0.8, width * 0.4, angle), WHITE, 200)
    if food is not None:
        heap = p.ellipse(tip[0], tip[1] - width * 0.9, width * 1.9, width * 1.1)
        p.part(heap, food, line=p.line * 0.5, shade=1.0, light=0)


def drawChopsticks(p, base, tip, noodles=False):
    for k in (-1, 1):
        stickTip = (tip[0] + k * 2.2, tip[1] + k * 0.5)
        stick = p.limb([(base[0] + k * 1.5, base[1]), stickTip], 3.0)
        p.part(stick, (196, 132, 78), line=p.line * 0.5, shade=0.8, light=0)
    if noodles:
        for k in range(4):
            x = tip[0] - 4 + k * 2.6
            strand = []
            for j in range(9):
                strand.append((x + 1.6 * math.sin(j * 1.3 + k), tip[1] + j * 2.4))
            p.stroke(strand, OUTLINE, 3.0)
            p.stroke(strand, (246, 226, 170), 1.8)


def drawRotiPiece(p, x, y, size):
    """A torn piece of roti canai dipped in curry (eaten by hand)."""
    rnd = random.Random(5)
    pts = []
    for i in range(9):
        a = math.radians(i * 40)
        r = size * rnd.uniform(0.75, 1.1)
        pts.append((x + math.cos(a) * r * 1.2, y + math.sin(a) * r * 0.8))
    roti = p.poly(pts, smooth=True)
    p.part(roti, (240, 196, 110), line=p.line * 0.7)
    for i in range(5):
        p.fill(p.circle(x + rnd.uniform(-0.7, 0.7) * size, y + rnd.uniform(-0.4, 0.5) * size, size * 0.12),
               (196, 130, 60), 200)
    curry = maskIntersect(roti, p.ellipse(x + size * 0.8, y - size * 0.5, size * 0.7, size * 0.6))
    p.fill(curry, (226, 120, 40), 235)


def drawPhone(p, ch, cx, cy, w, h, angle, showBack=True):
    """Farah's phone (seen from the back: coral case + camera lenses)."""
    body = p.poly(roundRectPoints(cx, cy, w, h, w * 0.22, angle))
    p.part(body, ch["phone"], line=p.line * 0.85, shade=w * 0.1, light=w * 0.05)
    if showBack:
        mx, my = rotatePoint(cx - w * 0.18, cy - h * 0.3, angle, cx, cy)
        module = p.poly(roundRectPoints(mx, my, w * 0.5, w * 0.5, w * 0.12, angle))
        p.part(module, (46, 40, 56), line=0, shade=0, light=0)
        for ox, oy in ((-0.1, -0.1), (0.1, 0.1)):
            lx, ly = rotatePoint(mx + ox * w, my + oy * w, angle, mx, my)
            lens = p.circle(lx, ly, w * 0.09)
            p.fill(lens, (90, 110, 150))
            p.fill(p.circle(lx - w * 0.03, ly - w * 0.03, w * 0.03), WHITE, 220)
        fx, fy = rotatePoint(mx + w * 0.12, my - w * 0.12, angle, mx, my)
        p.fill(p.circle(fx, fy, w * 0.04), (255, 240, 180))


# ======================================================================
# AIMAN - side view walk cycle and idle
# ======================================================================
FRAME_W = 128
FRAME_H = 160
BASE_Y = 154          # where the soles touch the ground (6 px padding)
HIP_X = 60
LEG_LENGTH = 30
FOOT_H = 8


def aimanWalkPose(frame):
    """Walk cycle maths: phase = frame / 8 of a full circle.
    Legs swing +-25 degrees with sin(phase); arms swing the opposite way;
    the leg that moves forward (cos > 0) is lifted a little."""
    phase = frame / 8.0 * 2 * math.pi
    s = math.sin(phase)
    c = math.cos(phase)
    return {"legNear": 25 * s, "legFar": -25 * s, "liftNear": 6 * max(0.0, c), "liftFar": 6 * max(0.0, -c),
            "armNear": -24 * s, "armFar": 24 * s, "breath": 0.0, "flutter": math.sin(phase * 2),
            "apronSwing": 2.0 * s, "eyes": "open"}


def aimanIdlePose(frame):
    """Idle: chest rises and falls (breathing) and frame 3 is a blink."""
    breath = [0.0, -1.0, -2.0, -1.0][frame]
    return {"legNear": 3.0, "legFar": -3.0, "liftNear": 0.0, "liftFar": 0.0,
            "armNear": -4.0 + breath, "armFar": 6.0 - breath, "breath": breath,
            "flutter": [0.0, 0.5, 1.0, 0.5][frame], "apronSwing": 0.0,
            "eyes": "closed" if frame == 3 else "open"}


def drawSideLeg(p, ch, hip, angle, lift, far):
    a = math.radians(angle)
    length = LEG_LENGTH - lift
    ankle = (hip[0] + length * math.sin(a), hip[1] + length * math.cos(a))
    knee = (hip[0] + length * 0.5 * math.sin(a) + lift * 0.8, hip[1] + length * 0.5 * math.cos(a))
    pants = ch["pants"]
    shoe = ch["shoe"]
    if far:
        pants = darkOf(pants, 0.25)
        shoe = darkOf(shoe, 0.18)
    # sneaker (rotated a little: toe lifts while the foot swings forward)
    footAngle = -lift * 1.6
    ax = ankle[0]
    ay = ankle[1]
    outline = [(-6, -4), (-7.5, 3), (-7, FOOT_H - 1.5), (14, FOOT_H - 1.5), (16, FOOT_H - 4.5), (14, -0.5),
               (7, -2.5), (2, -5)]
    shoePts = []
    for ox, oy in outline:
        shoePts.append(rotatePoint(ax + ox, ay + oy, footAngle, ax, ay))
    shoeMask = p.poly(shoePts, smooth=True)
    p.part(shoeMask, shoe, shade=1.6, light=0.7)
    solePts = []
    for ox, oy in ((-9, FOOT_H - 4.2), (18, FOOT_H - 4.2), (18, FOOT_H + 2), (-9, FOOT_H + 2)):
        solePts.append(rotatePoint(ax + ox, ay + oy, footAngle, ax, ay))
    p.fill(maskIntersect(p.poly(solePts), shoeMask), ch["sole"] if not far else darkOf(ch["sole"], 0.2))
    stripe = [rotatePoint(ax - 3, ay + 1.5, footAngle, ax, ay), rotatePoint(ax + 8, ay - 0.5, footAngle, ax, ay)]
    p.fill(maskIntersect(p.limb(stripe, 2.2), shoeMask), ch["shoeStripe"] if not far else darkOf(ch["shoeStripe"], 0.2))
    # trouser leg on top of the shoe collar
    legMask = p.limb([hip, knee, (ankle[0], ankle[1] - 1)], 13)
    p.part(legMask, pants, shade=2.2, light=0.8)
    cuff = p.limb([lerpPoint(knee, ankle, 0.82), (ankle[0], ankle[1] - 1)], 14)
    p.fill(maskIntersect(cuff, legMask), darkOf(pants, 0.12), 180)


def drawSideArm(p, ch, shoulder, angle, far):
    a = math.radians(angle)
    elbow = (shoulder[0] + 13 * math.sin(a), shoulder[1] + 13 * math.cos(a))
    bend = 14 + max(0.0, angle) * 0.6
    b = math.radians(angle + bend)
    wrist = (elbow[0] + 12 * math.sin(b), elbow[1] + 12 * math.cos(b))
    skin = ch["skin"]
    shirt = ch["shirt"]
    if far:
        skin = darkOf(skin, 0.2)
        shirt = darkOf(shirt, 0.2)
    fore = p.limb([elbow, wrist], 9.5)
    p.part(fore, skin, shade=1.8, light=0, shadeColour=skinShadeOf(skin))
    handC = (wrist[0] + 2.5 * math.sin(b), wrist[1] + 2.5 * math.cos(b))
    hand = p.circle(handC[0], handC[1], 5.4)
    p.part(hand, skin, shade=1.4, light=0, shadeColour=skinShadeOf(skin))
    sleeve = p.limb([shoulder, lerpPoint(shoulder, elbow, 0.95)], 13)
    p.part(sleeve, shirt, shade=2.0, light=0.8)


def drawAimanSide(ch, pose):
    p = Painter(FRAME_W, FRAME_H, line=2.2, shade=2.4, light=0.9)
    spread = max(abs(pose["legNear"]), abs(pose["legFar"]))
    # hip height keeps the planted foot exactly on the baseline -> no jitter
    hipY = BASE_Y - FOOT_H - LEG_LENGTH * math.cos(math.radians(spread))
    upperY = hipY + pose["breath"]
    shoulderY = upperY - 36
    head = makeHead(HIP_X + 5, shoulderY - 24, 64, 58, 0.55)
    head["flutter"] = pose["flutter"]
    expr = {"eyes": pose["eyes"], "mouth": "grin", "brows": (0.02, 0.02)}

    # soft ground shadow (drawn on the "under" layer, no outline)
    p.softEllipse(HIP_X + 3, BASE_Y - 1, 25, 4.5, (18, 10, 30), 110, 2.0, target=p.under)

    drawSideArm(p, ch, (HIP_X - 1, shoulderY + 6), pose["armFar"], True)
    drawSideLeg(p, ch, (HIP_X - 3, hipY), pose["legFar"], pose["liftFar"], True)
    drawHead(p, ch, head, expr, "back")

    # hips + T-shirt torso
    hips = p.rect(HIP_X - 13, hipY - 7, HIP_X + 13, hipY + 8, radius=6)
    p.part(hips, ch["pants"], shade=2.0)
    torso = p.poly([(HIP_X - 13, hipY + 3), (HIP_X - 15, upperY - 22), (HIP_X - 10, upperY - 35),
                    (HIP_X + 8, upperY - 36), (HIP_X + 15, upperY - 25), (HIP_X + 14, hipY + 3)], smooth=True)
    p.part(torso, ch["shirt"])

    drawSideLeg(p, ch, (HIP_X + 3, hipY), pose["legNear"], pose["liftNear"], False)

    # red apron: bib + long skirt that sways with the stride
    sw = pose["apronSwing"]
    bib = [(HIP_X - 2, upperY - 31), (HIP_X + 12, upperY - 31), (HIP_X + 15, hipY - 8), (HIP_X - 8, hipY - 8)]
    skirt = [(HIP_X - 12, hipY - 11), (HIP_X + 16, hipY - 11), (HIP_X + 18 + sw, hipY + 16),
             (HIP_X + 14 + sw, hipY + 20), (HIP_X - 8 + sw, hipY + 20), (HIP_X - 11 + sw, hipY + 16)]
    apron = maskUnion(p.poly(bib), p.poly(skirt, smooth=False))
    apron = maskShrink(maskGrow(apron, 1.5), 1.5)
    p.part(apron, ch["apron"])
    # waist tie and bow at the back
    band = p.limb([(HIP_X - 13, hipY - 10), (HIP_X + 16, hipY - 10)], 3.2)
    p.fill(maskIntersect(band, maskGrow(apron, 0.5)), darkOf(ch["apron"], 0.12))
    for dy in (-3, 3):
        loop = p.ellipse(HIP_X - 17, hipY - 10 + dy, 4, 2.4, dy * 8)
        p.part(loop, ch["apron"], line=1.6, shade=0.8, light=0)
    # pocket with the golden Aura Ladle handle peeking out
    handle = p.limb([(HIP_X + 8, hipY + 3), (HIP_X + 11, hipY - 12)], 2.6)
    p.part(handle, GOLD, line=1.5, shade=0.6, light=0)
    p.part(p.circle(HIP_X + 11.3, hipY - 13, 2.4), GOLD, line=1.5, shade=0.5, light=0)
    pocket = p.rect(HIP_X + 1 + sw * 0.5, hipY, HIP_X + 14 + sw * 0.5, hipY + 10, radius=2)
    p.part(pocket, ch["apron"], line=1.4, shade=1.2, light=0)
    p.stroke([(HIP_X + 3 + sw * 0.5, hipY + 2), (HIP_X + 12 + sw * 0.5, hipY + 2)], lightOf(ch["apron"], 0.4), 0.8, 200)
    # neck strap
    p.stroke([(HIP_X + 1, upperY - 31), (HIP_X + 3, upperY - 38)], darkOf(ch["apron"], 0.1), 2.4)

    drawHead(p, ch, head, expr, "front")
    drawSideArm(p, ch, (HIP_X + 3, shoulderY + 6), pose["armNear"], False)
    return p.result(0.9)


# ======================================================================
# SEATED GUESTS (front view, 4 frames: idle, blink, eat, cheer)
# ======================================================================
GUEST_W = 180
GUEST_H = 220
GUEST_EXPRESSIONS = [
    {"eyes": "open", "mouth": None},
    {"eyes": "closed", "mouth": None},
    {"eyes": "open", "mouth": "eat", "brows": (0.02, 0.02)},
    {"eyes": "happy", "mouth": "cheer", "brows": (0.04, 0.04), "blush": 170},
]


def guestArmPoints(pose, side, cx):
    """side = -1 for the arm on the viewer's left (the character's right arm)."""
    if pose == "rest":
        points = [(cx - 50, 138), (cx - 58, 182), (cx - 26, 200)]
    elif pose == "eat":
        points = [(cx - 50, 138), (cx - 62, 178), (cx - 24, 132)]
    elif pose == "up":
        points = [(cx - 48, 136), (cx - 69, 94), (cx - 63, 48)]
    else:  # "phone": hand held at the chest
        points = [(cx - 50, 138), (cx - 60, 180), (cx - 34, 170)]
    if side == 1:
        mirrored = []
        for pt in points:
            mirrored.append(mirrorPoint(pt, cx))
        return mirrored
    return points


def drawGuestTorso(p, ch, cx, sy):
    """Clothes for the seated front view (cut at the waist)."""
    cid = ch["id"]
    if cid == "leon":
        torso = p.poly([(cx - 18, sy - 9), (cx - 50, sy - 3), (cx - 60, sy + 14), (cx - 56, 236), (cx + 56, 236),
                        (cx + 60, sy + 14), (cx + 50, sy - 3), (cx + 18, sy - 9)], smooth=True)
        p.part(torso, ch["suit"])
        shirt = maskIntersect(p.poly([(cx - 16, sy - 10), (cx + 16, sy - 10), (cx + 3, sy + 58), (cx - 3, sy + 58)]),
                              torso)
        p.part(shirt, ch["shirt"], line=p.line * 0.7, shade=1.5)
        for side in (-1, 1):
            lapel = [(cx + side * 16, sy - 10), (cx + side * 3, sy + 58), (cx + side * 13, sy + 40),
                     (cx + side * 27, sy + 12), (cx + side * 24, sy - 4)]
            p.part(maskIntersect(p.poly(lapel), torso), lightOf(ch["suit"], 0.3), line=p.line * 0.7, shade=1.6)
        for by in (sy + 68, sy + 84):
            p.part(p.circle(cx, by, 3.2), GOLD, line=1.4, shade=0.8, light=0)
        # gold pocket square and a small lapel pin
        pocket = p.poly([(cx + 24, sy + 34), (cx + 30, sy + 25), (cx + 34, sy + 31), (cx + 39, sy + 24),
                         (cx + 42, sy + 34)])
        p.part(pocket, GOLD, line=1.4, shade=1.0, light=0)
        p.stroke([(cx + 22, sy + 34), (cx + 44, sy + 34)], darkOf(ch["suit"], 0.2), 1.6)
        p.part(p.circle(cx - 22, sy + 20, 2.6), GOLD, line=1.2, shade=0, light=0)
        drawBowTie(p, cx, sy - 7, 1.0)
    elif cid == "lim":
        torso = p.poly([(cx - 20, sy - 9), (cx - 56, sy - 2), (cx - 68, sy + 18), (cx - 70, 236), (cx + 70, 236),
                        (cx + 68, sy + 18), (cx + 56, sy - 2), (cx + 20, sy - 9)], smooth=True)
        p.part(torso, ch["skin"], shadeColour=skinShadeOf(ch["skin"]))
        singlet = p.poly([(cx - 38, sy - 8), (cx - 27, sy - 8), (cx - 14, sy + 16), (cx, sy + 22), (cx + 14, sy + 16),
                          (cx + 27, sy - 8), (cx + 38, sy - 8), (cx + 48, sy + 34), (cx + 62, sy + 48),
                          (cx + 64, 236), (cx - 64, 236), (cx - 62, sy + 48), (cx - 48, sy + 34)])
        singlet = maskIntersect(maskShrink(maskGrow(singlet, 3), 3), torso)
        p.part(singlet, ch["singlet"], line=p.line * 0.8, shade=3.0, shadeColour=(206, 212, 226))
        for k in range(-5, 6):
            x = cx + k * 10
            p.fill(maskIntersect(p.limb([(x, sy + 20), (x, 236)], 1.2), maskShrink(singlet, 2)), (220, 224, 234), 160)
        belly = art_kit.bezier((cx - 44, sy + 76), (cx, sy + 96), (cx + 44, sy + 76), steps=16)
        p.fill(maskIntersect(p.limb(belly, 2.0), singlet), (200, 206, 222), 200)
    elif cid == "kavitha":
        torso = p.poly([(cx - 18, sy - 9), (cx - 48, sy - 3), (cx - 57, sy + 14), (cx - 53, 236), (cx + 53, 236),
                        (cx + 57, sy + 14), (cx + 48, sy - 3), (cx + 18, sy - 9)], smooth=True)
        p.part(torso, ch["blouse"])
        neck = maskIntersect(p.poly(art_kit.bezier((cx - 22, sy - 12), (cx, sy + 36), (cx + 22, sy - 12), steps=16)),
                             torso)
        p.part(neck, ch["skin"], line=p.line * 0.7, shade=2.0, light=0, shadeColour=skinShadeOf(ch["skin"]))
        chain = art_kit.bezier((cx - 17, sy - 6), (cx, sy + 26), (cx + 17, sy - 6), steps=16)
        p.stroke(chain, OUTLINE, 3.0)
        p.stroke(chain, GOLD, 1.8)
        pend = p.poly([(cx - 4, sy + 13), (cx + 4, sy + 13), (cx, sy + 22)])
        p.part(pend, (214, 40, 70), line=1.4, shade=0, light=0)
        p.fill(p.circle(cx, sy + 13, 2.2), GOLD)
    elif cid == "farah":
        torso = p.poly([(cx - 18, sy - 9), (cx - 48, sy - 3), (cx - 57, sy + 14), (cx - 53, 236), (cx + 53, 236),
                        (cx + 57, sy + 14), (cx + 48, sy - 3), (cx + 18, sy - 9)], smooth=True)
        p.part(torso, ch["top"])
        for side in (-1, 1):
            panel = [(cx + side * 12, sy - 6), (cx + side * 48, sy - 3), (cx + side * 58, sy + 14),
                     (cx + side * 54, 236), (cx + side * 14, 236), (cx + side * 11, sy + 30)]
            panelMask = maskIntersect(p.poly(panel), torso)
            p.part(panelMask, ch["cardigan"], line=p.line * 0.8)
            for k in range(3):
                p.fill(p.circle(cx + side * 17, sy + 48 + k * 14, 1.8), darkOf(ch["cardigan"], 0.25))
        # APU student lanyard with an ID card
        for side in (-1, 1):
            p.stroke([(cx + side * 12, sy + 20), (cx + side * 4, sy + 58)], OUTLINE, 3.6)
            p.stroke([(cx + side * 12, sy + 20), (cx + side * 4, sy + 58)], ch["lanyard"], 2.4)
        card = p.rect(cx - 9, sy + 56, cx + 9, sy + 80, radius=2)
        p.part(card, (252, 252, 255), line=1.6, shade=1.0, light=0)
        p.fill(p.rect(cx - 9, sy + 56, cx + 9, sy + 62), (214, 48, 49))
        p.fill(p.rect(cx - 6, sy + 65, cx + 1, sy + 73), (140, 170, 210))
        p.stroke([(cx + 3, sy + 67), (cx + 7, sy + 67)], (160, 160, 180), 1.0)
        p.stroke([(cx + 3, sy + 71), (cx + 7, sy + 71)], (160, 160, 180), 1.0)
    elif cid == "nenek":
        torso = p.poly([(cx - 18, sy - 9), (cx - 50, sy - 3), (cx - 60, sy + 14), (cx - 60, 236), (cx + 60, 236),
                        (cx + 60, sy + 14), (cx + 50, sy - 3), (cx + 18, sy - 9)], smooth=True)
        p.part(torso, ch["outfit"])
        for side in (-1, 1):
            fold = [(cx + side * 30, sy + 50), (cx + side * 34, 236)]
            p.fill(maskIntersect(p.limb(fold, 1.6), torso), darkOf(ch["outfit"], 0.1), 150)


def drawBowTie(p, x, y, scale):
    for side in (-1, 1):
        wing = p.poly([(x, y), (x + side * 13 * scale, y - 7 * scale), (x + side * 15 * scale, y),
                       (x + side * 13 * scale, y + 7 * scale)], smooth=True)
        p.part(wing, GOLD, line=1.6 * scale, shade=1.4 * scale, light=0.6 * scale)
    knot = p.ellipse(x, y, 3.6 * scale, 4.2 * scale)
    p.part(knot, darkOf(GOLD, 0.05), line=1.4 * scale, shade=0.8, light=0)


def drawGuestOverArms(p, ch, cx, sy):
    """Clothes that lie on top of the resting arms (towel, saree pallu)."""
    cid = ch["id"]
    if cid == "lim":
        towel = p.poly([(cx - 60, sy - 2), (cx - 44, sy - 12), (cx - 30, sy - 8), (cx - 28, sy + 8),
                        (cx - 32, sy + 66), (cx - 52, sy + 70), (cx - 56, sy + 20)], smooth=True)
        p.part(towel, ch["towel"], shade=2.6, shadeColour=(214, 220, 232))
        for k, colour in enumerate((ch["towelStripe"], ch["towelRed"], ch["towelStripe"])):
            y = sy + 48 + k * 6
            p.fill(maskIntersect(p.limb([(cx - 60, y), (cx - 26, y - 2)], 2.2), towel), colour)
        p.fill(maskIntersect(p.limb([(cx - 54, sy + 2), (cx - 36, sy)], 1.2), towel), (206, 212, 226), 200)
    elif cid == "kavitha":
        # saree pallu: a wide band from the waist over the LEFT shoulder
        centre = [(cx - 40, 240), (cx - 6, sy + 60), (cx + 26, sy + 18), (cx + 50, sy - 2)]
        band = p.poly(ribbonPoints(catmullRom(centre, False, 8), 44, 34))
        cap = p.ellipse(cx + 49, sy + 6, 15, 13)
        pallu = maskUnion(band, cap)
        p.part(pallu, ch["saree"])
        inner = maskShrink(pallu, 4.5)
        border = maskSubtract(pallu, inner)
        p.fill(border, ch["sareeBorder"])
        p.fill(maskSubtract(maskShrink(pallu, 2.0), maskShrink(pallu, 3.2)), darkOf(ch["sareeBorder"], 0.2), 200)
        rnd = random.Random(3)
        motifs, draw = p.blank()
        for i in range(40):
            x = rnd.uniform(cx - 60, cx + 70)
            y = rnd.uniform(sy - 10, 236)
            r = 1.5
            draw.ellipse(art_kit.box(x - r, y - r, x + r, y + r), fill=255)
        p.fill(maskIntersect(motifs, maskShrink(pallu, 6)), GOLD, 220)
        for k in range(3):
            fold = art_kit.bezier((cx - 30 + k * 12, 236), (cx + k * 10, sy + 50), (cx + 38 + k * 4, sy + 4), steps=12)
            p.fill(maskIntersect(p.limb(fold, 1.4), inner), darkOf(ch["saree"], 0.18), 170)


def drawHeldItem(p, ch, pose, hand):
    """Utensils in the character's right hand (viewer's left)."""
    cid = ch["id"]
    x, y = hand
    if pose == "eat":
        mouthTip = (GUEST_W / 2 + 2, 104)
        if cid == "leon":
            drawSpoon(p, (x - 6, y + 14), mouthTip, GOLD, food=(250, 210, 120))
        elif cid == "lim":
            drawChopsticks(p, (x - 8, y + 12), (mouthTip[0] - 2, mouthTip[1] + 2), noodles=True)
        elif cid == "kavitha":
            drawRotiPiece(p, x + 12, y - 16, 11)
        else:
            drawSpoon(p, (x - 6, y + 14), mouthTip, (206, 212, 222), food=(252, 250, 240))
    elif pose == "rest":
        if cid == "leon":
            drawSpoon(p, (x - 1, y + 10), (x - 4, y - 30), GOLD)
        elif cid == "lim":
            drawChopsticks(p, (x + 2, y + 8), (x - 14, y - 34))
    elif pose == "up":
        if cid == "leon":
            drawSpoon(p, (x + 1, y + 8), (x + 4, y - 30), GOLD)


def drawGuestFrame(ch, frame):
    p = Painter(GUEST_W, GUEST_H, line=2.5, shade=3.0, light=1.2)
    cx = GUEST_W / 2
    sy = 128
    head = makeHead(cx, 70, 104, 96, 0.0)
    head["shoulderY"] = sy
    if ch["id"] == "nenek":
        head["drapeBottom"] = sy + 64
        head["drapeW"] = 60
    else:
        head["drapeBottom"] = sy + 34
        head["drapeW"] = 56
    expr = dict(GUEST_EXPRESSIONS[frame])
    if expr["mouth"] is None:
        expr["mouth"] = ch.get("mouth", "smile")
    if ch["id"] == "leon" and frame == 2:
        expr["eyes"] = "closed"
        expr["brows"] = (0.05, 0.05)
    if ch["id"] == "leon" and frame in (0, 1):
        expr["brows"] = (0.0, 0.05)
        expr["lid"] = 0.22 if frame == 0 else 0.0
    if ch["id"] == "lim" and frame in (0, 1):
        expr["browTilt"] = (-0.02, -0.02)

    # which pose each arm uses in this frame
    leftPose = ["rest", "rest", "eat", "up"][frame]
    rightPose = ["rest", "rest", "rest", "up"][frame]
    if ch["id"] == "farah" and rightPose == "rest":
        rightPose = "phone"

    # neck (shadow from the chin) and torso
    neck = p.rect(cx - 11, 108, cx + 11, sy + 6, radius=5)
    p.part(neck, ch["skin"], shade=2.0, light=0, shadeColour=skinShadeOf(ch["skin"]))
    p.fill(maskIntersect(neck, p.ellipse(cx, 112, 30, 10)), skinShadeOf(ch["skin"]))
    drawGuestTorso(p, ch, cx, sy)

    # resting arms go under the head; raised arms are drawn after it
    late = []
    for side, pose in ((-1, leftPose), (1, rightPose)):
        if pose in ("rest", "phone"):
            drawGuestArm(p, ch, pose, side, cx)
        else:
            late.append((side, pose))
    drawGuestOverArms(p, ch, cx, sy)
    drawHead(p, ch, head, expr)
    for side, pose in late:
        drawGuestArm(p, ch, pose, side, cx)

    if frame == 3:
        p.sparkle(24, 22, 7)
        p.sparkle(158, 30, 5.5)
        p.sparkle(150, 102, 4)
    return p.result(0.9)


def drawGuestArm(p, ch, pose, side, cx):
    shoulder, elbow, hand = guestArmPoints(pose, side, cx)
    bangles = ch["id"] == "kavitha"
    thumb = -side
    if pose == "up":
        thumb = side
    if side == -1 and pose in ("eat", "rest") and ch["id"] in ("leon", "lim") and pose == "rest":
        drawHeldItem(p, ch, pose, hand)
    if side == -1 and pose in ("eat", "up"):
        drawHeldItem(p, ch, pose, hand)
    if ch["id"] == "farah" and side == 1:
        # the phone sits in her left hand (viewer's right)
        if pose == "phone":
            drawArm(p, ch, shoulder, elbow, hand, thumb, bangles=bangles)
            drawPhone(p, ch, hand[0] + 2, hand[1] - 16, 20, 34, 6)
            drawHand(p, ch, hand[0] + 1, hand[1] - 2, 10, -1)
            return
        if pose == "up":
            drawPhone(p, ch, hand[0] + 1, hand[1] - 18, 18, 30, 8)
    drawArm(p, ch, shoulder, elbow, hand, thumb, bangles=bangles)


# ======================================================================
# PORTRAITS (256x256, head and shoulders, facing slightly right)
# ======================================================================
PORTRAIT_EXPR = {
    "aiman": {"eyes": "open", "mouth": "grin", "brows": (0.03, 0.03), "blush": 130},
    "nenek": {"eyes": "open", "mouth": "smile", "lid": 0.12, "brows": (0.0, 0.0), "browTilt": (0.03, 0.03)},
    "leon": {"eyes": "open", "mouth": "smirk", "lid": 0.25, "brows": (-0.01, 0.07)},
    "lim": {"eyes": "open", "mouth": "open", "brows": (0.02, 0.06), "browTilt": (-0.02, 0.0)},
    "kavitha": {"eyes": "open", "mouth": "grin", "brows": (0.02, 0.03)},
    "farah": {"eyes": "open", "mouth": "open", "brows": (0.04, 0.04), "blush": 150},
}


def drawPortraitBust(p, ch, cx, sy):
    """Shoulders and outfit for a 3/4 bust (the body turns slightly right)."""
    cid = ch["id"]
    bust = p.poly([(cx - 22, sy - 12), (cx - 72, sy - 2), (cx - 98, sy + 26), (cx - 106, 290), (cx + 106, 290),
                   (cx + 96, sy + 24), (cx + 68, sy - 4), (cx + 26, sy - 12)], smooth=True)
    if cid == "aiman":
        p.part(bust, ch["shirt"])
        apron = p.poly([(cx - 40, sy + 14), (cx + 44, sy + 14), (cx + 54, 290), (cx - 52, 290)], smooth=False)
        apron = maskIntersect(maskShrink(maskGrow(apron, 4), 4), bust)
        p.part(apron, ch["apron"])
        for side in (-1, 1):
            strap = [(cx + side * 36, sy + 16), (cx + side * 24 + 3, sy - 12)]
            p.stroke(strap, OUTLINE, 9)
            p.stroke(strap, darkOf(ch["apron"], 0.08), 5.6)
        # bib pocket with the glowing Aura Ladle
        p.softEllipse(cx + 22, sy + 16, 20, 26, (255, 210, 110), 150, 7, target=p.under)
        handle = p.limb([(cx + 18, sy + 46), (cx + 30, sy - 2)], 6)
        p.part(handle, GOLD, line=2.2, shade=1.4, light=0.6)
        p.part(p.circle(cx + 31, sy - 5, 5.5), GOLD, line=2.2, shade=1.2, light=0)
        p.fill(p.circle(cx + 31, sy - 5, 2.3), (214, 48, 49))
        pocket = p.rect(cx - 2, sy + 36, cx + 40, sy + 64, radius=5)
        p.part(pocket, ch["apron"], line=2.4, shade=2.4, light=0)
        p.stroke([(cx + 2, sy + 41), (cx + 36, sy + 41)], lightOf(ch["apron"], 0.4), 1.4, 200)
        p.sparkle(cx + 44, sy - 14, 6)
    elif cid == "nenek":
        p.part(bust, ch["outfit"])
    elif cid == "leon":
        p.part(bust, ch["suit"])
        shirt = maskIntersect(p.poly([(cx - 26, sy - 14), (cx + 30, sy - 14), (cx + 8, 290), (cx - 4, 290)]), bust)
        p.part(shirt, ch["shirt"], line=2.2, shade=2.0)
        for side, x0 in ((-1, cx - 26), (1, cx + 30)):
            lapel = [(x0, sy - 14), (cx + 2 + side * 3, 290), (cx + side * 34, sy + 50), (x0 + side * 22, sy + 8),
                     (x0 + side * 16, sy - 6)]
            p.part(maskIntersect(p.poly(lapel), bust), lightOf(ch["suit"], 0.3), line=2.2, shade=2.4)
        pocket = p.poly([(cx + 50, sy + 44), (cx + 57, sy + 31), (cx + 63, sy + 40), (cx + 69, sy + 30), (cx + 74, sy + 44)])
        p.part(pocket, GOLD, line=2.0, shade=1.4, light=0)
        p.stroke([(cx + 46, sy + 44), (cx + 78, sy + 44)], darkOf(ch["suit"], 0.2), 2.2)
        p.part(p.circle(cx - 42, sy + 24, 4), GOLD, line=1.8, shade=0, light=0)
        drawBowTie(p, cx + 3, sy - 10, 1.5)
    elif cid == "lim":
        p.part(bust, ch["skin"], shadeColour=skinShadeOf(ch["skin"]))
        singlet = p.poly([(cx - 58, sy - 6), (cx - 42, sy - 8), (cx - 20, sy + 30), (cx + 4, sy + 38),
                          (cx + 30, sy + 28), (cx + 46, sy - 8), (cx + 60, sy - 6), (cx + 74, sy + 40),
                          (cx + 86, 290), (cx - 90, 290), (cx - 76, sy + 40)])
        singlet = maskIntersect(maskShrink(maskGrow(singlet, 4), 4), bust)
        p.part(singlet, ch["singlet"], shade=4.0, shadeColour=(206, 212, 226))
        for k in range(-6, 7):
            x = cx + k * 13
            p.fill(maskIntersect(p.limb([(x, sy + 30), (x, 290)], 1.5), maskShrink(singlet, 3)), (220, 224, 234), 160)
        towel = p.poly([(cx - 104, sy + 20), (cx - 88, sy - 6), (cx - 58, sy - 16), (cx - 42, sy - 10),
                        (cx - 40, sy + 12), (cx - 50, 290), (cx - 96, 290)], smooth=True)
        p.part(towel, ch["towel"], shade=3.5, shadeColour=(214, 220, 232))
        for k, colour in enumerate((ch["towelStripe"], ch["towelRed"], ch["towelStripe"])):
            y = sy + 36 + k * 8
            p.fill(maskIntersect(p.limb([(cx - 108, y + 4), (cx - 36, y)], 3.0), towel), colour)
    elif cid == "kavitha":
        p.part(bust, ch["blouse"])
        neck = maskIntersect(p.poly(art_kit.bezier((cx - 30, sy - 16), (cx + 4, sy + 50), (cx + 34, sy - 16),
                                                   steps=16)), bust)
        p.part(neck, ch["skin"], line=2.2, shade=3.0, light=0, shadeColour=skinShadeOf(ch["skin"]))
        chain = art_kit.bezier((cx - 22, sy - 8), (cx + 4, sy + 36), (cx + 28, sy - 8), steps=16)
        p.stroke(chain, OUTLINE, 4.4)
        p.stroke(chain, GOLD, 2.6)
        pend = p.poly([(cx - 3, sy + 18), (cx + 9, sy + 18), (cx + 3, sy + 30)])
        p.part(pend, (214, 40, 70), line=2.0, shade=0, light=0)
        centre = [(cx - 40, 300), (cx + 10, sy + 70), (cx + 56, sy + 20), (cx + 86, sy + 2)]
        band = p.poly(ribbonPoints(catmullRom(centre, False, 8), 70, 54))
        cap = p.ellipse(cx + 82, sy + 12, 22, 18)
        pallu = maskIntersect(maskUnion(band, cap), maskGrow(bust, 3))
        p.part(pallu, ch["saree"])
        inner = maskShrink(pallu, 7)
        p.fill(maskSubtract(pallu, inner), ch["sareeBorder"])
        p.fill(maskSubtract(maskShrink(pallu, 3), maskShrink(pallu, 5)), darkOf(GOLD, 0.2), 200)
        rnd = random.Random(4)
        motifs, draw = p.blank()
        for i in range(50):
            x = rnd.uniform(cx - 60, cx + 110)
            y = rnd.uniform(sy - 10, 260)
            r = 2.2
            draw.ellipse(art_kit.box(x - r, y - r, x + r, y + r), fill=255)
        p.fill(maskIntersect(motifs, maskShrink(pallu, 9)), GOLD, 220)
        for k in range(3):
            fold = art_kit.bezier((cx - 30 + k * 18, 262), (cx + 10 + k * 14, sy + 60), (cx + 60 + k * 6, sy + 8),
                                  steps=12)
            p.fill(maskIntersect(p.limb(fold, 2.0), inner), darkOf(ch["saree"], 0.18), 170)
    elif cid == "farah":
        p.part(bust, ch["top"])
        for side in (-1, 1):
            panel = [(cx + side * 14, sy - 8), (cx + side * 72, sy - 2), (cx + side * 98, sy + 26),
                     (cx + side * 106, 290), (cx + side * 20, 290), (cx + side * 16, sy + 40)]
            p.part(maskIntersect(p.poly(panel), bust), ch["cardigan"])
        for side in (-1, 1):
            p.stroke([(cx + side * 18, sy + 30), (cx + side * 6, sy + 64)], OUTLINE, 5.0)
            p.stroke([(cx + side * 18, sy + 30), (cx + side * 6, sy + 64)], ch["lanyard"], 3.2)


def drawPortrait(ch):
    p = Painter(256, 256, line=3.1, shade=4.0, light=1.6)
    cx = 124
    sy = 196
    head = makeHead(cx, 102, 150, 138, 0.3)
    head["shoulderY"] = sy
    head["drapeW"] = 104
    head["drapeBottom"] = 300 if ch["id"] == "nenek" else 240
    expr = PORTRAIT_EXPR[ch["id"]]

    if ch["id"] == "aiman":
        drawHead(p, ch, head, expr, "back")
    elif ch["id"] == "kavitha":
        drawHead(p, ch, head, expr, "back")
    neck = p.rect(cx - 16 + 4, 150, cx + 16 + 4, sy + 6, radius=8)
    p.part(neck, ch["skin"], shade=3.0, light=0, shadeColour=skinShadeOf(ch["skin"]))
    p.fill(maskIntersect(neck, p.ellipse(cx + 4, 162, 40, 14)), skinShadeOf(ch["skin"]))
    drawPortraitBust(p, ch, cx, sy)
    drawHead(p, ch, head, expr, "front")

    # hands and props in front of the bust
    if ch["id"] == "leon":
        # the golden spoon held thoughtfully near his chin
        sleeve = p.limb([(cx + 118, 300), (cx + 66, 232)], 34)
        p.part(sleeve, ch["suit"])
        cuff = p.limb([(cx + 72, 240), (cx + 64, 229)], 36)
        p.part(cuff, ch["shirt"], line=2.4)
        drawSpoon(p, (cx + 52, 226), (cx + 30, 176), GOLD, width=5.5)
        drawHand(p, ch, cx + 52, 214, 16, -1)
        p.sparkle(cx + 12, 168, 7)
    elif ch["id"] == "farah":
        sleeve = p.limb([(cx + 120, 300), (cx + 94, 236)], 30)
        p.part(sleeve, ch["cardigan"])
        drawPhone(p, ch, cx + 92, 188, 40, 70, -8)
        drawHand(p, ch, cx + 94, 226, 15, -1)
    return p.result(1.0)


# ======================================================================
# OYEN the cat
# ======================================================================
def drawOyenHead(p, head, expr):
    """3/4 cat head: ears, fluffy cheeks, tabby M, white muzzle, green eyes."""
    ch = OYEN
    cx = head["cx"]
    cy = head["cy"]
    w = head["w"]
    h = head["h"]
    t = head["turn"]
    fx = cx + w * 0.16 * t
    fur = ch["fur"]
    # ears (behind the head)
    ears = [((cx - w * 0.44, cy - h * 0.10), (cx - w * 0.36, cy - h * 0.66), (cx - w * 0.06, cy - h * 0.38)),
            ((cx + w * 0.10, cy - h * 0.40), (cx + w * 0.36, cy - h * 0.68), (cx + w * 0.46, cy - h * 0.12))]
    for a, tip, b in ears:
        ear = p.poly([a, lerpPoint(a, tip, 0.8), tip, lerpPoint(b, tip, 0.8), b], smooth=True)
        p.part(ear, fur)
        inner = p.poly([lerpPoint(a, b, 0.2), lerpPoint(lerpPoint(a, b, 0.5), tip, 0.75), lerpPoint(a, b, 0.8)],
                       smooth=True)
        p.fill(maskIntersect(maskShift(inner, 0, h * 0.04), ear), ch["innerEar"])
    # head: a wide round shape with fluffy cheek tufts at the bottom sides
    pts = []
    for i in range(60):
        a = 2 * math.pi * i / 60
        c = math.cos(a)
        s = math.sin(a)
        r = 1.0
        if s > 0.1:
            r = 1.0 + 0.06 * math.sin(a * 9) * s
        pts.append((cx + c * w / 2 * r + t * w * 0.05 * max(0, s), cy + s * h / 2 * r))
    face = p.poly(pts)
    p.part(face, fur, shade=w * 0.05)
    p.masks["face"] = face
    # tabby stripes on the forehead ("M")
    for k in (-1, 0, 1):
        x = fx + k * w * 0.1
        stripe = [(x, cy - h * 0.48), (x + k * w * 0.02, cy - h * 0.34), (x + k * w * 0.01, cy - h * 0.24)]
        p.fill(maskIntersect(p.taper(catmullRom(stripe, False, 6), w * 0.06, w * 0.015), face), ch["stripe"])
    for side in (-1, 1):
        cheek = [(cx + side * w * 0.5, cy + h * 0.02), (cx + side * w * 0.36, cy + h * 0.06)]
        p.fill(maskIntersect(p.taper(cheek, w * 0.05, w * 0.02), face), ch["stripe"])
    # white muzzle and chin
    muzzle = maskUnion(p.ellipse(fx - w * 0.1, cy + h * 0.24, w * 0.15, h * 0.14),
                       p.ellipse(fx + w * 0.1, cy + h * 0.24, w * 0.14, h * 0.14))
    muzzle = maskUnion(muzzle, p.ellipse(fx, cy + h * 0.34, w * 0.12, h * 0.1))
    p.fill(maskIntersect(muzzle, face), ch["white"])
    # eyes
    eyeY = cy + h * 0.02
    ew = w * 0.17
    eh = h * 0.24
    lx = cx + w * (-0.19 + 0.15 * t)
    rx = cx + w * (0.20 + 0.08 * t)
    state = expr.get("eyes", "open")
    lid = expr.get("lid", 0.0)
    eyeCh = {"eyes": ch["eyes"], "fur": fur}
    drawEye(p, eyeCh, lx, eyeY, ew * (1 + 0.05 * t), eh, state, -1, lid, isCat=True)
    drawEye(p, eyeCh, rx, eyeY, ew * (1 - 0.25 * t), eh, state, 1, lid, isCat=True)
    # nose
    ny = cy + h * 0.15
    nose = p.poly([(fx - w * 0.045, ny - h * 0.02), (fx + w * 0.045, ny - h * 0.02), (fx, ny + h * 0.04)],
                  smooth=True)
    p.part(nose, ch["nose"], line=p.line * 0.6, shade=0, light=0)
    # mouth: mischievous grin with one little fang
    my = ny + h * 0.07
    p.stroke([(fx, ny + h * 0.03), (fx, my)], OUTLINE, w * 0.02)
    if expr.get("mouth", "grin") == "grin":
        grin = art_kit.bezier((fx - w * 0.15, my - h * 0.03), (fx + w * 0.02, my + h * 0.22),
                              (fx + w * 0.2, my - h * 0.06), steps=18)
        mouth = p.poly(grin)
        p.fill(maskGrow(mouth, w * 0.012), OUTLINE)
        p.fill(mouth, MOUTH_DARK)
        p.fill(maskIntersect(p.ellipse(fx + w * 0.04, my + h * 0.11, w * 0.07, h * 0.05), mouth), TONGUE)
        fang = p.poly([(fx - w * 0.06, my - h * 0.005), (fx - w * 0.02, my + h * 0.005), (fx - w * 0.045, my + h * 0.06)])
        p.fill(fang, WHITE)
    else:
        for side in (-1, 1):
            curve = art_kit.bezier((fx, my), (fx + side * w * 0.05, my + h * 0.07), (fx + side * w * 0.11, my - h * 0.01),
                                   steps=10)
            p.stroke(curve, OUTLINE, w * 0.02)
    # blush
    p.softEllipse(lx - w * 0.03, eyeY + h * 0.2, w * 0.07, h * 0.035, BLUSH, 110, w * 0.02, clip=face)
    p.softEllipse(rx + w * 0.04, eyeY + h * 0.2, w * 0.06, h * 0.035, BLUSH, 110, w * 0.02, clip=face)
    # whiskers (drawn on the top layer so they get no outline)
    for side in (-1, 1):
        for k in range(3):
            x0 = fx + side * w * 0.2
            y0 = cy + h * (0.2 + k * 0.06)
            x1 = x0 + side * w * 0.34
            y1 = y0 + (k - 1) * h * 0.1
            p.stroke([(x0, y0), (x1, y1)], (255, 250, 240), max(1.0, w * 0.012), 230, target=p.over)


def oyenTailMask(p, control, startWidth, endWidth):
    centre = catmullRom(control, False, 10)
    return p.poly(ribbonPoints(centre, startWidth, endWidth)), centre


def drawOyenTail(p, control, startWidth, endWidth):
    ch = OYEN
    tail, centre = oyenTailMask(p, control, startWidth, endWidth)
    tip = p.circle(centre[-1][0], centre[-1][1], endWidth / 2)
    tail = maskUnion(tail, tip)
    p.part(tail, ch["fur"])
    n = len(centre)
    for k in range(1, 5):
        c = centre[int(n * k / 5)]
        p.fill(maskIntersect(p.circle(c[0], c[1], startWidth * 0.45), maskShrink(tail, 0.6)), ch["stripe"], 230)
    return tail


def drawOyenLeg(p, top, angle, lift, length, far):
    ch = OYEN
    fur = ch["fur"]
    white = ch["white"]
    if far:
        fur = darkOf(fur, 0.2)
        white = darkOf(white, 0.18)
    a = math.radians(angle)
    paw = (top[0] + (length - lift) * math.sin(a), top[1] + (length - lift) * math.cos(a))
    leg = p.limb([top, paw], 10)
    p.part(leg, fur, shade=1.6, light=0)
    foot = p.ellipse(paw[0] + 2, paw[1] + 1, 6.5, 4.2)
    p.part(foot, white, shade=1.2, light=0)


def drawOyenWalk(frame):
    """Sneaky low crouch walk: diagonal leg pairs move together."""
    ch = OYEN
    p = Painter(128, 96, line=2.1, shade=2.4, light=0.9)
    phase = frame / 4.0 * 2 * math.pi
    s = math.sin(phase)
    c = math.cos(phase)
    bob = 1.5 * abs(s)
    base = 90
    p.softEllipse(62, base - 1, 40, 4, (18, 10, 30), 110, 2.0, target=p.under)
    # tail up and curling, swaying with the steps
    drawOyenTail(p, [(30, 60 + bob), (16, 52 + bob), (10, 38 + 2 * s), (16, 24 + 2 * c), (26 + 3 * s, 20)], 10, 7)
    legTopY = 70 + bob
    legs = [((42, legTopY), 20 * s, 3 * max(0, c), True), ((84, legTopY), -20 * s, 3 * max(0, -c), True)]
    for top, angle, lift, far in legs:
        drawOyenLeg(p, top, angle, lift, 17 - bob, far)
    body = maskUnion(p.ellipse(60, 63 + bob, 33, 17), p.ellipse(40, 63 + bob, 17, 16))
    body = maskUnion(body, p.ellipse(84, 63 + bob, 16, 15))
    p.part(body, ch["fur"])
    belly = maskIntersect(p.ellipse(66, 77 + bob, 26, 8), body)
    p.fill(belly, ch["white"])
    for x in (44, 56, 68):
        stripe = catmullRom([(x + 3, 46 + bob), (x, 55 + bob), (x - 3, 64 + bob)], False, 6)
        p.fill(maskIntersect(p.taper(stripe, 6, 2), body), ch["stripe"])
    for top, angle, lift in (((46, legTopY), -20 * s, 3 * max(0, -c)), ((80, legTopY), 20 * s, 3 * max(0, c))):
        drawOyenLeg(p, top, angle, lift, 17 - bob, False)
    head = makeHead(99, 54 + bob * 0.5, 44, 38, 0.5)
    drawOyenHead(p, head, {"eyes": "open", "lid": 0.32, "mouth": "grin"})
    return p.result(0.8)


def drawOyenSit(frame):
    """Sitting, facing right; the tail swishes and frame 3 is a smug blink."""
    ch = OYEN
    p = Painter(128, 96, line=2.1, shade=2.4, light=0.9)
    base = 90
    p.softEllipse(58, base - 1, 34, 4, (18, 10, 30), 110, 2.0, target=p.under)
    tails = [
        [(42, 84), (26, 89), (12, 84), (8, 70), (14, 60)],
        [(42, 84), (26, 89), (10, 80), (6, 64), (12, 52)],
        [(42, 84), (26, 90), (12, 88), (4, 80), (6, 70)],
        [(42, 84), (26, 89), (11, 83), (6, 68), (10, 57)],
    ]
    drawOyenTail(p, tails[frame], 10, 7)
    # far front leg behind the body
    drawOyenLeg(p, (64, 64), 0, 0, 22, True)
    body = maskUnion(p.ellipse(50, 70, 23, 19), p.ellipse(64, 60, 17, 24))
    p.part(body, ch["fur"])
    chest = maskIntersect(p.ellipse(71, 60, 10, 18), body)
    p.fill(chest, ch["white"])
    for k in range(3):
        stripe = catmullRom([(36 + k * 7, 54 + k * 2), (33 + k * 7, 64 + k * 2), (36 + k * 7, 74 + k * 2)], False, 6)
        p.fill(maskIntersect(p.taper(stripe, 5, 2), body), ch["stripe"])
    haunchPaw = p.ellipse(42, 86, 9, 4)
    p.part(haunchPaw, ch["white"], shade=1.0, light=0)
    drawOyenLeg(p, (74, 64), -4, 0, 22, False)
    expr = {"eyes": "open", "lid": 0.25, "mouth": "smile"}
    if frame == 0:
        expr["mouth"] = "grin"
    if frame == 3:
        expr = {"eyes": "smug", "mouth": "grin"}
    head = makeHead(72, 38, 48, 40, 0.35)
    drawOyenHead(p, head, expr)
    return p.result(0.8)


def drawOyenPortrait():
    ch = OYEN
    p = Painter(256, 256, line=3.1, shade=4.0, light=1.6)
    # chubby body and white chest
    body = p.poly([(50, 290), (58, 200), (90, 168), (170, 168), (206, 200), (214, 290)], smooth=True)
    p.part(body, ch["fur"])
    chest = maskIntersect(p.ellipse(136, 230, 46, 60), body)
    p.fill(chest, ch["white"])
    for k in range(3):
        stripe = catmullRom([(64 + k * 6, 196 + k * 16), (74 + k * 6, 206 + k * 16), (70 + k * 6, 218 + k * 16)],
                            False, 6)
        p.fill(maskIntersect(p.taper(stripe, 8, 3), body), ch["stripe"])
    head = makeHead(126, 112, 176, 150, 0.3)
    drawOyenHead(p, head, {"eyes": "open", "lid": 0.3, "mouth": "grin"})
    # a raised paw: "hehe"
    paw = p.ellipse(206, 206, 22, 20)
    p.part(paw, ch["white"])
    for k in range(3):
        x = 196 + k * 10
        p.stroke([(x, 192), (x, 200)], darkOf(ch["white"], 0.25), 2.0)
    p.fill(p.ellipse(208, 214, 8, 6), ch["nose"], 200)
    return p.result(1.0)


# ======================================================================
# CROWD silhouettes (background walkers)
# ======================================================================
CROWD_W = 110
CROWD_H = 190
CROWD_BASE = 184
CROWD_TOP = (70, 76, 118)
CROWD_BOTTOM = (36, 40, 72)
RIM = (255, 172, 96)


def personMask(p, cx, height, bodyW, stride, headScale=1.0, arms=True, legs=True):
    """A simple walking person: head, torso, two legs and two arms."""
    headR = height * 0.105 * headScale
    headY = CROWD_BASE - height + headR
    shoulderY = headY + headR + height * 0.05
    hipY = CROWD_BASE - height * 0.46
    shape = p.circle(cx + 1, headY, headR)
    shape = maskUnion(shape, p.rect(cx - headR * 0.35, headY, cx + headR * 0.35, shoulderY + 4))
    torso = p.poly([(cx - bodyW * 0.42, shoulderY), (cx + bodyW * 0.42, shoulderY), (cx + bodyW * 0.5, shoulderY + 6),
                    (cx + bodyW * 0.44, hipY + 4), (cx - bodyW * 0.44, hipY + 4), (cx - bodyW * 0.5, shoulderY + 6)],
                   smooth=True)
    shape = maskUnion(shape, torso)
    if legs:
        for side in (-1, 1):
            foot = (cx + side * stride, CROWD_BASE - 3)
            leg = p.limb([(cx + side * bodyW * 0.18, hipY), foot], bodyW * 0.3)
            shape = maskUnion(shape, leg)
            shape = maskUnion(shape, p.ellipse(foot[0] + 4, CROWD_BASE - 2.5, 7, 3.5))
    if arms:
        for side in (-1, 1):
            hand = (cx - side * stride * 0.7, hipY + height * 0.02)
            arm = p.limb([(cx + side * bodyW * 0.1, shoulderY + 5), hand], bodyW * 0.22)
            shape = maskUnion(shape, arm)
    return shape, headY, headR, shoulderY, hipY


def drawCrowdPerson(index):
    p = Painter(CROWD_W, CROWD_H, line=0, shade=0, light=0)
    extra = []
    glow = None
    if index == 0:
        # uncle with a plastic bag of food
        shape, hy, hr, sy, hipY = personMask(p, 54, 164, 40, 10)
        shape = maskUnion(shape, p.ellipse(58, hipY - 18, 22, 26))  # round belly
        shape = maskUnion(shape, p.limb([(64, sy + 6), (76, hipY + 6)], 8))
        bag = p.poly([(70, hipY + 4), (86, hipY + 4), (90, hipY + 30), (68, hipY + 30)], smooth=True)
        shape = maskUnion(shape, bag)
        shape = maskUnion(shape, p.ellipse(54, hy - hr * 0.6, hr * 1.1, hr * 0.45))  # cap
    elif index == 1:
        # girl with a ponytail and a skirt
        shape, hy, hr, sy, hipY = personMask(p, 56, 150, 30, 12)
        pony = p.poly(ribbonPoints(catmullRom([(46, hy - 4), (36, hy + 4), (32, hy + 22)], False, 6), 11, 4))
        shape = maskUnion(shape, pony)
        skirt = p.poly([(42, hipY - 10), (70, hipY - 10), (78, hipY + 26), (34, hipY + 26)], smooth=True)
        shape = maskUnion(shape, skirt)
    elif index == 2:
        # couple holding hands
        a, hy, hr, sy, hipY = personMask(p, 36, 166, 32, 9)
        b, hy2, hr2, sy2, hipY2 = personMask(p, 74, 150, 28, 9)
        shape = maskUnion(a, b)
        shape = maskUnion(shape, p.limb([(46, sy + 8), (56, hipY - 4), (66, sy2 + 10)], 6))
        shape = maskUnion(shape, p.poly(ribbonPoints([(70, hy2 - 8), (64, hy2 + 10), (66, hy2 + 26)], 16, 8)))
    elif index == 3:
        # makcik in a tudung and baju kurung carrying a tiffin carrier
        shape, hy, hr, sy, hipY = personMask(p, 54, 156, 36, 6)
        tudung = p.poly([(54 - hr * 1.2, hy - 2), (54, hy - hr * 1.25), (54 + hr * 1.25, hy - 2), (54 + 20, sy + 26),
                         (54, sy + 34), (54 - 20, sy + 26)], smooth=True)
        shape = maskUnion(shape, tudung)
        kurung = p.poly([(34, sy + 10), (74, sy + 10), (80, CROWD_BASE - 16), (28, CROWD_BASE - 16)], smooth=True)
        shape = maskUnion(shape, kurung)
        shape = maskUnion(shape, p.rect(74, hipY - 4, 90, hipY + 26, radius=3))
        shape = maskUnion(shape, p.limb([(76, hipY - 4), (82, hipY - 14), (88, hipY - 4)], 2.5))
    elif index == 4:
        # child holding a paper lantern on a stick (the lantern glows)
        shape, hy, hr, sy, hipY = personMask(p, 46, 104, 28, 7, headScale=1.35)
        shape = maskUnion(shape, p.limb([(52, sy + 8), (70, sy + 2)], 6))
        shape = maskUnion(shape, p.limb([(70, sy + 2), (84, sy - 30)], 2.5))
        extra.append((84, sy - 16))
    else:
        # student with a cap and backpack, reading his phone
        shape, hy, hr, sy, hipY = personMask(p, 52, 168, 34, 11, arms=False)
        shape = maskUnion(shape, p.poly([(52 - hr, hy - 3), (52 + hr * 1.9, hy - 3), (52 + hr, hy - hr * 1.1),
                                         (52 - hr * 0.8, hy - hr * 0.9)], smooth=True))
        shape = maskUnion(shape, p.rect(26, sy + 2, 44, hipY - 4, radius=6))
        shape = maskUnion(shape, p.limb([(56, sy + 6), (60, sy + 36), (70, sy + 24)], 8))
        shape = maskUnion(shape, p.limb([(48, sy + 6), (46, hipY + 4)], 8))
        glow = (72, sy + 20)
    p.gradient(shape, CROWD_TOP, CROWD_BOTTOM)
    # warm rim light on the right edge (the lanterns are ahead of them)
    rim = maskSubtract(shape, maskShift(shape, -2.2, 0.6))
    p.fill(rim, RIM, 210)
    top = maskSubtract(shape, maskShift(shape, -1.0, 1.6))
    p.fill(top, RIM, 90)
    cool = maskSubtract(shape, maskShift(shape, 1.6, 0))
    p.fill(cool, (120, 140, 220), 90)
    for x, y in extra:
        p.softEllipse(x, y, 16, 16, (255, 160, 70), 170, 5, target=p.under)
        lantern = p.ellipse(x, y, 8, 10)
        p.fill(maskGrow(lantern, 1.2), (60, 30, 40))
        p.gradient(lantern, (255, 210, 110), (230, 70, 50))
        p.fill(p.rect(x - 4, y - 12, x + 4, y - 9), GOLD)
        p.fill(p.rect(x - 4, y + 9, x + 4, y + 12), GOLD)
    if glow is not None:
        p.softEllipse(glow[0] + 2, glow[1] - 6, 10, 10, (150, 220, 255), 120, 4, target=p.over)
        p.fill(p.rect(glow[0] - 2, glow[1] - 5, glow[0] + 4, glow[1] + 4, radius=1), (170, 230, 255))
    return p.result(0)


# ======================================================================
# MAIN
# ======================================================================
def makeAll():
    # Aiman: 8-frame walk (4 columns x 2 rows) and 4-frame idle
    walkFrames = []
    for i in range(8):
        walkFrames.append(drawAimanSide(AIMAN, aimanWalkPose(i)))
    art_kit.save(art_kit.spriteSheet(walkFrames, 4), "characters/aiman_walk.png")
    idleFrames = []
    for i in range(4):
        idleFrames.append(drawAimanSide(AIMAN, aimanIdlePose(i)))
    art_kit.save(art_kit.spriteSheet(idleFrames, 4), "characters/aiman_idle.png")

    # dialogue portraits
    for ch in (AIMAN, NENEK, LEON, LIM, KAVITHA, FARAH):
        art_kit.save(drawPortrait(ch), "characters/portrait_" + ch["id"] + ".png")
    art_kit.save(drawOyenPortrait(), "characters/portrait_oyen.png")

    # seated banquet guests
    for ch in (LEON, LIM, KAVITHA, FARAH, NENEK):
        frames = []
        for f in range(4):
            frames.append(drawGuestFrame(ch, f))
        art_kit.save(art_kit.spriteSheet(frames, 4), "characters/guest_" + ch["id"] + ".png")

    # Oyen the cat
    frames = []
    for f in range(4):
        frames.append(drawOyenWalk(f))
    art_kit.save(art_kit.spriteSheet(frames, 4), "characters/oyen_walk.png")
    frames = []
    for f in range(4):
        frames.append(drawOyenSit(f))
    art_kit.save(art_kit.spriteSheet(frames, 4), "characters/oyen_sit.png")

    # background crowd
    frames = []
    for i in range(6):
        frames.append(drawCrowdPerson(i))
    art_kit.save(art_kit.spriteSheet(frames, 6), "characters/crowd_people.png")


if __name__ == "__main__":
    random.seed(88446)
    makeAll()
