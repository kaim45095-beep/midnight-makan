"""
assets.py
---------
Every image and font in the game is loaded through this file, so:
  * paths are always built from settings.IMAGE_DIR (no hard-coded paths),
  * each file is loaded only ONCE (a dictionary cache),
  * a missing file never crashes the game: we print one warning and use a
    placeholder picture instead.

It also does RUNTIME IMAGE PROCESSING with Pillow (PIL), which is one of
the module topics:
  * cropSlices()    - image.crop() into strips (Week 4 tutorial exercise)
  * darkerTexture() - ImageEnhance brightness + colour (burnt food)
  * glowTexture()   - GaussianBlur on the alpha channel (glow / aura halo)
  * tintedTexture() - colour shifting by blending the RGB channels

Coordinates: PIL images start at the TOP-LEFT, Arcade draws from the
BOTTOM-LEFT. That only matters when cropping (we crop in PIL space).
"""
import math
import os

import arcade
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

import settings

# Caches (dictionaries) so each file is only processed once
textureCache = {}
imageCache = {}
framesCache = {}
processedCache = {}
warnedPaths = set()
usedTextureNames = {}
fontsLoaded = False
placeholderTexture = None


# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
def imagePath(relativePath):
    """Full path of a file inside assets/images, e.g. imagePath("ui/logo.png")."""
    return os.path.join(settings.IMAGE_DIR, relativePath)


def imageExists(relativePath):
    return os.path.isfile(imagePath(relativePath))


def warnMissing(relativePath):
    """Print a warning only the first time a file is missing."""
    if relativePath in warnedPaths:
        return
    warnedPaths.add(relativePath)
    print(f"[assets] WARNING: missing image '{relativePath}' - using a placeholder")


# ----------------------------------------------------------------------
# Textures
# ----------------------------------------------------------------------
def textureFromImage(pilImage, name):
    """
    Turn a PIL image into an Arcade texture.
    The name is the texture's hash (its ID in the GPU texture atlas). Two
    different pictures must never share a name, otherwise the atlas would
    show the old one - so if a name was already used we add a number.
    """
    if name in usedTextureNames:
        usedTextureNames[name] += 1
        name = f"{name}#{usedTextureNames[name]}"
    else:
        usedTextureNames[name] = 0
    if pilImage.mode != "RGBA":
        pilImage = pilImage.convert("RGBA")
    return arcade.Texture(pilImage, hash=name)


def getPlaceholderTexture():
    """Magenta checkerboard: easy to spot on screen when an image is missing."""
    global placeholderTexture
    if placeholderTexture is None:
        placeholderTexture = textureFromImage(makePlaceholderImage(64, 64), "placeholder-magenta")
    return placeholderTexture


def loadTexture(relativePath):
    """Load assets/images/<relativePath> once and cache it (missing -> placeholder)."""
    if relativePath in textureCache:
        return textureCache[relativePath]

    texture = None
    path = imagePath(relativePath)
    if os.path.isfile(path):
        try:
            texture = arcade.load_texture(path)
        except Exception as error:
            print(f"[assets] WARNING: could not load '{relativePath}': {error}")
    if texture is None:
        warnMissing(relativePath)
        fallback = makeFallbackImage(relativePath)
        if fallback is not None:
            texture = textureFromImage(fallback, "fallback:" + relativePath)
        else:
            texture = getPlaceholderTexture()

    textureCache[relativePath] = texture
    return texture


def loadImage(relativePath):
    """A PIL RGBA copy of an image for runtime processing (the original stays cached)."""
    if relativePath not in imageCache:
        path = imagePath(relativePath)
        image = None
        if os.path.isfile(path):
            try:
                image = Image.open(path).convert("RGBA")
            except Exception as error:
                print(f"[assets] WARNING: could not open '{relativePath}': {error}")
        if image is None:
            warnMissing(relativePath)
            image = makeFallbackImage(relativePath)
            if image is None:
                image = makePlaceholderImage(64, 64)
        imageCache[relativePath] = image
    return imageCache[relativePath].copy()


def loadFrames(relativePath, columns, rows):
    """
    Cut a sprite sheet into animation frames (Week 4 method):
        sheet = arcade.load_spritesheet(path)
        frames = sheet.get_texture_grid((frameWidth, frameHeight), columns, count)
    Frames are ordered left-to-right, top-to-bottom.
    """
    key = (relativePath, columns, rows)
    if key in framesCache:
        return framesCache[key]

    count = columns * rows
    frames = []
    path = imagePath(relativePath)
    if os.path.isfile(path):
        try:
            sheet = arcade.load_spritesheet(path)
            frameWidth = sheet.image.width // columns
            frameHeight = sheet.image.height // rows
            frames = sheet.get_texture_grid((frameWidth, frameHeight), columns, count)
        except Exception as error:
            print(f"[assets] WARNING: could not cut sprite sheet '{relativePath}': {error}")
            frames = []
    if len(frames) == 0:
        warnMissing(relativePath)
        for i in range(count):
            frames.append(getPlaceholderTexture())

    framesCache[key] = frames
    return frames


def loadFramesMirrored(relativePath, columns, rows):
    """Same frames flipped left-right (a mirroring / reflection transformation)."""
    key = ("mirrored", relativePath, columns, rows)
    if key in framesCache:
        return framesCache[key]
    mirrored = []
    for frame in loadFrames(relativePath, columns, rows):
        mirrored.append(frame.flip_left_right())
    framesCache[key] = mirrored
    return mirrored


def loadFonts():
    """Register every .ttf in assets/fonts with Arcade (so font names like 'Kenney Future' work)."""
    global fontsLoaded
    if fontsLoaded:
        return
    fontsLoaded = True
    if not os.path.isdir(settings.FONT_DIR):
        print("[assets] WARNING: font folder not found - using system fonts")
        return
    for fileName in sorted(os.listdir(settings.FONT_DIR)):
        if fileName.lower().endswith(".ttf"):
            try:
                arcade.load_font(os.path.join(settings.FONT_DIR, fileName))
            except Exception as error:
                print(f"[assets] WARNING: could not load font {fileName}: {error}")


# ----------------------------------------------------------------------
# Runtime image processing (Pillow)
# ----------------------------------------------------------------------
def cropSlices(relativePath, cutFractions):
    """
    Week 4 exercise: cut an image into vertical strips with image.crop().
    cutFractions are x positions from 0 (left edge) to 1 (right edge),
    e.g. [0.25, 0.5, 0.75] gives 4 strips.
    Returns a list of (texture, leftFraction, rightFraction).
    """
    fractions = []
    for value in sorted(cutFractions):
        value = max(0.0, min(1.0, float(value)))
        if 0.0 < value < 1.0 and value not in fractions:
            fractions.append(value)
    key = ("slices", relativePath, tuple(fractions))
    if key in processedCache:
        return processedCache[key]

    image = loadImage(relativePath)
    width, height = image.size
    edges = [0.0] + fractions + [1.0]
    slices = []
    for i in range(len(edges) - 1):
        leftFraction = edges[i]
        rightFraction = edges[i + 1]
        left = int(round(leftFraction * width))
        right = int(round(rightFraction * width))
        if right <= left:
            right = min(width, left + 1)
        # crop box is (left, top, right, bottom) in PIL coordinates
        piece = image.crop((left, 0, right, height))
        texture = textureFromImage(piece, f"slice:{relativePath}:{i}:{fractions}")
        slices.append((texture, leftFraction, rightFraction))

    processedCache[key] = slices
    return slices


def splitAlpha(image):
    """Return (rgbImage, alphaChannel) so filters do not change the transparency."""
    rgba = image.convert("RGBA")
    alpha = rgba.getchannel("A")
    return rgba.convert("RGB"), alpha


def darkerTexture(relativePath, factor=0.45, saturation=0.6):
    """
    A 'burnt' version of an image: lower brightness and colour saturation
    with ImageEnhance, then a small brown tint. The alpha channel is kept.
    """
    key = ("darker", relativePath, factor, saturation)
    if key in processedCache:
        return processedCache[key]
    rgb, alpha = splitAlpha(loadImage(relativePath))
    rgb = ImageEnhance.Color(rgb).enhance(saturation)
    rgb = ImageEnhance.Brightness(rgb).enhance(factor)
    brown = Image.new("RGB", rgb.size, (60, 34, 18))
    rgb = Image.blend(rgb, brown, 0.18)
    result = rgb.convert("RGBA")
    result.putalpha(alpha)
    texture = textureFromImage(result, f"darker:{relativePath}:{factor}:{saturation}")
    processedCache[key] = texture
    return texture


def glowTexture(relativePath, blurRadius=12, rgb=(255, 210, 120)):
    """
    A soft coloured halo with the same shape as the image:
    take the alpha channel (the silhouette), blur it with GaussianBlur,
    and fill it with one colour.
    NOTE: the result is BIGGER than the source image by 'pad' pixels on
    every side (pad = blurRadius * 2) so the blur is not cut off.
    Draw it at the same centre as the sprite, at (texture.width * scale).
    """
    key = ("glow", relativePath, blurRadius, tuple(rgb))
    if key in processedCache:
        return processedCache[key]
    source = loadImage(relativePath)
    pad = int(blurRadius * 2)
    width = source.width + pad * 2
    height = source.height + pad * 2
    alpha = Image.new("L", (width, height), 0)
    alpha.paste(source.getchannel("A"), (pad, pad))
    alpha = alpha.filter(ImageFilter.GaussianBlur(blurRadius))
    # Blurring makes the silhouette fainter, so boost it with a lookup table
    boostTable = []
    for i in range(256):
        boostTable.append(min(255, int(i * 1.8)))
    alpha = alpha.point(boostTable)
    glow = Image.new("RGBA", (width, height), (rgb[0], rgb[1], rgb[2], 0))
    glow.putalpha(alpha)
    texture = textureFromImage(glow, f"glow:{relativePath}:{blurRadius}:{rgb}")
    processedCache[key] = texture
    return texture


def tintedTexture(relativePath, rgb, amount):
    """Colour-shifted copy: blend the RGB channels towards 'rgb' by 'amount' (0..1)."""
    key = ("tint", relativePath, tuple(rgb), amount)
    if key in processedCache:
        return processedCache[key]
    base, alpha = splitAlpha(loadImage(relativePath))
    colour = Image.new("RGB", base.size, (rgb[0], rgb[1], rgb[2]))
    mixed = Image.blend(base, colour, max(0.0, min(1.0, amount)))
    result = mixed.convert("RGBA")
    result.putalpha(alpha)
    texture = textureFromImage(result, f"tint:{relativePath}:{rgb}:{amount}")
    processedCache[key] = texture
    return texture


# ----------------------------------------------------------------------
# Drawing helper
# ----------------------------------------------------------------------
def additiveBlend():
    """
    Additive blending that still respects each pixel's alpha:
        result = source * sourceAlpha + destination
    (Arcade's BLEND_ADDITIVE is ONE, ONE and ignores alpha, which turns
    soft white particle textures into hard white squares.)
    """
    ctx = arcade.get_window().ctx
    return (ctx.SRC_ALPHA, ctx.ONE)


def drawTexture(texture, cx, cy, width=None, height=None, angle=0.0, alpha=255, rgb=None, additive=False):
    """
    Draw a texture centred on (cx, cy).
    width/height: size in pixels (give one of them to keep the aspect ratio,
    none to use the image size). angle: degrees clockwise.
    rgb: optional colour tint (multiplies the image). additive: glow blending.
    """
    if width is None and height is None:
        width = texture.width
        height = texture.height
    elif width is None:
        width = texture.width * (height / texture.height)
    elif height is None:
        height = texture.height * (width / texture.width)
    if width <= 0 or height <= 0 or alpha <= 0:
        return
    if rgb is None:
        tint = arcade.color.WHITE
    else:
        tint = arcade.types.Color(int(rgb[0]), int(rgb[1]), int(rgb[2]), 255)
    rect = arcade.XYWH(cx, cy, width, height)
    if additive:
        ctx = arcade.get_window().ctx
        ctx.blend_func = additiveBlend()
        arcade.draw_texture_rect(texture, rect, color=tint, angle=angle, alpha=int(alpha))
        ctx.blend_func = ctx.BLEND_DEFAULT
    else:
        arcade.draw_texture_rect(texture, rect, color=tint, angle=angle, alpha=int(alpha))


# ----------------------------------------------------------------------
# Fallback pictures (drawn with PIL) for files that do not exist yet.
# Particles and overlays get a proper simple shape so effects still work;
# everything else gets the magenta checkerboard.
# ----------------------------------------------------------------------
def makePlaceholderImage(width, height):
    image = Image.new("RGBA", (width, height), (255, 0, 255, 200))
    draw = ImageDraw.Draw(image)
    cell = max(4, width // 4)
    for row in range(0, height, cell):
        for col in range(0, width, cell):
            if ((row // cell) + (col // cell)) % 2 == 0:
                draw.rectangle((col, row, col + cell - 1, row + cell - 1), fill=(40, 0, 40, 200))
    return image


def radialAlpha(size, power=2.0, inner=0.0):
    """Square L image: 255 at the centre fading to 0 at the edge (lookup table on a radial gradient)."""
    gradient = Image.radial_gradient("L")   # 256x256, 0 in the centre, 255 at radius 128
    table = []
    for i in range(256):
        d = i / 255.0
        if d <= inner:
            value = 1.0
        else:
            value = max(0.0, 1.0 - (d - inner) / (1.0 - inner))
        table.append(int(255 * value ** power))
    return gradient.point(table).resize((size, size), Image.BICUBIC)


def whiteWithAlpha(alpha):
    image = Image.new("RGBA", alpha.size, (255, 255, 255, 0))
    image.putalpha(alpha)
    return image


def starPoints(cx, cy, outer, inner, points=5, rotation=-90.0):
    result = []
    for i in range(points * 2):
        radius = outer if i % 2 == 0 else inner
        angle = math.radians(rotation + i * 180.0 / points)
        result.append((cx + radius * math.cos(angle), cy + radius * math.sin(angle)))
    return result


def heartPoints(cx, cy, scale):
    """Parametric heart curve: x = 16 sin^3 t, y = 13 cos t - 5 cos 2t - 2 cos 3t - cos 4t."""
    result = []
    for i in range(60):
        t = i / 60 * 2 * math.pi
        x = 16 * math.sin(t) ** 3
        y = 13 * math.cos(t) - 5 * math.cos(2 * t) - 2 * math.cos(3 * t) - math.cos(4 * t)
        result.append((cx + x * scale, cy - y * scale))   # PIL y goes down
    return result


def drawShape(width, height, drawFunction, blur=0):
    """Draw 4x bigger then shrink (supersampling = anti-aliased edges)."""
    big = Image.new("L", (width * 4, height * 4), 0)
    drawFunction(ImageDraw.Draw(big), 4)
    small = big.resize((width, height), Image.LANCZOS)
    if blur > 0:
        small = small.filter(ImageFilter.GaussianBlur(blur))
    return small


def drawDroplet(draw, k):
    draw.ellipse((2 * k, 16 * k, 22 * k, 38 * k), fill=255)
    draw.polygon([(12 * k, 1 * k), (3 * k, 24 * k), (21 * k, 24 * k)], fill=255)


def drawSpark(draw, k):
    draw.polygon(starPoints(24 * k, 24 * k, 23 * k, 4 * k, 4, -90), fill=255)


def drawStar5(draw, k):
    draw.polygon(starPoints(24 * k, 25 * k, 22 * k, 10 * k, 5, -90), fill=255)


def drawHeart(draw, k):
    draw.polygon(heartPoints(24 * k, 22 * k, 1.35 * k), fill=255)


def drawHardCircle(draw, k):
    draw.ellipse((1 * k, 1 * k, 31 * k, 31 * k), fill=255)


def drawConfetti(draw, k):
    draw.rectangle((0, 0, 20 * k, 12 * k), fill=255)


def drawRing(draw, k):
    draw.ellipse((10 * k, 10 * k, 118 * k, 118 * k), outline=255, width=6 * k)


def drawBubble(draw, k):
    draw.ellipse((2 * k, 2 * k, 30 * k, 30 * k), outline=255, width=3 * k)
    draw.ellipse((9 * k, 8 * k, 15 * k, 14 * k), fill=255)


def makeEdgeGlow(width, height, falloff, maxAlpha):
    """Alpha that is strong at the screen edges and 0 in the middle (small then resized = smooth)."""
    smallW = 160
    smallH = 90
    alpha = Image.new("L", (smallW, smallH), 0)
    pixels = alpha.load()
    for y in range(smallH):
        for x in range(smallW):
            edgeDistance = min(x, y, smallW - 1 - x, smallH - 1 - y)
            pixels[x, y] = int(maxAlpha * math.exp(-edgeDistance / falloff))
    return alpha.resize((width, height), Image.BICUBIC)


def makeFallbackImage(relativePath):
    """Simple procedural stand-ins for important images (None = use the magenta placeholder)."""
    name = os.path.splitext(os.path.basename(relativePath))[0]
    if name == "soft_circle":
        return whiteWithAlpha(radialAlpha(64, 1.8))
    if name == "light_glow":
        return whiteWithAlpha(radialAlpha(256, 2.4))
    if name == "smoke_puff":
        return whiteWithAlpha(radialAlpha(128, 1.3, 0.1))
    if name == "flame":
        alpha = radialAlpha(64, 1.5).resize((64, 96), Image.BICUBIC)
        return whiteWithAlpha(alpha)
    if name == "hard_circle":
        return whiteWithAlpha(drawShape(32, 32, drawHardCircle))
    if name == "droplet":
        return whiteWithAlpha(drawShape(24, 40, drawDroplet))
    if name == "spark":
        spark = drawShape(48, 48, drawSpark)
        glow = radialAlpha(48, 3.0)
        return whiteWithAlpha(Image.composite(spark, glow, spark))
    if name == "star5":
        return whiteWithAlpha(drawShape(48, 48, drawStar5))
    if name == "heart":
        return whiteWithAlpha(drawShape(48, 48, drawHeart))
    if name == "confetti":
        return whiteWithAlpha(drawShape(20, 12, drawConfetti))
    if name == "ring":
        return whiteWithAlpha(drawShape(128, 128, drawRing, blur=2))
    if name == "bubble":
        return whiteWithAlpha(drawShape(32, 32, drawBubble))
    if name == "vignette":
        gradient = Image.radial_gradient("L").resize((settings.SCREEN_WIDTH, settings.SCREEN_HEIGHT), Image.BICUBIC)
        table = []
        for i in range(256):
            t = max(0.0, min(1.0, (i - 120) / 150.0))
            table.append(int(255 * t * t * (3 - 2 * t)))
        image = Image.new("RGBA", gradient.size, (6, 6, 18, 0))
        image.putalpha(gradient.point(table))
        return image
    if name == "aura_frame":
        image = Image.new("RGBA", (settings.SCREEN_WIDTH, settings.SCREEN_HEIGHT), (255, 196, 90, 0))
        image.putalpha(makeEdgeGlow(settings.SCREEN_WIDTH, settings.SCREEN_HEIGHT, 7.0, 235))
        return image
    if name == "star_full" or name == "star_empty":
        return makeStarIcon(name == "star_full")
    if name == "cursor_hand" or name == "cursor_grab":
        return makeCursorIcon(name == "cursor_grab")
    if name == "icon_clock":
        return makeClockIcon()
    if name == "icon_ladle":
        return makeLadleIcon()
    return None


def makeStarIcon(full):
    big = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
    draw = ImageDraw.Draw(big)
    points = starPoints(128, 136, 118, 54, 5, -90)
    if full:
        draw.polygon(points, fill=(255, 200, 70, 255), outline=(90, 50, 20, 255), width=14)
        draw.polygon(starPoints(118, 118, 50, 22, 5, -90), fill=(255, 238, 170, 255))
    else:
        draw.polygon(points, fill=(40, 42, 70, 230), outline=(110, 104, 140, 255), width=12)
    return big.resize((64, 64), Image.LANCZOS)


def makeCursorIcon(grab):
    big = Image.new("RGBA", (192, 192), (0, 0, 0, 0))
    draw = ImageDraw.Draw(big)
    if grab:
        draw.rounded_rectangle((40, 50, 150, 150), radius=40, fill=(255, 244, 224, 255),
                               outline=(44, 30, 38, 255), width=12)
    else:
        arrow = [(20, 10), (20, 150), (58, 116), (86, 176), (114, 164), (86, 106), (136, 106)]
        draw.polygon(arrow, fill=(255, 244, 224, 255), outline=(44, 30, 38, 255), width=12)
    return big.resize((48, 48), Image.LANCZOS)


def makeClockIcon():
    big = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
    draw = ImageDraw.Draw(big)
    draw.ellipse((16, 16, 240, 240), fill=(255, 244, 224, 255), outline=(44, 30, 38, 255), width=18)
    draw.line((128, 128, 128, 60), fill=(44, 30, 38, 255), width=18)
    draw.line((128, 128, 180, 150), fill=(214, 48, 49, 255), width=16)
    return big.resize((64, 64), Image.LANCZOS)


def makeLadleIcon():
    big = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
    draw = ImageDraw.Draw(big)
    draw.line((70, 190, 200, 40), fill=(200, 150, 60, 255), width=26)
    draw.ellipse((20, 140, 130, 240), fill=(255, 200, 87, 255), outline=(90, 56, 20, 255), width=14)
    return big.resize((64, 64), Image.LANCZOS)
