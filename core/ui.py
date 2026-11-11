"""
ui.py
-----
User-interface pieces shared by all scenes:

  drawRoundedRect / drawRoundedRectOutline / drawPanel  - rounded panels
  drawProgressBar, drawKeyHint, drawStars, drawText      - small widgets
  Button       - menu button with a hover animation
  DialogueBox  - typewriter dialogue with portraits and voice blips
  Hud          - top bar: scene name, timer, score, combo and Aura meter
  ResultCard   - end-of-course card: dish, counting score, stamped stars

Rounded rectangles are drawn once with Pillow as white pictures, cached,
and then drawn as ONE tinted texture per shape (see getShapeTexture) -
far fewer GPU draw calls than building them from arcs every frame.

Small text (key caps, name tags, HUD labels) uses the body font: the
Kenney Future title font draws K and X almost like H at small sizes.

Text rule: arcade.Text objects are created ONCE and only moved/recoloured
each frame (creating text every frame is slow). drawText() keeps a small
cache so the same string reuses the same Text object.
"""
import math
import os
import random

import arcade
from PIL import Image, ImageDraw, ImageFont

from core import assets
from core.assets import drawTexture
from core.helpers import (approach, clamp, easeOutBack, easeOutQuad, lerp, lerpColor)
from settings import (STAR_THRESHOLDS, AURA_MODE_TIME, BAD_RED, COMBO_FOR_AURA, CREAM, FONT_BODY, FONT_DIR,
                      FONT_NARROW, FONT_TITLE, GOLD, LAVENDER, SCREEN_HEIGHT,
                      SCREEN_WIDTH, SPEAKERS, UI_PANEL, UI_PANEL_LIGHT, WARM_LIGHT, WHITE)

DARK_TEXT = (34, 22, 32)
SHADOW_COLOR = (8, 6, 18)


def withAlpha(color, alpha=255):
    """(r, g, b) or (r, g, b, a) -> (r, g, b, alpha)."""
    return (int(color[0]), int(color[1]), int(color[2]), int(clamp(alpha, 0, 255)))


def getAlpha(color):
    if len(color) > 3:
        return color[3]
    return 255


def setTextColor(textObject, rgba):
    """
    Set an arcade.Text colour only when it really changes. (Every colour
    change makes pyglet re-style the whole text, which is slow if it is
    done for dozens of texts every frame.)
    """
    if len(rgba) > 3:
        alpha = int(rgba[3])
    else:
        alpha = 255
    colour = (int(rgba[0]), int(rgba[1]), int(rgba[2]), alpha)
    if getattr(textObject, "lastColor", None) != colour:
        textObject.lastColor = colour
        textObject.color = colour


# ----------------------------------------------------------------------
# Cached text drawing
# ----------------------------------------------------------------------
textCache = {}


def getText(text, size, fontName=FONT_BODY, bold=False, anchorX="left", anchorY="baseline",
            width=None, align="left", multiline=False, italic=False, role="main"):
    """
    Return a cached arcade.Text for these settings (made only the first time).
    role="shadow" gives a second copy, so the shadow and the main text never
    have to swap colours. (Avoid multiline=True: pyglet can drop spaces.)
    """
    key = (text, size, fontName, bold, anchorX, anchorY, width, align, multiline, italic, role)
    textObject = textCache.get(key)
    if textObject is None:
        if len(textCache) > 600:
            textCache.clear()
        if multiline and width is None:
            width = 1000
        textObject = arcade.Text(text, 0, 0, WHITE, size, width=width, align=align, font_name=fontName,
                                 bold=bold, italic=italic, anchor_x=anchorX, anchor_y=anchorY,
                                 multiline=multiline)
        textCache[key] = textObject
    return textObject


def drawText(text, x, y, rgb=CREAM, size=18, fontName=FONT_BODY, anchorX="left", anchorY="baseline",
             bold=False, alpha=255, shadow=True, width=None, align="left", multiline=False, italic=False):
    """
    Draw text with an optional soft drop shadow (easier to read on busy art).
    Returns the arcade.Text so the caller can read .content_width.
    """
    textObject = getText(str(text), size, fontName, bold, anchorX, anchorY, width, align, multiline, italic)
    if alpha <= 0:
        return textObject
    if shadow:
        shadowObject = getText(str(text), size, fontName, bold, anchorX, anchorY, width, align, multiline,
                               italic, "shadow")
        shadowObject.x = x + 2
        shadowObject.y = y - 2
        setTextColor(shadowObject, withAlpha(SHADOW_COLOR, alpha * 0.75))
        shadowObject.draw()
    textObject.x = x
    textObject.y = y
    setTextColor(textObject, withAlpha(rgb, alpha))
    textObject.draw()
    return textObject


def getPixelRatio():
    """Real pixels per screen point (1.25 with 125% Windows display scaling)."""
    window = arcade.get_window()
    if window is None:
        return 1.0
    return max(1.0, window.get_pixel_ratio())


# ----------------------------------------------------------------------
# Text rendered into a texture (for big text that zooms)
# ----------------------------------------------------------------------
textTextureCache = {}
FONT_FILES = {                      # font family name -> file in assets/fonts
    "Kenney Future": "Kenney_Future.ttf",
    "Kenney Future Narrow": "Kenney_Future_Narrow.ttf",
    "Kenney Pixel": "Kenney_Pixel.ttf",
    "Liberation Sans": "Liberation_Sans_Bold.ttf",
}


def makeTextTexture(text, size, fontFile="Kenney_Future.ttf", rgb=WHITE, outlineWidth=0,
                    outlineRgb=(20, 10, 24)):
    """
    Render text ONCE with Pillow (ImageFont + a stroke outline) into a cached
    texture. A texture can be scaled smoothly every frame with drawTexture,
    while arcade.Text would have to build a new font size for every frame
    of a zoom animation (slow). size is in screen points. fontFile: a file in
    assets/fonts or a family name such as FONT_TITLE ("Liberation Sans" gives
    the bold file).
    """
    key = (text, size, fontFile, tuple(rgb), outlineWidth, tuple(outlineRgb))
    if key in textTextureCache:
        return textTextureCache[key]
    if len(textTextureCache) > 300:
        textTextureCache.clear()     # e.g. many different "+120" score popups
    if fontFile in FONT_FILES:
        fontFile = FONT_FILES[fontFile]
    ratio = getPixelRatio()
    try:
        font = ImageFont.truetype(os.path.join(FONT_DIR, fontFile), max(4, int(round(size * ratio))))
    except OSError:
        font = ImageFont.load_default()
    stroke = int(round(outlineWidth * ratio))
    left, top, right, bottom = font.getbbox(text, stroke_width=stroke)
    pad = 4
    image = Image.new("RGBA", (right - left + pad * 2, bottom - top + pad * 2), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.text((pad - left, pad - top), text, font=font, fill=(rgb[0], rgb[1], rgb[2], 255),
              stroke_width=stroke, stroke_fill=(outlineRgb[0], outlineRgb[1], outlineRgb[2], 255))
    texture = assets.textureFromImage(image, f"text:{text}:{size}:{fontFile}:{rgb}:{outlineWidth}")
    textTextureCache[key] = texture
    return texture


def drawTextTexture(texture, cx, cy, scale=1.0, alpha=255, rgb=None, angle=0.0):
    """Draw a makeTextTexture() result centred on (cx, cy) at 'scale' times its normal size."""
    ratio = getPixelRatio()
    drawTexture(texture, cx, cy, texture.width / ratio * scale, texture.height / ratio * scale,
                angle=angle, alpha=alpha, rgb=rgb)


# ----------------------------------------------------------------------
# Rounded rectangles and panels
# ----------------------------------------------------------------------
#
# Why textures? Drawing a rounded rectangle from 3 rectangles + 4 arcs costs
# 7 separate GPU draw calls, and a panel with a border needs about 30. With
# dozens of panels per frame the game dropped below 30 FPS. So each shape
# is made ONCE with Pillow as a WHITE picture (runtime image processing:
# the corners are supersampled 4x and shrunk with a LANCZOS filter for
# smooth anti-aliased curves), cached, and drawn as one tinted texture
# (colour multiply).
shapeCache = {}
cornerCache = {}
SHAPE_SUPERSAMPLE = 4


def snapSize(value):
    """Sizes are rounded to 2 px (4 px for big shapes) so animated sizes reuse a few textures."""
    step = 2
    if value > 300:
        step = 4
    return max(step, int(round(value / step)) * step)


def getCornerImage(radius, border):
    """
    Anti-aliased top-left corner (a quarter circle, or a quarter ring when
    border > 0) of size radius x radius pixels, as a greyscale mask.
    Drawn 4x bigger and shrunk (supersampling) so the curve is smooth.
    """
    key = (radius, border)
    if key in cornerCache:
        return cornerCache[key]
    big = radius * SHAPE_SUPERSAMPLE
    image = Image.new("L", (big, big), 0)
    draw = ImageDraw.Draw(image)
    # a circle of radius 'big' centred on the corner square's bottom-right point
    draw.ellipse((0, 0, big * 2 - 1, big * 2 - 1), fill=255)
    if border > 0:
        inner = border * SHAPE_SUPERSAMPLE
        if inner < big:
            draw.ellipse((inner, inner, big * 2 - 1 - inner, big * 2 - 1 - inner), fill=0)
    corner = image.resize((radius, radius), Image.LANCZOS)
    cornerCache[key] = corner
    return corner


def getShapeTexture(w, h, radius, borderWidth):
    """
    White rounded rectangle texture of w x h (screen pixels).
    borderWidth 0 = filled; otherwise only the outline, centred on the edge,
    so the picture is (w + borderWidth) x (h + borderWidth).
    It is built from a plain mask plus 4 cached anti-aliased corners, so even
    a big panel is made in about a millisecond (no big image filtering).
    """
    key = (w, h, int(round(radius)), borderWidth)
    texture = shapeCache.get(key)
    if texture is not None:
        return texture
    if len(shapeCache) > 800:
        shapeCache.clear()      # unused textures are then freed from the GPU atlas
    # Match the real pixel density (125% display scaling -> 1.25 pixels per point)
    window = arcade.get_window()
    pixelRatio = 1.0
    if window is not None:
        pixelRatio = max(1.0, window.get_pixel_ratio())
    width = max(2, int(round((w + borderWidth) * pixelRatio)))
    height = max(2, int(round((h + borderWidth) * pixelRatio)))
    corner = int(round((radius + borderWidth / 2) * pixelRatio))
    corner = max(1, min(corner, width // 2, height // 2))
    border = 0
    if borderWidth > 0:
        border = max(1, int(round(borderWidth * pixelRatio)))

    if border == 0:
        mask = Image.new("L", (width, height), 255)
    else:
        # outline: four straight bands, the corners are pasted on top
        mask = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask)
        draw.rectangle((0, 0, width - 1, border - 1), fill=255)
        draw.rectangle((0, height - border, width - 1, height - 1), fill=255)
        draw.rectangle((0, 0, border - 1, height - 1), fill=255)
        draw.rectangle((width - border, 0, width - 1, height - 1), fill=255)
    cornerImage = getCornerImage(corner, border)
    # the same corner mirrored into all four corners (flip = mirroring transformation)
    mask.paste(cornerImage, (0, 0))
    mask.paste(cornerImage.transpose(Image.FLIP_LEFT_RIGHT), (width - corner, 0))
    mask.paste(cornerImage.transpose(Image.FLIP_TOP_BOTTOM), (0, height - corner))
    mask.paste(cornerImage.transpose(Image.ROTATE_180), (width - corner, height - corner))

    white = Image.new("L", (width, height), 255)
    image = Image.merge("RGBA", (white, white, white, mask))
    texture = assets.textureFromImage(image, f"shape:{w}x{h}:{key[2]}:{borderWidth}")
    shapeCache[key] = texture
    return texture


def drawShape(cx, cy, w, h, rgba, radius, borderWidth):
    if w <= 0 or h <= 0:
        return
    alpha = getAlpha(rgba)
    if alpha <= 0:
        return
    radius = max(0.0, min(radius, w / 2, h / 2))
    snappedW = snapSize(w)
    snappedH = snapSize(h)
    texture = getShapeTexture(snappedW, snappedH, radius, borderWidth)
    drawTexture(texture, cx, cy, w + borderWidth, h + borderWidth, alpha=alpha, rgb=rgba)


def drawRoundedRect(cx, cy, w, h, rgba, radius=16):
    """Filled rounded rectangle centred on (cx, cy)."""
    if radius < 1:
        arcade.draw_lbwh_rectangle_filled(cx - w / 2, cy - h / 2, w, h, rgba)
        return
    drawShape(cx, cy, w, h, rgba, radius, 0)


def drawRoundedRectOutline(cx, cy, w, h, rgba, radius=16, borderWidth=3):
    """Outline of a rounded rectangle (the border is centred on the edge)."""
    if borderWidth <= 0:
        return
    drawShape(cx, cy, w, h, rgba, radius, borderWidth)


def drawPanel(cx, cy, w, h, fill=(24, 26, 52, 230), border=GOLD, radius=18, shadow=True):
    """A game panel: soft shadow, translucent fill, top highlight and a double golden frame."""
    if shadow:
        drawRoundedRect(cx + 4, cy - 8, w + 4, h + 2, (0, 0, 0, 95), radius + 2)
    drawRoundedRect(cx, cy, w, h, fill, radius)
    # a faint light line near the top edge makes the panel look glassy
    r = min(radius, w / 2, h / 2)
    if w > 2 * r + 8 and h > 20:
        arcade.draw_lbwh_rectangle_filled(cx - w / 2 + r, cy + h / 2 - 7, w - 2 * r, 2, (255, 255, 255, 26))
    if border is not None:
        alpha = getAlpha(border)
        drawRoundedRectOutline(cx, cy, w, h, withAlpha(border, alpha), radius, 3)
        if w > 24 and h > 24:
            drawRoundedRectOutline(cx, cy, w - 12, h - 12, withAlpha(border, alpha * 0.25), max(2, radius - 5), 1.5)


def drawProgressBar(left, bottom, width, height, fraction, fillRgb, backRgba=(0, 0, 0, 140), borderRgb=None):
    """A pill-shaped bar filled to 'fraction' (0..1)."""
    fraction = clamp(fraction, 0.0, 1.0)
    radius = height / 2
    drawRoundedRect(left + width / 2, bottom + height / 2, width, height, backRgba, radius)
    fillWidth = width * fraction
    if fillWidth > 0.5:
        fillRadius = min(radius, fillWidth / 2)
        drawRoundedRect(left + fillWidth / 2, bottom + height / 2, fillWidth, height,
                        withAlpha(fillRgb, getAlpha(fillRgb)), fillRadius)
        if fillWidth > 10 and height >= 8:
            # a light stripe on the upper part gives the bar a rounded 3D look
            drawRoundedRect(left + fillWidth / 2, bottom + height * 0.7, fillWidth - 6, height * 0.24,
                            (255, 255, 255, 60), height * 0.12)
    if borderRgb is not None:
        drawRoundedRectOutline(left + width / 2, bottom + height / 2, width, height,
                               withAlpha(borderRgb, getAlpha(borderRgb)), radius, 2)


def drawKeyCap(cx, cy, keyLabel, alpha=255):
    """A small keyboard key picture with a label. Returns its width."""
    label = getText(keyLabel, 13, FONT_BODY, bold=True, anchorX="center", anchorY="center")
    capWidth = max(34, label.content_width + 22)
    capHeight = 30
    drawRoundedRect(cx, cy - 3, capWidth, capHeight, (120, 92, 74, int(alpha)), 7)      # key side
    drawRoundedRect(cx, cy, capWidth, capHeight - 2, (255, 244, 224, int(alpha)), 7)    # key top
    label.x = cx
    label.y = cy + 1
    setTextColor(label, withAlpha(DARK_TEXT, alpha))
    label.draw()
    return capWidth


def drawKeyHint(cx, cy, keyLabel, text):
    """Key cap centred on (cx, cy) with a label to its right, e.g. drawKeyHint(100, 40, "SPACE", "Chop")."""
    capWidth = drawKeyCap(cx, cy, keyLabel)
    if text:
        label = drawText(text, cx + capWidth / 2 + 10, cy, CREAM, 16, FONT_BODY, anchorY="center", bold=True)
        return capWidth + 10 + label.content_width
    return capWidth


def drawStars(cx, cy, count, maxStars=3, size=40):
    """A row of stars: 'count' full ones and the rest empty (ui/star_full.png, ui/star_empty.png)."""
    fullTexture = assets.loadTexture("ui/star_full.png")
    emptyTexture = assets.loadTexture("ui/star_empty.png")
    spacing = size * 1.15
    startX = cx - (maxStars - 1) * spacing / 2
    for i in range(maxStars):
        x = startX + i * spacing
        y = cy
        starSize = size
        if maxStars == 3 and i == 1:
            starSize = size * 1.18      # the middle star is a little bigger and higher
            y = cy + size * 0.12
        if i < count:
            drawTexture(fullTexture, x, y, starSize, starSize)
        else:
            drawTexture(emptyTexture, x, y, starSize, starSize)


def getClock():
    """Game clock of the window (used for animations inside draw functions)."""
    window = arcade.get_window()
    if window is None:
        return 0.0
    return getattr(window, "gameClock", 0.0)


def playUiSound(name, volume=0.6, speed=1.0):
    window = arcade.get_window()
    sound = getattr(window, "sound", None)
    if sound is not None:
        sound.play(name, volume, 0.0, speed)


# ----------------------------------------------------------------------
# Button
# ----------------------------------------------------------------------
class Button:
    def __init__(self, label, cx, cy, w=320, h=62, fontSize=20):
        self.label = label
        self.cx = cx
        self.cy = cy
        self.w = w
        self.h = h
        self.fontSize = fontSize
        self.hovered = False
        self.enabled = True
        self.hoverAmount = 0.0      # 0..1, animated in draw()
        self.lastClock = None
        self.text = arcade.Text(label, cx, cy, CREAM, fontSize, font_name=FONT_TITLE,
                                anchor_x="center", anchor_y="center")
        self.shadowText = arcade.Text(label, cx + 2, cy - 2, SHADOW_COLOR, fontSize, font_name=FONT_TITLE,
                                      anchor_x="center", anchor_y="center")

    def setLabel(self, label):
        self.label = label
        self.text.text = label
        self.shadowText.text = label

    def contains(self, x, y):
        return abs(x - self.cx) <= self.w / 2 and abs(y - self.cy) <= self.h / 2

    def updateHover(self, mx, my):
        """Call on mouse motion. Returns True if the hover state changed."""
        newHover = self.enabled and self.contains(mx, my)
        if newHover != self.hovered:
            self.hovered = newHover
            if newHover:
                playUiSound("ui_hover", 0.5)
            return True
        return False

    def draw(self):
        now = getClock()
        if self.lastClock is None:
            self.lastClock = now
        dt = clamp(now - self.lastClock, 0.0, 0.1)
        self.lastClock = now
        if self.hovered:
            target = 1.0
        else:
            target = 0.0
        self.hoverAmount = approach(self.hoverAmount, target, 14.0, dt)

        h = self.hoverAmount
        scale = 1.0 + 0.05 * h
        w = self.w * scale
        height = self.h * scale
        if self.enabled:
            fill = lerpColor(withAlpha(UI_PANEL, 225), withAlpha(UI_PANEL_LIGHT, 245), h)
            border = lerpColor((120, 110, 150, 200), withAlpha(GOLD, 255), h)
            textColor = lerpColor(CREAM, WHITE, h)
        else:
            fill = (30, 30, 44, 200)
            border = (80, 80, 100, 200)
            textColor = (130, 130, 150)
        drawRoundedRect(self.cx + 3, self.cy - 6, w, height, (0, 0, 0, 90), 16)
        drawRoundedRect(self.cx, self.cy, w, height, fill, 16)
        if h > 0.02:
            # warm glow strip along the bottom when hovered
            drawRoundedRect(self.cx, self.cy - height / 2 + 6, w - 24, 5, withAlpha(GOLD, 120 * h), 2)
        drawRoundedRectOutline(self.cx, self.cy, w, height, border, 16, 3)
        if h > 0.05:
            # little lantern dots on both sides of the label
            dotX = w / 2 - 22
            arcade.draw_circle_filled(self.cx - dotX, self.cy, 5 * h, withAlpha(GOLD, 255 * h))
            arcade.draw_circle_filled(self.cx + dotX, self.cy, 5 * h, withAlpha(GOLD, 255 * h))
        self.shadowText.x = self.cx + 2
        self.shadowText.y = self.cy - 1
        self.shadowText.draw()
        self.text.x = self.cx
        self.text.y = self.cy + 1
        setTextColor(self.text, textColor)
        self.text.draw()


# ----------------------------------------------------------------------
# Dialogue box
# ----------------------------------------------------------------------
PANEL_WIDTH = 1180
PANEL_HEIGHT = 190
PANEL_CY = 112
TYPE_SPEED = 45           # characters per second (typewriter effect)
AUTO_ADVANCE_WAIT = 1.3   # seconds to wait on a finished line in autoplay
BODY_SIZE = 21
BODY_WIDTH = 880
NARRATOR_WIDTH = 1060
MAX_LINES = 4             # the text is wrapped into at most this many lines
LINE_HEIGHT = 30


class DialogueBox:
    def __init__(self):
        self.window = arcade.get_window()
        self.lines = []
        self.index = 0
        self.active = False
        self.speakerId = "narrator"
        self.wrappedText = ""
        self.shownChars = 0.0
        self.lineDone = False
        self.waitTime = 0.0
        self.letterCount = 0
        self.appear = 0.0
        self.time = 0.0
        # Text objects made once
        # One single-line Text per wrapped line. (pyglet's multiline=True
        # layout sometimes swallows spaces, e.g. "looks correct" -> "lookscorrect")
        self.lineTexts = []
        self.lineShadows = []
        for i in range(MAX_LINES):
            self.lineTexts.append(arcade.Text("", 0, 0, CREAM, BODY_SIZE, font_name=FONT_BODY, anchor_y="top"))
            self.lineShadows.append(arcade.Text("", 0, 0, SHADOW_COLOR, BODY_SIZE, font_name=FONT_BODY,
                                                anchor_y="top"))
        self.italic = False
        self.measureText = arcade.Text("", 0, 0, CREAM, BODY_SIZE, font_name=FONT_BODY)
        self.nameText = arcade.Text("", 0, 0, DARK_TEXT, 17, font_name=FONT_BODY, bold=True,
                                    anchor_x="center", anchor_y="center")

    # --- control -------------------------------------------------------
    def start(self, lines):
        """lines = [(speakerId, text), ...]"""
        self.lines = list(lines)
        self.index = 0
        if len(self.lines) == 0:
            self.active = False
            return
        self.active = True
        self.appear = 0.0
        self.prepareLine()

    def isActive(self):
        return self.active

    def skipAll(self):
        self.active = False

    def getSpeaker(self):
        if self.speakerId in SPEAKERS:
            return SPEAKERS[self.speakerId]
        return SPEAKERS["narrator"]

    def wrapText(self, text, maxWidth):
        """
        Word wrap done BEFORE the typewriter starts, so words never jump to
        the next line while they are being typed. We measure each candidate
        line with a hidden Text object (content_width).
        """
        words = text.split(" ")
        lines = []
        current = ""
        for word in words:
            if current == "":
                candidate = word
            else:
                candidate = current + " " + word
            self.measureText.text = candidate
            if self.measureText.content_width <= maxWidth or current == "":
                current = candidate
            else:
                lines.append(current)
                current = word
        if current != "":
            lines.append(current)
        return "\n".join(lines)

    def prepareLine(self):
        speakerId, text = self.lines[self.index]
        self.speakerId = speakerId
        speaker = self.getSpeaker()
        if speaker["portrait"] is None:
            maxWidth = NARRATOR_WIDTH
            italic = True      # the narrator speaks in italics
        else:
            maxWidth = BODY_WIDTH
            italic = False
        if italic != self.italic:
            self.italic = italic
            self.measureText.italic = italic
            for i in range(MAX_LINES):
                self.lineTexts[i].italic = italic
                self.lineShadows[i].italic = italic
        self.wrappedText = self.wrapText(str(text), maxWidth)
        self.shownChars = 0.0
        self.lineDone = False
        self.waitTime = 0.0
        self.letterCount = 0
        self.setShownText("")
        self.nameText.text = speaker["name"]
        # Speakers without a talking voice (the cat) make one sound per line
        voice = speaker["voice"]
        if voice is not None and not voice.startswith("voice_"):
            self.playVoice(voice, speaker["pitch"])

    def setShownText(self, text):
        """Put the visible part of the line into the single-line Text objects."""
        parts = text.split("\n")
        for i in range(MAX_LINES):
            if i < len(parts):
                lineString = parts[i]
            else:
                lineString = ""
            if self.lineTexts[i].text != lineString:
                self.lineTexts[i].text = lineString
                self.lineShadows[i].text = lineString

    def playVoice(self, voice, pitch):
        sound = getattr(self.window, "sound", None)
        if sound is not None and voice is not None:
            sound.play(voice, 0.45, 0.0, pitch * random.uniform(0.9, 1.12))

    def advance(self):
        """Still typing -> show the whole line. Line finished -> next line (or close)."""
        if not self.active:
            return
        if not self.lineDone:
            self.shownChars = len(self.wrappedText)
            self.lineDone = True
            self.setShownText(self.wrappedText)
            return
        self.index += 1
        if self.index >= len(self.lines):
            self.active = False
            playUiSound("ui_click", 0.35)
        else:
            self.prepareLine()

    def update(self, dt, autoAdvance=False):
        if not self.active:
            return
        self.time += dt
        self.appear = min(1.0, self.appear + dt / 0.25)
        if not self.lineDone:
            before = int(self.shownChars)
            self.shownChars += TYPE_SPEED * dt
            after = min(len(self.wrappedText), int(self.shownChars))
            if after != before:
                self.setShownText(self.wrappedText[:after])
                # Voice blip every 2nd letter (not for spaces or punctuation)
                speaker = self.getSpeaker()
                voice = speaker["voice"]
                blipDone = False
                for i in range(before, after):
                    if self.wrappedText[i].isalnum():
                        self.letterCount += 1
                        isTalkingVoice = voice is not None and voice.startswith("voice_")
                        if self.letterCount % 2 == 1 and not blipDone and isTalkingVoice:
                            self.playVoice(voice, speaker["pitch"])
                            blipDone = True
            if after >= len(self.wrappedText):
                self.lineDone = True
        elif autoAdvance:
            self.waitTime += dt
            if self.waitTime >= AUTO_ADVANCE_WAIT:
                self.advance()

    # --- drawing -------------------------------------------------------
    def draw(self):
        if not self.active:
            return
        speaker = self.getSpeaker()
        slide = easeOutQuad(self.appear)
        cy = PANEL_CY - (1 - slide) * 50
        alpha = 255 * slide
        speakerColor = speaker["color"]
        drawPanel(SCREEN_WIDTH / 2, cy, PANEL_WIDTH, PANEL_HEIGHT, (18, 20, 44, int(236 * slide)),
                  withAlpha(GOLD, alpha), 20)
        panelLeft = SCREEN_WIDTH / 2 - PANEL_WIDTH / 2
        panelTop = cy + PANEL_HEIGHT / 2

        if speaker["portrait"] is not None:
            # Portrait inside a rounded frame with a colour backdrop
            frameX = panelLeft + 100
            drawRoundedRect(frameX, cy, 164, 164, withAlpha(lerpColor(speakerColor, (20, 20, 40), 0.55), alpha), 16)
            bob = 0.0
            if not self.lineDone:
                bob = math.sin(self.time * 16) * 2.0    # little talking bob
            portrait = assets.loadTexture(speaker["portrait"])
            drawTexture(portrait, frameX, cy + 2 + bob, 156, 156, alpha=alpha)
            drawRoundedRectOutline(frameX, cy, 164, 164, withAlpha(speakerColor, alpha), 16, 3)
            textLeft = panelLeft + 206
            # Name tag sitting on the top edge of the panel
            tagWidth = self.nameText.content_width + 40
            drawRoundedRect(textLeft + tagWidth / 2 + 2, panelTop - 1 - 3, tagWidth, 36,
                            withAlpha((0, 0, 0), 90 * slide), 12)
            drawRoundedRect(textLeft + tagWidth / 2, panelTop + 1, tagWidth, 36, withAlpha(speakerColor, alpha), 12)
            self.nameText.x = textLeft + tagWidth / 2
            self.nameText.y = panelTop + 1
            setTextColor(self.nameText, withAlpha(DARK_TEXT, alpha))
            self.nameText.draw()
            textTop = panelTop - 36
            textColor = CREAM
        else:
            textLeft = panelLeft + 60
            textTop = panelTop - 30
            textColor = (226, 222, 255)

        for i in range(MAX_LINES):
            lineText = self.lineTexts[i]
            if lineText.text == "":
                continue
            lineShadow = self.lineShadows[i]
            lineShadow.x = textLeft + 2
            lineShadow.y = textTop - 2 - i * LINE_HEIGHT
            setTextColor(lineShadow, withAlpha(SHADOW_COLOR, alpha * 0.8))
            lineShadow.draw()
            lineText.x = textLeft
            lineText.y = textTop - i * LINE_HEIGHT
            setTextColor(lineText, withAlpha(textColor, alpha))
            lineText.draw()

        # Progress dots (which line of how many)
        count = len(self.lines)
        if count > 1:
            startX = SCREEN_WIDTH / 2 - (count - 1) * 9
            for i in range(count):
                if i == self.index:
                    colour = withAlpha(GOLD, alpha)
                else:
                    colour = (120, 110, 150, int(alpha * 0.8))
                arcade.draw_circle_filled(startX + i * 18, cy - PANEL_HEIGHT / 2 + 16, 4, colour)

        # Blinking "SPACE" prompt when the line is complete
        if self.lineDone:
            blink = 0.55 + 0.45 * math.sin(self.time * 6)
            capX = panelLeft + PANEL_WIDTH - 70
            capY = cy - PANEL_HEIGHT / 2 + 26
            drawKeyCap(capX, capY, "SPACE", 255 * blink * slide)
            arrowX = capX - 52
            arrowY = capY + math.sin(self.time * 8) * 3
            arcade.draw_triangle_filled(arrowX - 7, arrowY + 5, arrowX + 7, arrowY + 5, arrowX, arrowY - 6,
                                        withAlpha(GOLD, 255 * blink * slide))
        drawText("TAB  skip", panelLeft + PANEL_WIDTH - 22, cy + PANEL_HEIGHT / 2 - 22, (150, 140, 180), 11,
                 FONT_NARROW, anchorX="right", anchorY="center", alpha=alpha * 0.8, shadow=False)


# ----------------------------------------------------------------------
# HUD (heads-up display) for the dining scenes
# ----------------------------------------------------------------------
class Hud:
    def __init__(self):
        self.displayScore = 0.0
        self.lastClock = None
        self.lastCombo = 0
        self.comboBump = 0.0
        self.titleText = arcade.Text("", 30, 694, GOLD, 18, font_name=FONT_BODY, bold=True, anchor_y="center")
        self.stallText = arcade.Text("", 30, 670, CREAM, 12, font_name=FONT_BODY, anchor_y="center")
        self.timerText = arcade.Text("00:00", 652, 686, CREAM, 26, font_name=FONT_TITLE,
                                     anchor_x="center", anchor_y="center")
        self.scoreLabel = arcade.Text("SCORE", 1040, 686, LAVENDER, 12, font_name=FONT_NARROW,
                                      anchor_x="right", anchor_y="center")
        self.scoreText = arcade.Text("0", 1244, 686, WHITE, 26, font_name=FONT_TITLE,
                                     anchor_x="right", anchor_y="center")
        self.comboText = arcade.Text("", 948, 628, GOLD, 18, font_name=FONT_TITLE,
                                     anchor_x="right", anchor_y="center")
        self.comboShadow = arcade.Text("", 950, 625, SHADOW_COLOR, 18, font_name=FONT_TITLE,
                                       anchor_x="right", anchor_y="center")
        self.auraText = arcade.Text("AURA LADLE", 1012, 640, CREAM, 11, font_name=FONT_NARROW, anchor_y="center")
        self.timerFontSize = 26
        self.comboFontSize = 18

    def draw(self, scene):
        now = getClock()
        if self.lastClock is None:
            self.lastClock = now
        dt = clamp(now - self.lastClock, 0.0, 0.1)
        self.lastClock = now

        # Score counts up smoothly towards the real score
        self.displayScore = approach(self.displayScore, scene.score, 9.0, dt)
        if abs(self.displayScore - scene.score) < 0.5:
            self.displayScore = scene.score

        # --- top bar ---
        drawPanel(SCREEN_WIDTH / 2, 686, 1256, 56, (14, 16, 38, 205), (255, 200, 87, 150), 16, shadow=False)
        self.titleText.text = scene.sceneTitle.upper()
        self.titleText.draw()
        self.stallText.text = scene.stallName
        self.stallText.draw()

        # --- timer in the centre ---
        if scene.timeLimit > 0:
            timeLeft = max(0.0, scene.timeLeft)
            seconds = int(math.ceil(timeLeft))
            self.timerText.text = f"{seconds // 60:02d}:{seconds % 60:02d}"
            clockIcon = assets.loadTexture("ui/icon_clock.png")
            if timeLeft < 10:
                pulse = 0.5 + 0.5 * math.sin(now * 10)
                arcade.draw_circle_filled(640, 686, 44 + 6 * pulse, withAlpha(BAD_RED, 50 + 50 * pulse))
                fontSize = 26 + int(round(4 * pulse))
                colour = lerpColor(BAD_RED, (255, 200, 200), pulse * 0.4)
            else:
                fontSize = 26
                colour = CREAM
            if fontSize != self.timerFontSize:
                self.timerFontSize = fontSize
                self.timerText.font_size = fontSize
            drawTexture(clockIcon, 572, 686, 30, 30)
            setTextColor(self.timerText, colour)
            self.timerText.draw()

        # --- score on the right ---
        self.scoreLabel.draw()
        self.scoreText.text = str(int(self.displayScore))
        self.scoreText.draw()

        # --- combo + Aura meter under the right side ---
        auraCharge = getattr(scene, "auraCharge", scene.combo)
        if scene.combo != self.lastCombo:
            if scene.combo > self.lastCombo:
                self.comboBump = 1.0
            self.lastCombo = scene.combo
        self.comboBump = max(0.0, self.comboBump - dt * 4)
        if scene.combo >= 2:
            fontSize = 18 + int(round(8 * easeOutQuad(self.comboBump)))
            if fontSize != self.comboFontSize:
                self.comboFontSize = fontSize
                self.comboText.font_size = fontSize
                self.comboShadow.font_size = fontSize
            comboString = f"COMBO {scene.combo}"
            if self.comboText.text != comboString:
                self.comboText.text = comboString
                self.comboShadow.text = comboString
            self.comboShadow.draw()
            setTextColor(self.comboText, lerpColor(GOLD, WHITE, self.comboBump))
            self.comboText.draw()

        drawPanel(1116, 626, 268, 44, (14, 16, 38, 200), (255, 200, 87, 110), 14, shadow=False)
        ladle = assets.loadTexture("ui/icon_ladle.png")
        glow = 0.0
        if scene.auraMode:
            glow = 0.6 + 0.4 * math.sin(now * 8)
            drawTexture(assets.loadTexture("particles/light_glow.png"), 1004, 626, 90, 90,
                        rgb=GOLD, alpha=200 * glow, additive=True)
        drawTexture(ladle, 1004, 626, 36, 36)
        barLeft = 1030
        barWidth = 230
        if scene.auraMode:
            fraction = scene.auraTimer / AURA_MODE_TIME
            self.auraText.text = "AURA MODE - DOUBLE POINTS"
            setTextColor(self.auraText, GOLD)
            shimmer = lerpColor(GOLD, (255, 250, 210), 0.5 + 0.5 * math.sin(now * 12))
            drawProgressBar(barLeft, 608, barWidth, 13, fraction, shimmer, (0, 0, 0, 150), None)
            # a bright band sweeping along the bar
            sweepX = barLeft + (now * 260) % max(1.0, barWidth * fraction)
            drawRoundedRect(sweepX, 614.5, 18, 9, (255, 255, 255, 110), 4)
        else:
            fraction = auraCharge / COMBO_FOR_AURA
            self.auraText.text = "AURA LADLE"
            setTextColor(self.auraText, CREAM)
            barColour = lerpColor((150, 120, 200), GOLD, fraction)
            drawProgressBar(barLeft, 608, barWidth, 13, fraction, barColour, (0, 0, 0, 150), None)
        self.auraText.x = barLeft
        self.auraText.y = 638
        self.auraText.draw()


# ----------------------------------------------------------------------
# Result card
# ----------------------------------------------------------------------
CARD_WIDTH = 640
CARD_HEIGHT = 470
CARD_CY = 360
POP_TIME = 0.35
SCORE_START = 0.55
SCORE_END = 1.75
STAR_TIMES = [2.0, 2.4, 2.8]
INFO_TIME = 3.1
DONE_TIME = 3.5


class ResultCard:
    def __init__(self, title, dishName, dishTexture, score, maxScore, stars, bestCombo, extraLines):
        self.window = arcade.get_window()
        self.title = title
        self.dishName = dishName
        self.dishTexture = dishTexture
        self.score = int(score)
        self.maxScore = int(maxScore)
        self.stars = int(clamp(stars, 0, 3))
        self.bestCombo = int(bestCombo)
        self.extraLines = []
        if extraLines:
            self.extraLines = list(extraLines)[:3]
        self.time = 0.0
        self.shownScore = 0
        self.lastTickTime = -1.0
        self.stamped = [False, False, False]
        self.shakeTime = 1.0
        self.titleText = arcade.Text(title.upper(), 640, 566, GOLD, 26, font_name=FONT_TITLE,
                                     anchor_x="center", anchor_y="center")
        # long titles are made smaller until they fit inside the card
        while self.titleText.content_width > CARD_WIDTH - 60 and self.titleText.font_size > 14:
            self.titleText.font_size = self.titleText.font_size - 2
        self.dishText = arcade.Text(dishName, 640, 402, CREAM, 22, font_name=FONT_BODY, bold=True,
                                    anchor_x="center", anchor_y="center")
        self.scoreText = arcade.Text("0", 640, 354, WHITE, 32, font_name=FONT_TITLE,
                                     anchor_x="center", anchor_y="center")

    def playSound(self, name, volume=0.8):
        sound = getattr(self.window, "sound", None)
        if sound is not None:
            sound.play(name, volume)

    def update(self, dt):
        self.time += dt
        self.shakeTime += dt
        # Score counts up (linear interpolation eased) with little ticks
        t = clamp((self.time - SCORE_START) / (SCORE_END - SCORE_START), 0.0, 1.0)
        self.shownScore = int(round(lerp(0, self.score, easeOutQuad(t))))
        if 0.0 < t < 1.0 and self.time - self.lastTickTime > 0.07:
            self.lastTickTime = self.time
            self.playSound("score_tick", 0.35)
        # Stars stamp in one by one
        for i in range(self.stars):
            if not self.stamped[i] and self.time >= STAR_TIMES[i]:
                self.stamped[i] = True
                self.playSound(f"star_{i + 1}", 0.9)
                self.shakeTime = 0.0
                if i == self.stars - 1:
                    self.playSound("result_stamp", 0.7)

    def skip(self):
        """ENTER during the animation jumps to the end."""
        if self.time < DONE_TIME:
            self.time = DONE_TIME
            self.shownScore = self.score
            for i in range(self.stars):
                self.stamped[i] = True

    def isDone(self):
        return self.time >= DONE_TIME

    def draw(self):
        now = self.time
        # Darken the scene behind the card
        dim = clamp(now / 0.3, 0.0, 1.0)
        arcade.draw_lbwh_rectangle_filled(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT, (6, 6, 16, int(150 * dim)))

        # Small shake when a star lands
        shakeX = 0.0
        shakeY = 0.0
        if self.shakeTime < 0.25:
            amount = 6 * (1 - self.shakeTime / 0.25)
            shakeX = math.sin(self.shakeTime * 90) * amount
            shakeY = math.cos(self.shakeTime * 70) * amount

        pop = easeOutBack(clamp(now / POP_TIME, 0.0, 1.0))
        cx = 640 + shakeX
        cy = CARD_CY + shakeY
        drawPanel(cx, cy, CARD_WIDTH * pop, CARD_HEIGHT * pop, (22, 24, 52, 242), GOLD, 26)
        if now < POP_TIME * 0.8:
            return
        fade = clamp((now - POP_TIME * 0.8) / 0.25, 0.0, 1.0)
        alpha = int(255 * fade)

        self.titleText.x = cx
        self.titleText.y = 566 + shakeY
        setTextColor(self.titleText, withAlpha(GOLD, alpha))
        self.titleText.draw()

        # Dish on a glowing plate
        glowPulse = 0.8 + 0.2 * math.sin(now * 3)
        drawTexture(assets.loadTexture("particles/light_glow.png"), cx, 470 + shakeY, 380, 260,
                    rgb=WARM_LIGHT, alpha=150 * fade * glowPulse, additive=True)
        arcade.draw_ellipse_filled(cx, 446 + shakeY, 270, 64, (40, 30, 40, int(120 * fade)))
        arcade.draw_ellipse_filled(cx, 450 + shakeY, 256, 60, (250, 244, 232, alpha))
        arcade.draw_ellipse_outline(cx, 450 + shakeY, 212, 44, (220, 200, 170, alpha), 2)
        if self.dishTexture is not None:
            grow = easeOutBack(clamp((now - 0.3) / 0.45, 0.0, 1.0))
            scale = min(230 / self.dishTexture.width, 125 / self.dishTexture.height) * grow
            bob = math.sin(now * 2.2) * 2
            if scale > 0.01:
                drawTexture(self.dishTexture, cx, 488 + shakeY + bob, self.dishTexture.width * scale,
                            self.dishTexture.height * scale, alpha=alpha)

        self.dishText.x = cx
        self.dishText.y = 406 + shakeY
        setTextColor(self.dishText, withAlpha(CREAM, alpha))
        self.dishText.draw()

        # Aura Mode doubles points, so a great run can pass maxScore: show the
        # score and the 3-star target instead of "score / max"
        threeStarTarget = int(self.maxScore * STAR_THRESHOLDS[1])
        drawText(f"SCORE   (3 stars at {threeStarTarget})", cx, 378 + shakeY, LAVENDER, 12, FONT_NARROW,
                 anchorX="center", anchorY="center", alpha=alpha, shadow=False)
        self.scoreText.text = str(self.shownScore)
        self.scoreText.x = cx
        self.scoreText.y = 348 + shakeY
        setTextColor(self.scoreText, withAlpha(WHITE, alpha))
        self.scoreText.draw()

        # Stars: empty slots first, full stars stamp in (scale 2 -> 1)
        fullTexture = assets.loadTexture("ui/star_full.png")
        emptyTexture = assets.loadTexture("ui/star_empty.png")
        for i in range(3):
            size = 58
            x = cx + (i - 1) * 74
            y = 292 + shakeY
            if i == 1:
                size = 68
                y = y + 8
            drawTexture(emptyTexture, x, y, size, size, alpha=alpha)
            if self.stamped[i]:
                t = clamp((now - STAR_TIMES[i]) / 0.25, 0.0, 1.0)
                if now >= DONE_TIME and t < 1.0:
                    t = 1.0
                starScale = lerp(2.0, 1.0, easeOutQuad(t))
                drawTexture(fullTexture, x, y, size * starScale, size * starScale, alpha=255 * t)

        # Best combo and extra lines fade in
        infoFade = clamp((now - INFO_TIME) / 0.3, 0.0, 1.0)
        if infoFade > 0:
            infoAlpha = 255 * infoFade
            drawText(f"Best combo  x{self.bestCombo}", cx, 244 + shakeY, GOLD, 18, FONT_BODY,
                     anchorX="center", anchorY="center", bold=True, alpha=infoAlpha)
            y = 216
            for line in self.extraLines:
                drawText(line, cx, y + shakeY, CREAM, 16, FONT_BODY, anchorX="center", anchorY="center",
                         alpha=infoAlpha)
                y -= 24

        if self.isDone():
            blink = 0.5 + 0.5 * math.sin(now * 5)
            drawText("ENTER / CLICK to continue", cx, 146 + shakeY, lerpColor(CREAM, GOLD, blink), 15, FONT_TITLE,
                     anchorX="center", anchorY="center", alpha=150 + 105 * blink)
