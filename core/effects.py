"""
effects.py
----------
Screen-wide special effects shared by every scene:

  ScreenShake          - camera offset that decays over time ("juice" on impacts)
  ScreenFlash          - a full-screen colour that fades out (camera flash, flare)
  FloatingTextManager  - "+100", "PERFECT!" popups that pop, rise and fade
  Transition           - IRIS WIPE between scenes (a growing / shrinking hole)
  LightList            - point lights as additive glow sprites (Week 8 lighting)
  drawVignette, drawAuraOverlay, drawDarkness - full-screen overlays

Lighting idea (Week 8): the scene is first darkened (ambient light level,
drawDarkness), then each point light ADDS brightness with additive blending,
so overlapping lights get brighter, just like real light.
"""
import math

import arcade

from core import assets
from core.helpers import clamp, easeInQuad, easeOutBack, easeOutQuad, lerp
from core.ui import drawTextTexture, makeTextTexture
from settings import (BLACK, CREAM, FONT_TITLE, GOLD, NIGHT_DEEP, SCREEN_HEIGHT,
                      SCREEN_WIDTH)


# ----------------------------------------------------------------------
# Screen shake
# ----------------------------------------------------------------------
class ScreenShake:
    def __init__(self):
        self.strength = 0.0
        self.duration = 0.0
        self.time = 0.0
        self.offsetX = 0.0
        self.offsetY = 0.0

    def currentStrength(self):
        if self.duration <= 0 or self.time >= self.duration:
            return 0.0
        remaining = 1.0 - self.time / self.duration
        return self.strength * remaining * remaining

    def start(self, strength=8, duration=0.3):
        # A new shake never makes a stronger running shake weaker
        if strength >= self.currentStrength():
            self.strength = strength
            self.duration = duration
            self.time = 0.0

    def update(self, dt):
        self.time += dt
        amount = self.currentStrength()
        if amount <= 0.01:
            self.offsetX = 0.0
            self.offsetY = 0.0
            return
        # Two sine waves with odd frequencies look like random shaking,
        # but are smooth and repeatable (no random numbers needed)
        t = self.time
        self.offsetX = amount * (math.sin(t * 71.0 + 1.3) * 0.7 + math.sin(t * 131.0) * 0.3)
        self.offsetY = amount * (math.sin(t * 53.0 + 0.4) * 0.7 + math.cos(t * 97.0 + 2.1) * 0.3)

    def getOffset(self):
        return self.offsetX, self.offsetY


# ----------------------------------------------------------------------
# Screen flash
# ----------------------------------------------------------------------
class ScreenFlash:
    def __init__(self):
        self.rgb = (255, 255, 255)
        self.duration = 0.25
        self.maxAlpha = 200
        self.time = 1.0
        self.active = False

    def start(self, rgb=(255, 255, 255), duration=0.25, maxAlpha=200):
        self.rgb = rgb
        self.duration = max(0.01, duration)
        self.maxAlpha = maxAlpha
        self.time = 0.0
        self.active = True

    def update(self, dt):
        if self.active:
            self.time += dt
            if self.time >= self.duration:
                self.active = False

    def draw(self):
        if not self.active:
            return
        t = self.time / self.duration
        alpha = self.maxAlpha * (1 - t) * (1 - t)   # bright at once, then fades quickly
        arcade.draw_lbwh_rectangle_filled(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT,
                                          (self.rgb[0], self.rgb[1], self.rgb[2], int(alpha)))


# ----------------------------------------------------------------------
# Floating score popups
# ----------------------------------------------------------------------
OUTLINE_COLOR = (26, 16, 30)
MAX_POPUPS = 14


class Popup:
    """
    One floating text. The words are rendered ONCE into a texture with a dark
    outline (see ui.makeTextTexture), so the "pop" can scale it smoothly and
    each popup costs only two draw calls (shadow + text).
    """

    def __init__(self, text, x, y, rgb, size, duration, rise):
        self.x = x
        self.startY = y
        self.y = y
        self.rgb = rgb
        self.size = size
        self.duration = max(0.2, duration)
        self.rise = rise
        self.age = 0.0
        self.alpha = 255
        outline = max(2, int(round(size / 9)))
        self.texture = makeTextTexture(text, size, FONT_TITLE, rgb, outline, OUTLINE_COLOR)

    def update(self, dt):
        self.age += dt
        t = min(1.0, self.age / self.duration)
        self.y = self.startY + self.rise * easeOutQuad(t)
        if t < 0.6:
            self.alpha = 255
        else:
            self.alpha = int(255 * (1 - (t - 0.6) / 0.4))
        return self.age < self.duration

    def draw(self):
        # "Pop": start 45% bigger and settle to normal size with easeOutBack
        popTime = min(1.0, self.age / 0.22)
        scale = lerp(1.45, 1.0, easeOutBack(popTime))
        drawTextTexture(self.texture, self.x + 2, self.y - 3, scale, self.alpha * 0.55, (0, 0, 0))
        drawTextTexture(self.texture, self.x, self.y, scale, self.alpha)


class FloatingTextManager:
    def __init__(self):
        self.popups = []

    def add(self, text, x, y, rgb=CREAM, size=26, duration=1.0, rise=70):
        if len(self.popups) >= MAX_POPUPS:
            self.popups.pop(0)
        popup = Popup(str(text), x, y, rgb, size, duration, rise)
        self.popups.append(popup)
        return popup

    def update(self, dt):
        alive = []
        for popup in self.popups:
            if popup.update(dt):
                alive.append(popup)
        self.popups = alive

    def draw(self):
        for popup in self.popups:
            popup.draw()

    def clear(self):
        self.popups = []


# ----------------------------------------------------------------------
# Iris wipe transition
# ----------------------------------------------------------------------
class Transition:
    """
    IRIS WIPE: the screen is covered by a huge black ring with a round
    hole in the middle. The hole radius is interpolated between 0 and the
    screen diagonal: closing = the hole shrinks to a point, opening = it grows.
    """

    def __init__(self):
        self.state = "none"      # "none", "closing", "closed", "opening"
        self.time = 0.0
        self.duration = 0.55
        self.x = SCREEN_WIDTH / 2
        self.y = SCREEN_HEIGHT / 2
        self.onDone = None

    def setCentre(self, x, y):
        if x is None:
            x = SCREEN_WIDTH / 2
        if y is None:
            y = SCREEN_HEIGHT / 2
        self.x = x
        self.y = y

    def maxRadius(self):
        # Distance from the centre to the farthest screen corner
        farX = max(self.x, SCREEN_WIDTH - self.x)
        farY = max(self.y, SCREEN_HEIGHT - self.y)
        return math.hypot(farX, farY) + 12

    def startClose(self, x=None, y=None, duration=0.55, onDone=None):
        self.setCentre(x, y)
        self.state = "closing"
        self.time = 0.0
        self.duration = max(0.01, duration)
        self.onDone = onDone

    def startOpen(self, x=None, y=None, duration=0.55):
        self.setCentre(x, y)
        self.state = "opening"
        self.time = 0.0
        self.duration = max(0.01, duration)
        self.onDone = None

    def update(self, dt):
        if self.state == "closing":
            self.time += dt
            if self.time >= self.duration:
                self.state = "closed"
                callback = self.onDone
                self.onDone = None
                if callback is not None:
                    callback()
        elif self.state == "opening":
            self.time += dt
            if self.time >= self.duration:
                self.state = "none"

    def isActive(self):
        return self.state != "none"

    def getRadius(self):
        t = clamp(self.time / self.duration, 0.0, 1.0)
        if self.state == "closing":
            return lerp(self.maxRadius(), 0.0, easeInQuad(t))
        if self.state == "opening":
            return lerp(0.0, self.maxRadius(), easeOutQuad(t))
        if self.state == "closed":
            return 0.0
        return self.maxRadius()

    def draw(self):
        if self.state == "none":
            return
        radius = self.getRadius()
        if radius < 1.0:
            arcade.draw_lbwh_rectangle_filled(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT, BLACK)
            return
        thickness = self.maxRadius() * 2
        # In Arcade 3.3 the outline border grows INWARDS from the given
        # radius, so radius + thickness with border thickness covers r .. r+T
        arcade.draw_circle_outline(self.x, self.y, radius + thickness, BLACK, border_width=thickness)
        # A thin golden rim with a soft glow on the edge of the hole
        arcade.draw_circle_outline(self.x, self.y, radius + 10, (GOLD[0], GOLD[1], GOLD[2], 60), border_width=10)
        arcade.draw_circle_outline(self.x, self.y, radius + 3, GOLD, border_width=4)


# ----------------------------------------------------------------------
# Point lights
# ----------------------------------------------------------------------
class Light:
    def __init__(self, x, y, radius, rgb, intensity, flicker, texture):
        self.x = x
        self.y = y
        self.radius = radius
        self.rgb = rgb
        self.intensity = intensity
        self.flicker = flicker
        self.enabled = True
        self.phase = (x * 0.137 + y * 0.071) % (math.pi * 2)
        self.sprite = arcade.Sprite(texture, 1.0, x, y)
        self.currentIntensity = intensity

    def wobble(self, time):
        """Smooth random-looking value 0..1: three sine waves with odd speeds added together."""
        p = self.phase
        value = math.sin(time * 9.1 + p) * 0.5 + math.sin(time * 17.3 + p * 2.0) * 0.3
        value += math.sin(time * 31.7 + p * 3.0) * 0.2
        return 0.5 + 0.5 * value


class LightList:
    """Point lights drawn as additive glow sprites (particles/light_glow.png)."""

    def __init__(self):
        self.lights = []
        self.spriteList = arcade.SpriteList()
        self.time = 0.0
        self.texture = assets.loadTexture("particles/light_glow.png")

    def add(self, x, y, radius, rgb, intensity=1.0, flicker=0.0):
        light = Light(x, y, radius, rgb, intensity, flicker, self.texture)
        self.lights.append(light)
        self.spriteList.append(light.sprite)
        self.updateLight(light)
        return light

    def remove(self, light):
        if light in self.lights:
            self.lights.remove(light)
            light.sprite.remove_from_sprite_lists()

    def clear(self):
        for light in list(self.lights):
            self.remove(light)

    def updateLight(self, light):
        k = light.intensity
        if light.flicker > 0:
            k = k * (1.0 - light.flicker * light.wobble(self.time))
        k = clamp(k, 0.0, 1.5)
        light.currentIntensity = k
        if not light.enabled:
            k = 0.0
        # Intensity above 1 also makes the glow a bit bigger
        size = light.radius * 2 * (1.0 + max(0.0, k - 1.0) * 0.5)
        light.sprite.center_x = light.x
        light.sprite.center_y = light.y
        light.sprite.size = (max(1.0, size), max(1.0, size))
        light.sprite.color = (light.rgb[0], light.rgb[1], light.rgb[2], int(clamp(k, 0.0, 1.0) * 255))

    def update(self, dt):
        self.time += dt
        for light in self.lights:
            self.updateLight(light)

    def draw(self):
        self.spriteList.draw(blend_function=assets.additiveBlend())

    def count(self):
        return len(self.lights)


# ----------------------------------------------------------------------
# Full-screen overlays (drawn with the GUI camera)
# ----------------------------------------------------------------------
def drawVignette(alpha=190):
    """Dark screen edges focus the eye on the centre (ui/vignette.png)."""
    texture = assets.loadTexture("ui/vignette.png")
    assets.drawTexture(texture, SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2, SCREEN_WIDTH, SCREEN_HEIGHT, alpha=alpha)


def drawAuraOverlay(time, strength):
    """Aura Mode: a warm tint plus a pulsing golden glow around the screen edges."""
    if strength <= 0.01:
        return
    strength = clamp(strength, 0.0, 1.0)
    arcade.draw_lbwh_rectangle_filled(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT, (255, 180, 70, int(20 * strength)))
    pulse = 0.72 + 0.28 * math.sin(time * 5.0)
    texture = assets.loadTexture("ui/aura_frame.png")
    assets.drawTexture(texture, SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2, SCREEN_WIDTH, SCREEN_HEIGHT,
                       alpha=255 * strength * pulse, additive=True)


def drawDarkness(alpha):
    """Ambient light level (Week 8): cover the scene with night colour; lights are drawn after it."""
    if alpha <= 0:
        return
    arcade.draw_lbwh_rectangle_filled(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT,
                                      (NIGHT_DEEP[0], NIGHT_DEEP[1], NIGHT_DEEP[2], int(clamp(alpha, 0, 255))))
