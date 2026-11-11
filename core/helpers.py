"""
helpers.py
----------
Small maths helpers used everywhere in the game.

Most animation in Midnight Makan is built from two ideas from the lectures:
  * Linear interpolation (Week 7): value = start + (end - start) * t,
    where t goes from 0 to 1 over time.
  * Easing curves: instead of feeding t in directly, we pass it through a
    curve (easeOutBack, easeOutBounce ...) so the motion speeds up, slows
    down or overshoots, which looks more alive than straight lines.

A Tween object wraps this up: it animates ONE number from start to end.
This file does not import arcade, so it can be tested on its own.
"""
import colorsys
import math


# ----------------------------------------------------------------------
# Interpolation
# ----------------------------------------------------------------------
def lerp(a, b, t):
    """Linear interpolation (Week 7): the point a fraction t of the way from a to b."""
    return a + (b - a) * t


def clamp(value, low, high):
    """Keep a value inside [low, high] (same idea as checkBoundary() in the tutorials)."""
    return max(low, min(high, value))


def lerpColor(c1, c2, t):
    """
    Interpolate two colours channel by channel (RGB colour model).
    Works for RGB or RGBA tuples. If only one colour has alpha, the other
    one is treated as fully opaque (255). Returns a tuple of ints.
    """
    t = clamp(t, 0.0, 1.0)
    size = max(len(c1), len(c2))
    result = []
    for i in range(size):
        if i < len(c1):
            a = c1[i]
        else:
            a = 255
        if i < len(c2):
            b = c2[i]
        else:
            b = 255
        result.append(int(round(a + (b - a) * t)))
    return tuple(result)


def smoothstep(t):
    """S-shaped curve: slow start, slow end (3t^2 - 2t^3)."""
    t = clamp(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


# ----------------------------------------------------------------------
# Easing functions. Each one takes t (0..1) and returns the eased value.
# ----------------------------------------------------------------------
def easeInQuad(t):
    """Starts slow, ends fast (accelerating)."""
    return t * t


def easeOutQuad(t):
    """Starts fast, ends slow (decelerating) - good for things coming to rest."""
    return 1 - (1 - t) * (1 - t)


def easeInOutSine(t):
    """Gentle slow-fast-slow curve based on a cosine wave."""
    return -(math.cos(math.pi * t) - 1) / 2


def easeOutBack(t):
    """Goes a little past the target, then settles back - a friendly 'pop'."""
    c1 = 1.70158
    c3 = c1 + 1
    return 1 + c3 * (t - 1) ** 3 + c1 * (t - 1) ** 2


def easeOutElastic(t):
    """Springy wobble around the target (a damped sine wave)."""
    if t <= 0:
        return 0.0
    if t >= 1:
        return 1.0
    c4 = (2 * math.pi) / 3
    return 2 ** (-10 * t) * math.sin((t * 10 - 0.75) * c4) + 1


def easeOutBounce(t):
    """Bounces on the floor like a dropped ball (pieces of parabolas)."""
    n1 = 7.5625
    d1 = 2.75
    if t < 1 / d1:
        return n1 * t * t
    elif t < 2 / d1:
        t = t - 1.5 / d1
        return n1 * t * t + 0.75
    elif t < 2.5 / d1:
        t = t - 2.25 / d1
        return n1 * t * t + 0.9375
    else:
        t = t - 2.625 / d1
        return n1 * t * t + 0.984375


# ----------------------------------------------------------------------
# Other small maths helpers
# ----------------------------------------------------------------------
def approach(current, target, rate, dt):
    """
    Exponential smoothing: move 'current' part of the way to 'target'.
    The same rate gives the same feel at any frame rate because we use
    1 - e^(-rate * dt) instead of a fixed fraction per frame.
    """
    return current + (target - current) * (1 - math.exp(-rate * dt))


def distance(x1, y1, x2, y2):
    """Straight-line distance between two points (Pythagoras)."""
    return math.hypot(x2 - x1, y2 - y1)


def hsvColor(h, s, v, alpha=255):
    """
    Build a colour with the HSV colour model and return RGBA ints.
    h: hue 0..1 (0 = red, 1/3 = green, 2/3 = blue). Values above 1 are
       treated as degrees (0..360) to be forgiving.
    s: saturation 0..1, v: value (brightness) 0..1.
    """
    if h > 1.0:
        h = h / 360.0
    r, g, b = colorsys.hsv_to_rgb(h % 1.0, clamp(s, 0.0, 1.0), clamp(v, 0.0, 1.0))
    return (int(r * 255), int(g * 255), int(b * 255), int(alpha))


def angleDifference(a, b):
    """Smallest signed difference b - a between two angles in degrees (-180..180)."""
    difference = (b - a) % 360
    if difference > 180:
        difference -= 360
    return difference


def pingPong(value, length):
    """Goes 0 -> length -> 0 -> length ... as value grows (like a knife sweeping)."""
    if length <= 0:
        return 0
    value = value % (length * 2)
    if value > length:
        return length * 2 - value
    return value


def pulse(time, speed=4.0, low=0.0, high=1.0):
    """A smooth value that goes up and down forever (sine wave) between low and high."""
    wave = (math.sin(time * speed) + 1) / 2
    return low + (high - low) * wave


# ----------------------------------------------------------------------
# Tween: animates one number over time with an optional easing curve
# ----------------------------------------------------------------------
class Tween:
    """
    Example:
        self.knifeTween = Tween(400, 250, 0.08, easeInQuad)
        ... in update:  self.knifeTween.update(dt); y = self.knifeTween.value
    """

    def __init__(self, start, end, duration, easing=None, delay=0.0):
        self.start = start
        self.end = end
        self.duration = duration
        self.easing = easing
        self.delay = delay
        self.time = 0.0
        self.value = start
        self.progress = 0.0   # 0..1 before easing
        self.done = False

    def update(self, dt):
        if self.done:
            return self.value
        self.time += dt
        if self.time < self.delay:
            self.value = self.start
            return self.value
        if self.duration <= 0:
            t = 1.0
        else:
            t = (self.time - self.delay) / self.duration
        t = clamp(t, 0.0, 1.0)
        self.progress = t
        if self.easing is not None:
            eased = self.easing(t)
        else:
            eased = t
        self.value = lerp(self.start, self.end, eased)
        if t >= 1.0:
            self.value = self.end
            self.done = True
        return self.value

    def restart(self, start=None, end=None):
        if start is not None:
            self.start = start
        if end is not None:
            self.end = end
        self.time = 0.0
        self.progress = 0.0
        self.value = self.start
        self.done = False
