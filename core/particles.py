"""
particles.py
------------
Particle system (Week 8): EMITTER -> PARTICLES -> FORCES -> LIFE CYCLE.

  * A Particle is a small sprite with a velocity and a life time.
  * Every frame FORCES change its velocity (gravity pulls, drag slows it
    down), then the velocity changes its position (simple Euler steps).
  * During its LIFE CYCLE the size, colour and transparency are linearly
    interpolated from the start values to the end values, and when the
    life is over the particle is removed.
  * An Emitter creates new particles at a steady rate (a time accumulator),
    and the preset functions (spawnFire, spawnJuiceSplash ...) create
    particles with good-looking random values.

All particle pictures are WHITE on a transparent background, so the code
can colour them with sprite.color (colour multiplication).
Particles are kept in an arcade.SpriteList so hundreds of them are drawn
in one GPU call. A ParticleSystem with additive=True adds light instead of
covering what is behind it - perfect for fire, sparks and glows.
"""
import math
import random

import arcade

from core import assets
from core.helpers import clamp, easeOutQuad, hsvColor, lerp, lerpColor
from settings import GOLD

PARTICLE_FOLDER = "particles/"


def particleTexture(name):
    """Texture from assets/images/particles/<name>.png (a procedural stand-in if it is missing)."""
    return assets.loadTexture(PARTICLE_FOLDER + name + ".png")


# ----------------------------------------------------------------------
# One particle
# ----------------------------------------------------------------------
class Particle(arcade.Sprite):
    """
    Sizes are in pixels (sprite width). Colours are RGB tuples.
    gravity: px/s^2 (negative = falls). drag: 0..5, how quickly it slows down.
    spin: degrees per second. stretch=True points the sprite along its
    velocity and stretches it with speed (for liquid droplets).
    """

    def __init__(self, texture, x, y, vx, vy, life, startSize, endSize, startColor, endColor=None,
                 startAlpha=255, endAlpha=0, gravity=0.0, drag=0.0, spin=0.0, angle=0.0, stretch=False):
        super().__init__(texture, 1.0, x, y, angle)
        self.vx = vx
        self.vy = vy
        self.life = max(0.01, life)
        self.age = 0.0
        self.startSize = startSize
        self.endSize = endSize
        self.startColor = startColor
        if endColor is None:
            endColor = startColor
        self.endColor = endColor
        self.startAlpha = startAlpha
        self.endAlpha = endAlpha
        self.gravity = gravity
        self.drag = drag
        self.spin = spin
        self.stretch = stretch
        self.aspect = texture.height / max(1, texture.width)

        # Optional extras that the preset functions can switch on
        self.midColor = None          # 3-colour gradient: start -> mid -> end (fire)
        self.fadeIn = 0.0             # fraction of the life used to fade in (smoke)
        self.swayAmount = 0.0         # sideways wobble in px/s (steam, hearts, confetti)
        self.swaySpeed = 3.0
        self.swayPhase = random.uniform(0, math.pi * 2)
        self.sizeEasing = None        # easing function for the size (e.g. easeOutQuad)
        self.stretchFactor = 0.0022   # how much a droplet stretches per px/s of speed
        self.updateLook(0.0)

    def updateLook(self, t):
        """Life cycle: size, colour and alpha interpolated with t = age / life."""
        if self.sizeEasing is not None:
            sizeT = self.sizeEasing(t)
        else:
            sizeT = t
        size = max(0.5, lerp(self.startSize, self.endSize, sizeT))

        if self.midColor is not None:
            if t < 0.5:
                rgb = lerpColor(self.startColor, self.midColor, t * 2)
            else:
                rgb = lerpColor(self.midColor, self.endColor, (t - 0.5) * 2)
        else:
            rgb = lerpColor(self.startColor, self.endColor, t)

        alpha = lerp(self.startAlpha, self.endAlpha, t)
        if self.fadeIn > 0 and t < self.fadeIn:
            alpha = alpha * (t / self.fadeIn)
        self.color = (rgb[0], rgb[1], rgb[2], int(clamp(alpha, 0, 255)))

        if self.stretch:
            speed = math.hypot(self.vx, self.vy)
            stretchAmount = 1.0 + min(speed * self.stretchFactor, 2.2)
            self.size = (size, size * self.aspect * stretchAmount)
            if speed > 1:
                # The droplet picture points UP, so its tail should point
                # against the velocity: rotate by -90 - direction (clockwise degrees)
                self.angle = -90 - math.degrees(math.atan2(self.vy, self.vx))
        else:
            self.size = (size, size * self.aspect)

    def updateParticle(self, dt):
        """Move and age the particle. Returns False when it should be removed."""
        self.age += dt
        if self.age >= self.life:
            return False
        # 1. forces change the velocity
        self.vy += self.gravity * dt
        if self.drag > 0:
            keep = max(0.0, 1.0 - self.drag * dt)
            self.vx *= keep
            self.vy *= keep
        # 2. velocity changes the position (plus an optional sine sway)
        swayVelocity = 0.0
        if self.swayAmount != 0:
            swayVelocity = math.sin(self.age * self.swaySpeed + self.swayPhase) * self.swayAmount
        self.center_x += (self.vx + swayVelocity) * dt
        self.center_y += self.vy * dt
        if self.spin != 0 and not self.stretch:
            self.angle += self.spin * dt
        # 3. life cycle look
        self.updateLook(self.age / self.life)
        return True


# ----------------------------------------------------------------------
# A group of particles drawn together
# ----------------------------------------------------------------------
class ParticleSystem:
    def __init__(self, additive=False, maxParticles=900):
        self.additive = additive
        self.maxParticles = maxParticles
        self.spriteList = arcade.SpriteList()

    def add(self, particle):
        # If the system is full, remove the oldest particle to make room
        if len(self.spriteList) >= self.maxParticles:
            self.spriteList[0].remove_from_sprite_lists()
        self.spriteList.append(particle)
        return particle

    def update(self, dt):
        deadParticles = []
        for particle in self.spriteList:
            if not particle.updateParticle(dt):
                deadParticles.append(particle)
        for particle in deadParticles:
            particle.remove_from_sprite_lists()

    def draw(self):
        if self.additive:
            self.spriteList.draw(blend_function=assets.additiveBlend())
        else:
            self.spriteList.draw()

    def clear(self):
        self.spriteList.clear()

    def count(self):
        return len(self.spriteList)


# ----------------------------------------------------------------------
# Emitter: calls spawnFunction(x, y) 'rate' times per second
# ----------------------------------------------------------------------
class Emitter:
    def __init__(self, x, y, rate, spawnFunction):
        self.x = x
        self.y = y
        self.rate = rate
        self.spawnFunction = spawnFunction
        self.active = True
        self.accumulator = 0.0

    def update(self, dt):
        if not self.active or self.rate <= 0:
            self.accumulator = 0.0
            return
        # Time accumulator: 30 per second at 60 FPS = one particle every 2nd frame
        self.accumulator += self.rate * dt
        spawned = 0
        while self.accumulator >= 1.0 and spawned < 60:
            self.accumulator -= 1.0
            self.spawnFunction(self.x, self.y)
            spawned += 1


# ----------------------------------------------------------------------
# Preset spawn helpers (system first, then position)
# ----------------------------------------------------------------------
def varyColor(rgb, amount=0.15):
    """A slightly lighter or darker copy of a colour so particles are not all identical."""
    k = random.uniform(1.0 - amount, 1.0 + amount)
    return (int(clamp(rgb[0] * k, 0, 255)), int(clamp(rgb[1] * k, 0, 255)), int(clamp(rgb[2] * k, 0, 255)))


def darker(rgb, factor=0.6):
    return (int(rgb[0] * factor), int(rgb[1] * factor), int(rgb[2] * factor))


def lighter(rgb, amount=0.5):
    return (int(rgb[0] + (255 - rgb[0]) * amount), int(rgb[1] + (255 - rgb[1]) * amount),
            int(rgb[2] + (255 - rgb[2]) * amount))


def spawnJuiceSplash(system, x, y, rgb, count=18, power=1.0):
    """Juice droplets burst up and out, then fall with gravity (projectile motion)."""
    droplet = particleTexture("droplet")
    blob = particleTexture("soft_circle")
    created = []
    for i in range(count):
        angle = math.radians(random.uniform(15, 165))
        speed = random.uniform(180, 520) * power
        vx = math.cos(angle) * speed
        vy = math.sin(angle) * speed
        colour = varyColor(rgb, 0.12)
        if i % 4 == 0:
            size = random.uniform(10, 18)
            particle = Particle(blob, x, y, vx * 0.6, vy * 0.6, random.uniform(0.4, 0.7), size, size * 0.5,
                                lighter(colour, 0.2), colour, 230, 0, gravity=-900, drag=1.0)
        else:
            size = random.uniform(6, 12)
            particle = Particle(droplet, x, y, vx, vy, random.uniform(0.6, 1.1), size, size * 0.7,
                                colour, darker(colour, 0.85), 255, 40, gravity=-1200, drag=0.3, stretch=True)
        created.append(system.add(particle))
    return created


def spawnSparkles(system, x, y, rgb=GOLD, count=20, speed=260):
    """Star-shaped sparks flying out in every direction (use an additive system)."""
    created = []
    textures = [particleTexture("spark"), particleTexture("star5")]
    for i in range(count):
        angle = random.uniform(0, math.pi * 2)
        velocity = speed * random.uniform(0.3, 1.0)
        size = random.uniform(16, 32)
        particle = Particle(random.choice(textures), x, y, math.cos(angle) * velocity,
                            math.sin(angle) * velocity, random.uniform(0.45, 0.9), size, 3,
                            (255, 255, 235), rgb, 255, 0, gravity=-90, drag=2.4,
                            spin=random.uniform(-360, 360), angle=random.uniform(0, 360))
        created.append(system.add(particle))
    return created


def spawnFire(system, x, y, intensity=1.0, width=120):
    """ONE flame particle: white-yellow -> orange -> red, rising and shrinking."""
    offset = random.uniform(-width / 2, width / 2)
    size = random.uniform(34, 58) * (0.6 + 0.4 * intensity)
    particle = Particle(particleTexture("flame"), x + offset, y, -offset * 0.4, random.uniform(90, 190) * intensity,
                        random.uniform(0.45, 0.85), size, size * 0.25, (255, 250, 215), (190, 40, 20),
                        230, 0, gravity=90, drag=0.6)
    particle.midColor = (255, 146, 40)
    particle.swayAmount = 18
    particle.swaySpeed = random.uniform(5, 9)
    return system.add(particle)


def spawnSmoke(system, x, y, dark=False, width=80):
    """A grey puff that grows, rises slowly and fades (drawn with normal blending)."""
    if dark:
        start = (70, 62, 60)
        end = (30, 28, 30)
        alpha = 190
    else:
        start = (190, 190, 200)
        end = (120, 120, 135)
        alpha = 120
    size = random.uniform(40, 70)
    particle = Particle(particleTexture("smoke_puff"), x + random.uniform(-width / 2, width / 2), y,
                        random.uniform(-20, 20), random.uniform(40, 95), random.uniform(1.6, 2.6),
                        size, size * random.uniform(2.2, 3.0), start, end, alpha, 0, gravity=12, drag=0.35,
                        spin=random.uniform(-40, 40), angle=random.uniform(0, 360))
    particle.fadeIn = 0.15
    return system.add(particle)


def spawnSteam(system, x, y, width=40):
    """A soft white wisp that rises and sways sideways (sine wave)."""
    size = random.uniform(18, 30)
    particle = Particle(particleTexture("soft_circle"), x + random.uniform(-width / 2, width / 2), y,
                        random.uniform(-8, 8), random.uniform(45, 80), random.uniform(1.2, 2.0),
                        size, size * 2.6, (255, 255, 255), (230, 235, 245), 110, 0, gravity=10, drag=0.2)
    particle.fadeIn = 0.2
    particle.swayAmount = random.uniform(15, 30)
    particle.swaySpeed = random.uniform(2.0, 3.5)
    return system.add(particle)


def spawnDust(system, x, y, count=6):
    """Small brown/grey puffs kicked up on the ground (footsteps, misses)."""
    created = []
    texture = particleTexture("smoke_puff")
    for i in range(count):
        direction = random.choice([-1, 1])
        colour = random.choice([(150, 128, 104), (130, 124, 120), (170, 150, 120)])
        size = random.uniform(12, 22)
        particle = Particle(texture, x + random.uniform(-10, 10), y + random.uniform(0, 8),
                            direction * random.uniform(40, 130), random.uniform(10, 60),
                            random.uniform(0.45, 0.85), size, size * 2.2, colour, darker(colour, 0.8),
                            170, 0, gravity=-40, drag=2.2, spin=random.uniform(-90, 90))
        created.append(system.add(particle))
    return created


def spawnHearts(system, x, y, count=6):
    """Hearts float up with a gentle sway (happy customers)."""
    created = []
    texture = particleTexture("heart")
    for i in range(count):
        colour = random.choice([(255, 90, 130), (240, 60, 90), (255, 140, 170)])
        size = random.uniform(20, 32)
        particle = Particle(texture, x + random.uniform(-30, 30), y + random.uniform(-10, 10),
                            random.uniform(-40, 40), random.uniform(70, 150), random.uniform(1.1, 1.8),
                            size * 0.6, size, colour, lighter(colour, 0.3), 255, 0, gravity=20, drag=0.5,
                            spin=random.uniform(-30, 30))
        particle.swayAmount = 30
        particle.swaySpeed = random.uniform(3, 5)
        created.append(system.add(particle))
    return created


def spawnConfetti(system, x, y, count=60, spread=600):
    """Colourful paper pieces that burst up, then flutter down while spinning."""
    created = []
    texture = particleTexture("confetti")
    for i in range(count):
        colour = hsvColor(random.random(), 0.75, 1.0)
        rgb = (colour[0], colour[1], colour[2])
        size = random.uniform(12, 20)
        particle = Particle(texture, x + random.uniform(-spread / 2, spread / 2), y + random.uniform(-20, 20),
                            random.uniform(-120, 120), random.uniform(-80, 320), random.uniform(2.5, 4.0),
                            size, size, rgb, darker(rgb, 0.8), 255, 0, gravity=-260, drag=1.3,
                            spin=random.uniform(-540, 540), angle=random.uniform(0, 360))
        particle.swayAmount = random.uniform(20, 50)
        particle.swaySpeed = random.uniform(3, 6)
        created.append(system.add(particle))
    return created


def fireworkVelocity(shape, i, count):
    """
    Direction pattern of one firework spark (returns vx, vy before scaling).
    circle: evenly spaced angles. ring: a tilted flat ring. heart: the
    parametric heart curve. star: points along a 5-point star outline.
    """
    t = i / count * math.pi * 2
    if shape == "heart":
        hx = 16 * math.sin(t) ** 3
        hy = 13 * math.cos(t) - 5 * math.cos(2 * t) - 2 * math.cos(3 * t) - math.cos(4 * t)
        return hx / 16.0, hy / 16.0
    if shape == "ring":
        return math.cos(t), math.sin(t) * 0.42
    if shape == "star":
        # Walk around the 10 corners of a star (outer, inner, outer, ...)
        corners = []
        for c in range(10):
            radius = 1.0 if c % 2 == 0 else 0.42
            angle = math.pi / 2 + c * math.pi / 5
            corners.append((math.cos(angle) * radius, math.sin(angle) * radius))
        position = i / count * 10
        index = int(position) % 10
        f = position - int(position)
        a = corners[index]
        b = corners[(index + 1) % 10]
        return lerp(a[0], b[0], f), lerp(a[1], b[1], f)
    return math.cos(t), math.sin(t)


def spawnFirework(system, x, y, rgb, count=90, shape="circle"):
    """
    A firework burst: bright streaks fly out in a shape, slow down (drag) and
    droop (gravity). Every third streak also drops a twinkling glitter spark.
    The streaks use stretch=True, so they point along their own velocity.
    """
    created = []
    streak = particleTexture("droplet")
    glitter = particleTexture("spark")
    speed = random.uniform(300, 380)
    for i in range(count):
        dirX, dirY = fireworkVelocity(shape, i, count)
        jitter = random.uniform(0.94, 1.06)
        size = random.uniform(13, 20)
        particle = Particle(streak, x, y, dirX * speed * jitter, dirY * speed * jitter,
                            random.uniform(1.4, 2.1), size, 4, lighter(rgb, 0.4), darker(rgb, 0.75),
                            255, 0, gravity=-120, drag=1.1, stretch=True)
        created.append(system.add(particle))
        if i % 3 == 0:
            sparkle = Particle(glitter, x, y, dirX * speed * 0.55 * jitter, dirY * speed * 0.55 * jitter,
                               random.uniform(1.6, 2.4), 12, 2, (255, 255, 235), rgb, 255, 0,
                               gravity=-60, drag=1.4, spin=random.uniform(-360, 360))
            created.append(system.add(sparkle))
    # A bright flash in the middle of the burst
    flash = Particle(particleTexture("light_glow"), x, y, 0, 0, 0.4, 160, 380, lighter(rgb, 0.5), rgb, 235, 0)
    created.append(system.add(flash))
    return created


def spawnDroplet(system, x, y, vx, vy, rgb, size=10, life=1.5):
    """ONE fluid droplet (teh tarik stream, sauce): stretched along its velocity, gravity -1400."""
    particle = Particle(particleTexture("droplet"), x, y, vx, vy, life, size, size * 0.9, rgb, rgb,
                        255, 200, gravity=-1400, drag=0.05, stretch=True)
    return system.add(particle)


def spawnRing(system, x, y, rgb, size=260):
    """An expanding shockwave ring that fades out."""
    particle = Particle(particleTexture("ring"), x, y, 0, 0, 0.5, 20, size, lighter(rgb, 0.4), rgb, 240, 0)
    particle.sizeEasing = easeOutQuad
    return system.add(particle)


def spawnEmbers(system, x, y, count=4):
    """Tiny orange sparks that drift upwards (fire, lanterns)."""
    created = []
    texture = particleTexture("hard_circle")
    for i in range(count):
        size = random.uniform(3, 7)
        particle = Particle(texture, x + random.uniform(-20, 20), y + random.uniform(-6, 6),
                            random.uniform(-40, 40), random.uniform(60, 170), random.uniform(0.8, 1.6),
                            size, 1.5, (255, 214, 110), (255, 70, 20), 255, 0, gravity=30, drag=0.4)
        particle.swayAmount = 25
        particle.swaySpeed = random.uniform(4, 8)
        created.append(system.add(particle))
    return created


def spawnBubble(system, x, y, rgb=(255, 245, 230)):
    """A small foam bubble that rises a little and pops (fades) at the end."""
    size = random.uniform(6, 12)
    particle = Particle(particleTexture("bubble"), x, y, random.uniform(-8, 8), random.uniform(18, 45),
                        random.uniform(0.6, 1.2), size, size * 1.25, rgb, rgb, 230, 0, gravity=0, drag=0.5)
    particle.swayAmount = 6
    return system.add(particle)


def spawnGlowBurst(system, x, y, rgb, size=300, life=0.4):
    """Extra helper: one big soft flash of light (dish reveal, explosions)."""
    particle = Particle(particleTexture("light_glow"), x, y, 0, 0, life, size * 0.4, size,
                        lighter(rgb, 0.5), rgb, 230, 0)
    return system.add(particle)
