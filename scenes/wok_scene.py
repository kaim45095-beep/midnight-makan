"""
wok_scene.py
------------
Scene 2 - "Wok Hei Fire" at Ah Seng's Wok Station.

Aiman cooks Char Kuey Teow. The player drags each ingredient into the
wok in the order on the recipe ticket, keeps the gas flame in the
"wok hei" zone with UP / DOWN, and tosses the wok with SPACE. A toss
when the wok is very hot throws a huge flare of fire (wok hei).

Techniques used (lecture links):
  * Week 3  - mouse press / drag / release (drag and drop), key hold
  * Week 5-6 - rotation of the wok around a pivot point
              (translate -> rotate -> translate back), projectile motion
  * Week 7  - interpolation: the wok temperature follows the gas knob
              smoothly, HSV colour interpolation for the heat gauge
  * Week 8  - particle systems (flames, smoke, steam, oil sparks),
              point light that flickers with the fire, screen flash/shake
  * Image processing - burnt food uses darkerTexture() (Pillow
              ImageEnhance: lower brightness and saturation at runtime)
"""
import math
import random

import arcade

from core import assets
from core.base_scene import BaseScene
from core.helpers import Tween, approach, clamp, distance, easeOutBack, hsvColor, lerp
from core.particles import (Emitter, ParticleSystem, spawnEmbers, spawnFire, spawnGlowBurst, spawnRing,
                            spawnSmoke, spawnSparkles, spawnSteam)
from core.ui import drawKeyHint, drawPanel, drawProgressBar, drawRoundedRect, drawText, withAlpha
from settings import (BAD_RED, CREAM, DISH_NAMES, FONT_BODY, FONT_NARROW, FONT_TITLE, GOLD, GOOD_GREEN,
                      LANTERN_ORANGE, SCENE_NAMES, STALL_NAMES, WARM_LIGHT)

# ----------------------------------------------------------------------
# Layout
# ----------------------------------------------------------------------
WOK_X = 640
WOK_BACK_Y = 255            # centre of wok_back.png
WOK_FRONT_Y = 214           # centre of wok_front.png (the front lip drawn over the food)
WOK_PIVOT_Y = 240           # the wok rotates around this point when it is tossed
BURNER_Y = 166
DROP_RADIUS = 170           # release a bowl this close to the wok to add it
BOWL_Y = 95

ZONE_LOW = 65               # the "wok hei" heat zone (percent)
ZONE_HIGH = 85
FLARE_HEAT = 80             # toss at or above this heat for a flare
BURN_HEAT = 90
BURN_TIME = 2.5             # seconds above BURN_HEAT before the food burns
TOSS_COOLDOWN = 0.6
KNOB_SPEED = 60             # percent per second while UP/DOWN is held

# Recipe order (what the ticket asks for)
RECIPE = [
    ("Garlic", "props/bowl_garlic.png", "food/bit_garlic.png", "Garlic fragrant!"),
    ("Prawns", "props/bowl_prawns.png", "food/bit_prawn.png", "Prawns turn pink!"),
    ("Flat Noodles", "props/bowl_noodles.png", "food/bit_noodle.png", "Noodles charred!"),
    ("Soy Sauce", "props/bottle_soy.png", "food/bit_soy.png", "Soy caramelised!"),
    ("Egg", "props/bowl_egg.png", "food/bit_egg.png", "Egg scrambled!"),
    ("Bean Sprouts", "props/bowl_sprouts.png", "food/bit_sprout.png", "Sprouts crunchy!"),
]
# Bowls are NOT in recipe order on the counter, so the player must read the ticket
BOWL_SLOTS_X = {"Prawns": 170, "Bean Sprouts": 305, "Garlic": 440,
                "Soy Sauce": 840, "Flat Noodles": 975, "Egg": 1110}

MAX_SCORE = 2300


def rotateAroundPivot(offsetX, offsetY, angleDegrees):
    """
    2D rotation matrix (Week 5-6) for Arcade's clockwise angles:
        x' =  x cos(t) + y sin(t)
        y' = -x sin(t) + y cos(t)
    The offset is measured from the pivot, so this is the classic
    translate -> rotate -> translate back composite transformation.
    """
    t = math.radians(angleDegrees)
    newX = offsetX * math.cos(t) + offsetY * math.sin(t)
    newY = -offsetX * math.sin(t) + offsetY * math.cos(t)
    return newX, newY


class Bowl:
    """An ingredient bowl on the counter that can be dragged into the wok."""

    def __init__(self, recipeIndex):
        name, bowlPath, bitPath, doneText = RECIPE[recipeIndex]
        self.recipeIndex = recipeIndex
        self.name = name
        self.texture = assets.loadTexture(bowlPath)
        self.glow = assets.glowTexture(bowlPath, 10, (255, 220, 120))
        self.homeX = BOWL_SLOTS_X[name]
        self.homeY = BOWL_Y
        if name == "Soy Sauce":
            self.homeY = BOWL_Y + 22          # the bottle is taller than a bowl
        self.x = self.homeX
        self.y = self.homeY
        self.used = False
        self.dragging = False
        self.returnTween = None
        self.returnStart = (0, 0)

    def contains(self, x, y):
        return abs(x - self.x) < self.texture.width / 2 and abs(y - self.y) < self.texture.height / 2

    def sendHome(self):
        self.dragging = False
        self.returnStart = (self.x, self.y)
        self.returnTween = Tween(0.0, 1.0, 0.35, easeOutBack)

    def update(self, dt):
        if self.returnTween is not None:
            self.returnTween.update(dt)
            t = self.returnTween.value
            self.x = lerp(self.returnStart[0], self.homeX, t)
            self.y = lerp(self.returnStart[1], self.homeY, t)
            if self.returnTween.done:
                self.returnTween = None

    def draw(self, highlight, time):
        if highlight and not self.used:
            pulse = 0.6 + 0.4 * math.sin(time * 6)
            assets.drawTexture(self.glow, self.x, self.y, self.glow.width, self.glow.height, alpha=255 * pulse)
        alpha = 255
        if self.used:
            alpha = 90                          # empty bowls fade out
        scale = 1.12 if self.dragging else 1.0
        assets.drawTexture(self.texture, self.x, self.y, self.texture.width * scale, self.texture.height * scale,
                           alpha=alpha)


class FoodBit:
    """A small piece of food inside the wok. Tossing throws it up (projectile motion)."""

    def __init__(self, texture, burntTexture, x, y):
        self.texture = texture
        self.burntTexture = burntTexture
        self.burnt = False
        self.restX = x
        self.restY = y
        self.x = x
        self.y = y + 90
        self.vx = 0.0
        self.vy = -200.0
        self.inAir = True
        self.angle = random.uniform(0, 360)
        self.spin = random.uniform(-300, 300)
        self.tint = (255, 255, 255)

    def toss(self, power):
        self.inAir = True
        self.vx = random.uniform(-110, 110)
        self.vy = random.uniform(380, 620) * power
        self.spin = random.uniform(-500, 500)
        # land somewhere new, so tossing really mixes the food
        self.restX = random.uniform(495, 785)
        self.restY = random.uniform(238, 272)

    def update(self, dt):
        if not self.inAir:
            return
        self.vy -= 1700 * dt                  # gravity
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.x = approach(self.x, self.restX, 1.5, dt)
        self.angle += self.spin * dt
        if self.vy < 0 and self.y <= self.restY:
            self.y = self.restY
            self.inAir = False

    def draw(self, wokOffsetY):
        texture = self.burntTexture if self.burnt else self.texture
        y = self.y
        if not self.inAir:
            y = self.y + wokOffsetY
        assets.drawTexture(texture, self.x, y, texture.width, texture.height, self.angle, rgb=self.tint)


class WokScene(BaseScene):
    sceneId = "wok"
    sceneTitle = "Scene 2 - " + SCENE_NAMES["wok"]
    stallName = STALL_NAMES["wok"]
    dishName = DISH_NAMES["wok"]
    musicName = "music_wok"
    auraMusicName = "music_wok_aura"
    ambienceName = "amb_kitchen"
    timeLimit = 90
    maxScore = MAX_SCORE
    isGameplayScene = True
    controlsHelp = [("DRAG", "Drag a bowl into the wok, in ticket order"),
                    ("UP / DOWN", "Turn the gas knob (mouse wheel works too)"),
                    ("SPACE", "Toss the wok - at 80% heat or more: WOK HEI!")]

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def setupScene(self):
        self.background = assets.loadTexture("backgrounds/wok_kitchen_bg.png")
        self.wokBack = assets.loadTexture("props/wok_back.png")
        self.wokFront = assets.loadTexture("props/wok_front.png")
        self.plateTexture = assets.loadTexture("food/plate_ckt.png")

        self.bowls = []
        for i in range(len(RECIPE)):
            self.bowls.append(Bowl(i))
        self.draggedBowl = None

        # Cooking state
        self.stepIndex = 0
        self.stepState = "waiting"        # waiting (for an ingredient) -> cooking -> next step
        self.stepProgress = 0.0
        self.bits = []
        self.knob = 30.0                  # gas knob, 0..100
        self.heat = 18.0                  # wok temperature, follows the knob with a lag
        self.burnTimer = 0.0
        self.tossTimer = 0.0
        self.flareCooldown = 0.0
        self.hintTimer = 0.0
        self.heatHistory = []             # for the F2 heat graph
        self.historyTimer = 0.0

        # Counters for the result card
        self.flareCount = 0
        self.tossCount = 0
        self.burnCount = 0
        self.wrongCount = 0

        # Wok tilt and bounce (tweened)
        self.wokAngle = 0.0
        self.wokLift = 0.0
        self.tiltTween = None

        # Plating (food transformation at the end)
        self.plating = False
        self.platingTime = 0.0
        self.plateShown = False

        # Flames live in their own additive particle system that is drawn BEHIND the wok,
        # so they lick up around the wok's curved sides (painter's algorithm)
        self.flameParticles = ParticleSystem(additive=True, maxParticles=400)
        self.flameEmitter = Emitter(WOK_X, BURNER_Y, 30, self.spawnBurnerFlame)
        self.steamEmitter = Emitter(WOK_X, 300, 4, self.spawnWokSteam)

        # Lighting (Week 8): the fire is a point light that glows on the tiled wall BEHIND the
        # wok (drawn in drawWorld), so the dark steel wok itself is not washed out
        self.fireGlow = 0.3
        self.flareBoost = 0.0
        self.glowTexture = assets.loadTexture("particles/light_glow.png")
        self.hoodLight = self.lights.add(WOK_X, 578, 300, WARM_LIGHT, 0.25, 0.03)

        self.sizzleLoop = None
        self.fireLoop = None

        self.gaugeText = arcade.Text("", 1216, 488, CREAM, 16, font_name=FONT_TITLE,
                                     anchor_x="center", anchor_y="center")
        self.debugText = arcade.Text("", 640, 548, (170, 255, 190), 13, font_name=FONT_BODY,
                                     anchor_x="center", anchor_y="center")

        # Demo bot state
        self.botDrag = None
        self.botTossTimer = 1.5
        self.botWrongDone = False
        self.botBurnDone = False
        self.botHeldKey = None

    def getIntroLines(self):
        return [
            ("nenek", "Second course: char kuey teow! Ah Seng's wok is the hottest one on Pelita Street."),
            ("nenek", "Drag each bowl into the wok in the order on the ticket. UP and DOWN turn the gas."),
            ("aiman", "Keep the heat in the green zone and toss with SPACE. Toss when it is really hot and... WOK HEI!"),
            ("nenek", "But above 90 percent for too long, everything burns. Then Ah Seng will cry."),
        ]

    def onPlayStart(self):
        self.sizzleLoop = self.startLoop("sizzle_loop", 0.0)
        self.fireLoop = self.startLoop("fire_loop", 0.2)

    # ------------------------------------------------------------------
    # Emitter spawn functions
    # ------------------------------------------------------------------
    def spawnBurnerFlame(self, x, y):
        heatFraction = self.knob / 100
        width = 220 + 200 * heatFraction
        spawnFire(self.flameParticles, x, y, 0.55 + 0.9 * heatFraction, width)

    def spawnWokSteam(self, x, y):
        if len(self.bits) == 0:
            return
        if self.burnTimer > 0.9:
            spawnSmoke(self.particles, x + random.uniform(-90, 90), y, True, 60)
        else:
            spawnSteam(self.particles, x + random.uniform(-110, 110), y, 50)

    # ------------------------------------------------------------------
    # Ingredients and cooking steps
    # ------------------------------------------------------------------
    def currentRecipeName(self):
        if self.stepIndex < len(RECIPE):
            return RECIPE[self.stepIndex][0]
        return ""

    def dropBowl(self, bowl, x, y):
        """Called when a dragged bowl is released."""
        nearWok = distance(x, y, WOK_X, WOK_PIVOT_Y + 40) < DROP_RADIUS
        if not nearWok:
            bowl.sendHome()
            return
        if bowl.recipeIndex != self.stepIndex:
            # Wrong order: the ingredient goes back and the combo breaks
            self.wrongCount += 1
            self.playSfx("buzzer_wrong", 0.8, x=x)
            self.registerHit("miss", WOK_X, 330, "Not yet!")
            self.addScore(-20)
            self.popups.add(f"Add the {self.currentRecipeName()} first", WOK_X, 440, BAD_RED, 20, 1.6, 30)
            bowl.sendHome()
            return
        if self.stepState == "cooking":
            self.popups.add(f"Let the {self.currentRecipeName()} cook first!", WOK_X, 440, CREAM, 20, 1.4, 30)
            bowl.sendHome()
            return
        self.addIngredient(bowl)

    def addIngredient(self, bowl):
        bowl.used = True
        bowl.sendHome()
        name, bowlPath, bitPath, doneText = RECIPE[bowl.recipeIndex]
        bitTexture = assets.loadTexture(bitPath)
        # Image processing at runtime: a darker, less saturated copy for burnt food
        burntTexture = assets.darkerTexture(bitPath)
        count = 9
        if name == "Flat Noodles":
            count = 14
        for i in range(count):
            bit = FoodBit(bitTexture, burntTexture, random.uniform(495, 785), random.uniform(238, 272))
            bit.x = WOK_X + random.uniform(-60, 60)
            bit.y = 380 + random.uniform(0, 60)
            self.bits.append(bit)
        if name == "Soy Sauce":
            # soy sauce browns the noodles: tint their colour (RGB multiply)
            for bit in self.bits:
                bit.tint = (205, 170, 140)
        self.stepState = "cooking"
        self.stepProgress = 0.0
        self.playSfx("sizzle_burst", 0.9, x=WOK_X, pitchVariation=0.08)
        for i in range(8):
            spawnSteam(self.particles, WOK_X + random.uniform(-100, 100), 290, 60)
        spawnEmbers(self.glowParticles, WOK_X, 270, 10)
        self.addScore(20, WOK_X, 470, f"+ {name}", CREAM)

    def finishStep(self):
        name, bowlPath, bitPath, doneText = RECIPE[self.stepIndex]
        self.registerHit("good", WOK_X, 390, doneText)
        self.stepIndex += 1
        self.burnTimer = 0.0
        if self.stepIndex >= len(RECIPE):
            self.startPlating()
        else:
            self.stepState = "waiting"
            self.stepProgress = 0.0

    def burnFood(self):
        """Too hot for too long: the food burns (smoke, alarm, darker textures)."""
        self.burnCount += 1
        self.burnTimer = 0.0
        self.stepProgress = max(0.0, self.stepProgress - 0.3)
        for bit in self.bits:
            bit.burnt = True
        self.registerHit("miss", WOK_X, 330, "BURNT!")
        self.addScore(-100, WOK_X, 420)
        self.playSfx("smoke_alarm", 0.8)
        for i in range(14):
            spawnSmoke(self.particles, WOK_X + random.uniform(-120, 120), 300, True, 80)
        self.popups.add("Turn the heat DOWN!", WOK_X, 470, BAD_RED, 22, 1.6, 20)

    # ------------------------------------------------------------------
    # Tossing
    # ------------------------------------------------------------------
    def tossWok(self):
        if self.tossTimer > 0 or len(self.bits) == 0 or self.plating:
            return
        self.tossTimer = TOSS_COOLDOWN
        self.tossCount += 1
        # the wok tilts back and bounces up (keyframes 0 -> -12 degrees -> 0)
        self.tiltTween = Tween(0.0, 1.0, 0.4)
        power = 0.8 + 0.4 * (self.heat / 100)
        for bit in self.bits:
            bit.toss(power)
        self.playSfx("wok_clang", 0.8, x=WOK_X, pitchVariation=0.1)
        if self.stepState == "cooking":
            self.stepProgress += 0.1
            if ZONE_LOW <= self.heat <= ZONE_HIGH + 10:
                self.addScore(15)
        if self.heat >= FLARE_HEAT and self.stepState == "cooking" and self.flareCooldown <= 0:
            self.wokHeiFlare()
        elif self.heat >= 60:
            spawnEmbers(self.glowParticles, WOK_X, 290, 6)

    def wokHeiFlare(self):
        """The signature effect: oil vapour catches fire above the wok."""
        self.flareCount += 1
        self.flareCooldown = 1.6
        # the flames rise from the rim and above it, so the food stays visible
        for i in range(70):
            spawnFire(self.glowParticles, WOK_X + random.uniform(-160, 160), 320, 1.9, 80)
        spawnGlowBurst(self.glowParticles, WOK_X, 400, (255, 160, 60), 460, 0.4)
        spawnEmbers(self.glowParticles, WOK_X, 340, 26)
        self.flareBoost = 0.6
        self.flash.start(LANTERN_ORANGE, 0.22, 70)
        self.shake.start(10, 0.35)
        self.playSfx("flare_whoosh", 1.0, x=WOK_X)
        self.playSfx("vo_wokhei", 1.0)
        self.registerHit("perfect", WOK_X, 440, "WOK HEI!")
        self.addScore(50)

    def startPlating(self):
        self.plating = True
        self.platingTime = 0.0
        self.stepState = "done"
        self.playSfx("plate_slide", 0.8, x=960)
        self.knob = 10

    # ------------------------------------------------------------------
    # Input
    # ------------------------------------------------------------------
    def onPlayKeyPress(self, key, modifiers):
        if key == arcade.key.SPACE:
            self.tossWok()

    def onPlayMousePress(self, x, y, button):
        if button != arcade.MOUSE_BUTTON_LEFT or self.plating:
            return
        # Week 3 drag & drop: pick up the bowl under the mouse
        for bowl in self.bowls:
            if not bowl.used and bowl.returnTween is None and bowl.contains(x, y):
                bowl.dragging = True
                self.draggedBowl = bowl
                self.playSfx("ui_click", 0.5, x=x)
                return

    def onPlayMouseDrag(self, x, y, dx, dy):
        if self.draggedBowl is not None:
            self.draggedBowl.x = x
            self.draggedBowl.y = y

    def onPlayMouseRelease(self, x, y, button):
        if self.draggedBowl is not None:
            bowl = self.draggedBowl
            self.draggedBowl = None
            self.dropBowl(bowl, x, y)

    def onPlayMouseScroll(self, x, y, scrollY):
        self.changeKnob(scrollY * 8)

    def changeKnob(self, amount):
        old = self.knob
        self.knob = clamp(self.knob + amount, 0, 100)
        if int(old / 10) != int(self.knob / 10):
            self.playSfx("knob_click", 0.4, cooldown=0.06)

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------
    def updateAlways(self, dt):
        self.flameEmitter.update(dt)
        self.flameParticles.update(dt)
        self.steamEmitter.update(dt)
        for bowl in self.bowls:
            bowl.update(dt)

    def updatePlay(self, dt):
        # Gas knob with the keyboard (held keys)
        if self.isKeyHeld(arcade.key.UP) or self.isKeyHeld(arcade.key.W):
            self.changeKnob(KNOB_SPEED * dt)
        if self.isKeyHeld(arcade.key.DOWN) or self.isKeyHeld(arcade.key.S):
            self.changeKnob(-KNOB_SPEED * dt)

        # The wok temperature follows the knob with a lag (exponential smoothing)
        self.heat = approach(self.heat, self.knob, 0.9, dt)
        self.flameEmitter.rate = 18 + 70 * self.knob / 100
        self.steamEmitter.rate = 2 + 10 * self.heat / 100

        # Light: brighter with more heat, flickering like a real flame, and a boost after a flare
        flicker = 0.88 + 0.08 * math.sin(self.sceneTime * 23) * math.sin(self.sceneTime * 7.3)
        flicker += random.uniform(-0.04, 0.04)
        self.fireGlow = (0.2 + 0.8 * self.heat / 100) * flicker
        self.flareBoost = approach(self.flareBoost, 0.0, 2.5, dt)

        # Sounds follow the fire: sizzle volume = heat (only with food), roar = knob
        sound = self.window.sound
        sizzle = 0.05
        if len(self.bits) > 0:
            sizzle = 0.2 + 0.65 * self.heat / 100
        sound.setVolume(self.sizzleLoop, sizzle)
        sound.setVolume(self.fireLoop, 0.12 + 0.45 * self.knob / 100)

        self.tossTimer = max(0.0, self.tossTimer - dt)
        self.flareCooldown = max(0.0, self.flareCooldown - dt)
        self.hintTimer = max(0.0, self.hintTimer - dt)
        self.updateWokTilt(dt)
        for bit in self.bits:
            bit.update(dt)

        if self.stepState == "cooking":
            self.updateCooking(dt)
        if self.plating:
            self.updatePlating(dt)

        # heat history for the debug graph (one sample every 0.1 s)
        self.historyTimer += dt
        if self.historyTimer >= 0.1:
            self.historyTimer = 0.0
            self.heatHistory.append(self.heat)
            if len(self.heatHistory) > 80:
                self.heatHistory.pop(0)

    def updateCooking(self, dt):
        # Cooking speed depends on the heat: best inside the wok hei zone
        if self.heat >= ZONE_LOW:
            rate = 0.2
        else:
            rate = 0.2 * (self.heat / ZONE_LOW) * 0.4
            if self.heat < 45 and self.hintTimer <= 0:
                self.hintTimer = 4.0
                self.popups.add("Turn up the heat!", 1080, 520, CREAM, 18, 1.4, 20)
        self.stepProgress += rate * dt
        # Burning
        if self.heat > BURN_HEAT:
            self.burnTimer += dt
            if self.burnTimer > BURN_TIME:
                self.burnFood()
        else:
            self.burnTimer = max(0.0, self.burnTimer - dt * 2)
        if self.stepProgress >= 1.0:
            self.finishStep()

    def updateWokTilt(self, dt):
        if self.tiltTween is None:
            self.wokAngle = 0.0
            self.wokLift = 0.0
            return
        self.tiltTween.update(dt)
        t = self.tiltTween.value
        # a quick keyframe curve: up to -12 degrees and 10 px lift, then back (sin shape)
        shape = math.sin(t * math.pi)
        self.wokAngle = -12 * shape
        self.wokLift = 10 * shape
        if self.tiltTween.done:
            self.tiltTween = None

    def updatePlating(self, dt):
        self.platingTime += dt
        plateX = 960
        plateY = 250
        if self.platingTime > 0.4 and len(self.bits) > 0 and self.platingTime < 1.6:
            # the food jumps from the wok to the plate a few pieces at a time
            for i in range(2):
                if len(self.bits) == 0:
                    break
                bit = self.bits.pop(0)
                spawnSparkles(self.glowParticles, bit.x, bit.y, (255, 220, 150), 3, 160)
            self.wokAngle = 18
        if self.platingTime > 1.2 and not self.plateShown:
            # Food transformation: the loose pieces become the finished dish
            self.plateShown = True
            self.bits = []
            spawnGlowBurst(self.glowParticles, plateX, plateY, WARM_LIGHT, 420, 0.6)
            spawnSparkles(self.glowParticles, plateX, plateY, GOLD, 34, 320)
            spawnRing(self.glowParticles, plateX, plateY, GOLD, 360)
            self.playSfx("happy_chime", 0.8, x=plateX)
        if self.platingTime > 2.3:
            self.plating = False
            bonus = int(self.timeLeft) * 3
            if bonus > 0:
                self.score += bonus
                self.popups.add(f"TIME BONUS +{bonus}", 640, 470, (120, 190, 255), 28, 1.5, 50)
            self.finishScene("Smoky, charred, full of wok hei!")

    # ------------------------------------------------------------------
    # Demo bot
    # ------------------------------------------------------------------
    def holdBotKey(self, key):
        """Hold one arrow key (or none) like a player turning the knob."""
        if self.botHeldKey == key:
            return
        if self.botHeldKey is not None:
            self.botReleaseKey(self.botHeldKey)
        self.botHeldKey = key
        if key is not None:
            self.botPressKey(key)

    def autoPlay(self, dt):
        if self.dialogue.isActive() or self.plating:
            self.holdBotKey(None)
            return

        # 1) Heat: normally 74%, higher for flares, and one burn on purpose (step 3)
        targetKnob = 74
        if self.burnCount > 0:
            self.botBurnDone = True
        burnDemo = self.stepIndex == 2 and not self.botBurnDone
        if burnDemo:
            targetKnob = 100          # a deliberate mistake: far too hot
        elif self.stepState == "cooking" and self.stepIndex in (1, 3, 4) and 0.15 < self.stepProgress < 0.75:
            targetKnob = 87           # hot enough for a flare, below the burn line
        if self.knob < targetKnob - 3:
            self.holdBotKey(arcade.key.UP)
        elif self.knob > targetKnob + 3:
            self.holdBotKey(arcade.key.DOWN)
        else:
            self.holdBotKey(None)

        # 2) Ingredients: drag the right bowl (one wrong bowl early, to show the feedback)
        if self.botDrag is None and self.stepState == "waiting":
            wanted = self.stepIndex
            if self.stepIndex == 1 and not self.botWrongDone:
                wanted = 4
                self.botWrongDone = True
            self.botDrag = {"bowl": self.bowls[wanted], "stage": "toBowl"}
        if self.botDrag is not None:
            bowl = self.botDrag["bowl"]
            if self.botDrag["stage"] == "toBowl":
                if self.botMoveCursor(bowl.x, bowl.y, dt, 1300):
                    self.botMouseDown()
                    self.botDrag["stage"] = "toWok"
            elif self.botDrag["stage"] == "toWok":
                if self.botMoveCursor(WOK_X + random.uniform(-2, 2), 300, dt, 800):
                    self.botMouseUp()
                    self.botDrag = None
            return

        # 3) Toss every 1.2-1.9 seconds while cooking (not during the burn demo)
        if self.stepState == "cooking" and not burnDemo:
            self.botTossTimer -= dt
            if self.botTossTimer <= 0:
                self.botTossTimer = random.uniform(1.2, 1.9)
                self.botTapKey(arcade.key.SPACE)

    def cleanupScene(self):
        self.flameParticles.clear()

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------
    def getResultLines(self):
        return [
            f"Wok hei flares: {self.flareCount}    Tosses: {self.tossCount}",
            f"Burnt: {self.burnCount}    Wrong order: {self.wrongCount}",
        ]

    def getDishTexture(self):
        return self.plateTexture

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------
    def drawWorld(self):
        assets.drawTexture(self.background, 640, 360, 1280, 720)
        # the fire lights up the wall behind the wok (additive glow = point light)
        glow = min(1.6, self.fireGlow + self.flareBoost)
        assets.drawTexture(self.glowTexture, WOK_X, 330, 980 + 200 * self.flareBoost, 620, rgb=LANTERN_ORANGE,
                           alpha=min(255, 150 * glow), additive=True)
        # flames next: they sit behind the wok and lick up around its sides
        self.flameParticles.draw()
        self.drawWok()
        for bowl in self.bowls:
            if bowl is not self.draggedBowl:
                bowl.draw(bowl.recipeIndex == self.stepIndex and self.stepState == "waiting", self.sceneTime)
        if self.plateShown:
            bob = math.sin(self.sceneTime * 2) * 3
            assets.drawTexture(self.plateTexture, 960, 250 + bob, 300, None)

    def drawWok(self):
        # Both wok layers rotate around the same pivot, so they stay lined up
        backX, backY = rotateAroundPivot(0, WOK_BACK_Y - WOK_PIVOT_Y, self.wokAngle)
        frontX, frontY = rotateAroundPivot(0, WOK_FRONT_Y - WOK_PIVOT_Y, self.wokAngle)
        pivotY = WOK_PIVOT_Y + self.wokLift
        assets.drawTexture(self.wokBack, WOK_X + backX, pivotY + backY, 460, 210, self.wokAngle)
        # heat tint: the wok glows faintly red when it is very hot
        if self.heat > 70:
            glow = (self.heat - 70) / 30
            assets.drawTexture(assets.loadTexture("particles/light_glow.png"), WOK_X, 200, 420, 120,
                               rgb=(255, 90, 30), alpha=90 * glow, additive=True)
        for bit in self.bits:
            bit.draw(self.wokLift)
        assets.drawTexture(self.wokFront, WOK_X + frontX, pivotY + frontY, 460, 110, self.wokAngle)

    def drawWorldTop(self):
        # the dragged bowl is drawn last so it floats above everything
        if self.draggedBowl is not None:
            self.draggedBowl.draw(False, self.sceneTime)
            near = distance(self.draggedBowl.x, self.draggedBowl.y, WOK_X, WOK_PIVOT_Y + 40) < DROP_RADIUS
            if near:
                arcade.draw_circle_outline(WOK_X, WOK_PIVOT_Y + 40, DROP_RADIUS, withAlpha(GOLD, 120), 3)

    def drawPlayHud(self):
        self.drawRecipeTicket()
        self.drawHeatGauge()
        # cooking progress bar above the wok
        if self.stepState == "cooking":
            name = self.currentRecipeName()
            drawPanel(640, 602, 300, 50, (14, 16, 38, 200), (255, 200, 87, 120), 14, shadow=False)
            drawText(f"COOKING {name.upper()}", 640, 615, CREAM, 13, FONT_BODY, anchorX="center",
                     anchorY="center", bold=True)
            drawProgressBar(510, 588, 260, 12, clamp(self.stepProgress, 0.0, 1.0), GOOD_GREEN)
        elif self.stepState == "waiting" and self.stepIndex < len(RECIPE):
            name = self.currentRecipeName()
            drawPanel(640, 602, 380, 40, (14, 16, 38, 200), (255, 200, 87, 120), 14, shadow=False)
            drawText(f"Drag the {name.upper()} into the wok", 640, 602, GOLD, 15, FONT_BODY,
                     anchorX="center", anchorY="center", bold=True)
        drawKeyHint(470, 28, "UP/DOWN", "Gas")
        drawKeyHint(660, 28, "SPACE", "Toss")
        drawKeyHint(830, 28, "DRAG", "Add food")

    def drawRecipeTicket(self):
        # a paper order ticket pinned to the wall
        left = 26
        top = 590
        width = 250
        height = 58 + len(RECIPE) * 34
        drawRoundedRect(left + width / 2 + 4, top - height / 2 - 4, width, height, (0, 0, 0, 90), 10)
        drawRoundedRect(left + width / 2, top - height / 2, width, height, (250, 244, 226, 245), 10)
        arcade.draw_circle_filled(left + width / 2, top - 8, 7, BAD_RED)
        drawText("ORDER: CHAR KUEY TEOW", left + width / 2, top - 30, (90, 40, 30), 13, FONT_BODY,
                 anchorX="center", anchorY="center", bold=True, shadow=False)
        for i in range(len(RECIPE)):
            y = top - 64 - i * 34
            name = RECIPE[i][0]
            if i < self.stepIndex:
                colour = (70, 140, 70)
                assets.drawTexture(assets.loadTexture("ui/icon_check.png"), left + 24, y, 24, 24)
            elif i == self.stepIndex:
                colour = (180, 60, 20)
                arcade.draw_triangle_filled(left + 16, y + 7, left + 16, y - 7, left + 28, y, colour)
            else:
                colour = (110, 100, 95)
            drawText(f"{i + 1}. {name}", left + 44, y, colour, 16, FONT_BODY, anchorY="center",
                     bold=(i == self.stepIndex), shadow=False)
            if i < self.stepIndex:
                arcade.draw_line(left + 44, y, left + 44 + 130, y, (70, 140, 70, 180), 2)

    def drawHeatGauge(self):
        cx = 1216
        bottom = 180
        height = 270
        drawPanel(cx, bottom + height / 2 + 20, 84, height + 110, (14, 16, 38, 215), (255, 200, 87, 150), 16,
                  shadow=False)
        drawText("HEAT", cx, bottom + height + 52, GOLD, 14, FONT_TITLE, anchorX="center", anchorY="center")
        left = cx - 14
        arcade.draw_lbwh_rectangle_filled(left, bottom, 28, height, (0, 0, 0, 160))
        # the wok hei zone band
        zoneBottom = bottom + height * ZONE_LOW / 100
        zoneTop = bottom + height * ZONE_HIGH / 100
        arcade.draw_lbwh_rectangle_filled(left - 6, zoneBottom, 40, zoneTop - zoneBottom, (110, 220, 130, 90))
        burnY = bottom + height * BURN_HEAT / 100
        arcade.draw_lbwh_rectangle_filled(left - 6, burnY, 40, bottom + height - burnY, (240, 84, 84, 70))
        # temperature fill: HSV hue slides from blue (cold) to red (hot) - colour interpolation
        fraction = self.heat / 100
        colour = hsvColor(lerp(0.6, 0.0, fraction), 0.85, 1.0)
        arcade.draw_lbwh_rectangle_filled(left + 3, bottom + 3, 22, max(0, (height - 6) * fraction), colour)
        # knob marker (where the gas is set)
        knobY = bottom + height * self.knob / 100
        arcade.draw_triangle_filled(left - 12, knobY + 8, left - 12, knobY - 8, left - 2, knobY, CREAM)
        self.gaugeText.text = f"{int(self.heat)}%"
        self.gaugeText.draw()
        if self.burnTimer > 0.3:
            blink = 0.5 + 0.5 * math.sin(self.sceneTime * 14)
            drawText("TOO HOT!", cx, bottom - 22, BAD_RED, 13, FONT_BODY, anchorX="center", anchorY="center",
                     bold=True, alpha=150 + 105 * blink)
        else:
            drawText("WOK HEI", cx, bottom - 22, GOOD_GREEN, 12, FONT_NARROW, anchorX="center",
                     anchorY="center", alpha=200)

    def drawDebugWorld(self):
        """F2: pivot point, drop radius, food area and a heat graph."""
        arcade.draw_circle_outline(WOK_X, WOK_PIVOT_Y + 40, DROP_RADIUS, (80, 255, 120), 1)
        arcade.draw_circle_filled(WOK_X, WOK_PIVOT_Y + self.wokLift, 5, (255, 120, 255))
        arcade.draw_lbwh_rectangle_outline(495, 238, 290, 34, (120, 180, 255), 1)
        # heat graph: last 8 seconds, with the zone lines
        left = 330
        bottom = 470
        width = 320
        height = 90
        arcade.draw_lbwh_rectangle_filled(left, bottom, width, height, (0, 0, 0, 170))
        for level in (ZONE_LOW, ZONE_HIGH, BURN_HEAT):
            y = bottom + height * level / 100
            arcade.draw_line(left, y, left + width, y, (110, 220, 130, 120), 1)
        points = []
        for i in range(len(self.heatHistory)):
            points.append((left + i * width / 80, bottom + height * self.heatHistory[i] / 100))
        if len(points) > 1:
            arcade.draw_line_strip(points, (255, 160, 60), 2)
        self.debugText.text = (f"heat {self.heat:5.1f}  knob {self.knob:5.1f}  step {self.stepIndex} "
                               f"progress {self.stepProgress:4.2f}  burn {self.burnTimer:3.1f}/{BURN_TIME}  "
                               f"flames {self.flameParticles.count()}")
        # a dark strip behind the text keeps it readable on busy backgrounds
        width = self.debugText.content_width
        arcade.draw_lbwh_rectangle_filled(self.debugText.x - width / 2 - 8, self.debugText.y - 12, width + 16, 24,
                                          (0, 0, 0, 180))
        self.debugText.draw()
