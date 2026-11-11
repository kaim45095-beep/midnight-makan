"""
finale_scene.py
---------------
Finale - "Lantern Bloom" on the rooftop at midnight.

  1. LANTERNS  - click the eight sky lanterns (or press SPACE) to light
                 them; each one plays a note of a pentatonic scale and rises
  2. FIREWORKS - a short fireworks show (circle, heart, star, ring bursts)
  3. VERDICT   - Dato' Leon gives his verdict; the ending depends on the
                 total stars (golden / silver / bronze)
  4. PHOTO     - Farah takes a group photo. The screen is captured and
                 "developed" into a polaroid with Pillow image processing,
                 then saved in the memories folder
  5. CREDITS   - the credits scroll up, then back to the title screen

Techniques used (lecture links):
  * Week 1-2 - colour channels: the photo is colour-graded by changing the
              R, G and B channels separately (a warm "film" look)
  * Week 3-4 - image enhancement with Pillow (contrast, saturation), a
              vignette mask and film grain (a digital artefact added on purpose)
  * Week 5-6 - parametric curves: the heart-shaped firework uses the heart
              equation; the credits scroll is a translation
  * Week 8  - particle systems (fireworks, embers, confetti), point lights
  * Audio   - lantern notes use playback speed as pitch: speed = 2 ** (n / 12)
"""
import math
import os
import random

import arcade
from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter, ImageFont

from core import assets
from core.base_scene import BaseScene
from core.helpers import Tween, approach, easeOutBack
from core.particles import spawnConfetti, spawnEmbers, spawnFirework, spawnGlowBurst, spawnSparkles
from core.ui import drawKeyHint, drawPanel, drawText
from settings import (CREAM, FONT_BODY, FONT_DIR, FONT_TITLE, GOLD, GOOD_GREEN, LANTERN_ORANGE, LAVENDER,
                      MEMORY_DIR, PINK, SKY_BLUE, TEAL, WHITE)

LANTERN_Y = 215
PENTATONIC = [0, 2, 4, 7, 9, 12, 14, 16]          # semitones of a major pentatonic scale
FIREWORK_COLOURS = [GOLD, PINK, SKY_BLUE, GOOD_GREEN, LANTERN_ORANGE, LAVENDER, TEAL]
FIREWORK_SHAPES = ["circle", "heart", "star", "ring"]
FIREWORK_TIME = 11.0

VERDICTS = {
    "golden": [
        ("leon", "Rojak with real crunch. Char kuey teow with true wok hei. Teh tarik pulled to the sky."),
        ("leon", "Magnificent. Grandma Siti's stall wins the last spot in the Festival Food Hall!"),
        ("nenek", "Aiman... look at the ladle. It is glowing gold!"),
        ("aiman", "We did it, Grandma! Pelita Street did it together!"),
    ],
    "silver": [
        ("leon", "Good. Very good. A few cuts were rushed, and the fire got a little wild."),
        ("leon", "Bring your stall to the Food Hall next month as a guest. Come back stronger."),
        ("nenek", "A guest stall! That is a real start, Aiman."),
    ],
    "bronze": [
        ("leon", "Not yet... but I can taste the heart in every bowl."),
        ("nenek", "Then we cook again tomorrow, Aiman. Together, like always."),
    ],
}
VERDICT_VOICE = {"golden": "vo_leon_golden", "silver": "vo_leon_silver", "bronze": "vo_leon_bronze"}

CREDIT_LINES = [
    ("MIDNIGHT MAKAN", GOLD, 30),
    ("A Night-Market Dining Adventure", CREAM, 17),
    ("", CREAM, 10),
    ("Created by", LAVENDER, 15),
    ("Mohammad Kaymul Islam (TP088446)", CREAM, 20),
    ("APU2F2602CS(AI)", CREAM, 15),
    ("", CREAM, 10),
    ("CT029-3-2 Imaging and Special Effects", LAVENDER, 15),
    ("Asia Pacific University of Technology and Innovation", CREAM, 15),
    ("", CREAM, 10),
    ("Engine", LAVENDER, 15),
    ("Python Arcade 3.3  -  Pillow  -  NumPy  -  OpenCV", CREAM, 16),
    ("", CREAM, 10),
    ("Art, music and sound effects", LAVENDER, 15),
    ("generated procedurally by the Python scripts in tools/", CREAM, 16),
    ("Developed with declared GenAI assistance (APU Yellow / restricted use)", CREAM, 14),
    ("", CREAM, 10),
    ("Fonts", LAVENDER, 15),
    ("Kenney Future (CC0)  -  Liberation Sans (SIL OFL)", CREAM, 16),
    ("", CREAM, 10),
    ("Thank you for playing!", GOLD, 24),
]


class SkyLantern:
    def __init__(self, x, index):
        self.x = x
        self.baseX = x
        self.y = LANTERN_Y
        self.index = index
        self.state = "waiting"          # waiting -> lit -> rising -> gone
        self.time = 0.0
        self.light = None
        self.phase = random.uniform(0, 6.28)

    def contains(self, x, y):
        return self.state == "waiting" and abs(x - self.x) < 55 and abs(y - self.y) < 65


class Rocket:
    """A firework rocket: rises with a spark trail, then bursts at the top."""

    def __init__(self, x, apexY, colour, shape):
        self.x = x
        self.y = 150.0
        self.apexY = apexY
        self.colour = colour
        self.shape = shape
        self.speed = random.uniform(560, 680)
        self.done = False


class FinaleScene(BaseScene):
    sceneId = "finale"
    sceneTitle = "Lantern Bloom"
    musicName = "music_finale"
    ambienceName = "amb_rooftop"
    isGameplayScene = False
    controlsHelp = [("CLICK", "Light a sky lantern"), ("SPACE", "Light the next lantern"),
                    ("ENTER", "Skip the credits")]

    def setupScene(self):
        self.background = assets.loadTexture("backgrounds/rooftop_bg.png")
        self.lanternOff = assets.loadTexture("props/sky_lantern.png")
        self.lanternOn = assets.loadTexture("props/sky_lantern_lit.png")
        self.glow = assets.loadTexture("particles/light_glow.png")
        self.ladle = assets.loadTexture("props/aura_ladle.png")
        self.ending = self.window.gameState.endingType()

        self.lanterns = []
        for i in range(8):
            self.lanterns.append(SkyLantern(250 + i * 112, i))
        self.launched = 0

        self.phase = "lanterns"          # lanterns -> fireworks -> verdict -> photo -> polaroid -> credits
        self.phaseTime = 0.0
        self.rockets = []
        self.rocketTimer = 0.0
        self.rocketCount = 0
        self.groupAlpha = 0.0
        self.confettiTimer = 0.0

        # the group photo line-up (seated guest frames: 0 idle, 3 cheer)
        self.group = []
        for speaker, x, y in (("farah", 250, 108), ("lim", 420, 120), ("leon", 640, 134),
                              ("kavitha", 860, 120), ("nenek", 1030, 108)):
            self.group.append((assets.loadFrames(f"characters/guest_{speaker}.png", 4, 1), x, y))
        self.aimanFrames = assets.loadFrames("characters/aiman_idle.png", 4, 1)
        self.oyenFrames = assets.loadFrames("characters/oyen_sit.png", 4, 1)

        self.photoPending = False
        self.polaroidTexture = None
        self.polaroidPath = ""
        self.polaroidTween = Tween(0.0, 1.0, 0.01)
        self.creditsY = 0.0
        self.creditsDone = False
        self.closeTimer = 0.0
        self.botTimer = 0.0

    def getIntroLines(self):
        return [
            ("nenek", "Midnight, Aiman! The whole street is up here. Light the sky lanterns for the festival!"),
        ]

    # ------------------------------------------------------------------
    # Part 1: lanterns
    # ------------------------------------------------------------------
    def lightLantern(self, lantern):
        if lantern.state != "waiting":
            return
        lantern.state = "lit"
        lantern.time = 0.0
        lantern.light = self.lights.add(lantern.x, lantern.y, 160, LANTERN_ORANGE, 1.0, 0.25)
        # each lantern plays the next note of a pentatonic scale: speed = 2 ** (semitones / 12)
        semitones = PENTATONIC[self.launched % len(PENTATONIC)]
        self.playSfx("lantern_light", 0.6, x=lantern.x)
        self.playSfx("chime", 0.8, x=lantern.x, speed=2 ** (semitones / 12))
        spawnSparkles(self.glowParticles, lantern.x, lantern.y, GOLD, 14, 200)
        self.launched += 1

    def lightNextLantern(self):
        for lantern in self.lanterns:
            if lantern.state == "waiting":
                self.lightLantern(lantern)
                return

    def updateLanterns(self, dt):
        for lantern in self.lanterns:
            if lantern.state == "waiting":
                continue
            lantern.time += dt
            if lantern.state == "lit" and lantern.time > 0.6:
                lantern.state = "rising"
            if lantern.state == "rising":
                # rise faster and faster, swaying on the night breeze (sine)
                lantern.y += (30 + lantern.time * 28) * dt
                lantern.x = lantern.baseX + math.sin(lantern.time * 1.4 + lantern.phase) * 28
                if random.random() < 0.2:
                    spawnEmbers(self.glowParticles, lantern.x, lantern.y - 40, 1)
                if lantern.y > 800:
                    lantern.state = "gone"
                    self.lights.remove(lantern.light)
                    lantern.light = None
            if lantern.light is not None:
                lantern.light.x = lantern.x
                lantern.light.y = lantern.y

    # ------------------------------------------------------------------
    # Part 2: fireworks
    # ------------------------------------------------------------------
    def updateFireworks(self, dt):
        self.rocketTimer -= dt
        if self.phaseTime < FIREWORK_TIME - 2.0 and self.rocketTimer <= 0:
            self.rocketTimer = random.uniform(0.3, 0.55)
            colour = FIREWORK_COLOURS[self.rocketCount % len(FIREWORK_COLOURS)]
            shape = FIREWORK_SHAPES[self.rocketCount % len(FIREWORK_SHAPES)]
            self.rockets.append(Rocket(random.uniform(200, 1080), random.uniform(430, 610), colour, shape))
            self.rocketCount += 1
            self.playSfx("firework_launch", 0.5, x=self.rockets[-1].x, pitchVariation=0.1)
        for rocket in self.rockets:
            rocket.y += rocket.speed * dt
            rocket.speed *= 0.985
            if random.random() < 0.6:
                spawnEmbers(self.glowParticles, rocket.x, rocket.y - 10, 1)
            if rocket.y >= rocket.apexY:
                rocket.done = True
                spawnFirework(self.glowParticles, rocket.x, rocket.y, rocket.colour, 110, rocket.shape)
                spawnGlowBurst(self.glowParticles, rocket.x, rocket.y, rocket.colour, 260, 0.35)
                boom = "firework_boom_1" if self.rocketCount % 2 == 0 else "firework_boom_2"
                self.playSfx(boom, 0.7, x=rocket.x, pitchVariation=0.1)
                self.playSfx("crackle", 0.35, x=rocket.x, cooldown=0.4)
        alive = []
        for rocket in self.rockets:
            if not rocket.done:
                alive.append(rocket)
        self.rockets = alive

    # ------------------------------------------------------------------
    # Part 3-4: verdict and the polaroid
    # ------------------------------------------------------------------
    def startVerdict(self):
        self.phase = "verdict"
        self.phaseTime = 0.0
        self.dialogue.start(VERDICTS[self.ending])
        self.playSfx(VERDICT_VOICE[self.ending], 1.0)
        if self.ending == "golden":
            self.playSfx("fanfare", 0.8)
            self.playSfx("applause", 0.7)
        elif self.ending == "silver":
            self.playSfx("applause", 0.6)
        else:
            self.playSfx("happy_chime", 0.7)

    def startPhoto(self):
        self.phase = "photo"
        self.phaseTime = 0.0
        self.dialogue.start([("farah", "Everyone, squeeze in for a photo! One, two, three... say CHEESE!")])

    def takePhoto(self):
        """Capture the screen and 'develop' it into a polaroid (Pillow image processing)."""
        self.photoPending = False
        shot = self.window.grabScreenImage()
        polaroid = self.developPolaroid(shot)
        os.makedirs(MEMORY_DIR, exist_ok=True)
        number = 1
        path = os.path.join(MEMORY_DIR, f"polaroid_{number}.png")
        while os.path.exists(path):
            number += 1
            path = os.path.join(MEMORY_DIR, f"polaroid_{number}.png")
        polaroid.save(path)
        self.polaroidPath = path
        self.polaroidTexture = assets.textureFromImage(polaroid, "polaroid")
        self.polaroidTween = Tween(0.0, 1.0, 0.6, easeOutBack)
        self.flash.start(WHITE, 0.45, 255)
        self.playSfx("camera_shutter", 1.0)
        self.phase = "polaroid"
        self.phaseTime = 0.0

    def developPolaroid(self, shot):
        # 1. crop the middle of the screen (crop box = left, top, right, bottom) and shrink it
        photo = shot.crop((150, 100, 1130, 700)).resize((490, 300), Image.LANCZOS)

        # 2. colour grading per channel (RGB model): warmer reds, softer blues
        red, green, blue = photo.split()
        redTable = []
        greenTable = []
        blueTable = []
        for value in range(256):
            redTable.append(min(255, int(value * 1.10 + 8)))
            greenTable.append(min(255, int(value * 1.02 + 4)))
            blueTable.append(min(255, int(value * 0.86)))
        photo = Image.merge("RGB", (red.point(redTable), green.point(greenTable), blue.point(blueTable)))

        # 3. image enhancement: a little more contrast and colour
        photo = ImageEnhance.Contrast(photo).enhance(1.12)
        photo = ImageEnhance.Color(photo).enhance(1.15)

        # 4. vignette: multiply by a mask that is white in the middle and darker at the edges
        mask = Image.new("L", photo.size, 0)
        maskDraw = ImageDraw.Draw(mask)
        maskDraw.ellipse([-60, -60, photo.width + 60, photo.height + 60], fill=255)
        mask = mask.filter(ImageFilter.GaussianBlur(45))
        mask = mask.point(vignetteLevels())
        photo = ImageChops.multiply(photo, Image.merge("RGB", (mask, mask, mask)))

        # 5. film grain: random noise blended in (a digital artefact added on purpose)
        noise = Image.effect_noise(photo.size, 22).convert("RGB")
        photo = Image.blend(photo, ImageChops.overlay(photo, noise), 0.35)

        # 6. the white polaroid frame with a hand-written style caption
        card = Image.new("RGB", (530, 400), (250, 247, 238))
        card.paste(photo, (20, 20))
        draw = ImageDraw.Draw(card)
        stars = self.window.gameState.totalStars()
        caption = f"Pelita Street - Lantern Night - {stars}/12 stars"
        try:
            font = ImageFont.truetype(os.path.join(FONT_DIR, "Liberation_Sans_Italic.ttf"), 22)
        except OSError:
            font = ImageFont.load_default()
        draw.text((265, 358), caption, fill=(60, 50, 70), font=font, anchor="mm")
        return card

    # ------------------------------------------------------------------
    # Input
    # ------------------------------------------------------------------
    def onPlayMousePress(self, x, y, button):
        if self.phase == "lanterns":
            for lantern in self.lanterns:
                if lantern.contains(x, y):
                    self.lightLantern(lantern)
                    return
        elif self.phase == "credits":
            self.endCredits()

    def onPlayKeyPress(self, key, modifiers):
        if self.phase == "lanterns" and key == arcade.key.SPACE:
            self.lightNextLantern()
        elif self.phase == "credits" and key in (arcade.key.ENTER, arcade.key.RETURN, arcade.key.SPACE):
            self.endCredits()
        elif self.phase == "polaroid" and key in (arcade.key.ENTER, arcade.key.RETURN, arcade.key.SPACE):
            self.startCredits()

    def endCredits(self):
        if self.window.recording:
            return
        self.changeScene("title")

    def startCredits(self):
        self.phase = "credits"
        self.phaseTime = 0.0
        self.creditsY = 0.0

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------
    def updateAlways(self, dt):
        self.updateLanterns(dt)

    def updatePlay(self, dt):
        self.phaseTime += dt
        if self.phase == "lanterns":
            allGone = True
            for lantern in self.lanterns:
                if lantern.state == "waiting":
                    allGone = False
            if allGone and self.phaseTime > 1.0 and self.launched >= len(self.lanterns):
                self.phase = "fireworks"
                self.phaseTime = 0.0
        elif self.phase == "fireworks":
            self.updateFireworks(dt)
            if self.phaseTime > FIREWORK_TIME:
                self.startVerdict()
        elif self.phase == "verdict":
            self.groupAlpha = approach(self.groupAlpha, 1.0, 3.0, dt)
            if self.ending == "golden":
                self.confettiTimer -= dt
                if self.confettiTimer <= 0 and self.phaseTime < 6:
                    self.confettiTimer = 0.5
                    spawnConfetti(self.particles, random.uniform(200, 1080), 700, 30, 500)
            if not self.dialogue.isActive() and self.phaseTime > 1.0:
                self.startPhoto()
        elif self.phase == "photo":
            if not self.dialogue.isActive() and self.phaseTime > 0.5 and self.polaroidTexture is None:
                self.photoPending = True          # captured during the next draw
        elif self.phase == "polaroid":
            self.polaroidTween.update(dt)
            if self.phaseTime > 5.5:
                self.startCredits()
        elif self.phase == "credits":
            self.groupAlpha = approach(self.groupAlpha, 0.0, 2.0, dt)      # the group fades away
            self.creditsY += 42 * dt
            if self.creditsY > 1150 and not self.creditsDone:
                self.creditsDone = True
            if self.creditsDone:
                self.closeTimer += dt
                if self.window.recording and self.closeTimer > 2.0:
                    self.window.finishAndClose()
                elif not self.window.recording and self.closeTimer > 1.5:
                    self.changeScene("title")

    def autoPlay(self, dt):
        if self.dialogue.isActive():
            return
        if self.phase == "lanterns":
            self.botTimer += dt
            for lantern in self.lanterns:
                if lantern.state == "waiting":
                    if self.botMoveCursor(lantern.x, lantern.y, dt, 1100) and self.botTimer > 0.7:
                        self.botTimer = 0.0
                        self.botClick()
                    return

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------
    def drawWorld(self):
        assets.drawTexture(self.background, 640, 360, 1280, 720)
        for lantern in self.lanterns:
            if lantern.state == "gone":
                continue
            if lantern.state == "waiting":
                pulse = 0.5 + 0.5 * math.sin(self.sceneTime * 3 + lantern.index)
                assets.drawTexture(self.glow, lantern.x, lantern.y, 110, 110, rgb=GOLD, alpha=60 * pulse,
                                   additive=True)
                assets.drawTexture(self.lanternOff, lantern.x, lantern.y, 72, 88)
            else:
                assets.drawTexture(self.glow, lantern.x, lantern.y, 190, 190, rgb=LANTERN_ORANGE, alpha=170,
                                   additive=True)
                sway = math.sin(lantern.time * 1.4 + lantern.phase) * 6
                assets.drawTexture(self.lanternOn, lantern.x, lantern.y, 72, 88, sway)
        if self.groupAlpha > 0.01:
            self.drawGroup()

    def drawGroup(self):
        cheer = self.phase in ("photo", "polaroid") or (self.phase == "verdict" and self.ending != "bronze")
        frameIndex = 3 if cheer else 0
        alpha = 255 * self.groupAlpha
        for frames, x, y in self.group:
            assets.drawTexture(frames[frameIndex], x, y, 190, 232, alpha=alpha)
        aimanFrame = self.aimanFrames[int(self.sceneTime * 3) % 4]
        assets.drawTexture(aimanFrame, 560, 96, 170, 212, alpha=alpha)
        assets.drawTexture(self.oyenFrames[int(self.sceneTime * 4) % 4], 760, 62, 150, 112, alpha=alpha)
        # the Aura Ladle glows in Aiman's hand when the ending is golden
        if self.ending == "golden":
            pulse = 0.7 + 0.3 * math.sin(self.sceneTime * 4)
            assets.drawTexture(self.glow, 612, 150, 170, 170, rgb=GOLD, alpha=220 * pulse * self.groupAlpha,
                               additive=True)
            assets.drawTexture(self.ladle, 612, 150, 84, 84, -20, alpha=alpha)

    def drawPlayHud(self):
        if self.photoPending:
            self.takePhoto()                  # the world is drawn, the dialogue box is not yet
        if self.phase == "lanterns":
            drawPanel(640, 660, 520, 50, (14, 16, 38, 205), (255, 200, 87, 150), 16, shadow=False)
            drawText(f"Light the sky lanterns  {self.launched} / {len(self.lanterns)}", 640, 660, CREAM, 17,
                     FONT_BODY, anchorX="center", anchorY="center", bold=True)
            drawKeyHint(560, 28, "CLICK", "Light a lantern")
            drawKeyHint(760, 28, "SPACE", "Next lantern")
        elif self.phase == "polaroid":
            self.drawPolaroid()
        elif self.phase == "credits":
            self.drawCredits()

    def drawPolaroid(self):
        grow = self.polaroidTween.value
        if self.polaroidTexture is None or grow <= 0.01:
            return
        arcade.draw_lbwh_rectangle_filled(0, 0, 1280, 720, (6, 6, 16, int(120 * min(1.0, grow))))
        width = 530 * grow
        height = 400 * grow
        # soft shadow under the photo, then the photo tilted by 4 degrees (rotation)
        assets.drawTexture(self.polaroidTexture, 650, 372, width, height, 4, alpha=110, rgb=(0, 0, 0))
        assets.drawTexture(self.polaroidTexture, 640, 382, width, height, 4)
        drawText("Memory saved to the memories folder", 640, 150, CREAM, 16, FONT_BODY, anchorX="center",
                 anchorY="center", alpha=255 * min(1.0, grow))

    def drawCredits(self):
        arcade.draw_lbwh_rectangle_filled(0, 0, 1280, 720, (6, 6, 16, 205))
        y = self.creditsY - 20
        for text, colour, size in CREDIT_LINES:
            if text != "" and -40 < y < 760:
                font = FONT_TITLE if size >= 24 else FONT_BODY
                drawText(text, 640, y, colour, size, font, anchorX="center", anchorY="center")
            y -= size + 22
        if not self.window.recording:
            drawText("ENTER to skip", 1240, 24, CREAM, 12, FONT_BODY, anchorX="right", anchorY="center", alpha=150)


def vignetteLevels():
    """Lookup table that keeps the vignette gentle: 0..255 becomes 110..255."""
    table = []
    for value in range(256):
        table.append(int(110 + value * 145 / 255))
    return table
