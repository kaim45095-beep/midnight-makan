"""
story_scene.py
--------------
The opening story, told like a comic: four illustrated panels, one after
the other, with dialogue underneath.

Each panel is composed while the game runs from art that already exists
(backgrounds, portraits, props, the newspaper poster) and moves with a
"Ken Burns" effect: a slow zoom and pan. That is simply scaling and
translation (Week 5-6) interpolated over time (Week 7).

Everything outside the panel frame is covered again afterwards, so the
zoomed picture never spills out of its frame (a simple mask).
"""
import math
import random

import arcade

from core import assets
from core.base_scene import BaseScene
from core.helpers import Tween, clamp, easeInOutSine, easeOutBack, lerp
from core.particles import spawnEmbers, spawnSparkles
from core.ui import drawRoundedRectOutline, drawText
from settings import CREAM, FONT_BODY, GOLD, WARM_LIGHT

FRAME_X = 640
FRAME_Y = 452
FRAME_W = 1120
FRAME_H = 470
PAGE_COLOUR = (16, 18, 40)

PANELS = [
    {
        "caption": "1. Pelita Street",
        "lines": [("narrator", "Kuala Lumpur. Lantern Festival night on Pelita Street, the busiest night market in town.")],
        "zoom": (1.0, 1.1), "pan": (180, -180),
    },
    {
        "caption": "2. The Aura Ladle",
        "lines": [("nenek", "My knees are tired, Aiman. Tonight you cook for our family stall."),
                  ("nenek", "This ladle has fed our family for sixty years. It glows only when food is cooked with heart.")],
        "zoom": (1.05, 1.18), "pan": (0, 0),
    },
    {
        "caption": "3. The Golden Spoon",
        "lines": [("narrator", "Tonight the famous critic Dato' Leon Tan visits. The best cook wins the last spot "
                               "in the Festival Food Hall - and the rent for a whole year.")],
        "zoom": (1.0, 1.3), "pan": (0, 0),
    },
    {
        "caption": "4. Trouble on four paws",
        "lines": [("aiman", "Oyen... don't you dare touch the food tonight."),
                  ("nenek", "Four courses, Aiman: rojak, char kuey teow, teh tarik, then the banquet. Now go!")],
        "zoom": (1.1, 1.0), "pan": (-60, 60),
    },
]


class StoryScene(BaseScene):
    sceneId = "story"
    sceneTitle = "The Story"
    musicName = "music_title"
    ambienceName = "amb_night_market"
    isGameplayScene = False
    controlsHelp = [("SPACE / CLICK", "Next line"), ("TAB", "Skip this panel's lines"),
                    ("ESC", "Skip the whole story")]

    def setupScene(self):
        self.sky = assets.loadTexture("backgrounds/title_sky.png")
        self.farLayer = assets.loadTexture("backgrounds/street_far.png")
        self.midLayer = assets.loadTexture("backgrounds/street_mid.png")
        self.mainLayer = assets.loadTexture("backgrounds/street_main.png")
        self.kitchen = assets.loadTexture("backgrounds/wok_kitchen_bg.png")
        self.mamak = assets.loadTexture("backgrounds/mamak_bg.png")
        self.poster = assets.loadTexture("ui/poster_critic.png")
        self.ladle = assets.loadTexture("props/aura_ladle.png")
        self.ladleGlow = assets.glowTexture("props/aura_ladle.png", 18, (255, 210, 110))
        self.glow = assets.loadTexture("particles/light_glow.png")
        self.nenek = assets.loadTexture("characters/portrait_nenek.png")
        self.aiman = assets.loadTexture("characters/portrait_aiman.png")
        self.oyenFrames = assets.loadFrames("characters/oyen_walk.png", 4, 1)

        self.panelIndex = -1
        self.panelTime = 0.0
        self.slide = Tween(0.0, 0.0, 0.01)
        self.waitNext = 0.0
        self.captionText = arcade.Text("", FRAME_X - FRAME_W / 2 + 16, FRAME_Y + FRAME_H / 2 - 26, GOLD, 16,
                                       font_name=FONT_BODY, bold=True, anchor_y="center")
        self.nextPanel()

    # ------------------------------------------------------------------
    # Panel flow
    # ------------------------------------------------------------------
    def nextPanel(self):
        self.panelIndex += 1
        if self.panelIndex >= len(PANELS):
            self.changeScene("street")
            return
        self.panelTime = 0.0
        # the new panel slides in from the right with a little overshoot
        self.slide = Tween(1300.0, 0.0, 0.55, easeOutBack)
        self.captionText.text = PANELS[self.panelIndex]["caption"]
        self.dialogue.start(PANELS[self.panelIndex]["lines"])
        self.playSfx("knife_whoosh", 0.4, speed=0.7)

    def onEscape(self):
        self.changeScene("street")

    def updateAlways(self, dt):
        self.slide.update(dt)
        self.panelTime += dt
        if self.state != "play" or self.sceneEnded:
            return
        if not self.dialogue.isActive():
            self.waitNext += dt
            if self.waitNext > 0.5:
                self.waitNext = 0.0
                self.nextPanel()
        # a few sparkles / embers keep the panels alive
        if self.panelIndex == 1 and random.random() < 0.25:
            x, y = self.toScreen(random.uniform(-80, 80), random.uniform(-40, 80))
            spawnSparkles(self.glowParticles, x, y, (255, 214, 120), 1, 60)
        if self.panelIndex == 0 and random.random() < 0.15:
            x, y = self.toScreen(random.uniform(-500, 500), random.uniform(-120, 60))
            spawnEmbers(self.glowParticles, x, y, 1)

    # ------------------------------------------------------------------
    # Ken Burns maths: local panel coordinates -> screen coordinates
    # ------------------------------------------------------------------
    def kenBurns(self):
        panel = PANELS[self.panelIndex]
        t = easeInOutSine(clamp(self.panelTime / 14.0, 0.0, 1.0))
        zoom = lerp(panel["zoom"][0], panel["zoom"][1], t)          # scaling
        pan = lerp(panel["pan"][0], panel["pan"][1], t)             # translation
        return zoom, pan

    def toScreen(self, localX, localY):
        zoom, pan = self.kenBurns()
        return FRAME_X + self.slide.value + (localX + pan) * zoom, FRAME_Y + localY * zoom

    def drawLocal(self, texture, localX, localY, width, height, angle=0.0, alpha=255, rgb=None, additive=False):
        zoom, pan = self.kenBurns()
        x, y = self.toScreen(localX, localY)
        assets.drawTexture(texture, x, y, width * zoom, height * zoom, angle, alpha, rgb, additive)

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------
    def drawWorld(self):
        arcade.draw_lbwh_rectangle_filled(0, 0, 1280, 720, PAGE_COLOUR)
        if 0 <= self.panelIndex < len(PANELS):
            if self.panelIndex == 0:
                self.drawStreetPanel()
            elif self.panelIndex == 1:
                self.drawLadlePanel()
            elif self.panelIndex == 2:
                self.drawPosterPanel()
            else:
                self.drawOyenPanel()

    def drawStreetPanel(self):
        self.drawLocal(self.sky, 0, 60, 1280, 720)
        self.drawLocal(self.farLayer, 250, 0, 1920, 720)
        self.drawLocal(self.midLayer, 500, 0, 2560, 720)
        self.drawLocal(self.mainLayer, 700, 40, 3200, 720)

    def drawLadlePanel(self):
        self.drawLocal(self.kitchen, 0, 0, 1280, 720, rgb=(255, 190, 150))
        arcade.draw_lbwh_rectangle_filled(0, 0, 1280, 720, (40, 16, 8, 110))
        pulse = 0.7 + 0.3 * math.sin(self.panelTime * 3)
        self.drawLocal(self.glow, 0, 20, 700, 520, rgb=WARM_LIGHT, alpha=170 * pulse, additive=True)
        self.drawLocal(self.nenek, -330, -40, 340, 340)
        self.drawLocal(self.aiman, 330, -40, 340, 340)
        self.drawLocal(self.ladleGlow, 0, 40, self.ladleGlow.width * 1.2, self.ladleGlow.height * 1.2,
                       alpha=255 * pulse, additive=True)
        bob = math.sin(self.panelTime * 2) * 8
        self.drawLocal(self.ladle, 0, 40 + bob, 260, 260, -8)

    def drawPosterPanel(self):
        self.drawLocal(self.mamak, 0, 0, 1280, 720, rgb=(150, 150, 170))
        self.drawLocal(self.poster, 0, 0, 620, 430, -3)

    def drawOyenPanel(self):
        self.drawLocal(self.sky, 0, 60, 1280, 720)
        self.drawLocal(self.mainLayer, 300, 40, 3200, 720)
        arcade.draw_lbwh_rectangle_filled(0, 0, 1280, 720, (8, 8, 24, 170))
        # a spotlight follows Oyen as he sneaks across (additive light)
        catX = lerp(-420, 260, clamp(self.panelTime / 9.0, 0.0, 1.0))
        frame = int(self.panelTime / 0.16) % 4
        self.drawLocal(self.glow, catX, -110, 520, 380, rgb=(255, 230, 190), alpha=210, additive=True)
        self.drawLocal(self.oyenFrames[frame], catX, -150, 230, 172)
        self.drawLocal(self.aiman, 420, -40, 300, 300)

    def drawWorldTop(self):
        # Mask: cover everything outside the frame so the zoomed picture stays inside
        left = FRAME_X - FRAME_W / 2 + self.slide.value
        right = left + FRAME_W
        bottom = FRAME_Y - FRAME_H / 2
        top = FRAME_Y + FRAME_H / 2
        arcade.draw_lbwh_rectangle_filled(0, top, 1280, 720 - top, PAGE_COLOUR)
        arcade.draw_lbwh_rectangle_filled(0, 0, 1280, bottom, PAGE_COLOUR)
        arcade.draw_lbwh_rectangle_filled(0, bottom, max(0, left), FRAME_H, PAGE_COLOUR)
        arcade.draw_lbwh_rectangle_filled(right, bottom, max(0, 1280 - right), FRAME_H, PAGE_COLOUR)
        # comic frame: thick dark border and a thin gold line
        centreX = left + FRAME_W / 2
        drawRoundedRectOutline(centreX, FRAME_Y, FRAME_W + 10, FRAME_H + 10, (6, 6, 14, 255), 10, 8)
        drawRoundedRectOutline(centreX, FRAME_Y, FRAME_W + 22, FRAME_H + 22, GOLD, 14, 2)
        self.captionText.x = left + 16
        arcade.draw_lbwh_rectangle_filled(left, top - 44, 260, 36, (10, 10, 24, 200))
        self.captionText.draw()

    def drawPlayHud(self):
        # panel counter dots (like pages)
        for i in range(len(PANELS)):
            colour = GOLD if i == self.panelIndex else (90, 90, 120)
            arcade.draw_circle_filled(640 - 36 + i * 24, 700, 6, colour)
        drawText("ESC skips the story", 1240, 700, CREAM, 12, FONT_BODY, anchorX="right", anchorY="center",
                 alpha=150)
