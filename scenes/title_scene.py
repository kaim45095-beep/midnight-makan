"""
title_scene.py
--------------
The title screen of Midnight Makan: a night sky over the Kuala Lumpur
skyline, sky lanterns floating up, a glowing logo and the main menu.

Sub-screens (drawn as panels on top): Scene Select, How to Play, Credits.

Techniques used (lecture links):
  * Week 3  - keyboard (UP / DOWN / ENTER) and mouse (hover / click) events
  * Week 7  - interpolation: floating logo (sine), smooth button hover
  * Week 8  - additive glow (logo, moon, lanterns) = simple lighting,
              a star field made of many small sprites in one SpriteList
"""
import math
import random

import arcade

from core import assets
from core.base_scene import BaseScene
from core.helpers import approach
from core.ui import Button, drawPanel, drawRoundedRect, drawRoundedRectOutline, drawStars, drawText
from settings import (CREAM, FONT_BODY, FONT_TITLE, GOLD, LANTERN_ORANGE, LAVENDER, SCENE_NAMES, SCENE_ORDER,
                      WARM_LIGHT)

MENU_ITEMS = ["START STORY", "SCENE SELECT", "HOW TO PLAY", "CREDITS", "QUIT"]

# Scene Select cards: scene id, title, thumbnail picture
CARDS = [
    ("street", "Pelita Street", "backgrounds/street_main.png"),
    ("rojak", "1  " + SCENE_NAMES["rojak"], "backgrounds/fruit_stall_bg.png"),
    ("wok", "2  " + SCENE_NAMES["wok"], "backgrounds/wok_kitchen_bg.png"),
    ("tehtarik", "3  " + SCENE_NAMES["tehtarik"], "backgrounds/mamak_bg.png"),
    ("banquet", "4  " + SCENE_NAMES["banquet"], "backgrounds/banquet_bg.png"),
    ("finale", "Finale: Lantern Bloom", "backgrounds/rooftop_bg.png"),
]
CARD_W = 330
CARD_H = 200

HOW_TO_PLAY = [
    ("Pelita Street", "LEFT / RIGHT walk, UP enters a stall"),
    ("1 Rojak Rush", "SPACE / click when the cleaver meets a dotted line; click Oyen to shoo him"),
    ("2 Wok Hei Fire", "Drag bowls into the wok in ticket order, UP / DOWN gas, SPACE tosses"),
    ("3 The Great Pull", "Hold to pour milk, then move the jug and hold to pull; ENTER serves"),
    ("4 Lazy Susan", "LEFT / RIGHT spins the table, SPACE serves the guest who is lined up"),
    ("Finale", "Click the sky lanterns to light them and watch the fireworks"),
    ("Everywhere", "F1 help   F2 maths / debug view   F3 music   F12 screenshot   ESC pause"),
]

CREDITS = [
    ("MIDNIGHT MAKAN", GOLD, 22),
    ("A Night-Market Dining Adventure", CREAM, 15),
    ("", CREAM, 8),
    ("Created by Mohammad Kaymul Islam (TP088446)", CREAM, 15),
    ("APU2F2602CS(AI) - Asia Pacific University", CREAM, 14),
    ("CT029-3-2 Imaging and Special Effects", CREAM, 14),
    ("", CREAM, 8),
    ("Built with Python Arcade 3.3, Pillow, NumPy and OpenCV", LAVENDER, 14),
    ("All art, music and sound effects are generated procedurally", LAVENDER, 14),
    ("by the Python scripts in the tools/ folder", LAVENDER, 14),
    ("Fonts: Kenney Future (CC0), Liberation Sans (SIL OFL)", LAVENDER, 14),
    ("Developed with declared GenAI assistance (APU Yellow / restricted use)", LAVENDER, 13),
]


class SkyLantern:
    """A glowing paper lantern drifting up into the sky."""

    def __init__(self, startLow):
        self.reset(startLow)

    def reset(self, startLow):
        self.depth = random.uniform(0.35, 1.0)        # small + slow = far away
        self.x = random.uniform(40, 1240)
        if startLow:
            self.y = random.uniform(-80, 720)
        else:
            self.y = -80
        self.speed = 18 + 34 * self.depth
        self.phase = random.uniform(0, 6.28)

    def update(self, dt):
        self.y += self.speed * dt
        self.phase += dt
        if self.y > 800:
            self.reset(False)


class TitleScene(BaseScene):
    sceneId = "title"
    sceneTitle = "Midnight Makan"
    musicName = "music_title"
    ambienceName = "amb_rooftop"
    isGameplayScene = False
    useVignette = True
    controlsHelp = [("UP / DOWN", "Choose a menu item"), ("ENTER / CLICK", "Select"),
                    ("ESC", "Back")]

    def setupScene(self):
        self.sky = assets.loadTexture("backgrounds/title_sky.png")
        self.skyline = assets.loadTexture("backgrounds/title_skyline.png")
        self.logo = assets.loadTexture("ui/logo.png")
        self.logoGlow = assets.loadTexture("ui/logo_glow.png")
        self.lanternTexture = assets.loadTexture("props/sky_lantern_lit.png")
        self.glowTexture = assets.loadTexture("particles/light_glow.png")

        # Star field: 130 tiny sprites that twinkle (their alpha follows a sine wave)
        self.stars = arcade.SpriteList()
        starTexture = assets.loadTexture("particles/soft_circle.png")
        for i in range(130):
            star = arcade.Sprite(starTexture)
            star.center_x = random.uniform(0, 1280)
            star.center_y = random.uniform(330, 720)
            size = random.uniform(3, 8)
            star.scale = size / starTexture.width
            star.phase = random.uniform(0, 6.28)
            star.speed = random.uniform(1.0, 3.2)
            self.stars.append(star)

        self.lanterns = []
        for i in range(14):
            self.lanterns.append(SkyLantern(True))
        self.moonLight = self.lights.add(1040, 572, 330, (255, 240, 210), 0.35, 0.05)

        self.buttons = []
        y = 320
        for label in MENU_ITEMS:
            self.buttons.append(Button(label, 640, y, 330, 54, 18))
            y -= 62
        self.selected = 0
        self.screen = "menu"               # menu / select / howto / credits
        self.cardIndex = 0
        self.cardTextures = []
        for sceneId, name, path in CARDS:
            if sceneId == "street":
                self.cardTextures.append(self.makeStreetThumbnail())
            else:
                self.cardTextures.append(assets.loadTexture(path))
        self.fade = 0.0
        self.botTimer = 0.0

    def makeStreetThumbnail(self):
        """Image compositing with Pillow: sky + far + mid + a 1280 px window of the street."""
        picture = assets.loadImage("backgrounds/title_sky.png").convert("RGBA")
        for path, left in (("backgrounds/street_far.png", 0), ("backgrounds/street_mid.png", 150),
                           ("backgrounds/street_main.png", 300)):
            layer = assets.loadImage(path).convert("RGBA")
            window = layer.crop((left, 0, left + 1280, 720))       # crop box: left, top, right, bottom
            picture.alpha_composite(window)
        return assets.textureFromImage(picture, "street-thumbnail")

    # ------------------------------------------------------------------
    # Menu actions
    # ------------------------------------------------------------------
    def activate(self, index):
        label = MENU_ITEMS[index]
        self.playSfx("ui_confirm", 0.7)
        if label == "START STORY":
            self.window.gameState.reset()
            self.changeScene("story")
        elif label == "SCENE SELECT":
            self.screen = "select"
            self.cardIndex = 0
        elif label == "HOW TO PLAY":
            self.screen = "howto"
        elif label == "CREDITS":
            self.screen = "credits"
        elif label == "QUIT":
            arcade.exit()

    def startCard(self, index):
        sceneId = CARDS[index][0]
        gameState = self.window.gameState
        self.playSfx("ui_confirm", 0.7)
        if sceneId == "street":
            gameState.reset()                  # a fresh story from the street
        elif sceneId == "finale":
            gameState.storyMode = False
            if not gameState.allCompleted():
                # jump straight to the finale: pretend every dish earned 2 stars
                for stallId in SCENE_ORDER:
                    if not gameState.isCompleted(stallId):
                        gameState.recordResult(stallId, 60, 100, 5)
        else:
            gameState.storyMode = False
        self.changeScene(sceneId)

    def goBack(self):
        if self.screen != "menu":
            self.screen = "menu"
            self.playSfx("ui_back", 0.6)

    # ------------------------------------------------------------------
    # Input
    # ------------------------------------------------------------------
    def onPlayKeyPress(self, key, modifiers):
        if self.screen == "menu":
            if key in (arcade.key.UP, arcade.key.W):
                self.selected = (self.selected - 1) % len(MENU_ITEMS)
                self.playSfx("ui_hover", 0.5)
            elif key in (arcade.key.DOWN, arcade.key.S):
                self.selected = (self.selected + 1) % len(MENU_ITEMS)
                self.playSfx("ui_hover", 0.5)
            elif key in (arcade.key.ENTER, arcade.key.RETURN, arcade.key.SPACE):
                self.activate(self.selected)
        elif self.screen == "select":
            if key in (arcade.key.LEFT, arcade.key.A):
                self.cardIndex = (self.cardIndex - 1) % len(CARDS)
                self.playSfx("ui_hover", 0.5)
            elif key in (arcade.key.RIGHT, arcade.key.D):
                self.cardIndex = (self.cardIndex + 1) % len(CARDS)
                self.playSfx("ui_hover", 0.5)
            elif key in (arcade.key.UP, arcade.key.DOWN):
                self.cardIndex = (self.cardIndex + 3) % len(CARDS)
                self.playSfx("ui_hover", 0.5)
            elif key in (arcade.key.ENTER, arcade.key.RETURN, arcade.key.SPACE):
                self.startCard(self.cardIndex)
            elif key == arcade.key.BACKSPACE:
                self.goBack()
        else:
            if key in (arcade.key.ENTER, arcade.key.RETURN, arcade.key.SPACE, arcade.key.BACKSPACE):
                self.goBack()

    def onEscape(self):
        self.goBack()

    def cardPosition(self, index):
        column = index % 3
        row = index // 3
        return 290 + column * 350, 430 - row * 230

    def onPlayMouseMotion(self, x, y, dx, dy):
        if self.screen == "menu":
            for i in range(len(self.buttons)):
                if self.buttons[i].contains(x, y) and self.selected != i:
                    self.selected = i
                    self.playSfx("ui_hover", 0.4)
        elif self.screen == "select":
            for i in range(len(CARDS)):
                cx, cy = self.cardPosition(i)
                if abs(x - cx) < CARD_W / 2 and abs(y - cy) < CARD_H / 2 and self.cardIndex != i:
                    self.cardIndex = i
                    self.playSfx("ui_hover", 0.4)

    def onPlayMousePress(self, x, y, button):
        if self.screen == "menu":
            for i in range(len(self.buttons)):
                if self.buttons[i].contains(x, y):
                    self.activate(i)
                    return
        elif self.screen == "select":
            for i in range(len(CARDS)):
                cx, cy = self.cardPosition(i)
                if abs(x - cx) < CARD_W / 2 and abs(y - cy) < CARD_H / 2:
                    self.startCard(i)
                    return
            self.goBack()
        else:
            self.goBack()

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------
    def updateAlways(self, dt):
        self.fade = approach(self.fade, 1.0, 2.0, dt)
        for star in self.stars:
            star.phase += star.speed * dt
            star.alpha = int(120 + 120 * (0.5 + 0.5 * math.sin(star.phase)))
        for lantern in self.lanterns:
            lantern.update(dt)
        for i in range(len(self.buttons)):
            self.buttons[i].hovered = (i == self.selected and self.screen == "menu")

    def autoPlay(self, dt):
        """Demo: peek at How to Play and Scene Select, then start the story."""
        self.botTimer += dt
        plan = [
            (2.0, "move", 196),         # glide to HOW TO PLAY
            (2.2, "click", None),
            (6.0, "wait", None),
            (6.2, "key", arcade.key.ENTER),
            (7.6, "move", 258),         # glide to SCENE SELECT
            (7.8, "click", None),
            (11.0, "wait", None),
            (11.2, "key", arcade.key.ESCAPE),
            (12.6, "move", 320),        # glide to START STORY
            (99.0, "click", None),
        ]
        for endTime, action, value in plan:
            if self.botTimer < endTime:
                if action == "move":
                    self.botMoveCursor(640, value, dt, 700)
                elif action == "click" and self.botStep != endTime:
                    self.botStep = endTime          # remember: click only once per plan step
                    self.botClick()
                elif action == "key" and self.botStep != endTime:
                    self.botStep = endTime
                    self.botTapKey(value)
                return

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------
    def drawWorld(self):
        assets.drawTexture(self.sky, 640, 360, 1280, 720)
        self.stars.draw()
        # far lanterns first, near lanterns later
        for lantern in sorted(self.lanterns, key=lanternDepth):
            size = 70 * lantern.depth
            sway = math.sin(lantern.phase * 1.3) * 12 * lantern.depth
            flicker = 0.8 + 0.2 * math.sin(lantern.phase * 7)
            assets.drawTexture(self.glowTexture, lantern.x + sway, lantern.y, size * 2.6, size * 2.6,
                               rgb=LANTERN_ORANGE, alpha=110 * flicker * lantern.depth, additive=True)
            assets.drawTexture(self.lanternTexture, lantern.x + sway, lantern.y, size * 0.82, size,
                               alpha=140 + 115 * lantern.depth)
        assets.drawTexture(self.skyline, 640, 360, 1280, 720)

    def drawPlayHud(self):
        if self.screen == "menu":
            self.drawMenu()
        elif self.screen == "select":
            self.drawSceneSelect()
        elif self.screen == "howto":
            self.drawHowTo()
        elif self.screen == "credits":
            self.drawCredits()

    def drawMenu(self):
        bob = math.sin(self.sceneTime * 1.6) * 6
        pulse = 0.65 + 0.35 * math.sin(self.sceneTime * 2.2)
        assets.drawTexture(self.logoGlow, 640, 540 + bob, 860, 314, alpha=200 * pulse * self.fade, additive=True)
        assets.drawTexture(self.logo, 640, 540 + bob, 820, 300, alpha=255 * self.fade)
        for button in self.buttons:
            button.draw()
        drawText("UP / DOWN + ENTER, or click", 640, 18, CREAM, 12, FONT_BODY, anchorX="center", alpha=160)

    def drawSceneSelect(self):
        arcade.draw_lbwh_rectangle_filled(0, 0, 1280, 720, (6, 6, 16, 150))
        drawText("SCENE SELECT", 640, 640, GOLD, 30, FONT_TITLE, anchorX="center", anchorY="center")
        gameState = self.window.gameState
        for i in range(len(CARDS)):
            sceneId, name, path = CARDS[i]
            cx, cy = self.cardPosition(i)
            chosen = i == self.cardIndex
            scale = 1.05 if chosen else 1.0
            border = GOLD if chosen else (120, 110, 150)
            drawPanel(cx, cy, CARD_W * scale, CARD_H * scale, (18, 20, 44, 235), border, 14)
            texture = self.cardTextures[i]
            assets.drawTexture(texture, cx, cy + 18, 300 * scale, 168 * scale)
            drawRoundedRectOutline(cx, cy + 18, 302 * scale, 170 * scale, (0, 0, 0, 160), 6, 3)
            drawRoundedRect(cx, cy - 78 * scale, CARD_W * scale - 8, 40, (10, 12, 30, 230), 10)
            drawText(name, cx - 150 * scale, cy - 78 * scale, CREAM, 14, FONT_BODY, anchorY="center", bold=True)
            if sceneId in SCENE_ORDER:
                drawStars(cx + 110 * scale, cy - 78 * scale, gameState.getStars(sceneId), 3, 22)
        drawText("Click a scene (or arrows + ENTER).  ESC to go back", 640, 70, CREAM, 14, FONT_BODY,
                 anchorX="center", anchorY="center", alpha=200)

    def drawHowTo(self):
        arcade.draw_lbwh_rectangle_filled(0, 0, 1280, 720, (6, 6, 16, 170))
        drawPanel(640, 360, 1080, 520)
        drawText("HOW TO PLAY", 640, 575, GOLD, 28, FONT_TITLE, anchorX="center", anchorY="center")
        y = 510
        for title, text in HOW_TO_PLAY:
            drawText(title, 150, y, GOLD, 15, FONT_BODY, anchorY="center", bold=True)
            drawText(text, 340, y, CREAM, 15, FONT_BODY, anchorY="center")
            y -= 52
        drawText("Every dining scene: 5 good actions in a row start AURA MODE (double points).", 640, 148,
                 WARM_LIGHT, 14, FONT_BODY, anchorX="center", anchorY="center")
        drawText("ENTER / ESC / click to go back", 640, 120, CREAM, 13, FONT_BODY, anchorX="center",
                 anchorY="center", alpha=180)

    def drawCredits(self):
        arcade.draw_lbwh_rectangle_filled(0, 0, 1280, 720, (6, 6, 16, 170))
        drawPanel(640, 360, 900, 520)
        y = 560
        for text, colour, size in CREDITS:
            if text != "":
                font = FONT_TITLE if size >= 20 else FONT_BODY
                drawText(text, 640, y, colour, size, font, anchorX="center", anchorY="center")
            y -= size + 18
        drawText("ENTER / ESC / click to go back", 640, 130, CREAM, 13, FONT_BODY, anchorX="center",
                 anchorY="center", alpha=180)


def lanternDepth(lantern):
    """Sort key for drawing: far (small) lanterns first."""
    return lantern.depth
