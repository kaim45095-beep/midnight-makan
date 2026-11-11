"""
street_scene.py
---------------
Pelita Street - the night-market street that links the four dining scenes.

Aiman walks left and right (arrow keys or A / D) past the stalls and
presses UP (or ENTER) at a stall to go inside. The stalls open one by
one; finished stalls show the stars that were earned.

Techniques used (lecture links):
  * Week 3  - arrow-key movement and a checkBoundary() method, exactly
              like the tutorial, but in a world that is wider than the window
  * Week 4  - sprite sheet walk cycle (8 frames) and idle animation
  * Week 5-6 - translation: a camera follows Aiman; mirroring: the walk
              frames are flipped when he walks left
  * Parallax scrolling - every background layer moves at its own speed
              (far = slow, near = fast), which makes the flat pictures feel deep
  * Week 8  - swinging lanterns with flickering point lights, dust particles
"""
import math
import random

import arcade

from core import assets
from core.base_scene import BaseScene
from core.helpers import approach, clamp
from core.particles import spawnDust, spawnEmbers
from core.ui import drawKeyHint, drawPanel, drawProgressBar, drawStars, drawText
from settings import (CREAM, FONT_BODY, FONT_TITLE, GOLD, LANTERN_ORANGE, PLAYER_WALK_SPEED, SCENE_ORDER,
                      STALL_NAMES, STALL_X, STREET_FLOOR_Y, STREET_WIDTH, WARM_LIGHT)

# Parallax factors: how much each layer moves compared with the camera
FAR_FACTOR = 0.25
MID_FACTOR = 0.6
CROWD_FACTOR = 0.8
LANTERN_FACTOR = 1.15

ENTER_DISTANCE = 90          # how close Aiman must be to a stall door
PLAYER_WIDTH = 128
PLAYER_HEIGHT = 160
OYEN_X = 1650
OYEN_Y = 243

# a short line from Grandma Siti each time Aiman comes back with a finished dish
PROGRESS_LINES = {
    0: [("nenek", "Here we are, Pelita Street! Walk with LEFT and RIGHT, then press UP at a stall to go in."),
        ("nenek", "Start with Uncle Din's fruit stall. Dato' Leon arrives at midnight, so be quick!")],
    1: [("nenek", "Beautiful rojak! Ah Seng's wok is fired up next door. Go, go!")],
    2: [("aiman", "That wok was HOT. Now to the Sri Pelita Cafe for the teh tarik!")],
    3: [("nenek", "Everything is ready. To the banquet hall, Aiman. Dato' Leon is waiting!")],
}


class Walker:
    """A background silhouette strolling along the street (crowd life)."""

    def __init__(self, texture, x, speed):
        self.texture = texture
        self.x = x
        self.speed = speed
        self.phase = random.uniform(0, 6.28)

    def update(self, dt):
        self.x += self.speed * dt
        if self.x > STREET_WIDTH + 200:
            self.x = -200
        elif self.x < -200:
            self.x = STREET_WIDTH + 200
        self.phase += dt * 7


class StreetScene(BaseScene):
    sceneId = "street"
    sceneTitle = "Pelita Street"
    musicName = "music_street"
    ambienceName = "amb_night_market"
    isGameplayScene = False
    controlsHelp = [("LEFT / RIGHT", "Walk along Pelita Street (A / D work too)"),
                    ("UP / ENTER", "Go into the stall in front of you"),
                    ("ESC", "Back to the title screen")]

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def setupScene(self):
        gameState = self.window.gameState
        self.sky = assets.loadTexture("backgrounds/title_sky.png")
        self.farLayer = assets.loadTexture("backgrounds/street_far.png")
        self.midLayer = assets.loadTexture("backgrounds/street_mid.png")
        self.mainLayer = assets.loadTexture("backgrounds/street_main.png")
        self.lanternTexture = assets.loadTexture("props/lantern_red.png")
        self.lockTexture = assets.loadTexture("ui/icon_lock.png")

        # Aiman's animations (Week 4 sprite sheets) and mirrored copies (Week 5-6)
        self.walkRight = assets.loadFrames("characters/aiman_walk.png", 4, 2)
        self.walkLeft = assets.loadFramesMirrored("characters/aiman_walk.png", 4, 2)
        self.idleRight = assets.loadFrames("characters/aiman_idle.png", 4, 1)
        self.idleLeft = assets.loadFramesMirrored("characters/aiman_idle.png", 4, 1)
        self.oyenFrames = assets.loadFrames("characters/oyen_sit.png", 4, 1)

        self.playerX = gameState.streetX
        self.playerY = STREET_FLOOR_Y
        self.facing = 1
        self.walking = False
        self.frame = 0
        self.animationTime = 0.0
        self.frameDuration = 0.09
        self.oyenFrame = 0
        self.oyenTime = 0.0

        # the next stall is to the right, so face it
        nextStall = gameState.nextSceneToPlay()
        if nextStall is not None and STALL_X[nextStall] < self.playerX:
            self.facing = -1

        self.cameraX = clamp(self.playerX - 640, 0, STREET_WIDTH - 1280)
        self.nearStall = None
        self.promptPop = 0.0

        # background crowd
        crowdFrames = assets.loadFrames("characters/crowd_people.png", 6, 1)
        self.walkers = []
        for i in range(8):
            speed = random.choice([-1, 1]) * random.uniform(40, 75)
            self.walkers.append(Walker(crowdFrames[i % 6], random.uniform(0, STREET_WIDTH), speed))

        # lantern strings: each span sags like a hanging rope (a parabola)
        self.lanterns = []
        for i in range(34):
            baseX = -150 + i * 105
            self.lanterns.append({"x": baseX, "phase": random.uniform(0, 6.28),
                                  "light": self.lights.add(0, 0, 90, LANTERN_ORANGE, 0.55, 0.25)})

        self.debugText = arcade.Text("", 640, 548, (170, 255, 190), 13, font_name=FONT_BODY,
                                     anchor_x="center", anchor_y="center")
        self.botStep = 0
        self.botHeldKey = None

    def getIntroLines(self):
        gameState = self.window.gameState
        done = 0
        for sceneId in SCENE_ORDER:
            if gameState.isCompleted(sceneId):
                done += 1
        key = f"street{done}"
        if gameState.flags.get(key):
            return []
        gameState.flags[key] = True
        return PROGRESS_LINES.get(done, [])

    # ------------------------------------------------------------------
    # Movement (the tutorial style: move, then checkBoundary)
    # ------------------------------------------------------------------
    def checkBoundary(self):
        self.playerX = max(60, min(self.playerX, STREET_WIDTH - 60))

    def updatePlay(self, dt):
        direction = 0
        if self.isKeyHeld(arcade.key.LEFT) or self.isKeyHeld(arcade.key.A):
            direction -= 1
        if self.isKeyHeld(arcade.key.RIGHT) or self.isKeyHeld(arcade.key.D):
            direction += 1
        wasWalking = self.walking
        self.walking = direction != 0
        if self.walking:
            self.facing = direction
            self.playerX += direction * PLAYER_WALK_SPEED * dt
            self.checkBoundary()
        if self.walking != wasWalking:
            self.frame = 0
            self.animationTime = 0.0

        # sprite sheet animation: next frame every frameDuration seconds
        self.frameDuration = 0.09 if self.walking else 0.28
        self.animationTime += dt
        if self.animationTime >= self.frameDuration:
            self.animationTime = 0.0
            frameCount = 8 if self.walking else 4
            self.frame = (self.frame + 1) % frameCount
            if self.walking and self.frame in (1, 5):
                name = "footstep_1" if self.frame == 1 else "footstep_2"
                self.playSfx(name, 0.45, x=self.playerX - self.cameraX, pitchVariation=0.08)
                spawnDust(self.particles, self.playerX - self.facing * 20, self.playerY + 4, 2)

        # which stall is Aiman standing in front of?
        self.nearStall = None
        for sceneId in SCENE_ORDER:
            if abs(self.playerX - STALL_X[sceneId]) < ENTER_DISTANCE:
                self.nearStall = sceneId
        if self.nearStall is not None:
            self.promptPop = approach(self.promptPop, 1.0, 8.0, dt)
        else:
            self.promptPop = approach(self.promptPop, 0.0, 10.0, dt)

    def updateAlways(self, dt):
        # the camera follows Aiman smoothly (Week 5-6 translation + Week 7 smoothing)
        target = clamp(self.playerX - 640, 0, STREET_WIDTH - 1280)
        self.cameraX = approach(self.cameraX, target, 5.0, dt)
        self.worldCamera.position = (self.cameraX + 640, 360)

        for walker in self.walkers:
            walker.update(dt)
        self.oyenTime += dt
        if self.oyenTime > 0.3:
            self.oyenTime = 0.0
            self.oyenFrame = (self.oyenFrame + 1) % 4

        # lanterns swing (a sine wave) and their lights follow them
        for lantern in self.lanterns:
            lantern["phase"] += dt * 1.6
            x, y = self.lanternPosition(lantern)
            lantern["light"].x = x
            lantern["light"].y = y - 12
        if random.random() < 0.05:
            lantern = random.choice(self.lanterns)
            x, y = self.lanternPosition(lantern)
            spawnEmbers(self.glowParticles, x, y - 30, 1)

    def lanternPosition(self, lantern):
        """World position of a lantern on its sagging string (foreground parallax)."""
        worldX = lantern["x"] - (LANTERN_FACTOR - 1.0) * self.cameraX
        spanT = ((lantern["x"] + 150) % 420) / 420          # 0..1 along one span of string
        sag = 4 * spanT * (1 - spanT)                        # parabola: 0 at the poles, 1 in the middle
        y = 640 - 40 * sag
        return worldX, y

    # ------------------------------------------------------------------
    # Entering a stall
    # ------------------------------------------------------------------
    def tryEnter(self):
        sceneId = self.nearStall
        if sceneId is None:
            return
        gameState = self.window.gameState
        if not gameState.isUnlocked(sceneId):
            self.playSfx("locked", 0.8)
            self.popups.add("Closed! Finish the stall before it first.", self.playerX - self.cameraX,
                            self.playerY + 230, CREAM, 18, 1.4, 30)
            return
        gameState.streetX = self.playerX
        self.playSfx("door_chime", 0.9)
        self.changeScene(sceneId, self.playerX - self.cameraX, self.playerY + 80)

    def onPlayKeyPress(self, key, modifiers):
        if key in (arcade.key.UP, arcade.key.W, arcade.key.ENTER, arcade.key.RETURN):
            self.tryEnter()

    def onEscape(self):
        self.changeScene("title")

    # ------------------------------------------------------------------
    # Demo bot: walk to the next open stall and go in
    # ------------------------------------------------------------------
    def holdBotKey(self, key):
        if self.botHeldKey == key:
            return
        if self.botHeldKey is not None:
            self.botReleaseKey(self.botHeldKey)
        self.botHeldKey = key
        if key is not None:
            self.botPressKey(key)

    def autoPlay(self, dt):
        if self.dialogue.isActive() or self.sceneEnded:
            self.holdBotKey(None)
            return
        target = self.window.gameState.nextSceneToPlay()
        if target is None:
            self.holdBotKey(None)
            return
        gap = STALL_X[target] - self.playerX
        if gap > 12:
            self.holdBotKey(arcade.key.RIGHT)
        elif gap < -12:
            self.holdBotKey(arcade.key.LEFT)
        else:
            self.holdBotKey(None)
            self.botTimer += dt
            if self.botTimer > 0.6:
                self.botTimer = 0.0
                self.botTapKey(arcade.key.UP)

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------
    def drawLayer(self, texture, factor):
        """Parallax: the layer's left edge sits at cameraX * (1 - factor) in world space."""
        left = self.cameraX * (1.0 - factor)
        assets.drawTexture(texture, left + texture.width / 2, 360, texture.width, 720)

    def drawWorld(self):
        # the sky never moves (it is infinitely far away)
        assets.drawTexture(self.sky, self.cameraX + 640, 360, 1280, 720)
        self.drawLayer(self.farLayer, FAR_FACTOR)
        self.drawLayer(self.midLayer, MID_FACTOR)
        self.drawCrowd()
        self.drawLayer(self.mainLayer, 1.0)
        self.drawStallMarkers()
        # Oyen naps on his crate
        assets.drawTexture(self.oyenFrames[self.oyenFrame], OYEN_X, OYEN_Y, 128, 96)
        self.drawPlayer()

    def drawCrowd(self):
        shift = self.cameraX * (1.0 - CROWD_FACTOR)
        for walker in self.walkers:
            bob = abs(math.sin(walker.phase)) * 4
            angle = math.sin(walker.phase) * 2
            assets.drawTexture(walker.texture, walker.x + shift, 208 + bob, 77, 133, angle, alpha=190,
                               rgb=(150, 150, 190))

    def drawPlayer(self):
        if self.walking:
            frames = self.walkRight if self.facing > 0 else self.walkLeft
            texture = frames[self.frame % 8]
        else:
            frames = self.idleRight if self.facing > 0 else self.idleLeft
            texture = frames[self.frame % 4]
        # soft contact shadow under the feet
        arcade.draw_ellipse_filled(self.playerX, self.playerY + 4, 70, 14, (0, 0, 0, 90))
        assets.drawTexture(texture, self.playerX, self.playerY + PLAYER_HEIGHT / 2 - 6, PLAYER_WIDTH, PLAYER_HEIGHT)

    def drawStallMarkers(self):
        gameState = self.window.gameState
        for sceneId in SCENE_ORDER:
            x = STALL_X[sceneId]
            if gameState.isCompleted(sceneId):
                drawPanel(x, 560, 150, 46, (14, 16, 38, 200), (255, 200, 87, 150), 14, shadow=False)
                drawStars(x, 560, gameState.getStars(sceneId), 3, 32)
            elif not gameState.isUnlocked(sceneId):
                assets.drawTexture(self.lockTexture, x, 560, 46, 46, alpha=220)

    def drawWorldTop(self):
        # foreground lantern strings (drawn after the particles so they are the nearest layer)
        previous = None
        for lantern in self.lanterns:
            x, y = self.lanternPosition(lantern)
            if previous is not None and x - previous[0] < 120:
                arcade.draw_line(previous[0], previous[1] + 30, x, y + 30, (40, 30, 30), 2)
            previous = (x, y)
        for lantern in self.lanterns:
            x, y = self.lanternPosition(lantern)
            swing = math.sin(lantern["phase"] + lantern["x"] * 0.01) * 7
            assets.drawTexture(self.lanternTexture, x, y, 52, 70, swing)

        # the "enter" prompt above the stall
        if self.nearStall is not None and self.promptPop > 0.05:
            self.drawEnterPrompt()

    def drawEnterPrompt(self):
        sceneId = self.nearStall
        x = STALL_X[sceneId]
        y = 470 + 6 * math.sin(self.sceneTime * 4)
        scale = 0.7 + 0.3 * self.promptPop
        gameState = self.window.gameState
        if gameState.isUnlocked(sceneId):
            drawPanel(x, y, 360 * scale, 58 * scale, (14, 16, 38, 230), GOLD, 16)
            drawText(f"UP - Enter {STALL_NAMES[sceneId]}", x, y, CREAM, 15, FONT_BODY, anchorX="center",
                     anchorY="center", bold=True, alpha=255 * self.promptPop)
            assets.drawTexture(assets.loadTexture("particles/light_glow.png"), x, 300, 520, 420,
                               rgb=WARM_LIGHT, alpha=70 * self.promptPop, additive=True)
        else:
            drawPanel(x, y, 300 * scale, 50 * scale, (40, 20, 24, 220), (200, 120, 120), 14)
            drawText("Closed for now", x, y, CREAM, 14, FONT_BODY, anchorX="center", anchorY="center",
                     alpha=255 * self.promptPop)

    def drawPlayHud(self):
        gameState = self.window.gameState
        drawPanel(170, 664, 300, 76, (14, 16, 38, 205), (255, 200, 87, 150), 16, shadow=False)
        drawText("PELITA STREET", 40, 682, GOLD, 18, FONT_TITLE, anchorY="center")
        drawText(f"Stars {gameState.totalStars()} / 12", 40, 652, CREAM, 14, FONT_BODY, anchorY="center")
        assets.drawTexture(assets.loadTexture("ui/icon_ladle.png"), 170, 652, 26, 26)
        drawProgressBar(188, 645, 118, 14, gameState.auraFraction(), GOLD)
        nextStall = gameState.nextSceneToPlay()
        if nextStall is not None:
            arrow = ">>" if STALL_X[nextStall] > self.playerX else "<<"
            drawPanel(640, 672, 460, 44, (14, 16, 38, 205), (255, 200, 87, 110), 14, shadow=False)
            drawText(f"Next: {STALL_NAMES[nextStall]}  {arrow}", 640, 672, CREAM, 15, FONT_BODY,
                     anchorX="center", anchorY="center", bold=True)
        drawKeyHint(470, 28, "LEFT/RIGHT", "Walk")
        drawKeyHint(700, 28, "UP", "Enter stall")

    def drawDebugWorld(self):
        """F2: layer edges and the camera, to show how the parallax works."""
        for texture, factor, colour in ((self.farLayer, FAR_FACTOR, (120, 180, 255)),
                                        (self.midLayer, MID_FACTOR, (255, 200, 87)),
                                        (self.mainLayer, 1.0, (80, 255, 120))):
            left = self.cameraX * (1.0 - factor)
            arcade.draw_line(left, 0, left, 720, colour, 2)
            arcade.draw_line(left + texture.width, 0, left + texture.width, 720, colour, 2)
        arcade.draw_lbwh_rectangle_outline(self.playerX - 40, self.playerY, 80, 150, (255, 120, 255), 1)
        for sceneId in SCENE_ORDER:
            x = STALL_X[sceneId]
            arcade.draw_lbwh_rectangle_outline(x - ENTER_DISTANCE, STREET_FLOOR_Y - 10, ENTER_DISTANCE * 2, 20,
                                               (255, 160, 80), 1)
        self.debugText.x = self.cameraX + 640
        self.debugText.text = (f"cameraX {self.cameraX:6.1f}   playerX {self.playerX:6.1f}   parallax far {FAR_FACTOR} "
                               f"mid {MID_FACTOR} crowd {CROWD_FACTOR} main 1.0 lanterns {LANTERN_FACTOR}")
        # a dark strip behind the text keeps it readable on busy backgrounds
        width = self.debugText.content_width
        arcade.draw_lbwh_rectangle_filled(self.debugText.x - width / 2 - 8, self.debugText.y - 12, width + 16, 24,
                                          (0, 0, 0, 180))
        self.debugText.draw()
