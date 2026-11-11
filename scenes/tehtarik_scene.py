"""
tehtarik_scene.py
-----------------
Scene 3 - "The Great Pull" at the Sri Pelita Cafe.

Aiman makes three pulled drinks (teh tarik, kopi tarik, milo tarik) for
three regulars. Each drink has two steps:
  A) MIX  - hold to pour condensed milk until the colour matches the target
  B) PULL - move the jug with the mouse and hold to pour a long stream into
            the glass. The higher the pull, the thicker the foam.

Techniques used (lecture links):
  * Week 1-2 - RGB colour model: the drink colour is a linear interpolation
              between the dark tea colour and the milky colour
  * Week 2  - media capture: optional webcam "pull" mode (OpenCV frame
              differencing in core/motion_tracker.py, press T)
  * Week 3  - mouse press / motion / release events
  * Week 5-6 - rotation of the jug around its centre (the spout follows)
  * Week 8  - particle system: a fluid stream of droplets with gravity,
              splashes, foam bubbles and steam; flickering neon glow
  * Audio   - the pour sound's pitch rises as the glass fills (like a real
              glass: the air space above the drink gets smaller)
"""
import math
import random

import arcade

from core import assets
from core.base_scene import BaseScene
from core.helpers import Tween, approach, clamp, easeOutBack, lerp, lerpColor
from core.particles import spawnBubble, spawnDroplet, spawnHearts, spawnJuiceSplash, spawnSparkles, spawnSteam
from core.ui import Button, drawKeyHint, drawPanel, drawProgressBar, drawRoundedRect, drawText, withAlpha
from settings import (BAD_RED, CREAM, DISH_NAMES, FONT_BODY, FONT_NARROW, GOLD, GOOD_GREEN, SCENE_NAMES,
                      SPEAKERS, STALL_NAMES, TEAL)

# ----------------------------------------------------------------------
# Layout
# ----------------------------------------------------------------------
GLASS_X = 640
GLASS_Y = 290               # centre of glass_front.png (120 x 200), standing on the counter
COUNTER_Y = 190             # top edge of the steel counter
LIQUID_LEFT = GLASS_X - 60 + 14      # inside of the glass (from the asset notes)
LIQUID_RIGHT = GLASS_X - 60 + 106
LIQUID_BOTTOM = COUNTER_Y + 16
LIQUID_HEIGHT = 174
MOUTH_Y = LIQUID_BOTTOM + LIQUID_HEIGHT

JUG_REST_X = 455            # where the jug waits during mixing
JUG_REST_Y = 272
SPOUT_OFFSET = (67, 58)     # spout tip relative to the jug centre (from the asset notes)
POUR_ANGLE = 38             # the jug tilts this far (clockwise) to pour
NEON_X = 640
NEON_Y = 596

FILL_MIN = 0.75
FILL_MAX = 0.95
FILL_TARGET = 0.85
DROPLETS_PER_SECOND = 75
DROP_AMOUNT = 0.0031        # how much each droplet fills the glass

# Orders: customer, drink, dark base colour, fully milky colour, perfect milk amount
ORDERS = [
    ("lim", "Teh Tarik", (70, 35, 15), (228, 188, 142), 0.62),
    ("kavitha", "Kopi Tarik", (45, 28, 18), (214, 168, 124), 0.55),
    ("farah", "Milo Tarik", (92, 56, 36), (205, 152, 112), 0.50),
]
ORDER_LINES = {
    "lim": "One teh tarik! Pull it high, with thick foam!",
    "kavitha": "Kopi tarik please, dear. Not too sweet.",
    "farah": "One Milo tarik! Make it pretty, I'm taking a photo!",
}

MAX_SCORE = 1560


def rotateOffset(offsetX, offsetY, angleDegrees):
    """2D rotation matrix for Arcade's clockwise angles (Week 5-6)."""
    t = math.radians(angleDegrees)
    return (offsetX * math.cos(t) + offsetY * math.sin(t),
            -offsetX * math.sin(t) + offsetY * math.cos(t))


class TehTarikScene(BaseScene):
    sceneId = "tehtarik"
    sceneTitle = "Scene 3 - " + SCENE_NAMES["tehtarik"]
    stallName = STALL_NAMES["tehtarik"]
    dishName = DISH_NAMES["tehtarik"]
    musicName = "music_tehtarik"
    auraMusicName = "music_tehtarik_aura"
    ambienceName = "amb_mamak"
    timeLimit = 100
    maxScore = MAX_SCORE
    isGameplayScene = True
    controlsHelp = [("HOLD", "Mix: hold the mouse (or SPACE) to pour milk, let go on the target colour"),
                    ("MOUSE", "Pull: move the jug; hold to pour into the glass"),
                    ("ENTER", "Serve the drink (or click SERVE)"),
                    ("T", "Webcam pull mode: raise and lower your hand")]

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def setupScene(self):
        self.background = assets.loadTexture("backgrounds/mamak_bg.png")
        self.glassTexture = assets.loadTexture("props/glass_front.png")
        self.jugTexture = assets.loadTexture("props/jug.png")
        self.canTexture = assets.loadTexture("props/milk_can.png")
        self.neonTexture = assets.loadTexture("props/neon_24jam.png")
        self.neonGlow = assets.loadTexture("props/neon_24jam_glow.png")
        self.stainTexture = assets.loadTexture("particles/soft_circle.png")
        self.dishTexture = assets.loadTexture("food/dish_tehtarik.png")
        self.portraits = {}
        for order in ORDERS:
            speaker = order[0]
            self.portraits[speaker] = assets.loadTexture(SPEAKERS[speaker]["portrait"])

        self.orderIndex = 0
        self.stains = arcade.SpriteList()
        self.droplets = []                # stream droplets we follow for collisions
        self.pourLoop = None
        self.results = []                 # score of every drink (for the result card)
        self.messCount = 0
        self.totalMess = 0

        # jug (follows the mouse smoothly)
        self.jugX = JUG_REST_X
        self.jugY = JUG_REST_Y
        self.jugAngle = 0.0
        self.targetX = JUG_REST_X
        self.targetY = JUG_REST_Y
        self.lastSpoutX = 0.0

        # neon sign flicker
        self.neonOn = True
        self.neonTimer = random.uniform(3, 6)
        self.neonFlickerTime = 0.0
        self.neonBuzz = None

        # optional webcam motion tracking (Week 2 media capture + image processing)
        self.tracker = None
        self.previewTexture = None
        self.previewTimer = 0.0
        self.simulatedCamera = None      # only with --fake-camera (testing and the demo video)
        self.webcamTime = 0.0

        self.serveButton = Button("SERVE", GLASS_X, 110, 200, 54, 20)
        self.levelText = arcade.Text("", LIQUID_RIGHT + 76, 300, CREAM, 13, font_name=FONT_BODY,
                                     anchor_x="center", anchor_y="center")
        self.debugText = arcade.Text("", 640, 548, (170, 255, 190), 13, font_name=FONT_BODY,
                                     anchor_x="center", anchor_y="center")

        # demo bot
        self.botStage = "start"
        self.botTimer = 0.0
        self.botPullTime = 0.0

        self.startOrder()

    def getIntroLines(self):
        return [
            ("lim", "Hey, Aiman! One teh tarik, please. Pull it high. I want thick foam!"),
            ("nenek", "First the condensed milk: hold the mouse and let go when the colour matches the target."),
            ("aiman", "Then I move the jug with the mouse and hold to pour. The higher the pull, the more foam!"),
            ("nenek", "Fill between the MIN and MAX lines, then SERVE. And do not splash Muthu's counter!"),
        ]

    def onPlayStart(self):
        self.neonBuzz = self.startLoop("neon_buzz", 0.06)

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------
    def startOrder(self):
        speaker, drink, base, milky, perfectMilk = ORDERS[self.orderIndex]
        self.speaker = speaker
        self.drinkName = drink
        self.baseColor = base
        self.milkyColor = milky
        self.targetMilk = perfectMilk
        self.targetColor = lerpColor(base, milky, perfectMilk)
        self.phase = "mix"                # mix -> pull -> served
        self.milkAmount = 0.0
        self.milkPouring = False
        self.mixScore = 0
        self.fill = 0.0
        self.foam = 0.0
        self.pouring = False
        self.mess = 0
        self.glassX = GLASS_X - 700.0     # a fresh glass slides in from the left
        self.glassTween = Tween(GLASS_X - 700, GLASS_X, 0.6, easeOutBack)
        self.reaction = None
        self.reactionTimer = 0.0
        self.bubbleTimer = 0.0
        self.steamTimer = 0.0
        self.targetX = JUG_REST_X
        self.targetY = JUG_REST_Y
        self.popups.add(f"{drink.upper()} for {SPEAKERS[speaker]['name']}", 640, 470, GOLD, 24, 1.6, 30)

    def drinkColor(self):
        # RGB linear interpolation: dark tea -> milky colour (Week 1-2 colour model + Week 7 lerp)
        return lerpColor(self.baseColor, self.milkyColor, clamp(self.milkAmount, 0.0, 1.0))

    def colourDistance(self):
        a = self.drinkColor()
        b = self.targetColor
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)

    def finishMixing(self):
        """The player let go of the milk: judge how close the colour is."""
        self.milkPouring = False
        gap = self.colourDistance()
        if gap <= 12:
            quality = "perfect"
            label = "PERFECT COLOUR!"
        elif gap <= 30:
            quality = "good"
            label = "NICE COLOUR"
        else:
            quality = "miss"
            label = "TOO DARK" if self.milkAmount < self.targetMilk else "TOO MILKY"
        self.registerHit(quality, JUG_REST_X, JUG_REST_Y + 120, label)
        self.mixScore = int(50 * clamp(1 - gap / 80, 0.0, 1.0))
        self.phase = "pull"
        self.popups.add("Now PULL! Move the jug and hold to pour", 640, 470, CREAM, 20, 1.8, 20)

    def serveDrink(self):
        if self.phase != "pull" or self.fill < 0.3:
            return
        self.stopPouring()
        self.phase = "served"
        foamRatio = 0.0
        if self.fill > 0:
            foamRatio = self.foam / self.fill
        fillScore = 150 * clamp(1 - abs(self.fill - FILL_TARGET) / 0.25, 0.0, 1.0)
        foamScore = 100 * clamp(foamRatio / 0.5, 0.0, 1.0)
        messPenalty = min(60, self.mess * 3)
        total = int(fillScore + foamScore + self.mixScore - messPenalty)
        self.results.append(total)
        self.totalMess += self.mess
        if total >= 230:
            quality = "perfect"
            self.reaction = "love"
        elif total >= 150:
            quality = "good"
            self.reaction = "happy"
        else:
            quality = "miss"
            self.reaction = "sad"
        self.registerHit(quality, GLASS_X, MOUTH_Y + 40, f"{self.drinkName.upper()} +{total}")
        self.addScore(total)
        self.glassTween = Tween(GLASS_X, 1000, 0.7, easeOutBack)
        self.reactionTimer = 0.0
        self.playSfx("glass_clink", 0.9, x=GLASS_X)

    # ------------------------------------------------------------------
    # Pouring
    # ------------------------------------------------------------------
    def startPouring(self):
        if self.phase == "mix":
            self.milkPouring = True
            self.pourLoopStart(0.7)
        elif self.phase == "pull":
            self.pouring = True
            self.pourLoopStart(0.8 + 0.8 * self.fill)

    def pourLoopStart(self, speed):
        if self.pourLoop is None:
            self.pourLoop = self.startLoop("pour_loop", 0.55, speed)

    def stopPouring(self):
        self.pouring = False
        if self.pourLoop is not None:
            self.window.sound.stop(self.pourLoop)
            self.pourLoop = None
        if self.phase == "mix" and self.milkPouring:
            self.finishMixing()

    def spoutPosition(self):
        """Where the stream leaves the jug: the spout offset rotated with the jug."""
        dx, dy = rotateOffset(SPOUT_OFFSET[0], SPOUT_OFFSET[1], self.jugAngle)
        return self.jugX + dx, self.jugY + dy

    def pullHeight(self):
        spoutX, spoutY = self.spoutPosition()
        return spoutY - MOUTH_Y

    # ------------------------------------------------------------------
    # Input
    # ------------------------------------------------------------------
    def setHandTarget(self, x, y):
        """The spout tip follows the mouse, so the jug centre sits left of it."""
        tipX, tipY = rotateOffset(SPOUT_OFFSET[0], SPOUT_OFFSET[1], POUR_ANGLE)
        self.targetX = clamp(x - tipX, 260, 980)
        self.targetY = clamp(y - tipY, MOUTH_Y + 30, 560)

    def onPlayMousePress(self, x, y, button):
        if button != arcade.MOUSE_BUTTON_LEFT:
            return
        if self.phase == "pull" and self.serveButton.contains(x, y):
            self.serveDrink()
            return
        self.startPouring()

    def onPlayMouseRelease(self, x, y, button):
        if button == arcade.MOUSE_BUTTON_LEFT:
            self.stopPouring()

    def onPlayMouseMotion(self, x, y, dx, dy):
        self.serveButton.updateHover(x, y)
        if self.phase == "pull":
            self.moveHand(x, y)

    def onPlayMouseDrag(self, x, y, dx, dy):
        if self.phase == "pull":
            self.moveHand(x, y)

    def moveHand(self, x, y):
        """Mouse mode: the mouse moves the jug. Webcam mode: the camera sets the height, the mouse only left/right."""
        if self.tracker is None:
            self.setHandTarget(x, y)
        else:
            tipX, tipY = rotateOffset(SPOUT_OFFSET[0], SPOUT_OFFSET[1], POUR_ANGLE)
            self.targetX = clamp(x - tipX, 260, 980)

    def onPlayKeyPress(self, key, modifiers):
        if key == arcade.key.SPACE:
            self.startPouring()
        elif key == arcade.key.ENTER or key == arcade.key.RETURN:
            self.serveDrink()
        elif key == arcade.key.T:
            self.toggleWebcam()

    def onPlayKeyRelease(self, key, modifiers):
        if key == arcade.key.SPACE:
            self.stopPouring()

    def toggleWebcam(self):
        useFakeCamera = self.window.options.fakeCamera
        if (self.window.autoplay or self.window.recording) and not useFakeCamera:
            return       # never open a real camera in demo / record mode
        if self.tracker is not None:
            self.tracker.stop()
            self.tracker = None
            self.simulatedCamera = None
            self.previewTexture = None
            self.popups.add("Webcam off - mouse control", 640, 470, CREAM, 20, 1.4, 20)
            return
        from core import motion_tracker
        tracker = motion_tracker.MotionTracker()
        if useFakeCamera and motion_tracker.OPENCV_AVAILABLE:
            # a pretend webcam feeds the same tracking code; no real camera is opened
            self.simulatedCamera = motion_tracker.SimulatedCamera()
            self.webcamTime = 0.0
            self.tracker = tracker
            self.playSfx("webcam_on", 0.7)
            self.popups.add("Webcam pull mode (simulated camera)", 640, 470, TEAL, 20, 1.6, 20)
        elif tracker.start():
            self.tracker = tracker
            self.playSfx("webcam_on", 0.7)
            self.popups.add("Webcam pull mode: raise your hand!", 640, 470, TEAL, 20, 1.6, 20)
        else:
            self.popups.add("No webcam found - using mouse", 640, 470, BAD_RED, 20, 1.6, 20)

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------
    def updateAlways(self, dt):
        self.updateNeon(dt)
        for stain in list(self.stains):
            stain.life -= dt
            if stain.life <= 0:
                stain.remove_from_sprite_lists()
            else:
                stain.alpha = int(140 * clamp(stain.life / 3.0, 0.0, 1.0))

    def updateNeon(self, dt):
        """The old neon sign sometimes flickers (random on/off bursts) with a buzz."""
        self.neonTimer -= dt
        if self.neonTimer <= 0:
            self.neonFlickerTime = random.uniform(0.35, 0.7)
            self.neonTimer = random.uniform(4, 9)
            self.playSfx("neon_buzz", 0.35, cooldown=0.5)
        if self.neonFlickerTime > 0:
            self.neonFlickerTime -= dt
            self.neonOn = random.random() > 0.45
        else:
            self.neonOn = True

    def updatePlay(self, dt):
        self.glassTween.update(dt)
        self.glassX = self.glassTween.value

        if self.tracker is not None:
            self.updateWebcam(dt)

        if self.phase == "mix":
            # jug waits on the counter while the milk goes in
            self.targetX = JUG_REST_X
            self.targetY = JUG_REST_Y
            if self.milkPouring:
                self.milkAmount = min(1.0, self.milkAmount + 0.3 * dt)
                # a thick white stream from the can into the jug
                spawnDroplet(self.particles, JUG_REST_X - 2 + random.uniform(-3, 3), JUG_REST_Y + 150,
                             random.uniform(-6, 6), -60, (250, 246, 236), 12, 0.2)

        # smooth hand movement (exponential smoothing = interpolation over time)
        self.jugX = approach(self.jugX, self.targetX, 12.0, dt)
        self.jugY = approach(self.jugY, self.targetY, 12.0, dt)
        wantedAngle = POUR_ANGLE if (self.phase == "pull" and self.pouring) else 0.0
        if self.phase == "pull" and not self.pouring:
            wantedAngle = 8.0
        self.jugAngle = approach(self.jugAngle, wantedAngle, 14.0, dt)

        if self.phase == "pull":
            if self.pouring:
                self.emitStream(dt)
            self.updatePourSound()
            if self.fill > 1.08:
                self.popups.add("OVERFLOW!", GLASS_X, MOUTH_Y + 60, BAD_RED, 26, 1.0, 30)
                self.serveDrink()
        self.updateDroplets()
        self.updateGlassEffects(dt)

        if self.phase == "served":
            self.reactionTimer += dt
            if self.reactionTimer > 0.7 and self.reaction is not None:
                self.showReaction()
            if self.reactionTimer > 2.4:
                self.orderIndex += 1
                if self.orderIndex >= len(ORDERS):
                    self.phase = "done"
                    bonus = int(self.timeLeft) * 3
                    if bonus > 0:
                        self.score += bonus
                        self.popups.add(f"TIME BONUS +{bonus}", 640, 470, (120, 190, 255), 28, 1.5, 50)
                    self.finishScene("Frothy, sweet, pulled sky-high!")
                else:
                    self.startOrder()

    def emitStream(self, dt):
        spoutX, spoutY = self.spoutPosition()
        colour = self.drinkColor()
        # how fast the hand moves adds to the droplet speed (inertia)
        handSpeed = (spoutX - self.lastSpoutX) / max(dt, 0.001)
        self.lastSpoutX = spoutX
        count = DROPLETS_PER_SECOND * dt
        whole = int(count)
        if random.random() < count - whole:
            whole += 1
        for i in range(whole):
            shade = random.uniform(0.9, 1.08)
            dropColour = (int(clamp(colour[0] * shade, 0, 255)), int(clamp(colour[1] * shade, 0, 255)),
                          int(clamp(colour[2] * shade, 0, 255)))
            droplet = spawnDroplet(self.particles, spoutX + random.uniform(-2, 2), spoutY,
                                   handSpeed * 0.15 + random.uniform(-10, 10), random.uniform(-70, -40),
                                   dropColour, random.uniform(9, 13), 2.2)
            self.droplets.append(droplet)

    def updateDroplets(self):
        """Follow every stream droplet: into the glass, or a splash on the counter."""
        surfaceY = LIQUID_BOTTOM + min(1.0, self.fill) * LIQUID_HEIGHT + self.foamHeight()
        keep = []
        for droplet in self.droplets:
            if len(droplet.sprite_lists) == 0:
                continue                       # it already died of old age
            x = droplet.center_x
            y = droplet.center_y
            insideMouth = LIQUID_LEFT + 4 <= x <= LIQUID_RIGHT - 4 and self.phase == "pull"
            if insideMouth and y <= min(surfaceY, MOUTH_Y) + 4:
                # absorbed by the drink: more fill, and foam from the fall height
                self.fill += DROP_AMOUNT
                height = clamp((self.pullHeight() - 60) / 300, 0.0, 1.0)
                self.foam += DROP_AMOUNT * height
                droplet.remove_from_sprite_lists()
                continue
            if y <= COUNTER_Y + 2 and not (GLASS_X - 62 <= x <= GLASS_X + 62 and self.phase == "pull"):
                self.splash(x, droplet)
                continue
            if y < COUNTER_Y - 40:
                droplet.remove_from_sprite_lists()
                continue
            keep.append(droplet)
        self.droplets = keep

    def splash(self, x, droplet):
        droplet.remove_from_sprite_lists()
        self.mess += 1
        if self.mess % 3 == 1:
            spawnJuiceSplash(self.particles, x, COUNTER_Y + 4, self.drinkColor(), 5, 0.45)
            stain = arcade.Sprite(self.stainTexture)
            stain.center_x = x + random.uniform(-10, 10)
            stain.center_y = random.uniform(168, 186)
            stain.scale = (random.uniform(26, 44) / 64, random.uniform(6, 10) / 64)
            colour = self.drinkColor()
            stain.color = (colour[0], colour[1], colour[2], 140)
            stain.life = 5.0
            self.stains.append(stain)
            self.playSfx("splash_" + str(random.randint(1, 2)), 0.6, x=x, pitchVariation=0.15, cooldown=0.12)
        if self.mess == 12:
            self.popups.add("Careful! You're splashing!", x, 260, BAD_RED, 20, 1.2, 30)

    def updatePourSound(self):
        if self.pourLoop is None:
            return
        # Audio effect: the pitch rises from 0.8 to 1.6 as the glass fills
        self.window.sound.setSpeed(self.pourLoop, 0.8 + 0.8 * clamp(self.fill, 0.0, 1.0))
        spoutX, spoutY = self.spoutPosition()
        self.window.sound.setPan(self.pourLoop, self.window.sound.panFromX(spoutX))

    def foamHeight(self):
        return min(40.0, self.foam * 70)

    def updateGlassEffects(self, dt):
        if self.fill <= 0.05:
            return
        surfaceY = LIQUID_BOTTOM + min(1.0, self.fill) * LIQUID_HEIGHT
        # foam bubbles appear on the surface and pop
        self.bubbleTimer -= dt
        if self.foam > 0.05 and self.bubbleTimer <= 0:
            self.bubbleTimer = 0.12
            spawnBubble(self.particles, self.glassX + random.uniform(-40, 40), surfaceY + self.foamHeight() * 0.6)
            if random.random() < 0.15:
                self.playSfx("bubble_pop", 0.25, x=self.glassX, pitchVariation=0.25, cooldown=0.25)
        # hot drink: steam wisps rise from the top
        self.steamTimer -= dt
        if self.steamTimer <= 0:
            self.steamTimer = 0.3
            spawnSteam(self.particles, self.glassX, MOUTH_Y + 6, 40)

    def showReaction(self):
        x = 1110
        y = 420
        voice = SPEAKERS[self.speaker]["voice"]
        pitch = SPEAKERS[self.speaker]["pitch"]
        self.playSfx("slurp", 0.8, x=x)
        if self.reaction == "love":
            spawnHearts(self.particles, x, y, 8)
            self.popups.add("DELICIOUS!", x - 30, y + 40, GOLD, 22, 1.4, 30)
            self.playSfx("happy_chime", 0.7, x=x)
        elif self.reaction == "happy":
            spawnSparkles(self.glowParticles, x, y, GOOD_GREEN, 12, 180)
            self.popups.add("Not bad!", x - 30, y + 40, GOOD_GREEN, 22, 1.4, 30)
        else:
            self.popups.add("Hmm... needs work", x - 30, y + 40, BAD_RED, 20, 1.4, 30)
            self.playSfx("grumble", 0.7, x=x)
        self.playSfx(voice, 0.8, speed=pitch)
        self.reaction = None

    def updateWebcam(self, dt):
        if self.simulatedCamera is not None:
            self.webcamTime += dt
            self.tracker.feedFrame(self.simulatedCamera.frameAt(self.webcamTime))
        height = self.tracker.getHandHeight()
        if height is not None and self.phase == "pull":
            self.targetY = lerp(MOUTH_Y + 30, 560, height)
        self.previewTimer -= dt
        if self.previewTimer <= 0:
            self.previewTimer = 0.2
            image = self.tracker.getPreviewImage()
            if image is not None:
                self.previewTexture = assets.textureFromImage(image, "webcam")

    def cleanupScene(self):
        if self.tracker is not None:
            self.tracker.stop()
            self.tracker = None

    # ------------------------------------------------------------------
    # Demo bot
    # ------------------------------------------------------------------
    def autoPlay(self, dt):
        if self.dialogue.isActive():
            return
        if self.botTimer > 0:
            self.botTimer -= dt
            return
        if self.phase == "mix" and self.glassTween.done:
            if not self.milkPouring:
                if self.botMoveCursor(JUG_REST_X, JUG_REST_Y + 160, dt, 1100):
                    self.botMouseDown()
            else:
                # let go near the target (the second drink goes a little too milky on purpose)
                wanted = self.targetMilk + (0.07 if self.orderIndex == 1 else 0.0)
                if self.milkAmount >= wanted:
                    self.botMouseUp()
                    self.botTimer = 0.6
                    self.botStage = "aim"
        elif self.phase == "pull":
            if self.window.options.fakeCamera and self.tracker is None:
                self.botTapKey(arcade.key.T)      # show webcam mode with the simulated camera
                self.botTimer = 0.8
                return
            if self.botStage == "aim":
                if self.botMoveCursor(GLASS_X, MOUTH_Y + 70, dt, 900):
                    self.botMouseDown()
                    self.botStage = "pull"
                    self.botPullTime = 0.0
            elif self.botStage == "pull":
                self.botPullTime += dt
                # raise the jug higher and higher while pouring (a "sky-high" pull)
                # (the jug cannot go higher than its clamp, so the cursor stops there too)
                height = MOUTH_Y + 70 + min(120, self.botPullTime * 75)
                wobble = math.sin(self.botPullTime * 5) * 3
                if self.orderIndex == 1 and 1.4 < self.botPullTime < 1.75:
                    wobble = 70          # a deliberate drift: the stream misses and splashes
                self.botMoveCursor(GLASS_X + wobble, height, dt, 500)
                if self.fill >= 0.87:
                    self.botMouseUp()
                    self.botStage = "serve"
                    self.botTimer = 0.4
            elif self.botStage == "serve":
                if self.botMoveCursor(self.serveButton.cx, self.serveButton.cy, dt, 1100):
                    self.botClick()
                    self.botStage = "wait"
        elif self.phase == "served":
            self.botStage = "start"

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------
    def getResultLines(self):
        drinks = "   ".join(f"{ORDERS[i][1]} {self.results[i]}" for i in range(len(self.results)))
        return [drinks, f"Splashes on the counter: {self.totalMess}"]

    def getDishTexture(self):
        return self.dishTexture

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------
    def drawWorld(self):
        assets.drawTexture(self.background, 640, 360, 1280, 720)
        self.drawNeon()
        self.stains.draw()
        self.drawGlass()
        if self.phase == "mix":
            self.drawMilkCan()
        self.drawJug()

    def drawNeon(self):
        assets.drawTexture(self.neonTexture, NEON_X, NEON_Y, 360, 110)
        if self.neonOn:
            pulse = 0.85 + 0.15 * math.sin(self.sceneTime * 3)
            assets.drawTexture(self.neonGlow, NEON_X, NEON_Y, 380, 120, alpha=255 * pulse, additive=True)

    def drawGlass(self):
        x = self.glassX
        left = x - 60 + 14
        width = 92
        if self.fill > 0:
            height = min(1.0, self.fill) * LIQUID_HEIGHT
            colour = self.drinkColor()
            arcade.draw_lbwh_rectangle_filled(left, LIQUID_BOTTOM, width, height, (colour[0], colour[1], colour[2], 240))
            # a darker band at the bottom and a lighter band at the top give depth
            arcade.draw_lbwh_rectangle_filled(left, LIQUID_BOTTOM, width, min(height, 18), (0, 0, 0, 40))
            foamH = self.foamHeight()
            if foamH > 1:
                arcade.draw_lbwh_rectangle_filled(left, LIQUID_BOTTOM + height, width, foamH, (246, 232, 206, 250))
                arcade.draw_lbwh_rectangle_filled(left, LIQUID_BOTTOM + height + foamH - 4, width, 4,
                                                  (255, 250, 240, 255))
        assets.drawTexture(self.glassTexture, x, GLASS_Y, 120, 200)
        if self.phase == "pull":
            # MIN and MAX lines on the glass
            for level, label in ((FILL_MIN, "MIN"), (FILL_MAX, "MAX")):
                y = LIQUID_BOTTOM + level * LIQUID_HEIGHT
                arcade.draw_line(left - 4, y, left + width + 4, y, withAlpha(GOLD, 200), 2)
                drawText(label, left + width + 10, y, GOLD, 11, FONT_NARROW, anchorY="center")

    def drawMilkCan(self):
        tilt = 40 if self.milkPouring else 20
        assets.drawTexture(self.canTexture, JUG_REST_X - 40, JUG_REST_Y + 190, 90, 120, tilt)

    def drawJug(self):
        assets.drawTexture(self.jugTexture, self.jugX, self.jugY, 150, 170, self.jugAngle)
        if self.phase == "mix":
            # the liquid surface inside the jug opening shows the current colour
            colour = self.drinkColor()
            arcade.draw_ellipse_filled(self.jugX + 7, self.jugY + 51, 58, 12, colour)

    def drawWorldTop(self):
        if self.phase == "pull":
            self.serveButton.enabled = self.fill >= 0.3
            self.serveButton.draw()

    def drawPlayHud(self):
        self.drawCustomer()
        if self.phase == "mix":
            self.drawColourPanel()
        elif self.phase == "pull":
            self.drawPullPanel()
        if self.previewTexture is not None:
            # under the customer card, so it never covers the order bubble or the glass
            drawPanel(1110, 112, 176, 128, (14, 16, 38, 220), TEAL, 12, shadow=False)
            assets.drawTexture(self.previewTexture, 1110, 104, 144, 108)
            drawText("WEBCAM (motion)", 1110, 167, TEAL, 11, FONT_BODY, anchorX="center", anchorY="center",
                     bold=True)
        drawKeyHint(80, 28, "HOLD", "Pour")
        drawKeyHint(250, 28, "ENTER", "Serve")
        drawKeyHint(960, 28, "T", "Webcam mode")

    def drawCustomer(self):
        speaker = self.speaker
        drawPanel(1110, 330, 250, 300, (14, 16, 38, 215), SPEAKERS[speaker]["color"], 18, shadow=True)
        assets.drawTexture(self.portraits[speaker], 1110, 350, 200, 200)
        drawText(SPEAKERS[speaker]["name"], 1110, 220, SPEAKERS[speaker]["color"], 16, FONT_BODY,
                 anchorX="center", anchorY="center", bold=True)
        drawText(f"Order: {self.drinkName}", 1110, 196, CREAM, 14, FONT_BODY, anchorX="center", anchorY="center")
        # speech bubble with the order
        drawRoundedRect(1036, 522, 420, 46, (250, 244, 226, 235), 14)
        drawText(ORDER_LINES[speaker], 1036, 522, (60, 40, 30), 12, FONT_BODY, anchorX="center",
                 anchorY="center", shadow=False)

    def drawColourPanel(self):
        drawPanel(180, 430, 300, 150, (14, 16, 38, 215), (255, 200, 87, 150), 16, shadow=False)
        drawText("MIX THE MILK", 180, 486, GOLD, 15, FONT_BODY, anchorX="center", anchorY="center", bold=True)
        current = self.drinkColor()
        drawRoundedRect(110, 430, 90, 56, current, 10)
        drawRoundedRect(250, 430, 90, 56, self.targetColor, 10)
        drawText("NOW", 110, 390, CREAM, 12, FONT_BODY, anchorX="center", anchorY="center")
        drawText("TARGET", 250, 390, CREAM, 12, FONT_BODY, anchorX="center", anchorY="center")
        gap = self.colourDistance()
        if gap <= 12:
            colour = GOLD
        elif gap <= 30:
            colour = GOOD_GREEN
        else:
            colour = CREAM
        drawText("=", 180, 430, colour, 26, FONT_BODY, anchorX="center", anchorY="center", bold=True)

    def drawPullPanel(self):
        drawPanel(180, 430, 300, 150, (14, 16, 38, 215), (255, 200, 87, 150), 16, shadow=False)
        drawText("PULL HEIGHT", 180, 486, GOLD, 15, FONT_BODY, anchorX="center", anchorY="center", bold=True)
        height = clamp(self.pullHeight(), 0, 400)
        drawProgressBar(60, 446, 240, 14, height / 360, (120, 190, 255))
        foamRatio = self.foam / self.fill if self.fill > 0 else 0.0
        drawText("FOAM", 60, 420, CREAM, 12, FONT_BODY, anchorY="center")
        drawProgressBar(110, 413, 190, 12, clamp(foamRatio / 0.5, 0.0, 1.0), (246, 232, 206))
        drawText("FILL", 60, 394, CREAM, 12, FONT_BODY, anchorY="center")
        fillColour = GOOD_GREEN if FILL_MIN <= self.fill <= FILL_MAX else (255, 200, 120)
        drawProgressBar(110, 387, 190, 12, clamp(self.fill, 0.0, 1.0), fillColour)
        self.levelText.text = f"{int(self.fill * 100)}%"
        self.levelText.draw()

    def drawDebugWorld(self):
        """F2: glass mouth hit box, pull height ruler and droplet velocity vectors."""
        arcade.draw_lbwh_rectangle_outline(LIQUID_LEFT, LIQUID_BOTTOM, LIQUID_RIGHT - LIQUID_LEFT, LIQUID_HEIGHT,
                                           (80, 255, 120), 1)
        spoutX, spoutY = self.spoutPosition()
        arcade.draw_line(spoutX + 20, MOUTH_Y, spoutX + 20, spoutY, (255, 120, 255), 2)
        arcade.draw_circle_filled(spoutX, spoutY, 4, (255, 120, 255))
        for droplet in self.droplets[-40:]:
            arcade.draw_line(droplet.center_x, droplet.center_y, droplet.center_x + droplet.vx * 0.05,
                             droplet.center_y + droplet.vy * 0.05, (120, 180, 255), 1)
        self.debugText.text = (f"pull height {self.pullHeight():5.1f} px   fill {self.fill:4.2f}   foam {self.foam:4.2f}   "
                               f"drops {len(self.droplets)}   colour distance {self.colourDistance():5.1f}")
        # a dark strip behind the text keeps it readable on busy backgrounds
        width = self.debugText.content_width
        arcade.draw_lbwh_rectangle_filled(self.debugText.x - width / 2 - 8, self.debugText.y - 12, width + 16, 24,
                                          (0, 0, 0, 180))
        self.debugText.draw()
