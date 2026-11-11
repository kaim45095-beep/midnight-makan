"""
rojak_scene.py
--------------
Scene 1 - "Rojak Rush" at Uncle Din's Fruit Stall.

Aiman chops six ingredients for a fruit rojak. A cleaver sweeps left and
right above the fruit; the player presses SPACE (or clicks) when the
knife lines up with a dotted cut line.

Techniques used (lecture links):
  * Week 3  - event-driven input (key / mouse press), clicking on a sprite
  * Week 4  - Pillow image.crop(): each fruit picture is cut into slices
              (core.assets.cropSlices), and Oyen's walk uses a sprite sheet
  * Week 5-6 - translation and rotation of the slices, mirroring for Oyen
  * Week 7  - interpolation: tweens with easing, knife motion trail
  * Week 8  - particle system (juice splash, sauce stream, peanuts),
              point lights, post-processing (red "spicy" tint, screen shake)
"""
import math
import random

import arcade
from PIL import Image, ImageDraw

from core import assets
from core.base_scene import BaseScene
from core.helpers import Tween, approach, clamp, easeOutBack, easeOutQuad, lerp
from core.particles import spawnDroplet, spawnDust, spawnGlowBurst, spawnJuiceSplash, spawnRing, spawnSparkles
from core.ui import drawKeyHint, drawPanel, drawText, withAlpha
from settings import (BAD_RED, CREAM, DISH_NAMES, FONT_BODY, FONT_TITLE, GOLD, GOOD_GREEN, LANTERN_ORANGE,
                      SCENE_NAMES, STALL_NAMES, WARM_LIGHT, WHITE)

# ----------------------------------------------------------------------
# Layout (Arcade coordinates: (0, 0) is the bottom-left corner)
# ----------------------------------------------------------------------
BOARD_X = 640
BOARD_Y = 235
FRUIT_X = 640               # where the fruit rests on the board
FRUIT_BOTTOM = 205          # y of the fruit's bottom edge
BOWL_X = 175
BOWL_Y = 120

KNIFE_WIDTH = 136           # cleaver drawn a bit smaller than the picture
KNIFE_REST_Y = 452          # centre y while the knife is raised
KNIFE_CHOP_Y = 312          # centre y at the bottom of a chop
KNIFE_DOWN_TIME = 0.07
KNIFE_UP_TIME = 0.14

PERFECT_RANGE = 10          # pixels between the knife and a cut line
GOOD_RANGE = 24
SLICE_GAP = 8               # how far the slices open after each cut

OYEN_Y = 215                # Oyen walks along the back of the counter
OYEN_STEAL_X = 112          # when he gets this far he reaches the bowl
OYEN_SNEAK_SPEED = 42
OYEN_FLEE_SPEED = 560

# name, picture, cut positions (0..1 across the picture), juice colour
INGREDIENTS = [
    ("Cucumber", "food/cucumber.png", [0.2, 0.4, 0.6, 0.8], (170, 230, 120)),
    ("Pineapple", "food/pineapple.png", [0.2, 0.4, 0.6, 0.8], (255, 220, 80)),
    ("Mango", "food/mango.png", [0.2, 0.4, 0.6, 0.8], (255, 170, 40)),
    ("Jicama", "food/jicama.png", [0.2, 0.4, 0.6, 0.8], (245, 245, 225)),
    ("Rose Apple", "food/jambu.png", [0.22, 0.42, 0.62, 0.8], (240, 80, 110)),
    ("Chili", "food/chili.png", [0.3, 0.5, 0.7], (230, 40, 30)),
]
KNIFE_SPEEDS = [250, 290, 330, 370, 410, 450]    # the knife gets faster for every ingredient

BONUS_BLUE = (120, 190, 255)     # colour of the time bonus popup
TOTAL_LINES = 23                                  # 5 fruits x 4 lines + chili x 3
# All perfect + two shoos + a good time bonus, times 1.3 because Aura Mode doubles points
# for a while: three stars need clean cuts AND good use of Aura Mode.
MAX_SCORE = int((TOTAL_LINES * 100 + 2 * 50 + 150) * 1.3)


class Ingredient:
    """One fruit on the board, drawn as slices so it can be cut piece by piece."""

    def __init__(self, name, path, cutFractions, juiceColor):
        self.name = name
        self.path = path
        self.cutFractions = cutFractions
        self.juiceColor = juiceColor
        self.texture = assets.loadTexture(path)
        self.width = self.texture.width
        self.height = self.texture.height
        # Week 4: Pillow crop() cuts the picture into vertical strips at the cut lines
        self.slices = assets.cropSlices(path, cutFractions)
        self.cut = [False] * len(cutFractions)
        self.sliceOffsets = [0.0] * len(self.slices)     # extra x shift of every slice (opens gaps)
        self.x = 1420.0
        self.y = FRUIT_BOTTOM + self.height / 2
        self.slideTween = Tween(1420, FRUIT_X, 0.55, easeOutBack)
        self.state = "sliding"         # sliding -> ready -> flying -> done
        self.squash = 0.0              # 0..1 "punch" after a cut
        self.wobble = 0.0              # small rotation after a miss
        self.flyPieces = []

    def left(self):
        return self.x - self.width / 2 - self.totalGap() / 2

    def right(self):
        return self.x + self.width / 2 + self.totalGap() / 2

    def totalGap(self):
        return self.sliceOffsets[-1]

    def lineX(self, index):
        """Screen x of cut line 'index' (it moves right when gaps open on its left)."""
        base = self.x - self.width / 2 + self.cutFractions[index] * self.width
        return base + self.sliceOffsets[index] - self.totalGap() / 2

    def nearestUncutLine(self, x):
        bestIndex = -1
        bestDistance = 99999.0
        for i in range(len(self.cut)):
            if not self.cut[i]:
                gap = abs(self.lineX(i) - x)
                if gap < bestDistance:
                    bestDistance = gap
                    bestIndex = i
        return bestIndex, bestDistance

    def allCut(self):
        return all(self.cut)

    def cutLine(self, index):
        self.cut[index] = True
        self.squash = 1.0
        # every slice to the right of this line moves over by one gap
        for j in range(index + 1, len(self.slices)):
            self.sliceOffsets[j] += SLICE_GAP

    def update(self, dt):
        if self.state == "sliding":
            self.slideTween.update(dt)
            self.x = self.slideTween.value
            if self.slideTween.done:
                self.state = "ready"
                return "landed"
        self.squash = approach(self.squash, 0.0, 9.0, dt)
        self.wobble = approach(self.wobble, 0.0, 7.0, dt)
        return None

    def draw(self):
        if self.state in ("flying", "done"):
            return
        scaleX = 1.0 + 0.05 * self.squash
        scaleY = 1.0 - 0.09 * self.squash
        angle = math.sin(self.wobble * 20) * 4 * self.wobble
        for i in range(len(self.slices)):
            texture, leftFraction, rightFraction = self.slices[i]
            middle = (leftFraction + rightFraction) / 2
            sliceX = self.x - self.width / 2 + middle * self.width + self.sliceOffsets[i] - self.totalGap() / 2
            sliceWidth = (rightFraction - leftFraction) * self.width
            # translation (Week 5-6): each slice keeps its own x offset
            sliceY = FRUIT_BOTTOM + self.height * scaleY / 2
            assets.drawTexture(texture, sliceX, sliceY, sliceWidth * scaleX, self.height * scaleY, angle)


class FlyingPiece:
    """A slice jumping into the bowl (projectile motion with gravity)."""

    def __init__(self, texture, x, y, width, height, targetX, targetY, flightTime):
        self.texture = texture
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.time = 0.0
        self.flightTime = flightTime
        self.gravity = -1900.0
        # choose the start velocity so the piece lands on the target after 'flightTime'
        self.vx = (targetX - x) / flightTime
        self.vy = (targetY - y - 0.5 * self.gravity * flightTime * flightTime) / flightTime
        self.angle = 0.0
        self.spin = random.uniform(-420, 420)

    def update(self, dt):
        self.time += dt
        self.vy += self.gravity * dt
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.angle += self.spin * dt
        return self.time < self.flightTime

    def draw(self):
        shrink = lerp(1.0, 0.45, clamp(self.time / self.flightTime, 0.0, 1.0))
        assets.drawTexture(self.texture, self.x, self.y, self.width * shrink, self.height * shrink, self.angle)


class Oyen:
    """The greedy market cat: sneaks towards the bowl, steals, or runs when shooed."""

    def __init__(self):
        self.walkFrames = assets.loadFrames("characters/oyen_walk.png", 4, 1)
        self.fleeFrames = assets.loadFramesMirrored("characters/oyen_walk.png", 4, 1)
        self.state = "hidden"          # hidden -> sneak -> flee -> hidden
        self.x = -90.0
        self.y = OYEN_Y
        self.frame = 0
        self.animationTime = 0.0
        self.frameDuration = 0.16
        self.sneakTime = 0.0
        self.carriesFruit = None       # texture of a stolen slice

    def appear(self):
        self.state = "sneak"
        self.x = -50.0
        self.sneakTime = 0.0
        self.carriesFruit = None

    def contains(self, x, y):
        return self.state == "sneak" and abs(x - self.x) < 78 and abs(y - self.y) < 60

    def flee(self, fruitTexture=None):
        self.state = "flee"
        self.carriesFruit = fruitTexture
        self.frameDuration = 0.07

    def update(self, dt):
        if self.state == "hidden":
            return None
        # sprite sheet animation (Week 4): change frame every frameDuration seconds
        self.animationTime += dt
        if self.animationTime >= self.frameDuration:
            self.animationTime = 0.0
            self.frame = (self.frame + 1) % 4
        if self.state == "sneak":
            self.sneakTime += dt
            self.x += OYEN_SNEAK_SPEED * dt
            if self.x >= OYEN_STEAL_X:
                return "steal"
        elif self.state == "flee":
            self.x -= OYEN_FLEE_SPEED * dt
            if self.x < -140:
                self.state = "hidden"
                self.frameDuration = 0.16
        return None

    def draw(self):
        if self.state == "hidden":
            return
        if self.state == "flee":
            texture = self.fleeFrames[self.frame]
            hop = abs(math.sin(self.x * 0.05)) * 14
        else:
            texture = self.walkFrames[self.frame]
            hop = 0.0
        assets.drawTexture(texture, self.x, self.y + hop, 164, 123)
        if self.carriesFruit is not None:
            # the stolen slice is carried in his mouth (at the front of a left-facing cat)
            assets.drawTexture(self.carriesFruit, self.x - 58, self.y + hop + 10, 34, None, 20)


class RojakScene(BaseScene):
    sceneId = "rojak"
    sceneTitle = "Scene 1 - " + SCENE_NAMES["rojak"]
    stallName = STALL_NAMES["rojak"]
    dishName = DISH_NAMES["rojak"]
    musicName = "music_rojak"
    auraMusicName = "music_rojak_aura"
    ambienceName = "amb_night_market"
    timeLimit = 75
    maxScore = MAX_SCORE
    isGameplayScene = True
    controlsHelp = [("SPACE", "Chop when the knife lines up with a dotted line"),
                    ("CLICK", "Chop (click anywhere on the stall)"),
                    ("CLICK CAT", "Shoo Oyen away before he steals fruit"),
                    ("S", "Shoo Oyen with the keyboard")]

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def setupScene(self):
        self.background = assets.loadTexture("backgrounds/fruit_stall_bg.png")
        self.boardTexture = assets.loadTexture("props/cutting_board.png")
        self.knifeTexture = assets.loadTexture("props/cleaver.png")
        self.knifeHeight = KNIFE_WIDTH * self.knifeTexture.height / self.knifeTexture.width
        self.bowlTextures = []
        for i in range(4):
            self.bowlTextures.append(assets.loadTexture(f"food/rojak_bowl_{i}.png"))
        self.dishTexture = assets.loadTexture("food/rojak_dish.png")
        self.stainTexture = assets.loadTexture("particles/soft_circle.png")
        self.dashTexture = self.makeDashTexture()
        self.iconTextures = []
        for name, path, cuts, juice in INGREDIENTS:
            self.iconTextures.append(assets.loadTexture(path))

        # Ingredients and progress
        self.ingredientIndex = 0
        self.ingredient = None
        self.bowlFill = 0                 # pieces in the bowl (drops when Oyen steals)
        self.bowlBump = 0.0
        self.stolenCount = 0
        self.shooCount = 0
        self.oyenVisits = 0

        # The knife (sweeps left and right, then chops down)
        self.knifeX = FRUIT_X
        self.knifeY = KNIFE_REST_Y
        self.knifeDirection = 1
        self.knifePhase = "idle"          # idle -> down -> hold -> up
        self.knifeTimer = 0.0
        self.holdTime = 0.05
        self.trail = []                   # recent knife y positions for the motion trail

        # Juice stains that stay on the board for a while (one SpriteList = one draw call)
        self.stains = arcade.SpriteList()

        # Effects
        self.spicyTimer = 0.0
        self.oyen = Oyen()
        self.finalePhase = None           # None -> "sauce" -> "peanuts" -> "done"
        self.finaleTimer = 0.0
        self.showingDish = False
        self.warningTimer = 0.0

        # Warm light from the two hanging bulbs in the background (Week 8 point lights)
        self.lights.add(470, 572, 170, WARM_LIGHT, 0.45, 0.08)
        self.lights.add(810, 572, 170, WARM_LIGHT, 0.45, 0.08)

        # Text objects that change often are made once (never inside on_draw)
        self.debugText = arcade.Text("", 640, 548, (170, 255, 190), 13, font_name=FONT_BODY,
                                     anchor_x="center", anchor_y="center")

        # Demo bot plan: how each chop should turn out (a few mistakes on purpose)
        self.botQualities = ["perfect", "perfect", "good", "perfect", "perfect",
                             "miss", "perfect", "good", "perfect", "perfect",
                             "perfect", "good", "perfect", "perfect", "miss",
                             "perfect", "perfect", "good", "perfect", "perfect",
                             "perfect", "perfect", "perfect", "perfect", "perfect"]
        self.botChopIndex = 0
        self.botWait = 0.0
        self.botSearchTime = 0.0       # how long the bot has waited for the knife to line up

        self.startNextIngredient()

    def makeDashTexture(self):
        """A dotted line picture made once with Pillow (one texture = one cheap draw call)."""
        image = Image.new("RGBA", (4, 220), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        for y in range(0, 220, 16):
            draw.rectangle([0, y, 3, y + 8], fill=(255, 255, 255, 255))
        return assets.textureFromImage(image, "rojak-dash-line")

    def getIntroLines(self):
        return [
            ("nenek", "First course: fruit rojak! Uncle Din lent us his fruit stall, so chop nice and neat."),
            ("nenek", "Watch the cleaver. Press SPACE or click when it lines up with a dotted line."),
            ("aiman", "Clean cuts build my combo. Five in a row and the Aura Ladle wakes up!"),
            ("nenek", "And keep an eye on Oyen. That cat loves fruit more than I do. Click him to shoo!"),
        ]

    # ------------------------------------------------------------------
    # Ingredient flow
    # ------------------------------------------------------------------
    def startNextIngredient(self):
        name, path, cuts, juice = INGREDIENTS[self.ingredientIndex]
        self.ingredient = Ingredient(name, path, cuts, juice)
        self.knifeX = FRUIT_X - self.ingredient.width / 2 - 30
        self.knifeDirection = 1
        # Oyen sneaks in when the third and the fifth ingredient arrive
        if self.ingredientIndex in (2, 4):
            self.oyen.appear()
            self.oyenVisits += 1
            self.warningTimer = 2.2
            self.playSfx("cat_meow", 0.8, x=80)

    def ingredientFinished(self):
        """All lines are cut: the slices jump into the bowl."""
        ingredient = self.ingredient
        ingredient.state = "flying"
        for i in range(len(ingredient.slices)):
            texture, leftFraction, rightFraction = ingredient.slices[i]
            middle = (leftFraction + rightFraction) / 2
            x = ingredient.x - ingredient.width / 2 + middle * ingredient.width + ingredient.sliceOffsets[i]
            x -= ingredient.totalGap() / 2
            piece = FlyingPiece(texture, x, ingredient.y, (rightFraction - leftFraction) * ingredient.width,
                                ingredient.height, BOWL_X + random.uniform(-50, 50), BOWL_Y + 40,
                                0.55 + i * 0.05)
            ingredient.flyPieces.append(piece)
        self.playSfx("knife_whoosh", 0.5, x=ingredient.x, speed=0.8)

    def piecesLanded(self):
        self.bowlFill += 1
        self.bowlBump = 1.0
        self.playSfx("plop", 0.8, x=BOWL_X, pitchVariation=0.1)
        spawnJuiceSplash(self.particles, BOWL_X, BOWL_Y + 60, self.ingredient.juiceColor, 10, 0.6)
        self.ingredientIndex += 1
        if self.ingredientIndex < len(INGREDIENTS):
            self.startNextIngredient()
        else:
            self.ingredient = None
            self.startFinale()

    def startFinale(self):
        """Sauce drizzle then crushed peanuts, then the dish is ready."""
        self.finalePhase = "sauce"
        self.finaleTimer = 0.0
        self.bowlFill = max(self.bowlFill, len(INGREDIENTS))      # Nenek tops the bowl back up
        self.playSfx("sauce_pour", 0.9, x=BOWL_X)
        self.popups.add("Pour the sauce!", BOWL_X + 90, 330, CREAM, 22, 1.4, 40)

    # ------------------------------------------------------------------
    # Chopping
    # ------------------------------------------------------------------
    def canChop(self):
        return (self.ingredient is not None and self.ingredient.state == "ready"
                and self.knifePhase == "idle" and self.finalePhase is None)

    def startChop(self):
        if not self.canChop():
            return
        self.knifePhase = "down"
        self.knifeTimer = 0.0
        self.trail = []
        self.playSfx("knife_whoosh", 0.55, x=self.knifeX, pitchVariation=0.12)

    def judgeChop(self):
        """The blade reached the board: was it on a cut line?"""
        ingredient = self.ingredient
        index, gap = ingredient.nearestUncutLine(self.knifeX)
        topOfFruit = FRUIT_BOTTOM + ingredient.height
        if index >= 0 and gap <= GOOD_RANGE:
            if gap <= PERFECT_RANGE:
                quality = "perfect"
                self.holdTime = 0.09          # hit-stop: the knife stays down a moment longer
                self.shake.start(4, 0.12)
            else:
                quality = "good"
                self.holdTime = 0.05
            ingredient.cutLine(index)
            cutX = ingredient.lineX(index)
            self.registerHit(quality, cutX, topOfFruit)
            self.playSfx("chop_" + str(random.randint(1, 3)), 0.9, x=cutX, pitchVariation=0.08)
            self.playSfx("juice_squish", 0.5, x=cutX, pitchVariation=0.15)
            power = 1.2 if quality == "perfect" else 0.8
            spawnJuiceSplash(self.particles, cutX, topOfFruit - 10, ingredient.juiceColor, 20, power)
            self.addStains(cutX, ingredient.juiceColor)
            if ingredient.name == "Chili":
                self.spicyTimer = 0.7
                spawnSparkles(self.glowParticles, cutX, topOfFruit, (255, 80, 40), 14, 240)
                self.playSfx("spicy_sizzle", 0.7, x=cutX)
                self.popups.add("SPICY!", cutX - 150, topOfFruit + 40, BAD_RED, 30, 0.9, 40)
            if ingredient.allCut():
                self.ingredientFinished()
        else:
            # A miss: the blade hits the board (or the fruit between the lines)
            self.holdTime = 0.04
            ingredient.wobble = 1.0
            self.registerHit("miss", self.knifeX, topOfFruit)
            self.playSfx("board_thud", 0.8, x=self.knifeX, pitchVariation=0.1)
            self.addScratch(self.knifeX)

    def addStains(self, x, rgb):
        """Juice drops that land on the board and fade out slowly (fluid splatter)."""
        for i in range(random.randint(3, 5)):
            stain = arcade.Sprite(self.stainTexture)
            stain.center_x = x + random.uniform(-70, 70)
            stain.center_y = random.uniform(150, 200)
            size = random.uniform(10, 26)
            stain.scale = size / self.stainTexture.width
            stain.color = (rgb[0], rgb[1], rgb[2], 150)
            stain.life = 6.0
            self.stains.append(stain)
        while len(self.stains) > 45:
            self.stains[0].remove_from_sprite_lists()

    def addScratch(self, x):
        stain = arcade.Sprite(self.stainTexture)
        stain.center_x = x
        stain.center_y = random.uniform(160, 185)
        stain.scale = (40 / self.stainTexture.width, 6 / self.stainTexture.height)
        stain.color = (70, 40, 20, 150)
        stain.life = 6.0
        self.stains.append(stain)
        spawnDust(self.particles, x, 175, 5)

    # ------------------------------------------------------------------
    # Oyen the cat
    # ------------------------------------------------------------------
    def shooOyen(self):
        if self.oyen.state != "sneak":
            return
        self.shooCount += 1
        self.oyen.flee()
        self.playSfx("cat_hiss", 0.9, x=self.oyen.x)
        spawnDust(self.particles, self.oyen.x, self.oyen.y - 40, 10)
        self.addScore(50, self.oyen.x + 40, self.oyen.y + 60, "SHOO! +50", GOOD_GREEN)

    def oyenSteals(self):
        self.stolenCount += 1
        self.bowlFill = max(0, self.bowlFill - 1)
        self.bowlBump = 1.0
        stolenPiece = self.iconTextures[max(0, self.ingredientIndex - 1)]
        self.oyen.flee(stolenPiece)
        self.playSfx("cat_meow", 1.0, x=self.oyen.x, speed=1.1)
        self.addScore(-50, BOWL_X + 150, BOWL_Y + 150, "Oyen stole a slice! -50", BAD_RED)

    # ------------------------------------------------------------------
    # Input
    # ------------------------------------------------------------------
    def onPlayKeyPress(self, key, modifiers):
        if key == arcade.key.SPACE:
            self.startChop()
        elif key == arcade.key.S:
            self.shooOyen()

    def onPlayMousePress(self, x, y, button):
        if button != arcade.MOUSE_BUTTON_LEFT:
            return
        # Week 3: check if the click landed inside Oyen's box first
        if self.oyen.contains(x, y):
            self.shooOyen()
        else:
            self.startChop()

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------
    def updateAlways(self, dt):
        self.spicyTimer = max(0.0, self.spicyTimer - dt)
        self.warningTimer = max(0.0, self.warningTimer - dt)
        self.bowlBump = approach(self.bowlBump, 0.0, 6.0, dt)
        # stains fade over their life time
        for stain in list(self.stains):
            stain.life -= dt
            if stain.life <= 0:
                stain.remove_from_sprite_lists()
            else:
                stain.alpha = int(150 * clamp(stain.life / 2.0, 0.0, 1.0))

    def updatePlay(self, dt):
        self.updateKnife(dt)

        ingredient = self.ingredient
        if ingredient is not None:
            if ingredient.update(dt) == "landed":
                self.playSfx("board_thud", 0.9, x=FRUIT_X)
                spawnDust(self.particles, FRUIT_X, FRUIT_BOTTOM + 5, 6)
            if ingredient.state == "flying":
                stillFlying = []
                for piece in ingredient.flyPieces:
                    if piece.update(dt):
                        stillFlying.append(piece)
                ingredient.flyPieces = stillFlying
                if len(stillFlying) == 0:
                    ingredient.state = "done"
                    self.piecesLanded()

        result = self.oyen.update(dt)
        if result == "steal":
            self.oyenSteals()

        if self.finalePhase is not None:
            self.updateFinale(dt)

    def updateKnife(self, dt):
        ingredient = self.ingredient
        if self.knifePhase == "idle":
            if ingredient is None or ingredient.state != "ready":
                # wait above the board while the next fruit slides in
                self.knifeX = approach(self.knifeX, FRUIT_X - 200, 3.0, dt)
                return
            # ping-pong sweep across the fruit
            speed = KNIFE_SPEEDS[self.ingredientIndex]
            minX = ingredient.left() - 40
            maxX = ingredient.right() + 40
            self.knifeX += self.knifeDirection * speed * dt
            if self.knifeX > maxX:
                self.knifeX = maxX
                self.knifeDirection = -1
            elif self.knifeX < minX:
                self.knifeX = minX
                self.knifeDirection = 1
            self.knifeY = approach(self.knifeY, KNIFE_REST_Y, 14.0, dt)
            return

        self.knifeTimer += dt
        if self.knifePhase == "down":
            t = clamp(self.knifeTimer / KNIFE_DOWN_TIME, 0.0, 1.0)
            self.knifeY = lerp(KNIFE_REST_Y, KNIFE_CHOP_Y, t * t)     # speeds up as it falls
            self.trail.append(self.knifeY)
            if t >= 1.0:
                self.knifePhase = "hold"
                self.knifeTimer = 0.0
                self.judgeChop()
        elif self.knifePhase == "hold":
            if self.knifeTimer >= self.holdTime:
                self.knifePhase = "up"
                self.knifeTimer = 0.0
        elif self.knifePhase == "up":
            t = clamp(self.knifeTimer / KNIFE_UP_TIME, 0.0, 1.0)
            self.knifeY = lerp(KNIFE_CHOP_Y, KNIFE_REST_Y, easeOutQuad(t))
            if t >= 1.0:
                self.knifePhase = "idle"
                self.trail = []

    def updateFinale(self, dt):
        self.finaleTimer += dt
        if self.finalePhase == "sauce":
            # a thick sauce stream (fluid particles) falls into the bowl
            for i in range(3):
                spawnDroplet(self.particles, BOWL_X + random.uniform(-10, 10), 360,
                             random.uniform(-25, 25), random.uniform(-80, -20), (86, 44, 20), 13, 0.5)
            if self.finaleTimer >= 1.6:
                self.finalePhase = "peanuts"
                self.finaleTimer = 0.0
                self.playSfx("sprinkle", 0.9, x=BOWL_X)
        elif self.finalePhase == "peanuts":
            if self.finaleTimer < 0.8:
                for i in range(2):
                    spawnDroplet(self.particles, BOWL_X + random.uniform(-60, 60), 300,
                                 random.uniform(-40, 40), random.uniform(-40, 40), (196, 150, 90), 6, 0.6)
            if self.finaleTimer >= 1.1 and not self.showingDish:
                self.showingDish = True
                self.bowlBump = 1.0
                spawnGlowBurst(self.glowParticles, BOWL_X, BOWL_Y + 20, WARM_LIGHT, 320, 0.5)
                spawnSparkles(self.glowParticles, BOWL_X, BOWL_Y + 40, GOLD, 26, 260)
                spawnRing(self.glowParticles, BOWL_X, BOWL_Y + 30, GOLD, 300)
                self.playSfx("happy_chime", 0.8, x=BOWL_X)
            if self.finaleTimer >= 1.9:
                self.finalePhase = "done"
                bonus = int(self.timeLeft) * 5
                if bonus > 0:
                    self.score += bonus
                    self.popups.add(f"TIME BONUS +{bonus}", 640, 470, BONUS_BLUE, 28, 1.5, 50)
                self.finishScene("Crunchy, sweet and spicy!")

    # ------------------------------------------------------------------
    # Demo bot (drives the scene through the same input methods as a player)
    # ------------------------------------------------------------------
    def autoPlay(self, dt):
        if self.dialogue.isActive():
            return
        # Oyen: shoo him on his first visit, let him steal on the second (to show both)
        if self.oyen.state == "sneak":
            if self.oyenVisits == 1 and self.oyen.sneakTime > 1.6:
                if self.botMoveCursor(self.oyen.x + 10, self.oyen.y + 10, dt, 1300):
                    self.botClick()
                return
        if self.botWait > 0:
            self.botWait -= dt
            return
        if not self.canChop():
            return

        # Use the mouse for some ingredients and the keyboard for others
        useMouse = self.ingredientIndex % 2 == 1
        if useMouse:
            # a click anywhere on the stall chops, so the cursor rests on the board
            if not self.botMoveCursor(560, 140, dt, 1400):
                return

        wanted = self.botQualities[self.botChopIndex % len(self.botQualities)]
        index, gap = self.ingredient.nearestUncutLine(self.knifeX)
        ready = False
        if wanted == "perfect" and gap <= 4:
            ready = True
        elif wanted == "good" and 14 <= gap <= 20:
            ready = True
        elif wanted == "miss" and gap >= 45 and self.ingredient.left() < self.knifeX < self.ingredient.right():
            ready = True
        # With a fixed time step the knife can keep skipping a narrow window, so after
        # 3 seconds the bot settles for any clean cut (that window is wider than one step).
        self.botSearchTime += dt
        if not ready and self.botSearchTime > 3.0 and gap <= PERFECT_RANGE:
            ready = True
        if ready:
            self.botSearchTime = 0.0
            self.botChopIndex += 1
            # a short "thinking" pause after every chop, like a real player
            self.botWait = random.uniform(0.35, 0.8)
            if useMouse:
                self.botClick()
            else:
                self.botTapKey(arcade.key.SPACE)

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------
    def getResultLines(self):
        return [
            f"Cuts: {self.perfectCount} perfect, {self.goodCount} good, {self.missCount} missed",
            f"Oyen shooed {self.shooCount} time(s), slices stolen: {self.stolenCount}",
        ]

    def getDishTexture(self):
        return self.dishTexture

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------
    def drawWorld(self):
        assets.drawTexture(self.background, 640, 360, 1280, 720)
        assets.drawTexture(self.boardTexture, BOARD_X, BOARD_Y, 700, 260)
        self.stains.draw()

        # Oyen walks behind the bowl, so he is drawn first (painter's algorithm)
        self.oyen.draw()
        self.drawBowl()

        ingredient = self.ingredient
        if ingredient is not None:
            ingredient.draw()
            for piece in ingredient.flyPieces:
                piece.draw()
            if ingredient.state == "ready":
                self.drawCutLines(ingredient)
        self.drawKnife()

    def drawBowl(self):
        if self.showingDish:
            texture = self.dishTexture
            width = 300
        else:
            stage = min(3, (self.bowlFill + 1) // 2)
            texture = self.bowlTextures[stage]
            width = 270
        bump = 1.0 + 0.08 * self.bowlBump
        assets.drawTexture(texture, BOWL_X, BOWL_Y + 4 * self.bowlBump, width * bump, None)

    def drawCutLines(self, ingredient):
        nearest, gap = ingredient.nearestUncutLine(self.knifeX)
        pulse = 0.5 + 0.5 * math.sin(self.sceneTime * 8)
        top = FRUIT_BOTTOM + ingredient.height + 18
        height = ingredient.height + 36
        for i in range(len(ingredient.cut)):
            if ingredient.cut[i]:
                continue
            x = ingredient.lineX(i)
            if i == nearest:
                colour = withAlpha(GOLD, 170 + 85 * pulse)
            else:
                colour = withAlpha(WHITE, 170)
            assets.drawTexture(self.dashTexture, x, top - height / 2, 3, height, rgb=colour, alpha=colour[3])

    def drawKnife(self):
        # Motion trail: faded copies along the path of the chop (Week 7 interpolation)
        if self.knifePhase in ("down", "hold") and len(self.trail) > 1:
            for i in range(0, len(self.trail) - 1):
                alpha = 40 + 30 * i
                assets.drawTexture(self.knifeTexture, self.knifeX, self.trail[i], KNIFE_WIDTH, self.knifeHeight,
                                   alpha=min(alpha, 150))
        # Aim line: shows where the blade will land (gold when it is on a cut line)
        if self.knifePhase == "idle" and self.ingredient is not None and self.ingredient.state == "ready":
            index, gap = self.ingredient.nearestUncutLine(self.knifeX)
            if gap <= GOOD_RANGE:
                colour = GOLD
            else:
                colour = (255, 255, 255)
            bladeBottom = self.knifeY - self.knifeHeight / 2 + 6
            arcade.draw_line(self.knifeX, bladeBottom, self.knifeX, FRUIT_BOTTOM - 6, withAlpha(colour, 120), 2)
        assets.drawTexture(self.knifeTexture, self.knifeX, self.knifeY, KNIFE_WIDTH, self.knifeHeight)

    def drawWorldTop(self):
        # Post-processing style tint: the chili makes the whole screen feel hot
        if self.spicyTimer > 0:
            alpha = int(90 * self.spicyTimer / 0.7)
            arcade.draw_lbwh_rectangle_filled(0, 0, 1280, 720, (255, 50, 20, alpha))

    def drawPlayHud(self):
        # Ingredient tracker: six icons with a tick for every finished fruit
        drawPanel(510, 610, 500, 66, (14, 16, 38, 200), (255, 200, 87, 120), 16, shadow=False)
        for i in range(len(INGREDIENTS)):
            x = 300 + i * 84
            texture = self.iconTextures[i]
            alpha = 255
            if i > self.ingredientIndex:
                alpha = 110
            size = 70
            if i == self.ingredientIndex:
                size = 70 + 8 * math.sin(self.sceneTime * 6)
            assets.drawTexture(texture, x, 610, size, None, alpha=alpha)
            if i < self.ingredientIndex:
                assets.drawTexture(assets.loadTexture("ui/icon_check.png"), x + 26, 596, 28, 28)

        if self.ingredient is not None and self.finalePhase is None:
            drawText(self.ingredient.name.upper(), FRUIT_X, 84, CREAM, 18, FONT_TITLE,
                     anchorX="center", anchorY="center")

        # Oyen warning banner
        if self.oyen.state == "sneak" and self.warningTimer > 0:
            alpha = 255 * clamp(self.warningTimer / 0.4, 0.0, 1.0)
            drawPanel(290, 330, 330, 44, (60, 20, 20, int(220 * alpha / 255)), withAlpha(LANTERN_ORANGE, alpha), 14,
                      shadow=False)
            drawText("OYEN IS SNEAKING! CLICK HIM!", 290, 330, CREAM, 15, FONT_BODY,
                     anchorX="center", anchorY="center", bold=True, alpha=alpha)

        drawKeyHint(560, 28, "SPACE", "Chop")
        drawKeyHint(760, 28, "S", "Shoo cat")
        drawKeyHint(960, 28, "F1", "Help")

    def drawDebugWorld(self):
        """F2: shows the maths behind the timing judgement."""
        ingredient = self.ingredient
        if ingredient is None or ingredient.state != "ready":
            return
        index, gap = ingredient.nearestUncutLine(self.knifeX)
        for i in range(len(ingredient.cut)):
            x = ingredient.lineX(i)
            colour = (120, 120, 120) if ingredient.cut[i] else (80, 255, 120)
            arcade.draw_line(x, 170, x, 380, colour, 1)
            # good and perfect windows around the line
            if not ingredient.cut[i]:
                arcade.draw_lbwh_rectangle_filled(x - GOOD_RANGE, 172, GOOD_RANGE * 2, 6, (110, 220, 130, 90))
                arcade.draw_lbwh_rectangle_filled(x - PERFECT_RANGE, 172, PERFECT_RANGE * 2, 6, (255, 200, 87, 160))
        arcade.draw_line(self.knifeX, 160, self.knifeX, 400, (255, 120, 255), 1)
        arcade.draw_lbwh_rectangle_outline(ingredient.left() - 40, 140, ingredient.right() - ingredient.left() + 80,
                                           270, (120, 180, 255), 1)
        if self.oyen.state == "sneak":
            arcade.draw_lbwh_rectangle_outline(self.oyen.x - 78, self.oyen.y - 60, 156, 120, (255, 160, 80), 1)
        self.debugText.text = (f"knifeX {self.knifeX:6.1f}  nearest line d = {gap:5.1f} px   "
                               f"(perfect <= {PERFECT_RANGE}, good <= {GOOD_RANGE})   speed {KNIFE_SPEEDS[self.ingredientIndex]} px/s")
        # a dark strip behind the text keeps it readable on busy backgrounds
        width = self.debugText.content_width
        arcade.draw_lbwh_rectangle_filled(self.debugText.x - width / 2 - 8, self.debugText.y - 12, width + 16, 24,
                                          (0, 0, 0, 180))
        self.debugText.draw()

