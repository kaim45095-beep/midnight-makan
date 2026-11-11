"""
banquet_scene.py
----------------
Scene 4 - "Lazy Susan Banquet" at Pelita Hall.

Everyone sits at one round banquet table. Six dishes stand on a glass
lazy susan (turntable). The player spins it with LEFT / RIGHT (or drags
it with the mouse) and serves a guest with SPACE (or a click on the
guest) when the dish they want is right in front of them. Dato' Leon,
the critic, wants the three dishes Aiman cooked: rojak, char kuey teow,
then teh tarik. When he has all three, the feast is complete.

Techniques used (lecture links):
  * Week 5-6 - 2D transformations: every dish position is
              translate(tableCentre) * scale(1, 0.35) * rotate(angle) * (radius, 0)
              written out with the rotation matrix formula
  * Week 5-6 - visibility: dishes are sorted back-to-front and drawn in
              that order (painter's algorithm); far dishes are drawn
              smaller (perspective scaling)
  * Week 7  - interpolation: HSV hue for the patience rings, tweened
              dish flights, smoothed spin speed
  * Week 3/4 - mouse drag on the table, sprite sheet frames for the guests
  * Week 8  - particles (hearts, sparkles, spills), camera flash effect
  * Audio   - the turntable rumble gets louder as it spins faster
"""
import math
import random

import arcade

from core import assets
from core.base_scene import BaseScene
from core.helpers import Tween, angleDifference, approach, clamp, easeOutQuad, hsvColor, lerp
from core.particles import (spawnDroplet, spawnDust, spawnGlowBurst, spawnHearts, spawnJuiceSplash, spawnRing,
                            spawnSparkles)
from core.ui import drawKeyHint, drawPanel, drawStars, drawText, withAlpha
from settings import (BAD_RED, CREAM, DISH_NAMES, FONT_BODY, GOLD, GOOD_GREEN, SCENE_NAMES,
                      SPEAKERS, STALL_NAMES, WHITE)

# ----------------------------------------------------------------------
# Layout
# ----------------------------------------------------------------------
TABLE_X = 640
TABLE_Y = 235               # centre of table_top.png
SUSAN_X = 640
SUSAN_Y = 252               # centre of the glass turntable (and of the dish ring)
DISH_RADIUS = 215           # radius of the dish ring on the turntable
PERSPECTIVE = 0.35          # the table is seen from above at an angle: y is squashed
GUEST_RX = 430              # guests sit on a bigger ellipse around the table
GUEST_RY = 150
GUEST_LIFT = 70             # lift so their upper body shows above the table
SERVE_WINDOW = 18           # degrees: how close a dish must be to a guest's seat

SPIN_ACCELERATION = 300     # degrees per second squared while an arrow key is held
SPIN_FRICTION = 1.4         # how quickly the spin slows down on its own
MAX_SPIN = 320
WOBBLE_SPEED = 220          # spinning faster than this makes the dishes wobble

# Slot order around the turntable (60 degrees apart). The critic's three
# courses are 120 degrees apart, so every course needs a real spin.
DISHES = [
    ("rojak", "Rojak", "food/dish_rojak.png"),
    ("nasilemak", "Nasi Lemak", "food/dish_nasilemak.png"),
    ("ckt", "Char Kuey Teow", "food/dish_ckt.png"),
    ("satay", "Satay", "food/dish_satay.png"),
    ("tehtarik", "Teh Tarik", "food/dish_tehtarik.png"),
    ("roticanai", "Roti Canai", "food/dish_roticanai.png"),
]
CRITIC_SAVOUR_TIME = 8.0    # seconds the critic takes to taste a course before the next one
CRITIC_ORDER = ["rojak", "ckt", "tehtarik"]

# speaker id, seat angle (degrees, 90 = back centre), patience drain per second
GUESTS = [
    ("farah", 160, 1 / 22),
    ("lim", 125, 1 / 14),       # Uncle Lim is the most impatient
    ("leon", 90, 1 / 30),
    ("kavitha", 55, 1 / 20),
    ("nenek", 20, 1 / 24),
]

MAX_SCORE = 2200


def rotatePoint(x, y, angleDegrees):
    """
    2D rotation about the origin (lecture formula, anticlockwise):
        x' = x cos(t) - y sin(t)
        y' = x sin(t) + y cos(t)
    As a matrix:  [x']   [cos t  -sin t] [x]
                  [y'] = [sin t   cos t] [y]
    """
    t = math.radians(angleDegrees)
    return x * math.cos(t) - y * math.sin(t), x * math.sin(t) + y * math.cos(t)


def turntablePoint(angleDegrees, radius):
    """
    Composite transformation for a point on the lazy susan:
      1. start at (radius, 0)
      2. rotate by the angle            (rotation matrix)
      3. scale y by PERSPECTIVE         (non-uniform scaling = seen at an angle)
      4. translate to the table centre  (translation)
    """
    x, y = rotatePoint(radius, 0, angleDegrees)
    y = y * PERSPECTIVE
    return SUSAN_X + x, SUSAN_Y + y


def backToFront(item):
    """Sort key: items with a bigger y are further back, so they come first."""
    return -item[0]


def guestBackToFront(guest):
    return -guest.y


def depthOf(angleDegrees):
    """0 at the front edge (closest to the player) and 1 at the back edge."""
    return (math.sin(math.radians(angleDegrees)) + 1) / 2


class Guest:
    def __init__(self, speaker, seatAngle, drain):
        self.speaker = speaker
        self.seatAngle = seatAngle
        self.drain = drain
        self.frames = assets.loadFrames(f"characters/guest_{speaker}.png", 4, 1)
        self.x = TABLE_X + GUEST_RX * math.cos(math.radians(seatAngle))
        self.y = TABLE_Y + GUEST_RY * math.sin(math.radians(seatAngle)) + GUEST_LIFT
        self.order = None            # dish id they want (None = not hungry right now)
        self.patience = 1.0
        self.cooldown = random.uniform(0.5, 2.5)
        self.frame = 0
        self.frameTimer = 0.0
        self.blinkTimer = random.uniform(2, 4)
        self.bubblePop = 0.0
        self.goldGlow = 0.0

    def contains(self, x, y):
        return abs(x - self.x) < 80 and abs(y - self.y) < 100

    def setOrder(self, dishId):
        self.order = dishId
        self.patience = 1.0
        self.bubblePop = 1.0

    def react(self, frame, seconds):
        self.frame = frame
        self.frameTimer = seconds

    def update(self, dt):
        self.bubblePop = approach(self.bubblePop, 0.0, 6.0, dt)
        self.goldGlow = approach(self.goldGlow, 0.0, 0.8, dt)
        if self.frameTimer > 0:
            self.frameTimer -= dt
            if self.frameTimer <= 0:
                self.frame = 0
        else:
            # blink now and then (frame 1)
            self.blinkTimer -= dt
            if self.blinkTimer <= 0:
                self.react(1, 0.13)
                self.blinkTimer = random.uniform(2.5, 4.5)

    def draw(self):
        if self.goldGlow > 0.02:
            assets.drawTexture(assets.loadTexture("particles/light_glow.png"), self.x, self.y + 20, 300, 300,
                               rgb=(255, 210, 110), alpha=220 * self.goldGlow, additive=True)
        assets.drawTexture(self.frames[self.frame], self.x, self.y, 180, 220)


class Dish:
    def __init__(self, slot, dishId, name, path):
        self.slot = slot                   # 0..5 around the turntable, 60 degrees apart
        self.dishId = dishId
        self.name = name
        self.texture = assets.loadTexture(path)
        self.present = True
        self.refillTimer = 0.0
        self.fadeIn = 1.0

    def angle(self, susanAngle):
        return susanAngle + self.slot * 60


class RidingCat:
    """Oyen jumps onto the turntable and rides it (he turns with the same rotation maths)."""

    def __init__(self):
        self.frames = assets.loadFrames("characters/oyen_sit.png", 4, 1)
        self.state = "away"               # away -> jumping -> riding -> leaving -> away
        self.slotAngle = 30               # sits between two dishes
        self.time = 0.0
        self.x = -100.0
        self.y = 300.0
        self.frame = 0
        self.frameTimer = 0.0
        self.jumpFrom = (0.0, 0.0)
        self.leaveTo = (0.0, 0.0)

    def update(self, dt, susanAngle):
        self.time += dt
        self.frameTimer += dt
        if self.frameTimer > 0.18:
            self.frameTimer = 0.0
            self.frame = (self.frame + 1) % 4
        if self.state == "jumping":
            # a parabola from the floor onto the turntable
            t = clamp(self.time / 0.7, 0.0, 1.0)
            targetX, targetY = turntablePoint(susanAngle + self.slotAngle, DISH_RADIUS * 0.55)
            self.x = lerp(self.jumpFrom[0], targetX, t)
            self.y = lerp(self.jumpFrom[1], targetY, t) + math.sin(t * math.pi) * 160
            if t >= 1.0:
                self.state = "riding"
                self.time = 0.0
                return "landed"
        elif self.state == "riding":
            self.x, self.y = turntablePoint(susanAngle + self.slotAngle, DISH_RADIUS * 0.55)
        elif self.state == "leaving":
            t = clamp(self.time / 0.6, 0.0, 1.0)
            self.x = lerp(self.jumpFrom[0], self.leaveTo[0], t)
            self.y = lerp(self.jumpFrom[1], self.leaveTo[1], t) + math.sin(t * math.pi) * 140
            if t >= 1.0:
                self.state = "away"
        return None

    def contains(self, x, y):
        return self.state == "riding" and abs(x - self.x) < 70 and abs(y - (self.y + 30)) < 60

    def leave(self):
        self.state = "leaving"
        self.time = 0.0
        self.jumpFrom = (self.x, self.y)
        self.leaveTo = (1420, 160)

    def draw(self):
        if self.state == "away":
            return
        assets.drawTexture(self.frames[self.frame], self.x, self.y + 34, 150, 112)


class BanquetScene(BaseScene):
    sceneId = "banquet"
    sceneTitle = "Scene 4 - " + SCENE_NAMES["banquet"]
    stallName = STALL_NAMES["banquet"]
    dishName = DISH_NAMES["banquet"]
    musicName = "music_banquet"
    auraMusicName = "music_banquet_aura"
    ambienceName = "amb_hall"
    timeLimit = 100
    maxScore = MAX_SCORE
    isGameplayScene = True
    controlsHelp = [("LEFT / RIGHT", "Spin the lazy susan (A / D work too)"),
                    ("DRAG", "Grab the turntable with the mouse and spin it"),
                    ("SPACE", "Serve the guest whose dish is in front of them"),
                    ("CLICK GUEST", "Serve that guest"),
                    ("S / CLICK CAT", "Shoo Oyen off the turntable")]

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def setupScene(self):
        self.background = assets.loadTexture("backgrounds/banquet_bg.png")
        self.tableTexture = assets.loadTexture("props/table_top.png")
        self.susanTexture = assets.loadTexture("props/lazy_susan.png")
        self.bubbleTexture = assets.loadTexture("ui/thought_bubble.png")
        self.dishTexture = assets.loadTexture("food/dish_ckt.png")

        self.guests = []
        for speaker, angle, drain in GUESTS:
            self.guests.append(Guest(speaker, angle, drain))
        self.critic = self.guests[2]
        self.dishes = []
        for i in range(len(DISHES)):
            dishId, name, path = DISHES[i]
            self.dishes.append(Dish(i, dishId, name, path))
        self.dishById = {}
        for dish in self.dishes:
            self.dishById[dish.dishId] = dish

        self.susanAngle = 0.0            # degrees (anticlockwise, maths convention)
        self.spinSpeed = 0.0             # degrees per second
        self.draggingSusan = False
        self.lastMouseAngle = 0.0
        self.wobble = 0.0
        self.spillCooldown = 0.0
        self.rumbleLoop = None
        self.flights = []                # dishes flying to a guest
        self.criticCourse = 0
        self.cat = RidingCat()
        self.catTimer = 0.0
        self.catVisited = False

        # counters for the result card
        self.servedCount = 0
        self.spillCount = 0
        self.wrongCount = 0
        self.catResult = ""

        # the critic starts with the first course
        self.critic.setOrder(CRITIC_ORDER[0])
        self.critic.cooldown = 0.0

        self.debugText = arcade.Text("", 640, 112, (170, 255, 190), 13, font_name=FONT_BODY,
                                     anchor_x="center", anchor_y="center")

        # demo bot
        self.botHeldKey = None
        self.botServeWait = 0.0
        self.botWrongDone = False
        self.botSpinDemo = "waiting"
        self.botCatTime = 0.0
        self.botTarget = None
        self.botDirection = 0

    def getIntroLines(self):
        return [
            ("nenek", "The last course: everything together at the banquet table, like a real family dinner!"),
            ("leon", "Hmm. I will taste the rojak first, then the char kuey teow, then the teh tarik."),
            ("nenek", "Spin the lazy susan with LEFT and RIGHT. When a dish faces the guest who wants it, press SPACE."),
            ("aiman", "Gently though. Spin too fast and the teh tarik will fly!"),
        ]

    def onPlayStart(self):
        self.rumbleLoop = self.startLoop("susan_rumble", 0.0)

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------
    def pickOrderFor(self, guest):
        wanted = []
        for dishId, name, path in DISHES:
            taken = False
            for other in self.guests:
                if other.order == dishId:
                    taken = True
            if not taken and dishId not in CRITIC_ORDER[self.criticCourse:]:
                wanted.append(dishId)
        if len(wanted) == 0:
            wanted = ["nasilemak", "satay", "roticanai"]
        return random.choice(wanted)

    def updateOrders(self, dt):
        for guest in self.guests:
            guest.update(dt)
            if guest.order is None:
                if guest is self.critic:
                    # the critic tastes slowly, then asks for his next course
                    if self.criticCourse < len(CRITIC_ORDER) and len(self.flights) == 0:
                        guest.cooldown -= dt
                        if guest.cooldown <= 0:
                            guest.setOrder(CRITIC_ORDER[self.criticCourse])
                            self.playSfx("voice_leon", 0.8, speed=0.8)
                    continue
                guest.cooldown -= dt
                if guest.cooldown <= 0:
                    guest.setOrder(self.pickOrderFor(guest))
                continue
            guest.patience -= guest.drain * dt
            if guest.patience <= 0:
                self.orderTimedOut(guest)

    def orderTimedOut(self, guest):
        self.registerHit("miss", guest.x, guest.y + 40, "Too slow!")
        self.addScore(-30)
        self.playSfx("grumble", 0.8, x=guest.x)
        if guest is self.critic:
            guest.patience = 1.0          # the critic sighs and waits again
            self.popups.add("Hmm... I am still waiting.", guest.x + 250, guest.y + 30, CREAM, 17, 1.8, 20)
        else:
            guest.order = None
            guest.cooldown = random.uniform(2.0, 3.5)

    # ------------------------------------------------------------------
    # Serving
    # ------------------------------------------------------------------
    def dishInFrontOf(self, guest):
        """The dish whose angle is closest to the guest's seat (within the serve window)."""
        best = None
        bestGap = SERVE_WINDOW + 0.1
        for dish in self.dishes:
            if not dish.present:
                continue
            gap = abs(angleDifference(dish.angle(self.susanAngle), guest.seatAngle))
            if gap < bestGap:
                bestGap = gap
                best = dish
        return best

    def tryServeAny(self):
        """SPACE: serve the first guest whose wanted dish is lined up."""
        for guest in self.guests:
            if guest.order is None:
                continue
            dish = self.dishInFrontOf(guest)
            if dish is not None and dish.dishId == guest.order:
                self.serve(guest, dish)
                return
        self.playSfx("buzzer_wrong", 0.6)
        self.registerHit("miss", SUSAN_X, SUSAN_Y + 120, "Line it up first!")

    def tryServeGuest(self, guest):
        dish = self.dishInFrontOf(guest)
        if guest.order is None:
            self.popups.add("I'm still full!", guest.x, guest.y + 150, CREAM, 18, 1.2, 20)
            return
        if dish is None:
            self.popups.add("Nothing in front of me!", guest.x, guest.y + 150, CREAM, 18, 1.2, 20)
            return
        if dish.dishId != guest.order:
            self.wrongCount += 1
            self.playSfx("buzzer_wrong", 0.7, x=guest.x)
            self.registerHit("miss", guest.x, guest.y + 40, "That's not mine!")
            self.playSfx(SPEAKERS[guest.speaker]["voice"], 0.8, speed=SPEAKERS[guest.speaker]["pitch"] * 0.9)
            return
        self.serve(guest, dish)

    def serve(self, guest, dish):
        startX, startY = turntablePoint(dish.angle(self.susanAngle), DISH_RADIUS)
        self.flights.append({"guest": guest, "texture": dish.texture, "from": (startX, startY),
                             "tween": Tween(0.0, 1.0, 0.45)})
        dish.present = False
        dish.refillTimer = 3.0
        quality = "perfect" if guest.patience > 0.6 else "good"
        self.registerHit(quality, guest.x, guest.y + 40)
        self.addScore(int(40 * guest.patience))
        self.playSfx("dish_serve", 0.9, x=guest.x)
        self.servedCount += 1
        guest.order = None
        guest.cooldown = random.uniform(3.0, 5.0)
        if guest is self.critic:
            self.criticCourse += 1
            guest.goldGlow = 1.0
            self.addScore(100, guest.x - 250, guest.y + 30, "GOLDEN SPOON +100", GOLD)
            if self.criticCourse == 1:
                self.playSfx("vo_leon_hmm", 1.0)
            if self.criticCourse < len(CRITIC_ORDER):
                guest.cooldown = CRITIC_SAVOUR_TIME
                self.popups.add("Hmm... let me savour this.", guest.x + 250, guest.y + 30, CREAM, 17, 2.2, 20)
        if guest.speaker == "farah":
            # Farah photographs every dish: a camera flash (post-processing flash)
            self.flash.start(WHITE, 0.3, 210)
            self.playSfx("camera_shutter", 0.9, x=guest.x)

    def updateFlights(self, dt):
        keep = []
        for flight in self.flights:
            flight["tween"].update(dt)
            if flight["tween"].done:
                guest = flight["guest"]
                guest.react(2, 1.3)                     # eating frame
                spawnHearts(self.particles, guest.x, guest.y + 60, 7)
                spawnSparkles(self.glowParticles, guest.x, guest.y + 40, (255, 220, 150), 10, 180)
                self.playSfx("nom", 0.9, x=guest.x, pitchVariation=0.1)
                self.playSfx(SPEAKERS[guest.speaker]["voice"], 0.8, speed=SPEAKERS[guest.speaker]["pitch"])
                if guest is self.critic and self.criticCourse >= len(CRITIC_ORDER):
                    self.feastComplete()
            else:
                keep.append(flight)
        self.flights = keep

    def feastComplete(self):
        self.critic.react(3, 3.0)
        for guest in self.guests:
            guest.react(3, 2.5)
        spawnGlowBurst(self.glowParticles, self.critic.x, self.critic.y, (255, 220, 140), 520, 0.6)
        spawnRing(self.glowParticles, self.critic.x, self.critic.y, GOLD, 480)
        self.playSfx("applause", 0.8)
        bonus = int(self.timeLeft) * 3
        if bonus > 0:
            self.score += bonus
            self.popups.add(f"TIME BONUS +{bonus}", 640, 470, (120, 190, 255), 28, 1.5, 50)
        self.finishScene("A feast for the whole table!")

    def updateDishes(self, dt):
        for dish in self.dishes:
            if not dish.present:
                dish.refillTimer -= dt
                if dish.refillTimer <= 0:
                    # Grandma Siti brings a fresh plate
                    dish.present = True
                    dish.fadeIn = 0.0
                    x, y = turntablePoint(dish.angle(self.susanAngle), DISH_RADIUS)
                    spawnSparkles(self.glowParticles, x, y, (255, 230, 170), 8, 140)
                    self.playSfx("plop", 0.5, x=x)
            dish.fadeIn = approach(dish.fadeIn, 1.0, 5.0, dt)

    # ------------------------------------------------------------------
    # Oyen on the turntable
    # ------------------------------------------------------------------
    def updateCat(self, dt):
        # Oyen shows up once, after the critic's first course
        if not self.catVisited and self.criticCourse >= 1 and len(self.flights) == 0:
            self.catVisited = True
            self.cat.state = "jumping"
            self.cat.time = 0.0
            self.cat.jumpFrom = (-60, 120)
            self.playSfx("cat_meow", 0.9, x=100)
            self.popups.add("OYEN! Get off the table!", 640, 520, BAD_RED, 24, 1.8, 20)
        result = self.cat.update(dt, self.susanAngle)
        if result == "landed":
            self.catTimer = 6.0
            self.playSfx("wobble", 0.6, x=self.cat.x)
        if self.cat.state == "riding":
            self.catTimer -= dt
            if self.catTimer <= 0:
                # too slow: he eats the satay and leaves happy
                satay = self.dishById["satay"]
                if satay.present:
                    satay.present = False
                    satay.refillTimer = 8.0
                self.addScore(-50, self.cat.x, self.cat.y + 110, "Oyen ate the satay! -50", BAD_RED)
                self.playSfx("cat_meow", 1.0, x=self.cat.x, speed=1.15)
                self.catResult = "Oyen ate the satay"
                self.cat.leave()

    def shooCat(self):
        if self.cat.state != "riding":
            return
        self.playSfx("cat_hiss", 0.9, x=self.cat.x)
        spawnDust(self.particles, self.cat.x, self.cat.y + 20, 10)
        self.addScore(50, self.cat.x, self.cat.y + 110, "SHOO! +50", GOOD_GREEN)
        self.catResult = "Oyen shooed away"
        self.cat.leave()

    # ------------------------------------------------------------------
    # Input
    # ------------------------------------------------------------------
    def mouseAngleOnTable(self, x, y):
        """Undo the perspective squash, then atan2 gives the angle on the turntable."""
        return math.degrees(math.atan2((y - SUSAN_Y) / PERSPECTIVE, x - SUSAN_X))

    def onTurntable(self, x, y):
        dx = (x - SUSAN_X) / 320
        dy = (y - SUSAN_Y) / 115
        return dx * dx + dy * dy <= 1.0

    def onPlayKeyPress(self, key, modifiers):
        if key == arcade.key.SPACE:
            self.tryServeAny()
        elif key == arcade.key.S:
            self.shooCat()

    def onPlayMousePress(self, x, y, button):
        if button != arcade.MOUSE_BUTTON_LEFT:
            return
        if self.cat.contains(x, y):
            self.shooCat()
            return
        for guest in self.guests:
            if guest.contains(x, y):
                self.tryServeGuest(guest)
                return
        if self.onTurntable(x, y):
            self.draggingSusan = True
            self.lastMouseAngle = self.mouseAngleOnTable(x, y)

    def onPlayMouseDrag(self, x, y, dx, dy):
        if self.draggingSusan:
            angle = self.mouseAngleOnTable(x, y)
            change = angleDifference(self.lastMouseAngle, angle)      # new angle - old angle
            self.susanAngle += change
            # remember the speed so the turntable keeps spinning after release (momentum)
            self.spinSpeed = clamp(change * 60, -MAX_SPIN, MAX_SPIN)
            self.lastMouseAngle = angle

    def onPlayMouseRelease(self, x, y, button):
        self.draggingSusan = False

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------
    def updatePlay(self, dt):
        # Spin physics: keys accelerate, friction slows it down (exponential smoothing)
        if self.isKeyHeld(arcade.key.RIGHT) or self.isKeyHeld(arcade.key.D):
            self.spinSpeed += SPIN_ACCELERATION * dt
        if self.isKeyHeld(arcade.key.LEFT) or self.isKeyHeld(arcade.key.A):
            self.spinSpeed -= SPIN_ACCELERATION * dt
        self.spinSpeed = clamp(self.spinSpeed, -MAX_SPIN, MAX_SPIN)
        if not self.draggingSusan:
            self.spinSpeed = approach(self.spinSpeed, 0.0, SPIN_FRICTION, dt)
            self.susanAngle += self.spinSpeed * dt
        self.susanAngle = self.susanAngle % 360

        # Too fast: the dishes wobble and the teh tarik spills
        speed = abs(self.spinSpeed)
        self.spillCooldown = max(0.0, self.spillCooldown - dt)
        if speed > WOBBLE_SPEED:
            self.wobble = approach(self.wobble, 1.0, 8.0, dt)
            if self.spillCooldown <= 0:
                self.spillTea()
        else:
            self.wobble = approach(self.wobble, 0.0, 4.0, dt)
        self.window.sound.setVolume(self.rumbleLoop, min(0.8, speed / 260))

        self.updateOrders(dt)
        self.updateFlights(dt)
        self.updateDishes(dt)
        self.updateCat(dt)

    def spillTea(self):
        self.spillCooldown = 2.0
        self.spillCount += 1
        tea = self.dishById["tehtarik"]
        x, y = turntablePoint(tea.angle(self.susanAngle), DISH_RADIUS)
        spawnJuiceSplash(self.particles, x, y + 20, (190, 140, 95), 16, 0.9)
        for i in range(8):
            spawnDroplet(self.particles, x, y + 30, random.uniform(-160, 160), random.uniform(150, 320),
                         (190, 140, 95), 10, 1.0)
        self.playSfx("wobble", 0.9, x=x)
        self.playSfx("splash_1", 0.7, x=x)
        self.addScore(-30, x, y + 100, "Careful! -30", BAD_RED)
        self.shake.start(6, 0.3)

    # ------------------------------------------------------------------
    # Demo bot
    # ------------------------------------------------------------------
    def holdBotKey(self, key):
        if self.botHeldKey == key:
            return
        if self.botHeldKey is not None:
            self.botReleaseKey(self.botHeldKey)
        self.botHeldKey = key
        if key is not None:
            self.botPressKey(key)

    def chooseBotTarget(self):
        """The critic first when he is waiting, otherwise the most impatient guest."""
        best = None
        for guest in self.guests:
            if guest.order is None:
                continue
            dish = self.dishById[guest.order]
            if not dish.present:
                continue
            if guest is self.critic and guest.patience < 0.85:
                return guest
            if best is None or guest.patience < best.patience:
                best = guest
        return best

    def autoPlay(self, dt):
        if self.dialogue.isActive():
            self.holdBotKey(None)
            return
        # Oyen first: shoo him after a short moment
        if self.cat.state == "riding":
            self.botCatTime += dt
            if self.botCatTime > 1.2:
                self.holdBotKey(None)
                if self.botMoveCursor(self.cat.x, self.cat.y + 30, dt, 1500):
                    self.botClick()
                return
        # a deliberate mistake at the start: click Uncle Lim with the wrong dish in front
        lim = self.guests[1]
        if not self.botWrongDone and lim.order is not None:
            inFront = self.dishInFrontOf(lim)
            if inFront is None or inFront.dishId == lim.order:
                self.botWrongDone = True          # nothing wrong to show, skip the mistake
                return
            if self.botMoveCursor(lim.x, lim.y, dt, 1200):
                self.botClick()
                self.botWrongDone = True
            return
        # a deliberate over-spin once, to show the wobble and the spill
        if self.botSpinDemo == "waiting" and self.criticCourse >= 1 and self.cat.state == "away":
            self.botSpinDemo = "spinning"
            self.botServeWait = 1.6
        if self.botSpinDemo == "spinning":
            self.holdBotKey(arcade.key.RIGHT)
            self.botServeWait -= dt
            if self.botServeWait <= 0 or self.spillCount > 0:
                self.botSpinDemo = "done"
                self.holdBotKey(None)
            return

        # keep one target until it is served (no dithering between guests),
        # but switch to the critic if he gets impatient
        target = self.botTarget
        if target is None or target.order is None or not self.dishById[target.order].present:
            target = self.chooseBotTarget()
            self.botTarget = target
            self.botDirection = 0
        elif target is not self.critic and self.critic.order is not None and self.critic.patience < 0.5:
            target = self.critic
            self.botTarget = target
            self.botDirection = 0
        if target is None:
            self.holdBotKey(None)
            return
        dish = self.dishById[target.order]
        # how far the dish still has to turn to reach the guest's seat (seat - dish, -180..180)
        gap = angleDifference(dish.angle(self.susanAngle), target.seatAngle)
        # near 180 degrees the shortest way flips between left and right, so
        # pick one direction and stick to it until the dish is close
        if abs(gap) > 140:
            if self.botDirection == 0:
                self.botDirection = 1 if gap > 0 else -1
            gap = abs(gap) * self.botDirection
        else:
            self.botDirection = 0
        # P-controller: wanted spin speed is proportional to how far away the dish is
        wantedSpeed = clamp(gap * 3.0, -170, 170)
        if self.spinSpeed < wantedSpeed - 12:
            self.holdBotKey(arcade.key.RIGHT)
        elif self.spinSpeed > wantedSpeed + 12:
            self.holdBotKey(arcade.key.LEFT)
        else:
            self.holdBotKey(None)
        if abs(gap) < 7 and abs(self.spinSpeed) < 45:
            self.botServeWait += dt
            if self.botServeWait > 0.15:
                self.botServeWait = 0.0
                if target.speaker in ("farah", "kavitha"):
                    # sometimes click the guest instead of pressing SPACE
                    if self.botMoveCursor(target.x, target.y, dt, 1600):
                        self.botClick()
                else:
                    self.botTapKey(arcade.key.SPACE)

    def cleanupScene(self):
        self.flights = []

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------
    def getResultLines(self):
        lines = [f"Guests served: {self.servedCount}    Critic courses: {self.criticCourse}/3",
                 f"Spills: {self.spillCount}    Wrong dishes: {self.wrongCount}"]
        if self.catResult:
            lines.append(self.catResult)
        return lines

    def getDishTexture(self):
        return self.dishTexture

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------
    def drawWorld(self):
        assets.drawTexture(self.background, 640, 360, 1280, 720)
        # guests sit behind the table: back guests first (painter's algorithm)
        order = sorted(self.guests, key=guestBackToFront)
        for guest in order:
            guest.draw()
        assets.drawTexture(self.tableTexture, TABLE_X, TABLE_Y, 980, 360)
        spinTilt = clamp(self.spinSpeed / MAX_SPIN, -1, 1) * 0.6
        assets.drawTexture(self.susanTexture, SUSAN_X, SUSAN_Y, 640, 230, spinTilt)
        self.drawTurntableItems()
        for flight in self.flights:
            t = easeOutQuad(flight["tween"].value)
            guest = flight["guest"]
            x = lerp(flight["from"][0], guest.x, t)
            y = lerp(flight["from"][1], guest.y + 10, t) + math.sin(t * math.pi) * 90
            size = lerp(150, 110, t)
            assets.drawTexture(flight["texture"], x, y, size, None)

    def drawTurntableItems(self):
        """Dishes (and Oyen) sorted by depth, drawn back to front."""
        items = []
        for dish in self.dishes:
            if dish.present:
                angle = dish.angle(self.susanAngle)
                x, y = turntablePoint(angle, DISH_RADIUS)
                items.append((y, "dish", dish, x, angle))
        if self.cat.state == "riding":
            items.append((self.cat.y, "cat", None, self.cat.x, 0))
        items.sort(key=backToFront)        # bigger y = further back = drawn first
        for y, kind, dish, x, angle in items:
            if kind == "cat":
                self.cat.draw()
                continue
            # perspective: far dishes are smaller than near ones
            scale = lerp(1.08, 0.72, depthOf(angle))
            jitterX = 0.0
            jitterY = 0.0
            wobbleAngle = 0.0
            if self.wobble > 0.05:
                jitterX = random.uniform(-3, 3) * self.wobble
                jitterY = random.uniform(-2, 2) * self.wobble
                wobbleAngle = random.uniform(-6, 6) * self.wobble
            assets.drawTexture(dish.texture, x + jitterX, y + 18 * scale + jitterY, 150 * scale, 110 * scale,
                               wobbleAngle, alpha=255 * dish.fadeIn)
        if self.cat.state in ("jumping", "leaving"):
            self.cat.draw()

    def drawWorldTop(self):
        for guest in self.guests:
            if guest.order is not None:
                self.drawOrderBubble(guest)

    def drawOrderBubble(self, guest):
        pop = 1.0 + 0.3 * guest.bubblePop
        bx = guest.x + 58
        by = guest.y + 118
        assets.drawTexture(self.bubbleTexture, bx, by, 118 * pop, 108 * pop)
        dish = self.dishById[guest.order]
        assets.drawTexture(dish.texture, bx + 4, by + 8, 78 * pop, None)
        # patience ring: HSV hue slides from green (0.33) to red (0.0) as patience runs out
        colour = hsvColor(lerp(0.0, 0.33, clamp(guest.patience, 0.0, 1.0)), 0.8, 0.95)
        arcade.draw_arc_outline(bx + 2, by + 6, 104 * pop, 96 * pop, colour, 90, 90 + 360 * guest.patience, 6)
        # the dish in front of this guest right now (a hint)
        inFront = self.dishInFrontOf(guest)
        if inFront is not None and inFront.dishId == guest.order:
            pulse = 0.5 + 0.5 * math.sin(self.sceneTime * 10)
            arcade.draw_circle_outline(bx + 2, by + 6, 58 + 4 * pulse, withAlpha(GOLD, 200), 3)

    def drawPlayHud(self):
        # the critic's Golden Spoon rating: one star per course he has eaten
        drawPanel(150, 604, 250, 70, (14, 16, 38, 205), (255, 200, 87, 150), 16, shadow=False)
        drawText("GOLDEN SPOON RATING", 150, 624, GOLD, 12, FONT_BODY, anchorX="center", anchorY="center",
                 bold=True)
        drawStars(150, 594, self.criticCourse, 3, 30)
        nextCourse = ""
        if self.criticCourse < len(CRITIC_ORDER):
            nextCourse = "Next: " + self.dishById[CRITIC_ORDER[self.criticCourse]].name
        drawText(nextCourse, 150, 556, CREAM, 12, FONT_BODY, anchorX="center", anchorY="center")
        if abs(self.spinSpeed) > WOBBLE_SPEED * 0.8:
            drawText("TOO FAST!", 640, 470, BAD_RED, 20, FONT_BODY, anchorX="center", anchorY="center",
                     bold=True, alpha=200)
        drawKeyHint(380, 28, "LEFT/RIGHT", "Spin")
        drawKeyHint(640, 28, "SPACE", "Serve")
        drawKeyHint(840, 28, "S", "Shoo cat")

    def drawDebugWorld(self):
        """F2: the maths - perspective ellipse, rotation vectors, seat angles, draw order."""
        points = []
        for i in range(0, 361, 10):
            points.append(turntablePoint(i, DISH_RADIUS))
        arcade.draw_line_strip(points, (80, 255, 120), 1)
        order = []
        for dish in self.dishes:
            angle = dish.angle(self.susanAngle)
            x, y = turntablePoint(angle, DISH_RADIUS)
            arcade.draw_line(SUSAN_X, SUSAN_Y, x, y, (120, 180, 255), 2)
            order.append((y, dish.slot, x))
        order.sort(key=backToFront)
        for i in range(len(order)):
            y, slot, x = order[i]
            drawText(str(i + 1), x - 40, y + 40, (255, 120, 255), 14, FONT_BODY, bold=True)
        for guest in self.guests:
            sx, sy = turntablePoint(guest.seatAngle, DISH_RADIUS + 40)
            arcade.draw_line(SUSAN_X, SUSAN_Y, sx, sy, (255, 200, 87, 160), 1)
        self.debugText.text = (f"susan angle {self.susanAngle:6.1f} deg   spin {self.spinSpeed:6.1f} deg/s   "
                               f"x' = x cos t - y sin t,  y' = (x sin t + y cos t) * {PERSPECTIVE}")
        # a dark strip behind the text keeps it readable on busy backgrounds
        width = self.debugText.content_width
        arcade.draw_lbwh_rectangle_filled(self.debugText.x - width / 2 - 8, self.debugText.y - 12, width + 16, 24,
                                          (0, 0, 0, 180))
        self.debugText.draw()
