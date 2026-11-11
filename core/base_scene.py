"""
base_scene.py
-------------
BaseScene(arcade.View) is the parent class of every scene in the game.
It does all the shared work so each scene file can focus on its own idea.

STATE MACHINE (a finite state machine keeps the flow easy to follow):

    "intro"  -> dialogue from getIntroLines()
    "countdown" (dining scenes only) -> 3, 2, 1, COOK!
    "play"   -> the scene's own gameplay (updatePlay / autoPlay)
    "outro"  -> dish reveal (the finished dish grows in with a rotating glow)
    "result" -> ResultCard, then ENTER / click goes to the next scene

Non-gameplay scenes (title, story, street, finale) go intro -> play and
manage their own flow inside "play".

It also owns the shared effect objects (particles, lights, screen shake,
flash, popups, iris transition, dialogue box, HUD) and routes input:
real keyboard/mouse events call handleKeyPress / handleMousePress ...
and in --autoplay mode the demo bot calls those SAME methods, so the bot
and a human player run exactly the same code.

Child scenes override the hook methods (setupScene, updatePlay, drawWorld,
onPlayKeyPress, autoPlay ...) - simple one-level inheritance.
"""
import math

import arcade

from core import assets
from core.assets import drawTexture
from core.effects import (FloatingTextManager, LightList, ScreenFlash, ScreenShake, Transition,
                          drawAuraOverlay, drawVignette)
from core.helpers import approach, clamp, easeOutBack, easeOutQuad, lerp
from core.particles import ParticleSystem, spawnDust, spawnGlowBurst, spawnRing, spawnSparkles
from core.ui import (Button, DialogueBox, Hud, ResultCard, drawKeyHint, drawPanel, drawText, drawTextTexture,
                     makeTextTexture, setTextColor, withAlpha)
from settings import (AURA_MODE_TIME, AURA_SCORE_MULTIPLIER, BAD_RED, COMBO_FOR_AURA, COUNTDOWN_TIME,
                      CREAM, DISH_NAMES, FONT_BODY, FONT_NARROW, FONT_TITLE, GOLD, GOOD_GREEN,
                      LAVENDER, MAX_DELTA_TIME, RECORD_FPS, SCREEN_HEIGHT, SCREEN_WIDTH, SKY_BLUE,
                      WARM_LIGHT, WHITE)

OUTRO_TIME = 2.6              # seconds of dish reveal before the result card
RESULT_AUTOPLAY_WAIT = 2.5    # autoplay presses ENTER on the result card after this
OVERLAY_AUTOPLAY_TIME = 2.5   # in autoplay, help / pause overlays close by themselves
COUNTDOWN_STEP = 0.8          # "3", "2", "1" each show for this long, then "COOK!"

HIT_POINTS = {"perfect": 100, "good": 60, "miss": 0}

GLOBAL_KEYS_HELP = [("F1", "Help"), ("F2", "Debug maths"), ("F3", "Music on/off"),
                    ("F12", "Screenshot"), ("ESC", "Pause / back")]


class BaseScene(arcade.View):
    # ------------------------------------------------------------------
    # Class attributes that each scene overrides
    # ------------------------------------------------------------------
    sceneId = "base"
    sceneTitle = ""
    stallName = ""
    dishName = ""
    musicName = None
    auraMusicName = None
    ambienceName = None
    timeLimit = 0             # seconds, 0 = no timer
    maxScore = 1000
    isGameplayScene = False   # True for the four dining scenes
    controlsHelp = []         # list of (key, text) for the F1 help panel
    useVignette = True

    def __init__(self):
        super().__init__()
        # Shared effect systems
        self.particles = ParticleSystem(additive=False)
        self.glowParticles = ParticleSystem(additive=True)
        self.lights = LightList()
        self.shake = ScreenShake()
        self.flash = ScreenFlash()
        self.popups = FloatingTextManager()
        self.transition = Transition()
        self.dialogue = DialogueBox()
        self.hud = Hud()
        self.resultCard = None

        # Input
        self.keysHeld = set()
        self.mouseButtonsHeld = set()
        self.mouseX = SCREEN_WIDTH / 2
        self.mouseY = SCREEN_HEIGHT / 2

        # State machine
        self.state = "intro"
        self.stateTime = 0.0
        self.sceneTime = 0.0
        self.paused = False
        self.showHelp = False
        self.overlayTime = 0.0
        self.sceneEnded = False
        self.nextSceneId = None
        self.finishMessage = ""
        self.screenshotPending = False
        self.countdownIndex = -1

        # Score, combo and Aura Mode
        self.score = 0
        self.combo = 0
        self.bestCombo = 0
        self.auraCharge = 0          # good actions since the last miss / last Aura Mode
        self.perfectCount = 0
        self.goodCount = 0
        self.missCount = 0
        self.auraMode = False
        self.auraTimer = 0.0
        self.auraGlow = 0.0          # 0..1 smoothed strength of the golden overlay
        self.timeLeft = self.timeLimit
        self.earnedStars = 0

        # Cameras: the world camera can move / shake, the GUI camera never moves
        self.worldCamera = arcade.Camera2D()
        self.guiCamera = arcade.Camera2D()

        # Autoplay (demo bot) fields
        self.botX = SCREEN_WIDTH / 2
        self.botY = SCREEN_HEIGHT / 2 - 60
        self.botMouseHeld = False
        self.botCursorVisible = False
        self.botTimer = 0.0
        self.botStep = 0
        self.botRipples = []         # little click circles drawn at the bot cursor

        # Text objects made once (never inside on_draw)
        self.revealTitle = arcade.Text("", SCREEN_WIDTH / 2, 170, GOLD, 40, font_name=FONT_TITLE,
                                       anchor_x="center", anchor_y="center")
        self.revealMessage = arcade.Text("", SCREEN_WIDTH / 2, 120, CREAM, 20, font_name=FONT_TITLE,
                                         anchor_x="center", anchor_y="center")
        # One Text per debug line (pyglet multiline text can drop spaces)
        self.debugTexts = []
        for i in range(5):
            self.debugTexts.append(arcade.Text("", 16, SCREEN_HEIGHT - 80 - i * 24, (200, 255, 200), 13,
                                               font_name=FONT_BODY, anchor_y="top"))
        self.pauseButtons = []
        self.makePauseButtons()

        self.setupScene()

    # ==================================================================
    # Hooks for child scenes (override the ones you need)
    # ==================================================================
    def setupScene(self):
        pass

    def getIntroLines(self):
        return []

    def onPlayStart(self):
        """Called once when the state becomes "play"."""
        pass

    def updatePlay(self, dt):
        pass

    def updateAlways(self, dt):
        pass

    def drawWorld(self):
        pass

    def drawWorldTop(self):
        pass

    def drawPlayHud(self):
        pass

    def drawDebugWorld(self):
        pass

    def onPlayKeyPress(self, key, modifiers):
        pass

    def onPlayKeyRelease(self, key, modifiers):
        pass

    def onPlayMousePress(self, x, y, button):
        pass

    def onPlayMouseRelease(self, x, y, button):
        pass

    def onPlayMouseDrag(self, x, y, dx, dy):
        pass

    def onPlayMouseMotion(self, x, y, dx, dy):
        pass

    def onPlayMouseScroll(self, x, y, scrollY):
        pass

    def autoPlay(self, dt):
        pass

    def onTimeUp(self):
        self.finishScene("TIME'S UP!")

    def getResultLines(self):
        lines = []
        lines.append(f"Perfect {self.perfectCount}   Good {self.goodCount}   Miss {self.missCount}")
        return lines

    def getDishTexture(self):
        return None

    def cleanupScene(self):
        pass

    def onEscape(self):
        """ESC in a non-gameplay scene (default: back to the title)."""
        if self.sceneId != "title":
            self.changeScene("title")

    def getNextSceneId(self):
        gameState = self.window.gameState
        if gameState.storyMode:
            if self.sceneId == "banquet":
                return "finale"
            return "street"
        return "title"

    # ==================================================================
    # Showing / hiding
    # ==================================================================
    def on_show_view(self):
        sound = self.window.sound
        if self.musicName is not None:
            sound.playMusic(self.musicName, self.auraMusicName)
        sound.playAmbience(self.ambienceName)
        self.transition.startOpen(duration=0.6)
        lines = self.getIntroLines()
        if lines:
            self.dialogue.start(lines)
            self.setState("intro")
        else:
            self.enterAfterIntro()

    def on_hide_view(self):
        self.window.sound.stopAllLoops()
        self.window.sound.pauseLoops(False)
        self.cleanupScene()

    # ==================================================================
    # Update
    # ==================================================================
    def on_update(self, delta_time):
        if self.window.recording:
            return    # in record mode the simulation runs inside on_draw
        dt = min(delta_time, MAX_DELTA_TIME)
        for i in range(self.window.turbo):
            if self.window.current_view is not self:
                break
            self.tick(dt)

    def setState(self, newState):
        self.state = newState
        self.stateTime = 0.0
        if newState == "play":
            self.onPlayStart()

    def enterAfterIntro(self):
        if self.isGameplayScene:
            self.setState("countdown")
            self.countdownIndex = -1
        else:
            self.setState("play")

    def tick(self, dt):
        """One simulation step (called by on_update, or by on_draw when recording)."""
        window = self.window
        window.gameClock += dt
        window.sound.update(dt)
        self.transition.update(dt)
        if window.current_view is not self:
            return    # the transition just switched to the next scene

        # Overlays: the game is frozen while paused or showing help
        if self.paused or self.showHelp:
            self.overlayTime += dt
            if window.autoplay and self.overlayTime > OVERLAY_AUTOPLAY_TIME:
                self.showHelp = False
                self.setPaused(False)
            return

        self.stateTime += dt
        self.sceneTime += dt
        self.dialogue.update(dt, autoAdvance=window.autoplay)
        self.shake.update(dt)
        self.flash.update(dt)
        self.popups.update(dt)
        self.particles.update(dt)
        self.glowParticles.update(dt)
        self.lights.update(dt)
        self.updateBotRipples(dt)
        self.updateAlways(dt)
        if window.current_view is not self:
            return

        if self.state == "intro":
            if not self.dialogue.isActive():
                self.enterAfterIntro()
        elif self.state == "countdown":
            self.updateCountdown()
        elif self.state == "play":
            self.updatePlayState(dt)
        elif self.state == "outro":
            if self.stateTime >= OUTRO_TIME:
                self.showResultCard()
        elif self.state == "result":
            if self.resultCard is not None:
                self.resultCard.update(dt)
                if window.autoplay and self.stateTime >= RESULT_AUTOPLAY_WAIT and self.resultCard.isDone():
                    self.continueFromResult()

        # Aura Mode golden overlay fades in and out smoothly
        if self.auraMode:
            self.auraGlow = approach(self.auraGlow, 1.0, 6.0, dt)
        else:
            self.auraGlow = approach(self.auraGlow, 0.0, 4.0, dt)
        # Music: the aura layer grows with the combo and is full in Aura Mode
        if self.auraMode:
            window.sound.setAuraLevel(1.0)
        else:
            window.sound.setAuraLevel(min(self.auraCharge / COMBO_FOR_AURA, 1.0) * 0.5)

    def updateCountdown(self):
        # 3 (0.0s), 2 (0.8s), 1 (1.6s), COOK! (2.4s) then play at COUNTDOWN_TIME
        index = int(self.stateTime / COUNTDOWN_STEP)
        if index != self.countdownIndex and index <= 3:
            self.countdownIndex = index
            if index < 3:
                self.playSfx("countdown_beep", 0.8)
            else:
                self.playSfx("go_whistle", 0.8)
                self.playSfx("vo_cook", 1.0)
        if self.stateTime >= COUNTDOWN_TIME:
            self.setState("play")

    def updatePlayState(self, dt):
        # Timer (frozen while a dialogue is open in the middle of play)
        if self.isGameplayScene and self.timeLimit > 0 and not self.dialogue.isActive():
            previousSeconds = int(math.ceil(self.timeLeft))
            self.timeLeft -= dt
            seconds = int(math.ceil(self.timeLeft))
            if seconds != previousSeconds and 0 < seconds <= 5:
                self.playSfx("countdown_beep", 0.45, speed=1.3)
            if self.timeLeft <= 0:
                self.timeLeft = 0
                self.playSfx("vo_timesup", 1.0)
                self.onTimeUp()
                if self.state != "play":
                    return
        # Aura Mode timer
        if self.auraMode:
            self.auraTimer -= dt
            if self.auraTimer <= 0:
                self.auraTimer = 0
                self.auraMode = False
                self.auraCharge = 0
        self.updatePlay(dt)
        if self.window.current_view is not self or self.state != "play":
            return
        if self.window.autoplay:
            self.autoPlay(dt)

    # ==================================================================
    # Scoring helpers
    # ==================================================================
    def addScore(self, points, x=None, y=None, label=None, rgb=None):
        """Add points (doubled in Aura Mode) and show a popup at (x, y). Returns the points added."""
        points = int(points)
        if self.auraMode and points > 0:
            points = points * AURA_SCORE_MULTIPLIER
        self.score = max(0, self.score + points)
        if x is not None and y is not None:
            if label is None:
                if points >= 0:
                    label = f"+{points}"
                else:
                    label = f"{points}"
            if rgb is None:
                if points >= 0:
                    rgb = GOLD
                else:
                    rgb = BAD_RED
            self.popups.add(label, x, y, rgb, 26)
        return points

    def registerHit(self, quality, x, y, label=None):
        """
        One player action judged as "perfect", "good" or "miss".
        Updates combo and counters, plays sounds, shows popups and sparkles,
        and starts Aura Mode after COMBO_FOR_AURA good actions in a row.
        """
        points = HIT_POINTS.get(quality, 0)
        if quality == "perfect" or quality == "good":
            self.combo += 1
            self.auraCharge += 1
            self.bestCombo = max(self.bestCombo, self.combo)
            if quality == "perfect":
                self.perfectCount += 1
                self.playSfx("perfect", 0.8, x=x)
                if label is None:
                    label = "PERFECT!"
                self.popups.add(label, x, y + 44, GOLD, 30, 1.0, 60)
                spawnSparkles(self.glowParticles, x, y, GOLD, 22, 280)
            else:
                self.goodCount += 1
                self.playSfx("good", 0.7, x=x)
                if label is None:
                    label = "GOOD"
                self.popups.add(label, x, y + 44, GOOD_GREEN, 26, 0.9, 55)
                spawnSparkles(self.glowParticles, x, y, (190, 255, 200), 10, 200)
            added = self.addScore(points, x, y)
            if self.combo >= 3 and self.combo % 5 == 0:
                self.popups.add(f"{self.combo} COMBO!", x, y + 90, SKY_BLUE, 22, 1.0, 50)
            if not self.auraMode and self.auraCharge >= COMBO_FOR_AURA:
                self.startAura()
            return added
        # miss
        self.combo = 0
        self.auraCharge = 0
        self.missCount += 1
        self.playSfx("miss", 0.8, x=x)
        if label is None:
            label = "MISS"
        self.popups.add(label, x, y + 30, BAD_RED, 28, 0.9, 45)
        self.shake.start(5, 0.22)
        spawnDust(self.particles, x, y, 5)
        return 0

    def startAura(self):
        """Aura Mode: 6 s of golden glow, double points, and the music aura layer at full volume."""
        self.auraMode = True
        self.auraTimer = AURA_MODE_TIME
        self.auraCharge = 0
        self.playSfx("aura_mode", 0.9)
        self.playSfx("vo_aura", 1.0)
        self.flash.start(GOLD, 0.4, 150)
        self.shake.start(6, 0.3)
        self.popups.add("AURA MODE!", SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2 + 80, GOLD, 46, 1.6, 50)
        self.popups.add("DOUBLE POINTS", SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2 + 30, CREAM, 22, 1.6, 50)
        spawnSparkles(self.glowParticles, SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2 + 60, GOLD, 40, 420)
        spawnRing(self.glowParticles, SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2 + 60, GOLD, 520)

    def finishScene(self, message=""):
        """End the gameplay: save the result, then dish reveal -> result card."""
        if self.state in ("outro", "result") or self.sceneEnded:
            return
        self.finishMessage = message
        self.auraMode = False
        self.auraTimer = 0
        self.window.sound.stopAllLoops()
        self.earnedStars = self.window.gameState.recordResult(self.sceneId, self.score, self.maxScore,
                                                              self.bestCombo)
        self.setState("outro")
        self.playSfx("fanfare", 0.8)
        self.playSfx("vo_delicious", 1.0)
        cx = SCREEN_WIDTH / 2
        cy = SCREEN_HEIGHT / 2 + 40
        spawnGlowBurst(self.glowParticles, cx, cy, WARM_LIGHT, 520, 0.6)
        spawnSparkles(self.glowParticles, cx, cy, GOLD, 45, 460)
        spawnRing(self.glowParticles, cx, cy, GOLD, 560)
        self.flash.start(WHITE, 0.35, 160)
        dishName = self.dishName
        if dishName == "" and self.sceneId in DISH_NAMES:
            dishName = DISH_NAMES[self.sceneId]
        self.revealTitle.text = dishName.upper()
        self.revealMessage.text = message

    def showResultCard(self):
        dishName = self.dishName
        if dishName == "" and self.sceneId in DISH_NAMES:
            dishName = DISH_NAMES[self.sceneId]
        self.resultCard = ResultCard(self.sceneTitle, dishName, self.getDishTexture(), self.score,
                                     self.maxScore, self.earnedStars, self.bestCombo, self.getResultLines())
        self.setState("result")

    def continueFromResult(self):
        if self.resultCard is None:
            return
        if not self.resultCard.isDone():
            self.resultCard.skip()
            return
        self.playSfx("ui_confirm", 0.7)
        self.changeScene(self.getNextSceneId())

    def changeScene(self, sceneId, x=None, y=None):
        """Iris-close, then show the new scene (only once)."""
        if self.sceneEnded:
            return
        self.sceneEnded = True
        self.nextSceneId = sceneId
        self.transition.startClose(x, y, 0.55, self.switchToNextScene)

    def switchToNextScene(self):
        # Imported here so scene files never import each other in a circle
        from scenes import makeScene
        self.window.show_view(makeScene(self.nextSceneId))

    # ==================================================================
    # Sound helpers
    # ==================================================================
    def playSfx(self, name, volume=1.0, x=None, pitchVariation=0.0, speed=1.0, cooldown=0.0):
        sound = self.window.sound
        if x is not None:
            return sound.playAt(name, x, volume, pitchVariation, speed, cooldown)
        if pitchVariation > 0:
            import random
            speed = speed * random.uniform(1.0 - pitchVariation, 1.0 + pitchVariation)
        return sound.play(name, volume, 0.0, speed, False, cooldown)

    def startLoop(self, name, volume=1.0, speed=1.0):
        """Looping sound effect; stopped automatically when the scene ends."""
        return self.window.sound.startLoop(name, volume, speed)

    def isKeyHeld(self, key):
        return key in self.keysHeld

    # ==================================================================
    # Input (real events are ignored in autoplay; the bot calls handle* directly)
    # ==================================================================
    def on_key_press(self, key, modifiers):
        if self.window.autoplay:
            return
        self.handleKeyPress(key, modifiers)

    def on_key_release(self, key, modifiers):
        if self.window.autoplay:
            return
        self.handleKeyRelease(key, modifiers)

    def on_mouse_press(self, x, y, button, modifiers):
        if self.window.autoplay:
            return
        self.handleMousePress(x, y, button)

    def on_mouse_release(self, x, y, button, modifiers):
        if self.window.autoplay:
            return
        self.handleMouseRelease(x, y, button)

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        if self.window.autoplay:
            return
        self.handleMouseDrag(x, y, dx, dy)

    def on_mouse_motion(self, x, y, dx, dy):
        if self.window.autoplay:
            return
        self.handleMouseMotion(x, y, dx, dy)

    def on_mouse_scroll(self, x, y, scroll_x, scroll_y):
        if self.window.autoplay:
            return
        self.handleMouseScroll(x, y, scroll_y)

    def handleKeyPress(self, key, modifiers):
        self.keysHeld.add(key)
        if self.sceneEnded:
            return
        # Global keys
        if key == arcade.key.F1:
            self.showHelp = not self.showHelp
            self.overlayTime = 0.0
            return
        if key == arcade.key.F2:
            self.window.showDebug = not self.window.showDebug
            return
        if key == arcade.key.F3:
            muted = self.window.sound.toggleMusicMute()
            if muted:
                self.popups.add("Music off", SCREEN_WIDTH / 2, 600, CREAM, 20, 1.0, 30)
            else:
                self.popups.add("Music on", SCREEN_WIDTH / 2, 600, CREAM, 20, 1.0, 30)
            return
        if key == arcade.key.F12:
            self.screenshotPending = True
            return
        if self.showHelp:
            if key in (arcade.key.ESCAPE, arcade.key.ENTER, arcade.key.RETURN, arcade.key.SPACE):
                self.showHelp = False
            return
        if self.paused:
            self.handlePauseKey(key)
            return
        if key == arcade.key.ESCAPE:
            if self.isGameplayScene and self.state in ("intro", "countdown", "play"):
                self.setPaused(True)
            else:
                self.onEscape()
            return
        if self.dialogue.isActive():
            if key in (arcade.key.SPACE, arcade.key.ENTER, arcade.key.RETURN):
                self.dialogue.advance()
            elif key == arcade.key.TAB:
                self.dialogue.skipAll()
            return
        if self.state == "result":
            if key in (arcade.key.ENTER, arcade.key.RETURN, arcade.key.SPACE):
                self.continueFromResult()
            return
        if self.state == "play":
            self.onPlayKeyPress(key, modifiers)

    def handleKeyRelease(self, key, modifiers):
        self.keysHeld.discard(key)
        if self.state == "play" and not self.paused and not self.sceneEnded:
            self.onPlayKeyRelease(key, modifiers)

    def handleMousePress(self, x, y, button):
        self.mouseX = x
        self.mouseY = y
        self.mouseButtonsHeld.add(button)
        if self.sceneEnded:
            return
        if self.showHelp:
            self.showHelp = False
            return
        if self.paused:
            for pauseButton in self.pauseButtons:
                if pauseButton.contains(x, y):
                    self.handlePauseKey(pauseButton.pauseKey)
            return
        if self.dialogue.isActive():
            if button == arcade.MOUSE_BUTTON_LEFT:
                self.dialogue.advance()
            return
        if self.state == "result":
            self.continueFromResult()
            return
        if self.state == "play":
            self.onPlayMousePress(x, y, button)

    def handleMouseRelease(self, x, y, button):
        self.mouseX = x
        self.mouseY = y
        self.mouseButtonsHeld.discard(button)
        if self.state == "play" and not self.paused and not self.sceneEnded:
            self.onPlayMouseRelease(x, y, button)

    def handleMouseDrag(self, x, y, dx, dy):
        self.mouseX = x
        self.mouseY = y
        if self.state == "play" and not self.paused and not self.sceneEnded:
            self.onPlayMouseDrag(x, y, dx, dy)

    def handleMouseMotion(self, x, y, dx, dy):
        self.mouseX = x
        self.mouseY = y
        if self.paused:
            for pauseButton in self.pauseButtons:
                pauseButton.updateHover(x, y)
            return
        if self.state == "play" and not self.sceneEnded:
            self.onPlayMouseMotion(x, y, dx, dy)

    def handleMouseScroll(self, x, y, scrollY):
        if self.state == "play" and not self.paused and not self.sceneEnded:
            self.onPlayMouseScroll(x, y, scrollY)

    # ------------------------------------------------------------------
    # Pause menu
    # ------------------------------------------------------------------
    def makePauseButtons(self):
        labels = [("RESUME", arcade.key.ESCAPE), ("RESTART", arcade.key.R),
                  ("GO TO STREET", arcade.key.B), ("QUIT TO TITLE", arcade.key.Q)]
        y = 420
        for label, key in labels:
            pauseButton = Button(label, SCREEN_WIDTH / 2, y, 360, 56, 18)
            pauseButton.pauseKey = key
            self.pauseButtons.append(pauseButton)
            y -= 72

    def setPaused(self, paused):
        if paused == self.paused:
            return
        self.paused = paused
        self.overlayTime = 0.0
        self.window.sound.pauseLoops(paused)
        if paused:
            self.playSfx("ui_pause", 0.6)
        else:
            self.playSfx("ui_back", 0.5)

    def handlePauseKey(self, key):
        if key == arcade.key.ESCAPE or key == arcade.key.P:
            self.setPaused(False)
        elif key == arcade.key.R:
            self.setPaused(False)
            self.changeScene(self.sceneId)
        elif key == arcade.key.B:
            self.setPaused(False)
            if self.window.gameState.storyMode:
                self.changeScene("street")
            else:
                self.changeScene("title")
        elif key == arcade.key.Q:
            self.setPaused(False)
            self.changeScene("title")

    # ==================================================================
    # Autoplay bot helpers (a virtual mouse cursor + key presses)
    # ==================================================================
    def botMoveCursor(self, tx, ty, dt, speed=900):
        """Glide the virtual cursor towards (tx, ty). Returns True when it has arrived."""
        self.botCursorVisible = True
        dx = tx - self.botX
        dy = ty - self.botY
        gap = math.hypot(dx, dy)
        if gap < 1.5:
            self.botX = tx
            self.botY = ty
            return True
        # Fast when far away, slowing down near the target (like a real hand)
        velocity = max(140.0, min(speed, gap * 9.0))
        step = velocity * dt
        if step >= gap:
            newX = tx
            newY = ty
        else:
            newX = self.botX + dx / gap * step
            newY = self.botY + dy / gap * step
        moveX = newX - self.botX
        moveY = newY - self.botY
        self.botX = newX
        self.botY = newY
        if self.botMouseHeld:
            self.handleMouseDrag(self.botX, self.botY, moveX, moveY)
        else:
            self.handleMouseMotion(self.botX, self.botY, moveX, moveY)
        return step >= gap

    def botMouseDown(self, button=arcade.MOUSE_BUTTON_LEFT):
        self.botCursorVisible = True
        self.botMouseHeld = True
        self.botRipples.append([self.botX, self.botY, 0.0])
        self.handleMousePress(self.botX, self.botY, button)

    def botMouseUp(self, button=arcade.MOUSE_BUTTON_LEFT):
        self.botMouseHeld = False
        self.handleMouseRelease(self.botX, self.botY, button)

    def botClick(self, button=arcade.MOUSE_BUTTON_LEFT):
        self.botMouseDown(button)
        self.botMouseUp(button)

    def botPressKey(self, key):
        self.handleKeyPress(key, 0)

    def botReleaseKey(self, key):
        self.handleKeyRelease(key, 0)

    def botTapKey(self, key):
        self.botPressKey(key)
        self.botReleaseKey(key)

    def updateBotRipples(self, dt):
        alive = []
        for ripple in self.botRipples:
            ripple[2] += dt
            if ripple[2] < 0.4:
                alive.append(ripple)
        self.botRipples = alive

    # ==================================================================
    # Drawing
    # ==================================================================
    def on_draw(self):
        window = self.window
        if window.recording:
            # Deterministic recording: exactly one simulation step per video frame
            self.tick(1.0 / RECORD_FPS)
        self.clear()

        # --- world layer (moves with the camera and the screen shake) ---
        shakeX, shakeY = self.shake.getOffset()
        basePosition = self.worldCamera.position
        self.worldCamera.position = (basePosition[0] + shakeX, basePosition[1] + shakeY)
        self.worldCamera.use()
        self.drawWorld()
        self.particles.draw()
        self.lights.draw()
        self.glowParticles.draw()
        self.drawWorldTop()
        if window.showDebug:
            self.drawDebugWorld()
        self.worldCamera.position = basePosition

        # --- GUI layer (fixed on screen) ---
        self.guiCamera.use()
        if self.useVignette:
            drawVignette()
        if self.auraGlow > 0.01:
            drawAuraOverlay(window.gameClock, self.auraGlow)
        if self.state in ("countdown", "play"):
            if self.isGameplayScene:
                self.hud.draw(self)
            self.drawPlayHud()
        self.popups.draw()
        if self.state == "countdown" or (self.state == "play" and self.isGameplayScene and self.stateTime < 0.6):
            self.drawCountdown()
        if self.state == "outro":
            self.drawOutroReveal()
        if self.state == "result" and self.resultCard is not None:
            self.resultCard.draw()
        self.dialogue.draw()
        if self.showHelp:
            self.drawHelpOverlay()
        elif self.paused:
            self.drawPauseMenu()
        if window.showDebug:
            self.drawDebugText()
        self.flash.draw()
        self.drawBotCursor()
        self.transition.draw()

        # F12 screenshot: taken now (before the "saved" popup exists)
        if self.screenshotPending:
            self.screenshotPending = False
            path = window.takeScreenshot()
            if path:
                self.popups.add("Screenshot saved", SCREEN_WIDTH / 2, 600, CREAM, 20, 1.2, 30)
        window.afterDraw()

    def drawCountdown(self):
        if self.state == "countdown":
            index = min(3, int(self.stateTime / COUNTDOWN_STEP))
            localTime = self.stateTime - index * COUNTDOWN_STEP
            fade = 1.0
        else:
            index = 3
            localTime = COUNTDOWN_TIME - 3 * COUNTDOWN_STEP + self.stateTime
            fade = clamp(1.0 - self.stateTime / 0.6, 0.0, 1.0)
        labels = ["3", "2", "1", "COOK!"]
        label = labels[index]
        popT = clamp(localTime / 0.35, 0.0, 1.0)
        scale = lerp(0.3, 1.0, easeOutBack(popT))
        if index == 3:
            baseSize = 96
            colour = GOLD
        else:
            baseSize = 130
            colour = CREAM
        alpha = int(255 * fade)
        cx = SCREEN_WIDTH / 2
        cy = SCREEN_HEIGHT / 2 + 40
        # glowing disc behind the number
        drawTexture(assets.loadTexture("particles/light_glow.png"), cx, cy, 420 * scale, 420 * scale,
                    rgb=colour, alpha=150 * fade, additive=True)
        # The word is rendered once into a texture, then only scaled (smooth zoom, no new font sizes)
        texture = makeTextTexture(label, baseSize, "Kenney_Future.ttf", colour, 5)
        drawTextTexture(texture, cx + 5, cy - 6, scale, alpha * 0.6, (0, 0, 0))      # drop shadow
        drawTextTexture(texture, cx, cy, scale, alpha)

    def drawOutroReveal(self):
        """The finished dish grows in (easeOutBack) with rotating light rays behind it."""
        t = self.stateTime
        cx = SCREEN_WIDTH / 2
        cy = SCREEN_HEIGHT / 2 + 40
        dim = clamp(t / 0.4, 0.0, 1.0)
        arcade.draw_lbwh_rectangle_filled(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT, (8, 8, 20, int(160 * dim)))
        # rotating rays (triangles around the centre, rotated with sin/cos)
        rayLength = 520 * easeOutQuad(clamp(t / 0.6, 0.0, 1.0))
        spin = t * 25
        ctx = self.window.ctx
        ctx.blend_func = assets.additiveBlend()
        for i in range(12):
            angle = math.radians(spin + i * 30)
            spread = math.radians(6)
            x1 = cx + math.cos(angle - spread) * rayLength
            y1 = cy + math.sin(angle - spread) * rayLength
            x2 = cx + math.cos(angle + spread) * rayLength
            y2 = cy + math.sin(angle + spread) * rayLength
            arcade.draw_triangle_filled(cx, cy, x1, y1, x2, y2, (255, 210, 120, int(38 * dim)))
        ctx.blend_func = ctx.BLEND_DEFAULT
        drawTexture(assets.loadTexture("particles/light_glow.png"), cx, cy, 560, 560, rgb=WARM_LIGHT,
                    alpha=200 * dim, additive=True)
        texture = self.getDishTexture()
        grow = easeOutBack(clamp((t - 0.15) / 0.8, 0.0, 1.0))
        if texture is not None and grow > 0.01:
            fit = min(420 / texture.width, 300 / texture.height)
            bob = math.sin(t * 2.5) * 4
            drawTexture(texture, cx, cy + bob, texture.width * fit * grow, texture.height * fit * grow)
        textFade = clamp((t - 0.6) / 0.4, 0.0, 1.0)
        if textFade > 0:
            setTextColor(self.revealTitle, withAlpha(GOLD, 255 * textFade))
            self.revealTitle.draw()
            if self.revealMessage.text != "":
                setTextColor(self.revealMessage, withAlpha(CREAM, 255 * textFade))
                self.revealMessage.draw()

    def drawHelpOverlay(self):
        arcade.draw_lbwh_rectangle_filled(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT, (6, 6, 16, 170))
        rows = len(self.controlsHelp) + len(GLOBAL_KEYS_HELP)
        height = 170 + rows * 40
        top = SCREEN_HEIGHT / 2 + height / 2
        drawPanel(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2, 640, height)
        title = self.sceneTitle
        if title == "":
            title = "Controls"
        drawText(f"HOW TO PLAY - {title.upper()}", SCREEN_WIDTH / 2, top - 44, GOLD, 22, FONT_TITLE,
                 anchorX="center", anchorY="center")
        y = top - 100
        for key, text in self.controlsHelp:
            drawKeyHint(420, y, key, text)
            y -= 40
        drawText("GLOBAL KEYS", 420 - 20, y - 4, LAVENDER, 13, FONT_NARROW, anchorY="center", shadow=False)
        y -= 36
        for key, text in GLOBAL_KEYS_HELP:
            drawKeyHint(420, y, key, text)
            y -= 40
        drawText("Press F1 or ESC to close", SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2 - height / 2 + 26, CREAM, 14,
                 FONT_BODY, anchorX="center", anchorY="center", alpha=200)

    def drawPauseMenu(self):
        arcade.draw_lbwh_rectangle_filled(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT, (6, 6, 16, 170))
        drawPanel(SCREEN_WIDTH / 2, 330, 460, 430)
        drawText("PAUSED", SCREEN_WIDTH / 2, 500, GOLD, 34, FONT_TITLE, anchorX="center", anchorY="center")
        for pauseButton in self.pauseButtons:
            pauseButton.draw()
        drawText("ESC / P resume    R restart    B street    Q title", SCREEN_WIDTH / 2, 138, CREAM, 13,
                 FONT_BODY, anchorX="center", anchorY="center", alpha=200)

    def drawDebugText(self):
        window = self.window
        lines = [
            f"DEBUG (F2)   FPS avg {window.averageFps:5.1f}",
            f"scene {self.sceneId}   state {self.state}   t {self.stateTime:5.1f}s   clock {window.gameClock:6.1f}s",
            f"particles {self.particles.count()} + glow {self.glowParticles.count()}   lights {self.lights.count()}",
            f"score {self.score}/{self.maxScore}   combo {self.combo}   aura {self.auraMode} {self.auraTimer:3.1f}s",
            f"sounds playing {len(window.sound.channels)}   mouse ({int(self.mouseX)}, {int(self.mouseY)})",
        ]
        arcade.draw_lbwh_rectangle_filled(8, SCREEN_HEIGHT - 204, 560, 132, (0, 0, 0, 170))
        for i in range(len(self.debugTexts)):
            if self.debugTexts[i].text != lines[i]:
                self.debugTexts[i].text = lines[i]
            self.debugTexts[i].draw()

    def drawBotCursor(self):
        if not (self.window.autoplay and self.botCursorVisible):
            return
        for ripple in self.botRipples:
            t = ripple[2] / 0.4
            arcade.draw_circle_outline(ripple[0], ripple[1], 8 + 26 * t, withAlpha(WHITE, 220 * (1 - t)), 3)
        if self.botMouseHeld:
            texture = assets.loadTexture("ui/cursor_grab.png")
        else:
            texture = assets.loadTexture("ui/cursor_hand.png")
        # The hotspot (fingertip) is near the top-left of the 48x48 picture
        drawTexture(texture, self.botX + 12, self.botY - 20, 48, 48)
