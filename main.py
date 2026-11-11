"""
main.py
-------
Starts Midnight Makan.

GameWindow(arcade.Window) is the one window of the game. It owns the things
that must survive when the scene changes:
  * gameState  - stars and scores of the whole night (game_state.py)
  * sound      - the SoundManager (all audio goes through it)
  * gameClock  - seconds of game time (advanced by BaseScene.tick)
and it handles the command-line options used for testing and for the demo
video (autoplay bot, muted runs, automatic screenshots, recording).

Usage:
    python main.py [--scene ID] [--autoplay] [--mute] [--seconds N]
                   [--shots DIR] [--shot-every S] [--turbo N]
                   [--record VIDEO.mp4] [--audio-log LOG.json] [--seed N]
                   [--master-volume V] [--debug] [--fake-camera]
"""
import argparse
import os
import random
import time

import arcade
from PIL import Image

from core import assets
from core.recorder import FrameRecorder
from core.sound_manager import SoundManager
from game_state import GameState
from scenes import SCENE_IDS, makeScene
from settings import (NIGHT_DEEP, RECORD_FPS, SCENE_ORDER, SCREEN_HEIGHT, SCREEN_TITLE,
                      SCREEN_WIDTH, SCREENSHOT_DIR)


class GameWindow(arcade.Window):
    def __init__(self, options):
        super().__init__(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE, resizable=False, center_window=True)
        self.background_color = NIGHT_DEEP

        # Options that the scenes read
        self.options = options
        self.autoplay = options.autoplay
        self.turbo = max(1, int(options.turbo))
        self.recording = options.record is not None
        self.showDebug = options.debug
        self.gameClock = 0.0

        # Shared game objects
        self.gameState = GameState()
        logEvents = options.audioLog is not None or self.recording
        self.sound = SoundManager(muted=options.mute, logEvents=logEvents, masterVolume=options.masterVolume)
        self.sound.loadAll()
        assets.loadFonts()

        # Demo video recorder (every drawn frame is one video frame)
        self.recorder = None
        if self.recording:
            self.recorder = FrameRecorder(options.record, RECORD_FPS, (SCREEN_WIDTH, SCREEN_HEIGHT))

        # Automatic screenshots every few seconds of game time
        self.shotsFolder = options.shots
        self.shotEvery = max(0.1, options.shotEvery)
        self.nextShotTime = self.shotEvery
        if self.shotsFolder is not None:
            os.makedirs(self.shotsFolder, exist_ok=True)

        # Frame rate measurement (shown in the F2 debug text)
        self.averageFps = 60.0
        self.frameTimeAverage = 1.0 / 60.0
        self.lastFrameTime = None
        self.frameCount = 0
        self.startRealTime = time.perf_counter()

        # Closing happens in three small steps so the last frame can still be
        # captured: finishAndClose() -> afterDraw() saves everything -> on_update() closes
        self.closeRequested = False
        self.shutdownDone = False

    # ------------------------------------------------------------------
    # Starting
    # ------------------------------------------------------------------
    def startGame(self):
        sceneId = self.options.scene
        if sceneId in SCENE_ORDER:
            # Starting straight at a dining scene = free play (it returns to the title)
            self.gameState.storyMode = False
        if sceneId == "finale" and len(self.gameState.results) == 0:
            # Demo results (2 stars each) so the critic's verdict has something to judge
            for diningScene in SCENE_ORDER:
                self.gameState.recordResult(diningScene, 600, 1000, 6)
        self.show_view(makeScene(sceneId))

    def show_view(self, new_view):
        # One console line per scene change makes test logs easy to read
        sceneId = getattr(new_view, "sceneId", "?")
        print(f"[main] clock {self.gameClock:6.1f}s  scene -> {sceneId}")
        super().show_view(new_view)

    # ------------------------------------------------------------------
    # Called by BaseScene at the very end of on_draw
    # ------------------------------------------------------------------
    def afterDraw(self):
        self.measureFps()
        if self.recorder is not None and not self.shutdownDone:
            self.recorder.capture()

        if self.shotsFolder is not None and self.gameClock >= self.nextShotTime and not self.shutdownDone:
            self.saveShot()
            while self.nextShotTime <= self.gameClock:
                self.nextShotTime += self.shotEvery

        seconds = self.options.seconds
        if seconds is not None and seconds > 0 and self.gameClock >= seconds:
            self.finishAndClose()

        if self.closeRequested and not self.shutdownDone:
            if self.shotsFolder is not None:
                self.saveShot()     # one final shot at exit
            self.shutdown()

    def measureFps(self):
        now = time.perf_counter()
        if self.lastFrameTime is not None:
            frameTime = now - self.lastFrameTime
            # Exponential moving average so the number does not jump around
            self.frameTimeAverage = self.frameTimeAverage * 0.95 + frameTime * 0.05
            self.averageFps = 1.0 / max(0.0001, self.frameTimeAverage)
        self.lastFrameTime = now
        self.frameCount += 1

    # ------------------------------------------------------------------
    # Screenshots
    # ------------------------------------------------------------------
    def grabScreenImage(self):
        """
        The current frame as a 1280x720 RGB PIL image.
        With Windows display scaling (125%) the real framebuffer is bigger
        (1600x900), so it is resized with the LANCZOS filter.
        """
        image = arcade.get_image()
        if image.size != (SCREEN_WIDTH, SCREEN_HEIGHT):
            image = image.resize((SCREEN_WIDTH, SCREEN_HEIGHT), Image.LANCZOS)
        return image.convert("RGB")

    def saveShot(self):
        sceneId = getattr(self.current_view, "sceneId", "scene")
        fileName = f"{sceneId}_{self.gameClock:06.1f}.png"
        path = os.path.join(self.shotsFolder, fileName)
        try:
            self.grabScreenImage().save(path)
        except Exception as error:
            print(f"[main] WARNING: could not save screenshot {path}: {error}")

    def takeScreenshot(self, path=None):
        """F12: save screenshots/shot_<n>.png and return the path (None if it failed)."""
        if path is None:
            os.makedirs(SCREENSHOT_DIR, exist_ok=True)
            number = 1
            path = os.path.join(SCREENSHOT_DIR, f"shot_{number}.png")
            while os.path.exists(path):
                number += 1
                path = os.path.join(SCREENSHOT_DIR, f"shot_{number}.png")
        try:
            self.grabScreenImage().save(path)
        except Exception as error:
            print(f"[main] WARNING: screenshot failed: {error}")
            return None
        print(f"[main] Screenshot saved: {path}")
        return path

    # ------------------------------------------------------------------
    # Closing
    # ------------------------------------------------------------------
    def finishAndClose(self):
        """Close the recorder, save the audio log and close the window (safe to call any time)."""
        self.closeRequested = True

    def shutdown(self):
        if self.shutdownDone:
            return
        self.shutdownDone = True
        if self.recorder is not None:
            self.recorder.close()
        if self.options.audioLog is not None:
            self.sound.saveLog(self.options.audioLog)
        self.sound.stopEverything()
        realTime = time.perf_counter() - self.startRealTime
        print(f"[main] Finished: game clock {self.gameClock:.1f}s, {self.frameCount} frames in "
              f"{realTime:.1f}s real time ({self.frameCount / max(0.001, realTime):.1f} FPS drawn, "
              f"last average {self.averageFps:.1f} FPS)")

    def on_update(self, delta_time):
        # The scenes do their own updating (BaseScene.on_update); the window
        # only closes itself once everything has been saved.
        if self.shutdownDone and not self.closed:
            self.close()

    def on_close(self):
        # The window's X button: still save the video and the audio log
        self.shutdown()
        super().on_close()


# ----------------------------------------------------------------------
# Command-line options
# ----------------------------------------------------------------------
def parseOptions():
    parser = argparse.ArgumentParser(description="Midnight Makan - a night-market dining adventure")
    parser.add_argument("--scene", default="title", choices=SCENE_IDS,
                        help="scene to start at")
    parser.add_argument("--autoplay", action="store_true", help="the demo bot plays the game")
    parser.add_argument("--mute", action="store_true", help="no audio output (sounds are still logged)")
    parser.add_argument("--seconds", type=float, default=None, help="quit after N seconds of game time")
    parser.add_argument("--shots", default=None, help="folder for automatic screenshots")
    parser.add_argument("--shot-every", dest="shotEvery", type=float, default=2.0,
                        help="seconds of game time between automatic screenshots")
    parser.add_argument("--turbo", type=int, default=1, help="simulation steps per update (fast tests)")
    parser.add_argument("--record", default=None, help="record a deterministic video to this .mp4 file")
    parser.add_argument("--audio-log", dest="audioLog", default=None, help="save the sound event log (JSON)")
    parser.add_argument("--seed", type=int, default=None, help="random seed for reproducible runs")
    parser.add_argument("--master-volume", dest="masterVolume", type=float, default=1.0,
                        help="overall volume 0..1")
    parser.add_argument("--debug", action="store_true", help="start with the F2 debug overlay on")
    parser.add_argument("--fake-camera", dest="fakeCamera", action="store_true",
                        help="Scene 3 webcam mode uses a simulated camera (no real camera is opened)")
    return parser.parse_args()


def main():
    options = parseOptions()
    if options.seed is not None:
        random.seed(options.seed)
    window = GameWindow(options)
    window.startGame()
    arcade.run()


if __name__ == "__main__":
    main()
