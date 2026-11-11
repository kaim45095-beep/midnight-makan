"""
sound_manager.py
----------------
SoundManager: the ONE place where the game makes noise.

Why one class for all audio?
  * Scenes ask for a sound by name ("chop_1") instead of loading files.
  * Every playing sound gets a small integer "handle", so a scene can later
    change its volume, speed (pitch) or pan, fade it, or stop it.
  * Everything is written into an event log. The demo recorder saves this
    log and tools/mix_audio.py rebuilds the exact soundtrack offline.
  * Muted mode (--mute) creates no real players, but still hands out
    handles and logs events, so tests run silently with the same code.

Audio ideas used (Week 3-4 digital audio):
  * playback speed = pitch (a sample played faster sounds higher),
  * stereo panning from the x position on screen,
  * volume fades (linear interpolation of the gain over time),
  * two music layers started together (base + "aura" layer) and mixed by
    changing only the aura layer's volume, so they stay in sync.
"""
import json
import math
import os
import random
import wave

import arcade

import settings
from core.helpers import approach, clamp, lerp

MUSIC_VOLUME = 0.55        # base music volume (sound effects are louder)
AURA_RATE = 3.0            # how fast the aura layer follows setAuraLevel
MUSIC_MUTE_RATE = 8.0      # how fast F3 music mute fades


class SoundManager:
    def __init__(self, muted=False, logEvents=False, masterVolume=1.0):
        self.muted = muted
        self.logEvents = logEvents
        self.masterVolume = clamp(masterVolume, 0.0, 1.0)

        self.sounds = {}        # name -> arcade.Sound (not loaded in muted mode)
        self.paths = {}         # name -> file path
        self.durations = {}     # name -> length in seconds
        self.channels = {}      # handle -> dictionary describing one playing sound
        self.nextHandle = 1
        self.clock = 0.0        # SoundManager time in seconds (advanced by update)
        self.lastPlayTime = {}  # name -> clock time (used by cooldown)
        self.warnedNames = set()
        self.events = []

        # Music has two synced layers: the base track and the Aura Mode layer
        self.musicHandle = None
        self.musicName = None
        self.auraHandle = None
        self.auraName = None
        self.auraLevel = 0.0
        self.auraTarget = 0.0
        self.musicMuted = False
        self.musicMuteFactor = 1.0
        # Ambience loop (crowd murmur, kitchen hum ...)
        self.ambienceHandle = None
        self.ambienceName = None

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def loadAll(self):
        """Load every .wav under assets/sounds (the file name without .wav is the sound name)."""
        self.loadFolder(settings.SOUND_DIR)

    def loadFolder(self, folder):
        """Load every .wav in a folder and its sub-folders (os.walk)."""
        if not os.path.isdir(folder):
            print(f"[sound] WARNING: sound folder not found: {folder}")
            return 0
        count = 0
        for root, dirs, files in os.walk(folder):
            for fileName in sorted(files):
                if fileName.lower().endswith(".wav"):
                    name = os.path.splitext(fileName)[0]
                    if self.loadSound(name, os.path.join(root, fileName)):
                        count += 1
        return count

    def loadSound(self, name, path):
        # The length comes from the WAV header (frames / sample rate), so it
        # is known even in muted mode where pyglet is never used.
        try:
            with wave.open(path, "rb") as waveFile:
                self.durations[name] = waveFile.getnframes() / float(waveFile.getframerate())
        except Exception as error:
            print(f"[sound] WARNING: cannot read {path}: {error}")
            return False
        self.paths[name] = path
        if not self.muted:
            try:
                self.sounds[name] = arcade.load_sound(path)
            except Exception as error:
                print(f"[sound] WARNING: cannot load {path}: {error}")
                self.sounds[name] = None
        return True

    def hasSound(self, name):
        return name in self.paths

    def getDuration(self, name):
        return self.durations.get(name, 0.0)

    def warnMissing(self, name):
        if name not in self.warnedNames:
            self.warnedNames.add(name)
            print(f"[sound] WARNING: missing sound '{name}'")

    # ------------------------------------------------------------------
    # Event log
    # ------------------------------------------------------------------
    def logEvent(self, event):
        if self.logEvents:
            event["t"] = round(self.clock, 4)
            self.events.append(event)

    def saveLog(self, path):
        """Save the event list as JSON (read by tools/mix_audio.py)."""
        folder = os.path.dirname(os.path.abspath(path))
        if not os.path.isdir(folder):
            os.makedirs(folder)
        with open(path, "w") as logFile:
            json.dump(self.events, logFile, indent=0)
        print(f"[sound] Saved {len(self.events)} audio events to {path}")

    # ------------------------------------------------------------------
    # Channels (one dictionary per playing sound)
    # ------------------------------------------------------------------
    def startChannel(self, name, volume, pan, speed, loop, kind):
        volume = clamp(volume, 0.0, 1.0)
        pan = clamp(pan, -1.0, 1.0)
        speed = clamp(speed, 0.1, 4.0)
        handle = self.nextHandle
        self.nextHandle += 1

        channel = {
            "name": name, "kind": kind, "loop": loop,
            "volume": volume, "speed": speed, "pan": pan,
            "player": None, "fade": None, "paused": False,
            "endTime": None, "sentVolume": -1.0,
        }
        if not loop:
            channel["endTime"] = self.clock + self.getDuration(name) / speed
        self.channels[handle] = channel

        mixVolume = self.getMixVolume(channel)
        if not self.muted and self.sounds.get(name) is not None:
            try:
                channel["player"] = arcade.play_sound(self.sounds[name], mixVolume * self.masterVolume,
                                                      pan, loop, speed)
            except Exception as error:
                print(f"[sound] WARNING: could not play {name}: {error}")
        channel["sentVolume"] = mixVolume
        channel["loggedVolume"] = mixVolume
        channel["loggedSpeed"] = speed
        channel["loggedPan"] = pan
        self.logEvent({"type": "play", "handle": handle, "name": name, "volume": round(mixVolume, 4),
                       "pan": round(pan, 4), "speed": round(speed, 4), "loop": loop})
        return handle

    def getMixVolume(self, channel):
        """Volume of one channel in the game mix (music mute and aura level included)."""
        volume = channel["volume"]
        if channel["kind"] == "music":
            volume = volume * self.musicMuteFactor
        elif channel["kind"] == "aura":
            volume = volume * self.musicMuteFactor * self.auraLevel
        return clamp(volume, 0.0, 1.0)

    def applyVolume(self, handle, channel):
        """Send the channel volume to the player and log it when it really changed."""
        mixVolume = self.getMixVolume(channel)
        if abs(mixVolume - channel["sentVolume"]) > 0.0005:
            channel["sentVolume"] = mixVolume
            player = channel["player"]
            if player is not None:
                try:
                    player.volume = mixVolume * self.masterVolume
                except Exception:
                    pass
        if abs(mixVolume - channel["loggedVolume"]) > 0.01 or (mixVolume == 0.0 and channel["loggedVolume"] != 0.0):
            channel["loggedVolume"] = mixVolume
            self.logEvent({"type": "volume", "handle": handle, "volume": round(mixVolume, 4)})

    # ------------------------------------------------------------------
    # Playing sounds
    # ------------------------------------------------------------------
    def play(self, name, volume=1.0, pan=0.0, speed=1.0, loop=False, cooldown=0.0):
        """Play a sound. Returns a handle (int) or None (missing sound / cooldown)."""
        if name is None:
            return None
        if name not in self.paths:
            self.warnMissing(name)
            return None
        if cooldown > 0 and name in self.lastPlayTime:
            if self.clock - self.lastPlayTime[name] < cooldown:
                return None
        self.lastPlayTime[name] = self.clock
        if loop:
            kind = "loop"
        else:
            kind = "sfx"
        return self.startChannel(name, volume, pan, speed, loop, kind)

    def panFromX(self, x):
        """Stereo panning from the screen position: left edge -0.8 ... right edge +0.8."""
        return clamp((x / settings.SCREEN_WIDTH) * 2 - 1, -1.0, 1.0) * 0.8

    def playAt(self, name, x, volume=1.0, pitchVariation=0.0, speed=1.0, cooldown=0.0):
        """Play a sound panned to screen x, with a small random pitch change so repeats sound natural."""
        if pitchVariation > 0:
            speed = speed * random.uniform(1.0 - pitchVariation, 1.0 + pitchVariation)
        return self.play(name, volume, self.panFromX(x), speed, False, cooldown)

    def startLoop(self, name, volume=1.0, speed=1.0):
        """A looping sound effect (sizzle, pour ...). Stop it with stop(handle)."""
        return self.play(name, volume, 0.0, speed, True)

    def isPlaying(self, handle):
        return handle is not None and handle in self.channels

    def stop(self, handle):
        if handle is None or handle not in self.channels:
            return
        channel = self.channels.pop(handle)
        player = channel["player"]
        if player is not None:
            try:
                arcade.stop_sound(player)
            except Exception:
                pass
        self.logEvent({"type": "stop", "handle": handle})

    def setVolume(self, handle, volume):
        if handle is None or handle not in self.channels:
            return
        channel = self.channels[handle]
        channel["volume"] = clamp(volume, 0.0, 1.0)
        channel["fade"] = None
        self.applyVolume(handle, channel)

    def setSpeed(self, handle, speed):
        """Change playback speed = pitch (e.g. pour sound rising as the glass fills)."""
        if handle is None or handle not in self.channels:
            return
        channel = self.channels[handle]
        speed = clamp(speed, 0.1, 4.0)
        if abs(speed - channel["speed"]) < 0.0005:
            return
        channel["speed"] = speed
        if channel["player"] is not None:
            try:
                channel["player"].pitch = speed
            except Exception:
                pass
        if abs(speed - channel["loggedSpeed"]) > 0.01:
            channel["loggedSpeed"] = speed
            self.logEvent({"type": "speed", "handle": handle, "speed": round(speed, 4)})

    def setPan(self, handle, pan):
        """Move a playing sound between the left (-1) and right (+1) speaker."""
        if handle is None or handle not in self.channels:
            return
        channel = self.channels[handle]
        pan = clamp(pan, -1.0, 1.0)
        if abs(pan - channel["pan"]) < 0.0005:
            return
        channel["pan"] = pan
        if channel["player"] is not None:
            try:
                # Same formula Arcade uses: a point on a circle around the listener
                channel["player"].position = (pan, 0.0, math.sqrt(1 - pan * pan))
            except Exception:
                pass
        if abs(pan - channel["loggedPan"]) > 0.01:
            channel["loggedPan"] = pan
            self.logEvent({"type": "pan", "handle": handle, "pan": round(pan, 4)})

    def fadeTo(self, handle, volume, duration, stopAtEnd=False):
        """Fade a channel to a new volume over 'duration' seconds (linear interpolation)."""
        if handle is None or handle not in self.channels:
            return
        channel = self.channels[handle]
        channel["fade"] = {
            "from": channel["volume"], "to": clamp(volume, 0.0, 1.0),
            "time": 0.0, "duration": max(0.001, duration), "stopAtEnd": stopAtEnd,
        }

    # ------------------------------------------------------------------
    # Music (base layer + aura layer) and ambience
    # ------------------------------------------------------------------
    def playMusic(self, name, auraName=None, fadeTime=1.2):
        """Crossfade to a new music track. The same name keeps playing (no restart)."""
        if name is None:
            self.stopMusic(fadeTime)
            return None
        if name == self.musicName and self.musicHandle in self.channels:
            # Already playing: make sure it is (fading back) to full volume
            self.fadeTo(self.musicHandle, MUSIC_VOLUME, fadeTime)
            if self.auraHandle in self.channels:
                self.fadeTo(self.auraHandle, MUSIC_VOLUME, fadeTime)
            return self.musicHandle

        self.stopMusic(fadeTime)
        self.musicName = name
        self.auraName = auraName
        self.auraLevel = 0.0
        self.auraTarget = 0.0
        if name not in self.paths:
            self.warnMissing(name)
            self.musicHandle = None
            self.auraHandle = None
            return None
        self.musicHandle = self.startChannel(name, 0.0, 0.0, 1.0, True, "music")
        self.fadeTo(self.musicHandle, MUSIC_VOLUME, fadeTime)
        self.auraHandle = None
        if auraName is not None:
            if auraName in self.paths:
                # Started at the same moment so both loops stay in sync.
                # It is silent until setAuraLevel() raises it.
                self.auraHandle = self.startChannel(auraName, 0.0, 0.0, 1.0, True, "aura")
                self.fadeTo(self.auraHandle, MUSIC_VOLUME, fadeTime)
            else:
                self.warnMissing(auraName)
        return self.musicHandle

    def setAuraLevel(self, level):
        """0..1: how loud the aura layer should be (smoothed in update)."""
        self.auraTarget = clamp(level, 0.0, 1.0)

    def stopMusic(self, fadeTime=1.0):
        if self.musicHandle in self.channels:
            self.fadeTo(self.musicHandle, 0.0, fadeTime, stopAtEnd=True)
        if self.auraHandle in self.channels:
            self.fadeTo(self.auraHandle, 0.0, fadeTime, stopAtEnd=True)
        self.musicHandle = None
        self.auraHandle = None
        self.musicName = None
        self.auraName = None

    def playAmbience(self, name, volume=0.45, fadeTime=1.5):
        """Background ambience loop with crossfade. name=None fades the ambience out."""
        if name is not None and name == self.ambienceName and self.ambienceHandle in self.channels:
            self.fadeTo(self.ambienceHandle, volume, fadeTime)
            return self.ambienceHandle
        if self.ambienceHandle in self.channels:
            self.fadeTo(self.ambienceHandle, 0.0, fadeTime, stopAtEnd=True)
        self.ambienceHandle = None
        self.ambienceName = name
        if name is None:
            return None
        if name not in self.paths:
            self.warnMissing(name)
            return None
        self.ambienceHandle = self.startChannel(name, 0.0, 0.0, 1.0, True, "ambience")
        self.fadeTo(self.ambienceHandle, volume, fadeTime)
        return self.ambienceHandle

    def stopAllLoops(self):
        """Stop every looping sound effect (music and ambience keep going)."""
        handles = list(self.channels.keys())
        for handle in handles:
            if self.channels[handle]["kind"] == "loop":
                self.stop(handle)

    def pauseLoops(self, paused):
        """Pause / resume looping sound effects (used by the pause menu)."""
        for handle in self.channels:
            channel = self.channels[handle]
            if channel["kind"] != "loop" or channel["paused"] == paused:
                continue
            channel["paused"] = paused
            if channel["player"] is not None:
                try:
                    if paused:
                        channel["player"].pause()
                    else:
                        channel["player"].play()
                except Exception:
                    pass

    def toggleMusicMute(self):
        self.musicMuted = not self.musicMuted
        return self.musicMuted

    # ------------------------------------------------------------------
    # Update (called every tick by BaseScene)
    # ------------------------------------------------------------------
    def update(self, dt):
        self.clock += dt
        self.auraLevel = approach(self.auraLevel, self.auraTarget, AURA_RATE, dt)
        if self.musicMuted:
            muteTarget = 0.0
        else:
            muteTarget = 1.0
        self.musicMuteFactor = approach(self.musicMuteFactor, muteTarget, MUSIC_MUTE_RATE, dt)
        if abs(self.musicMuteFactor - muteTarget) < 0.001:
            self.musicMuteFactor = muteTarget

        handles = list(self.channels.keys())
        for handle in handles:
            channel = self.channels.get(handle)
            if channel is None:
                continue
            fade = channel["fade"]
            if fade is not None:
                fade["time"] += dt
                t = min(1.0, fade["time"] / fade["duration"])
                channel["volume"] = lerp(fade["from"], fade["to"], t)
                if t >= 1.0:
                    channel["fade"] = None
                    if fade["stopAtEnd"]:
                        self.stop(handle)
                        continue
            if fade is not None or channel["kind"] in ("music", "aura"):
                self.applyVolume(handle, channel)
            # One-shot sounds are forgotten when they finish (pyglet deletes the player itself)
            if channel["endTime"] is not None and self.clock >= channel["endTime"]:
                del self.channels[handle]

    def stopEverything(self):
        """Used when the window closes."""
        handles = list(self.channels.keys())
        for handle in handles:
            self.stop(handle)
