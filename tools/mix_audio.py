"""
mix_audio.py
------------
Rebuilds the game's soundtrack offline for the demo video.

While the demo is recorded, the SoundManager writes every audio event
(play, stop, volume, speed = pitch, pan) with its time into a JSON log.
This script reads that log, loads the same WAV files, and mixes them
into one stereo WAV that lines up exactly with the recorded video.

How the mixing works:
  * time is processed in small blocks of 1/60 second
  * every playing sound is a "voice" with a read position in its WAV data
  * speed changes the step size of the read position (pitch shift by
    resampling, with linear interpolation between samples)
  * volume changes are ramped across a block (no clicks / zipper noise)
  * mono sounds are panned left / right, stereo sounds are left as they are
  * a soft limiter keeps the final mix below -1 dBFS

Usage:
    python tools/mix_audio.py LOG.json OUT.wav DURATION_SECONDS
"""
import json
import os
import sys
import wave

import numpy as np

OUT_RATE = 48000
BLOCK = OUT_RATE // 60             # samples per block (1/60 s)

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
SOUND_DIR = os.path.join(os.path.dirname(TOOLS_DIR), "assets", "sounds")


def loadWav(path):
    """Read a 16-bit WAV file into a float array of shape (samples, channels)."""
    with wave.open(path, "rb") as wavFile:
        channels = wavFile.getnchannels()
        rate = wavFile.getframerate()
        frames = wavFile.readframes(wavFile.getnframes())
    data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    data = data.reshape(-1, channels)
    return data, rate


def loadAllSounds():
    sounds = {}
    for folder, subfolders, files in os.walk(SOUND_DIR):
        for fileName in files:
            if fileName.lower().endswith(".wav"):
                name = os.path.splitext(fileName)[0]
                sounds[name] = loadWav(os.path.join(folder, fileName))
    return sounds


class Voice:
    """One playing sound in the offline mix."""

    def __init__(self, data, rate, volume, pan, speed, loop):
        self.data = data
        self.rate = rate
        self.volume = volume
        self.targetVolume = volume
        self.pan = pan
        self.speed = speed
        self.loop = loop
        self.position = 0.0
        self.finished = False

    def render(self, count):
        """Return (count, 2) stereo samples and move the read position forward."""
        length = self.data.shape[0]
        step = self.speed * self.rate / OUT_RATE
        positions = self.position + step * np.arange(count)
        self.position = self.position + step * count
        if self.loop:
            positions = np.mod(positions, length)
            self.position = self.position % length
        index = np.floor(positions).astype(np.int64)
        fraction = (positions - index)[:, None]
        valid = index < length - 1
        index = np.clip(index, 0, length - 2)
        # linear interpolation between two neighbour samples
        samples = self.data[index] * (1.0 - fraction) + self.data[index + 1] * fraction
        samples[~valid] = 0.0
        if not self.loop and self.position >= length - 1:
            self.finished = True

        # volume ramp from the old value to the new one across the block
        ramp = np.linspace(self.volume, self.targetVolume, count, dtype=np.float32)[:, None]
        self.volume = self.targetVolume
        samples = samples * ramp

        if samples.shape[1] == 1:
            # simple balance pan for mono sounds: centre = full volume on both sides
            left = min(1.0, 1.0 - self.pan)
            right = min(1.0, 1.0 + self.pan)
            return np.hstack([samples * left, samples * right])
        return samples[:, :2]


def mix(logPath, outPath, duration):
    with open(logPath) as logFile:
        events = json.load(logFile)
    events.sort(key=eventTime)
    sounds = loadAllSounds()
    totalSamples = int(round(duration * OUT_RATE))
    output = np.zeros((totalSamples, 2), dtype=np.float32)
    voices = {}
    eventIndex = 0
    missing = set()

    start = 0
    while start < totalSamples:
        count = min(BLOCK, totalSamples - start)
        blockEndTime = (start + count) / OUT_RATE
        # apply every event that happens before the end of this block
        while eventIndex < len(events) and events[eventIndex]["t"] < blockEndTime:
            event = events[eventIndex]
            eventIndex += 1
            handle = event.get("handle")
            kind = event["type"]
            if kind == "play":
                name = event["name"]
                if name not in sounds:
                    missing.add(name)
                    continue
                data, rate = sounds[name]
                voices[handle] = Voice(data, rate, event["volume"], event.get("pan", 0.0),
                                       event.get("speed", 1.0), event.get("loop", False))
            elif handle in voices:
                voice = voices[handle]
                if kind == "stop":
                    voice.targetVolume = 0.0          # fade out inside this block, then remove
                    voice.finished = True
                elif kind == "volume":
                    voice.targetVolume = event["volume"]
                elif kind == "speed":
                    voice.speed = event["speed"]
                elif kind == "pan":
                    voice.pan = event["pan"]
        for handle in list(voices.keys()):
            voice = voices[handle]
            output[start:start + count] += voice.render(count)
            if voice.finished:
                del voices[handle]
        start += count

    # loudness: make-up gain so the whole mix sits near -16.5 dBFS RMS (a normal level for video)
    peak = float(np.max(np.abs(output))) if totalSamples > 0 else 0.0
    rms = float(np.sqrt(np.mean(output ** 2))) if totalSamples > 0 else 0.0
    if rms > 0:
        output = output * min(4.0, (10 ** (-16.5 / 20)) / rms)
    output = softKneeLimit(output)
    pcm = (np.clip(output, -1.0, 1.0) * 32767).astype(np.int16)
    folder = os.path.dirname(os.path.abspath(outPath))
    os.makedirs(folder, exist_ok=True)
    with wave.open(outPath, "wb") as wavFile:
        wavFile.setnchannels(2)
        wavFile.setsampwidth(2)
        wavFile.setframerate(OUT_RATE)
        wavFile.writeframes(pcm.tobytes())
    if missing:
        print("[mix] missing sounds:", ", ".join(sorted(missing)))
    print(f"[mix] {len(events)} events -> {outPath} ({duration:.1f} s, raw peak {peak:.2f}, raw rms {rms:.3f})")
    return outPath


def softKneeLimit(samples, knee=0.6, ceiling=10 ** (-1.0 / 20)):
    """
    Soft-knee limiter: quiet samples pass unchanged, samples above the knee
    are bent smoothly (tanh curve) so they never go past the ceiling (-1 dBFS).
    """
    size = np.abs(samples)
    room = ceiling - knee
    bent = knee + room * np.tanh((size - knee) / room)
    size = np.where(size > knee, bent, size)
    return np.sign(samples) * size


def eventTime(event):
    return event["t"]


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("usage: python tools/mix_audio.py LOG.json OUT.wav DURATION_SECONDS")
        sys.exit(1)
    mix(sys.argv[1], sys.argv[2], float(sys.argv[3]))
