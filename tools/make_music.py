"""
make_music.py
-------------
Generates every background music loop (assets/sounds/music) and every
ambience loop (assets/sounds/ambience) for Midnight Makan. Nothing was
downloaded or recorded: every note is synthesised with NumPy maths and saved
with Python's "wave" module (the module used in the Week 2 "Sound Recording"
tutorial).

Music (STEREO, 32000 Hz, 16-bit PCM, seamless loops):
  music_title                  84 BPM  3/4  16 bars  dreamy pad, music-box arps, gamelan bells
  music_street                 96 BPM  4/4  12 bars  lo-fi stroll: FM electric piano, bells, swung drums
  music_rojak (+ _aura)       118 BPM  4/4  16 bars  bouncy marimba + plucks   | aura: bright lead + kompang
  music_wok (+ _aura)         128 BPM  4/4  16 bars  funk drums, bass, brass   | aura: fast arps + claps
  music_tehtarik (+ _aura)     92 BPM  4/4  12 bars  bossa / lo-fi e-piano     | aura: flute lead
  music_banquet (+ _aura)     100 BPM  3/4  16 bars  waltz: strings, pizz, erhu| aura: bell countermelody
  music_finale                 88 BPM  4/4  12 bars  piano, big pads, bells, builds to a climax
The three slowest 4/4 songs use 12 bars (three 4-bar phrases) and two songs
use 3/4 time so that every file stays under about 4.5 MB.

Ambience (STEREO, 32000 Hz, 24 s seamless loops, about -28 dBFS RMS):
  amb_night_market, amb_kitchen, amb_mamak, amb_hall, amb_rooftop

How one music track is made:
  1. Composition - chord progressions are written as chord symbols
                   ("Dmaj9", "Gm7 C7") and voice-led automatically: every chord
                   picks the voicing that moves the voices the least (also
                   across the loop point). Melodies are written by hand as
                   "note:beats" text, mostly in pentatonic scales for a
                   Malaysian / Southeast-Asian flavour.
  2. Performance - notes are "humanised" with a small seeded random timing
                   offset (a few ms) and velocity change; swing delays the
                   off-beat 16ths for a relaxed groove. Each section (every
                   4 bars) changes the arrangement: sparse -> full -> variation.
  3. Synthesis   - FM electric piano, wavetable pads / strings / brass (band-
                   limited additive waves, so no aliasing), Karplus-Strong
                   plucked strings, modal marimba / piano / music box,
                   inharmonic FM gamelan bells with "ombak" beating, phrase
                   leads with glides and vibrato (flute, erhu, synth lead),
                   drums (kick = pitched sine sweep, snare = tone + filtered
                   noise, shaker = high-passed noise) and kompang frame drums.
  4. Mixing      - instruments go to buses (pan, EQ, width, sidechain-style
                   ducking, auto-pan). Buses feed a ping-pong delay and a
                   Schroeder reverb (8 damped comb filters + 4 all-pass
                   filters). The master gets gentle compression and a
                   look-ahead peak limiter (peaks below -2 dBFS).
  5. Looping     - each track is rendered with a 6 s tail. Everything that
                   rings past the loop end (releases, echoes, reverb) is
                   added back onto the start, so the end flows into the start
                   with no gap and no click. The delay and reverb are linear
                   time-invariant systems, so their impulse responses are
                   rendered once with the difference equations and applied by
                   FFT convolution; the convolution tail is wrapped onto the
                   start in the same way.

Run from the project folder (plain Python with NumPy is enough):
    python tools/make_music.py                    (everything, about 1-2 minutes)
    python tools/make_music.py music_wok amb_hall (only the named files)
    python tools/make_music.py music_wok --buses   (also print the level of each bus)
"""
import itertools
import math
import os
import sys
import time
import wave

import numpy as np

SAMPLE_RATE = 32000          # 32 kHz keeps loops small; still covers up to 16 kHz
MUSIC_RMS_DB = -20.0         # average loudness of the base music loops
AURA_RMS_DB = -22.0          # aura layers sit a little under the base
MUSIC_CEILING_DB = -2.4      # no music sample may go above this
AMBIENCE_RMS_DB = -28.0      # ambience is quieter than the music
AMBIENCE_CEILING_DB = -9.0
AMBIENCE_SECONDS = 24.0
TAIL_SECONDS = 6.0           # extra render time for releases and echoes

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(TOOLS_DIR)
MUSIC_DIR = os.path.join(PROJECT_DIR, "assets", "sounds", "music")
AMBIENCE_DIR = os.path.join(PROJECT_DIR, "assets", "sounds", "ambience")

SHOW_BUSES = False           # set by the --buses option
LOOP_INFO = {}               # name -> (bars, beatsPerBar, bpm), used by the report


# ======================================================================
# 1. BASIC HELPERS
# ======================================================================
def samples(seconds):
    return int(round(seconds * SAMPLE_RATE))


def timeAxis(count):
    return np.arange(count) / SAMPLE_RATE


def dbToGain(db):
    return 10.0 ** (db / 20.0)


def gainToDb(gain):
    return 20.0 * math.log10(max(gain, 1e-12))


def rmsDb(signal):
    return gainToDb(float(np.sqrt(np.mean(np.square(signal)))))


def peakDb(signal):
    return gainToDb(float(np.max(np.abs(signal))))


def normalise(signal, peak=1.0):
    top = float(np.max(np.abs(signal)))
    if top < 1e-12:
        return signal
    return signal * (peak / top)


def midiToFreq(midi):
    """MIDI note number -> frequency in Hz (A4 = note 69 = 440 Hz).
    Works on single numbers and on NumPy arrays (for glides)."""
    return 440.0 * 2.0 ** ((midi - 69.0) / 12.0)


NOTE_OFFSETS = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def noteToMidi(name):
    """'C#5' -> 73, 'Bb3' -> 58 (C4 = 60 = middle C)."""
    letter = name[0].upper()
    rest = name[1:]
    shift = 0
    while rest and rest[0] in "#b":
        if rest[0] == "#":
            shift += 1
        else:
            shift -= 1
        rest = rest[1:]
    return 12 * (int(rest) + 1) + NOTE_OFFSETS[letter] + shift


def pitchClass(name):
    """'F#' -> 6 (pitch class 0..11, C = 0)."""
    return noteToMidi(name + "4") % 12


def nextPowerOfTwo(value):
    size = 1
    while size < value:
        size *= 2
    return size


def panGains(pan):
    """Constant-power pan law: left^2 + right^2 stays constant, so a sound
    keeps the same loudness wherever it is placed (1.0 each at the centre)."""
    angle = (pan + 1.0) * math.pi / 4.0
    return math.cos(angle) * math.sqrt(2.0), math.sin(angle) * math.sqrt(2.0)


def loopFrequency(freq, seconds):
    """Rounds a frequency so it completes a whole number of cycles in the
    loop - then a steady hum joins perfectly at the loop point."""
    return max(1.0, round(freq * seconds)) / seconds


def circularSmooth(values, width):
    """Moving average that wraps around the end of a loop (centred window)."""
    width = max(1, int(width))
    count = len(values)
    if width == 1:
        return values.copy()
    if width >= count:
        return np.full(count, float(np.mean(values)))
    padded = np.concatenate([values[count - width + 1:], values])
    sums = np.cumsum(np.concatenate([[0.0], padded]))
    causal = (sums[width:] - sums[:-width]) / width
    return np.roll(causal, -(width // 2))


def smoothSignal(values, width):
    """Plain (non-looping) moving average."""
    width = max(1, int(width))
    if width == 1:
        return values
    kernel = np.ones(width) / width
    return np.convolve(values, kernel, mode="same")


def loopSmoothCurve(rng, count, knots, low, high):
    """A slowly wandering random curve that joins up with itself at the
    loop point (random values at evenly spaced knots, wrapped, smoothed)."""
    values = rng.uniform(low, high, knots)
    positions = np.arange(knots) * count / knots
    curve = np.interp(np.arange(count), positions, values, period=count)
    return circularSmooth(curve, count // knots)


# ======================================================================
# 2. ENVELOPES (loudness over time)
# ======================================================================
def gateEnvelope(count, holdSeconds, releaseSeconds, attackSeconds=0.003):
    """1.0 while the key is held, then a smooth cosine fade to 0 (the release).
    The short attack ramp at the start avoids a click."""
    env = np.ones(count)
    attack = min(count, max(1, samples(attackSeconds)))
    env[:attack] = np.linspace(0.0, 1.0, attack, endpoint=False)
    hold = min(count, samples(holdSeconds))
    release = max(1, samples(releaseSeconds))
    end = min(count, hold + release)
    if end > hold:
        fade = 0.5 + 0.5 * np.cos(np.linspace(0.0, math.pi, release))
        env[hold:end] *= fade[:end - hold]
    env[end:] = 0.0
    return env


def adsrEnvelope(count, attack, decay, sustain, holdSeconds, release):
    """ADSR: Attack (smooth rise), Decay (to the sustain level), Sustain
    (while held), Release (fade after the key is let go)."""
    t = timeAxis(count)
    rise = np.clip(t / max(attack, 1e-4), 0.0, 1.0)
    rise = np.sin(rise * math.pi / 2.0) ** 2
    body = sustain + (1.0 - sustain) * np.exp(-np.maximum(t - attack, 0.0) / max(decay, 1e-4))
    return rise * body * gateEnvelope(count, holdSeconds, release, 0.001)


# ======================================================================
# 3. OSCILLATORS (wavetables built by additive synthesis)
# ======================================================================
TABLE_SIZE = 4096
tableCache = {}


def harmonicAmplitudes(shape, count):
    """Fourier series of the basic wave shapes (amplitude of harmonic k)."""
    k = np.arange(1, count + 1, dtype=float)
    if shape == "saw":
        return 1.0 / k
    if shape == "square":
        return np.where(k % 2 == 1, 1.0 / k, 0.0)
    if shape == "triangle":
        signs = np.where((k - 1) % 4 == 0, 1.0, -1.0)
        return np.where(k % 2 == 1, signs / (k * k), 0.0)
    if shape == "pulse":                       # 30% pulse: brighter, nasal
        return np.sin(math.pi * k * 0.3) / k
    if shape == "flute":                       # nearly a sine, soft upper partials
        amps = np.zeros(count)
        values = [1.0, 0.22, 0.09, 0.035, 0.015, 0.008]
        amps[:min(count, len(values))] = values[:min(count, len(values))]
        return amps
    return np.where(k == 1, 1.0, 0.0)          # "sine"


def waveTable(shape, harmonics):
    """One cycle of a band-limited wave: a sum of sines (additive synthesis)
    that stops below the Nyquist frequency, so the oscillator cannot alias."""
    key = (shape, harmonics)
    if key not in tableCache:
        x = np.arange(TABLE_SIZE) / TABLE_SIZE
        table = np.zeros(TABLE_SIZE + 1)
        amps = harmonicAmplitudes(shape, harmonics)
        for index in range(harmonics):
            if amps[index] != 0.0:
                table[:-1] += amps[index] * np.sin(2.0 * math.pi * (index + 1) * x)
        table[-1] = table[0]
        tableCache[key] = table / max(1e-9, float(np.max(np.abs(table))))
    return tableCache[key]


def tableOscillator(shape, freq, count, phase=0.0, maxHarmonics=48):
    """Wavetable oscillator. freq can be one number or an array (for glides
    and vibrato); the phase is the running sum of frequency / sample rate."""
    if np.isscalar(freq):
        topFreq = float(freq)
        cycles = phase + freq * np.arange(count) / SAMPLE_RATE
    else:
        topFreq = float(np.max(freq))
        cycles = phase + np.cumsum(freq[:count]) / SAMPLE_RATE
    harmonics = max(1, min(maxHarmonics, int(0.45 * SAMPLE_RATE / max(topFreq, 1.0))))
    table = waveTable(shape, harmonics)
    position = (cycles % 1.0) * TABLE_SIZE
    index = np.minimum(position.astype(np.int64), TABLE_SIZE - 1)
    frac = position - index
    return table[index] * (1.0 - frac) + table[index + 1] * frac


# ======================================================================
# 4. FILTERS (RBJ biquads, applied through their frequency response)
# ======================================================================
def biquadCoefficients(kind, freq, q=0.707, gainDb=0.0):
    """Biquad coefficients from the RBJ "Audio EQ Cookbook". The filter is
    the difference equation y[n] = b0 x[n] + b1 x[n-1] + b2 x[n-2]
    - a1 y[n-1] - a2 y[n-2]."""
    freq = min(freq, SAMPLE_RATE * 0.45)
    w0 = 2.0 * math.pi * freq / SAMPLE_RATE
    cosW = math.cos(w0)
    alpha = math.sin(w0) / (2.0 * q)
    amp = 10.0 ** (gainDb / 40.0)
    if kind == "lowpass":
        b = [(1 - cosW) / 2, 1 - cosW, (1 - cosW) / 2]
        a = [1 + alpha, -2 * cosW, 1 - alpha]
    elif kind == "highpass":
        b = [(1 + cosW) / 2, -(1 + cosW), (1 + cosW) / 2]
        a = [1 + alpha, -2 * cosW, 1 - alpha]
    elif kind == "bandpass":
        b = [alpha, 0.0, -alpha]
        a = [1 + alpha, -2 * cosW, 1 - alpha]
    elif kind == "peak":
        b = [1 + alpha * amp, -2 * cosW, 1 - alpha * amp]
        a = [1 + alpha / amp, -2 * cosW, 1 - alpha / amp]
    elif kind == "lowshelf":
        root = 2 * math.sqrt(amp) * alpha
        b = [amp * ((amp + 1) - (amp - 1) * cosW + root),
             2 * amp * ((amp - 1) - (amp + 1) * cosW),
             amp * ((amp + 1) - (amp - 1) * cosW - root)]
        a = [(amp + 1) + (amp - 1) * cosW + root,
             -2 * ((amp - 1) + (amp + 1) * cosW),
             (amp + 1) + (amp - 1) * cosW - root]
    else:                                      # "highshelf"
        root = 2 * math.sqrt(amp) * alpha
        b = [amp * ((amp + 1) + (amp - 1) * cosW + root),
             -2 * amp * ((amp - 1) + (amp + 1) * cosW),
             amp * ((amp + 1) + (amp - 1) * cosW - root)]
        a = [(amp + 1) - (amp - 1) * cosW + root,
             2 * ((amp - 1) - (amp + 1) * cosW),
             (amp + 1) - (amp - 1) * cosW - root]
    return b[0] / a[0], b[1] / a[0], b[2] / a[0], a[1] / a[0], a[2] / a[0]


delayCache = {}


def unitDelays(size):
    """z^-1 = e^(-jw) for every bin of a real FFT of the given size."""
    if size not in delayCache:
        freqs = np.fft.rfftfreq(size, 1.0 / SAMPLE_RATE)
        delayCache[size] = np.exp(-2j * math.pi * freqs / SAMPLE_RATE)
    return delayCache[size]


def filterResponse(filters, size):
    """Frequency response H(e^jw) = B(z) / A(z) of a chain of biquads.
    filters is a list like [("lowpass", 3000), ("peak", 800, 1.2, 6.0)]."""
    z1 = unitDelays(size)
    z2 = z1 * z1
    response = np.ones(len(z1), dtype=complex)
    for spec in filters:
        q = spec[2] if len(spec) > 2 else 0.707
        gainDb = spec[3] if len(spec) > 3 else 0.0
        b0, b1, b2, a1, a2 = biquadCoefficients(spec[0], spec[1], q, gainDb)
        response *= (b0 + b1 * z1 + b2 * z2) / (1.0 + a1 * z1 + a2 * z2)
    return response


def filterSound(signal, filters, padSeconds=0.3):
    """Filters a one-shot sound: multiply its spectrum by H(e^jw).
    Zero padding stops the filter ringing from wrapping around."""
    count = signal.shape[-1]
    size = nextPowerOfTwo(count + samples(padSeconds))
    spectrum = np.fft.rfft(signal, size, axis=-1) * filterResponse(filters, size)
    return np.fft.irfft(spectrum, size, axis=-1)[..., :count]


def filterLoop(signal, filters):
    """Circular filtering of a loop: identical to running the biquads for
    ever on the repeating loop, so the loop point stays seamless."""
    count = signal.shape[-1]
    spectrum = np.fft.rfft(signal, axis=-1) * filterResponse(filters, count)
    return np.fft.irfft(spectrum, count, axis=-1)


# ======================================================================
# 5. INSTRUMENTS (each returns a mono array, pads return stereo)
# ======================================================================
noteCache = {}


def electricPiano(freq, seconds, velocity=0.7):
    """FM electric piano (like the classic DX7 "Rhodes"). A sine modulator
    at the same frequency bends the phase of a sine carrier:
        y = sin(2 pi f t + I(t) sin(2 pi f t))
    The modulation index I(t) starts high (bright "bark") and decays (mellow
    tone). A fast 14:1 partial adds the metal "tine" ping. Harder notes are
    brighter, like a real piano."""
    key = ("ep", round(freq, 2), round(seconds, 2), round(velocity, 1))
    if key in noteCache:
        return noteCache[key]
    release = 0.3
    count = samples(seconds + release)
    t = timeAxis(count)
    index = (0.4 + 1.7 * velocity) * np.exp(-t / 0.28) + 0.25
    body = np.sin(2 * math.pi * freq * t + index * np.sin(2 * math.pi * freq * 1.0007 * t))
    tineRatio = 14.0
    if freq * tineRatio > 0.42 * SAMPLE_RATE:
        tineRatio = 7.0
    tine = np.zeros(count)
    if freq * tineRatio < 0.42 * SAMPLE_RATE:
        tine = np.sin(2 * math.pi * freq * tineRatio * t) * np.exp(-t / 0.012) * 0.2 * velocity
    decay = min(4.0, max(0.7, 2.8 * math.sqrt(220.0 / freq)))
    amp = np.exp(-t / decay) * gateEnvelope(count, seconds, release, 0.002)
    sound = (body + tine) * amp * (0.3 + 0.7 * velocity)
    noteCache[key] = sound
    return sound


def bassNote(freq, seconds, velocity=0.8, tone=0.35, sustain=0.72, decay=0.2):
    """Warm sine/triangle bass: sine fundamental plus a few soft harmonics
    (so small speakers can still hear it), a tiny pitch "thump" at the start
    and gentle tanh saturation."""
    release = 0.07
    count = samples(seconds + release)
    t = timeAxis(count)
    bend = freq * (1.0 + 0.025 * np.exp(-t / 0.02))
    phase = 2.0 * math.pi * np.cumsum(bend) / SAMPLE_RATE
    wave = (np.sin(phase) + tone * (0.55 * np.sin(2 * phase) - 0.2 * np.sin(3 * phase)
                                    + 0.08 * np.sin(4 * phase)))
    wave = np.tanh(1.3 * wave) / math.tanh(1.3)
    amp = (sustain + (1.0 - sustain) * np.exp(-t / decay)) * gateEnvelope(count, seconds, release, 0.004)
    return wave * amp * (0.4 + 0.6 * velocity)


def padNote(rng, freq, seconds, attack=0.8, release=1.6, brightness=0.5, detuneCents=7.0,
            vibratoCents=0.0, vibratoRate=5.2, shape="saw"):
    """Stereo pad / string ensemble: three slightly detuned band-limited saw
    waves per side (the detuning makes a slow chorus). Fewer harmonics =
    darker tone, like a low-pass filter. Optional delayed vibrato for strings."""
    count = samples(seconds + release)
    t = timeAxis(count)
    harmonics = int(3 + 22 * brightness)
    out = np.zeros((2, count))
    for channel in range(2):
        for voice in range(3):
            cents = (voice - 1) * detuneCents + (channel - 0.5) * detuneCents * 0.6
            voiceFreq = freq * 2.0 ** (cents / 1200.0)
            if vibratoCents > 0.0:
                onset = np.clip((t - 0.25) / 0.6, 0.0, 1.0)
                wobble = np.sin(2 * math.pi * (vibratoRate + 0.3 * voice) * t + rng.uniform(0, 6.28))
                voiceFreq = voiceFreq * 2.0 ** (vibratoCents * onset * wobble / 1200.0)
            out[channel] += tableOscillator(shape, voiceFreq, count, rng.random(), harmonics)
    env = adsrEnvelope(count, attack, 1.0, 0.85, seconds, release)
    return out * env / 3.0


def pluckNote(rng, freq, seconds, decaySeconds=1.5, brightness=0.5, pickPosition=0.2,
              releaseSeconds=0.06):
    """KARPLUS-STRONG plucked string: a short noise burst circulates in a
    delay line of D samples; every pass averages two neighbouring samples
    (a gentle low-pass), so high harmonics die first, like a real string:
        y[n] = g * (y[n-D] + y[n-D-1]) / 2
    The loop gain g is chosen so the note falls by 60 dB in decaySeconds.
    Subtracting a shifted copy of the burst imitates the plucking position.
    The loop is computed one block of D samples at a time (vectorised)."""
    delay = max(2, int(SAMPLE_RATE / freq - 0.5))
    realFreq = SAMPLE_RATE / (delay + 0.5)
    ring = samples(seconds + releaseSeconds)
    rate = freq / realFreq
    total = int(ring * rate) + delay + 8
    gain = 10.0 ** (-3.0 / (max(decaySeconds, 0.05) * realFreq))
    burst = rng.uniform(-1.0, 1.0, delay + 1)
    soft = int(1 + (1.0 - brightness) * 8)
    if soft > 1:
        burst = np.convolve(burst, np.ones(soft) / soft, mode="same")
    burst = burst - np.roll(burst, max(1, int(pickPosition * delay)))
    burst = normalise(burst - np.mean(burst))
    line = np.zeros(total)
    line[:delay + 1] = burst
    for start in range(delay + 1, total, delay):
        end = min(start + delay, total)
        line[start:end] = gain * 0.5 * (line[start - delay:end - delay] +
                                        line[start - delay - 1:end - delay - 1])
    # integer delays are slightly out of tune, so read the line at the exact pitch
    out = np.interp(np.arange(ring) * rate, np.arange(total), line)
    return out * gateEnvelope(ring, seconds, releaseSeconds, 0.0005)


def marimbaNote(rng, freq, velocity=0.8):
    """Modal synthesis of a marimba bar: a few decaying sines at the bar's
    tuned mode ratios (1 : 3.93 : 9.2) plus a soft mallet click."""
    decay = min(1.1, max(0.22, 0.65 * (262.0 / freq) ** 0.6))
    count = samples(decay * 5.0)
    t = timeAxis(count)
    out = np.zeros(count)
    modes = [(1.0, 1.0, decay), (3.93, 0.25 + 0.3 * velocity, decay * 0.2), (9.2, 0.1 * velocity, decay * 0.06)]
    for ratio, amp, modeDecay in modes:
        if freq * ratio < 0.45 * SAMPLE_RATE:
            out += amp * np.sin(2 * math.pi * freq * ratio * t) * np.exp(-t / modeDecay)
    mallet = smoothSignal(rng.uniform(-1, 1, count), 4) * np.exp(-t / 0.002) * 0.3 * velocity
    return (out + mallet) * gateEnvelope(count, decay * 4.5, decay * 0.5, 0.0008) * (0.4 + 0.6 * velocity)


def gamelanBell(rng, freq, seconds, velocity=0.8, ombak=4.5):
    """Gamelan-like metallophone (saron / gender). Inharmonic FM: the
    modulator runs at 1.41 x the carrier, a non-integer ratio, so the
    sidebands are NOT harmonics - that gives the metallic, bell-like colour.
    A second copy is tuned a few Hz higher: the two beat against each other,
    the shimmering "ombak" (wave) of Balinese paired tuning. Extra modes at
    2.76 and 5.40 x are the modes of a free metal bar."""
    ring = min(4.0, max(1.3, 3.0 * (440.0 / freq) ** 0.4))
    count = samples(ring)
    t = timeAxis(count)
    index = (0.5 + 1.4 * velocity) * np.exp(-t / 0.16)
    first = np.sin(2 * math.pi * freq * t + index * np.sin(2 * math.pi * 1.41 * freq * t))
    shifted = freq + ombak
    second = np.sin(2 * math.pi * shifted * t + rng.uniform(0, 6.28) +
                    index * np.sin(2 * math.pi * 1.41 * shifted * t))
    out = 0.5 * (first + second) * np.exp(-t / (ring * 0.3))
    if freq * 2.76 < 0.45 * SAMPLE_RATE:
        out += 0.3 * velocity * np.sin(2 * math.pi * 2.76 * freq * t) * np.exp(-t / (ring * 0.07))
    if freq * 5.40 < 0.45 * SAMPLE_RATE:
        out += 0.12 * velocity * np.sin(2 * math.pi * 5.40 * freq * t) * np.exp(-t / (ring * 0.025))
    out += smoothSignal(rng.uniform(-1, 1, count), 3) * np.exp(-t / 0.0015) * 0.15
    # players damp ("tutup") the previous key, so long notes ring, short ones are stopped
    hold = min(ring - 0.3, max(seconds + 0.25, 0.6))
    return out * gateEnvelope(count, hold, 0.3, 0.001) * (0.35 + 0.65 * velocity)


def musicBoxNote(freq, velocity=0.8):
    """Music box / celesta: a steel tine is a clamped beam, whose second
    mode sits at 6.27 x the fundamental (not a harmonic), plus a soft octave."""
    key = ("box", round(freq, 2), round(velocity, 1))
    if key in noteCache:
        return noteCache[key]
    ring = min(2.6, max(0.8, 1.9 * (660.0 / freq) ** 0.5))
    count = samples(ring)
    t = timeAxis(count)
    out = np.sin(2 * math.pi * freq * t) * np.exp(-t / (ring / 4.5))
    out += 0.12 * np.sin(2 * math.pi * 2.0 * freq * t) * np.exp(-t / (ring / 8.0))
    if freq * 6.27 < 0.45 * SAMPLE_RATE:
        out += 0.3 * velocity * np.sin(2 * math.pi * 6.27 * freq * t) * np.exp(-t / (ring / 25.0))
    out *= gateEnvelope(count, ring - 0.1, 0.1, 0.0008) * (0.35 + 0.65 * velocity)
    noteCache[key] = out
    return out


def pianoNote(freq, seconds, velocity=0.7):
    """Additive piano: stiff strings make the partials slightly sharp,
    f_n = n f sqrt(1 + B n^2) (inharmonicity). Each partial has a fast then
    slow decay (the "double decay" of real pianos); two strings per note are
    detuned a little for warmth; the hammer adds a short thump. The 7th
    partial is weak because the hammer strikes near 1/7 of the string."""
    key = ("piano", round(freq, 2), round(seconds, 2), round(velocity, 1))
    if key in noteCache:
        return noteCache[key]
    rng = np.random.default_rng(int(freq * 10))
    release = 0.3
    count = samples(seconds + release)
    t = timeAxis(count)
    out = np.zeros(count)
    tilt = 1.8 - 0.9 * velocity
    fast = min(0.9, 0.45 * (262.0 / freq) ** 0.5)
    slow = min(7.0, 3.4 * (262.0 / freq) ** 0.55)
    total = 0.0
    for n in range(1, 16):
        partial = n * freq * math.sqrt(1.0 + 0.00035 * n * n)
        if partial > 0.45 * SAMPLE_RATE:
            break
        amp = n ** -tilt
        if n % 7 == 0:
            amp *= 0.2
        env = 0.6 * np.exp(-t / (fast / (1.0 + 0.25 * (n - 1)))) + 0.4 * np.exp(-t / (slow / (1.0 + 0.18 * (n - 1))))
        pair = np.sin(2 * math.pi * partial * t) + np.sin(2 * math.pi * partial * 1.0006 * t + rng.uniform(0, 6.28))
        out += amp * env * pair * 0.5
        total += amp
    hammer = smoothSignal(rng.uniform(-1, 1, count), 6) * np.exp(-t / 0.004) * 0.25 * velocity
    out = (out / total + hammer) * gateEnvelope(count, seconds, release, 0.001) * (0.3 + 0.7 * velocity)
    noteCache[key] = out
    return out


def brassNote(freq, seconds, velocity=0.8):
    """Brass-like stab: a band-limited saw whose brightness jumps up at the
    start (the "blat" of a trumpet) and then settles, plus a small pitch
    scoop from 40 cents flat - that is what makes synth brass sound blown."""
    release = 0.09
    count = samples(seconds + release)
    t = timeAxis(count)
    curve = freq * 2.0 ** (-0.4 * np.exp(-t / 0.035) / 12.0)
    bright = tableOscillator("saw", curve, count, 0.0, 26)
    dark = tableOscillator("saw", curve, count, 0.0, 5)
    opening = np.clip(0.3 + 0.7 * velocity * np.exp(-t / 0.09), 0.0, 1.0)
    tone = dark + (bright - dark) * opening
    return tone * adsrEnvelope(count, 0.018, 0.12, 0.72, seconds, release) * (0.35 + 0.65 * velocity)


def fmPluck(freq, seconds, velocity=0.8):
    """Short bright FM pluck for fast arpeggios (modulator at 2 x carrier,
    index falls quickly -> "pling")."""
    key = ("fmpluck", round(freq, 2), round(seconds, 2), round(velocity, 1))
    if key in noteCache:
        return noteCache[key]
    count = samples(seconds + 0.08)
    t = timeAxis(count)
    index = (1.0 + 2.3 * velocity) * np.exp(-t / 0.05)
    tone = np.sin(2 * math.pi * freq * t + index * np.sin(2 * math.pi * 2.0 * freq * t))
    out = tone * np.exp(-t / 0.2) * gateEnvelope(count, seconds, 0.08, 0.001) * (0.4 + 0.6 * velocity)
    noteCache[key] = out
    return out


LEAD_STYLES = {
    "flute": {"shape": "flute", "harmonics": 6, "glide": 0.045, "vibRate": 4.8, "vibCents": 16.0,
              "vibDelay": 0.22, "attack": 0.045, "release": 0.16, "breath": 0.07, "scoop": 0.0},
    "erhu": {"shape": "saw", "harmonics": 40, "glide": 0.085, "vibRate": 6.2, "vibCents": 28.0,
             "vibDelay": 0.12, "attack": 0.07, "release": 0.18, "breath": 0.03, "scoop": 1.0},
    "lead": {"shape": "pulse", "harmonics": 22, "glide": 0.03, "vibRate": 5.6, "vibCents": 11.0,
             "vibDelay": 0.18, "attack": 0.01, "release": 0.1, "breath": 0.0, "scoop": 0.0},
}
ERHU_BODY = [("highpass", 280), ("peak", 750, 1.4, 6.0), ("peak", 1650, 2.0, 4.0),
             ("peak", 3100, 2.5, 3.0), ("highshelf", 5000, 0.7, -9.0)]


def renderLead(rng, notes, style):
    """Renders a whole melodic phrase with ONE continuous oscillator, so the
    pitch can glide between connected notes (portamento) and vibrato can
    grow on long notes, like a singer or an erhu player.
    notes = list of (startSeconds, seconds, midi, velocity), first at 0."""
    s = LEAD_STYLES[style]
    endTime = max(note[0] + note[1] for note in notes) + s["release"] + 0.05
    count = samples(endTime)
    t = timeAxis(count)
    pitch = np.full(count, float(notes[0][2]))
    amp = np.zeros(count)
    vibWeight = np.zeros(count)
    onsets = np.zeros(count)
    previous = None
    releaseCount = samples(s["release"])
    for start, length, midi, velocity in notes:
        a = samples(start)
        b = min(count, samples(start + length))
        if b <= a:
            continue
        local = t[a:b] - t[a]
        legato = previous is not None and start - previous[1] < 0.03
        if legato:
            pitch[a:b] = midi + (previous[0] - midi) * np.exp(-local / s["glide"])
            level = velocity * (1.0 - 0.15 * np.exp(-local / 0.04))
        else:
            pitch[a:b] = midi - s["scoop"] * np.exp(-local / 0.06)
            level = velocity * (1.0 - np.exp(-local / s["attack"]))
            onsets[a] = velocity
        if style == "erhu":
            level = level * (0.85 + 0.15 * np.minimum(1.0, local / 0.5))   # bow pressure swell
        amp[a:b] = level
        vibWeight[a:b] = np.clip((local - s["vibDelay"]) / 0.35, 0.0, 1.0)
        c = min(count, b + releaseCount)
        if c > b:
            fade = 0.5 + 0.5 * np.cos(np.linspace(0.0, math.pi, releaseCount))
            amp[b:c] = level[-1] * fade[:c - b]
            pitch[b:c] = pitch[b - 1]
            vibWeight[b:c] = vibWeight[b - 1]
        previous = (midi, start + length)
    vibrato = s["vibCents"] * vibWeight * np.sin(2 * math.pi * s["vibRate"] * t + rng.uniform(0, 6.28))
    freq = midiToFreq(pitch + vibrato / 100.0)
    tone = tableOscillator(s["shape"], freq, count, 0.0, s["harmonics"])
    if style == "erhu":
        tone = filterSound(tone, ERHU_BODY)
    if style == "flute":
        tone = filterSound(tone, [("lowpass", 5200)])
    out = tone * amp
    if s["breath"] > 0.0:
        air = rng.uniform(-1.0, 1.0, count)
        if style == "flute":
            air = filterSound(air, [("bandpass", 2400, 0.8)])
            chiff = np.convolve(onsets, np.exp(-np.arange(samples(0.05)) / samples(0.012)))[:count]
            air = air * (amp + 2.5 * chiff)
        else:
            air = filterSound(air, [("highpass", 2500), ("lowpass", 7000)]) * amp
        out += s["breath"] * 3.0 * air
    if style == "lead":
        # a second, slightly sharp copy on the other side = chorus / width
        other = tableOscillator(s["shape"], freq * 2.0 ** (9.0 / 1200.0), count, 0.37, s["harmonics"]) * amp
        return filterSound(np.vstack([out + 0.5 * other, other + 0.5 * out]), [("lowpass", 7000)])
    return out


# ---------------------------------------------------------------- drums
def noiseBurst(rng, count):
    return rng.uniform(-1.0, 1.0, count)


def kickDrum(rng, pitch=48.0, punch=1.0, length=0.45):
    """Kick = a sine whose pitch sweeps quickly down (about 150 Hz -> 48 Hz),
    plus a tiny filtered-noise click for the beater."""
    count = samples(length)
    t = timeAxis(count)
    freq = pitch + 100.0 * np.exp(-t / 0.028) + 20.0 * np.exp(-t / 0.2)
    body = np.sin(2 * math.pi * np.cumsum(freq) / SAMPLE_RATE) * np.exp(-t / 0.22)
    click = filterSound(noiseBurst(rng, count) * np.exp(-t / 0.003), [("lowpass", 3500)]) * 0.35 * punch
    out = np.tanh(1.6 * (body + click)) / math.tanh(1.6)
    return out * gateEnvelope(count, length - 0.06, 0.06, 0.0008)


def tomDrum(rng, pitch=140.0):
    count = samples(0.5)
    t = timeAxis(count)
    freq = pitch * (1.0 + 0.45 * np.exp(-t / 0.025))
    body = np.sin(2 * math.pi * np.cumsum(freq) / SAMPLE_RATE) * np.exp(-t / 0.18)
    skin = filterSound(noiseBurst(rng, count), [("bandpass", pitch * 4, 0.9)]) * np.exp(-t / 0.02) * 0.3
    return normalise(body + skin) * gateEnvelope(count, 0.44, 0.06, 0.0008)


def snareDrum(rng, brush=False):
    """Snare = two drum-head tones plus filtered noise (the snare wires).
    The brush version has a soft attack: a "swish" instead of a crack."""
    count = samples(0.35)
    t = timeAxis(count)
    tone = 0.55 * np.sin(2 * math.pi * 180 * t) * np.exp(-t / 0.05) + \
        0.3 * np.sin(2 * math.pi * 330 * t) * np.exp(-t / 0.03)
    noise = noiseBurst(rng, count)
    if brush:
        env = (1.0 - np.exp(-t / 0.012)) * np.exp(-t / 0.1)
        out = 0.15 * tone + filterSound(noise, [("bandpass", 4200, 0.6), ("lowpass", 7000)]) * env * 2.0
    else:
        wires = filterSound(noise, [("highpass", 1500), ("lowpass", 8000)]) * np.exp(-t / 0.085)
        out = tone + 0.9 * wires
    return normalise(out) * gateEnvelope(count, 0.3, 0.05, 0.0008)


def rimClick(rng):
    """Cross-stick rim click: a few short high modes (the wooden stick on
    the metal rim), used in the bossa nova clave pattern."""
    count = samples(0.15)
    t = timeAxis(count)
    out = np.zeros(count)
    for freq, decay, amp in ((1700, 0.011, 1.0), (520, 0.03, 0.5), (820, 0.018, 0.4), (2600, 0.006, 0.4)):
        out += amp * np.sin(2 * math.pi * freq * rng.uniform(0.98, 1.02) * t) * np.exp(-t / decay)
    out += noiseBurst(rng, count) * np.exp(-t / 0.0015) * 0.4
    return normalise(out) * gateEnvelope(count, 0.12, 0.03, 0.0005)


def shakerHit(rng, length=0.12):
    """Shaker = high-passed noise with a quick rise (the beads travel) and fall."""
    count = samples(length)
    t = timeAxis(count)
    env = (1.0 - np.exp(-t / 0.006)) * np.exp(-t / 0.035)
    out = filterSound(noiseBurst(rng, count), [("highpass", 5500), ("peak", 8000, 1.0, 4.0), ("lowpass", 12000)])
    return normalise(out * env) * gateEnvelope(count, length - 0.02, 0.02, 0.0005)


def hiHat(rng, openHat=False):
    """Hi-hat = inharmonic metallic partials + noise, high-passed."""
    length = 0.5 if openHat else 0.1
    count = samples(length)
    t = timeAxis(count)
    metal = np.zeros(count)
    for freq in (3150, 4280, 5550, 6720, 8100, 9350):
        metal += np.sin(2 * math.pi * freq * rng.uniform(0.99, 1.01) * t + rng.uniform(0, 6.28))
    decay = 0.25 if openHat else 0.032
    out = filterSound(0.4 * metal / 6.0 + noiseBurst(rng, count), [("highpass", 6500), ("lowpass", 13000)])
    return normalise(out * np.exp(-t / decay)) * gateEnvelope(count, length - 0.03, 0.03, 0.0005)


def handClap(rng):
    """Clap = several hands a few ms apart (4 noise bursts) + a short room tail."""
    count = samples(0.3)
    t = timeAxis(count)
    env = np.zeros(count)
    for delay in (0.0, 0.009, 0.019, 0.03):
        env += np.where(t >= delay, np.exp(-np.maximum(t - delay, 0) / 0.005), 0.0)
    env += np.where(t >= 0.03, 0.55 * np.exp(-np.maximum(t - 0.03, 0) / 0.11), 0.0)
    out = filterSound(noiseBurst(rng, count), [("bandpass", 1150, 1.1), ("highpass", 500)]) * env
    return normalise(out) * gateEnvelope(count, 0.27, 0.03, 0.0005)


MEMBRANE_RATIOS = [1.0, 1.59, 2.14, 2.30, 2.65, 2.92]      # modes of a circular drum skin


def kompangHit(rng, stroke="bung", pitch=190.0):
    """Kompang (Malay frame drum, goatskin on a wooden frame, hand-played).
    'bung' = open centre stroke: the skin rings at the circular-membrane mode
             ratios 1 : 1.59 : 2.14 : 2.30 ... and the pitch drops a little
             as the skin relaxes after the hit.
    'pak'  = slap near the rim: bright crack, short ring.
    'tak'  = muted slap: very short."""
    if stroke == "bung":
        count = samples(0.45)
        t = timeAxis(count)
        f0 = pitch * (1.0 + 0.07 * np.exp(-t / 0.03))
        phase = 2 * math.pi * np.cumsum(f0) / SAMPLE_RATE
        out = np.zeros(count)
        amps = [1.0, 0.45, 0.35, 0.25, 0.18, 0.12]
        decays = [0.16, 0.09, 0.07, 0.06, 0.045, 0.035]
        for index in range(6):
            out += amps[index] * np.sin(MEMBRANE_RATIOS[index] * phase) * np.exp(-t / decays[index])
        out += filterSound(noiseBurst(rng, count), [("bandpass", 1200, 0.8)]) * np.exp(-t / 0.006) * 0.4
    else:
        count = samples(0.25)
        t = timeAxis(count)
        decay = 0.035 if stroke == "pak" else 0.018
        out = np.zeros(count)
        for ratio in (2.14, 2.65, 3.5, 4.1):
            out += np.sin(2 * math.pi * pitch * ratio * rng.uniform(0.98, 1.02) * t) * np.exp(-t / decay) * 0.35
        crack = 0.012 if stroke == "pak" else 0.006
        out += filterSound(noiseBurst(rng, count), [("bandpass", 2400, 0.7)]) * np.exp(-t / crack) * 1.2
    return normalise(out) * gateEnvelope(count, count / SAMPLE_RATE - 0.04, 0.04, 0.0005)


def gongHit(rng, freq=98.0, length=3.5):
    """Chinese gong: inharmonic plate modes with long decays; the pitch
    bends slightly downwards after the strike (like an opera gong)."""
    count = samples(length)
    t = timeAxis(count)
    bend = freq * (1.0 + 0.03 * np.exp(-t / 0.4))
    phase = 2 * math.pi * np.cumsum(bend) / SAMPLE_RATE
    out = np.zeros(count)
    for ratio, decay, amp in ((1.0, 2.2, 1.0), (1.47, 1.5, 0.6), (2.09, 1.1, 0.5),
                              (2.56, 0.8, 0.35), (2.98, 0.6, 0.3), (3.93, 0.4, 0.2)):
        out += amp * np.sin(ratio * phase + rng.uniform(0, 6.28)) * np.exp(-t / decay)
    out += filterSound(noiseBurst(rng, count), [("bandpass", 900, 0.6)]) * np.exp(-t / 0.03) * 0.3
    return normalise(out) * adsrEnvelope(count, 0.006, 1.0, 1.0, length - 0.4, 0.4)


def cymbalSwell(rng, seconds):
    """Reverse-cymbal style swell: high noise that grows (cubic curve) and
    stops on the next downbeat - a classic transition into a new section."""
    count = samples(seconds)
    t = timeAxis(count)
    out = filterSound(noiseBurst(rng, count), [("highpass", 3500), ("lowpass", 11000)])
    return normalise(out) * (t / seconds) ** 3 * gateEnvelope(count, seconds - 0.02, 0.02, 0.001)


def crashCymbal(rng, length=2.5):
    count = samples(length)
    t = timeAxis(count)
    out = filterSound(noiseBurst(rng, count), [("highpass", 3800), ("peak", 6500, 1.0, 3.0), ("lowpass", 12000)])
    return normalise(out) * np.exp(-t / 0.7) * gateEnvelope(count, length - 0.2, 0.2, 0.002)


def fingerCymbal(rng, freq=2950.0):
    """Small bright cymbal ("ting") - a few inharmonic modes, long ring."""
    count = samples(2.2)
    t = timeAxis(count)
    out = np.zeros(count)
    for ratio, decay, amp in ((1.0, 0.9, 1.0), (1.51, 0.6, 0.5), (2.37, 0.35, 0.3)):
        if freq * ratio < 0.45 * SAMPLE_RATE:
            out += amp * np.sin(2 * math.pi * freq * ratio * t) * np.exp(-t / decay)
    return normalise(out) * gateEnvelope(count, 2.0, 0.2, 0.001)


def noiseRiser(rng, seconds):
    """Rising noise (brightness and loudness grow) used before a new section."""
    count = samples(seconds)
    t = timeAxis(count)
    low = filterSound(noiseBurst(rng, count), [("bandpass", 1500, 0.8)])
    high = filterSound(noiseBurst(rng, count), [("bandpass", 6000, 0.8)])
    mix = (t / seconds)
    out = low * (1.0 - mix) + high * mix
    return normalise(out) * (t / seconds) ** 2 * gateEnvelope(count, seconds - 0.03, 0.03, 0.001)


# ======================================================================
# 6. MUSIC THEORY: chords, automatic voice leading, melodies, patterns
# ======================================================================
CHORD_TYPES = {
    "": [0, 4, 7], "m": [0, 3, 7], "6": [0, 4, 7, 9], "m6": [0, 3, 7, 9],
    "7": [0, 4, 7, 10], "maj7": [0, 4, 7, 11], "m7": [0, 3, 7, 10],
    "9": [0, 4, 7, 10, 14], "maj9": [0, 4, 7, 11, 14], "m9": [0, 3, 7, 10, 14],
    "add9": [0, 4, 7, 14], "13": [0, 4, 10, 14, 21], "7b9": [0, 4, 7, 10, 13],
    "sus4": [0, 5, 7], "sus2": [0, 2, 7], "7sus": [0, 5, 7, 10], "m7b5": [0, 3, 6, 10],
}
# How much each chord tone matters when a voicing cannot hold every note
TONE_WEIGHTS = {0: 1.0, 2: 4.0, 3: 6.0, 4: 6.0, 5: 5.0, 6: 4.0, 7: 0.6, 9: 4.0,
                10: 5.0, 11: 5.0, 13: 3.0, 14: 2.5, 21: 3.0}


def parseChord(symbol):
    """'F#m7' -> (6, [0, 3, 7, 10], 6); 'Gm7/C' -> bass note C (0)."""
    bassName = None
    if "/" in symbol:
        symbol, bassName = symbol.split("/")
    rootLength = 2 if len(symbol) > 1 and symbol[1] in "#b" else 1
    root = pitchClass(symbol[:rootLength])
    intervals = CHORD_TYPES[symbol[rootLength:]]
    bass = root if bassName is None else pitchClass(bassName)
    return root, intervals, bass


def chordEvents(progression, beatsPerBar):
    """['Dmaj9', 'Gm7 C7', ...] -> [(startBeat, beats, symbol)]. A bar with
    two chord symbols gives each chord half of the bar."""
    events = []
    for bar in range(len(progression)):
        symbols = progression[bar].split()
        share = beatsPerBar / len(symbols)
        for index in range(len(symbols)):
            events.append((bar * beatsPerBar + index * share, share, symbols[index]))
    return events


def voiceChord(symbol, previous, low, high, voices, rootless):
    """VOICE LEADING: try every set of chord notes inside the range and keep
    the one with the lowest cost = total movement from the previous chord
    + penalties for missing important tones (3rd, 7th), doubled notes,
    muddy semitone clusters and very wide spreads."""
    root, intervals, bass = parseChord(symbol)
    weights = {}
    for interval in intervals:
        pc = (root + interval) % 12
        weight = TONE_WEIGHTS.get(interval, 2.0)
        if interval == 0 and rootless:
            weight = 0.2
        weights[pc] = max(weights.get(pc, 0.0), weight)
    candidates = []
    for midi in range(low, high + 1):
        if midi % 12 in weights:
            candidates.append(midi)
    best = None
    bestCost = 1e9
    centre = (low + high) / 2.0
    for combo in itertools.combinations(candidates, voices):
        gaps = [combo[i + 1] - combo[i] for i in range(voices - 1)]
        if min(gaps) < 2 or combo[-1] - combo[0] > 17:
            continue
        covered = set(note % 12 for note in combo)
        missing = sum(weights[pc] for pc in weights if pc not in covered)
        doubled = voices - len(covered)
        if previous is None:
            movement = sum(abs(note - centre) for note in combo) * 0.5
        else:
            movement = sum(abs(combo[i] - previous[i]) for i in range(voices))
        cost = movement + 3.0 * missing + 2.0 * doubled
        if cost < bestCost:
            bestCost = cost
            best = list(combo)
    return best


def voiceProgression(events, low, high, voices=4, rootless=False):
    """Voice-leads a whole progression. It runs twice: the second pass starts
    from the LAST chord of the first pass, so the loop point (last chord ->
    first chord) is voice-led smoothly too."""
    voicings = []
    previous = None
    for loopPass in range(2):
        voicings = []
        for event in events:
            voicing = voiceChord(event[2], previous, low, high, voices, rootless)
            voicings.append(voicing)
            previous = voicing
    return voicings


def bassMidi(symbol, low=36):
    """Bass note (root or slash note) placed inside [low, low + 11]."""
    bass = parseChord(symbol)[2]
    return low + (bass - low) % 12


def chordRoot(symbol, low=36):
    root = parseChord(symbol)[0]
    return low + (root - low) % 12


def parseMelody(bars, beatsPerBar, firstBar=0):
    """Hand-written melody: one string per bar, tokens 'note:beats', 'r' is a
    rest, '-' an empty bar. Every bar is checked so typing mistakes that
    break the bar length are caught straight away."""
    notes = []
    for index in range(len(bars)):
        barStart = (firstBar + index) * beatsPerBar
        position = 0.0
        text = bars[index]
        if text.strip() == "-":
            continue
        for token in text.split():
            name, length = token.split(":")
            length = float(length)
            if name != "r":
                notes.append((barStart + position, length, noteToMidi(name)))
            position += length
        if abs(position - beatsPerBar) > 1e-6:
            raise ValueError(f"melody bar {firstBar + index} has {position} beats: {text}")
    return notes


LEVELS = {"X": 1.0, "x": 0.7, "o": 0.42}


def patternHits(pattern, stepBeats=0.25):
    """Drum pattern text: X accent, x normal, o ghost note, . rest."""
    hits = []
    for index in range(len(pattern)):
        if pattern[index] in LEVELS:
            hits.append((index * stepBeats, LEVELS[pattern[index]]))
    return hits


# ======================================================================
# 7. EFFECTS: Schroeder reverb, ping-pong delay, dynamics
# ======================================================================
def combFilter(signal, delay, feedback, damping):
    """Damped feedback comb: y[n] = x[n] + g ((1-d) y[n-D] + d y[n-D-1]).
    The two-tap average in the loop makes high frequencies die faster
    (soft walls absorb treble). Computed one block of D samples at a time."""
    out = np.concatenate([np.zeros(delay + 1), signal])
    for start in range(delay + 1, len(out), delay):
        end = min(start + delay, len(out))
        out[start:end] += feedback * ((1.0 - damping) * out[start - delay:end - delay] +
                                      damping * out[start - delay - 1:end - delay - 1])
    return out[delay + 1:]


def allPassFilter(signal, delay, gain):
    """Schroeder all-pass y[n] = -g x[n] + x[n-D] + g y[n-D]: smears the
    echoes into a dense tail without changing the tone colour."""
    out = -gain * signal
    out[delay:] += signal[:-delay]
    for start in range(delay, len(out), delay):
        end = min(start + delay, len(out))
        out[start:end] += gain * out[start - delay:end - delay]
    return out


def schroederImpulse(roomSize, damping, preDelay=0.02):
    """Impulse response of a stereo Schroeder / Freeverb reverb (8 parallel
    damped combs -> 4 series all-passes; the right channel uses slightly
    longer delays for a wide, decorrelated image)."""
    feedback = 0.74 + 0.22 * roomSize
    scale = (SAMPLE_RATE / 44100.0) * (0.8 + 0.5 * roomSize)
    loopTime = 1400 * scale / SAMPLE_RATE
    decaySeconds = 3.0 * loopTime / -math.log10(feedback)
    count = samples(min(5.0, decaySeconds + 0.6))
    ir = np.zeros((2, count))
    for channel in range(2):
        spread = 23 * channel
        impulse = np.zeros(count)
        impulse[samples(preDelay)] = 1.0
        wet = np.zeros(count)
        for delay in (1116, 1188, 1277, 1356, 1422, 1491, 1557, 1617):
            wet += combFilter(impulse, int((delay + spread) * scale), feedback, damping)
        for delay in (556, 441, 341, 225):
            wet = allPassFilter(wet, int((delay + spread) * scale), 0.5)
        ir[channel] = wet
    ir *= gateEnvelope(count, count / SAMPLE_RATE - 0.4, 0.4, 0.0)
    return ir / math.sqrt(float(np.sum(ir ** 2)) / 2.0)


def pingPongImpulse(delaySeconds, feedback, repeats=8):
    """Tempo-synced ping-pong delay: echoes alternate left / right, each one
    quieter (feedback^k) and a little darker (wider smoothing kernel)."""
    delay = samples(delaySeconds)
    ir = np.zeros((2, delay * repeats + 64))
    for k in range(1, repeats + 1):
        width = 1 + 4 * k
        kernel = np.hanning(width + 2)[1:-1]
        kernel /= np.sum(kernel)
        channel = (k - 1) % 2
        ir[channel, k * delay:k * delay + width] += feedback ** (k - 1) * kernel
    return ir


def foldLoop(audio, loopLength):
    """THE SEAMLESS LOOP TRICK: everything rendered after the loop end
    (note releases, echoes, reverb tail) is added back onto the start, so
    the loop sounds as if it had been playing for ever."""
    out = audio[:, :loopLength].copy()
    rest = audio[:, loopLength:]
    while rest.shape[1] > 0:
        chunk = rest[:, :loopLength]
        out[:, :chunk.shape[1]] += chunk
        rest = rest[:, loopLength:]
    return out


def compressLoop(stereo, thresholdDb, ratio, detectSeconds=0.025, smoothSeconds=0.1):
    """Gentle bus compressor: the level above the threshold is reduced by
    the ratio (2:1 -> 4 dB over becomes 2 dB over). The level detector and
    the gain smoothing wrap around the loop, so the seam is not disturbed."""
    power = np.mean(stereo ** 2, axis=0)
    level = circularSmooth(power, samples(detectSeconds))
    levelDb = 10.0 * np.log10(level + 1e-12)
    reduction = np.maximum(0.0, levelDb - thresholdDb) * (1.0 - 1.0 / ratio)
    reduction = circularSmooth(reduction, samples(smoothSeconds))
    return stereo * 10.0 ** (-reduction / 20.0)


def limitLoop(stereo, ceilingDb, block=32, hold=9):
    """Look-ahead peak limiter: find the gain each 1 ms block needs to stay
    under the ceiling, hold it for +-9 blocks, smooth it (so the gain moves
    without clicks) and interpolate. Holding for longer than half the
    smoothing window guarantees the peak itself is always turned down enough."""
    ceiling = dbToGain(ceilingDb)
    count = stereo.shape[1]
    blocks = int(math.ceil(count / block))
    padded = np.zeros(blocks * block)
    padded[:count] = np.max(np.abs(stereo), axis=0)
    need = np.minimum(1.0, ceiling / np.maximum(padded.reshape(blocks, block).max(axis=1), 1e-9))
    held = need.copy()
    for shift in range(1, hold + 1):
        held = np.minimum(held, np.roll(need, shift))
        held = np.minimum(held, np.roll(need, -shift))
    smooth = circularSmooth(held, 2 * hold - 1)
    centres = (np.arange(blocks) + 0.5) * block
    gain = np.interp(np.arange(count), centres, smooth, period=blocks * block)
    return np.clip(stereo * gain, -ceiling, ceiling)


def activeRmsDb(stereo, block=1600):
    """Loudness of an instrument WHILE it plays: the RMS of the 50 ms
    blocks that are no more than 40 dB below its loudest block (so an
    instrument that only plays in half of the song is not under-rated)."""
    count = stereo.shape[1] // block * block
    power = np.mean(stereo[:, :count] ** 2, axis=0).reshape(-1, block).mean(axis=1)
    top = float(np.max(power))
    if top <= 1e-24:
        return -120.0
    return 10.0 * math.log10(float(np.mean(power[power > top * 1e-4])))


def masterLoop(stereo, targetDb, ceilingDb, compress=True):
    """Master chain: level -> gentle compression -> level -> limiter."""
    stereo = stereo * dbToGain(targetDb - rmsDb(stereo))
    if SHOW_BUSES:
        print(f"      master: peak {peakDb(stereo):5.1f} dBFS before compression/limiting")
    if compress:
        stereo = compressLoop(stereo, targetDb + 5.0, 2.0)
    for step in range(3):
        stereo = stereo * dbToGain(targetDb - rmsDb(stereo))
        stereo = limitLoop(stereo, ceilingDb)
    return stereo


def crossfadeLoop(stereo, loopCount, fadeCount):
    """Makes a seamless loop out of a longer texture: the audio after the
    loop end is faded OUT over the first second while the start fades IN
    (equal-power curves keep the loudness steady for noise-like sounds)."""
    out = stereo[:, :loopCount].copy()
    x = np.linspace(0.0, 1.0, fadeCount)
    out[:, :fadeCount] = stereo[:, :fadeCount] * np.sin(x * math.pi / 2) + \
        stereo[:, loopCount:loopCount + fadeCount] * np.cos(x * math.pi / 2)
    return out


# ======================================================================
# 8. THE ARRANGEMENT ENGINE (a tiny "DAW": track -> buses -> master)
# ======================================================================
class Bus:
    """One mixer channel: its own stereo audio plus effect settings."""

    def __init__(self, name, layer, length, level, reverb, delay, filters, width, duck, autoPan):
        self.name = name
        self.layer = layer
        self.audio = np.zeros((2, length))
        self.level = level
        self.reverb = reverb
        self.delay = delay
        self.filters = filters
        self.width = width
        self.duck = duck
        self.autoPan = autoPan


class Track:
    def __init__(self, name, bpm, beatsPerBar, bars, seed, swing=0.0, swingUnit=0.25,
                 reverbSize=0.6, reverbDamping=0.4, delayBeats=0.75, delayFeedback=0.35):
        self.name = name
        self.bpm = bpm
        self.secondsPerBeat = 60.0 / bpm
        self.beatsPerBar = beatsPerBar
        self.bars = bars
        self.totalBeats = beatsPerBar * bars
        self.loopSeconds = self.totalBeats * self.secondsPerBeat
        self.loopLength = samples(self.loopSeconds)
        self.bufferLength = self.loopLength + samples(TAIL_SECONDS)
        self.rng = np.random.default_rng(seed)
        self.swing = swing
        self.swingUnit = swingUnit
        self.timingSpread = 0.004
        self.velocitySpread = 0.07
        self.reverbSize = reverbSize
        self.reverbDamping = reverbDamping
        self.delayBeats = delayBeats
        self.delayFeedback = delayFeedback
        self.buses = {}
        self.busOrder = []
        self.kickTimes = []

    def addBus(self, name, layer="base", level=-20.0, reverb=0.15, delay=0.0, filters=None,
               width=1.0, duck=0.0, autoPan=None):
        """level = the fader: how loud (dB RMS) this instrument should be
        while it is playing. reverb / delay = send amounts."""
        if filters is None:
            filters = []
        self.buses[name] = Bus(name, layer, self.bufferLength, level, reverb, delay, filters,
                               width, duck, autoPan)
        self.busOrder.append(name)

    def beatTime(self, beat):
        """Beat -> seconds. SWING: notes on the off-beat 16th (or 8th) are
        pushed a little later, which gives a lazy, human groove."""
        if self.swing > 0.0:
            steps = beat / self.swingUnit
            nearest = round(steps)
            if abs(steps - nearest) < 1e-6 and nearest % 2 == 1:
                beat += self.swing * self.swingUnit
        return beat * self.secondsPerBeat

    def humanTime(self):
        return float(np.clip(self.rng.normal(0.0, self.timingSpread),
                             -2.5 * self.timingSpread, 2.5 * self.timingSpread))

    def humanGain(self):
        return float(np.clip(1.0 + self.rng.normal(0.0, self.velocitySpread), 0.8, 1.2))

    def placeAtSeconds(self, busName, seconds, sound, gain=1.0, pan=0.0):
        """Mixes a sound into a bus. Positions wrap around the loop, so a
        note that is humanised slightly before beat 0 lands at the loop end."""
        bus = self.buses[busName]
        start = samples(seconds) % self.loopLength
        if sound.ndim == 1:
            left, right = panGains(pan)
            parts = (sound * (gain * left), sound * (gain * right))
        else:
            parts = (sound[0] * (gain * min(1.0, 1.0 - pan)), sound[1] * (gain * min(1.0, 1.0 + pan)))
        length = len(parts[0])
        offset = 0
        position = start
        while offset < length:
            take = min(self.bufferLength - position, length - offset)
            bus.audio[0, position:position + take] += parts[0][offset:offset + take]
            bus.audio[1, position:position + take] += parts[1][offset:offset + take]
            offset += take
            position = (position + take) % self.loopLength

    def place(self, busName, beat, sound, gain=1.0, pan=0.0, humanise=True):
        seconds = self.beatTime(beat)
        if humanise:
            seconds += self.humanTime()
            gain *= self.humanGain()
        self.placeAtSeconds(busName, seconds, sound, gain, pan)

    def hit(self, busName, variants, beat, gain=1.0, pan=0.0):
        """Drum hit: picks one of several pre-made variants (round-robin
        samples) so repeated hits never sound machine-identical."""
        sound = variants[int(self.rng.integers(len(variants)))]
        if busName == "kick":
            self.kickTimes.append(self.beatTime(beat))
        self.place(busName, beat, sound, gain, pan)

    def playPattern(self, busName, variants, pattern, barStart, gain=1.0, pan=0.0, stepBeats=0.25):
        for offset, level in patternHits(pattern, stepBeats):
            self.hit(busName, variants, barStart + offset, gain * level, pan)

    def placeLead(self, busName, melody, style, gain=1.0, pan=0.0, transpose=0, articulation=1.0):
        """Splits a melody into phrases (at rests of a beat or more) and
        renders each phrase with renderLead so connected notes glide."""
        phrases = []
        current = []
        for note in melody:
            if current and note[0] - (current[-1][0] + current[-1][1]) >= 1.0:
                phrases.append(current)
                current = []
            current.append(note)
        if current:
            phrases.append(current)
        for phrase in phrases:
            times = [self.beatTime(note[0]) + self.humanTime() for note in phrase]
            noteList = []
            for index in range(len(phrase)):
                beat, beats, midi = phrase[index]
                length = beats * self.secondsPerBeat * articulation
                if articulation >= 1.0 and index + 1 < len(phrase):
                    if abs(phrase[index + 1][0] - (beat + beats)) < 1e-6:
                        length = times[index + 1] - times[index]
                accent = 0.95 if abs(beat % self.beatsPerBar) < 1e-6 else 0.82
                velocity = accent * float(np.clip(1.0 + self.rng.normal(0, 0.05), 0.85, 1.1))
                noteList.append((times[index] - times[0], length, midi + transpose, velocity))
            sound = renderLead(self.rng, noteList, style)
            self.placeAtSeconds(busName, times[0], sound, gain, pan)

    def duckCurve(self, depth):
        """Sidechain-style ducking: the bus dips briefly on every kick,
        which makes the groove "breathe" (a lo-fi / dance production trick)."""
        curve = np.zeros(self.loopLength)
        kernelLength = samples(0.35)
        t = timeAxis(kernelLength)
        kernel = (1.0 - np.exp(-t / 0.004)) * np.exp(-t / 0.11)
        for seconds in self.kickTimes:
            start = samples(seconds) % self.loopLength
            end = min(self.loopLength, start + kernelLength)
            curve[start:end] = np.maximum(curve[start:end], kernel[:end - start])
            if start + kernelLength > self.loopLength:
                over = start + kernelLength - self.loopLength
                curve[:over] = np.maximum(curve[:over], kernel[kernelLength - over:])
        return 1.0 - depth * curve

    def autoPanCurves(self, cyclesPerBeat, depth):
        """Stereo auto-pan (like a vintage e-piano's tremolo). The LFO is
        locked to the beat, so it completes whole cycles in the loop."""
        beats = np.arange(self.loopLength) / SAMPLE_RATE / self.secondsPerBeat
        lfo = np.sin(2 * math.pi * cyclesPerBeat * beats)
        return np.sqrt(1.0 - depth * lfo), np.sqrt(1.0 + depth * lfo)

    def renderLayer(self, layer):
        """Mixes every bus of one layer ("base" or "aura") into a stereo loop.
        Each bus is folded into a loop, EQ'd in the frequency domain, sent
        to the delay and reverb (FFT convolution with their impulse
        responses) and summed; the tails are folded onto the start."""
        reverbIr = schroederImpulse(self.reverbSize, self.reverbDamping)
        delayIr = pingPongImpulse(self.delayBeats * self.secondsPerBeat, self.delayFeedback)
        irLength = max(reverbIr.shape[1], delayIr.shape[1])
        size = nextPowerOfTwo(self.loopLength + 2 * irLength + 16)
        bins = size // 2 + 1
        mix = np.zeros((2, bins), dtype=complex)
        reverbSend = np.zeros((2, bins), dtype=complex)
        delaySend = np.zeros((2, bins), dtype=complex)
        for name in self.busOrder:
            bus = self.buses[name]
            if bus.layer != layer:
                continue
            audio = foldLoop(bus.audio, self.loopLength)
            if bus.duck > 0.0 and self.kickTimes:
                audio *= self.duckCurve(bus.duck)
            if bus.autoPan is not None:
                leftGain, rightGain = self.autoPanCurves(bus.autoPan[0], bus.autoPan[1])
                audio[0] *= leftGain
                audio[1] *= rightGain
            if bus.width != 1.0:
                mid = (audio[0] + audio[1]) * 0.5
                side = (audio[0] - audio[1]) * 0.5 * bus.width
                audio = np.vstack([mid + side, mid - side])
            playing = activeRmsDb(audio)
            if playing < -110.0:
                continue
            gain = dbToGain(bus.level - playing)
            if SHOW_BUSES:
                print(f"      bus {name:10s} level {bus.level:6.1f} dB  gain {gainToDb(gain):6.1f} dB  "
                      f"average {rmsDb(audio * gain):6.1f} dB")
            spectrum = np.fft.rfft(audio, size, axis=1) * gain
            if bus.filters:
                spectrum *= filterResponse(bus.filters, size)
            mix += spectrum
            reverbSend += spectrum * bus.reverb
            delaySend += spectrum * bus.delay
        delayWet = delaySend * np.fft.rfft(delayIr, size, axis=1) * filterResponse(
            [("highpass", 300), ("lowpass", 5000)], size)
        mix += delayWet
        reverbSend += delayWet * 0.4
        mix += reverbSend * np.fft.rfft(reverbIr, size, axis=1) * filterResponse(
            [("highpass", 200), ("lowpass", 6500)], size)
        mix *= filterResponse([("highpass", 30), ("lowpass", 14500)], size)
        return foldLoop(np.fft.irfft(mix, size, axis=1), self.loopLength)


def finishTrack(track, auraName=None, auraRmsDb=AURA_RMS_DB):
    """Renders and masters the base loop (and the aura loop if there is one)."""
    LOOP_INFO[track.name] = (track.bars, track.beatsPerBar, track.bpm)
    result = {track.name: masterLoop(track.renderLayer("base"), MUSIC_RMS_DB, MUSIC_CEILING_DB)}
    if auraName is not None:
        LOOP_INFO[auraName] = (track.bars, track.beatsPerBar, track.bpm)
        result[auraName] = masterLoop(track.renderLayer("aura"), auraRmsDb, MUSIC_CEILING_DB)
    return result


def makeKit(rng, names, count=4):
    """Pre-renders a few variants of each drum sound for one track."""
    kit = {}
    for name in names:
        variants = []
        for index in range(count):
            if name == "kick":
                variants.append(kickDrum(rng, 48.0, 0.8))
            elif name == "softkick":
                variants.append(kickDrum(rng, 52.0, 0.3, 0.35))
            elif name == "snare":
                variants.append(snareDrum(rng))
            elif name == "brush":
                variants.append(snareDrum(rng, brush=True))
            elif name == "rim":
                variants.append(rimClick(rng))
            elif name == "shaker":
                variants.append(shakerHit(rng))
            elif name == "hat":
                variants.append(hiHat(rng))
            elif name == "openhat":
                variants.append(hiHat(rng, openHat=True))
            elif name == "clap":
                variants.append(handClap(rng))
            elif name == "bung":
                variants.append(kompangHit(rng, "bung", 190.0 * rng.uniform(0.97, 1.03)))
            elif name == "pak":
                variants.append(kompangHit(rng, "pak", 190.0 * rng.uniform(0.97, 1.03)))
            elif name == "tak":
                variants.append(kompangHit(rng, "tak", 210.0 * rng.uniform(0.97, 1.03)))
        kit[name] = variants
    return kit


KOMPANG_STROKES = {"B": ("bung", 1.0), "b": ("bung", 0.65), "P": ("pak", 1.0), "p": ("pak", 0.6),
                   "T": ("tak", 0.9), "t": ("tak", 0.55)}


def playKompang(track, kit, busName, pattern, barStart, gain, pan, stepBeats=0.25):
    """Kompang rhythm text: B/b = bung (open), P/p = pak (slap), T/t = tak (muted)."""
    for index in range(len(pattern)):
        letter = pattern[index]
        if letter in KOMPANG_STROKES:
            stroke, level = KOMPANG_STROKES[letter]
            track.hit(busName, kit[stroke], barStart + index * stepBeats, gain * level, pan)


# ======================================================================
# 9. THE SONGS
# ======================================================================
def makeTitle():
    """84 BPM, 3/4, D major. Dreamy: slow-attack pad, music-box arpeggios,
    gamelan bell melody in D major pentatonic, soft sine bass."""
    track = Track("music_title", 84, 3, 16, seed=1101, reverbSize=0.85, reverbDamping=0.45,
                  delayBeats=1.5, delayFeedback=0.42)
    spb = track.secondsPerBeat
    track.addBus("pad", level=-21.0, reverb=0.45, filters=[("lowpass", 3200), ("highpass", 110)], width=1.4)
    track.addBus("box", level=-27.0, reverb=0.35, delay=0.3, filters=[("highpass", 300)])
    track.addBus("bells", level=-19.0, reverb=0.4, delay=0.28, filters=[("highpass", 200)])
    track.addBus("bass", level=-25.0, reverb=0.04, filters=[("lowpass", 600)])
    track.addBus("shaker", level=-36.0, reverb=0.3, filters=[("highpass", 4000)])
    kit = makeKit(track.rng, ["shaker"])

    progression = ["Dmaj9", "F#m7", "Gmaj9", "A7sus", "Dmaj9", "Bm9", "Em9", "A7sus",
                   "Gmaj9", "F#m7", "Bm9", "Dmaj9", "Gmaj9", "Bm9", "Em9", "A7sus"]
    events = chordEvents(progression, 3)
    voicings = voiceProgression(events, 55, 74, 4)
    for index in range(len(events)):
        beat, beats, symbol = events[index]
        bar = int(beat // 3)
        level = 0.6 if bar < 4 else (0.8 if bar < 8 else 1.0)
        brightness = 0.3 if bar < 8 else 0.45
        for midi in voicings[index]:
            sound = padNote(track.rng, midiToFreq(midi), beats * spb + 0.3, 0.9, 1.8, brightness)
            track.place("pad", beat - 0.25, sound, level, 0.0, humanise=False)
        # music-box arpeggio an octave above the pad (sparse intro, then flowing 8ths)
        notes = [midi + 12 for midi in voicings[index]]
        if bar < 4:
            steps = [(0.0, 0), (1.0, 2), (1.5, 3), (2.5, 1)]
        elif bar < 12:
            steps = [(0.0, 0), (0.5, 1), (1.0, 2), (1.5, 3), (2.0, 2), (2.5, 1)]
        else:
            steps = [(0.0, 3), (0.5, 2), (1.0, 1), (1.5, 0), (2.0, 1), (2.5, 2)]
        for offset, which in steps:
            velocity = 0.8 if offset == 0.0 else 0.55
            pan = -0.35 if which % 2 == 0 else 0.35
            track.place("box", beat + offset, musicBoxNote(midiToFreq(notes[which]), velocity), velocity, pan)
        if bar >= 4:
            bass = bassNote(midiToFreq(bassMidi(symbol, 38)), beats * spb - 0.15, 0.6, 0.25, 0.8, 0.5)
            track.place("bass", beat, bass, 1.0)
        if bar >= 8:
            for step in range(6):
                track.hit("shaker", kit["shaker"], beat + step * 0.5, [1.0, 0.4, 0.7, 0.4, 0.7, 0.45][step])

    melody = ["A5:2 F#5:1", "E5:1.5 F#5:0.5 D5:1", "B4:2 D5:1", "E5:3",
              "D5:1 E5:1 F#5:1", "A5:2 F#5:1", "B5:1.5 A5:0.5 F#5:1", "E5:3",
              "D6:1.5 B5:0.5 A5:1", "F#5:2 A5:1", "B5:1 A5:1 F#5:1", "E5:2 r:1"]
    for beat, beats, midi in parseMelody(melody, 3, firstBar=4):
        velocity = 0.85 if beat % 3 == 0 else 0.7
        track.place("bells", beat, gamelanBell(track.rng, midiToFreq(midi), beats * spb, velocity), velocity, 0.12)
    # a little music-box sparkle that opens every loop
    sparkle = ["D6", "B5", "A5", "F#5", "E5", "D5"]
    for index in range(len(sparkle)):
        track.place("box", index * 0.25, musicBoxNote(midiToFreq(noteToMidi(sparkle[index])), 0.5), 0.45,
                    0.5 - 0.2 * index)
    return finishTrack(track)


def makeStreet():
    """96 BPM, 4/4, F major, 12 bars. Lo-fi stroll: FM electric piano with
    tremolo, walking-style bass, swung 16th drums, gamelan bell melody."""
    track = Track("music_street", 96, 4, 12, seed=2202, swing=0.16, reverbSize=0.6,
                  reverbDamping=0.5, delayBeats=0.75, delayFeedback=0.35)
    spb = track.secondsPerBeat
    track.addBus("ep", level=-21.0, reverb=0.18, filters=[("lowpass", 4200), ("highpass", 120)],
                 duck=0.25, autoPan=(0.5, 0.35))
    track.addBus("bass", level=-19.0, reverb=0.02, filters=[("lowpass", 900)])
    track.addBus("kick", level=-19.0, reverb=0.0, filters=[("lowpass", 5000)])
    track.addBus("drums", level=-25.0, reverb=0.12, filters=[("lowpass", 7000)])
    track.addBus("hats", level=-31.0, reverb=0.08, filters=[("highpass", 3000), ("lowpass", 10000)])
    track.addBus("bells", level=-19.0, reverb=0.3, delay=0.25, filters=[("highpass", 250)])
    track.addBus("pluck", level=-23.0, reverb=0.22, delay=0.18, filters=[("highpass", 200)])
    kit = makeKit(track.rng, ["kick", "snare", "rim", "shaker", "hat", "openhat"])

    progression = ["Fmaj9", "Em7 A7", "Dm9", "Cm7 F7", "Bbmaj9", "Am7 D7",
                   "Gm9", "C7sus", "Fmaj9", "Dm9", "Gm9", "Gm7 C7"]
    events = chordEvents(progression, 4)
    voicings = voiceProgression(events, 52, 72, 4, rootless=True)
    for index in range(len(events)):
        beat, beats, symbol = events[index]
        bar = int(beat // 4)
        nextSymbol = events[(index + 1) % len(events)][2]
        if beats == 4:
            if bar < 4:
                comp = [(0.0, 1.9, 0.7), (2.5, 1.3, 0.55)]
            else:
                comp = [(0.0, 1.4, 0.75), (1.75, 0.25, 0.4), (2.5, 0.9, 0.6), (3.75, 0.25, 0.35)]
        else:
            comp = [(0.0, 1.4, 0.7), (1.5, 0.4, 0.4)]
        for offset, length, velocity in comp:
            for midi in voicings[index]:
                spread = 0.012 * (midi - voicings[index][0]) / 12.0     # tiny "strum" between fingers
                track.place("ep", beat + offset + spread, electricPiano(midiToFreq(midi), length * spb, velocity), 1.0)
        root = chordRoot(symbol)
        nextRoot = chordRoot(nextSymbol)
        approach = nextRoot - 1 if nextRoot > root else nextRoot + 1
        if beats == 4:
            line = [(0.0, root, 1.4), (1.5, root + 7, 0.9), (2.5, root + 12, 0.9), (3.5, approach, 0.45)]
        else:
            line = [(0.0, root, 1.4), (1.5, approach, 0.45)]
        for offset, midi, length in line:
            track.place("bass", beat + offset, bassNote(midiToFreq(midi), length * spb, 0.8), 1.0)

    for bar in range(12):
        start = bar * 4
        track.playPattern("kick", kit["kick"], "X.....x...x.....", start, 1.0)
        track.playPattern("hats", kit["shaker"], "XoxoXoxoXoxoXoxo", start, 1.0, 0.25)
        if bar < 4:
            track.playPattern("drums", kit["rim"], "....X.......X...", start, 0.8, -0.1)
        else:
            track.playPattern("drums", kit["snare"], "....X.......X...", start, 1.0)
            track.playPattern("drums", kit["rim"], "..o.......o....o", start, 0.6, -0.1)
            track.playPattern("hats", kit["hat"], "x.x.x.x.x.x.x.x.", start, 0.8, -0.2)
        if bar == 11:
            track.playPattern("drums", kit["snare"], "............xoxX", start, 0.8)
            track.hit("hats", kit["openhat"], start + 3.5, 0.9, -0.2)
        if bar in (3, 7):
            track.hit("hats", kit["openhat"], start + 3.5, 0.7, -0.2)

    bellBars = ["A5:1 C6:0.5 A5:0.5 G5:1 F5:1", "G5:1.5 F5:0.5 A5:1 C6:1", "D6:2 C6:1 A5:1",
                "A5:1.5 G5:0.5 A5:1 C6:1", "D6:1.5 C6:0.5 A5:1 G5:1", "F5:3 r:1",
                "A5:1 C6:0.5 D6:0.5 C6:1 A5:1", "F5:1.5 G5:0.5 A5:2"]
    for beat, beats, midi in parseMelody(bellBars, 4, firstBar=2):
        velocity = 0.85 if beat % 2 == 0 else 0.7
        track.place("bells", beat, gamelanBell(track.rng, midiToFreq(midi), beats * spb, velocity, 3.5), velocity, 0.15)
    answer = ["G5:1 A5:1 C6:1 D6:1", "Bb5:1 A5:1 G5:2"]
    for beat, beats, midi in parseMelody(answer, 4, firstBar=10):
        track.place("pluck", beat, pluckNote(track.rng, midiToFreq(midi), beats * spb, 1.2, 0.6, 0.18), 0.9, -0.25)
    # soft plucked arpeggio "footsteps" in the middle section
    for index in range(len(events)):
        beat, beats, symbol = events[index]
        bar = int(beat // 4)
        if 4 <= bar < 10:
            notes = voicings[index]
            for step in range(int(beats * 2)):
                if step % 2 == 1:
                    midi = notes[[0, 2, 1, 3][(step // 2) % 4]] + 12
                    track.place("pluck", beat + step * 0.5, pluckNote(track.rng, midiToFreq(midi), 0.4, 0.7, 0.45),
                                0.35, 0.3)
    return finishTrack(track)


ROJAK_A = ["B5:0.5 D6:0.5 E6:0.5 D6:0.5 B5:1 A5:0.5 G5:0.5", "E5:0.5 G5:0.5 A5:0.5 B5:1.5 r:1",
           "E6:0.5 D6:0.5 B5:0.5 G5:0.5 A5:1 G5:0.5 E5:0.5", "D5:1 E5:0.5 F#5:0.5 A5:2"]
ROJAK_B = [ROJAK_A[0], "D6:0.5 B5:0.5 A5:0.5 F#5:1.5 r:1", ROJAK_A[2], "G5:1 A5:0.5 B5:0.5 A5:1 r:1"]
ROJAK_C = ["G5:0.5 B5:0.5 D6:1 B5:0.5 A5:0.5 G5:1", "E5:1 G5:0.5 A5:0.5 B5:2",
           "C6:0.5 B5:0.5 A5:0.5 G5:0.5 E5:1 G5:1", "A5:2 G5:0.5 A5:0.5 r:1"]
ROJAK_D = [ROJAK_A[0], ROJAK_B[1], "E6:0.5 D6:0.5 B5:0.5 G5:0.5 A5:1 F#5:1",
           "C6:0.5 B5:0.5 A5:0.5 G5:0.5 A5:1 F#5:0.5 D5:0.5"]


def makeRojak():
    """118 BPM, 4/4, G major pentatonic. Bouncy marimba ostinato, octave-
    jumping bass, plucked (Karplus-Strong) hook. Aura: bright synth lead
    on the hook + interlocking kompang (two players, left and right)."""
    track = Track("music_rojak", 118, 4, 16, seed=3303, swing=0.1, reverbSize=0.5,
                  reverbDamping=0.5, delayBeats=0.75, delayFeedback=0.32)
    spb = track.secondsPerBeat
    track.addBus("marimba", level=-21.0, reverb=0.14, filters=[("highpass", 150)], duck=0.15)
    track.addBus("bass", level=-19.0, reverb=0.02, filters=[("lowpass", 1000)])
    track.addBus("kick", level=-18.0, filters=[("lowpass", 6000)])
    track.addBus("drums", level=-24.0, reverb=0.12, filters=[("lowpass", 8000)])
    track.addBus("hats", level=-31.0, reverb=0.06, filters=[("highpass", 3500)])
    track.addBus("pluck", level=-20.0, reverb=0.18, delay=0.16, filters=[("highpass", 180)])
    track.addBus("pad", level=-29.0, reverb=0.3, filters=[("lowpass", 2500), ("highpass", 150)], width=1.3, duck=0.3)
    track.addBus("lead", layer="aura", level=-18.0, reverb=0.2, delay=0.22, filters=[("highpass", 250)])
    track.addBus("kompang", layer="aura", level=-22.0, reverb=0.12, filters=[("highpass", 90)])
    track.addBus("sparkle", layer="aura", level=-30.0, reverb=0.35, filters=[("highpass", 800)])
    kit = makeKit(track.rng, ["kick", "clap", "rim", "shaker", "openhat", "bung", "pak", "tak"])

    progression = ["Gadd9", "Em7", "Cmaj7", "D", "Gadd9", "Bm7", "Cmaj7", "D7sus D7",
                   "Em7", "Cmaj7", "Am7", "D7sus", "Gadd9", "Bm7", "Cmaj7 D", "Am7 D7"]
    events = chordEvents(progression, 4)
    voicings = voiceProgression(events, 60, 79, 4)
    padVoicings = voiceProgression(events, 52, 71, 4)
    patternA = [(0.0, 0), (0.5, 2), (0.75, 1), (1.25, 3), (1.5, 2), (2.0, 0), (2.5, 2), (2.75, 1), (3.25, 3), (3.5, 1)]
    patternB = [(0.0, 3), (0.25, 2), (0.75, 1), (1.0, 2), (1.5, 0), (2.0, 3), (2.25, 2), (2.75, 1), (3.0, 0), (3.5, 1)]
    for index in range(len(events)):
        beat, beats, symbol = events[index]
        bar = int(beat // 4)
        pattern = patternB if 8 <= bar < 12 else patternA
        for offset, which in pattern:
            if offset < beats:
                velocity = 0.9 if offset % 1.0 == 0.0 else 0.65
                track.place("marimba", beat + offset, marimbaNote(track.rng, midiToFreq(voicings[index][which]), velocity),
                            velocity, -0.3 + 0.2 * which)
        root = chordRoot(symbol, 36)
        fifth = root + 7
        if beats == 4:
            line = [(0.0, root, 0.4), (0.75, root + 12, 0.2), (1.5, root, 0.4), (2.5, fifth, 0.4),
                    (3.0, root + 12, 0.2), (3.5, root, 0.4)]
        else:
            line = [(0.0, root, 0.4), (0.75, root + 12, 0.2), (1.5, root, 0.4)]
        if 8 <= bar < 12:
            line = [(0.0, root, 1.4), (1.5, root, 0.4), (2.5, fifth, 1.2)] if beats == 4 else [(0.0, root, 1.4)]
        for offset, midi, length in line:
            track.place("bass", beat + offset, bassNote(midiToFreq(midi), length * spb, 0.85, 0.45, 0.6, 0.12), 1.0)
        if bar >= 8:
            level = 1.0 if bar < 12 else 0.7
            for midi in padVoicings[index]:
                track.place("pad", beat, padNote(track.rng, midiToFreq(midi), beats * spb, 0.3, 0.6, 0.45), level, 0.0, False)

    for bar in range(16):
        start = bar * 4
        if 8 <= bar < 12:
            track.playPattern("kick", kit["kick"], "X.......X.......", start)
            track.playPattern("drums", kit["rim"], "....x.......x..o", start, 0.8)
        else:
            track.playPattern("kick", kit["kick"], "X.....X.X.....x.", start)
            if bar < 4:
                track.playPattern("drums", kit["rim"], "....X.......X...", start, 0.8)
            else:
                track.playPattern("drums", kit["clap"], "....X.......X...", start)
        track.playPattern("hats", kit["shaker"], "XoxoXoxoXoxoXoxo", start, 1.0, 0.25)
        if bar >= 12:
            track.playPattern("hats", kit["openhat"], "..x...x...x...x.", start, 0.5, -0.25)
        if bar in (7, 15):
            track.playPattern("drums", kit["clap"], "............xxXX", start, 0.7)

    for beat, beats, midi in parseMelody(ROJAK_B + ROJAK_C + ROJAK_D, 4, firstBar=4):
        velocity = 0.9 if beat % 1.0 == 0.0 else 0.72
        track.place("pluck", beat, pluckNote(track.rng, midiToFreq(midi - 12), beats * spb * 0.9, 0.9, 0.75, 0.15),
                    velocity, -0.1)

    # ---- aura: bright lead on the hook (all 16 bars) + kompang ensemble
    track.placeLead("lead", parseMelody(ROJAK_A + ROJAK_B + ROJAK_C + ROJAK_D, 4), "lead", 1.0, 0.0, 0, 0.85)
    playerOne = "B...P.b.B...P..."
    playerTwo = "..p....p..p...pp"
    fillOne = "B...P.b.PpPpPPPP"
    fillTwo = "..p....p.t.tpPpP"
    for bar in range(16):
        start = bar * 4
        fill = bar % 4 == 3
        playKompang(track, kit, "kompang", fillOne if fill else playerOne, start, 1.0, -0.45)
        playKompang(track, kit, "kompang", fillTwo if fill else playerTwo, start, 0.9, 0.45)
        if bar % 4 == 0:
            for index in range(4):
                midi = [86, 83, 81, 79][index]
                track.place("sparkle", start + index * 0.25, musicBoxNote(midiToFreq(midi), 0.6), 0.7, 0.4 - 0.25 * index)
    return finishTrack(track, "music_rojak_aura", -21.5)


def makeWok():
    """128 BPM, 4/4, E minor. Funk: syncopated kick / snare with ghost notes,
    16th hats, octave bass, brass stabs, palm-muted plucks, a Chinese gong
    on the big downbeats. Aura: 16th-note FM arpeggios with ping-pong delay,
    hand claps and noise risers."""
    track = Track("music_wok", 128, 4, 16, seed=4404, swing=0.0, reverbSize=0.45,
                  reverbDamping=0.55, delayBeats=0.75, delayFeedback=0.38)
    spb = track.secondsPerBeat
    track.addBus("kick", level=-17.0, filters=[("lowpass", 6000)])
    track.addBus("drums", level=-21.0, reverb=0.1, filters=[("lowpass", 9000)])
    track.addBus("hats", level=-29.0, reverb=0.05, filters=[("highpass", 4000)])
    track.addBus("bass", level=-18.0, reverb=0.0, filters=[("lowpass", 1400)], duck=0.2)
    track.addBus("brass", level=-20.0, reverb=0.16, delay=0.06, filters=[("highpass", 180), ("peak", 1200, 0.8, 2.0)])
    track.addBus("guitar", level=-28.0, reverb=0.08, filters=[("highpass", 300)])
    track.addBus("gong", level=-25.0, reverb=0.3, filters=[("highpass", 60)])
    track.addBus("arps", layer="aura", level=-20.0, reverb=0.15, delay=0.3, filters=[("highpass", 300)])
    track.addBus("claps", layer="aura", level=-22.0, reverb=0.15, filters=[("highpass", 400)])
    track.addBus("riser", layer="aura", level=-30.0, reverb=0.2)
    kit = makeKit(track.rng, ["kick", "snare", "hat", "openhat", "clap"])
    toms = [tomDrum(track.rng, 180.0), tomDrum(track.rng, 140.0), tomDrum(track.rng, 105.0)]

    progression = ["Em9", "Em9", "Cmaj7", "B7sus B7", "Em9", "G6", "Am9", "B7sus B7",
                   "Cmaj9", "D6", "Bm7", "Em9", "Am9", "Cmaj7", "B7sus", "B7"]
    events = chordEvents(progression, 4)
    voicings = voiceProgression(events, 55, 76, 4)
    arpVoicings = voiceProgression(events, 62, 81, 4)
    for index in range(len(events)):
        beat, beats, symbol = events[index]
        bar = int(beat // 4)
        if beats == 4:
            if bar < 4:
                stabs = [(0.0, 0.35, 0.9), (3.5, 0.3, 0.7)]
            elif 8 <= bar < 12:
                stabs = [(0.0, 1.7, 0.6), (2.5, 1.3, 0.65)]
            else:
                stabs = [(0.0, 0.3, 1.0), (0.75, 0.2, 0.7), (1.5, 0.3, 0.8), (2.75, 0.25, 0.75), (3.5, 0.4, 0.9)]
        else:
            stabs = [(0.0, 0.3, 0.9), (1.5, 0.3, 0.75)]
        for offset, length, velocity in stabs:
            for midi in voicings[index]:
                for cents in (-4.0, 4.0):
                    freq = midiToFreq(midi) * 2.0 ** (cents / 1200.0)
                    track.place("brass", beat + offset, brassNote(freq, length * spb, velocity), 0.5,
                                cents / 10.0)
        root = chordRoot(symbol, 36)
        seventh = root + (10 if 10 in parseChord(symbol)[1] else 7)
        if beats == 4:
            line = [(0.0, root, 0.4), (0.75, root, 0.2), (1.25, root + 12, 0.2), (1.5, root, 0.35),
                    (2.5, root + 7, 0.35), (3.0, seventh, 0.2), (3.25, root + 12, 0.2), (3.5, root, 0.35)]
        else:
            line = [(0.0, root, 0.4), (0.75, root, 0.2), (1.5, root + 12, 0.3)]
        for offset, midi, length in line:
            track.place("bass", beat + offset, bassNote(midiToFreq(midi), length * spb, 0.9, 0.55, 0.55, 0.1), 1.0)
        if bar >= 4:
            top = voicings[index][-2:]
            for step in range(int(beats * 4)):
                if "x.xxx.x.xx.xx.x."[step % 16] == "x":
                    midi = top[step % 2]
                    velocity = 0.9 if step % 4 == 0 else 0.6
                    track.place("guitar", beat + step * 0.25, pluckNote(track.rng, midiToFreq(midi), 0.1, 0.12, 0.55, 0.3),
                                velocity, 0.35)
        # ---- aura arpeggio: broken thirds climbing over two octaves
        notes = arpVoicings[index] + [midi + 12 for midi in arpVoicings[index]]
        order = [0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 6, 4, 5, 3]
        for step in range(int(beats * 4)):
            midi = notes[order[step % 16]]
            velocity = 0.95 if step % 4 == 0 else 0.7
            track.place("arps", beat + step * 0.25, fmPluck(midiToFreq(midi), 0.2, velocity), velocity,
                        -0.4 + 0.8 * ((step % 4) / 3.0))

    for bar in range(16):
        start = bar * 4
        track.playPattern("kick", kit["kick"], "X.........X....." if bar < 4 else "X..x......X..x..", start)
        track.playPattern("drums", kit["snare"], "....X..o.o..X..o", start)
        if bar < 4:
            track.playPattern("hats", kit["hat"], "x.x.x.x.x.x.x.x.", start, 1.0, 0.2)
        else:
            track.playPattern("hats", kit["hat"], "XxoxXxoxXxoxXxox", start, 1.0, 0.2)
        if bar % 2 == 1:
            track.hit("hats", kit["openhat"], start + 3.5, 0.8, 0.2)
        if bar in (7, 15):
            for step in range(8):
                track.hit("drums", [toms[min(2, step // 3)]], start + 2.0 + step * 0.25, 0.6 + 0.05 * step,
                          0.4 - 0.1 * step)
        # aura claps (with the snare backbeat plus a pickup every second bar)
        track.playPattern("claps", kit["clap"], "....X.......X..." if bar % 2 == 0 else "....X.......X.xx", start)
    track.place("gong", 0.0, gongHit(track.rng, 98.0), 1.0, -0.2)
    track.place("gong", 32.0, gongHit(track.rng, 98.0), 0.9, 0.2)
    track.place("riser", 7 * 4.0, noiseRiser(track.rng, 4 * spb), 1.0, 0.0, False)
    track.place("riser", 15 * 4.0, noiseRiser(track.rng, 4 * spb), 1.0, 0.0, False)
    riffs = {3: "r:2 B4:0.25 D5:0.25 E5:0.25 F#5:0.25 A5:0.5 B5:0.5",
             7: "r:2 B4:0.25 D5:0.25 E5:0.25 F#5:0.25 A5:0.5 B5:0.5",
             11: "r:2 E5:0.25 G5:0.25 A5:0.25 B5:0.25 D6:0.5 E6:0.5",
             15: "r:2 B4:0.25 D5:0.25 E5:0.25 F#5:0.25 A5:0.5 B5:0.5"}
    for bar in riffs:
        for beat, beats, midi in parseMelody([riffs[bar]], 4, firstBar=bar):
            for octave in (0, -12):
                track.place("brass", beat, brassNote(midiToFreq(midi + octave), beats * spb * 0.9, 0.9), 0.9, 0.0)
    return finishTrack(track, "music_wok_aura", -22.0)


def makeTehTarik():
    """92 BPM, 4/4, D major, 12 bars. Bossa nova / lo-fi: rootless e-piano
    voicings in the bossa rhythm, nylon guitar (Karplus-Strong) melody,
    root-fifth bass, cross-stick clave, brushes, a little vinyl crackle.
    Aura: breathy flute lead with vibrato."""
    track = Track("music_tehtarik", 92, 4, 12, seed=5505, swing=0.12, reverbSize=0.55,
                  reverbDamping=0.6, delayBeats=0.75, delayFeedback=0.3)
    spb = track.secondsPerBeat
    track.addBus("ep", level=-21.0, reverb=0.2, filters=[("lowpass", 3400), ("highpass", 110)], autoPan=(0.25, 0.4))
    track.addBus("guitar", level=-19.0, reverb=0.2, delay=0.08, filters=[("highpass", 110), ("peak", 220, 1.0, 2.0)])
    track.addBus("bass", level=-19.0, reverb=0.02, filters=[("lowpass", 700)])
    track.addBus("kick", level=-24.0, filters=[("lowpass", 3000)])
    track.addBus("drums", level=-27.0, reverb=0.12, filters=[("lowpass", 7000)])
    track.addBus("hats", level=-33.0, reverb=0.08, filters=[("highpass", 4000)])
    track.addBus("vinyl", level=-46.0, reverb=0.0, filters=[("highpass", 900), ("lowpass", 6000)])
    track.addBus("flute", layer="aura", level=-17.0, reverb=0.28, delay=0.16, filters=[("highpass", 200)])
    track.addBus("auraep", layer="aura", level=-28.0, reverb=0.3, delay=0.2, filters=[("highpass", 500)])
    kit = makeKit(track.rng, ["softkick", "rim", "brush", "shaker"])

    progression = ["Dmaj9", "Dmaj9", "Em9", "A13", "F#m7", "B7b9", "Em9", "A13",
                   "Gmaj9", "Gm6", "F#m7 B7b9", "Em9 A13"]
    events = chordEvents(progression, 4)
    voicings = voiceProgression(events, 53, 74, 4, rootless=True)
    for index in range(len(events)):
        beat, beats, symbol = events[index]
        bar = int(beat // 4)
        if beats == 4:
            comp = [(0.0, 0.9), (1.5, 0.9), (3.0, 0.9)] if bar % 2 == 0 else [(1.0, 0.9), (2.5, 1.4)]
        else:
            comp = [(0.0, 0.9), (1.5, 0.4)]
        for offset, length in comp:
            velocity = 0.6 if bar < 4 else 0.68
            for midi in voicings[index]:
                track.place("ep", beat + offset, electricPiano(midiToFreq(midi), length * spb, velocity), 1.0)
        root = bassMidi(symbol, 36)
        fifth = root + 7
        if beats == 4:
            line = [(0.0, root, 1.3), (1.5, root, 0.4), (2.0, fifth, 1.3), (3.5, fifth, 0.4)]
        else:
            line = [(0.0, root, 1.3), (1.5, root, 0.4)]
        for offset, midi, length in line:
            track.place("bass", beat + offset, bassNote(midiToFreq(midi), length * spb, 0.7, 0.3, 0.4, 0.35), 1.0)
        # aura: soft high e-piano "sparkle" chord an octave up on the 2-and
        if beats == 4:
            for midi in voicings[index][1:]:
                track.place("auraep", beat + 1.5, electricPiano(midiToFreq(midi + 12), 1.5 * spb, 0.5), 1.0)

    for bar in range(12):
        start = bar * 4
        clave = "X.....X.....X..." if bar % 2 == 0 else "....X.....X....."
        track.playPattern("drums", kit["rim"], clave, start, 0.75, -0.15)
        track.playPattern("hats", kit["shaker"], "XoxoXoxoXoxoXoxo", start, 1.0, 0.25)
        if bar >= 4:
            track.playPattern("drums", kit["brush"], "....x.......x...", start, 0.9, 0.1)
        if bar >= 8:
            track.playPattern("kick", kit["softkick"], "X......xX......x", start)

    guitarLine = ["A5:1.5 F#5:0.5 E5:1 F#5:1", "C#5:3 r:1", "r:0.5 B4:0.5 D5:0.5 E5:0.5 F#5:2", "E5:1 G5:1 F#5:2",
                  "A5:1.5 C#6:0.5 B5:1 A5:1", "A5:1 F#5:1 D#5:2", "E5:1 F#5:0.5 G5:0.5 B5:2", "A5:1 G5:0.5 F#5:0.5 E5:2",
                  "D6:1.5 B5:0.5 A5:1 F#5:1", "E5:1.5 D5:0.5 Bb4:2", "A5:1 C#6:1 C6:1 A5:1", "G5:1 F#5:1 E5:1 C#5:1"]
    melody = parseMelody(guitarLine, 4)
    for beat, beats, midi in melody:
        if beat >= 16:
            velocity = 0.85 if beat % 1.0 == 0.0 else 0.72
            track.place("guitar", beat, pluckNote(track.rng, midiToFreq(midi - 12), beats * spb, 2.2, 0.38, 0.14, 0.1),
                        velocity, -0.2)
    track.placeLead("flute", melody, "flute", 1.0, 0.1, 0, 1.0)

    # lo-fi vinyl crackle: random tiny clicks all over the loop
    for index in range(int(track.loopSeconds * 7)):
        count = samples(0.003)
        click = track.rng.uniform(-1, 1, count) * np.exp(-np.arange(count) / 20.0)
        track.placeAtSeconds("vinyl", track.rng.uniform(0, track.loopSeconds), click,
                             track.rng.uniform(0.2, 1.0) ** 2, track.rng.uniform(-0.6, 0.6))
    return finishTrack(track, "music_tehtarik_aura", -22.0)


BANQUET_ERHU = ["E5:2 G5:0.5 E5:0.5", "D5:1 C5:1 A4:1", "C5:1.5 D5:0.5 E5:1", "D5:3",
                "E5:1 A5:1.5 G5:0.5", "A5:1 C6:1 A5:1", "G5:1.5 E5:0.5 D5:1", "E5:3",
                "C6:2 A5:1", "G5:1.5 E5:0.5 D5:1", "E5:1 G5:1 D5:1", "A4:3",
                "D5:1 E5:1 A5:1", "G5:1.5 A5:0.5 C6:1", "D6:2 C6:0.5 A5:0.5", "A5:1 G5:1 D5:1"]
BANQUET_BELLS = ["r:1 A5:0.5 C6:0.5 E6:1", "r:1 E6:1 C6:1", "r:1.5 G5:0.5 C6:1", "E6:1 D6:1 G5:1",
                 "r:2 C6:1", "r:1.5 D6:0.5 C6:1", "B5:1 A5:1 E6:1", "r:1 G6:0.5 E6:0.5 D6:0.5 B5:0.5",
                 "r:2 A5:1", "r:1 B5:1 D6:1", "r:1 E6:1 G6:1", "C6:1 E6:1 A6:1",
                 "r:2 E6:1", "r:1 A6:1 E6:1", "r:1.5 B5:0.5 G5:1", "E6:1 B5:1 A5:1"]


def makeBanquet():
    """100 BPM, 3/4 waltz, A minor pentatonic. Elegant: bowed string pad
    with vibrato, pizzicato waltz (bass on 1, chords on 2 and 3), erhu-like
    lead with slides and vibrato, harp arpeggios. Aura: gamelan bell
    countermelody + soft muted hand-drum on beats 2 and 3."""
    track = Track("music_banquet", 100, 3, 16, seed=6606, reverbSize=0.75, reverbDamping=0.45,
                  delayBeats=1.0, delayFeedback=0.35)
    spb = track.secondsPerBeat
    track.addBus("strings", level=-21.0, reverb=0.35, filters=[("lowpass", 3800), ("highpass", 100)], width=1.4)
    track.addBus("pizz", level=-23.0, reverb=0.2, filters=[("highpass", 70)])
    track.addBus("cello", level=-23.0, reverb=0.25, filters=[("lowpass", 1500), ("highpass", 45)])
    track.addBus("erhu", level=-18.0, reverb=0.3, delay=0.1, filters=[("highpass", 250)])
    track.addBus("harp", level=-25.0, reverb=0.3, delay=0.12, filters=[("highpass", 200)])
    track.addBus("perc", level=-28.0, reverb=0.25, filters=[("highpass", 80)])
    track.addBus("bells", layer="aura", level=-18.0, reverb=0.35, delay=0.25, filters=[("highpass", 300)])
    track.addBus("hand", layer="aura", level=-29.0, reverb=0.2, filters=[("highpass", 200)])
    kit = makeKit(track.rng, ["bung", "tak"])
    ting = [fingerCymbal(track.rng, 2950.0), fingerCymbal(track.rng, 3010.0)]

    progression = ["Am9", "Fmaj7", "C", "G6", "Am9", "Dm7", "Esus4", "Em7",
                   "Fmaj7", "G6", "Em7", "Am9", "Dm9", "Fmaj7", "G6", "Esus4"]
    events = chordEvents(progression, 3)
    voicings = voiceProgression(events, 52, 72, 4)
    for index in range(len(events)):
        beat, beats, symbol = events[index]
        bar = int(beat // 3)
        level = [0.55, 0.75, 0.95, 1.0][bar // 4]
        for midi in voicings[index]:
            sound = padNote(track.rng, midiToFreq(midi), beats * spb + 0.15, 0.45, 1.0, 0.55, 5.0, 11.0, 5.3)
            track.place("strings", beat - 0.1, sound, level, 0.0, False)
        if bar >= 8:
            top = voicings[index][-1] + 12
            track.place("strings", beat - 0.1, padNote(track.rng, midiToFreq(top), beats * spb + 0.15, 0.5, 1.0, 0.5,
                                                       5.0, 12.0, 5.5), level * 0.8, 0.0, False)
        # bowed cello under the waltz: one warm sustained bass note per bar
        cello = padNote(track.rng, midiToFreq(bassMidi(symbol, 36)), beats * spb + 0.1, 0.25, 0.6, 0.35, 3.0, 9.0, 5.0)
        track.place("cello", beat - 0.05, cello, 0.7 if bar < 4 else 1.0, 0.0, False)
        # pizzicato waltz: bass on beat 1, chord on beats 2 and 3
        root = bassMidi(symbol, 40)
        track.place("pizz", beat, pluckNote(track.rng, midiToFreq(root), 0.5, 0.6, 0.35, 0.2), 1.0, -0.1)
        if bar >= 4:
            for offset in (1.0, 2.0):
                for midi in voicings[index][1:]:
                    track.place("pizz", beat + offset, pluckNote(track.rng, midiToFreq(midi), 0.2, 0.3, 0.45, 0.2),
                                0.4, 0.25)
        if bar >= 8:
            notes = voicings[index] + [voicings[index][1] + 12, voicings[index][2] + 12]
            for step in range(6):
                midi = notes[step]
                track.place("harp", beat + step * 0.5, pluckNote(track.rng, midiToFreq(midi), 1.2, 2.2, 0.6, 0.3),
                            0.5 if step else 0.7, -0.4 + 0.15 * step)
        if bar >= 4:
            track.hit("perc", kit["bung"], beat, 0.6, -0.1)
        if bar % 4 == 0:
            track.hit("perc", ting, beat, 0.35, 0.3)
        # aura hand drum: muted taps on beats 2 and 3 of the waltz
        track.hit("hand", kit["tak"], beat + 1.0, 0.7, 0.3)
        track.hit("hand", kit["tak"], beat + 2.0, 0.55, 0.35)
        if bar % 2 == 1:
            track.hit("hand", kit["tak"], beat + 2.5, 0.4, 0.35)

    track.placeLead("erhu", parseMelody(BANQUET_ERHU, 3), "erhu", 1.0, -0.05, 0, 1.0)
    for beat, beats, midi in parseMelody(BANQUET_BELLS, 3):
        velocity = 0.8 if beat % 3 == 0 else 0.7
        track.place("bells", beat, gamelanBell(track.rng, midiToFreq(midi), beats * spb, velocity, 4.0), velocity,
                    0.3 if midi > 84 else -0.2)
    return finishTrack(track, "music_banquet_aura", -22.5)


FINALE_MELODY = ["E5:1.5 D5:0.5 C5:1 A4:1", "G4:1 A4:0.5 B4:0.5 D5:2", "E5:2 G5:1 E5:1", "A5:2 G5:1 E5:1",
                 "F5:1.5 E5:0.5 D5:1 C5:1", "D5:1 C5:1 B4:1 D5:1", "E5:1 G5:1 C6:1.5 B5:0.5", "C6:2 Bb5:1 G5:1",
                 "A5:1 C6:1 E6:2", "D6:1.5 C6:0.5 B5:1 G5:1", "G5:1 B5:1 A5:1 C#6:1", "D6:2 C6:1 D6:1"]


def makeFinale():
    """88 BPM, 4/4, C major, 12 bars. The "royal road" progression
    (IV - V - iii - vi), loved in East-Asian pop ballads. Sparse piano ->
    bells + bass + pad -> full climax with half-time drums, kompang, crash."""
    track = Track("music_finale", 88, 4, 12, seed=7707, reverbSize=0.85, reverbDamping=0.4,
                  delayBeats=1.5, delayFeedback=0.38)
    spb = track.secondsPerBeat
    track.addBus("piano", level=-19.0, reverb=0.3, filters=[("highpass", 60), ("highshelf", 6000, 0.7, -3.0)])
    track.addBus("pad", level=-21.0, reverb=0.4, filters=[("lowpass", 3500), ("highpass", 120)], width=1.5)
    track.addBus("bells", level=-19.0, reverb=0.38, delay=0.24, filters=[("highpass", 250)])
    track.addBus("bass", level=-21.0, reverb=0.03, filters=[("lowpass", 700)])
    track.addBus("kick", level=-20.0, filters=[("lowpass", 4000)])
    track.addBus("drums", level=-25.0, reverb=0.3, filters=[("lowpass", 8000)])
    track.addBus("kompang", level=-25.0, reverb=0.25, filters=[("highpass", 90)])
    track.addBus("cymbal", level=-29.0, reverb=0.4, filters=[("highpass", 2500)])
    kit = makeKit(track.rng, ["kick", "snare", "bung", "pak"])

    progression = ["Fmaj7", "G", "Em7", "Am7", "Dm7", "G7sus G", "Cmaj7", "C7",
                   "Fmaj7", "G", "Em7 A7", "Dm7 G7sus"]
    events = chordEvents(progression, 4)
    leftHand = voiceProgression(events, 48, 64, 3)
    padVoicings = voiceProgression(events, 52, 74, 4)
    for index in range(len(events)):
        beat, beats, symbol = events[index]
        bar = int(beat // 4)
        root = bassMidi(symbol, 36)
        if bar < 4:
            # sparse: soft chords on beats 1 and 3
            for offset in (0.0, 2.0):
                for midi in leftHand[index]:
                    track.place("piano", beat + offset, pianoNote(midiToFreq(midi), 1.9 * spb, 0.45), 0.9)
        else:
            # flowing 8th-note arpeggio: low root, then the chord tones up and down
            notes = [root + 12] + leftHand[index] + [leftHand[index][1] + 12]
            order = [0, 1, 2, 3, 4, 3, 2, 1]
            velocity = 0.5 if bar < 8 else 0.62
            for step in range(int(beats * 2)):
                midi = notes[order[step % 8]]
                track.place("piano", beat + step * 0.5, pianoNote(midiToFreq(midi), 1.2 * spb, velocity), 0.75)
            if bar >= 8:
                track.place("piano", beat, pianoNote(midiToFreq(root), beats * spb, 0.7), 0.7)
            track.place("bass", beat, bassNote(midiToFreq(root), beats * spb - 0.1, 0.7, 0.25, 0.8, 0.5), 1.0)
        level = [0.5, 0.75, 1.0][bar // 4]
        for midi in padVoicings[index]:
            track.place("pad", beat - 0.2, padNote(track.rng, midiToFreq(midi), beats * spb + 0.2, 0.7, 1.5,
                                                   0.4 if bar < 8 else 0.55), level, 0.0, False)
        if bar >= 8:
            for midi in padVoicings[index][-2:]:
                track.place("pad", beat - 0.2, padNote(track.rng, midiToFreq(midi + 12), beats * spb + 0.2, 0.7,
                                                       1.5, 0.5), 0.6, 0.0, False)

    melody = parseMelody(FINALE_MELODY, 4)
    for beat, beats, midi in melody:
        if beat < 16:
            track.place("piano", beat, pianoNote(midiToFreq(midi), beats * spb + 0.3, 0.6), 1.0)
        else:
            velocity = 0.9 if beat % 2 == 0 else 0.75
            track.place("bells", beat, gamelanBell(track.rng, midiToFreq(midi), beats * spb, velocity, 3.0), velocity, 0.1)
            if beat >= 32:
                track.place("piano", beat, pianoNote(midiToFreq(midi + 12), beats * spb + 0.3, 0.55), 0.55)

    # the build: kompang roll and cymbal swell into bar 9, then half-time drums
    track.place("cymbal", 7 * 4.0, cymbalSwell(track.rng, 4 * spb), 1.0, 0.0, False)
    playKompang(track, kit, "kompang", "b...b...b.b.bbBB", 7 * 4.0, 0.7, -0.3)
    playKompang(track, kit, "kompang", "..p...p...p.pPPP", 7 * 4.0, 0.6, 0.3)
    track.place("cymbal", 32.0, crashCymbal(track.rng, 3.0), 1.4, 0.0, False)
    for bar in range(8, 12):
        start = bar * 4
        track.playPattern("kick", kit["kick"], "X.........X.....", start)
        track.playPattern("drums", kit["snare"], "........X.......", start, 0.9)
        playKompang(track, kit, "kompang", "B.....p.b...p.p.", start, 0.7, -0.35)
        playKompang(track, kit, "kompang", "..p.......p....p", start, 0.5, 0.35)
    playKompang(track, kit, "kompang", "........b.b.bpbp", 11 * 4.0, 0.6, 0.0)
    return finishTrack(track)


# ======================================================================
# 10. AMBIENCE
# ======================================================================
VOWELS = [(730, 1090), (530, 1840), (270, 2290), (570, 840), (300, 870), (660, 1720), (440, 1020)]


def babbleVoice(rng, seconds, pitch, talkiness, lowRate=8000, excited=0.0):
    """One distant talker: a glottal pulse (harmonics of a wandering pitch)
    shaped by two vowel formants that change every syllable, switched on in
    talk spurts. Rendered at 8 kHz (speech murmur needs no more) for speed."""
    count = int(seconds * lowRate)
    envelope = np.zeros(count)
    formant1 = np.full(count, 500.0)
    formant2 = np.full(count, 1500.0)
    clock = -rng.uniform(0.0, 3.0)
    while clock < seconds:
        spurtEnd = clock + rng.uniform(0.8, 3.5) * (0.5 + talkiness)
        while clock < spurtEnd:
            length = rng.uniform(0.09, 0.22) * (1.0 - 0.3 * excited)
            a = int(clock * lowRate)
            b = int((clock + length) * lowRate)
            if b > 0 and a < count and b > a:
                window = np.sin(np.linspace(0.0, math.pi, b - a)) ** 0.6
                a2 = max(a, 0)
                b2 = min(b, count)
                level = rng.uniform(0.45, 1.0)
                envelope[a2:b2] = np.maximum(envelope[a2:b2], level * window[a2 - a:b2 - a])
                vowel = VOWELS[int(rng.integers(len(VOWELS)))]
                scale = rng.uniform(0.9, 1.15)
                formant1[a2:b2] = vowel[0] * scale
                formant2[a2:b2] = vowel[1] * scale
            clock += length * rng.uniform(1.0, 1.35)
        clock += rng.uniform(0.25, 2.0) * (1.5 - talkiness)
    formant1 = smoothSignal(formant1, int(0.035 * lowRate))
    formant2 = smoothSignal(formant2, int(0.035 * lowRate))
    knots = rng.uniform(-1.0, 1.0, int(seconds * 2.5) + 2)
    curve = np.interp(np.linspace(0, len(knots) - 1, count), np.arange(len(knots)), knots)
    curve = smoothSignal(curve, int(0.15 * lowRate))
    f0 = pitch * 2.0 ** (curve * (3.0 + 4.0 * excited) / 12.0)
    phase = 2.0 * math.pi * np.cumsum(f0) / lowRate
    voice = np.zeros(count)
    harmonics = int(0.45 * lowRate / float(np.max(f0)))
    for k in range(1, harmonics + 1):
        fk = k * f0
        gain = 1.0 / (1.0 + ((fk - formant1) / 110.0) ** 2) + 0.55 / (1.0 + ((fk - formant2) / 160.0) ** 2) + 0.04
        voice += gain * k ** -0.5 * np.sin(k * phase)
    return voice * envelope


def crowdBabble(rng, seconds, voices, pitchLow, pitchHigh, talkiness, excited=0.0, panWidth=0.8):
    """Many babbleVoices at random pitches and positions = crowd murmur."""
    lowRate = 8000
    count = samples(seconds)
    total = np.zeros((2, count))
    positions = np.arange(count) * lowRate / SAMPLE_RATE
    for index in range(voices):
        pitch = math.exp(rng.uniform(math.log(pitchLow), math.log(pitchHigh)))
        voice = babbleVoice(rng, seconds, pitch, talkiness, lowRate, excited)
        voice = np.interp(positions, np.arange(len(voice)), voice)
        distance = rng.uniform(0.35, 1.0)
        left, right = panGains(rng.uniform(-panWidth, panWidth))
        total[0] += voice * left * distance
        total[1] += voice * right * distance
    return total


def loopedTexture(stereo, track):
    """Crossfades a texture rendered 3 s longer than the loop into a loop."""
    return crossfadeLoop(stereo, track.loopLength, samples(3.0))


def clinkSound(rng, kind):
    """Modal hits for table sounds (glass, ceramic plate, spoon, metal pan)."""
    if kind == "glass":
        base, ratios, decays = rng.uniform(1800, 2600), [1.0, 2.32, 4.25, 6.63], [0.35, 0.2, 0.1, 0.06]
    elif kind == "ceramic":
        base, ratios, decays = rng.uniform(1100, 1700), [1.0, 2.1, 3.9, 5.3], [0.12, 0.08, 0.05, 0.03]
    elif kind == "spoon":
        base, ratios, decays = rng.uniform(2500, 3500), [1.0, 2.76, 5.4], [0.22, 0.1, 0.05]
    else:
        base, ratios, decays = rng.uniform(350, 600), [1.0, 1.58, 2.35, 3.4, 4.6], [0.5, 0.35, 0.25, 0.18, 0.12]
    count = samples(max(decays) * 5.0)
    t = timeAxis(count)
    out = np.zeros(count)
    for index in range(len(ratios)):
        freq = base * ratios[index] * rng.uniform(0.99, 1.01)
        if freq < 0.45 * SAMPLE_RATE:
            out += np.sin(2 * math.pi * freq * t + rng.uniform(0, 6.28)) * np.exp(-t / decays[index]) / (index + 1)
    out += rng.uniform(-1, 1, count) * np.exp(-t / 0.001) * 0.3
    return normalise(out) * gateEnvelope(count, count / SAMPLE_RATE - 0.05, 0.05, 0.0003)


def cricketChirp(rng, freq):
    """One chirp = 3-4 quick pulses of a pure high tone."""
    pulses = int(rng.integers(3, 5))
    count = samples(0.03 * pulses + 0.02)
    t = timeAxis(count)
    env = np.zeros(count)
    for index in range(pulses):
        a = samples(index * 0.03)
        b = a + samples(0.014)
        env[a:b] = np.sin(np.linspace(0, math.pi, b - a))
    return np.sin(2 * math.pi * freq * t) * env


def addCrickets(track, busName, count, level, freqLow=4200.0, freqHigh=5200.0):
    """Crickets chirp regularly; the period is rounded so a whole number of
    chirps fits in the loop - the rhythm continues across the loop point."""
    for index in range(count):
        freq = track.rng.uniform(freqLow, freqHigh)
        period = track.rng.uniform(0.55, 0.9)
        period = track.loopSeconds / round(track.loopSeconds / period)
        pan = track.rng.uniform(-0.8, 0.8)
        gain = level * track.rng.uniform(0.5, 1.0)
        offset = track.rng.uniform(0, period)
        chirps = int(round(track.loopSeconds / period))
        for chirp in range(chirps):
            if track.rng.random() < 0.12:
                continue
            seconds = offset + chirp * period + track.rng.normal(0, 0.01)
            track.placeAtSeconds(busName, seconds, cricketChirp(track.rng, freq), gain * track.rng.uniform(0.8, 1.0), pan)


def addClinks(track, busName, rate, kinds, level):
    for index in range(int(track.loopSeconds * rate)):
        kind = kinds[int(track.rng.integers(len(kinds)))]
        seconds = track.rng.uniform(0, track.loopSeconds)
        gain = level * track.rng.uniform(0.3, 1.0)
        pan = track.rng.uniform(-0.8, 0.8)
        track.placeAtSeconds(busName, seconds, clinkSound(track.rng, kind), gain, pan)
        if track.rng.random() < 0.3:     # a double clink (a glass touching a plate twice)
            track.placeAtSeconds(busName, seconds + track.rng.uniform(0.08, 0.2), clinkSound(track.rng, kind),
                                 gain * 0.6, pan)


def motorbikePass(rng, seconds=6.0):
    """A small two-stroke motorbike ("kapcai") passing in the distance: a
    saw-wave engine whose pitch drops as it passes (Doppler effect) while
    its loudness rises and falls and it pans from left to right."""
    count = samples(seconds)
    t = timeAxis(count)
    centre = seconds / 2.0
    freq = 105.0 * (1.0 - 0.06 * np.tanh((t - centre) / 0.8)) * (1.0 + 0.03 * np.sin(2 * math.pi * 0.7 * t))
    engine = tableOscillator("saw", freq, count, 0.0, 30)
    engine = filterSound(engine, [("lowpass", 1500), ("peak", 600, 1.0, 4.0)])
    loudness = 1.0 / (1.0 + ((t - centre) / 1.2) ** 2)
    pan = np.clip((t - centre) / centre, -0.8, 0.8)
    angle = (pan + 1.0) * math.pi / 4.0
    body = engine * loudness * gateEnvelope(count, seconds - 0.3, 0.3, 0.3)
    return np.vstack([body * np.cos(angle), body * np.sin(angle)]) * math.sqrt(2.0)


def addTexture(track, busName, stereo, gain=1.0):
    track.placeAtSeconds(busName, 0.0, stereo, gain, 0.0)


def ambienceTrack(name, seed, reverbSize, reverbDamping):
    """An ambience uses the same Track engine: 60 BPM, 6 bars of 4 = 24 s,
    so one "beat" is exactly one second."""
    return Track(name, 60, 4, 6, seed, reverbSize=reverbSize, reverbDamping=reverbDamping,
                 delayBeats=0.37, delayFeedback=0.2)


def finishAmbience(track):
    stereo = track.renderLayer("base")
    return {track.name: masterLoop(stereo, AMBIENCE_RMS_DB, AMBIENCE_CEILING_DB, compress=True)}


def makeNightMarket():
    track = ambienceTrack("amb_night_market", 8101, 0.6, 0.5)
    seconds = track.loopSeconds + 3.0
    track.addBus("crowd", level=-20.0, reverb=0.25, filters=[("highpass", 120), ("lowpass", 2300)], width=1.2)
    track.addBus("traffic", level=-27.0, reverb=0.0)
    track.addBus("bike", level=-30.0, reverb=0.3)
    track.addBus("crickets", level=-38.0, reverb=0.25)
    track.addBus("clinks", level=-32.0, reverb=0.4, filters=[("lowpass", 7000)])
    addTexture(track, "crowd", loopedTexture(normalise(crowdBabble(track.rng, seconds, 18, 95, 250, 0.55)), track))
    rumble = filterLoop(track.rng.normal(size=(2, track.loopLength)), [("lowpass", 180), ("lowpass", 300), ("highpass", 30)])
    swell = loopSmoothCurve(track.rng, track.loopLength, 8, 0.5, 1.0)
    addTexture(track, "traffic", normalise(rumble) * swell)
    track.placeAtSeconds("bike", 9.0, motorbikePass(track.rng, 6.0), 1.0, 0.0)
    addCrickets(track, "crickets", 3, 1.0)
    addClinks(track, "clinks", 0.5, ["glass", "ceramic", "spoon"], 1.0)
    return finishAmbience(track)


def makeKitchen():
    track = ambienceTrack("amb_kitchen", 8202, 0.45, 0.5)
    count = track.loopLength
    seconds = track.loopSeconds
    t = timeAxis(count)
    track.addBus("fan", level=-21.0, reverb=0.05, width=0.6)
    track.addBus("clatter", level=-24.0, reverb=0.45, filters=[("lowpass", 6000)])
    track.addBus("voices", level=-27.0, reverb=0.4, filters=[("highpass", 150), ("lowpass", 1800)])
    # exhaust fan: 100 Hz motor hum (Malaysian mains = 50 Hz) + blade-pass tone + airflow
    motor = 0.5 * np.sin(2 * math.pi * loopFrequency(100.0, seconds) * t) + \
        0.2 * np.sin(2 * math.pi * loopFrequency(200.0, seconds) * t) + \
        0.08 * np.sin(2 * math.pi * loopFrequency(300.0, seconds) * t)
    blade = loopFrequency(7.5, seconds)
    bladeTone = 0.12 * np.sin(2 * math.pi * loopFrequency(45.0, seconds) * t)
    airflow = filterLoop(track.rng.normal(size=(2, count)), [("lowpass", 1200), ("highpass", 60), ("peak", 220, 1.0, 5.0)])
    airflow = normalise(airflow) * (1.0 + 0.25 * np.sin(2 * math.pi * blade * t))
    fan = airflow * 0.8 + (motor * 0.12 + bladeTone * 0.2) * np.ones((2, 1))
    addTexture(track, "fan", fan)
    addClinks(track, "clatter", 0.45, ["pan", "ceramic", "spoon", "pan"], 1.0)
    # a distant cleaver: two bursts of steady chopping
    for burstStart in (4.0, 15.5):
        for chop in range(int(track.rng.integers(6, 10))):
            size = samples(0.12)
            tt = timeAxis(size)
            thud = np.sin(2 * math.pi * 180 * tt) * np.exp(-tt / 0.03) + \
                filterSound(track.rng.uniform(-1, 1, size), [("bandpass", 1200, 1.0)]) * np.exp(-tt / 0.01)
            track.placeAtSeconds("clatter", burstStart + chop * 0.21 + track.rng.normal(0, 0.01), thud,
                                 0.5 * track.rng.uniform(0.7, 1.0), -0.5)
    addTexture(track, "voices", loopedTexture(normalise(crowdBabble(track.rng, seconds + 3.0, 3, 100, 180, 0.4)), track))
    return finishAmbience(track)


def makeMamak():
    track = ambienceTrack("amb_mamak", 8303, 0.5, 0.55)
    count = track.loopLength
    seconds = track.loopSeconds
    t = timeAxis(count)
    track.addBus("crowd", level=-20.0, reverb=0.22, filters=[("highpass", 130), ("lowpass", 3000)], width=1.3)
    track.addBus("tv", level=-28.0, reverb=0.3, filters=[("highpass", 450), ("lowpass", 3000), ("peak", 1800, 1.0, 5.0)])
    track.addBus("clinks", level=-30.0, reverb=0.3, filters=[("lowpass", 8000)])
    track.addBus("fan", level=-33.0, reverb=0.0, filters=[("lowpass", 700)])
    addTexture(track, "crowd", loopedTexture(normalise(crowdBabble(track.rng, seconds + 3.0, 22, 100, 260, 0.7)), track))
    # a TV on the wall: one excited commentator through a small speaker + crowd roar swells
    commentator = crowdBabble(track.rng, seconds + 3.0, 1, 150, 160, 1.0, 1.0, 0.0)
    commentator = np.tanh(2.0 * normalise(commentator)) * 0.8
    roar = filterSound(track.rng.normal(size=(2, samples(seconds + 3.0))), [("bandpass", 900, 0.7)])
    roarEnv = np.zeros(samples(seconds + 3.0))
    for centre in (6.0, 17.0):
        roarEnv += np.exp(-((timeAxis(len(roarEnv)) - centre) / 1.3) ** 2)
    tv = loopedTexture(commentator + 0.4 * normalise(roar) * roarEnv, track)
    track.placeAtSeconds("tv", 0.0, tv, 1.0, -0.45)
    # teh tarik glasses being stirred: quick bursts of spoon-on-glass clinks
    for burst in range(7):
        start = track.rng.uniform(0, seconds)
        pan = track.rng.uniform(-0.7, 0.7)
        glassBase = track.rng.uniform(0.95, 1.05)
        for step in range(int(track.rng.integers(4, 9))):
            clink = clinkSound(track.rng, "glass")
            track.placeAtSeconds("clinks", start + step * track.rng.uniform(0.12, 0.18), clink,
                                 0.5 * glassBase * track.rng.uniform(0.5, 1.0), pan)
    addClinks(track, "clinks", 0.35, ["ceramic", "spoon", "glass"], 0.9)
    # ceiling fan: slow whoosh, locked to the loop length
    whoosh = filterLoop(track.rng.normal(size=(2, count)), [("lowpass", 500), ("highpass", 80)])
    whoosh = normalise(whoosh) * (0.7 + 0.3 * np.sin(2 * math.pi * loopFrequency(1.6, seconds) * t))
    addTexture(track, "fan", whoosh)
    return finishAmbience(track)


def makeHall():
    track = ambienceTrack("amb_hall", 8404, 0.95, 0.35)
    count = track.loopLength
    seconds = track.loopSeconds
    track.addBus("crowd", level=-20.0, reverb=0.9, filters=[("highpass", 150), ("lowpass", 2400)], width=1.4)
    track.addBus("cutlery", level=-31.0, reverb=0.8, filters=[("lowpass", 7000)])
    track.addBus("room", level=-30.0, reverb=0.3)
    addTexture(track, "crowd", loopedTexture(normalise(crowdBabble(track.rng, seconds + 3.0, 16, 100, 240, 0.5)), track))
    addClinks(track, "cutlery", 0.7, ["spoon", "ceramic", "glass"], 1.0)
    room = filterLoop(track.rng.normal(size=(2, count)), [("lowpass", 250), ("highpass", 40), ("peak", 450, 1.5, 4.0)])
    addTexture(track, "room", normalise(room))
    return finishAmbience(track)


def makeRooftop():
    track = ambienceTrack("amb_rooftop", 8505, 0.7, 0.5)
    count = track.loopLength
    track.addBus("wind", level=-20.0, reverb=0.05)
    track.addBus("city", level=-27.0, reverb=0.1)
    track.addBus("crickets", level=-33.0, reverb=0.3)
    track.addBus("horn", level=-31.0, reverb=0.6, filters=[("lowpass", 1500)])
    # wind: a bank of band-pass filters over noise. A slow "master gust"
    # curve swells the whole wind; smaller curves move energy between the
    # bands; two very narrow resonances whistle through the railings.
    wind = np.zeros((2, count))
    masterGust = loopSmoothCurve(track.rng, count, 7, 0.0, 1.0)
    masterGust = 0.15 + 0.85 * (masterGust - masterGust.min()) / (masterGust.max() - masterGust.min())
    for channel in range(2):
        noise = track.rng.normal(size=count)
        for centre, q, level in ((150, 0.8, 1.0), (320, 1.2, 0.7), (640, 1.6, 0.4), (1200, 2.0, 0.2), (2200, 2.5, 0.07)):
            band = filterLoop(noise, [("bandpass", centre, q)])
            wind[channel] += band * level * loopSmoothCurve(track.rng, count, 10, 0.3, 1.0)
        for centre in (760.0, 1130.0):
            band = filterLoop(noise, [("bandpass", centre * track.rng.uniform(0.97, 1.03), 12.0)])
            whistle = np.clip(loopSmoothCurve(track.rng, count, 9, -0.6, 1.0), 0.0, None) ** 2
            wind[channel] += normalise(band) * 0.12 * whistle
        wind[channel] *= masterGust ** 1.5
    addTexture(track, "wind", normalise(wind))
    city = filterLoop(track.rng.normal(size=(2, count)), [("lowpass", 200), ("lowpass", 350), ("highpass", 30)])
    hiss = filterLoop(track.rng.normal(size=(2, count)), [("bandpass", 900, 0.5)])
    addTexture(track, "city", normalise(city) + 0.15 * normalise(hiss))
    addCrickets(track, "crickets", 2, 1.0, 4000.0, 4800.0)
    # one distant two-tone car horn far below
    size = samples(0.7)
    horn = (tableOscillator("saw", 420.0, size, 0.0, 12) + tableOscillator("saw", 525.0, size, 0.3, 12))
    horn *= adsrEnvelope(size, 0.02, 0.2, 0.9, 0.55, 0.12)
    track.placeAtSeconds("horn", 13.0, horn, 1.0, 0.4)
    return finishAmbience(track)


# ======================================================================
# 11. OUTPUT AND CHECKS
# ======================================================================
def writeWav(path, stereo, seed):
    """16-bit stereo PCM with a tiny TPDF dither (avoids quantisation
    distortion on quiet reverb tails)."""
    rng = np.random.default_rng(seed)
    dither = (rng.random(stereo.shape) - rng.random(stereo.shape)) / 32768.0
    data = np.clip((stereo + dither) * 32767.0, -32768, 32767).astype("<i2")
    wf = wave.open(path, "wb")
    wf.setnchannels(2)
    wf.setsampwidth(2)
    wf.setframerate(SAMPLE_RATE)
    wf.writeframes(data.T.tobytes())
    wf.close()


def readWav(path):
    wf = wave.open(path, "rb")
    channels = wf.getnchannels()
    rate = wf.getframerate()
    frames = wf.readframes(wf.getnframes())
    wf.close()
    data = np.frombuffer(frames, dtype="<i2").astype(float) / 32768.0
    return data.reshape(-1, channels).T, rate


def seamRatio(stereo):
    """How big is the jump from the last sample to the first, compared with
    normal sample-to-sample steps (99.9th percentile)? Below 1 = inaudible."""
    steps = np.abs(np.diff(stereo, axis=1))
    typical = float(np.percentile(steps, 99.9)) + 1e-9
    jump = float(np.max(np.abs(stereo[:, 0] - stereo[:, -1])))
    return jump / typical


def checkFile(path, name):
    stereo, rate = readWav(path)
    seconds = stereo.shape[1] / rate
    size = os.path.getsize(path) / (1024 * 1024)
    line = (f"  {name:22s} {seconds:6.2f} s  {stereo.shape[0]} ch  {rate} Hz  {size:5.2f} MB  "
            f"peak {peakDb(stereo):6.2f} dBFS  RMS {rmsDb(stereo):6.2f} dBFS  seam {seamRatio(stereo):4.2f}")
    print(line)
    if name in LOOP_INFO:
        bars = LOOP_INFO[name][0]
        per = stereo.shape[1] // bars
        levels = [rmsDb(stereo[:, index * per:(index + 1) * per]) for index in range(bars)]
        print("      bar RMS: " + " ".join(f"{level:5.1f}" for level in levels))
    return stereo


JOBS = [
    (["music_title"], makeTitle),
    (["music_street"], makeStreet),
    (["music_rojak", "music_rojak_aura"], makeRojak),
    (["music_wok", "music_wok_aura"], makeWok),
    (["music_tehtarik", "music_tehtarik_aura"], makeTehTarik),
    (["music_banquet", "music_banquet_aura"], makeBanquet),
    (["music_finale"], makeFinale),
    (["amb_night_market"], makeNightMarket),
    (["amb_kitchen"], makeKitchen),
    (["amb_mamak"], makeMamak),
    (["amb_hall"], makeHall),
    (["amb_rooftop"], makeRooftop),
]


def main():
    global SHOW_BUSES
    startTime = time.time()
    wanted = []
    for argument in sys.argv[1:]:
        if argument == "--buses":
            SHOW_BUSES = True
        elif not argument.startswith("--"):
            wanted.append(argument)
    os.makedirs(MUSIC_DIR, exist_ok=True)
    os.makedirs(AMBIENCE_DIR, exist_ok=True)
    print("Midnight Makan - generating music and ambience loops")
    for names, maker in JOBS:
        if wanted and not any(name in wanted for name in names):
            continue
        jobStart = time.time()
        loops = maker()
        rendered = {}
        for name in names:
            folder = MUSIC_DIR if name.startswith("music") else AMBIENCE_DIR
            path = os.path.join(folder, name + ".wav")
            writeWav(path, loops[name], sum(ord(letter) for letter in name))
            rendered[name] = checkFile(path, name)
        if len(names) == 2:
            both = rendered[names[0]] + rendered[names[1]]
            print(f"      base + aura together: peak {peakDb(both):6.2f} dBFS  RMS {rmsDb(both):6.2f} dBFS")
        print(f"      ({time.time() - jobStart:.1f} s)")
    print(f"Done in {time.time() - startTime:.1f} s")


if __name__ == "__main__":
    main()
