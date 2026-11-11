"""
make_sfx.py
-----------
Generates every sound effect (assets/sounds/sfx) and every voice clip
(assets/sounds/voice) for Midnight Makan. No sound was downloaded or
recorded: each one is built from maths with NumPy and saved with Python's
"wave" module (the same module used in the Week 2 "Sound Recording" tutorial).

How a sound is built (the same layering idea as a visual effect):
  1. Sources   - sine waves, band-limited saw/pulse waves (additive synthesis),
                 FM synthesis for bells, modal synthesis (sums of decaying
                 sines) for wood, metal, glass and ceramic, Karplus-Strong for
                 plucked strings, white/brown noise for air, fire and friction.
  2. Shaping   - envelopes (ADSR / exponential decay) control loudness over
                 time; pitch envelopes (sweeps) make whooshes, plops and meows.
  3. Filters   - one-pole low/high-pass, biquad filters (difference equations
                 from the RBJ "Audio EQ Cookbook") and a state-variable filter
                 whose cutoff can move while it plays (sweeps and formants).
  4. Space     - a small Schroeder/Freeverb-style reverb (comb + all-pass).
  5. Finishing - DC offset removal, silence trimming, short fades (no clicks),
                 peak normalisation to -3 dBFS, 16-bit PCM with TPDF dither.

Digital artefacts I avoid on purpose (Week 3-4 "Digital Artefact"):
  * aliasing   -> oscillators only add harmonics below 18 kHz,
  * clicks     -> every one-shot fades in/out; loops crossfade end into start,
  * clipping   -> everything is peak normalised to -3 dBFS,
  * DC offset  -> a 20 Hz high-pass removes it,
  * quantisation distortion -> tiny TPDF dither before rounding to 16-bit.

Voice lines use the Windows text-to-speech engine (System.Speech, called
through PowerShell) and are then processed with NumPy: high-pass, pause
shortening, trimming, resampling for character, EQ, compression and a short
room reverb. The raw TTS files stay outside the assets folder.

Run from the project folder:
    python tools/make_sfx.py              (everything, about one minute)
    python tools/make_sfx.py chop_1 nom   (only the named sounds)
    python tools/make_sfx.py --no-tts     (skip the voice lines)
"""
import math
import os
import subprocess
import sys
import tempfile
import time
import wave
import zlib

import numpy as np

SAMPLE_RATE = 44100      # samples per second (same rate as the Week 2 tutorial)
PEAK_DB = -3.0           # every file is normalised so its loudest sample is -3 dBFS

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(TOOLS_DIR)
SFX_DIR = os.path.join(PROJECT_DIR, "assets", "sounds", "sfx")
VOICE_DIR = os.path.join(PROJECT_DIR, "assets", "sounds", "voice")

# Raw text-to-speech renders are kept OUT of the game assets.
BUILD_DIR = os.path.join(os.path.dirname(PROJECT_DIR), "_build")
if os.path.isdir(BUILD_DIR):
    TTS_RAW_DIR = os.path.join(BUILD_DIR, "tts_raw")
else:
    TTS_RAW_DIR = os.path.join(tempfile.gettempdir(), "midnight_makan_tts_raw")


# ======================================================================
# 1. BASIC HELPERS (time, length, decibels)
# ======================================================================
def samples(seconds):
    """Number of samples in 'seconds' of audio."""
    return max(1, int(round(seconds * SAMPLE_RATE)))


def timeline(seconds):
    """Time in seconds of every sample: 0, 1/44100, 2/44100, ..."""
    return np.arange(samples(seconds)) / SAMPLE_RATE


def dbToGain(db):
    """Decibels -> linear gain (-6 dB is about half the amplitude)."""
    return 10.0 ** (db / 20.0)


def gainToDb(gain):
    return 20.0 * math.log10(max(gain, 1e-12))


def fitLength(values, count):
    """Returns exactly 'count' samples. A single number becomes a constant array."""
    if np.isscalar(values):
        return np.full(count, float(values))
    values = np.asarray(values, dtype=float)
    if len(values) >= count:
        return values[:count]
    padding = np.full(count - len(values), values[-1])
    return np.concatenate([values, padding])


def normalise(signal, peak=1.0):
    """Scales a signal so its loudest sample equals 'peak'."""
    loudest = np.max(np.abs(signal))
    if loudest < 1e-12:
        return signal
    return signal * (peak / loudest)


def silence(seconds):
    return np.zeros(samples(seconds))


def addAt(mix, sound, startSeconds, gain=1.0):
    """Mixes 'sound' into 'mix' starting at 'startSeconds' (layering).
    The last 4 ms of every layer is faded out, so a layer that gets cut
    short can never leave a click behind."""
    start = int(round(startSeconds * SAMPLE_RATE))
    if start >= len(mix):
        return
    end = min(len(mix), start + len(sound))
    piece = gain * sound[:end - start]
    tail = min(len(piece), samples(0.004))
    if tail > 1:
        piece[-tail:] *= np.linspace(1.0, 0.0, tail)
    mix[start:end] += piece


def resample(signal, speed):
    """Plays a signal faster (speed > 1, higher pitch) or slower (speed < 1).
    Linear interpolation between samples - like changing tape speed, so the
    pitch AND the formants (voice character) move together."""
    positions = np.arange(0, len(signal) - 1, speed)
    return np.interp(positions, np.arange(len(signal)), signal)


def softClip(signal, drive=1.5):
    """Gentle tape-style saturation (tanh). Rounds off peaks and adds warmth."""
    return np.tanh(drive * signal) / math.tanh(drive)


# ======================================================================
# 2. ENVELOPES AND CONTROL CURVES
# ======================================================================
def curve(seconds, times, values):
    """Breakpoint envelope: straight lines between (time, value) points."""
    return np.interp(timeline(seconds), times, values)


def expCurve(seconds, start, end):
    """Exponential glide from 'start' to 'end' (sounds natural for pitch)."""
    fraction = timeline(seconds) / seconds
    return start * (end / start) ** fraction


def decay(seconds, decayTime):
    """Exponential decay envelope exp(-t / decayTime) - how struck things fade."""
    return np.exp(-timeline(seconds) / decayTime)


def hitEnvelope(seconds, attack, decayTime):
    """Very short linear attack (avoids a click) followed by an exponential decay."""
    t = timeline(seconds)
    rise = np.clip(t / attack, 0.0, 1.0)
    return rise * np.exp(-np.maximum(t - attack, 0.0) / decayTime)


def adsr(seconds, attack, decayTime, sustain, release):
    """Attack-Decay-Sustain-Release envelope made of straight lines."""
    releaseStart = max(attack + decayTime, seconds - release)
    releaseStart = min(releaseStart, seconds * 0.999)
    decayEnd = min(attack + decayTime, releaseStart)
    times = [0.0, attack, decayEnd, releaseStart, seconds]
    levels = [0.0, 1.0, sustain, sustain, 0.0]
    return curve(seconds, times, levels)


def smoothRandom(rng, seconds, rate, periodic=False):
    """A slowly wandering random value between -1 and 1 ('rate' new targets
    per second, joined with cosine interpolation). Used for flicker,
    turbulence and wobble. periodic=True makes the curve end where it
    started, so it can loop."""
    count = samples(seconds)
    points = max(2, int(round(seconds * rate)))
    targets = rng.uniform(-1.0, 1.0, points + 1)
    if periodic:
        targets[-1] = targets[0]
    position = np.arange(count) * points / count
    index = np.floor(position).astype(int)
    fraction = position - index
    blend = (1.0 - np.cos(np.pi * fraction)) / 2.0      # cosine interpolation
    return targets[index] * (1.0 - blend) + targets[index + 1] * blend


def fadeEdges(signal, fadeIn=0.002, fadeOut=0.02):
    """Raised-cosine fade in/out so the file starts and ends at zero (no click)."""
    out = signal.copy()
    inCount = min(len(out), samples(fadeIn))
    outCount = min(len(out), samples(fadeOut))
    if inCount > 1:
        out[:inCount] *= 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, inCount))
    if outCount > 1:
        out[-outCount:] *= 0.5 + 0.5 * np.cos(np.linspace(0.0, np.pi, outCount))
    return out


# ======================================================================
# 3. SOUND SOURCES (oscillators and noise)
# ======================================================================
def phaseOf(freq, count):
    """Phase accumulation: add 2*pi*f/SR every sample. The frequency can then
    change smoothly (sweeps, vibrato) without the wave jumping."""
    return 2.0 * np.pi * np.cumsum(fitLength(freq, count)) / SAMPLE_RATE


def sine(freq, seconds):
    """Sine wave. 'freq' can be a number or an array (a pitch curve)."""
    return np.sin(phaseOf(freq, samples(seconds)))


def richWave(freq, seconds, tilt=1.0, oddOnly=False, maxHarmonics=60):
    """Band-limited saw (tilt=1) or softer pulse/square (oddOnly) built by
    ADDITIVE SYNTHESIS: harmonic k has amplitude 1/k^tilt. Harmonics are faded
    out between 15 and 18 kHz so nothing folds back above Nyquist (aliasing)."""
    count = samples(seconds)
    freqs = fitLength(freq, count)
    phase = phaseOf(freqs, count)
    out = np.zeros(count)
    for k in range(1, maxHarmonics + 1):
        if oddOnly and k % 2 == 0:
            continue
        weight = np.clip((18000.0 - k * freqs) / 3000.0, 0.0, 1.0)
        if weight.max() <= 0.0:
            break
        out += weight * np.sin(k * phase) / (k ** tilt)
    return normalise(out)


def fmTone(freq, seconds, ratio, indexStart, indexEnd, indexTime):
    """FM SYNTHESIS (Chowning): a modulator at freq*ratio wobbles the phase of
    the carrier. A big modulation index = bright; as the index decays the
    tone gets purer - exactly how a struck bell behaves."""
    t = timeline(seconds)
    index = indexEnd + (indexStart - indexEnd) * np.exp(-t / indexTime)
    modulator = np.sin(2.0 * np.pi * freq * ratio * t)
    return np.sin(2.0 * np.pi * freq * t + index * modulator)


def modes(seconds, freqs, decays, amps):
    """MODAL SYNTHESIS: a struck object rings at several resonant frequencies
    (its modes); each one is a sine that dies away at its own speed."""
    t = timeline(seconds)
    out = np.zeros(len(t))
    for i in range(len(freqs)):
        if freqs[i] < 18000:
            out += amps[i] * np.sin(2.0 * np.pi * freqs[i] * t) * np.exp(-t / decays[i])
    # damp the last 15 % smoothly so the ring never stops with a click
    return fadeEdges(out, 0.0, seconds * 0.15)


def whiteNoise(rng, seconds):
    """Every sample random: sounds like 'shhh'."""
    return rng.uniform(-1.0, 1.0, samples(seconds))


def brownNoise(rng, seconds):
    """Deep rumbling noise: white noise through a very low one-pole low-pass."""
    return normalise(onePoleLowPass(whiteNoise(rng, seconds), 60.0))


def bubble(freq, seconds, rise=1.6, decayFraction=0.3):
    """Water droplet / bubble 'blip': a sine whose pitch glides UP while it
    fades (the Minnaert resonance of a small air bubble)."""
    t = timeline(seconds)
    pitch = freq * (1.0 + (rise - 1.0) * t / seconds)
    blip = sine(pitch, seconds) * hitEnvelope(seconds, 0.0006, seconds * decayFraction)
    return fadeEdges(blip, 0.0, seconds * 0.2)


def karplusStrong(rng, freq, seconds, brightness=0.5, damping=0.996):
    """KARPLUS-STRONG plucked string: a short noise burst circulates in a
    delay line; each pass averages two neighbouring samples (a gentle
    low-pass), so the tone gets softer as it decays, like a real string.
    Integer delays are slightly out of tune, so the result is resampled
    to the exact pitch."""
    delay = max(2, int(SAMPLE_RATE / freq - 0.5))
    realFreq = SAMPLE_RATE / (delay + 0.5)
    total = int(samples(seconds) * freq / realFreq) + delay + 4
    line = np.zeros(total)
    burst = whiteNoise(rng, (delay + 1) / SAMPLE_RATE)[:delay + 1]
    burst = onePoleLowPass(burst, 800.0 + 9000.0 * brightness)
    line[:delay + 1] = normalise(burst)
    # y[n] = damping * (y[n-D] + y[n-D-1]) / 2, computed one block of D samples at a time
    for start in range(delay + 1, total, delay):
        end = min(start + delay, total)
        line[start:end] = damping * 0.5 * (line[start - delay:end - delay] +
                                           line[start - delay - 1:end - delay - 1])
    exact = resample(line, freq / realFreq)
    return fitLength(exact, samples(seconds))


# ======================================================================
# 4. FILTERS (difference equations)
# ======================================================================
def onePoleLowPass(signal, cutoff):
    """One-pole low-pass: y[n] = y[n-1] + a * (x[n] - y[n-1]).
    Smooths the signal and removes high frequencies (6 dB per octave)."""
    a = 1.0 - math.exp(-2.0 * math.pi * cutoff / SAMPLE_RATE)
    values = np.asarray(signal, dtype=float).tolist()
    y = 0.0
    for i in range(len(values)):
        y += a * (values[i] - y)
        values[i] = y
    return np.array(values)


def onePoleHighPass(signal, cutoff):
    """High-pass = original minus its low-passed copy (keeps the fast changes)."""
    return signal - onePoleLowPass(signal, cutoff)


def biquadCoefficients(kind, freq, q, gainDb):
    """Biquad coefficients from the RBJ Audio EQ Cookbook."""
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
    elif kind == "bandpass":           # 0 dB at the centre frequency
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
    else:                              # "highshelf"
        root = 2 * math.sqrt(amp) * alpha
        b = [amp * ((amp + 1) + (amp - 1) * cosW + root),
             -2 * amp * ((amp - 1) + (amp + 1) * cosW),
             amp * ((amp + 1) + (amp - 1) * cosW - root)]
        a = [(amp + 1) - (amp - 1) * cosW + root,
             2 * ((amp - 1) - (amp + 1) * cosW),
             (amp + 1) - (amp - 1) * cosW - root]
    return b[0] / a[0], b[1] / a[0], b[2] / a[0], a[1] / a[0], a[2] / a[0]


def biquad(signal, kind, freq, q=0.707, gainDb=0.0):
    """Biquad filter, Direct Form I difference equation:
    y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]"""
    b0, b1, b2, a1, a2 = biquadCoefficients(kind, freq, q, gainDb)
    values = np.asarray(signal, dtype=float).tolist()
    x1 = x2 = y1 = y2 = 0.0
    for i in range(len(values)):
        x0 = values[i]
        y0 = b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
        x2 = x1
        x1 = x0
        y2 = y1
        y1 = y0
        values[i] = y0
    return np.array(values)


def lowPass(signal, cutoff, q=0.707):
    return biquad(signal, "lowpass", cutoff, q)


def highPass(signal, cutoff, q=0.707):
    return biquad(signal, "highpass", cutoff, q)


def bandPass(signal, centre, q=1.0):
    return biquad(signal, "bandpass", centre, q)


def bandNoise(rng, seconds, centre, q=1.0):
    """White noise through a band-pass filter: a 'coloured' hiss."""
    return normalise(bandPass(whiteNoise(rng, seconds), centre, q))


def sweepFilter(signal, cutoff, q=0.7, mode="band"):
    """STATE-VARIABLE FILTER (trapezoidal / 'TPT' form) whose cutoff can change
    every sample - used for whooshes, flames and moving vowel formants.
    mode = "low", "band" (normalised to 0 dB peak) or "high"."""
    count = len(signal)
    cut = np.clip(fitLength(cutoff, count), 20.0, SAMPLE_RATE * 0.45)
    g = np.tan(np.pi * cut / SAMPLE_RATE)
    k = 1.0 / q
    a1 = 1.0 / (1.0 + g * (g + k))
    a2 = (g * a1).tolist()
    a3 = (g * g * a1).tolist()
    a1 = a1.tolist()
    values = np.asarray(signal, dtype=float).tolist()
    band = [0.0] * count
    low = [0.0] * count
    s1 = s2 = 0.0
    for i in range(count):
        v3 = values[i] - s2
        v1 = a1[i] * s1 + a2[i] * v3
        v2 = s2 + a2[i] * s1 + a3[i] * v3
        s1 = 2.0 * v1 - s1
        s2 = 2.0 * v2 - s2
        band[i] = v1
        low[i] = v2
    band = np.array(band)
    low = np.array(low)
    if mode == "low":
        return low
    if mode == "high":
        return np.asarray(signal, dtype=float) - k * band - low
    return k * band


def sampleAndHold(signal, newRate):
    """Lowers the sample rate by repeating samples (a deliberate 'digital' artefact)."""
    step = max(1, int(round(SAMPLE_RATE / newRate)))
    index = (np.arange(len(signal)) // step) * step
    return signal[index]


# ======================================================================
# 5. REVERB (comb + all-pass filters)
# ======================================================================
def combFilter(signal, delay, feedback):
    """Feedback comb filter y[n] = x[n] + g*y[n-D]: a train of echoes.
    Computed one block of D samples at a time (each block only needs the
    previous block), which keeps it fast in NumPy."""
    out = signal.copy()
    for start in range(delay, len(out), delay):
        end = min(start + delay, len(out))
        out[start:end] += feedback * out[start - delay:end - delay]
    return out


def allPassFilter(signal, delay, gain):
    """Schroeder all-pass y[n] = -g*x[n] + x[n-D] + g*y[n-D]: smears the echoes
    into a dense tail without changing the tone colour."""
    out = -gain * signal
    out[delay:] += signal[:-delay]
    for start in range(delay, len(out), delay):
        end = min(start + delay, len(out))
        out[start:end] += gain * out[start - delay:end - delay]
    return out


def reverb(signal, roomSize=0.5, wet=0.25, tail=1.0, brightness=6000.0):
    """Schroeder/Freeverb-style room: 8 parallel combs -> 4 all-passes.
    The reflections also get darker over time (a moving low-pass), because
    real walls and air absorb high frequencies first."""
    padded = np.concatenate([signal, np.zeros(samples(tail))])
    scale = 0.5 + roomSize
    feedback = 0.70 + 0.26 * roomSize
    wetSignal = np.zeros(len(padded))
    for delay in (1116, 1188, 1277, 1356, 1422, 1491, 1557, 1617):
        wetSignal += combFilter(padded, int(delay * scale), feedback)
    wetSignal /= 8.0
    for delay in (556, 441, 341, 225):
        wetSignal = allPassFilter(wetSignal, int(delay * scale), 0.5)
    t = np.arange(len(padded)) / SAMPLE_RATE
    darkening = brightness * (0.3 + 0.7 * np.exp(-t / (0.25 + roomSize)))
    wetSignal = sweepFilter(wetSignal, darkening, 0.6, "low")
    wetSignal = onePoleHighPass(wetSignal, 90.0)
    return padded + wet * wetSignal


# ======================================================================
# 6. EVENT HELPERS (random crackles, grains)
# ======================================================================
def randomTimes(rng, seconds, rate):
    """Random event times (a Poisson process): the gaps between events follow
    an exponential distribution, which sounds natural for crackles and drops."""
    times = []
    t = rng.exponential(1.0 / rate)
    while t < seconds:
        times.append(t)
        t += rng.exponential(1.0 / rate)
    return times


def crackleTrain(rng, seconds, rate, bands, density=None):
    """Random clicks. Each click is a single-sample impulse that 'rings'
    through a resonant band-pass filter and becomes a tiny crackle.
    'density' (0..1 per sample) thins the clicks out over time."""
    count = samples(seconds)
    out = np.zeros(count)
    for centre, q in bands:
        impulses = np.zeros(count)
        for t in randomTimes(rng, seconds, rate / len(bands)):
            index = min(count - 1, int(t * SAMPLE_RATE))
            if density is not None and rng.random() > density[index]:
                continue
            # mostly small clicks with the occasional loud pop
            impulses[index] += rng.choice([-1.0, 1.0]) * rng.random() ** 2.5
        out += bandPass(impulses, centre, q)
    return normalise(out)


def sparkleGrains(rng, seconds, count, lowFreq, highFreq, start=0.0, spread=0.5):
    """Glitter layer: tiny high sine pings scattered in time (granular idea)."""
    mix = silence(seconds)
    for i in range(count):
        when = start + spread * rng.random() ** 1.4
        freq = rng.uniform(lowFreq, highFreq)
        length = rng.uniform(0.03, 0.09)
        ping = sine(freq, length) * hitEnvelope(length, 0.002, length / 4.0)
        loudness = rng.uniform(0.25, 1.0) * (1.0 - 0.7 * (when - start) / max(spread, 1e-3))
        addAt(mix, ping, when, loudness)
    return mix


def seamlessLoop(signal, loopSeconds, fadeSeconds):
    """Makes a loop with no click at the seam: the extra audio after the loop
    end is crossfaded (equal-power) into the loop start, so the last sample
    flows straight into the first one when the file repeats."""
    loopCount = samples(loopSeconds)
    fadeCount = samples(fadeSeconds)
    out = signal[:loopCount].copy()
    ramp = np.linspace(0.0, 1.0, fadeCount)
    fadeIn = np.sin(0.5 * np.pi * ramp)
    fadeOut = np.cos(0.5 * np.pi * ramp)
    out[:fadeCount] = signal[:fadeCount] * fadeIn + signal[loopCount:loopCount + fadeCount] * fadeOut
    return out


# ======================================================================
# 7. REUSABLE INSTRUMENTS
# ======================================================================
def bellNote(freq, seconds, decayTime=0.6, brightness=1.0):
    """Small hand-bell: slightly inharmonic partials (1, 2, 2.76, 4.07, 5.4)
    plus a quick FM 'strike' shimmer at the start."""
    ratios = [1.0, 2.0, 2.76, 4.07, 5.40]
    amps = [1.0, 0.45, 0.35 * brightness, 0.2 * brightness, 0.12 * brightness]
    decays = [decayTime, decayTime * 0.55, decayTime * 0.4, decayTime * 0.25, decayTime * 0.15]
    freqs = []
    for r in ratios:
        freqs.append(freq * r)
    tone = modes(seconds, freqs, decays, amps)
    strike = fmTone(freq, seconds, 3.5, 2.5 * brightness, 0.0, 0.02) * decay(seconds, 0.03)
    return (tone + 0.3 * strike) * hitEnvelope(seconds, 0.0015, 1e9)


def chimeNote(freq, seconds, brightness=1.0):
    """Clean UI chime: FM with a harmonic ratio (2:1) that starts bright and
    mellows, plus a soft octave shimmer."""
    fm = fmTone(freq, seconds, 2.0, 2.2 * brightness, 0.15, 0.06)
    shimmer = sine(freq * 2.0, seconds) * decay(seconds, 0.12) * 0.25
    return (fm + shimmer) * hitEnvelope(seconds, 0.002, seconds * 0.28)


def softNote(freq, seconds, decayTime=0.15):
    """Mellow triangle-like note (sine + a little 3rd harmonic)."""
    tone = sine(freq, seconds) + 0.12 * sine(freq * 3.0, seconds)
    return tone * hitEnvelope(seconds, 0.003, decayTime)


def brassNote(freq, seconds, loudness=1.0, vibrato=True):
    """Brass-like tone. Like a real trumpet, LOUDER = BRIGHTER: a low-pass
    cutoff follows the loudness envelope. The pitch starts slightly flat
    (the lip 'blip') and two detuned copies sound like a section."""
    t = timeline(seconds)
    envelope = adsr(seconds, 0.035, 0.12, 0.8, min(0.18, seconds * 0.4))
    pitch = freq * (1.0 - 0.025 * np.exp(-t / 0.025))
    if vibrato:
        pitch = pitch * (1.0 + 0.006 * np.sin(2 * np.pi * 5.5 * t) * np.clip((t - 0.2) / 0.3, 0.0, 1.0))
    tone = richWave(pitch, seconds, 1.0, False, 40) + 0.8 * richWave(pitch * 1.004, seconds, 1.0, False, 40)
    cutoff = 350.0 + 4200.0 * envelope * loudness
    tone = sweepFilter(tone, cutoff, 0.9, "low")
    return tone * envelope


def timpaniHit(rng, freq, seconds):
    """Kettle drum: membrane modes (not harmonic) + a soft mallet thump."""
    tone = modes(seconds, [freq, freq * 1.504, freq * 1.742, freq * 2.0, freq * 2.245],
                 [0.9, 0.5, 0.4, 0.3, 0.2], [1.0, 0.5, 0.35, 0.25, 0.15])
    mallet = onePoleLowPass(whiteNoise(rng, seconds), 900.0) * decay(seconds, 0.012)
    return (tone + 2.0 * mallet) * hitEnvelope(seconds, 0.002, 1e9)


def formantVoice(rng, seconds, pitch, formants, tilt=1.2, breath=0.04, jitter=0.0):
    """Tiny voice synthesiser (source-filter model of speech):
    SOURCE = band-limited buzz at the voice pitch (like the vocal folds),
    FILTER = parallel band-pass 'formants' (the mouth shape = the vowel).
    formants = list of (frequency or curve, q, gain)."""
    count = samples(seconds)
    pitchCurve = fitLength(pitch, count)
    if jitter > 0.0:
        pitchCurve = pitchCurve * (1.0 + jitter * smoothRandom(rng, seconds, 60))
    source = richWave(pitchCurve, seconds, tilt, False, 50)
    source = source + breath * whiteNoise(rng, seconds)
    out = np.zeros(count)
    for freq, q, gain in formants:
        out += gain * sweepFilter(source, freq, q, "band")
    return out


# ======================================================================
# 8. UI SOUNDS
# ======================================================================
def makeUiHover(rng):
    """Soft tick: tiny high sine ping (slightly falling) + a breath of noise."""
    length = 0.07
    ping = sine(expCurve(length, 2500, 2250), length) * decay(length, 0.012)
    body = sine(1250, length) * decay(length, 0.018)
    air = bandNoise(rng, length, 5200, 1.0) * decay(length, 0.0025)
    return 0.6 * ping + 0.45 * body + 0.35 * air


def makeUiClick(rng):
    """Wood-block click: modal 'tock' + noise transient + a soft low body."""
    length = 0.12
    tock = modes(length, [820, 1910, 3150], [0.03, 0.016, 0.008], [1.0, 0.45, 0.2])
    transient = bandNoise(rng, length, 3500, 0.8) * decay(length, 0.0015)
    body = sine(expCurve(length, 320, 220), length) * decay(length, 0.025)
    return tock + 0.6 * transient + 0.5 * body


def makeUiConfirm(rng):
    """Two-note rising chime C6 -> G6 (an upward perfect fifth feels like 'yes')."""
    mix = silence(0.9)
    addAt(mix, chimeNote(1046.5, 0.6), 0.0, 0.7)
    addAt(mix, chimeNote(1568.0, 0.8), 0.085, 1.0)
    return reverb(mix, 0.3, 0.2, 0.5)


def makeUiBack(rng):
    """Falling two-note G5 -> C5, softer and darker than confirm ('back')."""
    mix = silence(0.6)
    addAt(mix, softNote(784.0, 0.35, 0.09), 0.0, 0.8)
    addAt(mix, softNote(523.25, 0.45, 0.13), 0.075, 1.0)
    return reverb(lowPass(mix, 3500), 0.25, 0.15, 0.4)


def makeUiPause(rng):
    """Soft dyad that sags down in pitch like a tape slowing (the game freezing)."""
    length = 0.55
    sag = expCurve(length, 1.0, 0.94)
    tone = (sine(659.25 * sag, length) + 0.7 * sine(987.77 * sag, length) +
            0.2 * sine(1318.5 * sag, length))
    tone = tone * adsr(length, 0.012, 0.1, 0.55, 0.3)
    air = bandNoise(rng, length, 1200, 0.7) * curve(length, [0, 0.04, 0.3], [0, 0.12, 0])
    return reverb(lowPass(tone + air, 4000), 0.3, 0.18, 0.45)


def makeCountdownBeep(rng):
    """Clean A5 beep (sine + 2nd/3rd harmonics) - clear but not harsh."""
    length = 0.2
    tone = sine(880, length) + 0.28 * sine(1760, length) + 0.08 * sine(2640, length)
    return reverb(tone * adsr(length, 0.004, 0.04, 0.75, 0.07), 0.2, 0.12, 0.3)


def makeGoWhistle(rng):
    """Referee whistle: the pea inside makes the tone trill (fast AM/FM at 38 Hz);
    breath noise is band-passed around the whistle pitch."""
    length = 0.62
    t = timeline(length)
    pitch = curve(length, [0, 0.05, 0.45, 0.62], [2050, 2300, 2330, 2250])
    trill = np.sin(2 * np.pi * 38.0 * t + 0.8 * smoothRandom(rng, length, 12))
    pitch = pitch * (1.0 + 0.012 * trill)
    tone = sine(pitch, length) + 0.18 * sine(pitch * 2.0, length)
    tone = tone * (0.75 + 0.25 * trill)
    breath = normalise(sweepFilter(whiteNoise(rng, length), pitch, 8.0, "band"))
    envelope = adsr(length, 0.02, 0.05, 0.9, 0.12)
    out = lowPass((tone + 0.35 * breath) * envelope, 7000)
    return reverb(out, 0.3, 0.15, 0.4)


def makeScoreTick(rng):
    """Tiny coin-like tick (3 high modes) for the score counting up."""
    length = 0.06
    ring = modes(length, [3150, 4720, 6900], [0.018, 0.011, 0.006], [1.0, 0.5, 0.25])
    click = bandNoise(rng, length, 6000, 1.0) * decay(length, 0.0012)
    return ring + 0.3 * click


def makeStar(rng, level):
    """Result star: soft 'stamp' thump + bell (E6, G6, C7 for stars 1, 2, 3)
    + glitter grains. Star 3 adds a quick arpeggio and more sparkle."""
    length = 1.6 + 0.2 * level
    mix = silence(length)
    thump = sine(expCurve(0.12, 170, 70), 0.12) * hitEnvelope(0.12, 0.002, 0.035)
    addAt(mix, thump, 0.0, 0.55)
    mainFreq = {1: 1318.5, 2: 1568.0, 3: 2093.0}[level]
    start = 0.0
    if level == 3:
        arpeggio = [1046.5, 1318.5, 1568.0]
        for i in range(3):
            addAt(mix, bellNote(arpeggio[i], 0.5, 0.25), 0.05 * i, 0.35)
        start = 0.15
    addAt(mix, bellNote(mainFreq, 1.3, 0.55), start, 1.0)
    addAt(mix, sine(mainFreq / 2.0, 0.8) * hitEnvelope(0.8, 0.003, 0.3), start, 0.3)
    addAt(mix, sparkleGrains(rng, 0.9, 8 + 6 * level, 3500, 8000, 0.0, 0.6), start + 0.02, 0.35)
    return reverb(mix, 0.4, 0.25, 0.8)


def makeResultStamp(rng):
    """Rubber-stamp THUMP (pitch-dropping sine + paper slap) + a bright shimmer."""
    length = 1.3
    mix = silence(length)
    thump = sine(expCurve(0.25, 150, 48), 0.25) * hitEnvelope(0.25, 0.002, 0.09)
    slap = bandNoise(rng, 0.05, 900, 0.7) * decay(0.05, 0.008)
    paper = normalise(highPass(whiteNoise(rng, 0.05), 3000)) * decay(0.05, 0.01)
    addAt(mix, softClip(thump, 2.0), 0.0, 1.0)
    addAt(mix, slap, 0.0, 0.8)
    addAt(mix, paper, 0.0, 0.2)
    for freq in (2093.0, 2637.0, 3136.0):
        addAt(mix, bellNote(freq, 1.0, 0.6, 0.6), 0.03, 0.22)
    addAt(mix, sparkleGrains(rng, 1.0, 18, 4000, 9000, 0.0, 0.7), 0.04, 0.3)
    return reverb(mix, 0.35, 0.2, 0.7)


def makeFanfare(rng):
    """Triumphant brass fanfare in C major: 'da-da-da DAAA da da DAAAAA'.
    Brass tones + timpani hits + a cymbal swell + hall reverb."""
    C4, E4, G4 = 261.63, 329.63, 392.0
    C5, E5, G5, C6 = 523.25, 659.25, 783.99, 1046.5
    # (start time, length, [notes], loudness)
    phrases = [
        (0.00, 0.10, [G4], 0.6),
        (0.12, 0.10, [G4], 0.6),
        (0.24, 0.10, [G4], 0.65),
        (0.36, 0.30, [C5, G4, E4], 0.85),
        (0.68, 0.14, [E5, C5, G4], 0.8),
        (0.84, 0.14, [G5, E5, C5], 0.85),
        (1.00, 1.50, [C6, G5, E5, C5, C4], 1.0),
    ]
    length = 2.7
    mix = silence(length)
    for start, noteLength, notes, loudness in phrases:
        for i in range(len(notes)):
            gain = loudness * (1.0 if i == 0 else 0.55)
            addAt(mix, brassNote(notes[i], noteLength, loudness, noteLength > 0.5), start, gain)
    addAt(mix, timpaniHit(rng, 130.81, 1.0), 0.36, 0.9)
    addAt(mix, timpaniHit(rng, 98.0, 1.4), 1.0, 1.1)
    cymbal = normalise(sweepFilter(whiteNoise(rng, 1.6), 7000, 0.8, "band"))
    cymbal = cymbal * hitEnvelope(1.6, 0.01, 0.5)
    addAt(mix, cymbal, 1.0, 0.18)
    return reverb(mix, 0.6, 0.25, 1.0, 5000)


def clapSound(rng):
    """One hand clap: a band-passed noise burst (the cupped hands' resonance)
    with two tiny micro-impacts."""
    length = 0.05
    burst = bandNoise(rng, length, rng.uniform(900, 2400), rng.uniform(1.2, 2.5))
    envelope = hitEnvelope(length, 0.0008, rng.uniform(0.006, 0.011))
    second = np.roll(envelope, samples(rng.uniform(0.002, 0.004))) * 0.5
    return burst * (envelope + second)


def makeApplause(rng):
    """3 s of applause: 26 'people' clapping at their own speed with timing
    jitter, built from a bank of 24 different claps, then a room reverb."""
    length = 3.2
    bank = []
    for i in range(24):
        bank.append(clapSound(rng))
    mix = silence(length)
    for person in range(26):
        rate = rng.uniform(3.4, 5.2)
        distance = rng.uniform(0.3, 1.0)
        t = rng.uniform(0.0, 0.35)
        stop = rng.uniform(2.3, 3.0)
        while t < stop:
            clap = bank[rng.integers(len(bank))]
            swell = min(1.0, t / 0.35 + 0.3)
            addAt(mix, clap, t, distance * rng.uniform(0.6, 1.0) * swell)
            t += (1.0 / rate) * rng.uniform(0.85, 1.15)
    return reverb(lowPass(mix, 7000), 0.6, 0.3, 0.8)


def makeAuraMode(rng):
    """Magical rising shimmer: fast pentatonic bell arpeggio + a pad gliding up
    an octave + a rising noise whoosh, blooming into a C major bell chord."""
    length = 1.35
    mix = silence(length)
    pentatonic = [0, 2, 4, 7, 9]
    notes = []
    for octave in range(3):
        for step in pentatonic:
            notes.append(523.25 * 2.0 ** ((12 * octave + step) / 12.0))
    t = 0.0
    for i in range(14):
        note = (sine(notes[i], 0.3) + 0.3 * sine(notes[i] * 2.0, 0.3)) * hitEnvelope(0.3, 0.002, 0.09)
        addAt(mix, note, t, 0.25 + 0.45 * i / 14.0)
        t += 0.066 * (0.97 ** i)
    riseLength = 1.0
    glide = expCurve(riseLength, 1.0, 2.0)
    tremolo = 0.8 + 0.2 * np.sin(2 * np.pi * 11.0 * timeline(riseLength))
    pad = (sine(261.63 * glide, riseLength) + 0.7 * sine(392.0 * glide, riseLength) +
           0.5 * sine(659.25 * glide, riseLength)) * tremolo
    pad = pad * curve(riseLength, [0, 0.7, 0.95, 1.0], [0, 0.5, 0.35, 0])
    addAt(mix, pad, 0.0, 0.35)
    whoosh = normalise(sweepFilter(whiteNoise(rng, riseLength), expCurve(riseLength, 400, 7000), 2.5, "band"))
    addAt(mix, whoosh * curve(riseLength, [0, 0.8, 1.0], [0, 1, 0]), 0.0, 0.25)
    for freq in (1046.5, 1318.5, 1568.0, 2093.0):
        addAt(mix, bellNote(freq, 0.9, 0.45, 0.7), 0.92, 0.3)
    addAt(mix, sparkleGrains(rng, 0.8, 26, 3500, 9000, 0.0, 0.4), 0.9, 0.35)
    return reverb(mix, 0.7, 0.32, 0.6)


def makePerfect(rng):
    """Bright bell ding (A6) with a warm octave below and glitter."""
    mix = silence(1.3)
    addAt(mix, bellNote(1760.0, 1.2, 0.55, 1.0), 0.0, 1.0)
    addAt(mix, sine(880.0, 0.6) * hitEnvelope(0.6, 0.002, 0.25), 0.0, 0.3)
    addAt(mix, sparkleGrains(rng, 0.7, 12, 4000, 9000, 0.0, 0.45), 0.02, 0.3)
    return reverb(mix, 0.35, 0.22, 0.7)


def makeGood(rng):
    """Soft ding (E6): fewer, gentler partials and a shorter ring than 'perfect'."""
    length = 0.6
    tone = modes(length, [1318.5, 2637.0, 3955.5], [0.28, 0.1, 0.05], [1.0, 0.25, 0.08])
    mallet = bandNoise(rng, length, 2000, 0.8) * decay(length, 0.003)
    out = lowPass((tone + 0.15 * mallet) * hitEnvelope(length, 0.002, 1e9), 5000)
    return reverb(out, 0.3, 0.15, 0.4)


def makeMiss(rng):
    """Dull 'boop' (falling sine) + wooden knock + low thud - disappointing but
    not harsh."""
    length = 0.35
    pitch = expCurve(length, 240, 140)
    boop = (sine(pitch, length) + 0.12 * sine(pitch * 3.0, length)) * adsr(length, 0.004, 0.08, 0.5, 0.15)
    knock = modes(length, [170, 395, 640], [0.07, 0.035, 0.02], [0.8, 0.5, 0.3])
    thud = normalise(onePoleLowPass(whiteNoise(rng, length), 600)) * decay(length, 0.02)
    return lowPass(0.7 * boop + knock + 0.6 * thud, 2500)


def buzzPulse(freq, seconds):
    """Soft buzzer tone: two detuned saws beat against each other, then a
    low-pass takes away the harsh top end."""
    tone = richWave(freq, seconds, 1.25, False, 30) + 0.8 * richWave(freq * 1.018, seconds, 1.25, False, 30)
    tone = sweepFilter(tone, 1300.0, 0.8, "low")
    return tone * adsr(seconds, 0.006, 0.03, 0.85, 0.035)


def makeBuzzerWrong(rng):
    """Short low double buzz ('uh-uh'): D#3 then D3."""
    mix = silence(0.4)
    addAt(mix, buzzPulse(155.56, 0.13), 0.0)
    addAt(mix, buzzPulse(146.83, 0.16), 0.18)
    return mix


def makePopup(rng):
    """Tiny pop: a sine that jumps up in pitch very fast, like a bubble."""
    length = 0.08
    t = timeline(length)
    pitch = 380.0 * (1.0 + 2.2 * (1.0 - np.exp(-t / 0.012)))
    pop = sine(pitch, length) * hitEnvelope(length, 0.001, 0.022)
    click = normalise(highPass(whiteNoise(rng, length), 3000)) * decay(length, 0.001)
    return pop + 0.2 * click


# ======================================================================
# 9. DIALOGUE VOICE BLIPS (one short syllable per character)
# ======================================================================
def makeVoiceAiman(rng):
    """'ba' - young man, 136 Hz, lips open so F1/F2 rise into an 'ah'."""
    length = 0.075
    pitch = expCurve(length, 132, 142)
    f1 = curve(length, [0, 0.022, length], [350, 750, 740])
    f2 = curve(length, [0, 0.022, length], [900, 1150, 1120])
    voice = formantVoice(rng, length, pitch, [(f1, 7, 1.0), (f2, 9, 0.6), (2500, 14, 0.25)], 1.2, 0.03)
    return voice * adsr(length, 0.006, 0.02, 0.85, 0.03)


def makeVoiceNenek(rng):
    """'ho' - grandmother, 222 Hz, breathy 'h' onset, soft 'o' and a gentle
    tremble (6 Hz vibrato) of an older voice."""
    length = 0.085
    t = timeline(length)
    pitch = 222.0 * (1.0 + 0.03 * np.sin(2 * np.pi * 6.0 * t)) * expCurve(length, 1.02, 0.97)
    voice = formantVoice(rng, length, pitch, [(490, 6, 1.0), (870, 8, 0.55), (2600, 12, 0.12)], 1.4, 0.12)
    hNoise = normalise(sweepFilter(whiteNoise(rng, length), 1500, 1.0, "band"))
    breathy = hNoise * curve(length, [0, 0.02, 0.035], [0.5, 0.3, 0.0])
    voice = voice * curve(length, [0, 0.015, 0.03], [0.2, 0.6, 1.0])
    return (voice + 0.3 * breathy * np.max(np.abs(voice))) * adsr(length, 0.008, 0.02, 0.85, 0.035)


def makeVoiceLeon(rng):
    """'mo' - the critic: deep (112 Hz, falling), rounded 'o', nasal 'm' onset."""
    length = 0.09
    t = timeline(length)
    pitch = expCurve(length, 114, 104)
    voice = formantVoice(rng, length, pitch, [(470, 6, 1.0), (800, 8, 0.7), (2300, 12, 0.2)], 1.3, 0.02)
    hum = onePoleLowPass(richWave(pitch, length, 1.3, False, 30), 300)
    mouth = np.clip(t / 0.025, 0.0, 1.0)            # 0 = closed 'm', 1 = open 'o'
    mixed = voice * mouth + normalise(hum) * np.max(np.abs(voice)) * 0.8 * (1.0 - mouth)
    return biquad(mixed, "lowshelf", 250, 0.7, 3.0) * adsr(length, 0.01, 0.02, 0.85, 0.035)


def makeVoiceLim(rng):
    """'leh' - Uncle Lim: 128 Hz, gruff (pitch jitter + 35 Hz vocal-fry roughness),
    a slightly nasal 'eh'."""
    length = 0.07
    t = timeline(length)
    f1 = curve(length, [0, 0.02, length], [350, 540, 520])
    f2 = curve(length, [0, 0.02, length], [1300, 1800, 1750])
    formants = [(f1, 7, 1.0), (f2, 9, 0.5), (2500, 12, 0.2), (260, 4, 0.35)]
    voice = formantVoice(rng, length, 128.0, formants, 1.2, 0.05, 0.03)
    roughness = 1.0 - 0.3 * (0.5 + 0.5 * np.sin(2 * np.pi * 35.0 * t))
    return voice * roughness * adsr(length, 0.006, 0.02, 0.85, 0.028)


def makeVoiceKavitha(rng):
    """'yi' - Auntie Kavitha: 205 Hz with a melodic lilt (pitch up then down),
    bright 'ee' vowel gliding from a 'y'."""
    length = 0.08
    pitch = curve(length, [0, 0.03, length], [200, 222, 196])
    f1 = curve(length, [0, 0.025, length], [300, 380, 370])
    f2 = curve(length, [0, 0.025, length], [2050, 2350, 2300])
    formants = [(f1, 6, 1.0), (f2, 10, 0.5), (3000, 12, 0.22)]
    voice = formantVoice(rng, length, pitch, formants, 1.2, 0.03)
    return voice * adsr(length, 0.007, 0.02, 0.85, 0.03)


def makeVoiceFarah(rng):
    """'tae' - Farah: bright student voice, 238 Hz rising, a crisp 't' burst and
    an open 'ae' vowel. The shortest, snappiest blip."""
    length = 0.065
    pitch = expCurve(length, 232, 252)
    formants = [(820, 7, 1.0), (1950, 9, 0.65), (2900, 12, 0.35)]
    voice = formantVoice(rng, length, pitch, formants, 1.0, 0.03)
    tBurst = normalise(highPass(whiteNoise(rng, 0.008), 3500)) * decay(0.008, 0.0015)
    voice = voice * curve(length, [0, 0.006, 0.012], [0.0, 0.3, 1.0])
    out = voice.copy()
    addAt(out, tBurst, 0.0, 0.5 * np.max(np.abs(voice)))
    return out * adsr(length, 0.002, 0.02, 0.85, 0.025)


def makeVoiceOyen(rng):
    """EXTRA: 'mrrp' - a tiny cat trill (28 Hz flutter) for Oyen's dialogue,
    much shorter than cat_meow so it works as a typewriter blip."""
    length = 0.09
    t = timeline(length)
    pitch = expCurve(length, 470, 610)
    f1 = curve(length, [0, length], [400, 850])
    f2 = curve(length, [0, length], [1200, 1850])
    voice = formantVoice(rng, length, pitch, [(f1, 5, 1.0), (f2, 8, 0.6)], 1.2, 0.04)
    trill = 0.65 + 0.35 * np.sin(2 * np.pi * 28.0 * t)
    return voice * trill * adsr(length, 0.008, 0.02, 0.85, 0.035)


# ======================================================================
# 10. SCENE 1 - ROJAK RUSH (fruit stall)
# ======================================================================
def makeChop(rng, variant):
    """Knife into fruit on a wooden board, in three layers:
    1) crisp transient: blade edge breaking the skin + a tiny steel ring,
    2) wet crunch: 28 noise grains (fruit cells bursting) + a juicy squelch,
    3) board knock: wooden modes + low thump when the blade reaches the board."""
    length = 0.32
    mix = silence(length)
    scale = [0.94, 1.0, 1.07][variant - 1]
    crunchLength = [0.045, 0.055, 0.04][variant - 1]
    edge = normalise(highPass(whiteNoise(rng, 0.01), 2500)) * decay(0.01, 0.0012)
    ring = modes(0.08, [3350 * scale, 4870 * scale], [0.02, 0.012], [0.25, 0.12])
    addAt(mix, edge, 0.0, 0.9)
    addAt(mix, ring, 0.0, 1.0)
    for i in range(28):
        when = crunchLength * rng.random() ** 1.5
        grain = bandNoise(rng, 0.004, rng.uniform(1200, 4200), 2.0) * decay(0.004, 0.0012)
        addAt(mix, grain, when, rng.uniform(0.2, 0.7))
    squelch = normalise(sweepFilter(whiteNoise(rng, 0.08), expCurve(0.08, 2200, 500), 2.5, "band"))
    addAt(mix, squelch * hitEnvelope(0.08, 0.002, 0.02), 0.004, 0.45)
    knockTime = crunchLength * 0.4
    knock = modes(0.2, [165 * scale, 385 * scale, 620 * scale, 1040 * scale],
                  [0.07, 0.04, 0.022, 0.012], [1.0, 0.6, 0.35, 0.2])
    thump = sine(expCurve(0.08, 120, 70), 0.08) * hitEnvelope(0.08, 0.001, 0.025)
    addAt(mix, knock, knockTime, 0.9)
    addAt(mix, thump, knockTime, 0.8)
    return mix


def makeKnifeWhoosh(rng):
    """Swish of the cleaver: noise through a band-pass that sweeps up then down,
    plus a thin high 'edge' band. Peaks early because the knife drops in 0.08 s."""
    length = 0.26
    cutoff = curve(length, [0, 0.07, length], [700, 2900, 1100])
    noise = whiteNoise(rng, length)
    air = normalise(sweepFilter(noise, cutoff, 1.3, "band"))
    edge = normalise(sweepFilter(noise, cutoff * 2.1, 5.0, "band"))
    envelope = curve(length, [0, 0.06, 0.1, length], [0, 1, 0.55, 0]) ** 2
    return (air + 0.35 * edge) * envelope


def makeJuiceSquish(rng):
    """Wet splat (downward resonant sweep + low-passed noise) followed by
    droplets (rising bubble chirps)."""
    length = 0.42
    mix = silence(length)
    splat = normalise(sweepFilter(whiteNoise(rng, 0.12), expCurve(0.12, 1600, 350), 3.0, "band"))
    addAt(mix, splat * hitEnvelope(0.12, 0.002, 0.035), 0.0, 1.0)
    body = normalise(onePoleLowPass(whiteNoise(rng, 0.08), 900)) * decay(0.08, 0.018)
    addAt(mix, body, 0.0, 0.7)
    for i in range(9):
        when = 0.02 + 0.3 * rng.random() ** 1.3
        drop = bubble(rng.uniform(900, 2600), rng.uniform(0.012, 0.03), 1.8)
        addAt(mix, drop, when, rng.uniform(0.2, 0.6) * (1.0 - when / length))
    return mix


def makeBoardThud(rng):
    """Fruit landing on the board: low thump + wooden modes + a small bounce."""
    length = 0.38
    mix = silence(length)
    for when, gain in ((0.0, 1.0), (0.07, 0.3)):
        thump = sine(expCurve(0.12, 120, 80), 0.12) * hitEnvelope(0.12, 0.002, 0.04)
        wood = modes(0.2, [180, 350, 540, 900], [0.07, 0.05, 0.03, 0.015], [0.8, 0.6, 0.35, 0.2])
        impact = normalise(onePoleLowPass(whiteNoise(rng, 0.02), 2000)) * decay(0.02, 0.004)
        addAt(mix, thump, when, gain)
        addAt(mix, wood, when, 0.7 * gain)
        addAt(mix, impact, when, 0.5 * gain)
    return mix


def makePlop(rng):
    """Fruit into the bowl: a rising 'plop' sine + soft thud + ceramic clink."""
    length = 0.32
    t = timeline(length)
    mix = silence(length)
    pitch = 380.0 * (1.0 + 1.3 * (1.0 - np.exp(-t / 0.03)))
    addAt(mix, sine(pitch, length) * hitEnvelope(length, 0.003, 0.045), 0.0, 1.0)
    addAt(mix, sine(expCurve(0.1, 150, 90), 0.1) * hitEnvelope(0.1, 0.002, 0.03), 0.0, 0.6)
    clink = modes(0.3, [2150, 3950, 5400], [0.12, 0.07, 0.04], [1.0, 0.6, 0.3])
    addAt(mix, clink, 0.01, 0.2)
    return mix


def makeCatMeow(rng):
    """Cartoon 'mee-ow': a buzz source with a pitch glide up then down, sent
    through three moving formants (mouth opening from 'm' to 'e' to 'a' to 'ow').
    The first 80 ms blend from a nasal hum (closed mouth)."""
    length = 0.62
    t = timeline(length)
    pitch = curve(length, [0, 0.1, 0.28, 0.48, length], [520, 700, 780, 590, 450])
    vibrato = 1.0 + 0.02 * np.sin(2 * np.pi * 7.0 * t) * np.clip((t - 0.12) / 0.1, 0.0, 1.0)
    pitch = pitch * vibrato
    source = richWave(pitch, length, 1.2, False, 30) + 0.04 * whiteNoise(rng, length)
    times = [0, 0.08, 0.28, 0.5, length]
    f1 = curve(length, times, [350, 750, 1100, 800, 500])
    f2 = curve(length, times, [1600, 2200, 1750, 1250, 950])
    f3 = curve(length, times, [2900, 3300, 3100, 2800, 2600])
    voice = (sweepFilter(source, f1, 6.0, "band") + 0.6 * sweepFilter(source, f2, 8.0, "band") +
             0.25 * sweepFilter(source, f3, 10.0, "band"))
    nasal = normalise(onePoleLowPass(source, 450)) * np.max(np.abs(voice))
    blend = np.clip(t / 0.08, 0.0, 1.0)
    out = voice * blend + 0.8 * nasal * (1.0 - blend)
    return out * curve(length, [0, 0.04, 0.2, 0.45, length], [0, 0.6, 1.0, 0.7, 0])


def makeCatHiss(rng):
    """Angry hiss: two noise bands (4.6 and 7.2 kHz) with a fast random flutter,
    a spit burst at the start and a quiet throaty growl underneath."""
    length = 0.7
    noise = whiteNoise(rng, length)
    hiss = sweepFilter(noise, 4600, 1.1, "band") + 0.5 * sweepFilter(noise, 7200, 1.5, "band")
    hiss = normalise(highPass(hiss, 1800))
    flutter = 1.0 + 0.3 * smoothRandom(rng, length, 30)
    envelope = curve(length, [0, 0.015, 0.05, 0.4, length], [0, 1, 0.8, 0.65, 0])
    spit = whiteNoise(rng, 0.03) * hitEnvelope(0.03, 0.0005, 0.006)
    growlPitch = 78.0 * (1.0 + 0.05 * smoothRandom(rng, length, 40))
    growl = normalise(sweepFilter(richWave(growlPitch, length, 1.1, False, 40), 600, 0.8, "low"))
    mix = hiss * flutter * envelope + 0.18 * growl * envelope
    addAt(mix, spit, 0.0, 0.8)
    return mix


def makeSpicySizzle(rng):
    """Chili heat: bright sizzle hiss + crackles + a quiet rising 'heat' whistle."""
    length = 0.9
    t = timeline(length)
    hiss = normalise(highPass(lowPass(whiteNoise(rng, length), 10000), 3000))
    hiss = hiss * (0.8 + 0.2 * smoothRandom(rng, length, 25))
    crackles = crackleTrain(rng, length, 90, [(2500, 4), (4200, 5), (6500, 6)])
    envelope = curve(length, [0, 0.02, 0.25, length], [0, 1, 0.55, 0])
    whistlePitch = curve(length, [0, length], [650, 1500]) * (1.0 + 0.01 * np.sin(2 * np.pi * 6 * t))
    whistle = sine(whistlePitch, length) * curve(length, [0, 0.2, 0.6, length], [0, 0.12, 0.1, 0])
    return (0.5 * hiss + 0.8 * crackles) * envelope + whistle


def makeSaucePour(rng):
    """Thick rojak sauce: a low-passed stream + a resonant 'gloopy' wobble +
    low glug bubbles, ending with one drip."""
    length = 1.6
    noise = whiteNoise(rng, length)
    stream = normalise(onePoleLowPass(noise, 450))
    wobble = 500.0 + 350.0 * smoothRandom(rng, length, 7)
    gloop = normalise(sweepFilter(noise, wobble, 4.0, "band"))
    mix = 0.5 * stream + 0.6 * gloop
    for when in randomTimes(rng, 1.3, 9):
        glug = bubble(rng.uniform(140, 320), rng.uniform(0.05, 0.09), 1.9, 0.35)
        addAt(mix, glug, 0.1 + when, rng.uniform(0.6, 1.0))
    mix = mix * curve(length, [0, 0.1, 1.2, 1.45, length], [0, 1, 0.9, 0.2, 0])
    addAt(mix, bubble(700, 0.03, 1.6), 1.45, 0.5)
    return mix


def makeSprinkle(rng):
    """Crushed peanuts: ~70 tiny hard grains (short resonant noise bursts)
    whose density swells then fades, plus softer landings on the food."""
    length = 1.0
    mix = silence(length)
    hardGrains = []
    softGrains = []
    for i in range(12):
        grain = bandNoise(rng, 0.006, rng.uniform(2500, 6500), rng.uniform(3, 6)) * decay(0.006, 0.0015)
        hardGrains.append(grain)
    for i in range(6):
        grain = bandNoise(rng, 0.01, rng.uniform(900, 1600), 2.0) * decay(0.01, 0.003)
        softGrains.append(grain)
    density = curve(length, [0, 0.1, 0.35, 0.7, 0.85], [0.2, 0.9, 1.0, 0.4, 0.0])
    for when in randomTimes(rng, 0.85, 110):
        if rng.random() > density[int(when * SAMPLE_RATE)]:
            continue
        if rng.random() < 0.75:
            grain = hardGrains[rng.integers(len(hardGrains))]
            addAt(mix, grain, when, rng.uniform(0.2, 1.0) ** 1.5)
        else:
            grain = softGrains[rng.integers(len(softGrains))]
            addAt(mix, grain, when + 0.03, rng.uniform(0.2, 0.6))
    rattle = normalise(onePoleLowPass(whiteNoise(rng, length), 1500)) * density
    rattle = rattle * (0.5 + 0.5 * np.abs(np.sin(2 * np.pi * 7.0 * timeline(length))))
    return mix + 0.06 * rattle


# ======================================================================
# 11. SCENE 2 - WOK HEI FIRE
# ======================================================================
def makeSizzleLoop(rng):
    """Seamless 4 s oil sizzle: bright hiss bed with a slow random swell +
    four bands of crackles (Poisson clicks) + small popping bubbles."""
    loopLength = 4.0
    fade = 0.35
    total = loopLength + fade
    hiss = normalise(biquad(highPass(whiteNoise(rng, total), 2600), "lowpass", 9500))
    hiss = hiss * (0.75 + 0.25 * smoothRandom(rng, total, 5))
    crackles = crackleTrain(rng, total, 140, [(2200, 3), (3600, 4), (5200, 5), (7400, 6)])
    pops = silence(total)
    for when in randomTimes(rng, total, 3.0):
        addAt(pops, bubble(rng.uniform(1200, 2600), rng.uniform(0.008, 0.015), 1.7), when, rng.uniform(0.3, 1.0))
    mix = 0.45 * hiss + 0.9 * crackles + 0.4 * pops
    return seamlessLoop(mix, loopLength, fade)


def makeFireLoop(rng):
    """Seamless 4 s gas-burner roar: brown-noise rumble + a mid 'body' band,
    modulated by slow turbulence and faster flutter, plus the thin hiss of
    the gas jet."""
    loopLength = 4.0
    fade = 0.4
    total = loopLength + fade
    roar = brownNoise(rng, total)
    body = normalise(sweepFilter(whiteNoise(rng, total), 520, 0.7, "band"))
    jet = normalise(sweepFilter(whiteNoise(rng, total), 1900, 0.9, "band"))
    turbulence = 0.7 + 0.3 * smoothRandom(rng, total, 3)
    flutter = 0.85 + 0.15 * smoothRandom(rng, total, 16)
    mix = (roar + 0.35 * body) * turbulence * flutter + 0.12 * jet * (0.9 + 0.1 * smoothRandom(rng, total, 2))
    mix = lowPass(mix, 6000)
    return seamlessLoop(mix, loopLength, fade)


def makeSizzleBurst(rng):
    """Food hitting hot oil: an impact 'splat', then a loud hiss and a storm of
    crackles that thin out exponentially over one second."""
    length = 1.1
    t = timeline(length)
    density = 0.9 * np.exp(-t / 0.3) + 0.1
    crackles = crackleTrain(rng, length, 450, [(2200, 3), (3600, 4), (5200, 5), (7400, 6)], density)
    hiss = normalise(lowPass(highPass(whiteNoise(rng, length), 1800), 10000))
    hiss = hiss * (0.15 + 0.85 * np.exp(-t / 0.28)) * (0.85 + 0.15 * smoothRandom(rng, length, 20))
    mix = 0.7 * hiss + 0.8 * crackles
    impact = normalise(onePoleLowPass(whiteNoise(rng, 0.08), 1400)) * hitEnvelope(0.08, 0.002, 0.02)
    thump = sine(expCurve(0.1, 130, 80), 0.1) * hitEnvelope(0.1, 0.002, 0.04)
    addAt(mix, impact, 0.0, 0.6)
    addAt(mix, thump, 0.0, 0.5)
    return mix * curve(length, [0, 0.003, length * 0.8, length], [0, 1, 1, 0])


def makeWokClang(rng):
    """Carbon-steel wok: 8 inharmonic metal modes, each doubled with a tiny
    detune so they beat (shimmer), + a spatula scrape that also excites two of
    the wok's own modes."""
    length = 0.9
    mix = silence(length)
    freqs = [310, 845, 1365, 1990, 2710, 3530, 4480, 5600]
    decays = [0.55, 0.4, 0.3, 0.22, 0.16, 0.11, 0.08, 0.05]
    amps = [0.55, 1.0, 0.75, 0.6, 0.45, 0.32, 0.22, 0.14]
    detuned = []
    for f in freqs:
        detuned.append(f * 1.0035)
    clang = modes(length, freqs, decays, amps) + 0.6 * modes(length, detuned, decays, amps)
    strike = normalise(highPass(whiteNoise(rng, 0.01), 2000)) * decay(0.01, 0.0015)
    addAt(mix, clang * hitEnvelope(length, 0.0008, 1e9), 0.0, 1.0)
    addAt(mix, strike, 0.0, 0.7)
    scrapeLength = 0.38
    noise = whiteNoise(rng, scrapeLength)
    scrapeNoise = (normalise(sweepFilter(noise, 2300 + 600 * smoothRandom(rng, scrapeLength, 8), 2.0, "band")) +
                   0.5 * normalise(sweepFilter(noise, 4200, 3.0, "band")))
    metallic = normalise(bandPass(scrapeNoise, 1365, 25) + bandPass(scrapeNoise, 2710, 25))
    stickSlip = 0.5 + 0.5 * np.abs(smoothRandom(rng, scrapeLength, 120))
    scrapeEnvelope = curve(scrapeLength, [0, 0.05, 0.25, scrapeLength], [0, 1, 0.8, 0])
    scrape = (0.5 * normalise(scrapeNoise) + 0.5 * metallic) * stickSlip * scrapeEnvelope
    addAt(mix, scrape, 0.09, 0.45)
    return mix


def makeFlareWhoosh(rng):
    """Wok hei ignition 'FWOOMP': noise through a low-pass that snaps open and
    slowly closes, a sub-bass pressure swell, a brown-noise roar and crackles,
    glued together with soft saturation."""
    length = 1.3
    noise = whiteNoise(rng, length)
    cutoff = curve(length, [0, 0.12, 0.35, length], [180, 2600, 1400, 500])
    whoosh = normalise(sweepFilter(noise, cutoff, 1.1, "low"))
    whoosh = whoosh * curve(length, [0, 0.06, 0.14, 0.5, length], [0, 0.8, 1.0, 0.55, 0])
    sub = sine(curve(length, [0, 0.15, 0.6], [45, 75, 60]), length)
    sub = sub * curve(length, [0, 0.05, 0.12, 0.6], [0, 1, 0.8, 0])
    roar = brownNoise(rng, length) * curve(length, [0, 0.1, 0.4, length], [0, 0.8, 0.5, 0])
    t = timeline(length)
    crackles = crackleTrain(rng, length, 60, [(2500, 4), (4500, 5)], np.exp(-t / 0.4))
    mix = whoosh + 0.7 * sub + 0.5 * roar + 0.35 * crackles
    return reverb(softClip(mix, 1.3), 0.35, 0.15, 0.5)


def makeSmokeAlarm(rng):
    """Beep-beep-beep: a softened square (odd harmonics, steep 1/k^1.8 roll-off)
    at 2.35 kHz with a 14 Hz warble - clearly an alarm, but not piercing."""
    mix = silence(1.4)
    beepLength = 0.28
    t = timeline(beepLength)
    pitch = 2350.0 * (1.0 + 0.012 * np.sin(2 * np.pi * 14.0 * t))
    tone = onePoleLowPass(richWave(pitch, beepLength, 1.8, True, 7), 6000)
    beep = normalise(tone) * adsr(beepLength, 0.006, 0.02, 0.9, 0.02)
    for i in range(3):
        addAt(mix, beep, i * 0.5, 1.0)
    return reverb(mix, 0.25, 0.12, 0.3)


def makeSteamHiss(rng):
    """Steam: band-passed noise sweeping 3 -> 5.2 kHz with a fast flutter."""
    length = 0.95
    cutoff = curve(length, [0, 0.1, length], [3000, 5200, 4200])
    noise = whiteNoise(rng, length)
    hiss = normalise(sweepFilter(noise, cutoff, 0.9, "band"))
    hiss = hiss + 0.3 * normalise(highPass(noise, 6000))
    flutter = 0.8 + 0.2 * smoothRandom(rng, length, 18)
    return hiss * flutter * curve(length, [0, 0.04, 0.35, length], [0, 1, 0.7, 0])


def makePlateSlide(rng):
    """Plate sliding on steel: friction noise with stick-slip roughness, then a
    ceramic 'tok' as it stops."""
    length = 0.75
    mix = silence(length)
    slideLength = 0.45
    noise = whiteNoise(rng, slideLength)
    friction = (normalise(sweepFilter(noise, 1500 + 300 * smoothRandom(rng, slideLength, 6), 1.6, "band")) +
                0.5 * normalise(sweepFilter(noise, 3300, 3.0, "band")))
    stickSlip = 0.55 + 0.45 * np.abs(smoothRandom(rng, slideLength, 70))
    envelope = curve(slideLength, [0, 0.06, 0.3, slideLength], [0, 1, 0.8, 0.1])
    addAt(mix, friction * stickSlip * envelope, 0.0, 0.6)
    tok = modes(0.3, [1850, 3350, 5100], [0.09, 0.05, 0.03], [1.0, 0.5, 0.25])
    thud = sine(210, 0.1) * hitEnvelope(0.1, 0.001, 0.04)
    addAt(mix, tok * hitEnvelope(0.3, 0.001, 1e9), 0.44, 0.7)
    addAt(mix, thud, 0.44, 0.5)
    return mix


def makeKnobClick(rng):
    """Gas knob detent: two micro clicks 18 ms apart (plastic + small metal ring)."""
    length = 0.12
    mix = silence(length)
    for when, gain in ((0.0, 0.6), (0.018, 1.0)):
        click = normalise(highPass(whiteNoise(rng, 0.004), 3000)) * decay(0.004, 0.0008)
        ring = modes(0.05, [2750, 4600, 1150], [0.009, 0.006, 0.012], [0.4, 0.25, 0.35])
        addAt(mix, click, when, gain)
        addAt(mix, ring, when, gain)
    return mix


# ======================================================================
# 12. SCENE 3 - THE GREAT PULL (teh tarik)
# ======================================================================
def makePourLoop(rng):
    """Seamless 3 s liquid stream: ~190 random bubbles per second (log-uniform
    pitch 450-2200 Hz, each a rising chirp) over a soft low-passed 'rush'.
    The game raises its playback speed as the glass fills."""
    loopLength = 3.0
    fade = 0.3
    total = loopLength + fade
    bubbles = silence(total)
    for when in randomTimes(rng, total, 190):
        freq = 450.0 * (2200.0 / 450.0) ** rng.random()
        length = rng.uniform(0.006, 0.022)
        addAt(bubbles, bubble(freq, length, rng.uniform(1.3, 2.1), 0.35), when, rng.random() ** 2)
    noise = whiteNoise(rng, total)
    rush = normalise(onePoleLowPass(noise, 1100)) * (0.85 + 0.15 * smoothRandom(rng, total, 4))
    body = normalise(sweepFilter(noise, 380, 1.2, "band"))
    mix = 0.8 * normalise(bubbles) + 0.25 * rush + 0.2 * body
    return seamlessLoop(mix, loopLength, fade)


def makeSplash(rng, size):
    """Tea splashing on the counter: a slap, a spray sweep and droplets.
    size 2 is bigger (lower slap, more drops, longer)."""
    length = 0.45 + 0.2 * size
    mix = silence(length)
    slap = normalise(onePoleLowPass(whiteNoise(rng, 0.06), 3500)) * hitEnvelope(0.06, 0.001, 0.012)
    body = sine(expCurve(0.06, 220 - 40 * size, 120), 0.06) * hitEnvelope(0.06, 0.001, 0.02)
    spray = normalise(sweepFilter(whiteNoise(rng, 0.25), expCurve(0.25, 2500, 1200), 1.2, "band"))
    addAt(mix, slap, 0.0, 1.0)
    addAt(mix, body, 0.0, 0.4)
    addAt(mix, spray * hitEnvelope(0.25, 0.003, 0.06), 0.0, 0.6)
    for i in range(10 + 6 * size):
        when = 0.03 + (length - 0.12) * rng.random() ** 1.6
        drop = bubble(rng.uniform(700, 2600), rng.uniform(0.01, 0.03), 1.8)
        addAt(mix, drop, when, rng.uniform(0.15, 0.55) * (1.0 - when / length))
    return mix


def makeBubblePop(rng):
    """One foam bubble popping: a tiny rising chirp + a click."""
    length = 0.06
    pop = bubble(1050, 0.03, 1.9, 0.3)
    click = normalise(highPass(whiteNoise(rng, 0.005), 3000)) * decay(0.005, 0.0007)
    mix = silence(length)
    addAt(mix, pop, 0.0, 1.0)
    addAt(mix, click, 0.0, 0.3)
    return mix


def makeGlassClink(rng):
    """Glass: 4 high modes (2.15, 4.83, 7.42, 9.95 kHz) doubled with a small
    detune for a shimmering ring, a hard click, and a second softer touch."""
    length = 1.1
    mix = silence(length)
    freqs = [2150, 4830, 7420, 9950]
    decays = [0.6, 0.32, 0.18, 0.1]
    amps = [1.0, 0.55, 0.3, 0.18]
    detuned = []
    for f in freqs:
        detuned.append(f * 1.0025)
    ring = (modes(length, freqs, decays, amps) + 0.7 * modes(length, detuned, decays, amps))
    ring = ring * hitEnvelope(length, 0.0005, 1e9)
    click = normalise(highPass(whiteNoise(rng, 0.003), 4000)) * decay(0.003, 0.0006)
    addAt(mix, ring, 0.0, 1.0)
    addAt(mix, click, 0.0, 0.5)
    addAt(mix, ring, 0.085, 0.3)
    return reverb(mix, 0.3, 0.15, 0.4)


def makeSlurp(rng):
    """Slurp: air through liquid = noise through a narrow band-pass sweeping
    up, chopped by a bubbly gurgle; then a small satisfied exhale."""
    length = 0.8
    mix = silence(length)
    suckLength = 0.52
    cutoff = curve(suckLength, [0, 0.45, suckLength], [800, 2600, 2000])
    air = normalise(sweepFilter(whiteNoise(rng, suckLength), cutoff, 3.0, "band"))
    gurgle = 0.4 + 0.6 * np.abs(smoothRandom(rng, suckLength, 28))
    envelope = curve(suckLength, [0, 0.03, 0.45, suckLength], [0, 1, 0.9, 0])
    addAt(mix, air * gurgle * envelope, 0.0, 0.8)
    for i in range(8):
        addAt(mix, bubble(rng.uniform(400, 1000), rng.uniform(0.02, 0.04), 1.7), rng.uniform(0.05, 0.45), 0.35)
    exhale = normalise(sweepFilter(whiteNoise(rng, 0.22), 1000, 1.5, "band"))
    addAt(mix, exhale * curve(0.22, [0, 0.05, 0.22], [0, 1, 0]), 0.56, 0.2)
    return mix


def makeNeonBuzz(rng):
    """Seamless 2 s neon hum. Malaysia's mains is 50 Hz, so the ballast hums
    at 100 Hz (+ harmonics) and the arc 'fizz' is noise gated 100 times a
    second. 100 Hz fits exactly 200 cycles in 2 s, so the tone loops
    perfectly; only the noise part needs the crossfade."""
    loopLength = 2.0
    fade = 0.25
    t = timeline(loopLength)
    hum = np.zeros(len(t))
    harmonicLevels = [1.0, 0.55, 0.4, 0.22, 0.18, 0.12, 0.08]
    for k in range(len(harmonicLevels)):
        hum += harmonicLevels[k] * np.sin(2 * np.pi * 100.0 * (k + 1) * t + 0.7 * k)
    noise = seamlessLoop(whiteNoise(rng, loopLength + fade), loopLength, fade)
    gate = np.abs(np.sin(2 * np.pi * 50.0 * t)) ** 12
    fizz = normalise(sweepFilter(noise, 3200, 0.8, "band")) * gate
    flicker = 1.0 + 0.08 * np.sin(2 * np.pi * 1.5 * t) + 0.05 * np.sin(2 * np.pi * 3.5 * t + 1.0)
    return (0.6 * normalise(hum) + 0.35 * normalise(fizz)) * flicker


def makeWebcamOn(rng):
    """Device 'boot' sound: relay click, an upward chirp and two soft square
    blips, then sample-and-hold to 11 kHz for a deliberately 'digital' edge."""
    length = 0.45
    mix = silence(length)
    click = normalise(highPass(whiteNoise(rng, 0.004), 2500)) * decay(0.004, 0.0008)
    chirp = sine(expCurve(0.12, 600, 2400), 0.12) * curve(0.12, [0, 0.02, 0.12], [0, 0.5, 0])
    blip1 = richWave(1318.5, 0.06, 1.5, True, 9) * adsr(0.06, 0.003, 0.01, 0.8, 0.02)
    blip2 = richWave(1975.5, 0.09, 1.5, True, 9) * adsr(0.09, 0.003, 0.02, 0.8, 0.04)
    addAt(mix, click, 0.0, 0.5)
    addAt(mix, chirp, 0.01, 0.6)
    addAt(mix, blip1, 0.14, 0.7)
    addAt(mix, blip2, 0.22, 0.8)
    return lowPass(sampleAndHold(mix, 11025), 9000)


# ======================================================================
# 13. SCENE 4 - LAZY SUSAN BANQUET
# ======================================================================
def makeSusanRumble(rng):
    """Seamless 3 s rolling rumble of the glass turntable on ball bearings:
    looped low noise, shaped by PERIODIC rolling bumps (6 Hz and 7/3 Hz both
    fit a whole number of times into 3 s) + a quiet gritty bearing rattle."""
    loopLength = 3.0
    fade = 0.3
    total = loopLength + fade
    rumble = normalise(onePoleLowPass(brownNoise(rng, total), 160))
    grind = normalise(sweepFilter(whiteNoise(rng, total), 95, 1.4, "band"))
    rumbleLoop = seamlessLoop(0.8 * rumble + 0.6 * grind, loopLength, fade)
    grit = normalise(sweepFilter(whiteNoise(rng, total), 2600, 2.0, "band"))
    gritLoop = seamlessLoop(grit, loopLength, fade)
    t = timeline(loopLength)
    roll = 0.75 + 0.15 * np.sin(2 * np.pi * 6.0 * t) + 0.1 * np.sin(2 * np.pi * (7.0 / 3.0) * t + 0.5)
    gritPulse = 0.3 + 0.7 * np.abs(np.sin(2 * np.pi * 6.0 * t)) ** 3
    return normalise(rumbleLoop) * roll + 0.08 * gritLoop * gritPulse


def ceramicClink(rng, baseFreq, seconds=0.3):
    """Plate/bowl clink: three ceramic modes with a little random variation."""
    freqs = [baseFreq, baseFreq * (2.32 + rng.uniform(-0.1, 0.1)), baseFreq * (3.9 + rng.uniform(-0.2, 0.2))]
    ring = modes(seconds, freqs, [0.09, 0.05, 0.03], [1.0, 0.5, 0.25])
    return ring * hitEnvelope(seconds, 0.0008, 1e9)


def makeDishServe(rng):
    """Serving: a quick whoosh, then the plate lands (ceramic clink + table knock)."""
    length = 0.7
    mix = silence(length)
    whoosh = normalise(sweepFilter(whiteNoise(rng, 0.24), curve(0.24, [0, 0.12, 0.24], [800, 2400, 1000]), 1.3, "band"))
    addAt(mix, whoosh * curve(0.24, [0, 0.14, 0.24], [0, 1, 0]) ** 2, 0.0, 0.5)
    freqs = [1650, 3720, 5900, 8100]
    detuned = []
    for f in freqs:
        detuned.append(f * 1.003)
    clink = modes(0.5, freqs, [0.2, 0.11, 0.06, 0.035], [1.0, 0.5, 0.3, 0.15])
    clink = (clink + 0.6 * modes(0.5, detuned, [0.2, 0.11, 0.06, 0.035], [1.0, 0.5, 0.3, 0.15]))
    clink = clink * hitEnvelope(0.5, 0.0008, 1e9)
    knock = sine(220, 0.1) * hitEnvelope(0.1, 0.001, 0.04) + modes(0.1, [210, 480], [0.04, 0.02], [0.6, 0.3])
    addAt(mix, clink, 0.2, 1.0)
    addAt(mix, knock, 0.2, 0.6)
    addAt(mix, clink, 0.265, 0.25)
    return mix


def nomSyllable(rng, seconds, basePitch):
    """One 'nom': the mouth goes closed (n) -> open (o) -> closed (m), done by
    moving formants and blending with a nasal hum."""
    pitch = basePitch * curve(seconds, [0, seconds * 0.4, seconds], [1.0, 1.06, 0.9])
    source = richWave(pitch, seconds, 1.3, False, 40)
    mouth = curve(seconds, [0, 0.03, seconds - 0.06, seconds], [0, 1, 1, 0])
    f1 = 280 + 300 * mouth
    f2 = 900 + 150 * mouth
    voice = sweepFilter(source, f1, 5.0, "band") + 0.5 * sweepFilter(source, f2, 7.0, "band")
    nasal = normalise(onePoleLowPass(source, 350)) * np.max(np.abs(voice))
    out = voice * mouth + 0.6 * nasal * (1.0 - mouth)
    return out * adsr(seconds, 0.01, 0.03, 0.85, 0.04)


def makeNom(rng):
    """Cartoon munch 'nom-nom': two voiced syllables + little chewing crunches."""
    length = 0.45
    mix = silence(length)
    addAt(mix, nomSyllable(rng, 0.16, 210), 0.0, 1.0)
    addAt(mix, nomSyllable(rng, 0.17, 185), 0.2, 0.9)
    loudest = np.max(np.abs(mix))
    for start in (0.02, 0.22):
        for i in range(6):
            grain = bandNoise(rng, 0.005, rng.uniform(2000, 4000), 2.0) * decay(0.005, 0.0015)
            addAt(mix, grain, start + rng.uniform(0, 0.1), loudest * rng.uniform(0.1, 0.3))
    return mix


def makeHappyChime(rng):
    """Happy customer: G major arpeggio (G5 B5 D6 G6) plucked with
    Karplus-Strong 'harp' strings, doubled softly by bells."""
    mix = silence(1.2)
    notes = [783.99, 987.77, 1174.66, 1567.98]
    for i in range(4):
        pluck = karplusStrong(rng, notes[i], 0.9, 0.6, 0.997) * hitEnvelope(0.9, 0.002, 1e9)
        addAt(mix, pluck, 0.07 * i, 0.8)
        addAt(mix, bellNote(notes[i], 0.9, 0.4, 0.5), 0.07 * i, 0.25)
    return reverb(mix, 0.45, 0.28, 0.8)


def makeGrumble(rng):
    """Unhappy customer 'hrmm-mph': low voice (110-125 Hz) with jitter and
    vocal-fry roughness; the second part falls in pitch and closes to a hum."""
    length = 0.6
    mix = silence(length)
    partA = 0.24
    pitchA = curve(partA, [0, partA], [112, 124])
    voiceA = formantVoice(rng, partA, pitchA, [(520, 6, 1.0), (1150, 8, 0.5), (2400, 12, 0.15)], 1.3, 0.1, 0.03)
    tA = timeline(partA)
    voiceA = voiceA * (1.0 - 0.35 * (0.5 + 0.5 * np.sin(2 * np.pi * 32.0 * tA)))
    addAt(mix, voiceA * adsr(partA, 0.02, 0.04, 0.8, 0.06), 0.0, 1.0)
    partB = 0.31
    tB = timeline(partB)
    pitchB = curve(partB, [0, partB], [120, 88])
    f1 = curve(partB, [0, 0.1, partB], [500, 420, 280])
    voiceB = formantVoice(rng, partB, pitchB, [(f1, 6, 1.0), (1000, 8, 0.4)], 1.3, 0.05, 0.03)
    closing = np.clip((tB - 0.1) / 0.12, 0.0, 1.0)
    hum = normalise(onePoleLowPass(richWave(pitchB, partB, 1.3, False, 30), 300)) * np.max(np.abs(voiceB))
    voiceB = voiceB * (1.0 - closing) + 0.8 * hum * closing
    voiceB = voiceB * (1.0 - 0.3 * (0.5 + 0.5 * np.sin(2 * np.pi * 30.0 * tB)))
    addAt(mix, voiceB * adsr(partB, 0.02, 0.05, 0.8, 0.1), 0.27, 1.0)
    return mix


def makeCameraShutter(rng):
    """Camera: 'click-clack' (two mechanical transients 60 ms apart) followed
    by a short film-advance whirr (gear-modulated buzz)."""
    length = 0.42
    mix = silence(length)
    for when, gain in ((0.0, 0.8), (0.06, 1.0)):
        click = normalise(highPass(whiteNoise(rng, 0.004), 2000)) * decay(0.004, 0.0008)
        mechanism = modes(0.04, [1900, 3400, 5200], [0.012, 0.008, 0.005], [0.5, 0.35, 0.25])
        body = sine(420, 0.04) * hitEnvelope(0.04, 0.0005, 0.01)
        addAt(mix, click, when, gain)
        addAt(mix, mechanism, when, gain)
        addAt(mix, body, when, 0.4 * gain)
    whirrLength = 0.24
    tw = timeline(whirrLength)
    gears = 0.6 + 0.4 * np.abs(np.sin(2 * np.pi * 45.0 * tw))
    whirr = lowPass(richWave(curve(whirrLength, [0, whirrLength], [95, 110]), whirrLength, 1.3, False, 40), 1800)
    whirr = whirr * gears * curve(whirrLength, [0, 0.03, 0.2, whirrLength], [0, 1, 0.8, 0])
    addAt(mix, whirr, 0.12, 0.2)
    return mix


def makeWobble(rng):
    """Dishes clattering on a spinning table: 10 random ceramic clinks clustered
    in half a second + a low rattle of plates shifting."""
    length = 0.75
    mix = silence(length)
    for i in range(10):
        when = 0.5 * rng.random()
        addAt(mix, ceramicClink(rng, rng.uniform(1300, 2700), 0.25), when, rng.uniform(0.3, 1.0))
    rattle = normalise(onePoleLowPass(whiteNoise(rng, length), 600))
    rattle = rattle * (0.5 + 0.5 * np.abs(smoothRandom(rng, length, 25)))
    rattle = rattle * curve(length, [0, 0.03, 0.45, length], [0, 1, 0.6, 0])
    return mix + 0.25 * rattle


# ======================================================================
# 14. STREET AND FINALE
# ======================================================================
def makeFootstep(rng, variant):
    """Soft sneaker on tiles: heel thud + rubber scuff, then the toe lands.
    Variant 2 is slightly lower with a faint rubber squeak."""
    length = 0.2
    mix = silence(length)
    heelPitch = 105 if variant == 1 else 92
    heel = sine(expCurve(0.06, heelPitch, 65), 0.06) * hitEnvelope(0.06, 0.002, 0.018)
    scuff = normalise(onePoleLowPass(whiteNoise(rng, 0.05), 1100)) * hitEnvelope(0.05, 0.002, 0.012)
    tile = bandNoise(rng, 0.01, 3200, 2.0) * decay(0.01, 0.002)
    toe = normalise(onePoleLowPass(whiteNoise(rng, 0.04), 1600)) * hitEnvelope(0.04, 0.002, 0.008)
    addAt(mix, heel, 0.0, 0.8)
    addAt(mix, scuff, 0.0, 0.6)
    addAt(mix, tile, 0.0, 0.15)
    addAt(mix, toe, 0.05 if variant == 1 else 0.045, 0.35)
    if variant == 2:
        squeak = sine(expCurve(0.03, 1850, 2050), 0.03) * adsr(0.03, 0.005, 0.01, 0.6, 0.01)
        addAt(mix, squeak, 0.055, 0.05)
    return mix


def shopBellStrike(rng, freq, seconds):
    """Small brass shop bell: 5 inharmonic modes with slightly random strengths
    (the clapper hits a different spot each swing) and a slow beating detune."""
    ratios = [1.0, 2.02, 2.78, 3.94, 5.21]
    decays = [1.1, 0.75, 0.55, 0.38, 0.25]
    baseAmps = [1.0, 0.55, 0.45, 0.3, 0.2]
    freqs = []
    detuned = []
    amps = []
    for i in range(5):
        freqs.append(freq * ratios[i])
        detuned.append(freq * ratios[i] * 1.002)
        amps.append(baseAmps[i] * rng.uniform(0.7, 1.3))
    ring = modes(seconds, freqs, decays, amps) + 0.5 * modes(seconds, detuned, decays, amps)
    return ring * hitEnvelope(seconds, 0.0008, 1e9)


def makeDoorChime(rng):
    """Shop door bells: two small bells jingling as the door swings open."""
    mix = silence(1.9)
    strikes = [(0.0, 1.0, 1480), (0.11, 0.7, 1975), (0.2, 0.6, 1480), (0.34, 0.35, 1975), (0.5, 0.15, 1480)]
    for when, gain, freq in strikes:
        addAt(mix, shopBellStrike(rng, freq, 1.4), when, gain)
    return reverb(mix, 0.35, 0.2, 0.6)


def makeLocked(rng):
    """Locked stall: two low 'tk-tk' rattles of a latch."""
    length = 0.22
    mix = silence(length)
    for when, gain in ((0.0, 1.0), (0.075, 0.7)):
        rattle = modes(0.08, [310, 760, 1480], [0.03, 0.018, 0.01], [1.0, 0.6, 0.35])
        click = normalise(onePoleLowPass(whiteNoise(rng, 0.006), 3000)) * decay(0.006, 0.0015)
        addAt(mix, rattle * hitEnvelope(0.08, 0.0008, 1e9), when, gain)
        addAt(mix, click, when, 0.5 * gain)
    return lowPass(mix, 3000)


def makeLanternLight(rng):
    """Lighting a sky lantern: match strike (rough granular rasp), the ignition
    'pff' with a few crackles, then the flame catching (soft low-pass whoosh)."""
    length = 1.15
    mix = silence(length)
    strikeLength = 0.18
    rasp = normalise(sweepFilter(whiteNoise(rng, strikeLength), 3200, 1.2, "band"))
    rasp = rasp * (0.3 + 0.7 * np.abs(smoothRandom(rng, strikeLength, 220)))
    rasp = rasp * curve(strikeLength, [0, 0.03, 0.15, strikeLength], [0, 0.7, 1, 0.3])
    ignite = normalise(onePoleLowPass(whiteNoise(rng, 0.12), 3000)) * hitEnvelope(0.12, 0.002, 0.03)
    crackles = crackleTrain(rng, 0.3, 40, [(3000, 4), (5000, 5)]) * curve(0.3, [0, 0.3], [1, 0])
    flameLength = 0.9
    cutoff = curve(flameLength, [0, 0.2, flameLength], [300, 1500, 500])
    flame = normalise(sweepFilter(whiteNoise(rng, flameLength), cutoff, 0.9, "low"))
    flame = flame * curve(flameLength, [0, 0.15, 0.4, flameLength], [0, 1, 0.6, 0])
    addAt(mix, rasp, 0.0, 0.55)
    addAt(mix, ignite, 0.15, 0.8)
    addAt(mix, crackles, 0.15, 0.25)
    addAt(mix, flame, 0.17, 0.7)
    return reverb(mix, 0.5, 0.12, 0.5)


def makeChime(rng):
    """Kalimba note at EXACTLY C5 (523.25 Hz) - the finale plays it at other
    speeds for a pentatonic melody. A kalimba tine is a clamped bar, so its
    first overtone is ~6.27x the fundamental (the metallic 'tink'); a
    Karplus-Strong pluck at the same pitch adds a woody attack."""
    length = 1.8
    f = 523.25
    tine = modes(length, [f, f * 2.0, f * 6.27], [0.9, 0.25, 0.07], [1.0, 0.12, 0.3])
    tine = tine * hitEnvelope(length, 0.0015, 1e9)
    pluck = karplusStrong(rng, f, length, 0.4, 0.996) * hitEnvelope(length, 0.001, 1e9)
    thumb = bandNoise(rng, 0.01, 3000, 1.0) * decay(0.01, 0.002)
    mix = tine + 0.35 * pluck
    addAt(mix, thumb, 0.0, 0.2)
    return reverb(lowPass(mix, 6000), 0.3, 0.22, 0.6)


def makeFireworkLaunch(rng):
    """Rocket: mortar thump, then a rising whistle (sine glide 850 -> 2500 Hz with
    vibrato + airy noise tracking the pitch) and a sparkler hiss trail."""
    length = 1.25
    mix = silence(length)
    mortar = sine(expCurve(0.15, 95, 45), 0.15) * hitEnvelope(0.15, 0.002, 0.04)
    puff = normalise(onePoleLowPass(whiteNoise(rng, 0.05), 1500)) * decay(0.05, 0.01)
    addAt(mix, mortar, 0.0, 0.8)
    addAt(mix, puff, 0.0, 0.4)
    whistleLength = 1.15
    tw = timeline(whistleLength)
    pitch = 850.0 + (2500.0 - 850.0) * (tw / whistleLength) ** 1.6
    pitch = pitch * (1.0 + 0.015 * np.sin(2 * np.pi * 9.0 * tw))
    whistle = sine(pitch, whistleLength) + 0.25 * sine(pitch * 2.0, whistleLength)
    airy = normalise(sweepFilter(whiteNoise(rng, whistleLength), pitch, 10.0, "band"))
    envelope = curve(whistleLength, [0, 0.1, 0.8, whistleLength], [0, 1, 0.8, 0])
    trail = normalise(highPass(whiteNoise(rng, whistleLength), 4000)) * envelope
    addAt(mix, (0.5 * whistle + 0.5 * airy) * envelope, 0.05, 0.7)
    addAt(mix, trail, 0.05, 0.08)
    return reverb(mix, 0.8, 0.2, 0.6, 4000)


def makeFireworkBoom(rng, variant):
    """Firework burst: a sharp crack, a deep pitch-dropping boom and a noise
    body, saturated for weight, then delayed darker 'city echoes' and a big
    outdoor reverb. Variant 2 is deeper and sparkles afterwards."""
    length = 2.8
    mix = silence(length)
    boomFreq = [70.0, 58.0][variant - 1]
    crack = whiteNoise(rng, 0.01) * hitEnvelope(0.01, 0.0003, 0.002)
    boom = sine(expCurve(0.9, boomFreq, boomFreq * 0.52), 0.9) * hitEnvelope(0.9, 0.003, 0.28)
    body = brownNoise(rng, 1.4) * hitEnvelope(1.4, 0.003, 0.35)
    punch = normalise(onePoleLowPass(whiteNoise(rng, 0.5), 1200)) * hitEnvelope(0.5, 0.002, 0.08)
    burst = silence(1.4)
    addAt(burst, crack, 0.0, 0.6)
    addAt(burst, boom, 0.0, 1.0)
    addAt(burst, body, 0.0, 0.8)
    addAt(burst, punch, 0.0, 0.5)
    burst = softClip(burst, 1.6)
    addAt(mix, burst, 0.0, 1.0)
    darkBurst = onePoleLowPass(burst, 900)
    for when, gain in ((0.23, 0.35), (0.52, 0.22), (0.9, 0.12)):
        addAt(mix, darkBurst, when, gain)
    if variant == 2:
        t = timeline(1.5)
        sparkle = crackleTrain(rng, 1.5, 160, [(3500, 4), (5500, 5), (8000, 6)], np.exp(-t / 0.6))
        addAt(mix, sparkle, 0.35, 0.3)
    return reverb(mix, 0.9, 0.3, 1.2, 2500)


def makeCrackle(rng):
    """Sparkle crackle (like a 'willow' firework): Poisson pops in four
    frequency bands whose density swells and dies away, over a faint fizz."""
    length = 1.6
    density = curve(length, [0, 0.1, 0.4, 1.0, 1.5], [0, 0.6, 1.0, 0.35, 0])
    pops = crackleTrain(rng, length, 260, [(2400, 3), (3800, 4), (5600, 5), (8200, 6)], density)
    fizz = normalise(highPass(whiteNoise(rng, length), 5000)) * density
    return reverb(pops + 0.08 * fizz, 0.6, 0.15, 0.5)


# ======================================================================
# 15. VOICE LINES (Windows text-to-speech + NumPy processing)
# ======================================================================
# Each style = which Windows voice to use, SSML prosody (pitch/rate) and my
# own post-processing settings.
VOICE_STYLES = {
    # Energetic game-show announcer (Zira): higher and faster, bright EQ,
    # compressed and gently saturated for punch, small hall + slapback echo.
    "announcer": {"voice": "Microsoft Zira Desktop", "pitch": "+20%", "rate": "+10%",
                  "speed": 1.03, "highPass": 120, "lowShelfDb": -1.5, "presenceDb": 4.0,
                  "drive": 1.6, "roomSize": 0.45, "wet": 0.14, "slapback": 0.09},
    # Dato' Leon Tan (David): slightly lower and slower, warm low end,
    # a calm dining-hall room.
    "leon": {"voice": "Microsoft David Desktop", "pitch": "-6%", "rate": "-14%",
             "speed": 0.95, "highPass": 70, "lowShelfDb": 2.5, "presenceDb": 1.5,
             "drive": 1.0, "roomSize": 0.35, "wet": 0.16, "slapback": 0.0},
}

# (file name, style, text). The text can hold SSML tags. "Wok hei" is written
# the way it sounds ("wok hay") so the English voice says it correctly.
VOICE_LINES = [
    ("vo_ready", "announcer", "Ready?"),
    ("vo_cook", "announcer", "Let's cook!"),
    ("vo_perfect", "announcer", "Perfect!"),
    ("vo_wokhei", "announcer", "Wok hay!"),
    ("vo_aura", "announcer", "Aura mode!"),
    ("vo_timesup", "announcer", "Time's up!"),
    ("vo_delicious", "announcer", "Delicious!"),
    ("vo_leon_hmm", "leon", '<prosody rate="-35%" pitch="-4%">Hmmm,</prosody> interesting.'),
    ("vo_leon_golden", "leon", "Magnificent. This stall has real heart."),
    ("vo_leon_silver", "leon", "Good. Very good. Come back stronger."),
    ("vo_leon_bronze", "leon", "Not yet. But I can taste the heart."),
]


def buildSsml(text, style):
    """Wraps the text in SSML so the TTS engine uses my pitch and speed."""
    return ('<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-US">'
            f'<prosody pitch="{style["pitch"]}" rate="{style["rate"]}">{text}</prosody></speak>')


def renderTtsLines(lines):
    """Writes a small PowerShell script that renders every line with
    System.Speech into TTS_RAW_DIR (44.1 kHz, 16-bit, mono). Returns True if
    PowerShell finished without an error."""
    os.makedirs(TTS_RAW_DIR, exist_ok=True)
    script = [
        "Add-Type -AssemblyName System.Speech",
        "$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer",
        "$format = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(44100, "
        "[System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, "
        "[System.Speech.AudioFormat.AudioChannel]::Mono)",
    ]
    for name, styleName, text in lines:
        style = VOICE_STYLES[styleName]
        ssmlPath = os.path.join(TTS_RAW_DIR, name + ".ssml")
        wavPath = os.path.join(TTS_RAW_DIR, name + "_raw.wav")
        with open(ssmlPath, "w", encoding="utf-8") as file:
            file.write(buildSsml(text, style))
        script.append(f'$synth.SelectVoice("{style["voice"]}")')
        script.append(f'$synth.SetOutputToWaveFile("{wavPath}", $format)')
        script.append(f'$synth.SpeakSsml([System.IO.File]::ReadAllText("{ssmlPath}"))')
    script.append("$synth.SetOutputToNull()")
    script.append("$synth.Dispose()")
    scriptPath = os.path.join(TTS_RAW_DIR, "render_tts.ps1")
    with open(scriptPath, "w", encoding="utf-8") as file:
        file.write("\n".join(script))
    try:
        result = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
                                 "-File", scriptPath], capture_output=True, text=True, timeout=180)
    except (OSError, subprocess.TimeoutExpired) as error:
        print("  TTS could not run:", error)
        return False
    if result.returncode != 0:
        print("  TTS error:", result.stderr.strip()[:400])
        return False
    return True


def readWav(path):
    """Reads a 16-bit WAV (mono or stereo) into floats between -1 and 1."""
    wf = wave.open(path, "rb")
    channels = wf.getnchannels()
    rate = wf.getframerate()
    data = np.frombuffer(wf.readframes(wf.getnframes()), dtype="<i2").astype(float) / 32768.0
    wf.close()
    if channels == 2:
        data = data.reshape(-1, 2).mean(axis=1)
    if rate != SAMPLE_RATE:
        data = resample(data, rate / SAMPLE_RATE)
    return data


def frameLevels(signal, frameSeconds=0.01):
    """Peak level of every 10 ms frame (used to find silence)."""
    size = samples(frameSeconds)
    count = len(signal) // size
    frames = np.abs(signal[:count * size]).reshape(count, size)
    return frames.max(axis=1), size


def shortenPauses(signal, maxPause=0.3, thresholdDb=-40.0):
    """Cuts long silent gaps inside a sentence down to 'maxPause' seconds."""
    levels, size = frameLevels(signal)
    quiet = levels < np.max(levels) * dbToGain(thresholdDb)
    keepFrames = int(maxPause / 0.01)
    pieces = []
    runStart = None
    lastEnd = 0
    for i in range(len(quiet) + 1):
        isQuiet = i < len(quiet) and quiet[i]
        if isQuiet and runStart is None:
            runStart = i
        if not isQuiet and runStart is not None:
            if i - runStart > keepFrames and runStart > 0 and i < len(quiet):
                cutFrom = (runStart + keepFrames // 2) * size
                cutTo = (i - keepFrames // 2) * size
                pieces.append(signal[lastEnd:cutFrom])
                lastEnd = cutTo
            runStart = None
    pieces.append(signal[lastEnd:])
    return np.concatenate(pieces)


def trimSilence(signal, thresholdDb=-60.0, keepStart=0.002, keepEnd=0.01):
    """Removes silence at the start and end (relative to the loudest sample)."""
    threshold = np.max(np.abs(signal)) * dbToGain(thresholdDb)
    loud = np.where(np.abs(signal) > threshold)[0]
    if len(loud) == 0:
        return signal
    start = max(0, loud[0] - samples(keepStart))
    end = min(len(signal), loud[-1] + samples(keepEnd))
    return signal[start:end]


def compressor(signal, thresholdDb=-20.0, ratio=3.0, attack=0.004, release=0.09):
    """Dynamic range compressor: an envelope follower measures the level and
    turns loud parts down, so every syllable is clear and punchy."""
    threshold = dbToGain(thresholdDb)
    attackCoef = math.exp(-1.0 / (attack * SAMPLE_RATE))
    releaseCoef = math.exp(-1.0 / (release * SAMPLE_RATE))
    values = signal.tolist()
    level = 0.0
    for i in range(len(values)):
        size = abs(values[i])
        if size > level:
            level = attackCoef * level + (1.0 - attackCoef) * size
        else:
            level = releaseCoef * level + (1.0 - releaseCoef) * size
        if level > threshold:
            values[i] *= (level / threshold) ** (1.0 / ratio - 1.0)
    return np.array(values)


def processVoice(raw, style):
    """Clean-up and character chain for one TTS line."""
    voice = highPass(raw, style["highPass"])
    voice = shortenPauses(voice, 0.3)
    voice = trimSilence(voice, -50.0, 0.015, 0.06)
    voice = resample(voice, style["speed"])            # pitch/formant character
    voice = biquad(voice, "lowshelf", 220, 0.7, style["lowShelfDb"])
    voice = biquad(voice, "peak", 3200, 0.9, style["presenceDb"])
    voice = lowPass(voice, 10000)
    voice = compressor(normalise(voice), -20.0, 3.5)
    if style["drive"] > 1.0:
        voice = softClip(normalise(voice, 0.9), style["drive"])
    if style["slapback"] > 0.0:
        echo = np.concatenate([np.zeros(samples(style["slapback"])), lowPass(voice, 3000)])
        voice = np.concatenate([voice, np.zeros(len(echo) - len(voice))]) + dbToGain(-14.0) * echo
    return reverb(voice, style["roomSize"], style["wet"], 0.6, 5000)


def makeAnnouncerFallback(rng):
    """Used only if Windows TTS is not available: a short bright two-note chime."""
    mix = silence(0.8)
    addAt(mix, chimeNote(783.99, 0.5), 0.0, 0.8)
    addAt(mix, chimeNote(1174.66, 0.6), 0.1, 1.0)
    return reverb(mix, 0.3, 0.2, 0.4)


# ======================================================================
# 16. FINISHING AND SAVING
# ======================================================================
def finishSound(signal, loop=False):
    """Final clean-up for every file:
    one-shots: 20 Hz high-pass (DC removal), trim silence, fade in/out;
    loops: only remove the average (DC) - a fade would break the seam;
    then peak normalise to PEAK_DB."""
    sound = np.asarray(signal, dtype=float)
    if loop:
        sound = sound - np.mean(sound)
    else:
        sound = onePoleHighPass(sound, 20.0)
        sound = trimSilence(sound, -60.0, 0.001, 0.01)
        fadeOut = min(0.06, max(0.008, len(sound) / SAMPLE_RATE * 0.08))
        sound = fadeEdges(sound, 0.0015, fadeOut)
    return normalise(sound, dbToGain(PEAK_DB))


def writeWav(path, signal):
    """Saves 16-bit mono PCM with the 'wave' module (as in the Week 2 tutorial).
    TPDF dither (two random numbers added together, +-1 LSB) turns the
    rounding error into a tiny smooth noise instead of distortion."""
    ditherRng = np.random.default_rng(zlib.crc32(os.path.basename(path).encode()))
    count = len(signal)
    dither = ditherRng.uniform(-0.5, 0.5, count) + ditherRng.uniform(-0.5, 0.5, count)
    pcm = np.clip(np.round(signal * 32767.0 + dither), -32768, 32767).astype("<i2")
    wf = wave.open(path, "wb")
    wf.setnchannels(1)                 # mono, so the game can pan it left/right
    wf.setsampwidth(2)                 # 2 bytes = 16-bit
    wf.setframerate(SAMPLE_RATE)
    wf.writeframes(pcm.tobytes())
    wf.close()


def saveSound(folder, name, signal, loop=False):
    finished = finishSound(signal, loop)
    path = os.path.join(folder, name + ".wav")
    writeWav(path, finished)
    peak = gainToDb(np.max(np.abs(finished)))
    rms = gainToDb(math.sqrt(np.mean(finished ** 2)))
    label = "loop" if loop else ""
    print(f"  saved {os.path.relpath(path, PROJECT_DIR):38s} {len(finished) / SAMPLE_RATE:5.2f} s  "
          f"peak {peak:5.1f} dBFS  rms {rms:6.1f} dBFS {label}")


def newRandom(name):
    """Every sound gets its own random generator seeded from its name, so the
    script always produces exactly the same files (deterministic)."""
    return np.random.default_rng(zlib.crc32(name.encode("utf-8")))


# (file name, function, extra argument or None)
SFX_LIST = [
    # UI
    ("ui_hover", makeUiHover, None), ("ui_click", makeUiClick, None),
    ("ui_confirm", makeUiConfirm, None), ("ui_back", makeUiBack, None),
    ("ui_pause", makeUiPause, None), ("countdown_beep", makeCountdownBeep, None),
    ("go_whistle", makeGoWhistle, None), ("score_tick", makeScoreTick, None),
    ("star_1", makeStar, 1), ("star_2", makeStar, 2), ("star_3", makeStar, 3),
    ("result_stamp", makeResultStamp, None), ("fanfare", makeFanfare, None),
    ("applause", makeApplause, None), ("aura_mode", makeAuraMode, None),
    ("perfect", makePerfect, None), ("good", makeGood, None), ("miss", makeMiss, None),
    ("buzzer_wrong", makeBuzzerWrong, None), ("popup", makePopup, None),
    # dialogue voice blips
    ("voice_aiman", makeVoiceAiman, None), ("voice_nenek", makeVoiceNenek, None),
    ("voice_leon", makeVoiceLeon, None), ("voice_lim", makeVoiceLim, None),
    ("voice_kavitha", makeVoiceKavitha, None), ("voice_farah", makeVoiceFarah, None),
    ("voice_oyen", makeVoiceOyen, None),
    # Scene 1 - Rojak Rush
    ("chop_1", makeChop, 1), ("chop_2", makeChop, 2), ("chop_3", makeChop, 3),
    ("knife_whoosh", makeKnifeWhoosh, None), ("juice_squish", makeJuiceSquish, None),
    ("board_thud", makeBoardThud, None), ("plop", makePlop, None),
    ("cat_meow", makeCatMeow, None), ("cat_hiss", makeCatHiss, None),
    ("spicy_sizzle", makeSpicySizzle, None), ("sauce_pour", makeSaucePour, None),
    ("sprinkle", makeSprinkle, None),
    # Scene 2 - Wok Hei Fire
    ("sizzle_loop", makeSizzleLoop, None), ("fire_loop", makeFireLoop, None),
    ("sizzle_burst", makeSizzleBurst, None), ("wok_clang", makeWokClang, None),
    ("flare_whoosh", makeFlareWhoosh, None), ("smoke_alarm", makeSmokeAlarm, None),
    ("steam_hiss", makeSteamHiss, None), ("plate_slide", makePlateSlide, None),
    ("knob_click", makeKnobClick, None),
    # Scene 3 - The Great Pull
    ("pour_loop", makePourLoop, None), ("splash_1", makeSplash, 1), ("splash_2", makeSplash, 2),
    ("bubble_pop", makeBubblePop, None), ("glass_clink", makeGlassClink, None),
    ("slurp", makeSlurp, None), ("neon_buzz", makeNeonBuzz, None),
    ("webcam_on", makeWebcamOn, None),
    # Scene 4 - Lazy Susan Banquet
    ("susan_rumble", makeSusanRumble, None), ("dish_serve", makeDishServe, None),
    ("nom", makeNom, None), ("happy_chime", makeHappyChime, None),
    ("grumble", makeGrumble, None), ("camera_shutter", makeCameraShutter, None),
    ("wobble", makeWobble, None),
    # Street and finale
    ("footstep_1", makeFootstep, 1), ("footstep_2", makeFootstep, 2),
    ("door_chime", makeDoorChime, None), ("locked", makeLocked, None),
    ("lantern_light", makeLanternLight, None), ("chime", makeChime, None),
    ("firework_launch", makeFireworkLaunch, None),
    ("firework_boom_1", makeFireworkBoom, 1), ("firework_boom_2", makeFireworkBoom, 2),
    ("crackle", makeCrackle, None),
]

# Seamless loops: no fades or trimming (they would break the seam)
LOOP_NAMES = ["sizzle_loop", "fire_loop", "pour_loop", "neon_buzz", "susan_rumble"]


def makeAllSfx(onlyNames):
    os.makedirs(SFX_DIR, exist_ok=True)
    for name, function, extra in SFX_LIST:
        if onlyNames and name not in onlyNames:
            continue
        rng = newRandom(name)
        if extra is None:
            signal = function(rng)
        else:
            signal = function(rng, extra)
        saveSound(SFX_DIR, name, signal, name in LOOP_NAMES)


def makeAllVoices(onlyNames):
    os.makedirs(VOICE_DIR, exist_ok=True)
    lines = []
    for line in VOICE_LINES:
        if not onlyNames or line[0] in onlyNames:
            lines.append(line)
    if not lines:
        return
    print(f"  rendering {len(lines)} TTS lines into {TTS_RAW_DIR}")
    ttsWorked = renderTtsLines(lines)
    for name, styleName, text in lines:
        rawPath = os.path.join(TTS_RAW_DIR, name + "_raw.wav")
        if ttsWorked and os.path.exists(rawPath):
            signal = processVoice(readWav(rawPath), VOICE_STYLES[styleName])
        else:
            print(f"  WARNING: no TTS for {name}, using a synthesised chime instead")
            signal = makeAnnouncerFallback(newRandom(name))
        saveSound(VOICE_DIR, name, signal)


def main():
    startTime = time.time()
    arguments = sys.argv[1:]
    skipTts = "--no-tts" in arguments
    onlyNames = []
    for argument in arguments:
        if not argument.startswith("--"):
            onlyNames.append(argument)
    print("Midnight Makan - generating sound effects")
    makeAllSfx(onlyNames)
    if not skipTts:
        print("Midnight Makan - generating voice lines")
        makeAllVoices(onlyNames)
    print(f"Done in {time.time() - startTime:.1f} s")


if __name__ == "__main__":
    main()
