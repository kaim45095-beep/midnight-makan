"""
record_demo.py
--------------
Records the demo video of Midnight Makan in three steps:

  1. run the game with the autoplay bot in record mode: every drawn frame
     is saved into a video (core/recorder.py pipes the frames into ffmpeg)
     and every sound event is written into a JSON log
  2. mix the soundtrack offline from that log (tools/mix_audio.py)
  3. join video + audio into the final MP4 with ffmpeg

Because the game advances exactly 1/30 s for every recorded frame, the
video and the audio line up perfectly, even on a slow computer.

Usage (from the project folder):
    python tools/record_demo.py --out demo.mp4 [--scene title] [--seed 7] [--keep-temp]
                                [--seconds N] [--fake-camera]
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(TOOLS_DIR)
sys.path.insert(0, TOOLS_DIR)

import mix_audio  # noqa: E402  (after the path change on purpose)


def videoDuration(path):
    """Ask ffprobe how long the video is (in seconds)."""
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", path],
                            capture_output=True, text=True)
    return float(json.loads(result.stdout)["format"]["duration"])


def main():
    parser = argparse.ArgumentParser(description="Record the Midnight Makan demo video")
    parser.add_argument("--out", default=os.path.join(PROJECT_DIR, "demo", "MidnightMakan_demo.mp4"))
    parser.add_argument("--scene", default="title")
    parser.add_argument("--seed", default="7")
    parser.add_argument("--keep-temp", action="store_true")
    parser.add_argument("--seconds", default=None, help="stop after N seconds of game time")
    parser.add_argument("--fake-camera", action="store_true",
                        help="record Scene 3 webcam mode with the simulated camera")
    options = parser.parse_args()

    if shutil.which("ffmpeg") is None:
        print("ffmpeg was not found on the PATH - please install it first.")
        sys.exit(1)

    outPath = os.path.abspath(options.out)
    tempFolder = os.path.splitext(outPath)[0] + "_temp"
    os.makedirs(tempFolder, exist_ok=True)
    rawVideo = os.path.join(tempFolder, "video_only.mp4")
    audioLog = os.path.join(tempFolder, "audio_log.json")
    mixedAudio = os.path.join(tempFolder, "soundtrack.wav")

    # 1) play + record (muted: the sound is rebuilt offline, so nothing plays through the speakers)
    print("[demo] 1/3 recording the game (a window opens and plays by itself)...")
    command = [sys.executable, os.path.join(PROJECT_DIR, "main.py"), "--autoplay", "--mute",
               "--scene", options.scene, "--seed", options.seed,
               "--record", rawVideo, "--audio-log", audioLog]
    if options.seconds is not None:
        command += ["--seconds", options.seconds]
    if options.fake_camera:
        command.append("--fake-camera")
    subprocess.run(command, cwd=PROJECT_DIR, check=True)
    duration = videoDuration(rawVideo)
    print(f"[demo] video length {duration:.1f} s")

    # 2) mix the soundtrack
    print("[demo] 2/3 mixing the soundtrack...")
    mix_audio.mix(audioLog, mixedAudio, duration)

    # 3) join video and audio
    print("[demo] 3/3 joining video and audio...")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", rawVideo, "-i", mixedAudio,
                    "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                    "-movflags", "+faststart", outPath], check=True)
    print(f"[demo] done: {outPath} ({videoDuration(outPath):.1f} s)")

    if not options.keep_temp:
        shutil.rmtree(tempFolder, ignore_errors=True)


if __name__ == "__main__":
    main()
