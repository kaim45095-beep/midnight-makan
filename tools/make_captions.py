"""
make_captions.py
----------------
Post-production for the demo video: adds a caption bar under the picture
that names the scene and explains each special effect when it happens.

The caption times are not typed by hand. They come from two logs that the
recording already makes:
  * the console log of record_demo.py ("scene -> rojak" lines), for scene changes
  * the SoundManager's audio_log.json, for effects (a sound plays with every effect)

Usage (from the project folder, needs ffmpeg):
    python tools/record_demo.py --out demo_raw.mp4 --keep-temp > record_log.txt
    python tools/make_captions.py demo_raw.mp4 demo_raw_temp/audio_log.json record_log.txt demo.mp4
"""
import json
import os
import re
import shutil
import subprocess
import sys

FONT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets", "fonts")

if len(sys.argv) < 5:
    sys.exit("usage: python tools/make_captions.py RAW.mp4 AUDIO_LOG.json GAME_LOG.txt OUT.mp4")
raw, audioLog, gameLog, out = sys.argv[1:5]
work = os.path.join(os.path.dirname(os.path.abspath(out)), "captions_work")
os.makedirs(work, exist_ok=True)
shutil.copy(os.path.join(FONT_DIR, "Liberation_Sans_Bold.ttf"), os.path.join(work, "bold.ttf"))
shutil.copy(os.path.join(FONT_DIR, "Liberation_Sans_Regular.ttf"), os.path.join(work, "regular.ttf"))

# scene segments from the game log
segments = []
for line in open(gameLog, encoding="utf-8", errors="ignore"):
    match = re.search(r"clock\s+([\d.]+)s\s+scene -> (\w+)", line)
    if match:
        segments.append((float(match.group(1)), match.group(2)))
probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", raw],
                       capture_output=True, text=True)
duration = float(json.loads(probe.stdout)["format"]["duration"])
ranges = []
for i, (start, name) in enumerate(segments):
    end = segments[i + 1][0] if i + 1 < len(segments) else duration
    ranges.append((start, end, name))

events = [(event["t"], event["name"]) for event in json.load(open(audioLog)) if event["type"] == "play"]


def first(namePrefix, scene, after=0.0, nth=1):
    """Time of the nth sound whose name starts with namePrefix, inside the given scene."""
    count = 0
    for start, end, sceneName in ranges:
        if sceneName != scene:
            continue
        for t, name in events:
            if start <= t < end and t >= start + after and name.startswith(namePrefix):
                count += 1
                if count == nth:
                    return t
    return None


def sceneStart(scene, nth=1):
    count = 0
    for start, end, sceneName in ranges:
        if sceneName == scene:
            count += 1
            if count == nth:
                return start
    return None


wanted = [
    (0.6, "Title screen: twinkling star sprites and rising sky lanterns with additive glow"),
    (7.8, "Menus: mouse hover and keyboard events; thumbnails composited with Pillow"),
    ((sceneStart("story") or 0) + 1.0, "Story intro: comic panels with a Ken Burns effect (interpolated zoom and pan)"),
    ((sceneStart("street") or 0) + 1.0, "Street hub: 8-frame sprite-sheet walk, parallax layers, swinging lantern lights"),
    (first("go_whistle", "rojak"), "Scene 1: press SPACE or click when the cleaver meets a dotted line (distance check)"),
    (first("chop_", "rojak"), "Each cut: Pillow crop() slices, juice particles, stain decals, chop panned by x"),
    (first("cat_meow", "rojak", 1.0), "Obstacle: Oyen sneaks in - clicking him is a mouse hit-test event"),
    (first("aura_mode", "rojak"), "5 good actions in a row: Aura Mode - double points, golden frame, music layer"),
    (first("sauce_pour", "rojak"), "Food transformation: sauce stream and peanut particles, then the dish reveal"),
    (first("go_whistle", "wok"), "Scene 2: drag bowls in ticket order, UP/DOWN sets the gas, SPACE tosses the wok"),
    (first("buzzer_wrong", "wok"), "Wrong ingredient: buzzer, the combo breaks, the bowl tweens back home"),
    (first("flare_whoosh", "wok"), "Toss above 80% heat: wok-hei flare - fire particles, glow burst, screen shake"),
    (first("smoke_alarm", "wok"), "Too hot for 2.5 s: food burns - darker texture (ImageEnhance), smoke, alarm"),
    (first("plate_slide", "wok"), "Plating: the food bits turn into the finished dish (food transformation)"),
    (first("go_whistle", "tehtarik"), "Scene 3: hold to pour condensed milk - the drink colour is an RGB interpolation"),
    (first("pour_loop", "tehtarik", 0.0, 2), "The pull: each droplet is a gravity particle; the pour pitch rises as it fills"),
    (first("splash_", "tehtarik"), "Missed droplets splash on the counter and leave stain decals"),
    (first("glass_clink", "tehtarik"), "Serve: score = fill + foam + colour - mess, and the customer reacts"),
    (first("go_whistle", "banquet"), "Scene 4: spin the lazy susan - dish = translate x scale(1, 0.35) x rotate x (r, 0)"),
    (first("cat_meow", "banquet"), "Oyen rides the turntable with the same rotation maths - click to shoo him"),
    (first("camera_shutter", "banquet"), "Farah is served: a camera-flash post effect and shutter sound"),
    (first("splash_1", "banquet"), "Spinning too fast: the dishes wobble, the teh tarik spills, the rumble grows"),
    (first("applause", "banquet"), "The critic has all three courses: the feast is complete"),
    (first("chime", "finale"), "Finale: each lantern plays the next pentatonic note (speed = 2 ^ (n / 12))"),
    (first("firework_launch", "finale"), "Fireworks: parametric heart, star, circle and ring bursts with glitter"),
    (first("camera_shutter", "finale"), "Polaroid: screen capture, RGB colour grading, vignette mask, film grain"),
    (duration - 22.0, "All art, music and sound effects were generated by the Python scripts in tools/"),
]


def captionTime(item):
    return item[0]


# every caption shows for 5 s and never overlaps the one before it
captions = []
lastEnd = -1.0
for t, text in sorted([item for item in wanted if item[0] is not None], key=captionTime):
    start = max(t, lastEnd + 0.3)
    end = min(duration, start + 5.0)
    captions.append((start, end, text))
    lastEnd = end

labels = {"title": "Title and menus", "story": "Story intro", "street": "Pelita Street", "rojak": "Scene 1 - Rojak Rush",
          "wok": "Scene 2 - Wok Hei Fire", "tehtarik": "Scene 3 - The Great Pull", "banquet": "Scene 4 - Lazy Susan",
          "finale": "Finale - Lantern Bloom"}

# ffmpeg filters: grow the frame by 70 px (the caption bar), a gold line, then one drawtext per caption.
# expansion=none stops drawtext from reading "%" in a caption as a special code.
filters = ["pad=1280:790:0:0:color=0x10122a", "drawbox=x=0:y=720:w=1280:h=3:color=0xffc857@0.9:t=fill"]
for i, (start, end, name) in enumerate(ranges):
    open(os.path.join(work, f"label_{i:02d}.txt"), "w", encoding="utf-8").write(labels.get(name, name))
    filters.append(f"drawtext=fontfile=bold.ttf:expansion=none:textfile=label_{i:02d}.txt:fontsize=19:"
                   f"fontcolor=0xffc857:x=18:y=744:enable='between(t,{start:.2f},{end:.2f})'")
for i, (start, end, text) in enumerate(captions):
    open(os.path.join(work, f"cap_{i:02d}.txt"), "w", encoding="utf-8").write(text)
    filters.append(f"drawtext=fontfile=regular.ttf:expansion=none:textfile=cap_{i:02d}.txt:fontsize=20:"
                   f"fontcolor=0xfff4e0:x=262+(1000-text_w)/2:y=744:enable='between(t,{start:.2f},{end:.2f})'")
open(os.path.join(work, "filter.txt"), "w", encoding="utf-8").write(",".join(filters))

command = ["ffmpeg", "-y", "-loglevel", "error", "-i", os.path.abspath(raw), "-/vf", "filter.txt",
           "-c:v", "libx264", "-preset", "medium", "-crf", "19", "-pix_fmt", "yuv420p", "-c:a", "copy",
           "-movflags", "+faststart", os.path.abspath(out)]
try:
    subprocess.run(command, cwd=work, check=True)
except subprocess.CalledProcessError:
    # ffmpeg older than version 7 reads a filter file with -filter_script:v instead
    command[command.index("-/vf")] = "-filter_script:v"
    subprocess.run(command, cwd=work, check=True)
print(f"captions: {len(captions)}")
for start, end, text in captions:
    print(f"  {start:6.1f}-{end:6.1f}  {text}")
print("saved", out)
