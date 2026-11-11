"""
recorder.py
-----------
FrameRecorder saves the demo video of the game.

Every drawn frame is read back from the screen with arcade.get_image() and
its raw RGBA bytes are sent (piped) into ffmpeg, which encodes them as an
H.264 MP4 file. Because main.py advances the game by exactly 1 / RECORD_FPS
seconds for every drawn frame, the video plays at the correct speed even if
the computer is too slow to record in real time.

The framebuffer can be bigger than 1280x720 (Windows display scaling of
125% gives 1600x900), so ffmpeg scales every frame back to the video size.

If ffmpeg is not installed the recorder falls back to saving numbered PNG
frames in a folder next to the video path.
"""
import os
import shutil
import subprocess

import arcade


class FrameRecorder:
    def __init__(self, outPath, fps=30, size=(1280, 720)):
        self.outPath = os.path.abspath(outPath)
        self.fps = fps
        self.size = size
        self.process = None
        self.inputSize = None        # framebuffer size, known after the first frame
        self.frameCount = 0
        self.closed = False
        self.failed = False
        self.frameFolder = None

        folder = os.path.dirname(self.outPath)
        os.makedirs(folder, exist_ok=True)
        self.ffmpegPath = shutil.which("ffmpeg")
        if self.ffmpegPath is None:
            self.frameFolder = os.path.splitext(self.outPath)[0] + "_frames"
            os.makedirs(self.frameFolder, exist_ok=True)
            print(f"[recorder] WARNING: ffmpeg not found - saving PNG frames to {self.frameFolder}")

    def startFfmpeg(self, width, height):
        """Start ffmpeg reading raw RGBA frames of width x height from its standard input."""
        command = [
            self.ffmpegPath, "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{width}x{height}", "-r", str(self.fps),
            "-i", "-",
            "-vf", f"scale={self.size[0]}:{self.size[1]}:flags=lanczos",
            "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
            "-r", str(self.fps),
            self.outPath,
        ]
        try:
            self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                            stderr=subprocess.PIPE)
            self.inputSize = (width, height)
            print(f"[recorder] Recording {width}x{height} frames -> {self.size[0]}x{self.size[1]} "
                  f"at {self.fps} fps: {self.outPath}")
        except Exception as error:
            print(f"[recorder] WARNING: could not start ffmpeg: {error}")
            self.failed = True

    def capture(self):
        """Grab the frame that was just drawn and add it to the video."""
        if self.closed or self.failed:
            return
        image = arcade.get_image()

        if self.frameFolder is not None:
            # Fallback without ffmpeg: numbered PNG files
            if image.size != self.size:
                image = image.resize(self.size)
            image.convert("RGB").save(os.path.join(self.frameFolder, f"frame_{self.frameCount:05d}.png"))
            self.frameCount += 1
            return

        if self.process is None:
            self.startFfmpeg(image.width, image.height)
            if self.failed:
                return
        if image.size != self.inputSize:
            # Should not happen (the window cannot be resized), but keep ffmpeg's input size fixed
            image = image.resize(self.inputSize)
        try:
            self.process.stdin.write(image.tobytes())
            self.frameCount += 1
        except Exception as error:
            print(f"[recorder] WARNING: ffmpeg stopped accepting frames: {error}")
            self.failed = True

    def close(self):
        """Finish the video file (ffmpeg writes the MP4 header when its input ends)."""
        if self.closed:
            return
        self.closed = True
        if self.process is not None:
            errorText = b""
            try:
                self.process.stdin.close()
            except Exception:
                pass
            try:
                errorText = self.process.stderr.read()
                self.process.wait(timeout=180)
            except Exception as error:
                print(f"[recorder] WARNING: waiting for ffmpeg failed: {error}")
            if self.process.returncode == 0:
                seconds = self.frameCount / float(self.fps)
                print(f"[recorder] Saved {self.frameCount} frames ({seconds:.1f} s) to {self.outPath}")
            else:
                print(f"[recorder] WARNING: ffmpeg exit code {self.process.returncode}")
                if errorText:
                    print(errorText.decode("utf-8", "replace")[-800:])
        elif self.frameFolder is not None:
            print(f"[recorder] Saved {self.frameCount} PNG frames to {self.frameFolder}")
