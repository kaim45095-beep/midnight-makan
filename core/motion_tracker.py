"""
motion_tracker.py
-----------------
Optional webcam hand tracking for Scene 3 (Webcam Pull Mode).

It uses FRAME DIFFERENCING, a classic motion-detection method from image
processing. For every new camera frame:

  1. mirror the frame (flip) so moving your hand right moves it right on screen
  2. convert it to grayscale (colour is not needed to find motion)
  3. GaussianBlur = low-pass filter that removes camera noise
  4. absdiff with the previous frame: pixels that changed = motion
  5. threshold: changed "enough" -> white (255), otherwise black (0)
  6. dilate (morphology) joins the small white blobs into bigger ones
  7. the largest contour is the moving hand; its centroid comes from the
     image moments: cx = m10 / m00, cy = m01 / m00

The camera is read in a background thread so the game never waits for it.
The game only asks for the latest smoothed hand height (0 = bottom, 1 = top).

The file never crashes the game: if OpenCV is not installed or there is no
camera, start() simply returns False and the scene uses the mouse instead.

SimulatedCamera (at the bottom) is a pretend webcam for testing and for the
recorded demo (python main.py --fake-camera): it draws a waving hand, and the
frames go through exactly the same processFrame() steps. No camera is opened.
"""
import math
import threading
import time

try:
    import cv2
    OPENCV_AVAILABLE = True
except Exception:
    cv2 = None
    OPENCV_AVAILABLE = False

import numpy
from PIL import Image

BLUR_SIZE = 21          # Gaussian kernel size (must be odd)
DIFF_THRESHOLD = 25     # grey-level change that counts as motion
DILATE_STEPS = 2        # how many times the white areas are grown
MIN_AREA = 400          # smaller moving blobs are ignored (noise)
SMOOTHING = 0.35        # 0..1, how quickly the smoothed height follows new values
HOLD_TIME = 2.0         # seconds the last height is kept after the motion stops
PREVIEW_SIZE = (160, 120)


def processFrame(previousGray, frame):
    """
    One step of frame differencing (a pure function, easy to test without a camera).
    previousGray: the blurred grayscale result of the last call (or None).
    frame: a BGR colour image (or a grayscale image) as a NumPy array from OpenCV.
    Returns (gray, mask, centroid): the new blurred gray image, the white-on-black
    motion mask and the (x, y) centroid of the biggest moving area in pixels
    (y grows DOWNWARDS like in every image), or None if nothing moved.
    """
    # 1. mirror (flip around the vertical axis)
    frame = cv2.flip(frame, 1)
    # 2. grayscale
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
    # 3. noise removal (Gaussian low-pass filter)
    gray = cv2.GaussianBlur(gray, (BLUR_SIZE, BLUR_SIZE), 0)

    if previousGray is None or previousGray.shape != gray.shape:
        mask = gray.copy()
        mask[:] = 0
        return gray, mask, None

    # 4. frame differencing
    difference = cv2.absdiff(previousGray, gray)
    # 5. threshold -> binary mask
    result, mask = cv2.threshold(difference, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
    # 6. morphology: dilate to fill the gaps inside the moving area
    mask = cv2.dilate(mask, None, iterations=DILATE_STEPS)

    # 7. largest contour -> centroid from the image moments
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    biggest = None
    biggestArea = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > biggestArea:
            biggestArea = area
            biggest = contour
    if biggest is None or biggestArea < MIN_AREA:
        return gray, mask, None
    moments = cv2.moments(biggest)
    if moments["m00"] == 0:
        return gray, mask, None
    cx = moments["m10"] / moments["m00"]
    cy = moments["m01"] / moments["m00"]
    return gray, mask, (cx, cy)


class MotionTracker:
    def __init__(self, cameraIndex=0, width=320, height=240):
        self.cameraIndex = cameraIndex
        self.width = width
        self.height = height
        self.capture = None
        self.thread = None
        self.running = False
        self.lock = threading.Lock()     # the thread and the game share the values below
        self.handHeight = None           # smoothed 0..1 (1 = top of the camera view)
        self.lastSeenTime = 0.0
        self.previewImage = None
        self.framesProcessed = 0
        self.previousGray = None         # only used by feedFrame()

    def start(self):
        """Open the camera and start the background thread. Returns False if there is no camera."""
        if self.running:
            return True
        if not OPENCV_AVAILABLE:
            print("[webcam] OpenCV is not installed - webcam mode is off")
            return False
        try:
            if hasattr(cv2, "CAP_DSHOW"):
                # DirectShow opens much faster on Windows
                capture = cv2.VideoCapture(self.cameraIndex, cv2.CAP_DSHOW)
            else:
                capture = cv2.VideoCapture(self.cameraIndex)
            if capture is None or not capture.isOpened():
                print("[webcam] No camera found")
                return False
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        except Exception as error:
            print(f"[webcam] Could not open the camera: {error}")
            return False
        self.capture = capture
        self.running = True
        # daemon=True: the thread never keeps the game open after the window closes
        self.thread = threading.Thread(target=self.captureLoop, daemon=True)
        self.thread.start()
        return True

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
            self.thread = None

    def captureLoop(self):
        """Runs in the background thread: read, process, store the result, repeat."""
        previousGray = None
        failures = 0
        while self.running:
            try:
                ok, frame = self.capture.read()
            except Exception:
                ok = False
                frame = None
            if not ok or frame is None:
                failures += 1
                if failures > 100:
                    print("[webcam] Camera stopped sending frames")
                    break
                time.sleep(0.03)
                continue
            failures = 0
            frame = cv2.resize(frame, (self.width, self.height))
            gray, mask, centroid = processFrame(previousGray, frame)
            previousGray = gray
            self.storeResult(gray, mask, centroid)
        self.running = False
        try:
            self.capture.release()
        except Exception:
            pass

    def storeResult(self, gray, mask, centroid):
        preview = self.makePreview(gray, mask, centroid)
        with self.lock:
            self.framesProcessed += 1
            self.previewImage = preview
            if centroid is not None:
                # Image y grows downwards, so the top of the picture is height 1.0
                rawHeight = 1.0 - centroid[1] / float(self.height)
                rawHeight = max(0.0, min(1.0, rawHeight))
                if self.handHeight is None:
                    self.handHeight = rawHeight
                else:
                    # exponential smoothing removes the jitter of single frames
                    self.handHeight = self.handHeight + (rawHeight - self.handHeight) * SMOOTHING
                self.lastSeenTime = time.time()

    def feedFrame(self, frame):
        """
        Process one frame given by the program instead of the camera thread
        (used by the SimulatedCamera). Same steps as captureLoop().
        """
        frame = cv2.resize(frame, (self.width, self.height))
        gray, mask, centroid = processFrame(self.previousGray, frame)
        self.previousGray = gray
        self.storeResult(gray, mask, centroid)

    def makePreview(self, gray, mask, centroid):
        """Small picture for the on-screen thumbnail: dim camera image + bright motion mask."""
        mixed = cv2.addWeighted(gray, 0.35, mask, 0.65, 0)
        if centroid is not None:
            cv2.circle(mixed, (int(centroid[0]), int(centroid[1])), 10, 255, 2)
        image = Image.fromarray(mixed).convert("RGBA")
        return image.resize(PREVIEW_SIZE)

    def getHandHeight(self):
        """Smoothed hand height 0..1 (1 = top of the camera view), or None if no hand was seen lately."""
        with self.lock:
            if self.handHeight is None:
                return None
            if time.time() - self.lastSeenTime > HOLD_TIME:
                return None
            return self.handHeight

    def getPreviewImage(self):
        """The latest motion-mask preview (PIL RGBA image), or None."""
        with self.lock:
            if self.previewImage is None:
                return None
            return self.previewImage.copy()

    def isRunning(self):
        return self.running


class SimulatedCamera:
    """
    A pretend webcam (python main.py --fake-camera). No real camera is opened.
    It draws a skin-coloured hand on a dark, slightly noisy background. The hand
    waves from side to side while it rises, because frame differencing only
    sees things that move.
    """
    def __init__(self, width=320, height=240):
        self.width = width
        self.height = height
        self.random = numpy.random.default_rng(5)     # fixed seed: the same noise every run

    def handHeightAt(self, seconds):
        """Where the pretend hand is held: 0 = bottom of the picture, 1 = top."""
        rise = min(1.0, seconds / 5.0)                 # the hand goes up over 5 seconds
        return 0.2 + 0.6 * rise + 0.05 * math.sin(seconds * 1.7)

    def frameAt(self, seconds):
        """One BGR camera frame (NumPy array, like cv2.VideoCapture.read gives)."""
        frame = numpy.full((self.height, self.width, 3), 40, dtype=numpy.uint8)
        noise = self.random.integers(0, 10, (self.height, self.width, 1), dtype=numpy.uint8)
        frame = frame + noise                          # camera noise (the blur step removes it)
        x = int(self.width / 2 + 30 * math.sin(seconds * 7.0))     # the wave
        y = int((1.0 - self.handHeightAt(seconds)) * self.height)
        skin = (120, 160, 215)                         # OpenCV colours are in B, G, R order
        cv2.ellipse(frame, (x, y), (30, 36), 0, 0, 360, skin, -1)                 # palm
        for i in range(4):
            cv2.ellipse(frame, (x - 21 + i * 14, y - 42), (6, 16), 0, 0, 360, skin, -1)   # fingers
        cv2.ellipse(frame, (x + 34, y - 4), (7, 15), -35, 0, 360, skin, -1)      # thumb
        return frame
