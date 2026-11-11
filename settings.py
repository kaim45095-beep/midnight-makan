"""
settings.py
-----------
All fixed values (constants) for Midnight Makan live in this one file.
Keeping them together means I can tune the game without hunting
through every scene file.
"""
import os

# ----------------------------------------------------------------------
# Window
# ----------------------------------------------------------------------
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
SCREEN_TITLE = "Midnight Makan - A Night-Market Dining Adventure"

# The biggest time step we allow in one update. If the computer freezes
# for a moment we do not want objects to jump across the screen.
MAX_DELTA_TIME = 1 / 20

# Frames per second used when the demo video is recorded
RECORD_FPS = 30

# ----------------------------------------------------------------------
# Folders (built from this file's location so the game runs from anywhere)
# ----------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSET_DIR = os.path.join(BASE_DIR, "assets")
IMAGE_DIR = os.path.join(ASSET_DIR, "images")
SOUND_DIR = os.path.join(ASSET_DIR, "sounds")
FONT_DIR = os.path.join(ASSET_DIR, "fonts")
SCREENSHOT_DIR = os.path.join(BASE_DIR, "screenshots")
MEMORY_DIR = os.path.join(BASE_DIR, "memories")

# ----------------------------------------------------------------------
# Fonts (the .ttf files are loaded from assets/fonts when the game starts)
# ----------------------------------------------------------------------
FONT_TITLE = "Kenney Future"          # big headings, buttons, popups
FONT_NARROW = "Kenney Future Narrow"  # HUD labels
FONT_BODY = "Liberation Sans"         # dialogue and long text (easy to read)

# ----------------------------------------------------------------------
# Colour palette (R, G, B). Warm lantern colours on a deep night-blue base.
# ----------------------------------------------------------------------
NIGHT_DEEP = (14, 16, 38)
NIGHT = (26, 29, 66)
NIGHT_BLUE = (44, 52, 110)
LANTERN_RED = (214, 48, 49)
LANTERN_ORANGE = (255, 140, 50)
GOLD = (255, 200, 87)
WARM_LIGHT = (255, 225, 160)
CREAM = (255, 244, 224)
TEAL = (46, 196, 182)
LAVENDER = (190, 150, 230)
SKY_BLUE = (120, 190, 255)
PINK = (236, 96, 170)
GOOD_GREEN = (110, 220, 130)
BAD_RED = (240, 84, 84)
UI_PANEL = (24, 26, 52)
UI_PANEL_LIGHT = (48, 52, 96)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# ----------------------------------------------------------------------
# Shared gameplay tuning
# ----------------------------------------------------------------------
COMBO_FOR_AURA = 5            # good/perfect actions in a row to start Aura Mode
AURA_MODE_TIME = 6.0          # seconds that Aura Mode lasts
AURA_SCORE_MULTIPLIER = 2     # points are doubled during Aura Mode
STAR_THRESHOLDS = (0.5, 0.75) # score fraction needed for 2 stars and 3 stars
COUNTDOWN_TIME = 3.2          # "3, 2, 1, COOK!" before each dining scene

# The four main dining scenes, in story order
SCENE_ORDER = ["rojak", "wok", "tehtarik", "banquet"]

SCENE_NAMES = {
    "rojak": "Rojak Rush",
    "wok": "Wok Hei Fire",
    "tehtarik": "The Great Pull",
    "banquet": "Lazy Susan Banquet",
}

STALL_NAMES = {
    "rojak": "Uncle Din's Fruit Stall",
    "wok": "Ah Seng's Wok Station",
    "tehtarik": "Sri Pelita Cafe",
    "banquet": "Pelita Banquet Hall",
}

DISH_NAMES = {
    "rojak": "Fruit Rojak",
    "wok": "Char Kuey Teow",
    "tehtarik": "Teh Tarik",
    "banquet": "The Midnight Feast",
}

# ----------------------------------------------------------------------
# Pelita Street (the walkable street hub)
# ----------------------------------------------------------------------
STREET_WIDTH = 3200          # width of the street world in pixels
STREET_FLOOR_Y = 118         # y position of Aiman's feet on the street
STALL_X = {                  # centre x of each stall entrance on the street
    "rojak": 620,
    "wok": 1320,
    "tehtarik": 2020,
    "banquet": 2720,
}
PLAYER_WALK_SPEED = 260      # pixels per second

# ----------------------------------------------------------------------
# Dialogue speakers: display name, portrait image, voice blip and name colour
# ----------------------------------------------------------------------
SPEAKERS = {
    "aiman": {"name": "Aiman", "portrait": "characters/portrait_aiman.png",
              "voice": "voice_aiman", "pitch": 1.1, "color": LANTERN_ORANGE},
    "nenek": {"name": "Grandma Siti", "portrait": "characters/portrait_nenek.png",
              "voice": "voice_nenek", "pitch": 0.95, "color": LAVENDER},
    "leon": {"name": "Dato' Leon Tan", "portrait": "characters/portrait_leon.png",
             "voice": "voice_leon", "pitch": 0.8, "color": GOLD},
    "lim": {"name": "Uncle Lim", "portrait": "characters/portrait_lim.png",
            "voice": "voice_lim", "pitch": 0.9, "color": SKY_BLUE},
    "kavitha": {"name": "Auntie Kavitha", "portrait": "characters/portrait_kavitha.png",
                "voice": "voice_kavitha", "pitch": 1.05, "color": PINK},
    "farah": {"name": "Farah", "portrait": "characters/portrait_farah.png",
              "voice": "voice_farah", "pitch": 1.2, "color": TEAL},
    "oyen": {"name": "Oyen", "portrait": "characters/portrait_oyen.png",
             "voice": "cat_meow", "pitch": 1.0, "color": LANTERN_ORANGE},
    "narrator": {"name": "", "portrait": None,
                 "voice": None, "pitch": 1.0, "color": CREAM},
}
