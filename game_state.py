"""
game_state.py
-------------
GameState remembers what happened during one night on Pelita Street:
the score and stars from every dining scene, where Aiman is standing on
the street, and a few story flags. Every scene can read it through
self.window.gameState.
"""
from settings import SCENE_ORDER, STAR_THRESHOLDS


class GameState:
    def __init__(self):
        self.reset()

    def reset(self):
        # sceneId -> {"score": int, "maxScore": int, "stars": int, "bestCombo": int}
        self.results = {}
        # Story mode: scenes unlock one by one and the street links them.
        # Free mode (Scene Select): every scene is open and returns to the title.
        self.storyMode = True
        # Aiman's x position on the street, so he comes back to the same stall
        self.streetX = 220
        # Small story flags, e.g. "metNenek" or "photosTaken"
        self.flags = {}

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------
    def recordResult(self, sceneId, score, maxScore, bestCombo=0):
        """Save a finished scene and return how many stars it earned (1-3)."""
        stars = self.starsForScore(score, maxScore)
        old = self.results.get(sceneId)
        # Keep the best attempt if the scene is replayed
        if old is None or score > old["score"]:
            self.results[sceneId] = {
                "score": int(score),
                "maxScore": int(maxScore),
                "stars": stars,
                "bestCombo": int(bestCombo),
            }
        return stars

    def starsForScore(self, score, maxScore):
        if maxScore <= 0:
            return 1
        fraction = score / maxScore
        if fraction >= STAR_THRESHOLDS[1]:
            return 3
        if fraction >= STAR_THRESHOLDS[0]:
            return 2
        return 1  # finishing a dish always earns at least one star

    def getStars(self, sceneId):
        if sceneId in self.results:
            return self.results[sceneId]["stars"]
        return 0

    def getScore(self, sceneId):
        if sceneId in self.results:
            return self.results[sceneId]["score"]
        return 0

    def totalStars(self):
        total = 0
        for sceneId in SCENE_ORDER:
            total += self.getStars(sceneId)
        return total

    def totalScore(self):
        total = 0
        for sceneId in SCENE_ORDER:
            total += self.getScore(sceneId)
        return total

    # ------------------------------------------------------------------
    # Progress through the story
    # ------------------------------------------------------------------
    def isCompleted(self, sceneId):
        return sceneId in self.results

    def isUnlocked(self, sceneId):
        if not self.storyMode:
            return True
        index = SCENE_ORDER.index(sceneId)
        if index == 0:
            return True
        previousScene = SCENE_ORDER[index - 1]
        return self.isCompleted(previousScene)

    def nextSceneToPlay(self):
        """The first dining scene that is not finished yet (or None)."""
        for sceneId in SCENE_ORDER:
            if not self.isCompleted(sceneId):
                return sceneId
        return None

    def allCompleted(self):
        return self.nextSceneToPlay() is None

    def auraFraction(self):
        """How bright the Aura Ladle glows: stars earned out of 12."""
        return self.totalStars() / (3 * len(SCENE_ORDER))

    def endingType(self):
        stars = self.totalStars()
        if stars >= 10:
            return "golden"
        if stars >= 6:
            return "silver"
        return "bronze"
