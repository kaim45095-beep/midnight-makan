"""
scenes package
--------------
makeScene() turns a scene id (a short text name) into a new View object.
The imports are inside the function so the scene files never import each
other in a circle.
"""

SCENE_IDS = ["title", "story", "street", "rojak", "wok", "tehtarik", "banquet", "finale"]


def makeScene(sceneId):
    if sceneId == "title":
        from scenes.title_scene import TitleScene
        return TitleScene()
    if sceneId == "story":
        from scenes.story_scene import StoryScene
        return StoryScene()
    if sceneId == "street":
        from scenes.street_scene import StreetScene
        return StreetScene()
    if sceneId == "rojak":
        from scenes.rojak_scene import RojakScene
        return RojakScene()
    if sceneId == "wok":
        from scenes.wok_scene import WokScene
        return WokScene()
    if sceneId == "tehtarik":
        from scenes.tehtarik_scene import TehTarikScene
        return TehTarikScene()
    if sceneId == "banquet":
        from scenes.banquet_scene import BanquetScene
        return BanquetScene()
    if sceneId == "finale":
        from scenes.finale_scene import FinaleScene
        return FinaleScene()
    raise ValueError(f"Unknown scene id: {sceneId}")
