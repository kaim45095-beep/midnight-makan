"""
core package
------------
The shared engine parts of Midnight Makan that every scene uses:

  helpers.py        maths: lerp, easing curves, clamp, HSV colours, Tween
  assets.py         loading images / sprite sheets / fonts + runtime image processing
  sound_manager.py  all audio (sound effects, loops, music layers, ambience)
  particles.py      particle system (emitter -> particles -> forces -> life cycle)
  effects.py        screen shake, flash, popups, iris transition, point lights
  ui.py             panels, buttons, dialogue box, HUD, result card
  base_scene.py     BaseScene: the state machine every scene inherits from
  motion_tracker.py optional webcam hand tracking (OpenCV frame differencing)
  recorder.py       saves the demo video by sending frames to ffmpeg
"""
