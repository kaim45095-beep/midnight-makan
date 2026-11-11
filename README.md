# midnight-makan
A 2D Python game inspired by Kuala Lumpur’s night markets. Make rojak, cook char kuey teow, and pull teh tarik with mouse controls or optional webcam hand tracking.
# Midnight Makan

![Midnight Makan title logo](assets/images/ui/logo.png)

**Midnight Makan** is a 2D game inspired by Kuala Lumpur’s night markets. Play as Aiman and prepare fruit rojak, char kuey teow, and teh tarik before serving guests at a banquet. Your scores across the cooking challenges determine the rooftop finale.

I built this game for the **Imaging and Special Effects** module at **Asia Pacific University**. It brings together interactive animation, sprite effects, audio, and image processing in a Malaysian food setting.

## Gameplay

| Challenge | What you do |
| --- | --- |
| **Rojak Rush** | Time your cuts to chop fruit for a rojak. |
| **Wok Hei Fire** | Drag ingredients into the wok in recipe order, control the flame, and toss the noodles. |
| **The Great Pull** | Mix and pull drinks with the mouse, or use the optional webcam mode to control the pull with hand movement. |
| **Lazy Susan Banquet** | Rotate the table and serve each guest the dish they want. |

The finale features a lantern-lighting sequence, animated fireworks, and an in-game Polaroid saved locally to `memories/`.

## Run locally

Use **Python 3.10–3.13**. Open a terminal in the folder containing `main.py` and `requirements.txt`, then run:

```bash
python -m pip install -r requirements.txt
python main.py
```

Choose **Start Story** for the full game, **Scene Select** to jump to a challenge, or **How to Play** for controls. The standard controls use the keyboard and mouse. In **The Great Pull**, press **T** to try webcam mode.

## How it works

- **Python and Arcade** handle the scenes, input, sprites, scoring, and particle effects.
- **OpenCV** detects movement between webcam frames and converts the tracked hand height into the tea-pouring motion.
- **Pillow** processes sprites and applies the finale’s Polaroid effects.

## Credits

Created by **Mohammad Kaymul Islam**. See the [font credits](assets/fonts/FONTS_README.txt) for font attributions.
