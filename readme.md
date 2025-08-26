# 🎭 Smart Beauty & Dress Filter

An advanced AI-powered virtual makeover application built with Python, OpenCV, and MediaPipe.
It allows users to try makeup filters (lipstick, blush, eyeshadow) and dress overlays in real-time using their webcam.
The system uses pose estimation and face landmarks for accurate placement, and comes with a modular architecture for extensibility.

---

## ✨ Features

#### 🎨 Makeup Filters

- Lipstick (10 color shades)
- Blush (10 color shades)
- Eyeshadow (10 color shades)

#### 👗 Dress Overlay Filter

- Dress images mapped to shoulders & torso
- Works even if full body isn’t visible
- Anchored from shoulders (cuts from bottom if frame is small)

#### 🌸 Skin Tone Detection

- Real-time skin calibration
- Dynamic color recommendations

#### 🖥 User Interface

- Status panel with current dress & makeup info
- Help menu with controls
- Color selection with keyboard shortcuts

#### ⚙️ Modular Architecture

- Clean module separation (filters/, processing/, core/, utils/)
- Easy to add new filters (hairstyles, jewelry, accessories, etc.)

---

## 📂 Project Structure

```
smart_beauty_filter/
│
├── main.py                        # Entry point (launches app)
├── config.py                      # Configuration (factors, constants)
│
├── core/                          # Core system modules
│   ├── camera_manager.py
│   ├── ui_manager.py
│   ├── logger_manager.py
│
├── filters/                       # Beauty & dress filters
│   ├── beauty_filter.py
│   ├── skin_detector.py
│   ├── dress_manager.py
│
├── processing/                    # Image & pose processing
│   ├── image_processor.py
│   ├── pose_estimator.py
│
├── app/
│   ├── smart_beauty_filter.py     # Orchestrates the app
│
├── utils/
│   ├── color_palette.py           # Centralized makeup colors
│   ├── helpers.py
│
├── assets/                        # Overlay PNGs
│   ├── dresses/
│   ├── makeup/
│
└── requirements.txt               # Python dependencies
```

---

## 🎮 Controls


| Key | Action                                  |
| --- | --------------------------------------- |
| `[` | Previous Dress                          |
| `]` | Next Dress                              |
| `d` | Get dress recommendation (season-based) |
| `1` | Cycle lipstick colors                   |
| `2` | Cycle blush colors                      |
| `3` | Cycle eyeshadow colors                  |
| `l` | Toggle lipstick                         |
| `b` | Toggle blush                            |
| `e` | Toggle eyeshadow                        |
| `h` | Toggle help menu                        |
| `q` | Quit application                        |

---

## ⚡ Installation

1. Clone this repo:

```
git clone https://github.com/yourusername/smart-beauty-filter.git
cd smart-beauty-filter
```

2. Install dependencies:

```
pip install -r requirements.txt
```

3. Add dress & makeup PNG overlays to the assets/ folder.
   <br><br>
4. Run the application:

```
python main.py
```

---

## 🛠 Dependencies

* Python 3.9+
* OpenCV
* MediaPipe
* NumPy
* Logging

---

## 🚀 Future Enhancements

* 👑 Hairstyle overlays
* 💍 Jewelry & accessory filters
* 🎤 Voice-controlled filter switching
* 📱 Mobile app version

---

## 📜 License

This project is licensed under the MIT License – feel free to use, modify, and distribute.



