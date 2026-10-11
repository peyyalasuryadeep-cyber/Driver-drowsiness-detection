# Driver Drowsiness Detection Using Computer Vision

## About
A Python-based computer vision project that monitors eye closure through a webcam and alerts the user when prolonged eye closure is detected.

## Features
- Real-time webcam monitoring
- Face detection using MediaPipe Face Mesh
- Facial landmark visualization
- Eye Aspect Ratio (EAR) calculation
- Drowsiness detection based on prolonged eye closure
- Audio alarm for warnings

## Technologies Used
- Python
- OpenCV
- MediaPipe
- NumPy
- Pygame

## Installation

1. Install Python.
2. Clone this repository.
3. Open the project folder in VS Code.
4. Create and activate a virtual environment.

```bash
python -m venv venv
```

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

5. Install dependencies:

```bash
python -m pip install opencv-python mediapipe==0.10.21 numpy pygame
```

6. Run the application:

```bash
python main.py
```

Press Q to close the webcam window.

## Safety Notice
This is an educational prototype. It is not a certified vehicle safety system and must not be relied upon to prevent accidents.