# AI_plays_Touhou_6 

[Project in work]

This project trains a Deep Q-Network (DQN) agent to play *Touhou 6: Embodiment of Scarlet Devil* 

Made by Brauxo in february 2025

## Features
- **Screen Capture**: Captures the game window using `mss` and processes it with OpenCV.
- **Input Control**: Simulates key presses with `pydirectinput` 
- **Game Over Detection**: Uses template matching to detect the "Game Over" screen and restart automatically with a simpel script.
- **DQN Training**: Implements a DQN with TensorFlow to learn optimal actions.

## Requirements
- Python 3.9