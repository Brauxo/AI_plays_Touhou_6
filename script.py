import pygetwindow as gw
from mss import mss
import cv2
import numpy as np
from config import SCREEN_REGION  
import pyautogui

#This script is used to capture the screen and save it as a screenshot.png file, I use it to see if the game is currently in the window (change config.py).
#It also prints the titles of all the windows currently open

windows = gw.getAllTitles()
for title in windows:
    print(title)

sct = mss()

screenshot = sct.grab(SCREEN_REGION)

img = np.array(screenshot)
cv2.imwrite("img/screenshot.png", img)  
print(f"Screenshot saved as 'screenshot.png' with region: {SCREEN_REGION}")

pyautogui.mouseInfo()
