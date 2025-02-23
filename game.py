import cv2
import numpy as np
import pygetwindow as gw
import pydirectinput
from mss import mss
import time
from config import SCREEN_REGION, STATE_SIZE, ACTIONS, SHOOT_KEY

class TouhouEnv:
    def __init__(self):
        self.sct = mss()
        self.current_keys = set()  # Track currently pressed keys
        self.focus_game()
        # Start with shoot key pressed permanently
        pydirectinput.keyDown(SHOOT_KEY)
        self.current_keys.add(SHOOT_KEY)
        # Load the Game Over template from img folder
        self.game_over_template = cv2.imread("img/game_over.png", cv2.IMREAD_GRAYSCALE)
        if self.game_over_template is None:
            raise FileNotFoundError("Could not load img/game_over.png. Ensure it’s in the img folder.")

    def focus_game(self):
        windows = gw.getWindowsWithTitle("Touhou Scarlet Devil Land ~ The Embodiment of Scarlet Devil v1.02h")
        if windows:
            windows[0].activate()
            time.sleep(0.2)  # delay important

    def reset_keys(self):
        # Release all keys except the permanent shoot key
        for key in self.current_keys:
            if key != SHOOT_KEY:
                pydirectinput.keyUp(key)
        self.current_keys = {SHOOT_KEY}  # Keep shoot key pressed

    def capture_screen(self):
        screenshot = self.sct.grab(SCREEN_REGION)
        img = np.array(screenshot)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (STATE_SIZE[0], STATE_SIZE[1]))
        return np.reshape(img, STATE_SIZE)

    def perform_action(self, action_idx):
        self.focus_game()  # focus game
        # Get the new set of keys to press from ACTIONS (tuple of keys)
        new_keys = set(ACTIONS[action_idx])  # Convert tuple to set
        for key in self.current_keys - new_keys:
            if key != SHOOT_KEY:
                pydirectinput.keyUp(key)

        for key in new_keys - self.current_keys:
            pydirectinput.keyDown(key)

        # Update the current state of pressed keys
        self.current_keys = new_keys | {SHOOT_KEY}  # keep the shoot key because it has no impact
        time.sleep(1/60)  # for 60 FPS

    def is_game_over(self, state):
        # Resize the template to match the state size (84x84)
        template = cv2.resize(self.game_over_template, (STATE_SIZE[0], STATE_SIZE[1]))
        # Perform template matching
        result = cv2.matchTemplate(state[:, :, 0], template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        return max_val > 0.8  # Threshold for match

    def get_reward(self, state, done):
        if done:
            return -100
        return 1  # Survivre = +1 (to be improved)

    def restart_game(self):
        """Execute the sequence to start a new episode after Game Over: Esc, down, down, z, z, z, z, z, z."""
        self.reset_keys() 
        time.sleep(1.0)  

        # Sequence: Esc, down (x2), z (x6)
        pydirectinput.keyDown("escape")
        time.sleep(0.2)
        pydirectinput.keyUp("escape")
        time.sleep(0.2)

        for _ in range(2):  # 2 'down' presses
            pydirectinput.keyDown("down")
            time.sleep(0.2)
            pydirectinput.keyUp("down")
            time.sleep(0.2)

        for _ in range(6):  # 6 'z' presses
            pydirectinput.keyDown("z")
            time.sleep(0.2)
            pydirectinput.keyUp("z")
            time.sleep(0.2)

        self.focus_game()
        pydirectinput.keyDown(SHOOT_KEY) # Keep the shoot key pressed
        self.current_keys = {SHOOT_KEY}

    def step(self, action_idx):
        self.perform_action(action_idx)
        next_state = self.capture_screen()
        done = self.is_game_over(next_state)
        reward = self.get_reward(next_state, done)
        
        if done:
            self.restart_game()  # Restart the game if Game Over detected
        
        return next_state, reward, done

    def cleanup(self):
        # Release all keys when done
        for key in self.current_keys:
            pydirectinput.keyUp(key)
        self.current_keys.clear()