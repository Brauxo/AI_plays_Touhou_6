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
        # Load character templates
        self.char_templates = [
            cv2.imread(f"img/char{i}.png", cv2.IMREAD_GRAYSCALE) for i in range(1, 4)
        ]
        for i, template in enumerate(self.char_templates, 1):
            self.char_templates[i-1] = cv2.resize(template, (10, 20))  # Scaled for 256x256 state

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
        time.sleep(1/60)  # for 60 FPS (single frame timing)


    def is_game_over(self, state):
        # Resize the template to match the state size (256x256)
        template = cv2.resize(self.game_over_template, (STATE_SIZE[0], STATE_SIZE[1]))
        # Perform template matching
        result = cv2.matchTemplate(state[:, :, 0], template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        return max_val > 0.8  # Threshold for match

    def get_reward(self, prev_state, next_state, done):
        if done:
            reward = -100
            print(f"Reward: {reward} (Game Over)")
            return reward
        
        frame = next_state[:, :, 0]
        prev_frame = prev_state[:, :, 0]
        diff = np.mean(np.abs(frame - prev_frame))
        hit_penalty = -10 if diff > 50 else 0  # Flash detection (adjust threshold)

        # Find player position using multiple templates
        player_pos = None
        max_val = 0
        for template in self.char_templates:
            result = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
            _, val, _, loc = cv2.minMaxLoc(result)
            if val > max_val and val > 0.6:  # Threshold for match
                max_val = val
                player_pos = loc

        if player_pos is None:
            # Fallback to center if player not found
            player_x, player_y = 128, 128  # Center 
        else:
            player_x, player_y = player_pos[0] + 10, player_pos[1] + 20  # Center 

        # Define danger zone around player 
        y_min, y_max = max(0, player_y-30), min(256, player_y+30)
        x_min, x_max = max(0, player_x-30), min(256, player_x+30)
        danger_zone = frame[y_min:y_max, x_min:x_max]

        # Detect projectiles
        projectile_mask = danger_zone > 200  # Adjust threshold
        projectile_count = np.sum(projectile_mask) / 255
        proximity_penalty = -1 * min(projectile_count, 5)

        reward = 2 + proximity_penalty + hit_penalty
        print(f"Reward: {reward} (Base: 2, Proximity Penalty: {proximity_penalty}, Hit Penalty: {hit_penalty}, Player at ({player_x}, {player_y}), Projectiles: {projectile_count:.1f})")
        return reward

    def restart_game(self):
        """Execute the sequence to start a new episode after Game Over: Esc, down, down, z, z, z, z, z, z."""
        self.reset_keys() 
        time.sleep(0.5)  
        pydirectinput.keyDown("escape")
        time.sleep(0.1)
        pydirectinput.keyUp("escape")
        time.sleep(0.1)

        for _ in range(2):  
            pydirectinput.keyDown("down")
            time.sleep(0.2)
            pydirectinput.keyUp("down")
            time.sleep(0.2)

        for _ in range(6):  
            pydirectinput.keyDown("z")
            time.sleep(0.2)
            pydirectinput.keyUp("z")
            time.sleep(0.2)

        self.focus_game()
        pydirectinput.keyDown(SHOOT_KEY) # Keep the shoot key pressed
        self.current_keys = {SHOOT_KEY}

    def step(self, action_idx):
        self.perform_action(action_idx)
        for _ in range(2):
            time.sleep(0.05)
            self.perform_action(action_idx)
        prev_state = self.capture_screen()
        next_state = self.capture_screen()
        done = self.is_game_over(next_state)
        reward = self.get_reward(prev_state, next_state, done)
        if done:
            self.restart_game()
        return next_state, reward, done

    def cleanup(self):
        # Release all keys when done
        for key in self.current_keys:
            pydirectinput.keyUp(key)
        self.current_keys.clear()