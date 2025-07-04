import cv2
import numpy as np
import pygetwindow as gw
import pydirectinput
from mss import mss
import time
import threading
import random
from config import SCREEN_REGION, STATE_SIZE, ACTIONS, SHOOT_KEY

class AsyncScreenCapture:
    def __init__(self, region, state_size):
        self.region = region
        self.state_size = state_size
        self.latest_frame = None
        self.lock = threading.Lock()
        self.running = True
        
        # Pre-allocate buffers for 256x256
        self.screenshot_buffer = np.zeros((region['height'], region['width'], 4), dtype=np.uint8)
        self.gray_buffer = np.zeros((region['height'], region['width']), dtype=np.uint8)
        self.resized_buffer = np.zeros(state_size[:2], dtype=np.uint8)  # (256, 256)
        
        # Thread-local mss instance
        self.thread = threading.Thread(target=self.update)
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        sct = mss()  # Initialize mss in the thread
        while self.running:
            raw = sct.grab(self.region)
            img = np.frombuffer(raw.bgra, dtype=np.uint8).reshape((raw.height, raw.width, 4))
            cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY, dst=self.gray_buffer)
            cv2.resize(self.gray_buffer, (self.state_size[0], self.state_size[1]), 
                      dst=self.resized_buffer, interpolation=cv2.INTER_NEAREST)
            with self.lock:
                self.latest_frame = self.resized_buffer.copy()

    def get_frame(self):
        with self.lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None

    def stop(self):
        self.running = False
        self.thread.join()

class TouhouEnv:
    def __init__(self):
        self.capture = AsyncScreenCapture(SCREEN_REGION, STATE_SIZE)
        self.current_keys = set()  # Track currently pressed keys
        self.focus_game()
        # Start with shoot key pressed permanently
        pydirectinput.keyDown(SHOOT_KEY)
        self.current_keys.add(SHOOT_KEY)
        # Load the Game Over template from img folder
        self.game_over_template = self._load_template("img/game_over.png", STATE_SIZE[:2])
        self.prev_state = None  # Store previous state

    def _load_template(self, path, size):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Could not load {path}. Ensure it’s in the img folder.")
        return cv2.resize(img, size, interpolation=cv2.INTER_AREA)

    def focus_game(self):
        windows = gw.getWindowsWithTitle("Touhou Scarlet Devil Land ~ The Embodiment of Scarlet Devil v1.02h")
        if windows:
            try:
                windows[0].activate()
            except Exception:
                pass  # Avoid focus exceptions
            time.sleep(0.2)  # delay important

    def reset_keys(self):
        # Release all keys except the permanent shoot key
        keys_to_release = self.current_keys - {SHOOT_KEY}
        for key in keys_to_release:
            pydirectinput.keyUp(key)
        self.current_keys = {SHOOT_KEY}  # Keep shoot key pressed

    def capture_screen(self):
        frame = self.capture.get_frame()
        if frame is None:  # Fallback using a new mss instance
            sct = mss()
            screenshot = sct.grab(SCREEN_REGION)
            img = np.array(screenshot)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
            frame = cv2.resize(img, (STATE_SIZE[0], STATE_SIZE[1]))
        return np.reshape(frame, STATE_SIZE)

    def perform_action(self, action_idx):
        # Reduced focus check frequency (1% chance)
        if random.random() < 0.01:  
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
        # Perform template matching
        result = cv2.matchTemplate(state[:, :, 0], self.game_over_template, cv2.TM_CCOEFF_NORMED)
        return np.max(result) > 0.8  # Threshold for match

    def get_reward(self, prev_state, next_state, done):
        if done:
            reward = -400
            print(f"Reward: {reward} (Game Over)")
            return reward
        
        frame = next_state[:, :, 0]
        prev_frame = prev_state[:, :, 0] if prev_state is not None else frame  # Use current if no prev

        # Simplified reward: survival + hit detection only
        reward = 2 
        print(f"Reward: {reward}")
        return reward

    def restart_game(self):
        """Execute the sequence to start a new episode after Game Over: Esc, down, down, z, z, z, z, z, z."""
        self.reset_keys() 
        time.sleep(2.5)  # Reduced initial delay

        pydirectinput.keyDown("enter")
        time.sleep(0.2) 
        pydirectinput.keyUp("enter")
        time.sleep(0.2)
    

        for _ in range(4): 
            pydirectinput.keyDown("escape")
            time.sleep(0.5) 
            pydirectinput.keyUp("escape")
            time.sleep(0.5)

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
        pydirectinput.keyDown(SHOOT_KEY) 
        self.current_keys = {SHOOT_KEY}
        self.prev_state = None  

    def step(self, action_idx):
        start = time.time()
        self.perform_action(action_idx)
        next_state = self.capture_screen()
        done = self.is_game_over(next_state)
        reward = self.get_reward(self.prev_state, next_state, done)
        if done:
            self.restart_game()
        self.prev_state = next_state  # Update prev_state for next step
        elapsed = time.time() - start
        print(f"Step time: {elapsed:.3f}s, FPS: {1/elapsed:.1f}")
        return next_state, reward, done

    def cleanup(self):
        self.capture.stop()
        # Release all keys when done
        for key in self.current_keys:
            pydirectinput.keyUp(key)
        self.current_keys.clear()

