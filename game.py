import cv2
import numpy as np
import pygetwindow as gw
import pydirectinput
from mss import mss
import time
import threading
import random
from collections import deque
from config import SCREEN_REGION, STATE_SIZE, ACTIONS, SHOOT_KEY, FRAME_STACK_SIZE

class AsyncScreenCapture:
    def __init__(self, region, state_size):
        self.region = region
        self.state_size = state_size
        self.latest_frame = None
        self.lock = threading.Lock()
        self.running = True
        self.gray_buffer = np.zeros((region['height'], region['width']), dtype=np.uint8)
        self.resized_buffer = np.zeros(state_size[:2], dtype=np.uint8)
        self.thread = threading.Thread(target=self.update)
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        sct = mss()
        while self.running:
            raw = sct.grab(self.region)
            img = np.frombuffer(raw.bgra, dtype=np.uint8).reshape((raw.height, raw.width, 4))
            cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY, dst=self.gray_buffer)
            cv2.resize(self.gray_buffer, self.state_size[:2], dst=self.resized_buffer, interpolation=cv2.INTER_AREA)
            with self.lock:
                self.latest_frame = self.resized_buffer.copy()
            time.sleep(0.001)

    def get_frame(self):
        while self.latest_frame is None:
            time.sleep(0.01)
        with self.lock:
            return self.latest_frame.copy()

    def stop(self):
        self.running = False
        self.thread.join()

class TouhouEnv:
    def __init__(self):
        self.capture = AsyncScreenCapture(SCREEN_REGION, STATE_SIZE)
        self.frame_stack = deque(maxlen=FRAME_STACK_SIZE)
        self.current_keys = set()
        
        self.game_over_templates = []

        template_files = ["img/game_over.png", "img/game_over_2.png"]
        
        for file_path in template_files:
            try:
                self.game_over_templates.append(self._load_template(file_path, STATE_SIZE[:2]))
                print(f"Loaded template: {file_path}")
            except FileNotFoundError:
                print(f"Warning: Template not found at {file_path}")
        if not self.game_over_templates: raise RuntimeError("No game over templates found!")
        
        self.focus_game()
        pydirectinput.keyDown(SHOOT_KEY)
        self.current_keys.add(SHOOT_KEY)

    def _load_template(self, path, size):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None: raise FileNotFoundError(f"Could not load {path}.")
        return cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        
    def _get_stacked_state(self):
        return np.stack(self.frame_stack, axis=-1)

    def focus_game(self):
        try:
            windows = gw.getWindowsWithTitle("Touhou Scarlet Devil Land ~ The Embodiment of Scarlet Devil v1.02h")
            if windows: windows[0].activate()
            time.sleep(0.2)
        except Exception: pass

    def reset(self):
        self.focus_game()
        self.frame_stack.clear()
        initial_frame = self.capture.get_frame()
        for _ in range(FRAME_STACK_SIZE):
            self.frame_stack.append(initial_frame)
        return self._get_stacked_state()

    def step(self, action_idx):
        self.perform_action(action_idx)
        
        new_frame = self.capture.get_frame()
        self.frame_stack.append(new_frame)
        next_state = self._get_stacked_state()

        done = self.is_game_over(next_state)
        reward = -400 if done else 2
        
        if done:
            self.restart_game()
            self.frame_stack.clear()
            fresh_frame = self.capture.get_frame()
            for _ in range(FRAME_STACK_SIZE):
                self.frame_stack.append(fresh_frame)
            next_state = self._get_stacked_state()
        
        return next_state, reward, done

    def perform_action(self, action_idx):
        if random.random() < 0.01: self.focus_game()
        new_keys = set(ACTIONS[action_idx])
        keys_to_release = self.current_keys - new_keys - {SHOOT_KEY}
        keys_to_press = new_keys - self.current_keys
        for key in keys_to_release: pydirectinput.keyUp(key)
        for key in keys_to_press: pydirectinput.keyDown(key)
        self.current_keys = new_keys | {SHOOT_KEY}
        time.sleep(0.01)

    def is_game_over(self, state):
        frame_to_check = state[:, :, 0]
        for template in self.game_over_templates:
            if np.max(cv2.matchTemplate(frame_to_check, template, cv2.TM_CCOEFF_NORMED)) > 0.8:
                return True
        return False

    def restart_game(self):
        print("\n--- GAME OVER: Executing your specific restart sequence. ---\n")
        keys_to_release = self.current_keys - {SHOOT_KEY}
        for key in keys_to_release:
            pydirectinput.keyUp(key)
        self.current_keys = {SHOOT_KEY}
        time.sleep(2.5)
        pydirectinput.keyDown("enter"); time.sleep(0.2); pydirectinput.keyUp("enter"); time.sleep(0.2)
        for _ in range(4):
            pydirectinput.keyDown("escape"); time.sleep(0.5); pydirectinput.keyUp("escape"); time.sleep(0.5)
        for _ in range(2):
            pydirectinput.keyDown("down"); time.sleep(0.2); pydirectinput.keyUp("down"); time.sleep(0.2)
        for _ in range(6):
            pydirectinput.keyDown("z"); time.sleep(0.2); pydirectinput.keyUp("z"); time.sleep(0.2)
        self.focus_game()
        pydirectinput.keyDown(SHOOT_KEY)
        self.current_keys = {SHOOT_KEY}
        time.sleep(1.5)
        print("--- Restart Complete ---")

    def cleanup(self):
        self.capture.stop()
        for key in self.current_keys:
            pydirectinput.keyUp(key)