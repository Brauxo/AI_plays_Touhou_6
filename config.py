SCREEN_REGION = {'top': 20, 'left': 302, 'width': 962, 'height': 747}
STATE_SIZE = (256, 256, 1)
ACTIONS = [
    ("Nothing"),         # 0: Nothing (just 'w' will still be held)
    ("up",),             # 1: Up
    ("down",),           # 2: Down
    ("left",),           # 3: Left
    ("right",),          # 4: Right      
    ("up", "left"),      # 5: Up + Left
    ("up", "right"),     # 6: Up + Right
    ("down", "left"),    # 7: Down + Left
    ("down", "right"),   # 8: Down + Right
    ("shift", "left"),   # 9: Shift + Left
    ("shift", "right"),  # 10: Shift + Right
    ("shift", "up"),     # 11: Shift + Up
    ("shift", "down"),   # 12: Shift + Down
]

#("x",), # 12: bomb not used because it's broken

SHOOT_KEY = "z" # Permanent key (shooting)
ACTION_SIZE = len(ACTIONS)
LEARNING_RATE = 0.001
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
BATCH_SIZE = 32
MEMORY_SIZE = 2000
MODEL_PATH = "models/touhou_AI.h5"