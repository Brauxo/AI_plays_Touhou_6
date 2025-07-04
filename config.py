# --- Screen and State Configuration ---
SCREEN_REGION = {'top': 20, 'left': 302, 'width': 962, 'height': 747}
FRAME_STACK_SIZE = 4  # Number of frames to stack for temporal information
STATE_SIZE = (256, 256, FRAME_STACK_SIZE) # The final shape of our state

# --- Action Space Configuration ---
ACTIONS = [
    ("Nothing"),         # 0
    ("up",),             # 1
    ("down",),           # 2
    ("left",),           # 3
    ("right",),          # 4
    ("shift",),          # 5
    ("shift", "left"),   # 6
    ("shift", "right"),  # 7
    ("shift", "up"),     # 8
    ("shift", "down"),   # 9
]
SHOOT_KEY = "z"
ACTION_SIZE = len(ACTIONS)

# --- DQN Hyperparameters ---
LEARNING_RATE = 0.00025
GAMMA = 0.99
MEMORY_SIZE = 100000
BATCH_SIZE = 64

# --- Epsilon (Exploration) Parameters ---
EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.9995

# --- File Paths ---
MODEL_PATH = "models/touhou_ai.keras"