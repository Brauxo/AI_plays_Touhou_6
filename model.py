import tensorflow as tf
from keras import layers
import numpy as np
from collections import deque
import random
from config import STATE_SIZE, ACTION_SIZE, LEARNING_RATE, GAMMA, MEMORY_SIZE, BATCH_SIZE, MODEL_PATH

class DQN:
    def __init__(self):
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.q_network = self.build_model()
        self.target_network = self.build_model()
        self.update_target() 

    def build_model(self):
        model = tf.keras.Sequential([
            layers.Input(shape=STATE_SIZE), # 
            layers.Conv2D(32, (8, 8), strides=4, activation='relu'),
            layers.Conv2D(64, (4, 4), strides=2, activation='relu'),
            layers.Conv2D(64, (3, 3), strides=1, activation='relu'),
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.Dense(ACTION_SIZE, activation='linear', dtype='float32') 
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss='huber') 
        return model

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def get_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randrange(ACTION_SIZE)
        q_values = self.q_network.predict(np.expand_dims(state, axis=0), verbose=0)
        return np.argmax(q_values[0])

    def train(self):
        if len(self.memory) < BATCH_SIZE:
            return

        minibatch = random.sample(self.memory, BATCH_SIZE)
        states = np.array([transition[0] for transition in minibatch])
        actions = np.array([transition[1] for transition in minibatch])
        rewards = np.array([transition[2] for transition in minibatch])
        next_states = np.array([transition[3] for transition in minibatch])
        dones = np.array([transition[4] for transition in minibatch])

        q_values_next = self.target_network.predict(next_states, verbose=0)

        targets_for_actions = rewards + GAMMA * np.max(q_values_next, axis=1) * (1 - dones)

        current_q_values = self.q_network.predict(states, verbose=0)

        batch_indices = np.arange(BATCH_SIZE)
        current_q_values[batch_indices, actions] = targets_for_actions

        self.q_network.fit(states, current_q_values, epochs=1, verbose=0, batch_size=BATCH_SIZE)

    def update_target(self):
        self.target_network.set_weights(self.q_network.get_weights())

    def save(self):
        self.q_network.save(MODEL_PATH)