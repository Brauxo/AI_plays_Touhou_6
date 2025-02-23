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
        self.target_network.set_weights(self.q_network.get_weights())

    def build_model(self):
        model = tf.keras.Sequential([
            layers.Conv2D(32, (8, 8), strides=4, activation='relu', input_shape=STATE_SIZE),
            layers.Conv2D(64, (4, 4), strides=2, activation='relu'),
            layers.Conv2D(64, (3, 3), strides=1, activation='relu'),
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.Dense(ACTION_SIZE, activation='linear')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(LEARNING_RATE), loss='mse')
        return model

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def get_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randrange(ACTION_SIZE)
        state = np.expand_dims(state, axis=0)
        q_values = self.q_network.predict(state, verbose=0)
        return np.argmax(q_values[0])

    def train(self):
        if len(self.memory) < BATCH_SIZE:
            return
        minibatch = random.sample(self.memory, BATCH_SIZE)
        states = np.array([t[0] for t in minibatch])
        actions = np.array([t[1] for t in minibatch])
        rewards = np.array([t[2] for t in minibatch])
        next_states = np.array([t[3] for t in minibatch])
        dones = np.array([t[4] for t in minibatch])

        targets = self.q_network.predict(states, verbose=0)
        next_q_values = self.target_network.predict(next_states, verbose=0)
        for i in range(BATCH_SIZE):
            if dones[i]:
                targets[i][actions[i]] = rewards[i]
            else:
                targets[i][actions[i]] = rewards[i] + GAMMA * np.max(next_q_values[i])
        self.q_network.fit(states, targets, epochs=1, verbose=0)

    def update_target(self):
        self.target_network.set_weights(self.q_network.get_weights())

    def save(self):
        self.q_network.save(MODEL_PATH)