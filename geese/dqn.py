import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

from collections import deque
from .StateTranslator import StateTranslator_Central


class dqnAgent:
    """
    Given an environment state, choose an action, and learn from the reward
    https://towardsdatascience.com/reinforcement-learning-w-keras-openai-dqns-1eed3a5338c
    https://towardsdatascience.com/deep-q-learning-tutorial-mindqn-2a4c855abffc
    https://www.researchgate.net/post/What-are-possible-reasons-why-Q-loss-is-not-converging-in-Deep-Q-Learning-algorithm
    """

    def __init__(self, model=None, epsilon=1.0, epsilon_min=0.15):

        self.StateTrans = StateTranslator_Central()
        self.state_shape = 97
        print('my state shape is:', self.state_shape)
        self.memory = deque(maxlen=5000)
        self.gamma = 0.95
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = 0.999
        self.learning_rate = 0.001
        self.tau = .125

        if model == None:
            self.model = self.create_model()
        else:
            self.model = model
        self.target_model = self.create_model()

    def create_model(self):
        model = Sequential()
        model.add(Dense(2000, input_dim=self.state_shape, activation="relu"))
        model.add(Dense(1000, activation="relu"))
        model.add(Dense(500, activation="relu"))
        model.add(Dense(1000, activation="relu"))
        model.add(Dense(500, activation="relu"))
        model.add(Dense(100, activation="relu"))
        model.add(Dense(4))
        model.compile(loss="MSE",
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def act(self, state):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        if np.random.random() < self.epsilon:
            # Set random choice to north or east as the agent is not moving in these directions
            return random.choice([0, 1, 2, 3])

        action_values = self.model.predict(state.reshape(-1, self.state_shape))[0]
        action = np.argmax(action_values)

        return action

    def translate_state(self, observation, configuration):
        state = self.StateTrans.get_state(observation, configuration)
        return state

    def __call__(self, observation, configuration):

        state = self.translate_state(observation, configuration)
        action = self.act(state)
        # State translator will take in 0, 1, 2, 3 and return straight, left or right, which in turn will
        # be translated into a kaggle Action
        action_text = self.StateTrans.translate_action_to_text(action)

        # Update our step number and actions
        self.StateTrans.set_last_action(action_text)

        return action_text

    def remember(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done])

    def replay(self):
        batch_size = 64
        if len(self.memory) < batch_size:
            return

        samples = random.sample(self.memory, batch_size)
        ########################
        # This can be sped up significantly, but processing all samples in batch rather than 1 at a time
        ####################
        states = np.array([])
        actions = np.array([])
        rewards = []
        dones = []
        new_states = np.array([])
        targets = np.array([])

        for sample in samples:
            state, action, reward, new_state, done = sample

            states = np.append(states, state)
            actions = np.append(actions, action)
            rewards.append(reward)
            new_states = np.append(new_states, new_state)
            dones.append(done)

        new_states = new_states.reshape(batch_size, self.state_shape)
        targets = self.target_model.predict(states.reshape(batch_size, self.state_shape))
        targets = targets.reshape(batch_size, 4)
        for i in range(batch_size):
            if dones[i]:
                targets[i][int(actions[i])] = rewards[i]

            else:
                Q_future = max(self.target_model.predict(new_states[i].reshape(-1, self.state_shape))[0])
                #                 print('targets i', targets[i])
                #                 print('actions[i]', actions[i])
                targets[i][int(actions[i])] = rewards[i] + Q_future * self.gamma

        self.model.fit(states.reshape(batch_size, self.state_shape), targets,
                       epochs=1, verbose=0)

    def target_train(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.target_model.set_weights(target_weights)

    def save_model(self, fn):
        self.model.save(fn)
