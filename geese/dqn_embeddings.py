from geese.dqn import dqnAgent
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, BatchNormalization, Input, Embedding, Flatten, concatenate, Reshape
from tensorflow. keras.optimizers import Adam
from geese.StateTranslator_Embeddings import StateTranslator_Embeddings
from collections import deque
import numpy as np
import random

class dqnEmbeddings(dqnAgent):

    def __init__(self, model=None, state_translator = None, epsilon=1.0, epsilon_min=0.05):

        if state_translator==None:
            self.StateTrans = StateTranslator_Embeddings()
        else:
            self.StateTrans = state_translator
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

    def create_model(self, num_input_size=20, embedding_size=2, all_trainable=True):
        """
        https://mmuratarat.github.io/2019-06-12/embeddings-with-numeric-variables-Keras

        :param num_input_size:
        :param embedding_size:
        :param all_trainable:
        :return:
        """

        num_layers = []
        emb_layers = []

        # Add numeric inputs
        num_layers.append(Input(shape=num_input_size))

        # Add cat inputs in embed layer
        m = Sequential()
        embedding = Embedding(17, embedding_size, input_length=7 * 11, trainable=all_trainable)
        embedding._name = f'embeddings_1'
        m.add(embedding)
        m.add(Flatten(name=f'flat_embeddings-1'))
        emb_layers.append(m)

        inputs = num_layers + [emb_layers[0].input]
        outputs = num_layers + [emb_layers[0].output]

        c = concatenate(outputs)

        model = Dense(2000, activation='elu', trainable=all_trainable)(c)
        # model = BatchNormalization(trainable=all_trainable)(model)
        # model = Dropout(rate=0.2, input_shape=(200,), trainable=all_trainable)(model)
        model = Dense(1000, activation='elu', trainable=all_trainable)(model)
        # model = Dropout(rate=0.2, input_shape=(100,), trainable=all_trainable)(model)
        # model = BatchNormalization(trainable=all_trainable)(model)
        model = Dense(500, activation='elu', trainable=all_trainable)(model)
        model = Dense(1000, activation='elu', trainable=all_trainable)(model)
        model = Dense(500, activation='elu', trainable=all_trainable)(model)
        # model = BatchNormalization(trainable=all_trainable)(model)
        model = Dense(100, activation='elu')(model)
        out = Dense(4)(model)

        m = Model(inputs=inputs, outputs=out)
        m.compile(loss="MSE",
                      optimizer=Adam(lr=self.learning_rate))

        return m

    def act(self, state):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        if np.random.random() < self.epsilon:
            # Set random choice to north or east as the agent is not moving in these directions
            return random.choice([0, 1, 2, 3])

        action_values = self.model.predict(state)[0]
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
        # ########################
        # # This can be sped up significantly, but processing all samples in batch rather than 1 at a time
        # ####################
        # for sample in samples:
        #     state, action, reward, new_state, done = sample
        #     target = self.target_model.predict(state)
        #     if done:
        #         target[0][action] = reward
        #     else:
        #         Q_future = max(self.target_model.predict(new_state)[0])
        #         target[0][action] = reward + Q_future * self.gamma
        #     self.model.fit(state, target, epochs=1, verbose=0)


        states_num = []
        states_cat = []
        new_states_num = []
        new_states_cat = []

        actions = np.array([])
        rewards = []
        dones = []


        for sample in samples:
            state, action, reward, new_state, done = sample

            states_num.append(state[0])
            states_cat.append(state[1])
            new_states_num.append(new_state[0])
            new_states_cat.append(new_state[1])

            actions = np.append(actions, action)
            rewards.append(reward)
            dones.append(done)

        #print(states_num)
        states_cat = np.array(states_cat)
        states_cat = states_cat.reshape(64, 77)
        states_num = np.array(states_num)
        states_num = states_num.reshape(64, 20)

        targets = self.target_model.predict([states_num, states_cat])
        targets = targets.reshape(batch_size, 4)

        for i in range(batch_size):
            if dones[i]:
                targets[i][int(actions[i])] = rewards[i]

            else:
                Q_future = max(self.target_model.predict([states_num[i].reshape(1,-1), states_cat[i].reshape(1,-1)])[0])

                targets[i][int(actions[i])] = rewards[i] + Q_future * self.gamma

        self.model.fit([states_num, states_cat], targets,
                       epochs=1, verbose=0)