from kaggle_environments import make
from geese.heuristic_agents import GreedyAgent, SuperGreedyAgent
from geese.dqn import dqnAgent
import pickle, os
from tensorflow import keras
from copy import deepcopy
from geese.StateTranslator import StateTranslator_TerminalRewards

import tensorflow as tf
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))

steps_per_ep = 200
num_episodes = 10000

env = make("hungry_geese", debug=True)
config = env.configuration
model_dir = 'Models'
train_name = 'gpu-test'
directory = os.sep.join([model_dir, train_name])

mod_num = 0# Which trial to load
state_translator = StateTranslator_TerminalRewards()
epsilon = .95
epsilon_min = .05

if mod_num > 0:
    model = keras.models.load_model(f'{directory}/trial-{mod_num}')
    dqn = dqnAgent(model=model,
                   state_translator=state_translator,
                   epsilon=epsilon,
                   epsilon_min=epsilon_min)
    dqn.target_model = model
else:
    dqn = dqnAgent(epsilon=epsilon,
                   epsilon_min=epsilon_min)

model_competitor = keras.models.load_model(f'Models/terminal_transfer_learning/trial-12000')

# Want a little epsilon to add some variablitity to actions so we don't overfit on this opponent
dqn_competitor = dqnAgent(model=model_competitor,
                          epsilon=0,
                          epsilon_min=0)

agent3 = SuperGreedyAgent()
agent4 = GreedyAgent()

agents = [dqn, dqn_competitor, agent3, agent4]

results_dic = {}
wins = 0
for ep in range(num_episodes):
    print('episode number: ', ep+mod_num)
    state_dict = env.reset(num_agents=4)[0]
    observation = state_dict['observation']
    my_goose_ind = observation['index']

    dqn.StateTrans.set_last_action(None)
    dqn.StateTrans.last_goose_length = 1
    cur_state = dqn.StateTrans.get_state(observation, config)

    done = False
    for step in range(steps_per_ep):
        actions = []
        for ind, agent in enumerate(agents):
            obs_copy = deepcopy(observation)
            obs_copy['index'] = ind
            action = agent(obs_copy, config)
            actions.append(action)

        state_dict = env.step(actions)[0]
        observation = state_dict['observation']

        action = state_dict['action']
        status = state_dict['status']

        action_for_model = dqn.StateTrans.translate_text_to_int(action)
        new_state = dqn.StateTrans.get_state(observation, config)

        # Set rewards based on if value was gained or lost
        reward = dqn.StateTrans.calculate_reward(observation)
        if reward > 100:
            wins += 1
        # Update our goose length based on prev state
        dqn.StateTrans.update_length()

        if status != "ACTIVE":
            done = True

        dqn.remember(cur_state, action_for_model, reward, new_state, done)

        cur_state = new_state

        # Check if my goose died

        if done:
            print('Done, Step: ', step)
            print('status, ', status)
            print('Reward: ', reward)
            print('Trial: ', ep)
            results_dic[ep] = reward

            if ep % 50 == 0:
                dqn.save_model(directory + f"/trial-{ep+mod_num}")
                with open(directory + "/results_dic.pkl", 'wb') as f:
                    pickle.dump(results_dic, f)

                # Have we won at least 22 of our last 50 games? Then we update the competitor model
                print('win percentage: ', wins/50)
                if wins > 22:
                    print('Updating competitor model')
                    # I'm scared of using the same instance of the model in two seperate classes... deepcopy didn't work so
                    # I'll just use this fix for now
                    model1 = keras.models.load_model(f'{directory}/trial-{ep + mod_num}')
                    dqn_competitor.model = model1
                wins = 0
            break


        if step % 5 == 0:
            dqn.replay()
            dqn.target_train()

        # Every 250 steps, we update the competitor model to not overfit to one agents strats
