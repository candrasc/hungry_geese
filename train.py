from kaggle_environments import make
from geese.heuristic_agents import GreedyAgent
from geese.dqn import dqnAgent
import pickle
import keras

steps_per_ep = 200
num_episodes = 10000

env = make("hungry_geese", debug=True)
config = env.configuration
train_name = 'central_agent'
#model = keras.models.load_model('submissions/submission_dqn_04_18/central_agent/trial-6890')

dqn = dqnAgent(#model = model,
               epsilon = 0.95,
               epsilon_min= 0.10)

agent2 = GreedyAgent()
agent3 = GreedyAgent()
agent4 = GreedyAgent()

agents = [dqn, agent2, agent3, agent4]

results_dic = {}
for ep in range(num_episodes):
    print('episode number: ', ep + mod_num)
    state_dict = env.reset(num_agents=4)[0]
    observation = state_dict['observation']
    my_goose_ind = observation['index']

    dqn.StateTrans.set_last_action(None)
    dqn.StateTrans.step_count = 0
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
        print(observation)

        action = state_dict['action']
        status = state_dict['status']

        action_for_model = dqn.StateTrans.translate_text_to_int(action)
        new_state = dqn.StateTrans.get_state(observation, config)

        # Set rewards based on if value was gained or lost
        reward = dqn.StateTrans.calculate_reward(observation)
        # Update our goose length based on prev state
        dqn.StateTrans.update_length()

        if status != "ACTIVE":
            done = True

        # Temp for just training agent to go get food

        #         if reward<-1:
        #             reward = -1
        print('reward: ', reward)
        dqn.remember(cur_state, action_for_model, reward, new_state, done)

        cur_state = new_state

        # Check if my goose died

        if done:
            print('Done, Step: ', step)
            print('status, ', status)
            results_dic[ep] = reward

            if ep % 50 == 0:
                directory = train_name
                dqn.save_model(directory + f"/trial-{ep + mod_num}")
                with open(directory + "/results_dic.pkl", 'wb') as f:
                    pickle.dump(results_dic, f)
            break

        if step % 5 == 0:
            dqn.replay()
            dqn.target_train()