from kaggle_environments.envs.hungry_geese.hungry_geese import Observation, Configuration, Action, \
                                                                row_col, adjacent_positions
import numpy as np
from copy import deepcopy

def geese_heads(obs_dict, config_dict):
    """
    Return the position of the geese's heads
    """
    configuration = Configuration(config_dict)

    observation = Observation(obs_dict)
    player_index = observation.index
    player_goose = observation.geese[player_index]
    player_head = player_goose[0]
    player_row, player_column = row_col(player_head, configuration.columns)
    positions = []
    for geese in observation.geese:
        if len(geese) > 0:
            geese_head = geese[0]
            row, column = row_col(geese_head, configuration.columns)
        else:
            row = None
            column = None
        positions.append((row, column))
    return positions


def get_last_actions(previous_geese_heads, heads_positions):
    def get_last_action(prev, cur):
        last_action = None

        prev_row = prev[0]
        prev_col = prev[1]
        cur_row = cur[0]
        cur_col = cur[1]

        if cur_row is not None:
            if (cur_row - prev_row == 1) | ((cur_row == 0) & (prev_row == 6)):
                last_action = Action.SOUTH.name
            elif (cur_row - prev_row == -1) | ((cur_row == 6) & (prev_row == 0)):
                last_action = Action.NORTH.name
            elif (cur_col - prev_col == 1) | ((cur_col == 0) & (prev_col == 10)):
                last_action = Action.EAST.name
            elif (cur_col - prev_col == -1) | ((cur_col == 10) & (prev_col == 0)):
                last_action = Action.WEST.name

        return last_action

    if len(previous_geese_heads) == 0:
        actions = [Action.SOUTH.name, Action.NORTH.name, Action.EAST.name, Action.WEST.name]
        nb_geeses = len(heads_positions)
        last_actions = [actions[np.random.randint(4)] for _ in range(nb_geeses)]
    else:
        last_actions = [get_last_action(*pos) for pos in zip(previous_geese_heads, heads_positions)]

    return last_actions


def central_state_space(obs_dict, config_dict, prev_head):
    """
    Recreating a board where my agent's head in the middle of the board
    (position (4,5)), and creating features accordingly
    """

    configuration = Configuration(config_dict)

    observation = Observation(obs_dict)
    player_index = observation.index
    player_goose = observation.geese[player_index]
    if len(player_goose) == 0:
        player_head = prev_head
    else:
        player_head = player_goose[0]
    player_row, player_column = row_col(player_head, configuration.columns)


    def centralize(row, col):
        if col > player_column:
            new_col = (5 + col - player_column) % 11
        else:
            new_col = 5 - (player_column - col)
            if new_col < 0:
                new_col += 11

        if row > player_row:
            new_row = (3 + row - player_row) % 7
        else:
            new_row = 3 - (player_row - row)
            if new_row < 0:
                new_row += 7
        return new_row, new_col

    # Create the grid
    board = np.zeros([7, 11])
    # Add food to board
    #     board[food1_row, food1_column] = 1
    #     board[food2_row, food2_column] = 1

    for ind, goose in enumerate(observation.geese):
        if ind != player_index:
            if len(goose) > 0:
                ap = adjacent_positions(goose[0], 11, 7)
                for p in ap:
                    row, col = centralize(*row_col(p, configuration.columns))
                    # 2 Represents adjacent heads
                    board[row, col] = 2

                # Set enemy heads to 3
                row, col = centralize(*row_col(goose[0], configuration.columns))
                board[row, col] = 3
        # Starting at 4 and going to 10, to embed goose bodies not including the head
        for ind, pos in enumerate(reversed(goose[1:])):
            # set bodies to 1
            if ind<=6:
                row, col = centralize(*row_col(pos, configuration.columns))
                board[row, col] = ind + 4
            if ind>6:
                row, col = centralize(*row_col(pos, configuration.columns))
                board[row, col] = 11


    # 1 represents the player head for embeddings
    board[3, 5] = 1

    return board


def min_dir(p1, p2, max_p):
    """
    min distance and direction from p1
    """
    direction = 'left'  # left by default
    d1 = abs(p1 - p2)  # Distance going across board
    d2 = min(abs(p1 - 0), abs(p1 - max_p)) + min(abs(p2 - 0), abs(p2 - max_p))  # Distance wrapping around board

    if p1 > p2 and d1 < d2:
        direction = 'left'

    elif p1 > p2 and d1 > d2:
        direction = 'right'

    elif p2 > p1 and d1 < d2:
        direction = 'right'

    elif p2 > p1 and d1 > d2:
        direction = 'left'

    dir_vec = np.zeros(2)

    if direction == 'left':
        dir_vec[0] = 1

    else:
        dir_vec[1] = 1

    min_dist = np.array([min(d1, d2)]) / max_p
    return dir_vec, min_dist


class StateTranslator_Embeddings:
    """
    Returns a board where we are always at the center
    """

    def __init__(self):

        self.last_action = None
        self.step_count = 0
        self.last_goose_length = 1
        self.last_goose_ind = 0
        self.observations = []

    def set_last_action(self, last_action):
        self.last_action = last_action

    def __get_last_action_vec(self):
        action_vec = np.zeros(4)

        if self.last_action == 'NORTH':
            action_vec[0] = 1
        elif self.last_action == 'SOUTH':
            action_vec[1] = 1
        elif self.last_action == 'EAST':
            action_vec[2] = 1
        elif self.last_action == 'WEST':
            action_vec[3] = 1

        return action_vec

    def translate_action_to_text(self, action):

        h = {0: 'WEST',
             1: 'EAST',
             2: 'NORTH',
             3: 'SOUTH'}

        return h[action]

    def translate_text_to_int(self, action):
        h = {'WEST': 0,
             'EAST': 1,
             'NORTH': 2,
             'SOUTH': 3}

        return h[action]

    def update_length(self):
        self.last_goose_length = self.current_goose_length

    def get_state(self, observation, config):

        #### This is exception handling for if our goose died this turn, to use the last
        ### known index as its postion for the state centralizer
        geese = observation['geese']
        if len(geese[observation['index']]) > 0:
            self.last_goose_ind = geese[observation['index']][0]
            self.my_goose = geese[observation['index']][0]

        board = central_state_space(observation, config, self.last_goose_ind)
        self.step_count = observation['step']

        self.current_goose_length = len(geese[observation['index']])

        food = observation['food']
        dir_vec1x, min_dist1x = min_dir(self.my_goose, food[0], 11)
        dir_vec1y, min_dist1y = min_dir(self.my_goose, food[0], 7)

        dir_vec2x, min_dist2x = min_dir(self.my_goose, food[1], 11)
        dir_vec2y, min_dist2y = min_dir(self.my_goose, food[1], 7)
        food_vec = np.concatenate((dir_vec1x, dir_vec1y, dir_vec2x, dir_vec2y,
                                   min_dist1x, min_dist1y, min_dist2x, min_dist2y))
        ####
        biggest_goose = 0
        alive_geese = 0
        for ind, goose in enumerate(geese):
            if len(goose) > biggest_goose:
                biggest_goose = len(goose)
            if ind != observation['index'] and len(goose) > 0:
                alive_geese += 1

        state = np.array([])
        state = np.append(state, self.__get_last_action_vec())
        state = np.append(state, self.current_goose_length / 15)
        state = np.append(state, biggest_goose / 15)
        state = np.append(state, alive_geese / 4)
        state = np.append(state, self.step_count / 200)
        state = np.append(state, food_vec)

        num_inp = deepcopy(state.reshape(1,-1))

        cat_input = board.flatten().reshape(1,-1)

        return [num_inp, cat_input]


    def calculate_reward(self, observation):

        current_geese = observation['geese']
        prev = self.last_goose_length
        cur = len(current_geese[observation['index']])

        reward = -1

        ## If we grow, reward is 100
        if cur > prev:
            reward = 100

        ### If we die -150
        if cur == 0:
            reward = -200

        ### see if any geese are alive

        alive_geese = 0
        for ind, goose in enumerate(current_geese):
            if ind != observation['index'] and len(goose) > 0:
                alive_geese += 1

        # If we are the last one standing
        if alive_geese == 0 and cur > 0:
            reward = 1000

        ## if the game ends and we are the biggest
        if self.step_count == 200:
            biggest_goose = 0
            biggest_goose_ind = None
            for ind, goose in enumerate(current_geese):
                if len(goose) > biggest_goose:
                    biggest_goose = len(goose)
                    biggest_goose_ind = ind

            if biggest_goose_ind == observation['index']:
                reward = 1000

        return reward