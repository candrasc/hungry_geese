from kaggle_environments.envs.hungry_geese.hungry_geese import Observation, Configuration, Action, min_distance, \
                                                                adjacent_positions, translate

from random import choice
from copy import deepcopy
import numpy as np

class GreedyAgent:
    def __init__(self):
        self.last_action = None
        self.observations = []

    def __call__(self, observation: Observation, configuration: Configuration):
        self.configuration = configuration

        board = np.zeros(self.configuration.rows * self.configuration.columns)
        board_shape = (self.configuration.rows, self.configuration.columns)

        board_heads = deepcopy(board)
        board_bodies = deepcopy(board)
        board_rewards = deepcopy(board)

        rows, columns = self.configuration.rows, self.configuration.columns

        food = observation.food
        geese = observation.geese

        opponents = [
            goose
            for index, goose in enumerate(geese)
            if index != observation.index and len(goose) > 0
        ]

        opponent_heads = [opponent[0] for opponent in opponents]
        # Don't move adjacent to any heads
        head_adjacent_positions = {
            opponent_head_adjacent
            for opponent_head in opponent_heads
            for opponent_head_adjacent in adjacent_positions(opponent_head, columns, rows)
        }

        # Don't move into any bodies
        # bodies, heads = [position for goose in geese for position in goose]

        heads = [i[0] for i in geese if len(i) > 1]
        bodies = [item for sublist in geese for item in sublist]

        board_bodies[list(bodies)] = 1
        board_heads[heads] = 1

        # Move to the closest food
        position = geese[observation.index][0]
        actions = {
            action: min_distance(new_position, food, columns)
            for action in Action
            for new_position in [translate(position, action, columns, rows)]
            if (
                    new_position not in head_adjacent_positions and
                    new_position not in bodies and
                    (self.last_action is None or action != self.last_action.opposite())
            )
        }

        action = min(actions, key=actions.get) if any(actions) else choice([action for action in Action])

        self.last_action = action
        return action.name


class SuperGreedyAgent:
    def __init__(self):
        self.last_action = None
        self.observations = []

    def __call__(self, observation: Observation, configuration: Configuration):
        self.configuration = configuration

        board = np.zeros(self.configuration.rows * self.configuration.columns)
        board_shape = (self.configuration.rows, self.configuration.columns)

        board_heads = deepcopy(board)
        board_bodies = deepcopy(board)
        board_rewards = deepcopy(board)

        rows, columns = self.configuration.rows, self.configuration.columns

        food = observation.food
        geese = observation.geese

        opponents = [
            goose
            for index, goose in enumerate(geese)
            if index != observation.index and len(goose) > 0
        ]

        opponent_heads = [opponent[0] for opponent in opponents]
        # Don't move adjacent to any heads
        head_adjacent_positions = {
            opponent_head_adjacent
            for opponent_head in opponent_heads
            for opponent_head_adjacent in adjacent_positions(opponent_head, columns, rows)
        }

        # Don't move into any bodies
        # bodies, heads = [position for goose in geese for position in goose]

        heads = [i[0] for i in geese if len(i) > 1]
        bodies = [item for sublist in geese for item in sublist]

        board_bodies[list(bodies)] = 1
        board_heads[heads] = 1

        # Move to the closest food
        position = geese[observation.index][0]
        actions = {
            action: min_distance(new_position, food, columns)
            for action in Action
            for new_position in [translate(position, action, columns, rows)]
            if (
                    new_position not in bodies and
                    (self.last_action is None or action != self.last_action.opposite())
            )
        }

        action = min(actions, key=actions.get) if any(actions) else choice([action for action in Action])

        self.last_action = action
        return action.name