from referee.game import \
    PlayerColor, Action, SpawnAction, SpreadAction, HexPos, HexDir, constants, Board
import copy
import math
import pickle
import numpy as np

# Load the saved model
# with open("agent/clf.pkl", 'rb') as f:
    # clf = pickle.load(f)
    # clf.feature_names_in_ = None

def get_action_list(state: Board):
    action_list = []
    total_power = 0
    for coord in state._state:
        total_power += state._state[coord].power
        if state._state[coord].player == state._turn_color:
            for direction in HexDir: # spread in 6 directions
                action_list.append(SpreadAction(coord, direction))

    if total_power < constants.MAX_TOTAL_POWER:
        for coord in state._state:
            if state._state[coord].player == None: # all the empty cells
                action_list.append(SpawnAction(coord))
    return action_list

def update_board(state: Board, action: Action):
    copied_board = copy.deepcopy(state)
    copied_board.apply_action(action)
    return copied_board

def coords_difference(state: Board, max_color: PlayerColor):
    max_coord = 0
    min_coord = 0
    for coord in state._state:
        if state._state[coord].player == max_color:
            max_coord += 1
        elif state._state[coord].player == max_color.opponent:
            min_coord += 1
    
    return max_coord - min_coord

def power_difference(state: Board, max_color: PlayerColor):
    power_diff = 0
    max_power = {}
    min_power = {}

    for power in range(1, 7):
        max_power[power] = 0
        min_power[power] = 0

    # find the difference of cell's power respectively from power 1 to 7
    for coord in state._state:
        if state._state[coord].player == max_color:
            max_power[state._state[coord].power] += 1
        elif state._state[coord].player == max_color.opponent:
            min_power[state._state[coord].power] += 1
    
    j = 1
    # 1, 6, 11, 16, 21, 26
    for i in range(1, 7):
        power_diff += (max_power[i] - min_power[i]) * j
        j += 5

    return power_diff

def spread_extra(state: Board, max_color: PlayerColor):
    spread_price = 0
    power_dict = {}

    # 5, 10, 15, 20, 25, 30
    j = 5
    for i in range(1, 7):
        power_dict[i] = j
        j += 5
    
    for item in state._history:
        if item.action.__class__ == SpreadAction:
            for cell in item.cell_mutations:
                if cell.prev.player == max_color.opponent and cell.next.player == max_color:
                    spread_price += power_dict[cell.prev.power]
                elif cell.prev.player == max_color and cell.next.player == max_color.opponent:
                    spread_price -= power_dict[cell.prev.power]
                elif cell.next.player == None and cell.prev.power == 6:
                    if cell.prev.player == max_color:
                        spread_price -= power_dict[cell.prev.power]
                    else:
                        spread_price += power_dict[cell.prev.power]
    
    return spread_price

def evaluate(state: Board, max_color: PlayerColor):
    evaluation = spread_extra(state, max_color) + 1.6 * coords_difference(state, max_color) + \
                    power_difference(state, max_color)
    return evaluation

def evaluate_1(state: Board, max_color: PlayerColor):
    if state._color_power(max_color) == 0:
        evaluation = 0
    elif state._color_power(max_color.opponent) == 0:
        evaluation = float('inf')
    else:
        evaluation = math.pow(state._color_power(max_color), len(state._player_cells(max_color))) \
                    / math.pow(state._color_power(max_color.opponent), len(state._player_cells(max_color.opponent)))
    return evaluation

""" def MLP_prediction(state: Board, max_color: PlayerColor):
    feature_dict = get_feature_vector(state)
    features = []
    for key in feature_dict:
        features.append(feature_dict[key])
    prob = clf.predict_proba([features])
    labels = clf.classes_

    return prob[0][np.where(labels == max_color.value)[0][0]] """


def get_feature_vector(state: Board):
    feature_dict = {}
    feature_dict["num_red_cells"] = len(state._player_cells(PlayerColor.RED))
    feature_dict["num_blue_cells"] = len(state._player_cells(PlayerColor.BLUE))
    feature_dict["num_blank_cells"] = constants.MAX_TOTAL_POWER - feature_dict["num_red_cells"] - feature_dict["num_blue_cells"]
    feature_dict["red_total_power"] = state._color_power(PlayerColor.RED)
    feature_dict["blue_total_power"] = state._color_power(PlayerColor.BLUE)
    feature_dict["turn_count"] = state.turn_count
    feature_dict["turn_color"] = state.turn_color.value
    return feature_dict


def top_k_actions(state: Board, actions, max_color: PlayerColor, k):
    states = [update_board(state, action) for action in actions]
    evaluations = map(lambda board: evaluate(board, max_color), states)
    top_k = list(zip(states, evaluations))
    top_k.sort(key = lambda x: x[1], reverse=True)
    if state.turn_color == max_color:
        top_k_states = [board[0] for board in top_k[:k]]
        # top_k_action = [board[0]._history[-1].action for board in top_k[:k]]
    else:
        top_k_states = [board[0] for board in top_k[-k:]]
        # top_k_action = [board[0]._history[-1].action for board in top_k[-k:]]
    return top_k_states
