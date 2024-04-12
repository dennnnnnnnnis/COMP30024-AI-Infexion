from referee.game import \
    PlayerColor, Action, SpawnAction, SpreadAction, HexPos, HexDir, constants, Board
from .boardRelated import *
import random
import math
import copy
import csv

class Node:
    def __init__(self, state: Board, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.num_wins = 0
        self.action = None
        self.simulations = 0
        self.evaluation = -1

    def simulate(self):
        copied_board = copy.deepcopy(self.state)
        curr = copied_board
        while curr.game_over != True:
            curr = self.playout(curr)
        # print(curr.render(True))
        return self.result(curr)
    
    def playout(self, state: Board):
        action_list = get_action_list(state)
        state.apply_action(random.choice(action_list))
        # top k controls the next possible action
        # children_list = sorted([update_board(state, action) for action in action_list], 
                   # key=lambda child : evaluate(child, state.turn_color), reverse=True)[:5]
        return state

    def result(self, state: Board):
        # get_feature_vector(state, state.winner_color)
        return state.winner_color
    
    def ubc1(self, c=1.0):
        if self.simulations == 0:
            return float('inf')
        exploitation = self.num_wins / self.simulations
        exploration = math.sqrt(math.log(self.parent.simulations) / self.simulations)
        return exploitation + c * exploration
    
    def select_best_child(self):
        curr = self
        while len(curr.children) != 0:
            curr = max(curr.children, key=lambda child: child.ubc1())
        return curr
    
    def expansion(self):
        action_list = get_action_list(self.state)
        for action in action_list:
            self.state.apply_action(action)
            child = Node(self.state, parent=self)
            child.evaluation = evaluate(child.state, child.state.turn_color)
            child.action = action
            self.state.undo_action()
            self.children.append(child)
    
    def backpropagation(self, result: PlayerColor):
        if self.state.turn_color == result:
            self.num_wins += 1
        self.simulations += 1
        if self.parent != None:
            self.parent.backpropagation(result)


def monte_carlo_tree_search(state: Board, num_simulations):
    root = Node(state)
    root.expansion()
    
    while num_simulations > 0:
        leaf = root.select_best_child()
        child = leaf
        if leaf.state.game_over == False:
            leaf.expansion()
            child = random.choice(leaf.children)
        result = child.simulate()
        # print(result)
        child.backpropagation(result)
        num_simulations -= 1
    
    best_child = max(root.children, key=lambda child: child.simulations)
    return best_child.action


def get_feature_vector(state: Board, result: PlayerColor):
    feature_dict = {}
    feature_dict["num_red_cells"] = len(state._player_cells(PlayerColor.RED))
    feature_dict["num_blue_cells"] = len(state._player_cells(PlayerColor.BLUE))
    feature_dict["num_blank_cells"] = constants.MAX_TOTAL_POWER - feature_dict["num_red_cells"] - feature_dict["num_blue_cells"]
    feature_dict["red_total_power"] = state._color_power(PlayerColor.RED)
    feature_dict["blue_total_power"] = state._color_power(PlayerColor.BLUE)
    feature_dict["turn_count"] = state.turn_count
    feature_dict["turn_color"] = state.turn_color.value
    if result == None:
        feature_dict["result"] = -1
    else:
        feature_dict["result"] = result.value

    with open('/Users/wangzeyu/Desktop/AI/part_b/agent/game_states.csv', 'a', newline='') as file:
        # Create a CSV writer object
        writer = csv.writer(file)

        # Write a new row to the CSV file
        writer.writerow([feature_dict["num_red_cells"], feature_dict["num_blue_cells"], 
                         feature_dict["num_blank_cells"], feature_dict["red_total_power"], 
                         feature_dict["blue_total_power"], feature_dict["turn_count"], 
                         feature_dict["turn_color"], feature_dict["result"]])
    return feature_dict
