# COMP30024 Artificial Intelligence, Semester 1 2023
# Project Part B: Game Playing Agent

from referee.game import \
    PlayerColor, Action, SpawnAction, SpreadAction, HexPos, HexDir, constants, Board
from .minimax import *
from .monteCarlo import *
from .boardRelated import *
import time


# This is the entry point for your game playing agent. Currently the agent
# simply spawns a token at the centre of the board if playing as RED, and
# spreads a token at the centre of the board if playing as BLUE. This is
# intended to serve as an example of how to use the referee API -- obviously
# this is not a valid strategy for actually playing the game!

class Agent:
    def __init__(self, color: PlayerColor, **referee: dict):
        """
        Initialise the agent.
        """
        self._color = color
        self.state = Board()
        # initialise the board state
        for i in range(constants.BOARD_N):
            for j in range(constants.BOARD_N):
                self.state[HexPos(i, j)]

        match color:
            case PlayerColor.RED:
                print("Testing: I am playing as red")
            case PlayerColor.BLUE:
                print("Testing: I am playing as blue")

    def action(self, **referee: dict) -> Action:
        """
        Return the next action to take.
        """
        max_color = self.state._turn_color
        transposition_table = {}
        # start_time = time.time()
        # access the next action through minimax
        next_action, _ = minimax_with_pruning(self.state, max_color, float('-inf'), float('inf'), 2, transposition_table)
        # next_action = monte_carlo_tree_search(self.state, 200)
        # end_time = time.time()
        # print(end_time - start_time)
        match self._color:
            case PlayerColor.RED:
                return next_action
            case PlayerColor.BLUE:
                return next_action

    def turn(self, color: PlayerColor, action: Action, **referee: dict):
        """
        Update the agent with the last player's action.
        """
        match action:
            case SpawnAction(cell):
                print(f"Testing: {color} SPAWN at {cell}")
                pass
            case SpreadAction(cell, direction):
                print(f"Testing: {color} SPREAD from {cell}, {direction}")
                pass
        # update the board state
        self.state.apply_action(action)
