from referee.game import \
    PlayerColor, Action, SpawnAction, SpreadAction, HexPos, HexDir, constants, Board
from .boardRelated import *

def minimax_with_pruning(state: Board, max_color: PlayerColor, alpha, beta, depth, tt):
    state_hash = compute_hash(state)
    if (state_hash in tt) and tt[state_hash][2] < depth:
        # print("Transition table is working!!")
        return tt[state_hash][0], tt[state_hash][1]
    
    # return all the possible next actions
    action_list = get_action_list(state)
    # top_k_list = top_k_actions(state, action_list, max_color, 5)
    # if reach to the end level
    if depth == 0 or state.game_over == True:
        return None, evaluate(state, max_color)
    # it's our turn
    if state._turn_color == max_color:
        max_value = float('-inf')
        best_action = None
        for action in action_list:
            state.apply_action(action)
            _, evaluation = minimax_with_pruning(state, max_color, alpha, beta, depth - 1, tt)
            state.undo_action()
            # find the maximum pay off by comparing each evaluations
            if evaluation > max_value:
                max_value = evaluation
                best_action = action
            # alpha beta pruing
            alpha = max(alpha, max_value)
            if alpha >= beta:
                break
        
        # transposition table recording
        tt[state_hash] = (best_action, max_value, depth)
        return best_action, max_value
    else: # opponent's turn
        min_value = float('inf')
        best_action = None
        for action in action_list:
            state.apply_action(action)
            _, evaluation = minimax_with_pruning(state, max_color, alpha, beta, depth - 1, tt)
            state.undo_action()
            if evaluation < min_value:
                min_value = evaluation
                best_action = action
            beta = min(min_value, beta)
            if beta <= alpha:
                break

        tt[state_hash] = (best_action, min_value, depth)
        return best_action, min_value

# find the hash value of the board state
def compute_hash(state: Board):
    return hash(state._state.__str__())