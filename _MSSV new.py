import numpy as np
from state import State_2
import math

def minimax_pruning(state, depth, alpha, beta, maximizing_player):
    if depth == 0 or state.game_over:
        return evaluate_state(state)

    legal_moves = state.get_valid_moves
    if maximizing_player:
        value = -math.inf
        for move in legal_moves:
            #print ("Move 2---: ")
            #print (move)
            next_state = State_2(state)
            next_state.free_move = state.free_move
            #print (next_state)
            #print ("Block: -------")
            #print (next_state.global_cells)
            next_state.act_move(move)
            value = max(value, minimax_pruning(next_state, depth - 1, alpha, beta, False))
            alpha = max(alpha, value)
            if beta <= alpha:
                break  # pruning
        return value
    else:
        value = math.inf
        for move in legal_moves:
            #print ("Move 3---: ")
            #print (move)
            next_state = State_2(state)
            next_state.free_move = state.free_move
            #print (next_state)
            #print ("Block: -------")
            #print (next_state.global_cells)
            next_state.act_move(move)
            value = min(value, minimax_pruning(next_state, depth - 1, alpha, beta, True))
            beta = min(beta, value)
            if beta <= alpha:
                break  # pruning
        return value

'''def evaluate_state(state):
    winner = state.game_result(state.global_cells.reshape(3, 3))
    #if winner == State_2.X:
    if winner == state.player_to_move:
        return 1  # Player wins
    elif winner == state.player_to_move*(-1):
        return -1  # Opponent wins
    elif state.game_over:
        return 0  # It's a draw
    
    # Custom scoring for the game state
    score = 0

    # Count the number of Xs and Os in the global cells
    count_X = state.count_X
    count_O = state.count_O

    # Ensure that count_X and count_O are scalar values
    score += count_X - count_O
    

    # Bonus for having winning moves in the global board
    winning_moves_X = sum(1 for block in state.blocks if state.game_result(block) == State_2.X)
    winning_moves_O = sum(1 for block in state.blocks if state.game_result(block) == State_2.O)
    
    score += (winning_moves_X - winning_moves_O) * 3
   
    # Bonus for controlling the center of the global board
    #center_cell = state.blocks[1][1]  # Center cell of the center block
    #score += center_cell * 2  # Add more weight if it's occupied
    
    return score'''



def evaluate_state(state):
    winner = state.game_result(state.global_cells.reshape(3, 3))
    # Winning the game
    if winner == state.player_to_move:
        return 50  # Player wins
    elif winner == state.player_to_move*(-1):
        return -50  # Opponent wins
    elif state.game_over:
        return 0  # It's a draw

    # Custom scoring for the game state
    score = 0

    # Feature: Small board wins add 5 points
    for x in [0,8]:

        #block_winner = state.game_result(block)
        if state.game_result(state.blocks[x]) == state.player_to_move:
            score += 5
            # Feature: Winning the center board adds 10
            if x == 4: score += 10
            # Feature: Winning the corner board
            if x in [0, 2, 6, 8]: score += 3

        elif state.game_result(state.blocks[x]) == state.player_to_move*(-1):
            score -= 5
            if x == 4: score -= 10
            if x in [0, 2, 6, 8]: score -= 3

        # Feature: Getting a center square in any small board is worth 3
        if state.blocks[x][1, 1] == state.player_to_move: 
            score += 3
            if x == 4: score += 3
        elif state.blocks[x][1, 1] == state.player_to_move*(-1): 
            score -= 3
            if x == 4: score -= 3



    return score
        
        
    

        # Feature: Two board wins which can be continued for a winning sequence are worth 4 points
        #for i in range(3):
        #    if (
        #        state.game_result(state.blocks[i]) == State_2.X
        #        and state.game_result(state.blocks[i + 3]) == State_2.X
        ##    ):
         #       score += 4
         #   elif (
        #        state.game_result(state.blocks[i]) == State_2.O
         #       and state.game_result(state.blocks[i + 3]) == State_2.O
         #   ):
          #      score -= 4

        # Feature: A similar sequence inside a small board is worth 2 points
        #for block in state.blocks:
         #   if (
          #      block[0, 0] == block[1, 1] == State_2.X
           #     or block[0, 2] == block[1, 1] == State_2.X
           # ):
            #    score += 2
           # elif (
            #    block[0, 0] == block[1, 1] == State_2.O
            #    or block[0, 2] == block[1, 1] == State_2.O
           # ):
            #    score -= 2

        # Feature: If you are sent to a small board that is full or won, add 2 points to the heuristic
        #if state.previous_move and state.blocks[
         #   state.previous_move.index_local_board
        #].all() != 0:
         #   score += 2'''
    


def select_move(cur_state, remain_time):
    legal_moves = cur_state.get_valid_moves
    best_move = None
    best_value = -math.inf

    if (cur_state.previous_move is None):
        return np.random.choice(legal_moves)

    for move in legal_moves:
        #print ("Move 1---: ")
        #print (move)
        next_state = State_2(cur_state)
        next_state.free_move = cur_state.free_move
        next_state.act_move(move)
        
        #next_state.player_to_move *= -1
        #next_state.previous_move = move
        value = minimax_pruning(next_state, depth=4, alpha=-math.inf, beta=math.inf, maximizing_player=False)

        if value > best_value:
            best_value = value
            best_move = move
    #print ("Best move: ")
    #print (best_move)
    
    return best_move

'''def select_move(cur_state, remain_time):
    # If player plays the first move => choose random cell
    valid_moves = cur_state.get_valid_moves
    if len(valid_moves) != 0:
        return np.random.choice(valid_moves)
    return None'''
