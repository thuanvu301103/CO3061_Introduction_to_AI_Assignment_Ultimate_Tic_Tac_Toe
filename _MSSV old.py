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
            
            next_state = State_2(state)
            next_state.free_move = state.free_move
            
            next_state.act_move(move)
            value = max(value, minimax_pruning(next_state, depth - 1, alpha, beta, False))
            alpha = max(alpha, value)
            if beta <= alpha:
                break  # pruning
        return value
    else:
        value = math.inf
        for move in legal_moves:
            
            next_state = State_2(state)
            next_state.free_move = state.free_move
            
            next_state.act_move(move)
            value = min(value, minimax_pruning(next_state, depth - 1, alpha, beta, True))
            beta = min(beta, value)
            if beta <= alpha:
                break  # pruning
        return value


def evaluate_state(state):
    winner = state.game_result(state.global_cells.reshape(3, 3))
    # Winning the game
    if winner == state.player_to_move:
        return 50  # Player wins
    elif winner == state.player_to_move*(-1):
        return -50  # Opponent wins
    elif state.game_over:
        return 0  # It's a draw

    score = 0;
    # Custom scoring for the game state
    for x in range(0,9):

        #block_winner = state.game_result(block)
        if state.game_result(state.blocks[x]) == state.player_to_move:
            # Feature: Small board wins add 5 points
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

        # small board
        # Row 
        for i in range(0,3):
            sum = 0
            for j in range(0,3): sum += state.blocks[x][i, j]
            if (sum == 2 or sum == -2):
                if sum/2 == state.player_to_move: 
                    score += 2
                else: score -= 2

        # Column boards potential
        for j in range(0,3):
            sum = 0
            for i in range(0,3): sum += state.blocks[x][j, i]
            if (sum == 2 or sum == -2):
                if sum/2 == state.player_to_move: 
                    score += 2
                else: score -= 2

        # Cross boards potential
        sum = 0
        for i in range(0,3):
            sum += state.blocks[x][i, i]
        if (sum == 2 or sum == -2):
            if sum/2 == state.player_to_move: 
                score += 2
            else: score -= 2

        sum = 0
        for i in range(0,3):
            sum += state.blocks[x][i, 2-i]
        if (sum == 2 or sum == -2):
            if sum/2 == state.player_to_move: 
                score += 2
            else: score -= 2

    # Row boards potential
    for j in [0,3,6]:
        sum = 0
        for i in range(0,3): sum += state.global_cells[i+j]
        if (sum == 2 or sum == -2):
            if sum/2 == state.player_to_move: 
                score += 4
            else: score -= 4

    # Column boards potential
    for j in range(0,3):
        sum = 0
        for i in [0,3,6]: sum += state.global_cells[i+j]
        if (sum == 2 or sum == -2):
            if sum/2 == state.player_to_move: 
                score += 4
            else: score -= 4

    # Cross boards potential
    sum = 0
    for i in [0,4,8]:
        sum += state.global_cells[i]
    if (sum == 2 or sum == -2):
        if sum/2 == state.player_to_move: 
            score += 4
        else: score -= 4

    sum = 0
    for i in [2,4,6]:
        sum += state.global_cells[i]
    if (sum == 2 or sum == -2):
        if sum/2 == state.player_to_move: 
            score += 4
        else: score -= 4

    return score
        

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
        value = minimax_pruning(next_state, depth=1, alpha=-math.inf, beta=math.inf, maximizing_player=False)

        if value >= best_value:
            best_value = value
            best_move = move
    #print ("Best move: ")
    #print (best_move)
    
    return best_move

