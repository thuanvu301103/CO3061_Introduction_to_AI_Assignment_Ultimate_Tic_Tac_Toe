import numpy as np
from state import *
import math
import random 

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
            if (state.free_move == True): value += 2
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
            if (state.free_move == True): value -= 2
            beta = min(beta, value)
            if beta <= alpha:
                break  # pruning
        return value

def evaluate_state(state):
    winner = state.game_result(state.global_cells.reshape(3, 3))

    if winner == state.player_to_move * (-1):
        return 100
    elif winner == state.player_to_move:
        return -100
    elif state.game_over:
        return 0

    score = 0
    #if (state.free_move == True and max_player == True): score += 2
    #elif (state.free_move == True and max_player == False): score -= 2

    # Cache results of game_result
    result_cache = {tuple(state.blocks[i].flatten()): state.game_result(state.blocks[i]) for i in range(9)}

    for x in range(9):
        block_result = result_cache[tuple(state.blocks[x].flatten())]

        if block_result == state.player_to_move* (-1):
            score += 5
            if x == 4:
                score += 10
            if x in [0, 2, 6, 8]:
                score += 3
        elif block_result == state.player_to_move:
            score -= 5
            if x == 4:
                score -= 10
            if x in [0, 2, 6, 8]:
                score -= 3

        center_square = state.blocks[x][1, 1]
        if center_square == state.player_to_move * (-1):
            score += 4
            if x == 4:
                score += 3
        elif center_square == state.player_to_move :
            score -= 4
            if x == 4:
                score -= 3

        for i in range(3):
            row_sum = np.sum(state.blocks[x][i, :])
            row_zero = np.count_nonzero(state.blocks[x][i, :])
            #print(row_zero)
            col_sum = np.sum(state.blocks[x][:, i])
            col_zero = np.count_nonzero(state.blocks[x][:, i])

            if abs(row_sum) == 2:
                score += -4 * (row_sum // 2 == state.player_to_move)
            if abs(col_sum) == 2:
                score += -4 * (col_sum // 2 == state.player_to_move)
            if abs(row_sum) == 1 and row_zero == 3: 
                score += 2 * (row_sum // 1 == state.player_to_move)
            if abs(col_sum) == 1 and col_zero == 3:
                score += 2 * (col_sum // 1 == state.player_to_move)

        diag_sum1 = np.sum(np.diag(state.blocks[x]))
        diag1_zero = np.count_nonzero(np.diag(state.blocks[x]))
        diag_sum2 = np.sum(np.diag(np.fliplr(state.blocks[x])))
        diag2_zero = np.count_nonzero(np.diag(np.fliplr(state.blocks[x])))

        if abs(diag_sum1) == 2:
            score += -4 * (diag_sum1 // 2 == state.player_to_move)
        if abs(diag_sum1) == 1 and diag1_zero == 3:
            score += 2 * (diag_sum1 // 1 == state.player_to_move)
        if abs(diag_sum2) == 2:
            score += -4 * (diag_sum2 // 2 == state.player_to_move)
        if abs(diag_sum2) == 1 and diag2_zero == 3:
            score += 2 * (diag_sum2 // 1 == state.player_to_move)

    for j in [0, 3, 6]:
        row_sum = np.sum(state.global_cells[j:j+3])
        row_zero = np.count_nonzero(state.global_cells[j:j+3])
        if abs(row_sum) == 2:
            score += -2 * (row_sum // 2 == state.player_to_move)
        if abs(row_sum) == 1 and row_zero == 3:
            score += 2 * (row_sum // 1 == state.player_to_move)

        col_sum = np.sum(state.global_cells[j::3])
        col_zero = np.count_nonzero(state.global_cells[j::3])
        if abs(col_sum) == 2:
            score += -2 * (col_sum // 2 == state.player_to_move)
        if abs(col_sum) == 1 and col_zero == 3:
            score += 2 * (col_sum // 1 == state.player_to_move)

    diag_sum1 = np.sum(np.diag(state.global_cells.reshape(3, 3)))
    diag1_zero = np.count_nonzero(np.diag(state.global_cells.reshape(3, 3)))
    diag_sum2 = np.sum(np.diag(np.fliplr(state.global_cells.reshape(3, 3))))
    diag2_zero = np.count_nonzero(np.diag(np.fliplr(state.global_cells.reshape(3, 3))))

    if abs(diag_sum1) == 2:
        score += -4 * (diag_sum1 // 2 == state.player_to_move)
    if abs(diag_sum2) == 2:
        score += -4 * (diag_sum2 // 2 == state.player_to_move)
    if abs(diag_sum1) == 1 and diag1_zero == 3:
        score += 2 * (diag_sum1 // 1 == state.player_to_move)
    if abs(diag_sum2) == 1 and diag2_zero == 3:
        score += 2 * (diag_sum2 // 1 == state.player_to_move)

    return score

        

def select_move(cur_state, remain_time):
    legal_moves = cur_state.get_valid_moves
    best_move = None
    best_moves = []  # List to store moves with the best value
    best_value = -math.inf
    alpha = -math.inf
    beta = math.inf

    '''if cur_state.previous_move is None:
        #return np.random.choice(legal_moves)
        return UltimateTTT_Move(4, 1, 1, cur_state.player_to_move)'''

    for move in legal_moves:
        next_state = State_2(cur_state)
        next_state.free_move = cur_state.free_move
        next_state.act_move(move)
        
        value = minimax_pruning(next_state, depth=2, alpha=alpha, beta=beta, maximizing_player=False)

        if value > best_value:
            best_value = value
            best_moves = [move]  # Start a new list for the best move
        elif value == best_value:
            best_moves.append(move)  # Add the move to the list

        alpha = max(alpha, best_value)
    if best_moves == []: return None
    # Choose a random move from the list of best moves
    return random.choice(best_moves)