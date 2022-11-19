import numpy as np

def random_eval(board):
    return np.random.normal()

def are_neighbors(loc1, loc2):
    dim = len(loc1)
    diff = np.asarray(loc1) - np.asarray(loc2)
    return np.square(diff).sum() <= np.ones(shape=(dim,)).sum()

def neighbors_eval(node):
    board = node.state
    score = 0
    player = board.pieces[(board.current_player + 1)%2]
    other_player = -player
    
    if board.get_winner() == player:
        return np.inf
    elif board.get_winner() == other_player:
        return -np.inf
    
    my_locs = list(zip(*np.where(board.board == player)))
    
    for loc in my_locs:
        for loc2 in my_locs:
            if are_neighbors(loc, loc2):
                score+=1
                
    return score

def dist(loc1, loc2):
    return np.square(loc1 - loc2).sum()

def dist_from_center(loc, board):
    center = np.asarray([board.shape[0]//2, board.shape[1]//2])
    return dist(np.asarray(loc), center)

def my_eval_function(node):
    board = node.state
    score = 0
    player = board.pieces[(board.current_player + 1)%2] # this *backtracks* to the previous board player 
    # i.e. gets the player we're evaluating the perspective of
    my_pieces = zip(*np.where(board.board == player))
    for loc in my_pieces:
        score -= dist_from_center(loc, board.board)
    return score + np.random.normal()