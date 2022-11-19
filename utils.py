import numpy as np
import copy

class Node:
    def __init__(self, state, parent, move):
        self.state = state
        self.parent = parent
        self.move = move
        
        self.who_placed = (self.state.current_player + 1)%2
        self.children = []
        
    def get_children(self):
        children = []
        for move in self.state.get_legal_moves():
            board_copy = copy.deepcopy(self.state)
            board_copy.place(move)
            children.append(
                Node(
                    state=board_copy,
                    parent=self,
                    move=move,
                )
            )
        self.children = children
        return children

def minimax(node, depth, maximizing_player, eval_fn):
    if depth == 0 or node.state.get_winner() != 0 or not list(node.state.get_legal_moves()):
        return eval_fn(node)
    if maximizing_player:
        value = -np.inf
        for child in node.get_children():
            value = max(value, minimax(child, depth-1, False, eval_fn))

    else:
        value = np.inf
        for child in node.get_children():
            value = min(value, minimax(child, depth-1, True, eval_fn))

    return value

