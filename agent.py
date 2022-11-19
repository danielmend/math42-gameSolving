import numpy as np
import copy
from utils import Node, minimax
import random 

class Agent:
    def __init__(self):
        pass
    
    def evaluate(self, board):
        '''
        board -> best move
        '''
        pass

class MiniMaxEvaluator:
    def __init__(self, eval_fn, depth):
        self.eval_fn = eval_fn
        self.depth = depth
        
    def evaluate(self, node):
        '''
        node -> best move
        '''
        best_move = None
        best_eval = -np.inf
        for move in node.state.get_legal_moves():
            board_copy = copy.deepcopy(node.state)
            board_copy.place(move)
            child_node = Node(
                state=board_copy,
                parent=self,
                move=move,
            )
            value = minimax(child_node, self.depth, False, self.eval_fn)
            
            if value > best_eval:
                best_move = move
                best_eval = value
                
        return best_move

class MiniMaxAgent(Agent):
    def __init__(self, eval_fn, depth):
        super().__init__()
        self.eval_fn = eval_fn
        self.evaluator = MiniMaxEvaluator(eval_fn, depth)
        
    def evaluate(self, board):
        '''
        board -> best move
        '''
        node = Node(board, None, None)
        return self.evaluator.evaluate(node)

class RandomAgent(Agent):
    def __init__(self):
        super().__init__()
    
    def evaluate(self, board):
        return random.choice(list(board.get_legal_moves()))