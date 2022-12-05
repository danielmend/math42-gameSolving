import numpy as np
import copy
from utils import Node, minimax, minimax_with_pruning
import random 
from collections import defaultdict
from utils import MonteCarloTreeSearchNode

class Agent:
    def __init__(self, name):
        pass
    
    def evaluate(self, board):
        '''
        board -> best move
        '''
        pass

class MiniMaxEvaluator:
    def __init__(self, eval_fn, depth, using_ab_pruning = False):
        self.eval_fn = eval_fn
        self.depth = depth
        self.using_ab_pruning = using_ab_pruning
        
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
            player = (child_node.state.current_player+1)%2
            
            if self.using_ab_pruning:
                value = minimax_with_pruning(child_node, self.depth, False, -np.inf, np.inf, self.eval_fn, player) 
            else:
                value = minimax(child_node, self.depth, False, self.eval_fn, player)
            
            
            if value > best_eval:
                best_move = move
                best_eval = value
           
        return best_move

class MiniMaxAgent(Agent):
    def __init__(self, eval_fn, depth, name='MiniMaxAgent', using_ab_pruning = False):
        super().__init__(name)
        self.name = name
        self.eval_fn = eval_fn
        self.evaluator = MiniMaxEvaluator(eval_fn, depth, using_ab_pruning = using_ab_pruning)
        
    def evaluate(self, board):
        '''
        board -> best move
        '''
        node = Node(board, None, None)
        return self.evaluator.evaluate(node)

class RandomAgent(Agent):
    def __init__(self, name='RandomAgent'):
        super().__init__(name)
        self.name = name
    
    def evaluate(self, board):
        moves = list(board.get_legal_moves())
        return random.choice(moves)
    
class MonteCarloAgent(Agent):
    def __init__(self, name='MctsAgent', num_sims=100):
        super().__init__(name)
        self.name = name
        self.num_sims = num_sims
    
    def evaluate(self, board):
        player = board.current_player
        return MonteCarloTreeSearchNode(board, player, num_sims=self.num_sims).best_action().parent_action