import numpy as np
import copy
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd

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

def minimax(node, depth, maximizing_player, eval_fn, player):
    if depth == 0 or node.state.get_winner() != 0 or not list(node.state.get_legal_moves()):
        piece = node.state.pieces[player]
        return eval_fn(node, piece)

    if maximizing_player:
        value = -np.inf
        for child in node.get_children():
            value = max(value, minimax(child, depth-1, False, eval_fn, player))
    else:
        value = np.inf
        for child in node.get_children():
            value = min(value, minimax(child, depth-1, True, eval_fn, player))
    
    return value

def minimax_with_pruning(node, depth, maximizing_player, alpha, beta, eval_fn, player):
    if depth == 0 or node.state.get_winner() != 0 or not list(node.state.get_legal_moves()):
        piece = node.state.pieces[player]
        return eval_fn(node, piece)

    if maximizing_player:
        value = -np.inf
        for child in node.get_children():
            value = max(value, minimax_with_pruning(child, depth-1, False, alpha, beta, eval_fn, player))
            alpha = max(alpha, value)
            if beta <= alpha:
                break
    else:
        value = np.inf
        for child in node.get_children():
            value = min(value, minimax_with_pruning(child, depth-1, True, alpha, beta, eval_fn, player))
            beta = min(beta, value)
            if beta <= alpha:
                break
    
    return value

class MonteCarloTreeSearchNode():
    def __init__(self, state, player, parent=None, parent_action=None, num_sims=100):
        self.state = state
        self.player = player
        self.parent = parent
        self.parent_action = parent_action
        self.children = []
        self._number_of_visits = 0
        self._results = defaultdict(int)
        self._results[1] = 0
        self._results[-1] = 0
        self._untried_actions = None
        self._untried_actions = self.untried_actions()
        self.num_sims = num_sims
        return
    
    def untried_actions(self):
        self._untried_actions = list(self.state.get_legal_moves())
        return self._untried_actions

    def q(self):
        wins = self._results[1]
        loses = self._results[-1]
        return wins - loses
    
    def n(self):
        return self._number_of_visits
    
    def expand(self):
        action = self._untried_actions.pop()
        board_copy = copy.deepcopy(self.state)
        board_copy.place(action)
        child_node = MonteCarloTreeSearchNode(
            board_copy, self.player, parent=self, parent_action=action
        )

        self.children.append(child_node)
        return child_node
    
    def is_terminal_node(self):
        return self.state.get_winner() != 0 or len(list(self.state.get_legal_moves())) == 0
    
    def rollout(self):
        current_rollout_state = copy.deepcopy(self.state)

        while current_rollout_state.get_winner() == 0 and len(list(current_rollout_state.get_legal_moves())) > 0:

            possible_moves = list(current_rollout_state.get_legal_moves())

            action = self.rollout_policy(possible_moves)
            current_rollout_state.place(action)
            
        winner = current_rollout_state.get_winner()
        if winner == self.player:
            return 1
        elif winner == -self.player:
            return -1
        else:
            return 0 
    
    def backpropagate(self, result):
        self._number_of_visits += 1.
        self._results[result] += 1.
        if self.parent:
            self.parent.backpropagate(result)
    
    def is_fully_expanded(self):
        return len(self._untried_actions) == 0

    def best_child(self, c_param=0.1):
        choices_weights = [(c.q() / c.n()) + c_param * np.sqrt((2 * np.log(self.n()) / c.n())) for c in self.children]
        if len(choices_weights) == 0:
            print(self.is_terminal_node())
        return self.children[np.argmax(choices_weights)]
    
    def rollout_policy(self, possible_moves):
        return list(possible_moves)[np.random.randint(len(possible_moves))]

    def _tree_policy(self):
        current_node = self
        while not current_node.is_terminal_node():

            if not current_node.is_fully_expanded():
                return current_node.expand()
            else:
                current_node = current_node.best_child()
        return current_node
    
    def best_action(self):
        simulation_no = self.num_sims

        for i in range(simulation_no):

            v = self._tree_policy()
            reward = v.rollout()
            v.backpropagate(reward)

        return self.best_child(c_param=0.1)
    
def plot_agent_results(results, agent1, agent2, title, x, xlabel='', ylabel='', xticks=None):
    sim_values = [list(x.values()) for x in results]
    draws = [x[0]/sum(x) for x in sim_values] # get percentage
    a1_winrate = [x[1]/sum(x) for x in sim_values]
    a2_winrate = [x[2]/sum(x) for x in sim_values]
    
    res_df = pd.DataFrame({'draws':draws, f'{agent1.name} agent wins':a1_winrate, f'{agent2.name} agent wins':a2_winrate})
    res_df.index = x
    
    res_df.plot(kind='bar', stacked=False, width=0.5, title=title, xlabel=xlabel, ylabel=ylabel)
    