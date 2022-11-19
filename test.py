import numpy as np
from agent import MiniMaxAgent, RandomAgent
from tictactoe import Board2D, Board3D, TicTacToe
from eval import random_eval

agent1 = MiniMaxAgent(random_eval, depth=2)
agent2 = MiniMaxAgent(random_eval, depth=2)

board = Board2D(board_size=3)
t = TicTacToe(board, agent1, agent2)

agent1 = RandomAgent()
agent2 = RandomAgent()

board = Board2D(board_size=3)

t = TicTacToe(board, agent1, agent2)

res = {
    'X': 0,
    'draw': 0,
    'O': 0
}

for _ in range(10000):
    outcome = t.sim_game(display=False)
    if outcome == -1:
        res['X'] += 1
    elif outcome == 1:
        res['O'] += 1
    else:
        res['draw'] += 1

print(res)