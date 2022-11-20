from tictactoe import TicTacToe

def simulate(board, agent1, agent2, num_sims, display=False):
    order1 = TicTacToe(board, agent1, agent2)
    order2 = TicTacToe(board, agent2, agent1)

    sim = {
        'draw': 0,
        agent1.name: 0,
        agent2.name: 0,
    }

    for _ in range(num_sims//2):
        res1 = order1.sim_game(display=display)
        res2 = order2.sim_game(display=display)
        
        sim[res1] = sim.get(res1, 0) + 1
        sim[res2] = sim.get(res2, 0) + 1

    return sim