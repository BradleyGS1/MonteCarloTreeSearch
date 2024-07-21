

import numpy as np
from pprint import pprint
from mcts import TicTacToe, Othello, MCTS


def main():

    def env_fn():
        return TicTacToe()

    mcts = MCTS()
    mcts.fit(env_fn, n_iters=10000, eval_every=1000, eval_iters=1000)
    hashes = [
        '0000000000000000000000000000000000000000000000000000000000000000',
        '0001010001000010010110111110000110101111100000110010001011001011',
        '0100011001000101101001110011000010100111111011100011110110110001',
        '0000110111000011111100000010001010110101101101001001010100010111'
    ]

    for hash in hashes:
        pprint(mcts.tree[hash])

    mcts.explore_factor = 0.0
    print(f"Best first action {mcts.expansion('0'*64, None)}")

    def env_fn():
        return Othello()

    mcts = MCTS()
    mcts.fit(env_fn, n_iters=10000, eval_every=2000, eval_iters=500)

    pprint(mcts.tree[mcts.root_hash])

    mcts.explore_factor = 0.0
    print(f"Best first action {mcts.expansion(mcts.root_hash, None)}")


if __name__ == "__main__":
    main()
