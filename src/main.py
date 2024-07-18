

from pprint import pprint
from mcts import TicTacToe, MCTS


def main():
    def env_fn():
        return TicTacToe()

    mcts = MCTS()

    mcts.fit(env_fn, n_iters=2000)
    hashes = [
        '0000000000000000000000000000000000000000000000000000000000000000',
        '0001010001000010010110111110000110101111100000110010001011001011',
        '0100011001000101101001110011000010100111111011100011110110110001',
        '0000110111000011111100000010001010110101101101001001010100010111'
    ]

    for hash in hashes:
        pprint(mcts.tree[hash])

    print(f"Best first action {mcts.expansion('0'*64, None)}")

if __name__ == "__main__":
    main()
