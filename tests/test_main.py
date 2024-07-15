

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from mcts import TicTacToe


# Zobrist related tests for tictactoe

def test_zobrist0():
    """ Test that the zobrist array is always the same regardless of user 
    provided seed input and each hash is always 64 characters long. """

    env0 = TicTacToe()
    arr0 = env0.zobrist_array

    env1 = TicTacToe(seed=1234)
    arr1 = env1.zobrist_array

    env2 = TicTacToe(seed=7293)
    arr2 = env2.zobrist_array

    for hash0, hash1, hash2 in zip(arr0, arr1, arr2):
        assert len(hash0) == 64
        assert hash0 == hash1 and hash1 == hash2
    