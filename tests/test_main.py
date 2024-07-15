

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import numpy as np

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

def test_zobrist1():
    """ Test that the _hash_xor method in tictactoe works as expected."""

    env = TicTacToe()
    hashes0 = [
        "0" * 16,
        "01101010",
        "10010010"
    ]
    hashes1 = [
        "1" * 16,
        "10101100",
        "01010101"
    ]
    results = [
        "1" * 16,
        "11000110",
        "11000111"
    ]

    for hash0, hash1, result in zip(hashes0, hashes1, results):
        assert result == env._hash_xor(hash0, hash1)

# Reset method related tests for tictactoe

def test_reset0():
    """ Test that the reset method in tictactoe returns state, env_info.
    With state being an all zero ndarray of length 10 and dtype np.int32.
    env_info having keys 'hash', 'win', 'legal_actions' with correct
    initial values. """

    # Perform the tests five times in case of stochasticity in env
    for _ in range(5):
        env = TicTacToe()
        state, env_info = env.reset()

        assert state.dtype == np.float32
        assert state.size == 10
        assert np.all(state[:9] == 0.0) and state[9] == 1.0

        assert env_info["hash"] == "0" * 64
        assert env_info["win"] == 0
        assert np.all(env_info["legal_actions"] == np.arange(9, dtype=np.int32))
