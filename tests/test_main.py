

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

def test_zobrist2():
    """ Test that the order in which actions are played does not change
    the resulting zobrist hash of the game state. The current state of
    the game should be the only thing that the hash depends on. """

    env0 = TicTacToe()
    env1 = TicTacToe()

    actions_list0 = [
        [0, 1, 2],
        [4, 0, 1, 2, 7],
        [8, 0, 7, 1, 6, 2, 5, 3, 4]  # includes moves after termination
    ]
    actions_list1 = [
        [2, 1, 0],
        [7, 2, 4, 0, 1],
        [7, 0, 4, 3, 8, 1, 6, 2, 5] # includes moves after termination
    ]

    for actions0, actions1 in zip(actions_list0, actions_list1):
        env0.reset()
        env1.reset()
        for action in actions0:
            hash0 = env0.step(action)[-1]["hash"]
        for action in actions1:
            hash1 = env1.step(action)[-1]["hash"]
        assert hash0 == hash1

# Reset method related tests for tictactoe

def test_reset0():
    """ Test that the reset method in tictactoe returns state, env_info.
    With state being an all zero ndarray of length 10 and dtype np.int32.
    env_info having keys 'hash', 'win', 'legal_actions' with correct
    initial values. """

    env = TicTacToe()

    # Perform the tests five times in case of stochasticity in env
    for _ in range(5):
        state, env_info = env.reset()

        assert state.dtype == np.float32
        assert state.size == 10
        assert np.all(state[:9] == 0.0) and state[9] == 1.0

        assert env_info["hash"] == "0" * 64
        assert env_info["win"] == 0
        assert np.all(
            env_info["legal_actions"] == np.arange(9, dtype=np.int32))

# Step method related tests for tictactoe

def test_step0():
    """ Test certain games play out as expected. """

    env = TicTacToe()

    actions_list = [
        [0, 1],
        [2, 5, 3, 4, 1],
        [8, 7, 6, 0, 4],
        [4, 0, 6, 2, 1, 7, 5, 3, 8],
        [0, 2, 1, 4, 6, 8, 7, 5, 3],
        [1, 4, 0, 2, 6, 3, 5, 8, 7],
        [5, 4, 2, 8, 0, 7, 3, 1],
        [0, 1, 4, 8, 6, 3, 2],
        [0, 3, 1, 7, 2],
    ]

    terminated_list = [0, 0, 0, 1, 1, 1, 1, 1, 1]
    win_list = [0, 0, 0, 0, 0, 0, 1, 1, 1]

    for actions, true_terminated, true_win in zip(
        actions_list, terminated_list, win_list):

        env.reset()
        terminated = 0
        win = 0
        for action in actions:
            _, _, terminated, _, env_info = env.step(action)
            win = env_info["win"]

        assert terminated == true_terminated and win == true_win

