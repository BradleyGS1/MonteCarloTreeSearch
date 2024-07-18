

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import numpy as np

from mcts import TicTacToe, MCTS


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
        assert env_info["legal_actions"] == set(range(9))

# Step method related tests for tictactoe

def test_step0():
    """ Test certain games play out as expected. """

    env = TicTacToe()

    actions_list = [
        [0, 1],
        [2, 5, 3, 4, 1],
        [8, 7, 6, 0, 4],
        [4, 0, 6, 2, 1, 7, 5, 3, 8],
        [0, 2, 1, 4, 6, 8, 7, 3, 5],
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

# Expansion method tests for MCTS

def test_expansion0():
    """ Test the return values are correct. Not
    testing selection method yet. """

    mcts = MCTS()
    env = TicTacToe()

    env_info = env.reset()[-1]
    legal_actions = env_info["legal_actions"]
    hash = env_info["hash"]

    action = mcts.expansion(hash, legal_actions)
    assert action == None

    mcts.cleanup()

    env_info = env.reset()[-1]
    legal_actions = env_info["legal_actions"]
    hash = env_info["hash"]

    action = mcts.expansion(hash, legal_actions)
    assert action != None

    env_info = env.step(0)[-1]
    legal_actions = env_info["legal_actions"]
    hash = env_info["hash"]

    action = mcts.expansion(hash, legal_actions)
    assert action == None

def test_expansion1():
    """ Test the tree information while performing
    expansion steps is correct. """

    mcts = MCTS()
    env = TicTacToe()

    assert len(mcts.tree) == 0

    hashes = []
    env_info = env.reset()[-1]
    legal_actions = env_info["legal_actions"]
    hash = env_info["hash"]
    hashes.append(hash)

    action = mcts.expansion(hash, legal_actions)
    assert mcts.visited == hashes
    assert len(mcts.tree) == 1
    assert len(mcts.tree[hash]["untried_actions"]) == 9
    assert mcts.tree[hash]["parents"] == set()
    assert mcts.tree[hash]["children"] == {
        a: [] for a in legal_actions
    }

    mcts.cleanup()

    hashes = []
    env_info = env.reset()[-1]
    legal_actions = env_info["legal_actions"]
    hash = env_info["hash"]
    hashes.append(hash)

    action = mcts.expansion(hash, legal_actions)
    assert mcts.visited == hashes
    assert len(mcts.tree) == 1
    assert len(mcts.tree[hash]["untried_actions"]) == 9
    assert mcts.tree[hash]["parents"] == set()
    assert mcts.tree[hash]["children"] == {
        a: [] for a in legal_actions
    }

    parent_hash = hash
    env_info = env.step(action)[-1]
    legal_actions = env_info["legal_actions"]
    hash = env_info["hash"]
    hashes.append(hash)

    prev_action = action
    action = mcts.expansion(hash, legal_actions)
    assert mcts.visited == hashes
    assert len(mcts.tree) == 2
    assert len(mcts.tree[hash]["untried_actions"]) == 8
    assert mcts.tree[hash]["parents"] == set([parent_hash])
    assert mcts.tree[hash]["children"] == {
        a: [] for a in legal_actions
    }

    assert len(mcts.tree[parent_hash]["untried_actions"]) == 8
    assert mcts.tree[parent_hash]["parents"] == set()
    parents_children_dict = {
        a: [] for a in range(9)
    }
    parents_children_dict[prev_action].append(hash)
    assert mcts.tree[parent_hash]["children"] == parents_children_dict

# Selection method tests for mcts

def test_selection0():
    """ Test the first time selection is required. """

    mcts = MCTS()
    env = TicTacToe()

    # It takes 9 full mcts loops before the initial state
    # has run out of untried actions

    for _ in range(9):
        env_info = env.reset()[-1]
        hash = env_info["hash"]
        legal_actions = env_info["legal_actions"]
        action = mcts.expansion(hash, legal_actions)

        player = 1
        while action is not None:
            player = 1 - player
            env_info = env.step(action)[-1]
            hash = env_info["hash"]
            legal_actions = env_info["legal_actions"]
            action = mcts.expansion(hash, legal_actions)

        terminated = False
        while not terminated:
            action = np.random.choice(list(legal_actions))
            player = 1 - player
            _, _, terminated, _, env_info = env.step(action)
            legal_actions = env_info["legal_actions"]
            win = env_info["win"]

        mcts.backprop(player, win)
        mcts.cleanup()

    env_info = env.reset()[-1]
    hash = env_info["hash"]
    init_node = mcts.tree[hash]
    print(init_node)
    
    for action, children in init_node["children"].items():
        print(action)
        for child in children:
            print(mcts.tree[child])
    print(mcts.expansion(hash, None))
