

import numpy as np


class TicTacToe:
    def __init__(self, seed: int = 0):
        # Set the fixed np.random seed for initialising the zobrist array
        np.random.seed(7284)
        self._init_zobrist()

        # Set the np.random seed for generating random actions
        np.random.seed(seed)

    # Initialise the zobrist hash element array
    def _init_zobrist(self) -> None:
        self.zobrist_array = []

        # There are 2*9+1=19 zobrist hashes (length 64) we need to encode
        # the full game state 2*9 for each position and player in that pos,
        # +1 for the player turn
        for _ in range(19):
            zobrist = "".join(np.random.choice(["0", "1"], size=64))
            self.zobrist_array.append(zobrist)

    # Perform an xor on two hashes and return result
    def _hash_xor(self, hash0: str, hash1: str) -> str:
        new_hash = []
        for char0, char1 in zip(hash0, hash1):
            new_hash.append("0" if char0 == char1 else "1")
        return "".join(char for char in new_hash)

    # Update the game state hash based on the action performed
    def _update_hash(self, action: int) -> None:
        is_player_two = (self.player_turn == -1)
        zobrist_idx = is_player_two * 9 + action
        zobrist = self.zobrist_array[zobrist_idx]
        # Update hash with new game state contribution
        self.hash = self._hash_xor(self.hash, zobrist)
        # Update hash with current turn contribution
        self.hash = self._hash_xor(self.hash, self.zobrist_array[18])

    # Return the legal actions
    def legal_actions(self) -> set[int]:
        return self.legal_moves

    # Reset the env
    def reset(self) -> tuple[np.ndarray[np.float32], dict[str,]]:
        """ Reset the environment ready for a new game.
        ## Returns:
        - initial state : np.nddary[np.float32]. First 9 values
        are 1.0 for player one, -1.0 for player two, 0.0 for
        available tiles. The final value is the player turn.
        - env info : dict[str, Any]. Mapping with keys such as
        'hash', 'win'. """

        # 1: player one goes first, -1: player two goes second
        self.player_turn = 1
        # All positions in the board can be played on init
        self.legal_moves = set(range(9))

        # Initialise the game state numpy array
        # the first 9 elements are the state of each position
        # the last element is the state of the current turn
        self.state = np.zeros(shape=10, dtype=np.float32)
        self.state[-1] = 1.0

        self.hash = "0" * 64

        # Create the env info dict
        env_info = dict()
        env_info["legal_actions"] = self.legal_actions()
        env_info["hash"] = self.hash
        env_info["winner"] = 0

        return self.state.copy(), env_info

    # Perform a win condition check
    def _win_check(self):
        state_matrix = np.reshape(self.state[:9], newshape=(3, 3))
        win_sum = 3 * self.player_turn
        if np.any(np.sum(state_matrix, axis=0) == win_sum):
            return True
        if np.any(np.sum(state_matrix, axis=1) == win_sum):
            return True
        if sum(state_matrix[i, i] for i in range(3)) == win_sum:
            return True
        if sum(state_matrix[i, 2-i] for i in range(3)) == win_sum:
            return True
        return False

    # Perform a single env step
    def step(self, action: int) -> tuple[
        np.ndarray[np.float32],
        np.ndarray[np.float32],
        bool,
        bool,
        dict[str,]
    ]:
        """ Performs a single env step. Follows the gymnasium
        convention for environments.
        ## Inputs:
        - action : int. action is equal to row * 3 + col to
        place the current players tile.
        ## Returns:
        - initial state : np.nddary[np.float32]. First 9 values
        are 1.0 for player one, -1.0 for player two, 0.0 for
        unowned tiles. The final value is the player turn.
        - reward vec : np.ndarray[np.float32]. First value is
        reward for player one. Second value is for player two.
        - terminated : bool. True if the game has ended.
        env.reset() should be called before another game is
        started.
        - truncated : bool. Unused (takes the same value as
        terminated).
        - env info : dict[str, Any]. Mapping with keys such as
        'hash', 'win'. """

        rewards = np.zeros(shape=2, dtype=np.float32)
        terminated = False
        env_info = dict()
        env_info["legal_actions"] = self.legal_actions()
        env_info["hash"] = self.hash
        env_info["winner"] = 0

        if action not in self.legal_moves:
            print("Illegal move made.")
            return (
                self.state.copy(),
                rewards,
                terminated,
                terminated,
                env_info
            )

        self.state[action] = self.player_turn
        self.state[-1] = self.player_turn
        self.legal_moves.remove(action)
        self._update_hash(action)
        env_info["legal_actions"] = self.legal_actions()
        env_info["hash"] = self.hash

        if self._win_check():
            player_idx = 0 if self.player_turn == 1 else 1
            rewards[player_idx] = 1.0
            rewards[1-player_idx] = -1.0
            terminated = True
            env_info["winner"] = player_idx + 1

        elif len(self.legal_moves) == 0:
            terminated = True

        self.player_turn *= -1

        return self.state.copy(), rewards, terminated, terminated, env_info
