

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
    def legal_actions(self) -> np.ndarray[np.int32]:
        return np.asarray(list(self.legal_moves), dtype=np.int32)

    # Reset the env
    def reset(self) -> tuple[np.ndarray[np.float32], dict[str,]]:
        """ Resets the game state.\nReturns: \n
        - initial state of shape=(10), dtype=np.float32.
        - env info mapping with keys such as 'hash', 'win'. """

        # 1: player one goes first, -1: player two goes second
        self.player_turn = 1
        # All positions in the board can be played on init
        self.legal_moves = set(i for i in range(9))

        # Initialise the game state numpy array
        # the first 9 elements are the state of each position
        # the last element is the state of the current turn
        self.state = np.zeros(shape=10, dtype=np.float32)
        self.state[-1] = 1.0

        self.hash = "0" * 64

        # Initialise the env info dict
        env_info = dict()
        env_info["legal_actions"] = self.legal_actions()
        env_info["hash"] = self.hash
        env_info["win"] = 0

    """
    # Perform a single env step
    def step(self, action: int) -> tuple[
        np.ndarray[np.float32], np.ndarray[np.float32], bool, bool, dict[str,]]:

        rewards = np.zeros(shape=2, dtype=np.float32)
        terminated = False
    """
