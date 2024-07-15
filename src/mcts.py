

import numpy as np


class TicTacToe:
    def __init__(self, seed: int = 0):
        # 1: player one goes first, -1: player two goes second
        self.player_turn: int = 1
        # All positions in the board can be played on init
        self.legal_actions: set[int] = set(i for i in range(9))

        # Set the fixed np.random seed for initialising the zobrist array
        np.random.seed(7284)
        self.hash = "0" * 64
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

    # Perform an xor on two hashes in place
    def _hash_xor(self, hash0: str, hash1: str) -> None:
        for i, chars in enumerate(zip(hash0, hash1)):
            char0, char1 = chars
            hash0[i] = ("0" if char0 == char1 else "1")

    # Update the game state hash based on the action performed
    def _update_hash(self, action: int) -> None:
        is_player_two = (self.player_turn == -1)
        zobrist_idx = is_player_two * 9 + action
        zobrist = self.zobrist_array[zobrist_idx]

        self._hash_xor(self.hash, zobrist)
        self._hash_xor(self.hash, self.zobrist_array[18])
