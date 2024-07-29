

import numpy as np


class Othello:
    def __init__(self, render: int = 0, verbose: int = 0, seed: int = 0):
        # Set the fixed np.random seed for initialising the zobrist array
        np.random.seed(3429)
        self._init_zobrist()

        self.action_space = 64

        self.render = render
        self.verbose = verbose

        # Set the np.random seed for generating random actions
        np.random.seed(seed)

    # Initialise the zobrist hash element array
    def _init_zobrist(self) -> None:
        self.zobrist_array = []

        # There are 2*64+1=129 zobrist hashes (length 64) we need to encode
        # the full game state 2*64 for each position and player in that pos,
        # +1 for the player turn
        for _ in range(129):
            zobrist = "".join(np.random.choice(["0", "1"], size=64))
            self.zobrist_array.append(zobrist)

    # Perform an xor on two hashes and return result
    def _hash_xor(self, hash0: str, hash1: str) -> str:
        new_hash = []
        for char0, char1 in zip(hash0, hash1):
            new_hash.append("0" if char0 == char1 else "1")
        return "".join(char for char in new_hash)

    # Initialise the zobrist hash
    def _init_hash(self) -> None:
        self.hash = "0" * 64
        zobrist_ids = [28, 35, 91, 100]
        for zobrist_id in zobrist_ids:
            zobrist = self.zobrist_array[zobrist_id]
            self.hash = self._hash_xor(self.hash, zobrist)

    # Update the legal actions array
    def _update_legal_moves(self):
        self.legal_moves = set()
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        player = self.player_turn
        opponent = self.player_turn * -1

        def is_on_board(x, y):
            return 0 <= x < 8 and 0 <= y < 8

        for x_init, y_init in self.check_tiles:
            for dx, dy in directions:
                x = x_init + dx
                y = y_init + dy
                found_opponent = False
                while (
                    is_on_board(x, y)
                    and
                    self.state[x, y] == opponent
                ):
                    x += dx
                    y += dy
                    found_opponent = True

                if (
                    found_opponent
                    and
                    is_on_board(x, y)
                    and
                    self.state[x, y] == player
                ):
                    flat_action_idx = x_init * 8 + y_init
                    self.legal_moves.add(flat_action_idx)
                    break

    # Find all grid coords of tiles to flip if a tile is placed
    # based on the value of action. Assuming the action is legal
    def _get_flips(self, action: int) -> list[tuple[int, int]]:
        flips = set()
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        player = self.player_turn
        opponent = self.player_turn * -1

        x_init = action // 8
        y_init = action % 8

        def is_on_board(x, y):
            return 0 <= x < 8 and 0 <= y < 8

        for dx, dy in directions:
            x = x_init + dx
            y = y_init + dy
            possible_flips = []
            while (
                is_on_board(x, y)
                and
                self.state[x, y] == opponent
            ):
                if (x, y) not in flips:
                    possible_flips.append((x, y))

                x += dx
                y += dy

            if (
                len(possible_flips) > 0
                and
                is_on_board(x, y)
                and
                self.state[x, y] == player
            ):
                for tile in possible_flips:
                    flips.add(tile)
        return flips

    # Update the set of tiles that should be checked
    # when finding the set of legal actions
    def _update_check_tiles(self, new_tile: tuple[int, int]) -> None:
        def is_on_board(x, y):
            return 0 <= x < 8 and 0 <= y < 8
        
        for i in range(-1, 2):
            for j in range(-1, 2):
                x = new_tile[0] + i
                y = new_tile[1] + j
                if (i != 0 or j != 0) and is_on_board(x, y) and self.state[x, y] == 0.0:
                    self.check_tiles.add((x, y))

        self.check_tiles.remove(new_tile)

    # Display the game board
    def display(self) -> None:
        player_one_count = int(np.sum(self.state == +1.0))
        player_two_count = int(np.sum(self.state == -1.0))
        board_str = f"X: {player_one_count}\tO: {player_two_count} \n"

        board_str += "  A B C D E F G H \n"
        for i in range(8):
            board_str += f"{i+1} "
            for j in range(8):
                char = '.'
                if (i, j) in self.check_tiles:
                    char = '#'
                if self.state[i, j] == 1.0:
                    char = 'X'
                elif self.state[i, j] == -1.0:
                    char = 'O'
                board_str += f"{char} "
            board_str += "\n"
        board_str += "\n"
        print(board_str)

    # Return the legal actions
    def legal_actions(self) -> set[int]:
        return self.legal_moves.copy()

    # Get the vectorised state
    def vec_state(self) -> np.ndarray[np.float32]:
        state = np.concatenate([
            self.state.flatten(),
            np.asarray([self.player_turn], dtype=np.float32)
        ])
        return state

    def reset(self) -> tuple[np.ndarray[np.float32], dict[str,]]:
        """ Reset the environment ready for a new game.
        ## Returns:
        - initial state : np.nddary[np.float32]. First 64 values
        are 1.0 for player one, -1.0 for player two, 0.0 for
        available tiles. The final value is the player turn.
        - env info : dict[str, Any]. Mapping with keys such as
        'hash', 'winner'. """

        # 1 is player one who goes first, -1 is player two who goes second
        self.player_turn = 1
        self.skip_count = 0

        # Initialise the game state numpy array
        # 0.0 is an unused tile, 1.0 is a tile belonging to player one
        # and -1.0 is a tile of player two
        self.state = np.zeros(shape=(8, 8), dtype=np.float32)
        self.state[3, 3] = -1.0
        self.state[4, 4] = -1.0
        self.state[3, 4] = 1.0
        self.state[4, 3] = 1.0

        self.check_tiles = set()
        for i in range(2, 6):
            for j in range(2, 6):
                tile = (i, j)
                if tile not in [(3, 3), (4, 4), (3, 4), (4, 3)]:
                    self.check_tiles.add(tile)

        self._init_hash()
        self._update_legal_moves()

        # Create the env info dict
        self.env_info = dict()
        self.env_info["legal_actions"] = self.legal_actions()
        self.env_info["hash"] = self.hash
        self.env_info["winner"] = 0

        if self.render:
            self.display()

        return self.vec_state(), self.env_info

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
        - action : int. action is equal to row * 64 + col to
        place the current players tile.
        ## Returns:
        - initial state : np.nddary[np.float32]. First 64 values
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
        self.env_info = dict()
        self.env_info["legal_actions"] = self.legal_actions()
        self.env_info["hash"] = self.hash
        self.env_info["winner"] = 0

        if len(self.legal_moves) == 0 and action == 0:
            if self.verbose:
                print("Skipping turn.")
            self.skip_count += 1

            if self.skip_count == 2:
                score = np.sum(self.state)
                if score != 0:
                    win_idx = score < 0
                    rewards[win_idx] = 1.0
                    rewards[1-win_idx] = -1.0
                    self.env_info["winner"] = win_idx + 1
                terminated = True

            self.player_turn *= -1
            self._update_legal_moves()
            zobrist = self.zobrist_array[128]
            self.hash = self._hash_xor(self.hash, zobrist)

            self.env_info["legal_actions"] = self.legal_actions()
            self.env_info["hash"] = self.hash

            return (
                self.vec_state(),
                rewards,
                terminated,
                terminated,
                self.env_info
            )

        self.skip_count = 0

        if action not in self.legal_moves:
            print("Illegal move made.")
            print(self.legal_moves, action)
            self.display()
            return (
                self.vec_state(),
                rewards,
                terminated,
                terminated,
                self.env_info
            )

        row = action // 8
        col = action % 8
        self.state[row, col] = self.player_turn
        self._update_check_tiles((row, col))

        zobrist_id = 64 * (self.player_turn == -1) + action
        zobrist = self.zobrist_array[zobrist_id]
        self.hash = self._hash_xor(self.hash, zobrist)

        zobrist = self.zobrist_array[128]
        self.hash = self._hash_xor(self.hash, zobrist)

        for i, j in self._get_flips(action):
            self.state[i, j] = self.player_turn
            flat_idx = i * 8 + j

            zobrist = self.zobrist_array[flat_idx]
            self.hash = self._hash_xor(self.hash, zobrist)

            zobrist = self.zobrist_array[64 + flat_idx]
            self.hash = self._hash_xor(self.hash, zobrist)

        self.player_turn *= -1
        self._update_legal_moves()

        self.env_info["legal_actions"] = self.legal_actions()
        self.env_info["hash"] = self.hash

        if self.render:
            self.display()

        return self.vec_state(), rewards, terminated, terminated, self.env_info
