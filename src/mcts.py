

import sys
from copy import deepcopy

import numpy as np
from tqdm import trange
from pprint import pprint


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


class Othello:
    def __init__(self, render: int = 0, verbose: int = 0, seed: int = 0):
        # Set the fixed np.random seed for initialising the zobrist array
        np.random.seed(3429)
        self._init_zobrist()

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

        for x_init in range(8):
            for y_init in range(8):
                if self.state[x_init, y_init] == 0.0:
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
    def _get_flip_list(self, action: int) -> list[tuple[int, int]]:
        flip_list = set()
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
                if (x, y) not in flip_list:
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
                for i, j in possible_flips:
                    flip_list.add((i, j))

        return list(flip_list)

    # Display the game board
    def display(self) -> None:
        board_str = "  A B C D E F G H \n"
        for i in range(8):
            board_str += f"{i+1} "
            for j in range(8):
                char = '.'
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

        self._init_hash()
        self._update_legal_moves()

        # Create the env info dict
        env_info = dict()
        env_info["legal_actions"] = self.legal_actions()
        env_info["hash"] = self.hash
        env_info["winner"] = 0

        if self.render:
            self.display()

        return self.vec_state(), env_info

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
        env_info = dict()
        env_info["legal_actions"] = self.legal_actions()
        env_info["hash"] = self.hash
        env_info["winner"] = 0

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
                    env_info["winner"] = win_idx + 1
                terminated = True

            self.player_turn *= -1
            self._update_legal_moves()
            zobrist = self.zobrist_array[128]
            self.hash = self._hash_xor(self.hash, zobrist)

            env_info["legal_actions"] = self.legal_actions()
            env_info["hash"] = self.hash

            return (
                self.vec_state(),
                rewards,
                terminated,
                terminated,
                env_info
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
                env_info
            )

        self.state[action // 8, action % 8] = self.player_turn

        zobrist_id = 64 * (self.player_turn == -1) + action
        zobrist = self.zobrist_array[zobrist_id]
        self.hash = self._hash_xor(self.hash, zobrist)

        zobrist = self.zobrist_array[128]
        self.hash = self._hash_xor(self.hash, zobrist)

        for i, j in self._get_flip_list(action):
            self.state[i, j] = self.player_turn
            flat_idx = i * 8 + j

            zobrist = self.zobrist_array[flat_idx]
            self.hash = self._hash_xor(self.hash, zobrist)

            zobrist = self.zobrist_array[64 + flat_idx]
            self.hash = self._hash_xor(self.hash, zobrist)

        self.player_turn *= -1
        self._update_legal_moves()

        env_info["legal_actions"] = self.legal_actions()
        env_info["hash"] = self.hash

        if self.render:
            self.display()

        return self.vec_state(), rewards, terminated, terminated, env_info


class MCTS:
    def __init__(self, explore_factor: float = 1.5, seed: int = 0):

        self.explore_factor = explore_factor
        np.random.seed(seed)

        self.tree = dict()
        self.cleanup()

        self.eval_history = []

    def size(self) -> float:
        """ Returns the size of the tree in MB rounded to four
        decimal places. """
        total_size = (
            sys.getsizeof(self.tree)
            +
            sys.getsizeof(self.prev_mcts.tree)
        )

        return round(total_size * 1e-6, 4)

    def cleanup(self) -> None:
        """ Cleanup must be performed immediately after backprop. """

        self.last_action = None
        self.visited = []

    def selection(self, hash: str) -> int:
        """ Gets the action, from the current state of the env
        encoded by the provided hash, which is optimal under the
        desired selection criteria (e.g. mean_uct). Selection
        should only be performed in the case where the current
        node/hash has no untried actions.
        ## Inputs:
        - hash : str. Zobrist hash representing the current
        state of the environment.
        ## Returns:
        - action : int. The action from the current state
        which is optimal under selection. """

        node = self.tree[hash]
        action_to_children = node["children"]

        best_action = 0
        best_uct = 0.0
        for action, children in action_to_children.items():
            uct_vals = np.zeros(shape=len(children), dtype=np.float32)

            for i, child in enumerate(children):
                parents = self.tree[child]["parents"]
                parent_nodes = [self.tree[parent] for parent in parents]

                if len(parents) > 0:
                    mean_parent_visits = np.mean(
                        [p_node["visits"] for p_node in parent_nodes],
                        dtype=np.float32
                    )
                    log_mpv = np.log(mean_parent_visits)
                else:
                    log_mpv = np.float32(0.0)

                wins = self.tree[child]["wins"]
                visits = self.tree[child]["visits"]

                win_freq = wins / visits
                explore_score = np.sqrt(log_mpv / visits)

                uct_value = win_freq + self.explore_factor * explore_score
                uct_vals[i] = uct_value

            mean_uct_val = np.mean(uct_vals)
            if mean_uct_val > best_uct:
                best_uct = mean_uct_val
                best_action = action

        return best_action

    def expansion(self, hash: str, legal_actions: set[int]) -> int:
        """ Performs a single expansion step from the current
        node/hash. This should be repeatedly called after each
        env step following the returned action value until the
        step returns None or the game terminates. A value of
        None indicates that the tree has actually expanded.
        ## Inputs:
        - hash : str. Zobrist hash representing the current
        state of the environment.
        - legal_actions : set[int]. Set containing the legal
        actions possible from the current state of the env.
        ## Returns:
        - action : int. The action from the current state
        which should be followed. While not None keep
        performing this operation. Once None is returned then
        random action simulation should be performed until the
        end of  the game is reached, at which point backprop
        should then be performed. """

        # If the current hash is new then add it to the tree
        if hash not in self.tree:
            action = None

            if len(self.tree) == 0:
                self.root_hash = hash

            if len(legal_actions) == 0:
                legal_actions = {0}

            node_info = {
                "untried_actions": legal_actions.copy(),
                "parents": set(),
                "children": {action: [] for action in legal_actions},
                "visits": 0,
                "wins": 0
            }
            self.tree[hash] = node_info

        # If the current hash has untried actions then return one of them
        # at random
        elif len(self.tree[hash]["untried_actions"]) > 0:
            untried_actions = self.tree[hash]["untried_actions"]
            action = int(np.random.choice(list(untried_actions)))

        # If the current hash has no untried actions then return the best
        # action in accordance with the selection method
        else:
            action = self.selection(hash)

        # Add parent and child information, remove the last action performed
        # from the set of untried actions for the previous hash
        parents = self.tree[hash]["parents"]
        if len(self.visited) > 0 and self.visited[-1] not in parents:
            parent_hash = self.visited[-1]
            parents.add(parent_hash)
            self.tree[parent_hash]["children"][self.last_action].append(hash)
            self.tree[parent_hash]["untried_actions"].discard(self.last_action)

        # Add the current hash to the list of visited hashes
        self.visited.append(hash)
        # Update the last action value
        self.last_action = action

        return action

    def backprop(self, winner: int) -> None:
        """ Performs the backpropagation step. This step should
        be performed after the random action simulation has
        terminated. Each node visited has their visits value
        incremented by 1 and those belonging to the winner have
        their wins value incremented by 1.0 also. If the game is
        a draw then wins are increased by 0.5 for all nodes visited.
        ## Inputs:
        - winner : int. The winner value should be 0 if the game
        ended in a draw, 1 if player one won and 2 if player two
        won. """

        for i, hash in enumerate(self.visited):
            # Add one visit to each visited node
            self.tree[hash]["visits"] += 1

            # Add one win to each of the visited
            # nodes belonging to the winner
            if winner == 0:
                self.tree[hash]["wins"] += 0.5

            elif i % 2 == winner % 2:
                self.tree[hash]["wins"] += 1.0

    def evaluate(self, env, eval_iters: int) -> dict[str,]:

        explore_factor = self.explore_factor
        self.explore_factor = 0.0
        eval_info = {
            "random": {"wins": 0, "draws": 0, "losses": 0, "score": 0.0},
            "previous": {"wins": 0, "draws": 0, "losses": 0, "score": 0.0}
        }
        for opp, eval_entry in eval_info.items():

            for _ in range(eval_iters):

                # Randomly choose which player the mcts will be
                tree_player = np.random.randint(2)

                # Reset the env
                player = 1
                env_info = env.reset()[-1]
                hash = env_info["hash"]

                # Perform selection until the new hash is unseen or
                # has untried actions. Then perform random actions
                # until the game is over
                terminated = False
                winner = 0
                while not terminated:

                    player = 1 - player
                    if player == tree_player:
                        curr_mcts = self
                    else:
                        curr_mcts = self.prev_mcts
                    curr_tree = curr_mcts.tree

                    get_random = (
                        (opp == "random" and player != tree_player)
                        or
                        hash not in curr_tree
                        or
                        len(curr_tree[hash]["untried_actions"]) > 0
                    )

                    legal_actions = env_info["legal_actions"]
                    if len(legal_actions) == 0:
                        action = 0
                    elif get_random:
                        action = np.random.choice(list(legal_actions))
                    else:
                        action = curr_mcts.selection(hash)
                        if action not in legal_actions:
                            action = np.random.choice(list(legal_actions))

                    _, _, terminated, _, env_info = env.step(action)
                    hash = env_info["hash"]
                    winner = env_info["winner"]

                # Update eval info
                if winner - 1 == player:
                    eval_entry["wins"] += 1
                    eval_entry["score"] += 1.0
                elif winner == 0:
                    eval_entry["draws"] += 1
                    eval_entry["score"] += 0.5
                else:
                    eval_entry["losses"] += 1

                win_rate = round(eval_entry["wins"] / eval_iters, 3)
                eval_entry["win_rate"] = win_rate

        self.explore_factor = explore_factor

        return eval_info

    def fit(
        self,
        env_fn: callable,
        n_iters: int,
        eval_every: int,
        eval_iters: int
    ) -> None:
        """ Fits the Monte Carlo Tree Search algorithm to the
        environment returned by env_fn(). This implementation
        assumes that the environment follows the gymnasium
        convention as seen in
        https://gymnasium.farama.org/index.html. The player
        whose turn it is should also alternate with prob 1.0
        after every step of the environment. It also assumes
        that the game returns an env_info dict after every
        env.reset() and env.step(action) call which contains
        items; "hash": zobrist_hash (str), "winner": who_won
        (int), "legal_actions": legal_actions (set[int]).
        ## Inputs:
        - env_fn : function. Function with no inputs that returns
        an instantiated environment.
        - n_iters : int. The number of backpropagation steps to
        perform. This is equivalent to the number of episodes
        to play. """

        env = env_fn()  # init the env

        if not hasattr(self, "prev_mcts"):
            self.prev_mcts = MCTS(self.explore_factor)

        pbar = trange(n_iters, ascii=True, desc="Fitting MCTS")
        for episode in pbar:
            pbar.set_postfix_str(
                f"tree_size={len(self.tree)}, memory_usage={self.size()}MB"
            )

            # Reset the env and perform initial expansion step
            player = 1
            env_info = env.reset()[-1]
            hash = env_info["hash"]
            legal_actions = env_info["legal_actions"]
            action = self.expansion(hash, legal_actions)

            # Perform expansion steps until either the game is
            # terminated or the expansion step returns None. If
            # None is returned this indicates we have discovered
            # a new node
            terminated = False
            winner = 0
            while not terminated and action is not None:
                player = 1 - player
                _, _, terminated, _, env_info = env.step(action)

                hash = env_info["hash"]
                legal_actions = env_info["legal_actions"]
                winner = env_info["winner"]
                action = self.expansion(hash, legal_actions)

            # Perform random simulation until the game is over
            while not terminated:
                player = 1 - player
                legal_actions = env_info["legal_actions"]
                action = 0
                if len(legal_actions) > 0:
                    action = np.random.choice(list(legal_actions))
                _, _, terminated, _, env_info = env.step(action)
                winner = env_info["winner"]

            # Perform backprop using the player who ended the
            # game and whether the game ended in a win or draw
            self.backprop(winner)
            self.cleanup()

            if episode % eval_every == eval_every - 1:
                eval_info = self.evaluate(env, eval_iters)
                self.eval_history.append(eval_info)
                self.prev_mcts.tree = deepcopy(self.tree)
                pprint(f"Evaluation: {self.eval_history[-1]}")
                print()
