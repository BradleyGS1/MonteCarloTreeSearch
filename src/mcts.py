

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
        """ ## Returns:
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
        env_info["win"] = 0

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
        env_info["win"] = 0

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
            env_info["win"] = 1

        elif len(self.legal_moves) == 0:
            terminated = True

        self.player_turn *= -1

        return self.state.copy(), rewards, terminated, terminated, env_info


class MCTS:
    def __init__(self, explore_factor: float = 1.5, seed: int = 0):

        self.explore_factor = explore_factor
        np.random.seed(seed)

        self.tree = dict()
        self.cleanup()

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
    
    def backprop(self, player: int, win: int) -> None:
        """ Performs the backpropagation step. This step should 
        be performed after the random action simulation has 
        terminated. Each node visited has their visits value 
        incremented by 1 and those belonging to the winner have 
        their wins value incremented by 1.0 also. If the game is 
        a draw then wins are increased by 0.5 for all nodes visited. 
        ## Inputs:
        - player : int. The player who made the move ending the
        game. Requires a value of 0 for player one and a value
        of 1 for player two.  
        - win : int. Is 0 if the game ended in a draw, otherwise
        takes the value 1. """

        for i, hash in enumerate(self.visited):
            # Add one visit to each visited node
            self.tree[hash]["visits"] += 1

            # Add one win to each of the visited 
            # nodes belonging to the winner
            if not win:
                self.tree[hash]["wins"] += 0.5

            elif win and i % 2 == player:
                self.tree[hash]["wins"] += 1
