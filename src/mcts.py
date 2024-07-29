

import sys
from copy import deepcopy

import numpy as np
from tqdm import trange
from pprint import pprint


class MCTS:
    def __init__(
        self,
        explore_factor: float = 1.5,
        prune_visits: int = 1000,
        prune_ratio: float = 0.35,
        seed: int = 0
    ):

        self.explore_factor = explore_factor
        self.prune_visits = prune_visits
        self.prune_ratio = prune_ratio
        np.random.seed(seed)

        self.tree = dict()
        self.tree_size = 0
        self.cleanup()

        self.max_depth = 0
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
        best_uct = -1.0
        for action, children in action_to_children.items():
            uct_vals = np.zeros(shape=len(children), dtype=np.float32)

            for i, child in enumerate(children):
                child_node = self.tree[child]
                if not child_node["pruned"]:
                    parents = child_node["parents"]
                    parent_nodes = [self.tree[parent] for parent in parents]

                    if len(parents) > 0:
                        mean_parent_visits = np.mean(
                            [p_node["visits"] for p_node in parent_nodes],
                            dtype=np.float32
                        )
                        log_mpv = np.log(mean_parent_visits)
                    else:
                        log_mpv = np.float32(0.0)

                    wins = child_node["wins"]
                    visits = child_node["visits"]

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
            depth = 10 ** 20

            if len(self.tree) == 0:
                self.root_hash = hash
                depth = 0

            if len(legal_actions) == 0:
                legal_actions = {0}

            node_info = {
                "untried_actions": legal_actions.copy(),
                "parents": set(),
                "children": {action: [] for action in legal_actions},
                "depth": depth,
                "pruned": False,
                "visits": 0,
                "wins": 0
            }
            self.tree[hash] = node_info
            self.tree_size += 1

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
        curr_depth = self.tree[hash]["depth"]
        parents = self.tree[hash]["parents"]
        if len(self.visited) > 0 and self.visited[-1] not in parents:
            parent_hash = self.visited[-1]
            parent_depth = self.tree[parent_hash]["depth"]
            parents.add(parent_hash)

            if parent_depth + 1 < curr_depth:
                self.tree[hash]["depth"] = parent_depth + 1
                self.max_depth = max(self.max_depth, parent_depth + 1)

            self.tree[parent_hash]["children"][self.last_action].append(hash)
            self.tree[parent_hash]["untried_actions"].discard(
                self.last_action)

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

            # Prune the node if it has visits greater than the
            # prune_visits value and a win_freq less than the
            # prune_ratio
            pruned = self.tree[hash]["pruned"]
            visits = self.tree[hash]["visits"]
            wins = self.tree[hash]["wins"]
            valid_visits = visits > self.prune_visits
            valid_ratio = wins / visits < self.prune_ratio
            if not pruned and valid_visits and valid_ratio:
                self.tree[hash]["pruned"] = True
                self.tree_size -= 1

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
                if winner - 1 == tree_player:
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
            # Initialise another mcts that is periodically updated
            # to the parent mcts object for evaluation purposes
            self.prev_mcts = MCTS(explore_factor=0.0)

        pbar = trange(n_iters, ascii=True, desc="Fitting MCTS")
        for episode in pbar:
            pbar.set_postfix_str(
                f"tree_size={self.tree_size}, max_depth={self.max_depth}, memory_usage={self.size()}MB"
            )

            # Reset the env and perform initial expansion step
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
                _, _, terminated, _, env_info = env.step(action)

                hash = env_info["hash"]
                legal_actions = env_info["legal_actions"]
                winner = env_info["winner"]
                action = self.expansion(hash, legal_actions)

            # Perform random simulation until the game is over
            while not terminated:
                action = 0
                if len(legal_actions) > 0:
                    action = np.random.choice(list(legal_actions))
                _, _, terminated, _, env_info = env.step(action)

                legal_actions = env_info["legal_actions"]
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
