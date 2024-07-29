

from copy import deepcopy
from time import time

import ray
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from pprint import pprint


@ray.remote
class AlphaZeroSelfPlayAgent:
    def __init__(
        self,
        agent_id: int,
    ):
        self.agent_id = agent_id

    def set_env(self, env_init) -> None:
        self.env_init = env_init

    def reset(self) -> None:
        self.env = deepcopy(self.env_init)

    def step(self, action: int) -> None:
        state, _, terminated, _, env_info = self.env.step(action)
        return state, terminated, env_info

class AlphaZeroMCTSController:
    def __init__(
        self,
        selfplay_agents,
        action_space: int,
        state: np.ndarray[np.float32],
        env_info: dict[str,],
        policy_network: tf.keras.Model,
        prev_inferences: dict[str,],
        simulations: int,
        explore_factor: float,
        epsilon: float,
    ):
        self.action_space = action_space
        self.state_init = state
        self.env_info_init = env_info

        self.selfplay_agents = selfplay_agents
        self.num_agents = len(selfplay_agents)
        self.sims_per_agent = simulations // self.num_agents
        self.policy_network = policy_network
        self.prev_inferences = prev_inferences
        self.simulations = simulations
        self.explore_factor = explore_factor
        self.epsilon = epsilon

        self.tree = dict()
        self.tree_size = 0
        self.max_depth = 0

        self.visited = {
            i: [] for i in range(self.num_agents)
        }
        self.performed = {
            i: [] for i in range(self.num_agents)
        }

        self.eval_history = []

    def expansion(self, env_info_list: list[dict[str, ]]) -> int:

        sampled_actions = np.zeros(shape=self.num_agents, dtype=np.int32)
        agent_indices = range(self.num_agents)

        for agent_idx, env_info in zip(agent_indices, env_info_list):
            hash = env_info["hash"]
            legal_actions = env_info["legal_actions"]

            visited = self.visited[agent_idx]
            performed = self.performed[agent_idx]

            # If the current hash is new then add it to the tree
            if hash not in self.tree:
                num_actions = len(legal_actions)
                if num_actions == 0:
                    legal_actions = [0]
                    num_actions = 1

                if len(self.tree) == 0:
                    parent = ""
                    depth = 0
                    self.root_hash = hash

                else:
                    # Add parent and child information
                    parent = visited[-1]
                    depth = self.tree[parent]["depth"] + 1
                    self.max_depth = max(self.max_depth, depth)
                    self.tree[parent]["children"][performed[-1]] = hash

                node_info = {
                    "parent": parent,
                    "actions": list(legal_actions),
                    "action_ids": {action: i for i, action in enumerate(legal_actions)},
                    "children": {action: "" for action in legal_actions},
                    "child_visits": np.zeros(shape=num_actions, dtype=np.int32),
                    "total_value": np.zeros(shape=num_actions, dtype=np.float32),
                    "mean_value": np.zeros(shape=num_actions, dtype=np.float32),
                    "prior_prob": np.zeros(shape=num_actions, dtype=np.float32),
                    "depth": depth,
                    "visits": 0,
                }

                self.tree[hash] = node_info
                self.tree_size += 1

                action = -1

            # If the current hash is already in the tree then
            # we perform selection using puct metric
            else:
                if np.random.random() < self.epsilon:
                    legal_actions = self.tree[hash]["actions"]
                    action = np.random.choice(legal_actions)
                else:
                    action = self.selection(hash)

            sampled_actions[agent_idx] = action

            # Add the current hash to the list of visited hashes
            visited.append(hash)
            # Add the last action to the list of performed actions
            performed.append(action)

        return sampled_actions

    def selection(self, hash: str) -> int:
        """ Gets the action, from the current state of the env
        encoded by the provided hash, which is optimal under the
        desired selection criteria. In this case the criteria is
        the PUCT (predictor upper confidence bound for trees)
        metric. This method uses proportional probability
        sampling to choose the action to perform. This is to
        avoid a determinstic sampling method from making all
        agents pick the same action as they act in parallel.
        ## Inputs:
        - hash : str. Zobrist hash representing the current
        state of the environment.
        ## Returns:
        - action : int. The action from the current state
        which is optimal under selection. """

        node = self.tree[hash]

        exploit_scores = node["mean_value"]
        explore_weights = self.explore_factor * node["prior_prob"]
        explore_scores = np.sqrt(node["visits"]) / (node["child_visits"] + 1)

        puct_values = exploit_scores + explore_weights * explore_scores
        exp_puct_values = np.exp(puct_values)
        puct_probs = exp_puct_values / np.sum(exp_puct_values)

        sampled_action_idx = np.random.choice(len(puct_values), p=puct_probs)
        sampled_action = node["actions"][sampled_action_idx]

        return sampled_action

    def fit(self) -> None:
        sim_counts = np.zeros_like(self.selfplay_agents, dtype=np.int32)
        to_reset = [i for i in range(self.num_agents)]

        state_list = [self.state_init for _ in range(self.num_agents)]
        env_info_list = [self.env_info_init for _ in range(self.num_agents)]

        while np.any(sim_counts < self.sims_per_agent):
            for agent_idx in to_reset:
                self.selfplay_agents[agent_idx].reset.remote()
                state_list[agent_idx] = self.state_init
                env_info_list[agent_idx] = self.env_info_init
                self.visited[agent_idx] = []
                self.performed[agent_idx] = []

            actions = self.expansion(env_info_list)

            to_reset = set()
            agent_indices = []
            agent_commands = []
            for agent_idx, action in enumerate(actions):
                if action == -1:
                    sim_counts[agent_idx] += 1
                    to_reset.add(agent_idx)
                    end_state = state_list[agent_idx]
                    with tf.device("/CPU:0"):
                        self.backprop(agent_idx, end_state)
                else:
                    agent = self.selfplay_agents[agent_idx]
                    agent_commands.append(
                        agent.step.remote(action)
                    )
                    agent_indices.append(agent_idx)

            env_step_res = ray.get(agent_commands)

            for agent_idx, result in zip(agent_indices, env_step_res):
                state, terminated, env_info = result
                state_list[agent_idx] = state
                env_info_list[agent_idx] = env_info

                if terminated:
                    sim_counts[agent_idx] += 1
                    to_reset.add(agent_idx)
                    end_state = state_list[agent_idx]
                    with tf.device("/CPU:0"):
                        self.backprop(agent_idx, end_state)

        action_space = self.action_space

        root_children_visits = self.tree[self.root_hash]["child_visits"]
        root_actions = self.tree[self.root_hash]["actions"]

        sim_probs = np.zeros(shape=action_space, dtype=np.float32)
        sim_probs[root_actions] = root_children_visits / np.sum(root_children_visits)

        return sim_probs

    def backprop(self, agent_idx: int, end_state: np.ndarray[np.float32]) -> None:

        visited = self.visited[agent_idx]
        performed = self.performed[agent_idx]

        last_hash = visited[-1]
        last_node = self.tree[last_hash]

        if last_hash in self.prev_inferences:
            action_logits, state_value = self.prev_inferences[last_hash]
        else:
            action_logits, state_value = self.policy_network(
                tf.expand_dims(end_state, axis=0)
            )
            action_logits = action_logits[0]
            state_value = state_value[0]
            self.prev_inferences[last_hash] = (action_logits, state_value)

        legal_actions = last_node["actions"]
        legal_logits = tf.gather(action_logits, legal_actions)
        action_probs = tf.nn.softmax(legal_logits).numpy()

        # Initialise the prior prob values for the last visited
        # nodes children
        last_node["prior_prob"] += action_probs

        for hash, action in zip(visited, performed):
            node = self.tree[hash]

            # Add one visit to each visited node
            node["visits"] += 1

            # Aggregate the child visits, state action value
            if action != -1:
                action_idx = node["action_ids"][action]

                node["total_value"][action_idx] += state_value.numpy()
                node["child_visits"][action_idx] += 1

                child_value = node["total_value"][action_idx]
                child_visits = node["child_visits"][action_idx]

                node["mean_value"][action_idx] = child_value / child_visits

class AlphaZeroMCTS:
    def __init__(
        self,
        env,
        state: np.ndarray[np.float32],
        env_info: dict[str,],
        policy_network: tf.keras.Model,
        prev_inferences: dict[str,],
        simulations: int,
        explore_factor: float,
        epsilon: float
    ):
        self.env_init = env
        self.state_init = state
        self.env_info_init = env_info

        self.policy_network = policy_network
        self.prev_inferences = prev_inferences
        self.simulations = simulations
        self.explore_factor = explore_factor
        self.epsilon = epsilon

        self.tree = dict()
        self.tree_size = 0
        self.cleanup()

        self.max_depth = 0
        self.eval_history = []

    def cleanup(self) -> None:
        """ Cleanup must be performed immediately after backprop. """

        self.visited = []
        self.performed = []
        self.last_state = None

    def expansion(self, state: np.ndarray[np.float32], hash: str, legal_actions: set[int]) -> int:
        """ Performs a single expansion step from the current
        node/hash. This function returns the next action
        which should be taken during the simulation. While
        the returned value is not None this function should
        be repeated. A return value of None indicates that
        the current state is a new leaf.
        ## Inputs:
        - state: np.ndarray[np.float32]. State array which
        represents all the information of the current state
        of the environment.
        - hash : str. Zobrist hash representing the current
        state of the environment.
        - legal_actions : set[int]. Set containing the legal
        actions possible from the current state of the env.
        ## Returns:
        - action : int. The action from the current state
        which should be followed. Once the game terminates
        the backprop function should be called. """

        # If the current hash is new then add it to the tree
        if hash not in self.tree:
            num_actions = len(legal_actions)
            if num_actions == 0:
                legal_actions = [0]
                num_actions = 1

            if len(self.tree) == 0:
                parent = ""
                depth = 0
                self.root_hash = hash

            else:
                # Add parent and child information
                parent = self.visited[-1]
                depth = self.tree[parent]["depth"] + 1
                self.max_depth = max(self.max_depth, depth)
                self.tree[parent]["children"][self.performed[-1]] = hash

            node_info = {
                "parent": parent,
                "actions": list(legal_actions),
                "action_ids": {action: i for i, action in enumerate(legal_actions)},
                "children": {action: "" for action in legal_actions},
                "child_visits": np.zeros(shape=num_actions, dtype=np.int32),
                "total_value": np.zeros(shape=num_actions, dtype=np.float32),
                "mean_value": np.zeros(shape=num_actions, dtype=np.float32),
                "prior_prob": np.zeros(shape=num_actions, dtype=np.float32),
                "depth": depth,
                "visits": 0,
            }

            self.tree[hash] = node_info
            self.tree_size += 1

            action = None

        # If the current hash is already in the tree then
        # we perform selection using puct metric
        else:
            if np.random.random() < self.epsilon:
                legal_actions = self.tree[hash]["actions"]
                action = np.random.choice(legal_actions)
            else:
                action = self.selection(hash)

        # Add the current hash to the list of visited hashes
        self.visited.append(hash)
        # Add the last action to the list of performed actions
        self.performed.append(action)
        # Update the last state value
        self.last_state = state

        return action

    def selection(self, hash: str) -> int:
        """ Gets the action, from the current state of the env
        encoded by the provided hash, which is optimal under the
        desired selection criteria. In this case the criteria is
        the PUCT (predictor upper confidence bound for trees)
        metric.
        ## Inputs:
        - hash : str. Zobrist hash representing the current
        state of the environment.
        ## Returns:
        - action : int. The action from the current state
        which is optimal under selection. """

        node = self.tree[hash]

        exploit_scores = node["mean_value"]
        explore_weights = self.explore_factor * node["prior_prob"]
        explore_scores = np.sqrt(node["visits"]) / (node["child_visits"] + 1)

        puct_values = exploit_scores + explore_weights * explore_scores
        best_action_idx = np.argmax(puct_values)

        best_action = node["actions"][best_action_idx]

        return best_action

    def backprop(self) -> None:
        """ Performs the backpropagation step. This step should
        be performed after the simulation has terminated. Each
        node visited has their visits value incremented by 1.
        Other attributes such as total_action_values and
        mean_action_values are also incremented by the value
        given by the value network. Prior probs for each action
        is also updated by the policy network. """

        last_hash = self.visited[-1]
        last_visited = self.tree[last_hash]

        if last_hash in self.prev_inferences:
            action_logits, state_value = self.prev_inferences[last_hash]
        else:
            action_logits, state_value = self.policy_network(
                tf.expand_dims(self.last_state, axis=0)
            )
            action_logits = action_logits[0]
            state_value = state_value[0]
            self.prev_inferences[last_hash] = (action_logits, state_value)

        legal_actions = last_visited["actions"]
        legal_logits = tf.gather(action_logits, legal_actions)
        action_probs = tf.nn.softmax(legal_logits).numpy()

        # Initialise the prior prob values for the last visited
        # nodes children
        last_visited["prior_prob"] = action_probs

        for hash, action in zip(self.visited, self.performed):
            node = self.tree[hash]

            # Add one visit to each visited node
            node["visits"] += 1

            # Aggregate the child visits, state action value
            if action is not None:
                action_idx = node["action_ids"][action]

                node["total_value"][action_idx] += state_value.numpy()
                node["child_visits"][action_idx] += 1

                child_value = node["total_value"][action_idx]
                child_visits = node["child_visits"][action_idx]

                node["mean_value"][action_idx] = child_value / child_visits

    def fit(self) -> dict[int, float]:
        """ Fits the Monte Carlo Tree Search algorithm to the
        deepcopied environment. This implementation
        assumes that the environment follows the gymnasium
        convention as seen in
        https://gymnasium.farama.org/index.html. The player
        whose turn it is should also alternate with prob 1.0
        after every step of the environment. It also assumes
        that the game returns an env_info dict after every
        env.reset() and env.step(action) call which contains
        items; "hash": zobrist_hash (str), "winner": who_won
        (int), "legal_actions": legal_actions (set[int]).
        ## Returns:
        - simulation_probs : dict[int, float]. An array
        containing the probabilities of each possible action
        being optimal from the root state. These probabilities
        are proportional to the number of visits for each
        child node from the root. """

        for _ in range(self.simulations):
            # Reset the env
            env = deepcopy(self.env_init)
            state = self.state_init
            env_info = self.env_info_init

            hash = env_info["hash"]
            legal_actions = env_info["legal_actions"]

            # Get the initial action via expansion
            action = self.expansion(state, hash, legal_actions)

            # Loop until the game ends or a new node is found
            terminated = False
            while not terminated and action is not None:
                state, _, terminated, _, env_info = env.step(action)

                hash = env_info["hash"]
                legal_actions = env_info["legal_actions"]

                if not terminated and action is not None:
                    action = self.expansion(state, hash, legal_actions)

            # Backpropogate through visited nodes updating
            # their inference values
            with tf.device("/CPU:0"):
                self.backprop()
            self.cleanup()

        action_space = self.env_init.action_space

        root_children_visits = self.tree[self.root_hash]["child_visits"]
        root_actions = self.tree[self.root_hash]["actions"]

        sim_probs = np.zeros(shape=action_space, dtype=np.float32)
        sim_probs[root_actions] = root_children_visits / (self.simulations - 1)

        return sim_probs

class AlphaZero:
    def __init__(
        self,
        epochs: int = 5,
        buffer_size: int = 100000,
        batch_size: int = 128,
        learning_rate: float = 1e-3,
        policy_layers: list[int] = [256, 256],
        search_agents: int = 8,
        search_sims: int = 100,
        search_explore_rate: float = 1.5,
        search_epsilon: float = 0.0,
    ):
        self.epochs = epochs
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=learning_rate
        )

        self.policy_layers = policy_layers

        self.search_agents = search_agents
        self.search_sims = search_sims
        self.search_explore_rate = search_explore_rate
        self.search_epsilon = search_epsilon

    def _init_policy(self, state_space, action_space) -> None:

        s = tf.keras.layers.Input(shape=state_space)
        x = tf.keras.layers.Dense(units=self.policy_layers[0], activation="tanh")(s)
        for units in self.policy_layers[1:]:
            x = tf.keras.layers.Dense(units=units, activation="tanh")(x)

        initializer = tf.keras.initializers.RandomNormal(
            mean=0,
            stddev=0.05
        )
        p = tf.keras.layers.Dense(
            units=action_space,
            activation=None,
            kernel_initializer=initializer
        )(x)

        v = tf.keras.layers.Dense(
            units=1,
            activation="tanh"
        )(x)

        self.policy_network = tf.keras.Model(inputs=[s], outputs=[p, v])
        self.policy_network.summary()

    @tf.function
    def _train_step(
        self,
        policy_network: tf.keras.Model,
        optimizer: tf.keras.optimizers.Optimizer,
        states: tf.Tensor,
        sim_probs: tf.Tensor,
        logit_masks: tf.Tensor,
        outcomes: tf.Tensor
    ) -> tuple[tf.Tensor, tf.Tensor]:

        with tf.GradientTape() as tape:
            action_logits, state_values = policy_network(states)
            action_logits += logit_masks

            policy_loss = tf.math.reduce_mean(
                tf.keras.losses.categorical_crossentropy(
                    y_true=sim_probs,
                    y_pred=action_logits,
                    from_logits=True
                )
            )

            critic_loss = tf.math.reduce_mean(
                tf.keras.losses.mean_squared_error(
                    y_true=outcomes,
                    y_pred=state_values
                )
            )

            total_loss = policy_loss + critic_loss

        grads = tape.gradient(total_loss, policy_network.trainable_variables)
        optimizer.apply_gradients(zip(grads, policy_network.trainable_variables))

        return policy_loss, critic_loss

    def _training_loop(
        self,
        states_buffer: list[np.ndarray],
        sim_probs_buffer: list[np.ndarray],
        logit_masks_buffer: list[np.ndarray],
        outcomes_buffer: list[np.ndarray]
    ) -> None:

        dataset = tf.data.Dataset.from_tensor_slices(
            (states_buffer, sim_probs_buffer, logit_masks_buffer, outcomes_buffer)
        )
        dataset = dataset.shuffle(buffer_size=1000)
        dataset = dataset.batch(batch_size=self.batch_size)
        dataset = dataset.repeat(count=self.epochs)

        dataset_size = 0
        for _ in dataset:
            dataset_size += 1

        pbar = tqdm(total=dataset_size, initial=0, ascii=True, desc="Training Network")
        i = 0
        for batch in dataset:
            states, sim_probs, logit_masks, outcomes = batch

            policy_loss, critic_loss = self._train_step(
                self.policy_network,
                self.optimizer,
                states,
                sim_probs,
                logit_masks,
                outcomes
            )
            pbar.set_postfix({
                "pi_loss": policy_loss.numpy().item(),
                "va_loss": critic_loss.numpy().item()
            })
            pbar.update(1)
            i += 1

        pbar.close()

    def _create_mcts(
        self,
        env,
        state,
        env_info,
        prev_inferences
    ):
        assert type(self.search_agents) == int
        assert self.search_agents > 0

        if self.search_agents == 1:
            return AlphaZeroMCTS(
                env,
                state,
                env_info,
                self.policy_network,
                prev_inferences,
                self.search_sims,
                self.search_explore_rate,
                self.search_epsilon
            )

        if not hasattr(self, "selfplay_agents"):
            ray.shutdown()
            ray.init()
            self.selfplay_agents = [
                AlphaZeroSelfPlayAgent.remote(i)
                for i in range(self.search_agents)]
        
        [agent.set_env.remote(env) for agent in self.selfplay_agents]

        return AlphaZeroMCTSController(
            self.selfplay_agents,
            env.action_space,
            state,
            env_info,
            self.policy_network,
            prev_inferences,
            self.search_sims,
            self.search_explore_rate,
            self.search_epsilon
        )

    def fit(
        self,
        env_fn: callable,
        n_episodes: int = 100000,
        update_every: int = 1000
    ) -> None:

        env = env_fn()
        state_space = env.reset()[0].shape
        action_space = env.action_space
        self._init_policy(state_space, action_space)

        prev_inferences = dict()
        states_buffer = []
        sim_probs_buffer = []
        logit_masks_buffer = []
        outcomes_buffer = []

        game_length_buffer = []

        pbar = tqdm(total=n_episodes, initial=0, ascii=True, desc="Self-Play")
        for episode in range(n_episodes):
            state, env_info = env.reset()

            num_steps = 0
            terminated = False
            while not terminated:
                # Initialise and fit the MCTS to get the simulation
                # action prob vector with the current state acting
                # as the root node
                mcts = self._create_mcts(
                    env,
                    state,
                    env_info,
                    prev_inferences
                )
                sim_probs = mcts.fit()
                logit_masks = np.where(sim_probs > 1e-6, np.float32(0.0), np.float32(-100.0))

                if len(states_buffer) == self.buffer_size:
                    states_buffer.pop(0)
                    sim_probs_buffer.pop(0)
                    logit_masks_buffer.pop(0)

                states_buffer.append(state)
                sim_probs_buffer.append(sim_probs)
                logit_masks_buffer.append(logit_masks)
                num_steps += 1

                action = np.random.choice(action_space, p=sim_probs)
                state, _, terminated, _, env_info = env.step(action)

            if episode >= 100:
                game_length_buffer.pop(0)
            game_length_buffer.append(num_steps)

            winner = env_info["winner"]
            for i in range(num_steps):
                if len(outcomes_buffer) == self.buffer_size:
                    outcomes_buffer.pop(0)

                if winner == 0:
                    outcomes_buffer.append(np.zeros(1, dtype=np.float32))
                else:
                    if i % 2 == winner - 1:
                        outcome = +1
                    else:
                        outcome = -1
                    outcomes_buffer.append(outcome * np.ones(1, dtype=np.float32))

            buff_size_disp = len(states_buffer)
            if buff_size_disp < 10**3:
                post_symbol = ''
            elif buff_size_disp < 10**6:
                buff_size_disp = round(buff_size_disp / 10**3, 2)
                post_symbol = 'k'
            else:
                buff_size_disp = round(buff_size_disp / 10**6, 2)
                post_symbol = 'M'

            pbar.update(1)
            pbar.set_postfix({
                "buffer_size": str(buff_size_disp) + post_symbol,
                "game_length": np.mean(game_length_buffer)
            })
            if episode % update_every == update_every - 1:
                pbar.close()
                self._training_loop(
                    states_buffer,
                    sim_probs_buffer,
                    logit_masks_buffer,
                    outcomes_buffer
                )
                prev_inferences = dict()
                print()
                pbar = tqdm(total=n_episodes, initial=episode, ascii=True, desc="Self-Play")
