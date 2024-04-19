class FMDP:
    """
    Finite Markov Decision Process (FMDP) class.

    Args:
        states (list): List of states in the environment.
        actions (list): List of actions available in the environment.
        transitions (dict): Dictionary representing the state transition probabilities.
        rewards (dict): Dictionary representing the rewards for each state-action pair.
        gamma (float, optional): Discount factor for future rewards. Defaults to 0.9.
        epsilon (float, optional): Convergence threshold for value iteration. Defaults to 0.0001.

    Attributes:
        states (list): List of states in the environment.
        actions (list): List of actions available in the environment.
        transitions (dict): Dictionary representing the state transition probabilities.
        rewards (dict): Dictionary representing the rewards for each state-action pair.
        gamma (float): Discount factor for future rewards.
        epsilon (float): Convergence threshold for value iteration.

    Methods:
        get_next_state(state, action): Returns the next state given the current state and action.
        get_reward(state, action): Returns the reward for a given state-action pair.
        value_iteration(): Performs value iteration to find the optimal state values and policy.
        get_action_value(state, action, V): Calculates the value of an action in a given state.
        extract_policy(V): Extracts the optimal policy based on the optimal state values.
        run_optimal_policy(env, policy_optimal, num_episodes, max_steps): Runs the optimal policy in the environment.

    """

    def __init__(self, states, actions, transitions, rewards, gamma=0.9, epsilon=0.0001):
        self.states = states
        self.actions = actions
        self.transitions = transitions
        self.rewards = rewards
        self.gamma = gamma
        self.epsilon = epsilon

    def get_next_state(self, state, action):
        return self.transitions[state][action]

    def get_reward(self, state, action):
        return self.rewards[state][action]

    def value_iteration(self):
        # Realiza a iteração de valor para encontrar os valores ótimos de estado e a política ótima
        V = {s: 0 for s in self.states}
        while True:
            delta = 0
            for state in self.states:
                v = V[state]
                max_q_value = float('-inf')
                for action in self.actions:
                    q_value = self.get_action_value(state, action, V)
                    if q_value > max_q_value:
                        max_q_value = q_value
                V[state] = max_q_value
                delta = max(delta, abs(v - V[state]))
            if delta < self.epsilon:
                break
        policy = self.extract_policy(V)
        return V, policy

    def get_action_value(self, state, action, V):
        # Calcula o valor de uma ação em um estado com base nos valores de estado
        next_state = self.get_next_state(state, action)
        reward = self.get_reward(state, action)
        return reward + self.gamma * V[next_state]

    def extract_policy(self, V):
        # Extrai a política ótima com base nos valores ótimos de estado
        policy = {}
        for state in self.states:
            max_q_value = float('-inf')
            best_action = None
            for action in self.actions:
                q_value = self.get_action_value(state, action, V)
                if q_value > max_q_value:
                    max_q_value = q_value
                    best_action = action
            policy[state] = best_action
        return policy
    
    def run_optimal_policy(self, env, policy_optimal, num_episodes, max_steps=10):
        rewards = []
        for _ in range(num_episodes):
            state = env.reset()
            done = False
            episode_reward = 0
            total_rewards = 0
            for i in range(max_steps):
                action = policy_optimal[state]
                state, reward, done, _ = env.step(action)
                episode_reward += reward
                if done:
                    break
            rewards.append(episode_reward)
        return rewards
    