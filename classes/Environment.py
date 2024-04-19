import numpy as np

class Environment:
    def __init__(self) -> None:
        self.states = ['s1', 's2', 's3']
        self.actions = ['a1', 'a2']
        self.transitions = {
            's1': {'a1': 's1', 'a2': 's2'},
            's2': {'a1': 's3', 'a2': 's1'},
            's3': {'a1': 's3', 'a2': 's2'}
        }
        self.rewards = {
            's1': {'a1': -1, 'a2': 5},
            's2': {'a1': 10, 'a2': 5},
            's3': {'a1': -1, 'a2': -15}
        }

    def reset(self):
        self.current_state = np.random.choice(self.states)
        return self.current_state
    
    def step(self, action):
        next_state = self.transitions[self.current_state][action]
        reward = self.rewards[self.current_state][action]
        self.current_state = next_state
        return next_state, reward, False, {}
