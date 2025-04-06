import numpy as np

class ReplayBuffer():
    def __init__(self, max_size, input_shape, action_size):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, input_shape))
        self.new_state_memory = np.zeros((self.mem_size, input_shape))
        self.action_memory = np.zeros((self.mem_size, action_size))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)

    def add_to_buffer(self, state, action, reward, n_state, done):
        # index can account for wrap around
        index = self.mem_cntr % self.mem_size

        # add the new info to the memory
        self.state_memory[index] = np.asarray(state)
        self.new_state_memory[index] = np.asarray(n_state)
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)

        # choose batch size amount of different indices to sample
        batch = np.random.choice(max_mem, batch_size)

        # sample from the memory based on the indices selected randomly above
        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        n_states = self.new_state_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, n_states, dones

