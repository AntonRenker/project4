import numpy as np

class ReplayMemory:
    '''
    This class is used to store the experiences of the agent.
    
    Attributes:
    capacity: int
        The maximum number of experiences that the memory can store.
    memory: list
        The list of experiences.
    idx: int
        The index of the last experience added to the memory.

    Methods:
    add: None
        Adds an experience to the memory.
    sample: tuple
        Samples a batch of experiences from the memory (uniform).
    '''
    def __init__(self, capacity: int)->None:
        self.capacity = capacity
        self.memory = []
        self.idx = 0
    
    def add(self, state, action, reward, next_state) -> None:
        expirience = (state, action, reward, next_state)
        if len(self.memory) < self.capacity:
            self.memory.append(expirience)
        else:
            self.memory[self.idx] = expirience
            self.idx = (self.idx + 1) % self.capacity
    
    def sample(self, batch_size) -> tuple:
        if len(self.memory) < batch_size:
            batch_size = len(self.memory)
        indeces = np.random.choice(len(self.memory), batch_size, replace=False)
        
        states, actions, rewards, next_states = [], [], [], []
        for i in indeces:
            state, action, reward, next_state = self.memory[i]
            states.append(list(state))
            actions.append(action)
            rewards.append(reward)
            next_states.append(list(next_state))
        return states, actions, rewards, next_states