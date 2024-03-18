from replay import ReplayMemory
from actionValueFunction import ActionValueFunction
import numpy as np
from environment import Environment as Env

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def train_agents(env, rows=4, columns=5, N=100_000, num_episodes=100, gamma=0.99, alpha=0.01, epsilon=0.1, batch_size=32, C=10):
    # Initialize replay memory D for both agents to capacity N
    D_1 = ReplayMemory(N)
    D_2 = ReplayMemory(N)

    # Initialize action-value function Q with random weights h for both agents
    Q_1 = ActionValueFunction(input_size=rows*columns + 1, output_size=columns)
    Q_2 = ActionValueFunction(input_size=rows*columns + 1, output_size=columns)

    # Initialize target action-value function Q' with weights h' = h for both agents
    Q_1_target = ActionValueFunction(input_size=rows*columns + 1, output_size=columns)
    Q_1_target.model.set_weights(Q_1.model.get_weights())
    Q_2_target = ActionValueFunction(input_size=rows*columns + 1, output_size=columns)
    Q_2_target.model.set_weights(Q_2.model.get_weights())

    for episode in range(num_episodes):
        print(f'Episode: {episode}')
        for _ in range(100):
            part_train(Q_1, Q_2, Q_1_target, D_1, env, gamma, alpha, epsilon, batch_size, C, columns, player=1)
            part_train(Q_2, Q_1, Q_2_target, D_2, env, gamma, alpha, epsilon, batch_size, C, columns, player=-1)

    return Q_1, Q_2
        
def part_train(Q_1, Q_2, Q_1_target, D, env, gamma, alpha, epsilon, batch_size, C, columns, player): 
    state = env.reset().copy() # TBD
    done = False
            
    t = 0

    while not done:
        # With probability epsilon select a random action a_t
        action_1 = get_action(columns, epsilon, Q_1, state)
                
        # Execute action at and observe reward r_t and next state s_t+1
        next_state, reward, done = env.step(action_1, player, Q_2) # TBD

        # Store transition in memory D_1
        D.add(state, action_1, reward, next_state.copy()) # TBD

        # Sample random minibatch of transitions (s_j, a_j, r_j, s_j+1) from D_1
        states, actions, rewards, next_states = D.sample(batch_size)

        # Set y_j = r_j if episode terminates at step j+1 
        # else r_j + gamma * max_a' Q'(s_j+1, a'; h')
        max_values_target = get_max_values(Q_1_target, next_states, columns)
        states_actions = get_states_actions(states, actions)

        y_j = rewards if done else rewards + gamma * np.array(max_values_target)
        
        # Perform a gradient descent step on (y_j - Q(s_j, a_j; h))^2 with respect to the network parameters h
        Q_1.model.fit(np.array(states_actions), np.array(y_j), epochs=1, batch_size=len(states), verbose=0)
        

        # Every C steps reset Q' = Q
        if t % C == 0:
            Q_1_target.model.set_weights(Q_1.model.get_weights())

        t += 1
        state = next_state.copy()

def get_states_actions(states, actions):
    states_actions = []
    for state, action in zip(states, actions):
        state_action = list(np.concatenate((state, [action])))
        states_actions.append(state_action)
    return states_actions
        

def get_action(columns, epsilon, Q, state):
    if np.random.rand() < epsilon:
        return np.random.randint(columns)
    else:
        # Wrong TBD
        return get_max_action(Q, state, columns)
    
def get_max_action(Q, state, columns):
    max_value = float('-inf')
    max_action = 0
    for i in range(columns):
        state_action = np.concatenate((state, [i])).reshape(1, -1)
        value = Q.predict(state_action)
        if value > max_value:
            max_value = value
            max_action = i
        return max_action
    
def get_max_values(Q, states, columns):
    max_values = []
    for state in states:
        max_value = float('-inf')
        for action in range(columns):
            state_action = np.concatenate((state, [action])).reshape(1, -1)
            value = Q.predict(state_action)[0][0]
            if value > max_value:
                max_value = value
        max_values.append(max_value)
    return max_values


rows = 4
columns = 5
num_wins = 3
env = Env(columns, rows , num_wins)

N = 100_000
num_episodes = 1_000
gamma = 0.99
alpha = 0.01
epsilon = 0.1
batch_size = 32
C = 10

Q_1, Q_2 = train_agents(env=env, rows=rows, columns=columns, N=N, num_episodes=num_episodes, gamma=gamma, alpha=alpha, epsilon=epsilon, batch_size=batch_size, C=C)

# Store the trained agents
Q_1.model.save('model/Q_1')
Q_2.model.save('model/Q_2')