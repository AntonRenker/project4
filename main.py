from replay import ReplayMemory
from actionValueFunction import ActionValueFunction
import numpy as np

def train_agents(env, rows=4, columns=4, N=100_000, num_episodes=100, gamma=0.99, alpha=0.01, epsilon=0.1, batch_size=32, C=10):
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
        for _ in range(1000):
            part_train(Q_1, Q_2, Q_1_target, D_1, env, gamma, alpha, epsilon, batch_size, C)
            part_train(Q_2, Q_1, Q_2_target, D_2, env, gamma, alpha, epsilon, batch_size, C)

    return Q_1, Q_2
        
def part_train(Q_1, Q_2, Q_1_target, D, env, gamma, alpha, epsilon, batch_size, C): 
    state = env.reset() # TBD
    done = False
            
    t = 0
    while not done:
        # With probability epsilon select a random action a_t
        action_1 = get_action(columns, epsilon, Q_1, state)
                
        # Execute action at and observe reward r_t and next state s_t+1
        next_state, reward, done = env.step(action_1, 1, Q_2) # TBD

        # Store transition in memory D_1
        D.add(state, action_1, reward, next_state)

        # Sample random minibatch of transitions (s_j, a_j, r_j, s_j+1) from D_1
        states, actions, rewards, next_states = D.sample(batch_size)

        # Set y_j = r_j if episode terminates at step j+1 
        # else r_j + gamma * max_a' Q'(s_j+1, a'; h')
        
        states = np.array(states)
        actions = np.array(actions).reshape(-1, 1)  # Reshape, um eine Spalte zu erzeugen
        states_actions = np.concatenate([states, actions], axis=1)

        y_j = rewards if done else rewards + gamma * np.max(Q_1_target.predict(states_actions), axis=1)

        # Perform a gradient descent step on (y_j - Q(s_j, a_j; h))^2 with respect to the network parameters h
        Q_1.model.fit(states_actions, y_j, epochs=1, batch_size=len(states), verbose=0, alpha=alpha)

        # Every C steps reset Q' = Q
        if t % C == 0:
            Q_1_target.model.set_weights(Q_1.model.get_weights())

def get_action(columns, epsilon, Q, state):
    if np.random.rand() < epsilon:
        return np.random.randint(columns)
    else:
        return np.argmax(Q.predict(state))

if __name__ == "main":
    rows = 4
    columns = 5
    N = 100_000

    Q_1, Q_2 = train_agents(rows, columns, N)

    # safe Q_1 and Q_2
