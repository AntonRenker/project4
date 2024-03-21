from replay import ReplayMemory
from actionValueFunction import ActionValueFunction
import numpy as np
from environment import Environment as Env
import time
import csv

def train_agents(env, rows=4, columns=5, N=100_000, num_episodes=100_000, gamma=1, alpha=0.001, epsilon=1, epsilon_min=0.01, epsilon_decay=0.9, batch_size=32, C=10):
    # Initialize replay memory D for both agents to capacity N
    D_1 = ReplayMemory(N)
    D_2 = ReplayMemory(N)

    # Initialize action-value function Q with random weights h for both agents
    Q_1 = ActionValueFunction(input_size=rows*columns, output_size=columns, alpha=alpha)
    Q_2 = ActionValueFunction(input_size=rows*columns, output_size=columns, alpha=alpha)

    # Initialize target action-value function Q' with weights h' = h for both agents
    Q_1_target = ActionValueFunction(input_size=rows*columns, output_size=columns, alpha=alpha)
    Q_1_target.model.set_weights(Q_1.model.get_weights())
    Q_2_target = ActionValueFunction(input_size=rows*columns, output_size=columns, alpha=alpha)
    Q_2_target.model.set_weights(Q_2.model.get_weights())

    initial_state_track_Q_1 = []
    initial_state_track_Q_2 = []

    for episode in range(num_episodes):
        print(f'Episode: {episode}')
        for _ in range(1):
            part_train_time(Q_1, Q_2, Q_1_target, D_1, env, gamma, alpha, epsilon, batch_size, C, columns, player=1)
            part_train_time(Q_2, Q_1, Q_2_target, D_2, env, gamma, alpha, epsilon, batch_size, C, columns, player=-1)
            epsilon = max(epsilon_min, epsilon * epsilon_decay)
            initial_state_track_Q_1.append(np.amax(Q_1.predict(np.array([np.zeros(columns * rows)]))[0]))
            initial_state_track_Q_2.append(np.amax(Q_2.predict(np.array([np.zeros(columns * rows)]))[0]))
        
        
        if episode % 100 == 0 and episode != 0:
            # Store the trained agents
            name_1 = 'model2/Q_1_' + str(episode)
            name_2 = 'model2/Q_2_' + str(episode)
            Q_1.model.save(name_1)
            Q_2.model.save(name_2)
            # Store the initial state track in csv
            name_3 = 'model2/track_Q_1_' + str(episode) + '.csv'
            name_4 = 'model2/track_Q_2_' + str(episode) + '.csv'
            np.savetxt(name_3, initial_state_track_Q_1, delimiter=',')
            np.savetxt(name_4, initial_state_track_Q_2, delimiter=',')

    return Q_1, Q_2, initial_state_track_Q_1, initial_state_track_Q_2
        

def part_train_time(Q_1, Q_2, Q_1_target, D, env, gamma, alpha, epsilon, batch_size, C, columns, player, csv_file='timing_data.csv'): 
    if player == 1:
        state = env.reset().copy()
    else:
        state = env.step(np.random.randint(columns), player, Q_2, epsilon)[0].copy()
    
    done = False
    t = 0

    while not done:
        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            # With probability epsilon select a random action a_t;
            start_time = time.time()
            action_1 = get_action(columns, epsilon, Q_1, state)
            end_time = time.time()
            writer.writerow(['Time to get action', end_time - start_time])

            # Execute action at and observe reward r_t and next state s_t+1
            start_time = time.time()
            next_state, reward, done = env.step(action_1, player, Q_2, epsilon) # TBD
            end_time = time.time()
            writer.writerow(['Time to step', end_time - start_time])

            # Store transition in memory D_1
            start_time = time.time()
            D.add(state, action_1, reward, next_state.copy(), done) # TBD
            end_time = time.time()
            writer.writerow(['Time to add to memory', end_time - start_time])

            # Sample random minibatch of transitions (s_j, a_j, r_j, s_j+1) from D_1
            start_time = time.time()
            states, actions, rewards, next_states, dones = D.sample(batch_size)
            end_time = time.time()
            writer.writerow(['Time to sample from memory', end_time - start_time])

            for state, action, reward, nexstate, done in zip(states, actions, rewards, next_states, dones):
                start_time = time.time()
                target = reward
                if not done:
                    target = reward + gamma * np.amax(Q_1_target.predict(np.array([nexstate]))[0])
                end_time = time.time()
                writer.writerow(['Time to calculate target', end_time - start_time])

                start_time = time.time()
                target_f = Q_1.predict(np.array([state]))
                end_time = time.time()
                writer.writerow(['Time to predict', end_time - start_time])

                start_time = time.time()
                target_f[0][action] = target
                Q_1.model.fit(np.array([state]), target_f, epochs=1, verbose=0)
                end_time = time.time()
                writer.writerow(['Time to fit', end_time - start_time])

                if t % C == 0:
                    start_time = time.time()
                    Q_1_target.model.set_weights(Q_1.model.get_weights())
                    end_time = time.time()
                    writer.writerow(['Time to set weights', end_time - start_time])

                t += 1
                state = next_state.copy()

def part_train(Q_1, Q_2, Q_1_target, D, env, gamma, alpha, epsilon, batch_size, C, columns, player): 
    if player == 1:
        state = env.reset().copy()
    else:
        state = env.step(np.random.randint(columns), player, Q_2, epsilon)[0].copy()

    done = False

    t = 0

    while not done:
        # With probability epsilon select a random action a_t;
        start_time = time.time()
        action_1 = get_action(columns, epsilon, Q_1, state)
        end_time = time.time()
        print(f'Time to get action: {end_time - start_time}')
              
        # Execute action at and observe reward r_t and next state s_t+1
        start_time = time.time()
        next_state, reward, done = env.step(action_1, player, Q_2, epsilon) # TBD
        end_time = time.time()
        print(f'Time to step: {end_time - start_time}')

        # Store transition in memory D_1
        start_time = time.time()
        D.add(state, action_1, reward, next_state.copy(), done) # TBD
        end_time = time.time()
        print(f'Time to add to memory: {end_time - start_time}')

        # Sample random minibatch of transitions (s_j, a_j, r_j, s_j+1) from D_1
        start_time = time.time()
        states, actions, rewards, next_states, dones = D.sample(batch_size)
        end_time = time.time()
        print(f'Time to sample from memory: {end_time - start_time}')

        for state, action, reward, nexstate, done in zip(states, actions, rewards, next_states, dones):
            start_time = time.time()
            target = reward
            if not done:
                target = reward + gamma * np.amax(Q_1_target.predict(np.array([nexstate]))[0])
            end_time = time.time()
            print(f'Time to calculate target: {end_time - start_time}')
            
            start_time = time.time()
            target_f = Q_1.predict(np.array([state]))
            end_time = time.time()
            print(f'Time to predict: {end_time - start_time}')

            start_time = time.time()
            target_f[0][action] = target
            Q_1.model.fit(np.array([state]), target_f, epochs=1, verbose=0)
            end_time = time.time()
            print(f'Time to fit: {end_time - start_time}')

            if t % C == 0:
                start_time = time.time()
                Q_1_target.model.set_weights(Q_1.model.get_weights())
                end_time = time.time()
                print(f'Time to set weights: {end_time - start_time}')

            t += 1
            state = next_state.copy()
        

def get_action(columns, epsilon, Q, state):
    rand_val = np.random.rand()
    if rand_val < epsilon:
        return np.random.randint(columns)
    else:
        state = np.array([state])
        return np.argmax(Q.predict(state))


rows = 4
columns = 5
num_wins = 3
env = Env(columns, rows , num_wins)

N = 100_000
num_episodes = 1_500
gamma = 1
alpha = 0.01
epsilon = 0.1
batch_size = 32
C = 5

Q_1, Q_2, track_Q_1, track_Q_2 = train_agents(env=env, rows=rows, columns=columns, N=N, num_episodes=num_episodes, gamma=gamma, alpha=alpha, epsilon=epsilon, batch_size=batch_size, C=C)

# Store the trained agents
Q_1.model.save('model2/Q_1')
Q_2.model.save('model2/Q_2')

# Store the initial state track in csv
np.savetxt('model2/track_Q_1.csv', track_Q_1, delimiter=',')
np.savetxt('model2/track_Q_2.csv', track_Q_2, delimiter=',')