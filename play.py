import numpy as np
import tensorflow as tf
import sys
from environment import Environment as Env

# Load Model with Keras
Q_1 = tf.keras.models.load_model('model/Q_1_300')
Q_2 = tf.keras.models.load_model('model/Q_2_300')
env = Env(num_columns=5, num_rows=4, num_win=3)

def get_action(Q, state):
    state = np.array([state])
    return np.argmax(Q.predict(state, verbose=0))

def play(Q_1, Q_2, env):
    #sys.stdout.write("\r")  # Carriage return
    #sys.stdout.write("\033[K")  # Clear to the end of line
    state = env.reset()
    env.render()
    while True:
        action_1 = get_action(Q_1, state)
        state, reward, done = env.single_step(action_1, 1)
        #sys.stdout.write("\r")  # Carriage return
        #sys.stdout.write("\033[K")  # Clear to the end of line
        env.render()
        print("Reward: {}".format(reward))
        print("Action: {}".format(action_1))
        input()
        if done:
            break
        action_2 = get_action(Q_2, state)
        state, reward, done = env.single_step(action_2, -1)
        #sys.stdout.write("\r")  # Carriage return
        #.stdout.write("\033[K")  # Clear to the end of line
        env.render()
        print("Reward: {}".format(reward))
        print("Action: {}".format(action_2))
        input()
        if done:
            break

if __name__ == "__main__":
    play(Q_1, Q_2, env)