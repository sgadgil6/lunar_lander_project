import numpy as np
import keras
from keras.activations import relu, linear
import lunar_lander as lander
from collections import deque
import gym
import random
from keras.utils import to_categorical


learning_rate = 0.001
model = keras.Sequential()
model.add(keras.layers.Dense(128, input_dim=8, activation=relu))
model.add(keras.layers.Dense(128, activation=relu))
model.add(keras.layers.Dense(4, activation=linear))
model.compile(loss="mse", optimizer=keras.optimizers.adam(lr=learning_rate ))

epsilon = 1
gamma = .99
batch_size = 64
memory = deque(maxlen=1000000)
min_eps = 0.01
model = model

def replay_experiences():
    if len(memory) >= batch_size:
        sample_choices = np.array(memory)
        mini_batch_index = np.random.choice(len(sample_choices), batch_size)
        #batch = random.sample(memory, batch_size)
        states = []
        actions = []
        next_states = []
        rewards = []
        finishes = []
        for index in mini_batch_index:
            states.append(memory[index][0])
            actions.append(memory[index][1])
            next_states.append(memory[index][2])
            rewards.append(memory[index][3])
            finishes.append(memory[index][4])
        states = np.array(states)
        actions = np.array(actions)
        next_states = np.array(next_states)
        rewards = np.array(rewards)
        finishes = np.array(finishes)
        states = np.squeeze(states)
        next_states = np.squeeze(next_states)
        q_vals_next_state = model.predict_on_batch(next_states)
        q_vals_target = model.predict_on_batch(states)
        max_q_values_next_state = np.amax(q_vals_next_state, axis=1)
        q_vals_target[np.arange(batch_size), actions] = rewards + gamma * (max_q_values_next_state) * (1 - finishes)
        model.fit(states, q_vals_target, verbose=0)
        global epsilon
        if epsilon > min_eps:
            epsilon *= 0.996

def get_obs(true_loc):
    tX = true_loc[0]
    tY = true_loc[1]

    nX = np.random.normal(loc=tX, scale=0.1)

    return (nX, tY)

if __name__ == '__main__':
    env = lander.LunarLander()
    # env.seed(0)
    num_episodes = 400
    np.random.seed(0)
    scores  = []
    for i in range(num_episodes+1):
        score = 0
        state = env.reset()
        finished = False
        if i != 0 and i % 50 == 0:
            model.save(".\saved_models\model_"+str(i)+"_episodes_128_noisy_obs.h5")
        iterations = 0
        while True:
            state = np.reshape(state, (1, 8))
            obs = get_obs((state[0][0], state[0][1]))
            state[0][0] = obs[0]
            state[0][1] = obs[1]
            if np.random.random() <= epsilon:
                action =  np.random.choice(4)
            else:
                action_values = model.predict(state)
                action = np.argmax(action_values[0])
            #env.render()
            next_state, reward, finished, metadata = env.step(action)
            next_state = np.reshape(next_state, (1, 8))
            memory.append((state, action, next_state, reward, finished))
            replay_experiences()
            score += reward
            state = next_state
            if finished or iterations > 2000:
                scores.append(score)
                break

            iterations += 1
        print("Episode = {}, Score = {}, Avg_Score = {}".format(i, score, np.mean(scores[-10:])))

