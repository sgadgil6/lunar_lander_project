import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optimizer
import lunar_lander as lander
from collections import deque
import gym
import random


class DQN(nn.Module):
    def __init__(self, learning_rate):
        super(DQN, self).__init__()
        """
        if torch.cuda.is_available():
            self.dev = "cuda:0"
        else:
            self.dev = "cpu"
        """
        self.dev = "cpu"
        self.num_actions = 4
        self.layer1 = nn.Linear(8, 150)
        self.layer2 = nn.Linear(150, 120)
        self.output_layer = nn.Linear(120, self.num_actions)
        self.optimizer = optimizer.Adam(self.parameters(), lr=learning_rate)
        self.loss_func = nn.MSELoss()
        self.to(self.dev)


    def forward(self, input):
        input = torch.Tensor(input).to(self.dev)
        input = func.relu(self.layer1(input))
        input = func.relu(self.layer2(input))
        output = self.output_layer(input)
        return output

class DQNAgent:
    def __init__(self):
        self.epsilon = 1
        self.gamma = .99
        self.batch_size = 64
        self.learning_rate = 0.001
        self.memory = deque(maxlen=100000)
        self.min_eps = 0.01
        self.model = DQN(learning_rate=self.learning_rate)

    def store_experience(self, state, action, next_state, reward, finished):
        self.memory.append((state, action, next_state, reward, finished))

    def take_action(self, input):
        if np.random.random() <= self.epsilon:
            return np.random.choice(4)
        action_values = self.model.forward(input)
        return torch.argmax(action_values).item()

    def replay_experiences(self):
        if len(self.memory) >= self.batch_size:
            self.model.optimizer.zero_grad()
            sample_choices = np.array(self.memory)
            mini_batch_index =np.random.choice(len(sample_choices), self.batch_size)
            batch = random.sample(self.memory, self.batch_size)
            states = []
            actions =[]
            next_states = []
            rewards = []
            finishes = []
            for index in batch:
                states.append(index[0])
                actions.append(index[1])
                next_states.append(index[2])
                rewards.append(index[3])
                finishes.append(index[4])
            states = np.array(states)
            actions = np.array(actions)
            next_states = np.array(next_states)
            rewards = np.array(rewards)
            finishes = np.array(finishes)
            #states = np.squeeze(states)
            #next_states = np.squeeze(next_states)
            #convert rewards and finishes to tensors
            rewards = torch.Tensor(rewards).to(self.model.dev)
            finishes = torch.Tensor(finishes).to(self.model.dev)
            q_vals_curr_state = self.model.forward(states)
            q_vals_next_state = self.model.forward(next_states)
            q_vals_target = q_vals_curr_state.clone()
            #print(q_vals_next_state.shape)
            max_q_values_next_state = torch.max(q_vals_next_state, dim=1)[0]
            q_vals_target[np.arange(self.batch_size), actions] = rewards + self.gamma * (max_q_values_next_state) * (1-finishes)
            loss = self.model.loss_func(q_vals_curr_state, q_vals_target).to(self.model.dev)
            loss.backward()
            self.model.optimizer.step()
            if self.epsilon > self.min_eps:
                self.epsilon *= 0.996



if __name__ == '__main__':
    env = lander.LunarLander()
    #env.seed(0)
    agent = DQNAgent()
    num_episodes = 400
    np.random.seed(0)
    for i in range(num_episodes):
        score = 0
        state = env.reset()
        #state = np.reshape(state, (1, 8))
        finished = False
        for j in range(3000):
            action = agent.take_action(state)
            env.render()
            next_state, reward, finished, metadata = env.step(action)

            #next_state = np.reshape(next_state, (1, 8))
            agent.store_experience(state, action, next_state, reward, finished)
            score += reward
            state = next_state
            agent.replay_experiences()
            if finished:
                print("Episode = {}, Score = {}".format(i, score))
                break

