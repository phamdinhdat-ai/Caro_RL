import os
import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from a2c import ActorCriticNet
class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        return np.array(self.states),\
                np.array(self.actions),\
                np.array(self.probs),\
                np.array(self.vals),\
                np.array(self.rewards),\
                np.array(self.dones),\
                batches

    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []


class Agent:
    def __init__(self, n_actions, input_dims, gamma=0.99, lr=0.0003, gae_lambda=0.95,
            policy_clip=0.2, batch_size=64, n_epochs=10):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.a2c = ActorCriticNet(in_dim=input_dims, out_dim=n_actions, lr= lr)
        self.memory = PPOMemory(batch_size)


    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def save_models(self):
        print('... saving models ...')
        self.a2c.save_checkpoint()
    def load_model(self):
        print('<<<<Loading model>>>>>')
        self.a2c.load_checkpoint()

    def choose_action(self, observation):
        
        # state = T.tensor([observation], dtype=T.float).to(self.actor.device)
        
        state = torch.tensor([observation],dtype=torch.float).view(1, 10, 10) .to(self.a2c.device)
        dist, value = self.a2c(state)
        dist = torch.distributions.Categorical(logits=dist)
        action = dist.sample()

        probs = torch.squeeze(dist.log_prob(action)).item()
        action = torch.squeeze(action).item()
        value = torch.squeeze(value).item()

        return action, probs, value
    
    def learn(self):
        loss = []
        for epoch in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr,\
            reward_arr, dones_arr, batches = \
                    self.memory.generate_batches()

            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            for t in range(len(reward_arr)-1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr)-1):
                    a_t += discount*(reward_arr[k] + self.gamma*values[k+1]*\
                            (1-int(dones_arr[k])) - values[k])
                    discount *= self.gamma*self.gae_lambda
                advantage[t] = a_t
            advantage = torch.tensor(advantage).to(self.a2c.device)
            loss_batch = []
            values = torch.tensor(values).to(self.a2c.device)
            for batch in batches:
                states = torch.tensor(state_arr[batch], dtype=torch.float).to(self.a2c.device)
                old_probs = torch.tensor(old_prob_arr[batch]).to(self.a2c.device)
                actions = torch.tensor(action_arr[batch]).to(self.a2c.device)
                states = states.view(states.size(0), 1, 10, 10) .to(self.a2c.device)
                dist, value  = self.a2c(states)
                dist = torch.distributions.categorical.Categorical(logits=dist)
                critic_value = torch.squeeze(value)

                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp()
                #prob_ratio = (new_probs - old_probs).exp()
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = torch.clamp(prob_ratio, 1-self.policy_clip,
                        1+self.policy_clip)*advantage[batch]
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()

                returns = advantage[batch] + values[batch]
                critic_loss = (returns-critic_value)**2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 0.5*critic_loss
                self.a2c.optimizer.zero_grad()
                total_loss.backward()
                self.a2c.optimizer.step()
                loss_batch.append(total_loss.item())
            loss.append(sum(loss_batch)/len(batches))
            # print(f"epochs:  {epoch} | Loss:  {sum(loss_batch)/len(batches)}")
        
        self.memory.clear_memory()               

        return sum(loss)/self.n_epochs
    

    def index_1D_to_2D(self, index):
        y = index % 10
        x = int(index / 10)
    
        return x,y

    def predict(self, state):
        action, _, _ = self.choose_action(state)
        x, y = self.index_1D_to_2D(action)
        return (x,y)