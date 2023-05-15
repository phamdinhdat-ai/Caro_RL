from helper import plot_as
from gomoku import Gomoku
# from agent import Agent
from a2c_agent import Agent
import numpy as np

if __name__ == '__main__':
    env = Gomoku(rows=10, cols=10, n_to_win=5)
    N = 20            # move taken before learning
    batch_size = 5
    n_epochs = 4
    alpha = 0.0003
    agent = Agent(n_actions=env.matrix, batch_size=batch_size, 
                    lr=alpha, n_epochs=n_epochs, 
                    input_dims=1)
    agent1 = Agent(n_actions=env.matrix, batch_size=batch_size, 
                    lr=alpha, n_epochs=n_epochs, 
                    input_dims=1)
    n_games = 300

    first_agent = 0
    learn_iters = 0
    
    n_steps = 0

    move_taken = 0
    losses = []
    rewards = []
    total_move_taken = 0
    
    for i in range(n_games):
        print("Game: ", i)
        env.reset()
        observation = env.get_state()
        done = False
        score = 0
        score1 = 0

        while not done or done1:
            #agent 0
            action, prob, val = agent.choose_action(observation)
            reward, done = env.take_action(action)
            observation_ = env.get_state()
            n_steps += 1
            score += reward
            agent.remember(observation, action, prob, val, reward, done)
            
            #agent 1
            action1, prob1, val1 = agent1.choose_action(observation_)
            reward1, done1 = env.take_action(action1)
            observation_1 = env.get_state()
            n_steps += 1
            score += reward1
            agent1.remember(observation_, action1, prob1, val1, reward1, done1)


            if n_steps % N == 0:
                loss_game = agent.learn()
                loss_game1 = agent1.learn()
                learn_iters += 1
            observation = observation_1

              
        print(f"Game: {i} | Loss Agent 0: {loss_game}")
        print(f"Game: {i} | Loss Agent 1: {loss_game1}")

        losses.append([loss_game,loss_game1])
        rewards.append([score,score1])
        # plot_as(plot_move_taken)
        
        