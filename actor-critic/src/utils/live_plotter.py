import plotext as plt
from typing import List, Dict
import time
from collections import deque

class LivePlotter:
    def __init__(self, window_size: int = 100):
        self.portfolio_values = deque(maxlen=window_size)
        self.rewards = deque(maxlen=window_size)
        self.episodes = deque(maxlen=window_size)
        self.current_episode = 0
        
    def update(self, portfolio_value: float, reward: float):
        self.portfolio_values.append(portfolio_value)
        self.rewards.append(reward)
        self.episodes.append(self.current_episode)
        
    def new_episode(self):
        self.current_episode += 1
        
    def plot(self):
        plt.clf()
        plt.subplot(2, 1, 1)
        plt.plot(list(self.episodes), list(self.portfolio_values))
        plt.title("Portfolio Value")
        plt.xlabel("Episode")
        plt.ylabel("Value")
        
        plt.subplot(2, 1, 2)
        plt.plot(list(self.episodes), list(self.rewards))
        plt.title("Reward")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        
        plt.show() 