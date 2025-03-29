import random
from typing import List
import numpy as np
import math

class DynamicWeights:
    def __init__(
            self,
            weights: List[float],
            smoothing_factor: float = 0.9,
    ):
        self.num_domains = len(weights)
        self.weights = weights
        self._estimated_reward = [0] * self.num_domains
        total_weights = np.sum(weights)
        self._probabilities = [weight / total_weights for weight in weights]
        self.eps = 1 / self.num_domains
        self.prev_eps = None
        self.smoothing_factor = smoothing_factor
        self.vars_to_log = ["_probabilities", "_estimated_reward"]

    def update(self, index: int, reward: float, iteration: int) -> List[float]:
        """
        Updates the weights based on the provided reward.
        """

        # update cumulative estimated reward
        self._estimated_reward[index] = self.smoothing_factor*self._estimated_reward[index] + (1-self.smoothing_factor)*math.exp(reward)

        # calculate epsilons
        self.prev_eps = self.eps
        self.eps = min(1/self.num_domains, math.sqrt(math.log(self.num_domains)/(self.num_domains*iteration)))

        # calculate scaling factor
        total_estimated_rewards = sum([math.exp(r*self.prev_eps) for r in self._estimated_reward])
        scaling_factor = (1-self.num_domains*self.eps)/total_estimated_rewards

        # update weights
        for i in range(self.num_domains):
            self.weights[i] = math.exp(self._estimated_reward[i]*self.prev_eps)*scaling_factor + self.eps

        # update probabilities
        total_weights = sum(self.weights)
        for i in range(self.num_domains):
            self._probabilities[i] = self.weights[i]/total_weights

        return self._probabilities

    def group_update(self, idx: List[int], rewards: List, iteration: int):
        # calculate epsilons
        self.prev_eps = self.eps
        self.eps = min(1/self.num_domains, math.sqrt(math.log(self.num_domains)/(self.num_domains*iteration)))

        # update cumulative estimated reward
        for index, reward in zip(idx, rewards):
            # smoothed mean
            # self._estimated_reward[name] = self.smoothing_factor*self._estimated_reward[name] + (1-self.smoothing_factor)*reward
            # smoothed exponentiated mean
            self._estimated_reward[index] = self.smoothing_factor*self._estimated_reward[index] + (1-self.smoothing_factor)*math.exp(reward)
        # print(f"Rank: {torch.distributed.get_rank()} -- estimated_reward {self._estimated_reward}")

        # calculate normalized scaling factor
        total_estimated_rewards = sum((r*self.prev_eps) for r in self._estimated_reward)
        scaling_factor = (1-self.num_domains*self.eps)/total_estimated_rewards

        # update weights
        for i in range(self.num_domains):
            # self.weights[self.dataset_map[name]] = math.exp(self._estimated_reward[name]*self.prev_eps)*scaling_factor + self.eps
            self.weights[i] = self._estimated_reward[i]*self.prev_eps*scaling_factor + self.eps

        # update probabilities
        total_weights = sum(self.weights)
        for i in range(self.num_domains):
            self._probabilities[i] = self.weights[i]/total_weights

        return self._probabilities