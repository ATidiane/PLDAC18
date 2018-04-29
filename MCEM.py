# -*- coding: utf-8 -*-

"""
@authors: Viniya CHANEMOUGAM and
          Ahmed Tidiane BALDE
"""

import random
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from Model import Model
from utils import loadEpisodes

train_file, test_file = "cascades_train.txt", "cascades_test.txt"


class MCEM(Model):
    def __init__(self, episodes):

        Model.__init__(self, episodes)
        self.createGraph()
        print([episode[:, 1] for episode in self.episodes])
        self.nbTimes = max([max(epi[:, 1]) for epi in self.episodes]) + 1
        self.Ic = {t: [] for t in range(1, nbTimes)}
        self.Uc = {t: [] for t in range(1, nbTimes)}
        self.setsOfIcAndUc()

    def setsOfIcAndUc(self):
        """ This method fills the set of episodes D+ which satisfies
            both u € D(t) and v € D(>t) """

        for d, episode in enumerate(self.episodes):
            print(d, episode)
            for i in range(0, len(episode)):
                for j in range(0, len(episode)):
                    u, v = episode[i][0], episode[j][0]
                    tu, tv = episode[i][1], episode[j][1]
                    print(u, v, tu, tv)
                    if u != v and tv > tu:                                      # If it's not the same user and v is infected
                        # after u, then add the episode to the set
                        self.dPlus[u, v].append(d)

    def setOfUc(self):
        """ This method fills the set of episodes D+ which satisfies
            both u € D(t) and v € D(>t) """

        for d, episode in enumerate(self.episodes):
            for i in range(0, len(episode)):
                for j in range(0, len(episode)):
                    u, v = episode[i][0], episode[j][0]
                    tu, tv = episode[i][1], episode[j][1]
                    print(u, v, tu, tv)
                    if u != v and tv > tu:                                      # If it's not the same user and v is infected
                        # after u, then add the episode to the set
                        self.dPlus[u, v].append(d)

    def puv(self, theta, phi_uv):
        """ Probability that node v gets infected by u | node u is infected

        :param theta:
        :param phi_uv:

        """

        return 1 / (1 + np.exp(np.dot(phi_uv, theta.T)))

    def conditional_distrib_Xcut(self, Xc_dot_tminus1, theta):
        """ Conditional distribution of Xc_dot_tminus1 given all predecessors

        :param Xc_dot_tminus1:
        :param theta:

        """

        # Proba that node ut is never infected
        proba_ictm1 = 0
        infected = np.zeros(())


if __name__ == "__main__":
    mcem = MCEM(loadEpisodes(train_file))
