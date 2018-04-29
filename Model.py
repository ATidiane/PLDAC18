# -*- coding: utf-8 -*-

"""
@authors: Viniya CHANEMOUGAM and
          Ahmed Tidiane BALDE
"""


from collections import defaultdict

import numpy as np


class Model:
    def __init__(self, episodes):
        """ Model of algorithm

        :param episodes: array of episodes

        """

        # Array of episodes
        self.episodes = episodes
        # Nomber of users or distincts nodes
        self.nbUser = max([max(epi[:, 0]) for epi in self.episodes]) + 1
        # Dictionary of predecessors for each user
        self.predecessors = defaultdict(dict)
        # Dictionary of successors for each user
        self.successors = defaultdict(dict)

    def createGraph(self):
        """ Creates the graph (tree) of episodes """

        for episode in self.episodes:
            # List of list of successors for each user
            listeSuccessors = [episode[episode[:, 1] > episode[i, 1]][:, 0]
                               for i in range(len(episode))]
            # for the list of successors of each user
            for i, successeur in enumerate(listeSuccessors):
                # for every successor of a user
                for v in successeur:
                    # Generate a probability so within (0,1)
                    u, proba = episode[i, 0], np.random.random()
                    # u ---(proba)---> v
                    self.successors[u][v] = proba
                    # v ---(proba)---> u
                    self.predecessors[v][u] = proba
