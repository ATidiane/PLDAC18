# -*- coding: utf-8 -*-

"""
@authors: Viniya CHANEMOUGAM and
          Ahmed Tidiane BALDE
"""

""" In this file, we implemented the Independant Cascade (IC)  model
"""

import random
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from Model import Model
from utils import loadEpisodes


class IC():
    def __init__(self, episodes, nbIteration=10, nbSimulation=10):
        """ Algorithme IC (Independent Cascade)
            Setting up the inference mechanism and the learning algorithm of
            infections probabilities.
        """

        Model.__init__(self, episodes)
        # Nb Iterations for reaching convergence
        self.nbIteration = nbIteration
        # Nb simulation of the inference
        self.nbSimulation = nbSimulation
        # Set of episodes D-
        self.dMoins = np.zeros((self.nbUser, self.nbUser))
        # Set of episodes D+
        self.dPlus = {(i, j): [] for i in range(0, self.nbUser)
                      for j in range(0, self.nbUser)}
        self.likelyHoods = np.zeros(nbIteration)

    def createGraph(self):
        """ Creates the graph (tree) of episodes"""

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

    def ptD(self):
        """ Estimates each success probability """

        p = dict()
        for d, episode in enumerate(self.episodes):
            # List of users of an episode and distinct time
            users, tempsU = episode[:, 0], np.unique(episode[:, 1])
            p[d] = np.ones(self.nbUser)
            # Nb: Si on voulait connaitre le temps de l'infection, on aurait
            # parcouru les temps.
            for u, user in enumerate(users):
                ptd, hasPred = 1., False
                # List of predecessors of user u at time tU
                predU = episode[episode[:, 1] < episode[u, 1]][:, 0]
                for v in predU:
                    # Proba que ça ne soit aucun des predecesseurs qui
                    # l'infectent
                    if v in self.predecessors[user]:
                        ptd *= (1 - self.successors[v][user])
                        hasPred = True
                if hasPred:
                    # Proba que ça soit l'un deux, probleme line.
                    p[d][user] = 1 - ptd

        return p

    def setOfdPlus(self):
        """ This method fills the set of episodes D+ which satisfies
            both u € D(t) and v € D(>t) """

        for d, episode in enumerate(self.episodes):
            for i in range(0, len(episode)):
                for j in range(0, len(episode)):
                    u, v = episode[i][0], episode[j][0]
                    tu, tv = episode[i][1], episode[j][1]
                    # If it's not the same user and v is infected after u,
                    # then add the episode to the set
                    if u != v and tv > tu:
                        self.dPlus[u, v].append(d)

    def setOfdMoins(self):
        """ This method fills the set of episodes D- which satisfies
            both u € D(t) and v not € D(t) """

        for episode in self.episodes:
            for i in range(0, len(episode)):
                for j in range(0, self.nbUser):
                    if (j not in episode[:, 0]):
                        self.dMoins[episode[i][0]][j] += 1

    def likelyhood(self, ptD):
        """ Calculate the likelyhood and returns it """

        vraissemblance = 0
        for d, episode in enumerate(self.episodes):
            # Proba that the infected users get infected
            vraissemblance += np.sum(np.log(ptD[d]))
            users = episode[:, 0]
            for u, user in enumerate(users):
                for v in range(self.nbUser):
                    if v not in users:
                        # Proba that the uninfected users stay uninfected
                        vraissemblance += np.log(1 - self.successors[user][v])

        return vraissemblance

    def fit(self):
        """ Estimates each diffusion probability """

        self.createGraph()
        self.setOfdPlus()
        self.setOfdMoins()

        print("\nHeure de vérité, Vraissemblance:")
        print("================================")

        for i in range(0, self.nbIteration):
            p = self.ptD()
            self.likelyHoods[i] = self.likelyhood(p)

            print("It°{} : ".format(i + 1), self.likelyHoods[i])

            for u in range(0, self.nbUser):
                # self.successors[u] is equivalent to self.nbUser in this case
                for v in self.successors[u]:
                    sumThetaPtd = 0
                    # D+(u,v) + D-(u,v)
                    sumDSets = len(self.dPlus[u, v]) + self.dMoins[u][v]
                    for d in self.dPlus[u, v]:
                        sumThetaPtd += self.successors[u][v] / p[d][v]

                    self.successors[u][v] = sumThetaPtd / sumDSets
                    self.predecessors[v][u] = sumThetaPtd / sumDSets

    def inference(self, S0):
        """ Chaque utilisateur tente d'infecter ses successeurs avec une
            probalité theta(u,v).

        :param S0: ensemble de sources, c a d, infecté au temps t0

        """

        S = []
        infected = defaultdict(bool)
        # We add users infected at time 0
        S.append(S0)
        t = 1
        # While there's an uninfected user
        while S[t - 1] != []:
            S.append([])
            for u in S[t - 1]:
                for v in self.successors[u]:
                    # If the user is not infected and we have a single chance
                    # (randomly) to infect him, then infect him
                    if ((not infected[v]) and
                            (random.random() < self.successors[u][v])):
                        infected[v] = True
                        S[t].append(v)
            t = t + 1
        return S, infected

    def predict(self, data):
        """ Applique l'inférence pendant nbIteration fois et considère la
            proportion d'infection de chaque utilisateur
        """

        sumInfected = defaultdict(float)
        for i in range(self.nbSimulation):
            s, infected = self.inference(data)
            for u in infected.keys():
                sumInfected[u] += infected[u]

        for u in sumInfected.keys():
            sumInfected[u] /= self.nbSimulation

        return sumInfected

    def score(self, data):
        """ Calcule la mésure de précision moyenne MAP """

        ic = IC(loadEpisodes(data))
        D = len(ic.episodes)
        sumD = 0
        for d, episode in enumerate(ic.episodes):
            users, tempsU = episode[:, 0], np.unique(episode[:, 1])
            # Infected user at (time 0) which is 1 in our data
            sourcesS0 = users[[episode[:, 1] == tempsU[0]]]
            # Predict the proportion of infection of each user
            prediction = self.predict(sourcesS0)
            # knowing SO, then, Sort those probabilities in descending order
            UD = np.array(list(prediction.keys()))[
                np.array(list(prediction.values())).argsort()[::-1]]
            sumI = 0
            lenEpisode = len(ic.episodes[d])
            for i in range(1, len(UD)):
                # first ième infected users in UD, which are UD[1:i]
                setOfUD = set(UD[1:i])
                # The real infected users in the corresponding episode
                setOfD = set(ic.episodes[d][:, 0])
                # The intersection of the two sets
                numerateur = len(setOfUD & setOfD)
                sumI += numerateur / i

            sumD += sumI / lenEpisode

        return sumD / D


def plotLikelyhood(ic):
    """ Plot the evoluation of x according to y """

    fig, ax = plt.subplots()

    plt.suptitle("Evolution of the likelyhood with IC model")
    ax.set_xlabel("nbIterations")
    ax.set_ylabel("likelyhoods")

    ax.plot(np.arange(ic.nbIteration), ic.likelyHoods,
            c='purple', marker='.', label="likelyhood")
    plt.legend()
    plt.show()


def main():
    """ We noticed that after each iteration, the values of theta decrease and
        MORE IMPORTANTLY, the likelyhood increases and converges after a certain
        number of iterations.
    """

    train, test = "cascades_train.txt", "cascades_test.txt"

    ic = IC(loadEpisodes(train), nbIteration=10, nbSimulation=10)

    #------------------------------ Apprentissage -----------------------------#
    ic.fit()
    theta = ic.successors[0]

    # print(theta)

    #------------------------------- Evaluation -------------------------------#
    scoreOn = test
    the_map = ic.score(test)

    print("\nScore MAP sur les données {} :\n".format(scoreOn), the_map)

    #---------------------- Plot the likelyhood evolution ---------------------#
    plotLikelyhood(ic)


if __name__ == "__main__":
    main()