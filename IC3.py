# -*- coding: utf-8 -*-

"""
@authors: Viniya CHANEMOUGAM and
          Ahmed Tidiane BALDE
"""

""" In this file, we try to implement the IC model
"""

import random
import numpy as np
from collections import defaultdict


def loadEpisodes(fichier):
    """ Loads episodes in arrays """
    
    # Load the file
    with open(fichier, 'r') as f:
        episodes = []
        for episode in f.readlines():                                           # Read lines one by one
            episode = np.array([p.split(':')                                    # Remove the last ';' and split':'
                                for p in episode[:-2].split(';')], float)
            episode = np.array(episode, int)
            episode = episode[episode[:,1].argsort()]                           # Sort the array in order of the 
            episodes.append(episode)                                            # infection time 
        
    return episodes       
    
    
class IC():
    def __init__(self, episodes, nbIteration=1):
        """ Algorithme IC (Independent Cascade)
            Setting up the inference mechanism and the learning algorithm of
            infections probabilities.
        """
        
        self.episodes = episodes                                                # Array of episodes
        self.nbIteration = nbIteration                                          # Nb Iterations for reaching convergence
        self.nbUser = max([max(epi[:,0]) for epi in self.episodes]) + 1         # Nomber of users or distincts nodes
        self.predecessors = defaultdict(dict)                                   # Dictionary of predecessors for each user
        self.successors = defaultdict(dict)                                     # Dictionary of successors for each user
        self.dMoins = np.zeros((self.nbUser, self.nbUser))                      # Set of episodes D-
        self.dPlus = {(i,j):[] for i in range(0,self.nbUser)                    # Set of episodes D+
                      for j in range(0, self.nbUser)}   
        
    
    def createGraph(self):
        """ Creates the graph (tree) of episodes"""
        
        for episode in self.episodes:
            listeSuccessors = [episode[episode[:,1] > episode[i,1]][:,0]        # List of list of successors for each user
                                for i in range(len(episode))]   
            for i, successeur in enumerate(listeSuccessors):                    # for the list of successors of each user
                for v in successeur:                                            # for every successor of a user
                    u, proba = episode[i,0], np.random.random()                 # Generate a probability so within (0,1)
                    self.successors[u][v] = proba                               # u ---(proba)---> v 
                    self.predecessors[v][u] = proba                             # v ---(proba)---> u

        
    def ptD(self):
        """ Estimates each success probability """
        
        p = dict()
        for d, episode in enumerate(self.episodes):
            users, tempsU = episode[:,0], np.unique(episode[:,1])               # List of users of an episode and distinct
            p[d] = np.ones(self.nbUser)                                         # time            
            # Nb: Si on voulait connaitre le temps de l'infection, on aurait 
            # parcouru les temps.            
            for u, user in enumerate(users):
                ptd, hasPred = 1., False
                predU = episode[episode[:,1] < episode[u,1]][:,0]               # List of predecessors of user u at time tU
                for v in predU:                                                 # Proba que ça ne soit aucun des predecesseurs 
                    if v in self.predecessors[user]:                            # qui l'infectent
                        ptd *= (1 - self.successors[v][user])
                        hasPred = True
                if hasPred:
<<<<<<< HEAD
                    p[d][user] = 1 - ptd                                           # Proba que ça soit l'un deux.
=======
                    p[d][user] = 1 - ptd                                        # Proba que ça soit l'un deux, probleme line.
>>>>>>> 71bf6cb3e23d92c20d5fc55e37ab021212b584de

        return p


    def phiD(self, episode, theta, ptd):

        user, sumPhi = episode[:,0], 0
        for i, u in enumerate(user):
            userV = user[episode[:,1] == episode[i,1]+1]
            for v in userV:
                if v in self.successors[u]:
                    div = theta[u][v] / ptd[v]
                    first = div * np.log(self.successors[u][v])
                    second = (1 - div) * np.log(1 - self.successors[u][v])
                    sumPhi += first + second

        return sumPhi
                
    
    def likelyhood(self, theta, ptd):
        
        vraissemblance = 0
        for d, episode in enumerate(self.episodes):
            users = episode[:,0]
            phi = self.phiD(episode, theta, ptd)
            sumlog = 0
            for i, u in enumerate(users):
                userV = users[episode[:,1] > episode[i,1]]
                for v in userV:
                    if v in self.successors[u]:
                        sumlog += np.log(1 - self.successors[u][v])
            
            vraissemblance += phi + sumlog

        return vraissemblance

    
    def setOfdPlus(self):
        """ This method fills the set of episodes D+ which satisfies
            both u € D(t) and v € D(>t) """
        
        for d, episode in enumerate(self.episodes):
            for i in range(0,len(episode)):
                for j in range(0,len(episode)):
                    u, v = episode[i][0], episode[j][0]
                    tu, tv = episode[i][1], episode[j][1]
                    if u != v and tv > tu:                                      # If it's not the same user and v is infected
                        self.dPlus[u,v].append(d)                               # after u, then add the episode to the set
                        
    
    def setOfdMoins(self):
        """ This method fills the set of episodes D- which satisfies
            both u € D(t) and v not € D(t) """
        
        for episode in self.episodes:
            for i in range(0,len(episode)):
                for j in range(0, self.nbUser):
                    if (j not in episode[:,0]):
                        self.dMoins[episode[i][0]][j] += 1
                        
    
    def fit(self):
        """ Estimates each diffusion probability """
        
        self.createGraph()
        self.setOfdPlus()
        self.setOfdMoins()
        self.vraissemblance = list()
        
        for i in range(0, self.nbIteration):
            p = self.ptD()

<<<<<<< HEAD
            vraissemblance = 0
            for d, episode in enumerate(self.episodes):
                vraissemblance += np.sum(np.log(list(p.values())))
                users = episode[:,0]
                for u, user in enumerate(users):
                    for v in range(self.nbUser):
                        if v not in users:
                            vraissemblance += np.log(1 - self.successors[user][v])

            print(vraissemblance)
            
=======
>>>>>>> 71bf6cb3e23d92c20d5fc55e37ab021212b584de
            for u in range(0, self.nbUser):
                for v in self.successors[u]:                                    # self.successors[u] is equivalent to self.nbUser
                    sumThetaPtd = 0
                    sumDSets = len(self.dPlus[u,v]) + self.dMoins[u][v]         # D+(u,v) + D-(u,v)
                    for d in self.dPlus[u,v]:
                        sumThetaPtd += self.successors[u][v]/p[d][v]

                    theta = self.successors.copy()
                    self.successors[u][v] = sumThetaPtd/sumDSets
                    self.predecessors[v][u] = sumThetaPtd/sumDSets
                    
        
            self.vraissemblance.append(self.likelyhood(theta, p))
            
    
    def inference(self, S0):
        """ Chaque utilisateur tente d'infecter ses successeurs avec une
            probalité theta(u,v).
            :param S0: ensemble de sources, c a d, infecté au temps t0            
        """
        
        S = []
        infected = defaultdict(bool)
        S.append(S0)                                                            # We add users infected at time 0
        t = 1
        while S[t-1] != []:                                                     # While there's an uninfected user
            S.append([])
            for u in S[t-1]:
                for v in self.successors[u]:
                    if ((not infected[v]) and                                   # If the user is not infected and we have a 
                    (np.random.random() < self.successors[u][v])):                 # single chance (randomly) to infect him, then
                        infected[v] = True                                      # infect him.
                        S[t].append(v)
            t = t + 1
        return S, infected
        
        
    def predict(self, data, nbIteration=1000):
        """ Applique l'inférence pendant nbIteration fois et considère le 
            nombre de fois ou chaque utilisateur est infecté
        """
            
        sumInfected = defaultdict(float)
        for i in range(nbIteration):
            s, infected = self.inference(data)
<<<<<<< HEAD
            for user in infected:
                sumInfected[user] += 1
                
        for user in sumInfected:
            sumInfected[user] /= nbIteration
=======
            for u in infected.keys():
                sumInfected[u] += infected[u]

        for u in sumInfected.keys():
            sumInfected[u] /= nbIteration
>>>>>>> 71bf6cb3e23d92c20d5fc55e37ab021212b584de

        return sumInfected
            
        
<<<<<<< HEAD
    def score(self, data, nbIteration=5000):
        """ Calcule la mésure de précision moyenne MAP """
        
        ic = IC(loadEpisodes(data))
        sumD, D = 0, len(ic.episodes)
        
=======
    
    def score(self, data, nbIteration=1000):
        """ Calcule la mésure de précision moyenne MAP """
        
        ic = IC(loadEpisodes(data))
        D = len(ic.episodes)
        sumD = 0
>>>>>>> 71bf6cb3e23d92c20d5fc55e37ab021212b584de
        for d, episode in enumerate(ic.episodes):
            users, tempsU = episode[:,0], np.unique(episode[:,1])
            sourcesS0 = users[[episode[:,1] == tempsU[0]]]
            prediction = self.predict(sourcesS0, nbIteration)
<<<<<<< HEAD
            UD = np.array(list(prediction.keys()))[(-np.array(list(prediction.values()))).argsort()]
            sumI, lenEpisode = 0, len(ic.episodes[d])
            
=======
            UD = np.array(list(prediction.keys()))[(np.array(list(prediction.values())).argsort())]

            sumI = 0
            lenEpisode = len(ic.episodes[d])
>>>>>>> 71bf6cb3e23d92c20d5fc55e37ab021212b584de
            for i in range(1, len(UD)):
                #if UD[i] est  infecte : 
                setOfUD = set(UD[1:i])
                setOfD = set(ic.episodes[d][:,0])
                numerateur  = len(setOfUD & setOfD)
                sumI += numerateur / i
            
            sumD += sumI / lenEpisode
            
        return sumD / D
    
    def test(self):
        pass

# On remarque que plus on augmente le nombre d'itérations,
# plus les valeurs de theta diminue

ic = IC(loadEpisodes("cascades_train.txt"), 10)

#======================= Apprentissage
ic.fit()
<<<<<<< HEAD
print(ic.successors[0])

#======================= Evaluation
the_map = ic.score("cascades_test.txt", 10)
print("score : ",the_map)
=======
theta = ic.successors[0]
print(theta)
print(ic.vraissemblance)


# Evaluation
the_map = ic.score("cascades_train.txt", 10) 
print(the_map)

>>>>>>> 71bf6cb3e23d92c20d5fc55e37ab021212b584de


#pour i infecte dans c:
#pour j non infecte dans c:
#sum+=log(1-theta(i,j))
            
