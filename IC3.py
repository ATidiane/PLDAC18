# -*- coding: utf-8 -*-

"""
@authors: Viniya CHANEMOUGAM and
          Ahmed Tidiane BALDE
"""

""" In this file, we try to implement the IC model
"""

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
                    p[d][u] = 1 - ptd                                           # Proba que ça soit l'un deux.

        return p
                    
        
    
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

        self.arrayStep = list()
        
        for i in range(0, self.nbIteration):
            p = self.ptD()
            for u in range(0, self.nbUser):
                for v in self.successors[u]:                                    # self.successors[u] is equivalent to self.nbUser
                    sumThetaPtd = 0
                    sumDSets = len(self.dPlus[u,v]) + self.dMoins[u][v]         # D+(u,v) + D-(u,v)
                    for d in self.dPlus[u,v]:
                        sumThetaPtd += self.successors[u][v]/p[d][v]
                    self.successors[u][v] = sumThetaPtd/sumDSets
                    self.predecessors[v][u] = sumThetaPtd/sumDSets
            
            self.arrayStep.append(self.successors[0].copy())
        
        self.arrayStep = np.asarray(self.arrayStep)            
                    
    
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
                    (random.random() < self.successors[u][v])):                 # single chance (randomly) to infect him, then
                        infected[v] = True                                      # infect him.
                        S[t].append(v)
            t = t + 1
        return S, infected
        
        
    def predict(self, data, nbIteration=5000):
        """ Applique l'inférence pendant nbIteration fois et considère le 
            nombre de fois ou chaque utilisateur est infecté
        """
            
        sumInfected = defaultdict(float)
        for i in range(nbIteration):
            s, infected = self.inference(data)
            for user in infected:
                sumInfected[user] += 1
                
        for user in sumInfected:
            sumInfected /= nbIteration

        return sumInfected
        
        
    def score2(self, data, nbIteration=5000):
        """ Calcule de la mésure de précision moyenne MAP """
        
        ic = IC(loadEpisodes(data))
        sumD, D = 0, len(ic.episodes)
        
        for episode in ic.episodes:
            users, tempsU = episode[:,0], np.unique(episode[:,1])
            sourcesS0 = users[[episode[:,1] == tempsU[0]]]
            prediction = self.predict(sourcesS0, nbIteration)
            rang = np.array(pred.keys())[(-np.array(pred.values())).argsort()]
            print(rang)
            break
            
            
        
    def score(self, data, nbIteration=5000):
        """ Calcule la mésure de précision moyenne MAP """
        
        ic = IC(loadEpisodes(data))
        D = len(ic.episodes)
        
        # UD pris au hasard, pas bien compris.        
        UD = [np.random.randint(1,100) for _ in range(100)]

        sumD = 0
        for d in range(5000):
            sumI = 0
            OneOnD = 1/len(ic.episodes[d])
            for i in range(1, len(UD)):
                setOfUD = set(UD[1:i])
                setOfD = set(ic.episodes[d][:,0])
                numerateur  = len(setOfUD & setOfD)
                sumI += OneOnD * (numerateur / i)
            
            sumD += sumI
            
        return sumD / D
    
    def test(self):
        pass

# On remarque que plus on augmente le nombre d'itérations, plus les valeurs de 
# theta diminue                
ic = IC(loadEpisodes("cascades_train.txt"), 10)
#print(ic.predict("cascades_test.txt"))

# Apprentissage
ic.fit()
ic.score2("cascades_test.txt")

#theta = ic.successors[1]
#print(theta)

#print(ic.arrayStep)
