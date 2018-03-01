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
        # ******A voir
        #self.theta = np.array([[ np.random.random()                             # Probability of the precedent step ô
        #                        for i in range(0, self.nbUser)]             
        #                       for j in range(0, self.nbUser)])
        
    
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
            
            # A voir ******
            
            for u, user in enumerate(users):
                ptd, hasPred = 1., False
                # Peut être parcourir seulement les users seraient plus interessants
                #for t in range(1, len(tempsU)):                                     # For each time tU of the episode D                
                predU = episode[episode[:,1] < episode[u,1]][:,0]               # List of predecessors of user u at time tU
                for v in predU:                                                 # Proba que ça ne soit aucun des predecesseurs 
                    if v in self.predecessors[user]:                            # qui l'infectent
                        ptd *= (1 - self.successors[v][user])
                        hasPred = True
                if hasPred:
                    p[d][u] = 1-ptd                                             # Proba que ça soit l'un deux.
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
                 # Aulieu de prendre tous les users pour la boucle ci-dessus
                 # prendre seulement les successeurs, serait plus efficace.
                 # Reviens à la même chose si trop d'épisodes
                for v in self.successors[u]:
                    sumThetaPtd = 0
                    sumDSets = len(self.dPlus[u, v]) + self.dMoins[u][v]
                    for d in self.dPlus[u,v]:
                        sumThetaPtd += self.successors[u][v]/p[d][v]
                    self.successors[u][v] = sumThetaPtd/sumDSets
                    self.predecessors[v][u] = sumThetaPtd/sumDSets
                    
             self.arrayStep.append(self.successors)
        
        self.arrayStep = np.asarray(self.arrayStep)            
                    
                    
    def predict(self, data):
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
            
        return sumD * (1 / D)
        
    
    def inference(self):
        St = set()
        
                    
                    
                
ic = IC(loadEpisodes("cascades_train.txt"), 5)
#ic.createGraph()
#print(ic.predict("cascades_test.txt"))
ic.fit()
print(ic.successors[0])
print(ic.successors[0][1])
#print(ic.arrayStep[0:2, 99])