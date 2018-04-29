# -*- coding: utf-8 -*-

"""
@authors: Viniya CHANEMOUGAM and
          Ahmed Tidiane BALDE
"""

import numpy as np


def loadEpisodes(fichier):
    """ Loads episodes in arrays

    :param fichier: file to load

    """

    with open(fichier, 'r') as f:
        episodes = []
        for episode in f.readlines():
            # Remove the last ';' and split':'
            episode = np.array([p.split(':')
                                for p in episode[:-2].split(';')], float)
            episode = np.array(episode, int)
            # Sort the array in order of the infection time
            episode = episode[episode[:, 1].argsort()]
            episodes.append(episode)

    return episodes
