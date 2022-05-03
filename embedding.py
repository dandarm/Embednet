import itertools
import numpy as np
from numpy import dot
from numpy.linalg import norm

class Embedding():
    def __init__(self, embeddings_array, embedding_labels, pER, test_loss_list=None):
        self.embeddings_array = embeddings_array
        self.coppie = None
        self.embedding_labels = embedding_labels
        self.coppie_labels = None
        self.cos_distances = []
        self.distances = []
        self.probabilities_ER = pER
        
        self.cos_intra_dists = None
        self.cos_inter_dists = None
        self.distance_of_means = None
        #self.interP1 = None
        #self.interP2 = None
        self.test_loss_list = test_loss_list
        
    def cosdist(self, a,b):
        return dot(a, b)/(norm(a)*norm(b))

    def calc_coppie(self):
        NN = self.embeddings_array.shape[0]
        coppie_numeric = list(itertools.combinations(range(NN), 2))
        assert len(coppie_numeric) == (NN * (NN-1))/2
        #print(f"{len(coppie_numeric)} possibili coppie")
        self.coppie = np.array([self.embeddings_array[c,:] for c in coppie_numeric])
        self.coppie_labels = [(self.embedding_labels[c[0]], self.embedding_labels[c[1]]) for c in coppie_numeric]

    def calc_distances(self):
        i = 0
        for a,b in self.coppie:
            cos_dist = self.cosdist(a,b) 
            dist = np.linalg.norm(a-b)
            label_a = self.coppie_labels[i][0]
            label_b = self.coppie_labels[i][1]
            self.cos_distances.append((cos_dist,(label_a, label_b)))
            self.distances.append((dist,(label_a, label_b)))
            i += 1
            
        self.cos_intra_dists = [d[0] for d in self.cos_distances if d[1] == (0,0) or d[1] == (1,1)]
        self.cos_inter_dists = [d[0] for d in self.cos_distances if d[1] == (1,0) or d[1] == (0,1)]
        
        self.intra_dists = [d[0] for d in self.distances if d[1] == (0,0) or d[1] == (1,1)]
        self.inter_dists = [d[0] for d in self.distances if d[1] == (1,0) or d[1] == (0,1)]
        
        # calculate average of each euclidean distribution
        mean_intra = np.mean(self.intra_dists)
        mean_inter = np.mean(self.inter_dists)
        self.distance_of_means = mean_inter - mean_intra
        