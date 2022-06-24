import itertools
import numpy as np
from numpy import dot
from numpy.linalg import norm
import networkx as nx

class Embedding():
    def __init__(self, embeddings_array, dataset_nx, embedding_labels, pER, test_loss_list=None, config=None, continous_p=False):
        self.embeddings_array = embeddings_array
        self.dataset_nx = dataset_nx
        self.coppie = None
        self.embedding_labels = embedding_labels
        self.coppie_labels = None
        self.cos_distances = []
        self.distances = []
        self.probabilities_ER = pER
        
        self.cos_intra_dists = None
        self.cos_inter_dists = []
        self.intra_dists = None
        self.inter_dists = []
        self.distance_of_means = None
        #self.interP1 = None
        #self.interP2 = None
        self.test_loss_list = test_loss_list
        self.config = config
        self.continous_p = continous_p
        
    def cosdist(self, a,b):
        return dot(a, b)/(norm(a)*norm(b))

    def calc_coppie(self):
        NN = self.embeddings_array.shape[0]
        coppie_numeric = list(itertools.combinations(range(NN), 2))
        assert len(coppie_numeric) == (NN * (NN-1))/2
        #print(f"{len(coppie_numeric)} possibili coppie")
        #self.coppie = np.array([self.embeddings_array[c,:] for c in coppie_numeric])
        self.coppie = self.embeddings_array[coppie_numeric] # è equivalente alla riga precedente: numpy integer mask, prende la shape della mask
        self.coppie_labels = [(self.embedding_labels[c[0]], self.embedding_labels[c[1]]) for c in coppie_numeric]

    def calc_distances(self):
        if self.config['model']['neurons_per_layer'][-1] > 1:
            self.calc_coppie()
            i = 0
            for a,b in self.coppie:
                cos_dist = self.cosdist(a,b) 
                dist = np.linalg.norm(a-b)
                label_a = self.coppie_labels[i][0]
                label_b = self.coppie_labels[i][1]
                self.cos_distances.append((cos_dist, (label_a, label_b)))
                self.distances.append((dist, (label_a, label_b)))
                i += 1

            #self.cos_intra_dists = [d[0] for d in self.cos_distances if d[1] == (label_1,label_1) or d[1] == (label_2,label_2)]
            self.cos_intra_dists = [d[0] for d in self.cos_distances if (d[1][0] == d[1][1]).all()]
            #self.cos_inter_dists = [d[0] for d in self.cos_distances if d[1] == (label_2,label_1) or d[1] == (label_1,label_2)]
            #guarda tutti i possibili accoppiamenti
            NN = len(set(tuple(e) for e in self.embedding_labels))
            #possibili_coppie_labels = set(list(itertools.combinations(range(NN), 2)))
            n_classi = len(self.config['graph_dataset']['list_p'])
            onehot_matrix = np.eye(n_classi)  ##   NN == n_classi?!
            possibili_coppie_labels = list(itertools.combinations(np.eye(4), 2))

            for label_1, label_2 in possibili_coppie_labels:
                #print(label_1, label_2)
                #print(self.cos_distances[0][1])
                r = [d[0] for d in self.cos_distances if d[1] == (label_2, label_1) or d[1] == (label_1, label_2)]
                #print(r)
                self.cos_inter_dists.append(r)

            self.intra_dists = [d[0] for d in self.distances if (d[1][0] == d[1][1]).all()]
            for label_1, label_2 in possibili_coppie_labels:
                self.inter_dists.append( [d[0] for d in self.distances if d[1] == (label_2,label_1) or d[1] == (label_1,label_2)] )

            # calculate average of each euclidean distribution
            mean_intra = np.mean(self.intra_dists)
            mean_inter = np.mean(self.inter_dists)
            self.distance_of_means = mean_inter - mean_intra
            
        else:
            print("Non serve calcolare le distanze nel caso di embedding scalare")

    def intorno(p_teorica, p_attuali, soglia):
        mask = []
        for p in p_attuali:
            if p > p_teorica - soglia and p < p_teorica + soglia:
                mask.append(True)
            else:
                mask.append(False)
        return mask

    def calc_correlation(self):
        # solo nel caso ddella regressione, no classificazione
        num_nodes = self.config['graph_dataset']['Num_nodes']
        if self.continous_p:
            correlazioni = np.corrcoef(self.embeddings_array.flatten(), self.embedding_labels)[0,1]

        else:
            actual_p = np.array([nx.to_numpy_matrix(t).sum(axis=1).mean() / (num_nodes - 1) for t in self.dataset_nx])
            correlazioni = []
            for p in self.probabilities_ER:
                mask_int = np.argwhere(self.embedding_labels == p).flatten()
                #plt.scatter(self.embeddings_array[mask_int].flatten(),actual_p[mask_int])
                # correlazione tra target e prediction
                correlaz = np.corrcoef(self.embeddings_array[mask_int].flatten(), actual_p[mask_int])[0, 1]
                correlazioni.append(correlaz)

        error = np.sqrt(np.sum(
            (self.embeddings_array.flatten() - self.embedding_labels) ** 2)) / len(self.embeddings_array)

        return correlazioni, error
