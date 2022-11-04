import itertools
import numpy as np
from numpy import dot
from numpy.linalg import norm
import networkx as nx

from config_valid import Config, TrainingMode

class Embedding():
    def __init__(self, embeddings_array, dataset, test_loss_list=None, config_c=None):
        self.embeddings_array = embeddings_array
        self.dataset = dataset
        self.dataset_nx = dataset.dataset_list
        self.coppie = None
        self.embedding_labels = dataset.labels
        self.original_class = dataset.original_class
        self.node_emb_pergraph = []
        self.coppie_labels = None
        self.cos_distances = []
        self.distances = []
        self.max_degree = []
        self.calc_max_degree()
        
        self.cos_intra_dists = None
        self.cos_inter_dists = []
        self.intra_dists = None
        self.inter_dists = []
        self.difference_of_means = None
        #self.interP1 = None
        #self.interP2 = None
        self.test_loss_list = test_loss_list
        self.config_class = config_c
        self.probabilities_ER = self.config_class.conf['graph_dataset']['list_p']
        self.continuous_p = self.config_class.conf['graph_dataset']['continuous_p']

    def calc_max_degree(self):
        if self.original_class is not None:
            for s in self.original_class:
                # la original class quì sarebbe la distribuzione del grado
                if isinstance(s, list):
                    self.max_degree.append(max(s))
                else:
                    self.max_degree.append(s)
        
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
        if self.original_class: # TODO: correggere perché ora original class contiene anche il modo ID
            self.coppie_orig_class = [(self.original_class[c[0]], self.original_class[c[1]]) for c in coppie_numeric]
            assert len(self.coppie_orig_class) == (NN * (NN-1))/2

    def calc_distances(self):
        self.calc_coppie()
        i = 0
        for a,b in self.coppie:
            cos_dist = self.cosdist(a,b)
            dist = np.linalg.norm(a-b)
            label_a = self.coppie_labels[i][0]
            label_b = self.coppie_labels[i][1]
            self.cos_distances.append((cos_dist, (label_a, label_b)))
            if self.original_class: # TODO: correggere perché ora original class contiene anche il modo ID
                orig_class_a = self.coppie_orig_class[i][0]
                orig_class_b = self.coppie_orig_class[i][1]
                self.distances.append((dist, (label_a, label_b), (orig_class_a, orig_class_b)))
            else:
                self.distances.append((dist, (label_a, label_b)))
            i += 1

        ### CALCOLO LA DISTANZA INTRA E INTER CLUSTER -> DEVO RITROVARE I CLUSTER IN BASE ALLE ETICHETTE

        ### CASO REGRESSION
        if self.config_class.get_mode() == TrainingMode.mode3 and not self.continuous_p:
            # non ha molto senso separare in classi quando la il target è la p stessa
            # però nel caso di distrib discreta lo voglio comunque fare, usando la original_class

            ## DISTANZE EUCLIDEE
            self.intra_dists = [d[0] for d in self.distances if (d[2][0] == d[2][1])]
            #possibili_coppie_labels = list(set(self.coppie_orig_class))
            #for orig_class_a, orig_class_b in possibili_coppie_labels:
            #    self.inter_dists.append(
            #        [d[0] for d in self.distances if d[2] == (orig_class_b, orig_class_a) or d[2] == (orig_class_a, orig_class_b)]
            #    )
            self.inter_dists = [d[0] for d in self.distances if (d[2][0] != d[2][1])]
            NN = self.embeddings_array.shape[0]
            assert len(self.intra_dists) == (NN/2 * (NN/2 - 1))
            assert len(self.inter_dists) == (NN/2)**2
        else:
            ### ORA NEL CASO DELLA CLASSIFICATION

            ## DISTANZE COSENO
            #self.cos_intra_dists = [d[0] for d in self.cos_distances if d[1] == (label_1,label_1) or d[1] == (label_2,label_2)]
            self.cos_intra_dists = [d[0] for d in self.cos_distances if (d[1][0] == d[1][1]).all()]
            #self.cos_inter_dists = [d[0] for d in self.cos_distances if d[1] == (label_2,label_1) or d[1] == (label_1,label_2)]

            #guarda tutti i possibili accoppiamenti
            #NN = len(set(tuple(e) for e in self.embedding_labels))
            #possibili_coppie_labels = set(list(itertools.combinations(range(NN), 2)))
            #n_classi = len(self.config['graph_dataset']['list_p'])
            #onehot_matrix = np.eye(n_classi)  ##   NN == n_classi?!

            # queste sono tutte le possibili sequenze di coppienumeric con 2classi!!!
            #possibili_coppie_labels = list(itertools.combinations(np.eye(2), 2))

            # for label_1, label_2 in possibili_coppie_labels:
            #     #print(label_1, label_2)
            #     #print(self.cos_distances[0][1])
            #     r = [d[0] for d in self.cos_distances if d[1] == (label_2, label_1) or d[1] == (label_1, label_2)]
            #     #print(r)
            #     self.cos_inter_dists.append(r)
            ### sembrava troppo difficile fare così?
            self.cos_inter_dists = [d[0] for d in self.cos_distances if (d[1][0] != d[1][1]).all()]
            # non ricordo perché facevo nel modo precedente

            ## DISTANZE EUCLIDEE
            self.intra_dists = [d[0] for d in self.distances if (d[1][0] == d[1][1]).all()]
            self.inter_dists = [d[0] for d in self.distances if (d[1][0] != d[1][1]).all()]
            #for label_1, label_2 in possibili_coppie_labels:
            #    self.inter_dists.append( [d[0] for d in self.distances if d[1] == (label_2,label_1) or d[1] == (label_1,label_2)] )

        # calculate average of each euclidean distribution
        mean_intra = np.mean(self.intra_dists)
        mean_inter = np.mean(self.inter_dists)
        self.difference_of_means = mean_inter - mean_intra

    def intorno(self, p_teorica, p_attuali, soglia):
        mask = []
        for p in p_attuali:
            if p > p_teorica - soglia and p < p_teorica + soglia:
                mask.append(True)
            else:
                mask.append(False)
        return mask

    def calc_graph_emb_correlation(self):
        # solo nel caso della dimensione di embedding = 1
        num_nodes = self.config_class.conf['graph_dataset']['Num_nodes']

        if self.config_class.conf['graph_dataset']['confmodel']:  # self.original_class ha un'array (degree sequence) per ciascun grafo
            #self.original_class # TODO: correggere perché ora original class contiene anche il modo ID
            correlazioni = np.corrcoef(self.embeddings_array.flatten(), self.original_class)[0, 1]
            embs = None
        elif self.continuous_p:
            correlazioni = np.corrcoef(self.embeddings_array.flatten(), self.embedding_labels)[0,1]
            embs = None
        else:
            actual_p = np.array([nx.to_numpy_matrix(t).sum(axis=1).mean() / (num_nodes - 1) for t in self.dataset_nx])
            correlazioni = []
            embs = []
            for p in self.probabilities_ER:
                #mask_int = np.argwhere(self.embedding_labels == p).flatten()
                mask_int = np.argwhere(self.intorno(p, self.embedding_labels, 0.05)).flatten()
                #plt.scatter(self.embeddings_array[mask_int].flatten(),actual_p[mask_int])
                emb = self.embeddings_array[mask_int].flatten()
                embs.append(emb)
                # correlazione tra target e prediction
                correlaz = np.corrcoef(emb, actual_p[mask_int])[0, 1]
                correlazioni.append(correlaz)

        error = np.sqrt(np.sum((self.embeddings_array.flatten() - self.embedding_labels) ** 2)) / len(self.embeddings_array)

        return correlazioni, error, embs

    def get_metrics(self, num_emb_neurons):
        if num_emb_neurons == 1:
            correlazioni, error, embeddings_per_cluster = self.calc_graph_emb_correlation()
            return correlazioni, error, embeddings_per_cluster
        else:
            self.calc_distances()  # calcola self.difference_of_means
            return self.difference_of_means

class Embedding_per_graph():
    def __init__(self, graph_embeddings_array, node_embedding_array, node_embeddings_array_id, graph_label, node_label_and_id):
        self.graph_embeddings_array = graph_embeddings_array
        self.node_embedding_array = node_embedding_array
        self.node_embeddings_array_id = node_embeddings_array_id
        self.graph_label = graph_label
        self.node_label_and_id = node_label_and_id

        self.node_label = [n[1] for n in list(node_label_and_id)]  # contiene anche i node ID

        self.align_embedding_id_with_degree_sequence_id()

    def align_embedding_id_with_degree_sequence_id(self):
        #print(self.node_embeddings_array_id)
        #print(self.node_label)
        pass

class NodeEmbedding(Embedding):
    def __init__(self, embeddings_array, embeddings_array_nodeid, dataset, test_loss_list=None, config_c=None):
        super().__init__(embeddings_array, dataset, test_loss_list, config_c)
        self.embeddings_array_nodeid = embeddings_array_nodeid
        self.node_emb_pergraphclass = []

        #if self.original_class is not None:
        #    self.get_emb_per_graph_class_cm()
        #    self.calc_max_degree()

    def get_emb_per_graph_class_cm(self, graph_embedding=None):
        """
        Suddivide i node embeddings per ciascun grafo, in self.node_emb_pergraph
        -> riprendo gli embeddings suddividendo i nodi come sono le lunghezze di original class (in questo caso la sequenza di grado, che quindi è lunga come il num di nodi)
        poiché gli embedding presi come output della rete vengono dal dataloader che suddivide in batch size, non in Num_grafi_per_tipo,
        né tantomeno in quanti nodi sono rimasti dopo il pruning dei nodi sconnessi, come cioè mi serve ora
        """

        r = 0
        for i, s in enumerate(self.original_class):  # tanti elementi quanti sono i grafi
            exp = self.dataset.labels[i]
            l = len(s)  # s ha tanti elementi quanti sono i nodi
            g_emb = None
            if graph_embedding is not None:
                g_emb = graph_embedding[i]
            # poiché nel training prepare ho shufflato coerentemente sia dataset_pyg che original_class che labels,
            # anche embeddings_array che arriva dal dataloader con shuffle=false ha il loro stesso ordine
            toappend = Embedding_per_graph(g_emb, self.embeddings_array[r:r + l], self.embeddings_array_nodeid[r:r + l], exp, s)
            self.node_emb_pergraphclass.append(toappend)
            r += l
            #print(f"l: {l}, r: {r}, toappend: {len(toappend)}, differenza: {(r + l)-r}")
        return self.node_emb_pergraphclass

    def get_emb_pergraph_cost(self):
        self.node_emb_pergraph = []
        Num_grafi_per_tipo = self.config_class.conf['graph_dataset']['Num_grafi_per_tipo']  # siamo sicuri che sia questo e non Num_nodes?
        tot = len(self.dataset_nx)
        j = 0
        for i in range(tot):
            self.node_emb_pergraph.append(self.embeddings_array[j:j + Num_grafi_per_tipo])
            j += Num_grafi_per_tipo
        #return self.node_emb_pergraph

    def get_average_corr_nodeemb(self):
        #print(self.node_emb_pergraph[0:5][0:5][0])
        correlations = []
        #print(self.node_emb_pergraph[0].flatten()[0:10])
        #print(self.original_class[0][0:10])
        #print(np.corrcoef(self.node_emb_pergraph[0].flatten()[0:10], self.original_class[0][0:10])[0, 1])
        for i, emb in enumerate(self.node_emb_pergraph):
            #print(f"emb len: {len(emb)}")
            #print(f"015: {emb_i[0:3]}")
            class_i = np.array(self.original_class[i]) # TODO: correggere perché ora original class contiene anche il modo ID
            #print(f"len class_i: {len(class_i)}")
            corr = np.corrcoef(emb, class_i)[0, 1]
            correlations.append(corr)
        #print(f"corr: {correlations}")
        return sum(correlations) / len(correlations)