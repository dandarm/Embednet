import itertools
import numpy as np
from numpy import dot
from numpy.linalg import norm
import networkx as nx

from config_valid import Config, TrainingMode
#from graph_generation import GraphType
from scipy.stats import kendalltau
import skdim

class Embedding():
    def __init__(self, graph_embedding_array, node_embedding_array, dataset, config_c=None, output_array=None, model_params=None):
        self.graph_embedding_array = graph_embedding_array
        self.node_embedding_array = node_embedding_array
        self.output_array = output_array
        self.model_params = model_params

        #dataset = dataset
        # tolgo il dataset dalla classe embedding
        self.training_labels = dataset.labels  # sono le label dettate dal training mode ad uso della training loss
        self.exponents = dataset.exponent
        self.original_node_class = dataset.original_node_class
        self.actual_node_class = dataset.actual_node_class
        self.scalar_class = dataset.scalar_label

        self.config_class = config_c

        self.emb_pergraph = []
        self.emb_perclass = []
        self.graph_correlation_per_class = None
        self.total_graph_correlation = None
        self.node_correlation_per_class = None
        self.total_node_correlation = None
        self.regression_error = None
        self.node_emb_dims = None
        self.graph_emb_dims = None
        self.total_node_emb_dim = None
        self.total_graph_emb_dim = None

        self.dataset_nx = dataset.dataset_list
        self.numgrafi = len(self.dataset_nx)
        self.coppie = None
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

        self.probabilities_ER = self.config_class.conf['graph_dataset']['list_p']
        self.continuous_p = self.config_class.conf['graph_dataset']['continuous_p']


# region handle embeddings
    def get_emb_per_graph(self):
        """
        Suddivide i node embeddings per ciascun grafo, e riempie in self.node_emb_pergraph
        -> riprendo gli embeddings suddividendo i nodi come sono le lunghezze di original class (in questo caso la sequenza di grado, che quindi è lunga come il num di nodi)
        poiché gli embedding presi come output della rete vengono dal dataloader che suddivide in batch size, non in Num_grafi_per_tipo,
        né tantomeno in quanti nodi sono rimasti dopo il pruning dei nodi sconnessi, come cioè mi serve ora
        """

        Num_nodi = self.config_class.conf['graph_dataset']['Num_nodes']
        if isinstance(Num_nodi, list):
            if self.config_class.conf['graph_dataset']['sbm']:
                Num_nodi = sum(Num_nodi)
            else:
                Num_nodi = Num_nodi[0]
        total_num_grafi = len(self.dataset_nx)
        r = 0
        for i in range(total_num_grafi):
            label = self.training_labels[i]

            exp = None
            if self.exponents is not None:
                exp = self.exponents[i][0]

            actual_node_label = None
            if self.actual_node_class is not None:
                actual_node_label = self.actual_node_class[i]   # [r:r + Num_nodi]

            original_nodel_label = None
            if self.original_node_class is not None:
                original_nodel_label = self.original_node_class[i]

            node_label_and_id = None
            #if self.original_node_class is not None:
            #    node_label_and_id = self.original_node_class[i]

            scalar_label = None
            if self.scalar_class:
                scalar_label = self.scalar_class[i]

            graph_emb = self.graph_embedding_array[i]
            node_emb = self.node_embedding_array[r:r + Num_nodi]
            graph_output = self.output_array[i]

            # poiché nel training prepare ho shufflato coerentemente sia dataset_pyg che original_class che labels,
            # anche embeddings_array che arriva dal dataloader con shuffle=false ha il loro stesso ordine
            toappend = Embedding_per_graph(graph_emb, node_emb, [], label, exp,
                                           node_label_and_id, actual_node_label, original_nodel_label, scalar_label,
                                           graph_output)
            self.emb_pergraph.append(toappend)
            r += Num_nodi

    def separate_embedding_by_classes(self):
        # devo gestire il caso in cui i target non siano scalari
        distinct_graph_labels = np.unique([n.graph_label for n in self.emb_pergraph], axis=0)
        for l in distinct_graph_labels:
            emb_perclass = [n for n in self.emb_pergraph if (n.graph_label == l).all()]
            self.emb_perclass.append(emb_perclass)

    def get_all_graph_emb_per_class(self):
        graph_emb_perclass = [[emb.graph_embedding.squeeze() for emb in emb_per_graph] for emb_per_graph in self.emb_perclass]
        return graph_emb_perclass

    def get_all_scalar_labels_per_class(self):
        labels_ripetute_pergraph = [[emb.scalar_label for emb in emb_per_graph] for emb_per_graph in self.emb_perclass]
        return np.array(labels_ripetute_pergraph).flatten()
    def get_unique_class_labels(self):
        labels = [emb_per_graph[0].scalar_label for emb_per_graph in self.emb_perclass]
        return labels

    def get_all_node_emb_per_class(self):
        node_emb_perclass = [[emb.node_embedding_array.squeeze() for emb in emb_per_graph] for emb_per_graph in self.emb_perclass]
        return node_emb_perclass

# endregion

    def get_average_corr_nodeemb(self):
        #print(self.node_emb_pergraph[0:5][0:5][0])
        correlations = []
        #print(self.node_emb_pergraph[0].flatten()[0:10])
        #print(self.original_class[0][0:10])
        #print(np.corrcoef(self.node_emb_pergraph[0].flatten()[0:10], self.original_class[0][0:10])[0, 1])
        for i, emb in enumerate(self.emb_pergraph):
            #print(f"emb len: {len(emb)}")
            #print(f"015: {emb_i[0:3]}")
            class_i = np.array(self.original_node_class[i])  # TODO: correggere perché ora original class contiene anche il modo ID
            #print(f"len class_i: {len(class_i)}")
            corr = np.corrcoef(emb, class_i)[0, 1]
            correlations.append(corr)
        #print(f"corr: {correlations}")
        return sum(correlations) / len(correlations)

    def calc_max_degree(self):
        if self.original_node_class is not None:
            for s in self.original_node_class:
                # la original class quì sarebbe la distribuzione del grado
                if isinstance(s, list):
                    self.max_degree.append(max(s))
                else:
                    self.max_degree.append(s)
        
    def cosdist(self, a,b):
        return dot(a, b)/(norm(a)*norm(b))

    def calc_coppie(self):
        NN = self.graph_embedding_array.shape[0]
        coppie_numeric = list(itertools.combinations(range(NN), 2))
        assert len(coppie_numeric) == (NN * (NN-1))/2
        #print(f"{len(coppie_numeric)} possibili coppie")
        #self.coppie = np.array([self.embeddings_array[c,:] for c in coppie_numeric])
        self.coppie = self.graph_embedding_array[coppie_numeric] # è equivalente alla riga precedente: numpy integer mask, prende la shape della mask
        self.coppie_labels = [(self.training_labels[c[0]], self.training_labels[c[1]]) for c in coppie_numeric]
        if self.original_node_class: # TODO: correggere perché ora original class contiene anche il modo ID
            self.coppie_orig_class = [(self.original_node_class[c[0]], self.original_node_class[c[1]]) for c in coppie_numeric]
            assert len(self.coppie_orig_class) == (NN * (NN-1))/2

    def calc_distances(self, num_emb_neurons):
        self.calc_coppie()
        i = 0
        for a,b in self.coppie:
            cos_dist = self.cosdist(a,b)
            dist = np.linalg.norm(a-b)
            label_a = self.coppie_labels[i][0]
            label_b = self.coppie_labels[i][1]
            self.cos_distances.append((cos_dist, (label_a, label_b)))
            if self.original_node_class:  # TODO: correggere perché ora original class contiene anche il modo ID
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
            if num_emb_neurons == 2:
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
            else:
                self.inter_dists = None
                self.intra_dists = None

            # calculate average of each euclidean distribution
            mean_intra = np.mean(self.intra_dists)
            mean_inter = np.mean(self.inter_dists)
            self.difference_of_means = mean_inter - mean_intra
            if self.inter_dists is not None and self.intra_dists is not None:
                print("calcolo l'overlap")
                #self.overlap = self.calc_overlap(self.inter_dists, self.intra_dists)



# region intra e inter distances many classes
    def calc_coppie_intracluster(self, graph_emb_cluster):
        NN = graph_emb_cluster.shape[0]
        coppie_numeric = list(itertools.combinations(range(NN), 2))
        coppie = graph_emb_cluster[coppie_numeric]
        # coppie_labels = le stesse per tutti
        distances = []
        for a, b in coppie:
            dist = np.linalg.norm(a - b)
            distances.append(dist)
        return distances

    def calc_coppie_multiclass(self):
        graphemb_perclass = np.array(self.get_all_graph_emb_per_class())

        # calcolo le inter-cluster distances
        NN = len(graphemb_perclass)
        coppie_classi = list(itertools.combinations(range(NN), 2))
        coppie_labels_class = [(self.emb_perclass[c[0]][0].scalar_label, self.emb_perclass[c[1]][0].scalar_label) for c in coppie_classi]
        coppie_gemb_perclass = [graphemb_perclass[c, :] for c in coppie_classi]

        inter_dists_perclass = []
        for cluster_a, cluster_b in coppie_gemb_perclass:
            inter_dists_ab = []
            NN = len(cluster_a)  # assumo sia la stessa per tutti i cluster
            coppie_ab = list(itertools.combinations(range(NN), 2))
            for ca, cb in coppie_ab:
                a = cluster_a[ca]
                b = cluster_b[cb]
                dist = np.linalg.norm(a - b)
                inter_dists_ab.append(dist)
            inter_dists_perclass.append(inter_dists_ab)

        # calcolo le intra-cluster distances
        intra_dists_perclass = []
        for i in range(len(graphemb_perclass)):
            intra_dists_perclass.append(self.calc_coppie_intracluster(graphemb_perclass[i]))
        single_labels = self.get_unique_class_labels()

        return inter_dists_perclass, intra_dists_perclass, coppie_labels_class, single_labels

# endregion
    def intorno(self, p_teorica, p_attuali, soglia):
        mask = []
        for p in p_attuali:
            if p > p_teorica - soglia and p < p_teorica + soglia:
                mask.append(True)
            else:
                mask.append(False)
        return mask

    def calc_overlap(self, inter_dists, intra_dists):
        massimo = max(inter_dists)
        bins = np.linspace(0, massimo, 200)
        v1, b = np.histogram(inter_dists, bins=bins)
        v2, b = np.histogram(intra_dists, bins=bins)
        overlap_product = (v1 * v2).sum()
        norm = (v1 * v1 + v2 * v2).sum()
        return overlap_product / norm

    def calc_graph_emb_correlation(self):
        # solo nel caso della dimensione di embedding = 1
        num_nodes = self.config_class.conf['graph_dataset']['Num_nodes']

        #if self.config_class.conf['graph_dataset']['confmodel']:  # self.original_class ha un'array (degree sequence) per ciascun grafo
        #    #self.original_class #
        #    correlazioni = np.corrcoef(self.graph_embedding_array.flatten(), self.original_class)[0, 1]
        self.graph_correlation_per_class = []
        #print((self.scalar_class))
        actual_p = np.array([nx.to_numpy_matrix(t).sum(axis=1).mean() / (200 - 1) for t in self.dataset_nx]) # num_nodes

        if self.continuous_p:
            self.total_graph_correlation = np.corrcoef(self.graph_embedding_array.flatten(), self.training_labels)[0,1]
            for p in self.probabilities_ER:
                mask_int = np.argwhere(self.intorno(p, self.training_labels, 0.05)).flatten()
                #plt.scatter(self.embeddings_array[mask_int].flatten(),actual_p[mask_int])
                emb = self.graph_embedding_array[mask_int].flatten()
                correlaz = np.corrcoef(emb, actual_p[mask_int])[0, 1]
                self.graph_correlation_per_class.append(correlaz)
        else:
            for p in self.probabilities_ER:
                # TODO: metodo elegante che dovrei usare a monte quando separo i graph embedding per classe
                mask_int = np.argwhere(np.array(self.scalar_class) == p).flatten()
                emb = self.graph_embedding_array[mask_int].flatten()
                correlaz = np.corrcoef(emb, actual_p[mask_int])[0, 1]
                self.graph_correlation_per_class.append(correlaz)

        self.total_graph_correlation = np.mean(self.graph_correlation_per_class)

    def calc_node_emb_correlation(self):
        avg_corr_classes = []
        avg_tau_classes = []
        for class_emb in self.emb_perclass:
            corrs = []
            kendall_tau = []
            for emb_pergraph in class_emb:
                emb_pergraph.get_correlation_with_degree_sequence()
                emb_pergraph.get_kendall_with_degree_sequence()
                corrs.append(emb_pergraph.correlation_with_degree)
                kendall_tau.append(emb_pergraph.kendall_with_degree)
            avg_corr_class0 = sum(corrs) / len(class_emb)
            avg_corr_classes.append(avg_corr_class0)
            avg_tau = sum(kendall_tau) / len(class_emb)
            avg_tau_classes.append(avg_tau)

        self.node_correlation_per_class = avg_corr_classes
        self.total_node_correlation = np.mean(avg_corr_classes)

        return avg_corr_classes, avg_tau_classes

    def calc_regression_error(self):
        self.regression_error = np.sqrt(np.sum((self.graph_embedding_array.flatten() - self.training_labels) ** 2)) / len(self.graph_embedding_array)

    def calc_instrinsic_dimension(self, num_emb_neurons, calc_perclass=False):
        metodo = skdim.id.TwoNN()
        # metodo = skdim.id.DANCo() troppo tempo
        # metodo = skdim.id.ESS() troppo tempo
        # metodo = skdim.id.FisherS()
        # metodo = skdim.id.CorrInt() quasi 10 minuti per 20 punti
        # metodo = skdim.id.lPCA()

        node_emb_dims = []
        graph_emb_dims = []
        if calc_perclass:
            num_classi = len(np.unique([n.graph_label for n in self.emb_pergraph], axis=0))
            node_emb_perclass_pergraph = np.array(self.get_all_node_emb_per_class())
            node_emb_perclass = node_emb_perclass_pergraph.reshape(num_classi, -1, num_emb_neurons)

            for n in node_emb_perclass:
                dim = metodo.fit(n).dimension_
                node_emb_dims.append(dim)

            graph_emb_perclass = np.array(self.get_all_graph_emb_per_class())
            for n in graph_emb_perclass:
                dim = metodo.fit(n).dimension_
                graph_emb_dims.append(dim)

        total_graph_emb_dim = metodo.fit(self.graph_embedding_array).dimension_
        total_node_emb_dim = metodo.fit(self.node_embedding_array).dimension_

        return node_emb_dims, graph_emb_dims, total_node_emb_dim, total_graph_emb_dim

    def get_metrics(self, num_emb_neurons, training_mode):
        if num_emb_neurons == 1:
            self.calc_graph_emb_correlation()  # calcola self.graph_correlation_per_class o self.total_graph_correlation
            self.calc_node_emb_correlation()
            if training_mode == TrainingMode.mode3:
                self.calc_regression_error()
        else:
            #self.calc_distances(num_emb_neurons)  # calcola self.difference_of_means
            self.node_emb_dims, self.graph_emb_dims, self.total_node_emb_dim, self.total_graph_emb_dim = self.calc_instrinsic_dimension(num_emb_neurons)


class Embedding_per_graph():
    def __init__(self, graph_embedding, node_embedding_array, node_embeddings_array_id,
                 graph_label, exponent, node_label_and_id=None,
                 actual_node_class=None, original_node_class=None, scalar_label=None,
                 graph_output=None, nx_recon_graph=None):
        self.graph_embedding = graph_embedding
        self.node_embedding_array = node_embedding_array
        self.node_embeddings_array_id = node_embeddings_array_id
        self.graph_label = graph_label
        self.scalar_label = scalar_label
        self.exponent = exponent  # posso aggiungere un controllo che sia sempre anche qu' exponent[1] == graph_label
        self.node_label_and_id = node_label_and_id
        self.correlation_with_degree = None
        self.kendall_with_degree = None
        self.output = graph_output

        if original_node_class is not None:
            self.original_node_class = original_node_class
        if actual_node_class is not None:
            self.actual_node_class = actual_node_class
        elif node_label_and_id is not None:
            self.actual_node_class = [n[1] for n in list(node_label_and_id)]  # contiene anche i node ID

        self.align_embedding_id_with_degree_sequence_id()

    def align_embedding_id_with_degree_sequence_id(self):
        #print(self.node_embeddings_array_id)
        #print(self.node_label)
        pass

    def get_correlation_with_degree_sequence(self):
        node_emb_array = np.array(self.node_embedding_array).flatten()
        self.correlation_with_degree = np.corrcoef(node_emb_array, self.actual_node_class)[0, 1]  # TODO: node_label flatten!!!

    def get_kendall_with_degree_sequence(self):
        node_emb_array = np.array(self.node_embedding_array).flatten()
        self.kendall_with_degree, p_value = kendalltau(node_emb_array, self.actual_node_class)  # TODO: node_label flatten!!!



