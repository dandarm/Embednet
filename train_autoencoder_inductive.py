import traceback
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.utils import class_weight, compute_class_weight

import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_adj
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d, LeakyReLU, Hardsigmoid, Tanh, ELU, Hardtanh, ReLU

from train import Trainer
from models import GCN, AutoencoderGCN, ConfModelDecoder, MLPDecoder
from models import view_parameters, get_parameters, new_parameters, modify_parameters, Inits, modify_parameters_linear, view_weights_gradients
#from Dataset_autoencoder import DatasetAutoencoder
from embedding import Embedding_autoencoder_per_graph, Embedding_autoencoder
from Metrics import Metrics
from torchmetrics import HammingDistance

class Trainer_Autoencoder(Trainer):
    def __init__(self, config_class, verbose=False, rootsave="."):
        super().__init__(config_class, verbose, rootsave)
        self.name_of_metric = ["auc", "pr_auc", "f1_score", "euclid"]

        #self.HD = HammingDistance(task="binary")

        self.HardTanh = Hardtanh(0,1)

        # la devo lasciare perche devo poter sistemare l'output dopo il prodotto scalare Z.Zt????
        #  TODO: no non mi serve la posso cancellare, perche  l'output deve essere gia attivato
        # self.loss_activation = None
        # if self.conf['model']['autoencoder']:
        #     self.loss_activation = torch.sigmoid
        # elif self.conf['model'].get('autoencoder_confmodel'):
        #     self.loss_activation = ReLU()    #nn.Identity()    # Hardtanh(0.01, 0.99)

        #self.num_graphs = self.config_class.conf['graph_dataset']['Num_grafi_per_tipo'] * self.config_class.num_classes
        #print(f"nodi per grafo e num grafi: {self.num_nodes_per_graph} {self.num_graphs}")



    def init_GCN(self, init_weights_gcn=None, init_weights_lin=None, verbose=False):
        """
        Returns the GCN model given the class of configurations
        :param config_class:
        :param verbose:
        :return:
        """
        if verbose: print("Initialize model")
        if self.config_class.conf['device'] == 'gpu':
            device = torch.device('cuda')
        else:
            device = "cpu"


        encoder = GCN(self.config_class)
        model = AutoencoderGCN.from_parent_instance(dic_attr="dict_attr", parent_instance=encoder)
        if self.config_class.conf['model']['autoencoder']:
            model.set_decoder(encoder)
        elif self.config_class.conf['model'].get('autoencoder_confmodel'):
            model.set_decoder(encoder, ConfModelDecoder())
        elif self.config_class.conf['model'].get('autoencoder_mlpdecoder'):
            # quì suppongo che i nodi siano gli stessi per tutti i grafi
            out_dim = self.config_class.conf['graph_dataset']['Num_nodes'][0]  # numero di nodi per grafo
            in_dim = encoder.convs[-1].out_channels  # dimensione dell'embedding: ultimo layer convolutional
            model.set_decoder(encoder, MLPDecoder(in_dim, out_dim))


        model.to(device)
        if init_weights_gcn is not None:
            modify_parameters(model, init_weights_gcn)
        if init_weights_lin is not None:
            modify_parameters_linear(model, init_weights_lin)
        if verbose:
            print(model)
        return model
        
    def gestisci_batch(self, complete_adjacency_matrix, batch_array, num_nodi_adj):
        # !SLOW
        # create a mask for pairs of nodes that belong to the same graph
        #################mask = (batch_array.unsqueeze(1) == batch_array.unsqueeze(0)).to(torch.float)
        # filtra con la maschera assegnando a 0 i prodotti incrociati tra diversi grafi
        
        #################block_adj = complete_adjacency_matrix * mask
        # in realtà non ho bisogno di portare gli elementi fuori dai blocchi diagonali a 0, 
        # tanto comunque ora estraggo questi blocchi contando il lato come numero di nodi 
        block_adj = complete_adjacency_matrix
        # estrai in un tensore dim=3 con tante matrici adj
        adjs = self.extract_block_diag(block_adj,  num_nodi_adj)        
        return adjs
    
    def extract_block_diag(self, A, step):
        # !SLOW
        diag_block = torch.stack([A[i:i+step, i:i+step] for i in range(0, A.shape[0], step)])
        return diag_block
    
    def calc_decoder_for_batches(self, total_z, num_nodes):
        """Necessario nel caso dei batch per togliere dal prodotto scalare i nodi di grafi diversi
        :param total_z: embedding complessivo di tutti i nodi del batch
        :param num_nodes: lista che rappresenta i nodi per ciascun grafo
        :return:
        """
        #print(f"{len(total_z)} - {sum(num_nodes)}")
        assert len(total_z) == sum(num_nodes), f"Num_nodes totali non torna con i node embedding di total_z, {len(total_z)} != {sum(num_nodes)}"
        # devo fare un padding: prendo il max tra i num nodi
        max_nodes = max(num_nodes)
        if list(set(num_nodes)) == [max_nodes]:
            need_padding = False  # l'array è tutto costante
        else:
            need_padding = True  # va reso costante col padding
        start_out = torch.empty((1, max_nodes, max_nodes), device=torch.device(self.device))

        #for i in range(0, len(total_z), num_nodes):
        i = 0
        for n in num_nodes:
            #print(f"i: {i}, n di num_nodes: {n}")   otengo ora l'mebedding di tutti i nodi del singolo grafo
            z = total_z[i:i+n]
            if self.conf['model']['autoencoder']:  # solo nel caso di autoencoder più semplice
                out = self.model.forward_all(z, sigmoid=False)
                #out = torch.nn.ReLU()(out)
                #print("siamo proprio quììììììììììììììììì")
                #print(np.histogram(out.clone().detach().cpu().numpy().ravel(), bins=20))
                out = self.HardTanh(out)
                #print(np.histogram(out.clone().detach().cpu().numpy().ravel(), bins=20))

            else:
                out = self.model.forward_all(z, sigmoid=False)
            #print(f"out shape!!! {out.shape}")
            #self.check_nans(out, z)
            #print(f"z shape {z.shape} -  out shape {out.shape}")
            if need_padding:
                d = max_nodes - out.shape[0]
                # here, pad = (padding_left, padding_right, padding_top, padding_bottom)
                out = F.pad(input=out, pad=(0, d, 0, d), mode='constant', value=0)
            #try:

            start_out = torch.cat((start_out, out.unsqueeze(0)))
            #except Exception as e:
            #print(e)
            i += n
        return start_out[1:]  # perché la prima riga è vuota

    def separate_per_graph_from_batch(self, batch_embedding, numnodes_list, input_adj=None):
        """
        :param batch_embedding:  embedding complessivo di tutti i nodi del batch
        :param numnodes_list: lista che rappresenta i nodi per ciascun grafo
        :return: una lista di Embedding_autoencoder (per graph)
        """
        embs = []
        i = 0
        for j, n in enumerate(numnodes_list):
            # z è il vettore di embedding dei nodi per un singolo grafo
            z = batch_embedding[i:i + n]
            if input_adj is not None:
                ia = input_adj[j]
            else:
                ia = None
            emb_auto = Embedding_autoencoder_per_graph(z, input_adj_mat=ia)
            embs.append(emb_auto)
            i += n
        return embs


    def transf_complementare(self, array):
        ##v_a2 = torch.tensor(adjusted_pred_adj.clone().ravel(),
        ##                   device='cuda',
        ##                   requires_grad=True,
        ##                   dtype=torch.float32)  ##.unsqueeze(1)
        complementare = 1 - array
        res = torch.cat((array, complementare), -1)
        return res
              
        
    def get_weighted_criterion(self, y_true):
        uni = y_true.sum()
        tot = len(y_true)
        # al primo posto è il peso della classe negativa (0)
        #class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_true), y=y_true)
        class_weights = [(tot-uni), uni]
        custom_weights = torch.tensor(class_weights, dtype=torch.float32, device='cuda')
        #pos_weight = torch.tensor(class_weights, dtype=torch.float32)
        # imposto come peso  la quantità di uni, la classe minoritaria
        #criterion = torch.nn.BCEWithLogitsLoss(pos_weight=custom_weights) ##class_weights[1])
        criterion = torch.nn.BCELoss(reduction=None)
        #criterion = nn.MSELoss()
        return criterion

    def get_all_weights_for_loss(self, y_true):
        """
            se y_true e fatto di tutti zeri non va bene
        """
        uni = y_true.sum()
        tot = len(y_true)
        # al primo posto è il peso della classe negativa (0)
        positive_weight = (tot - uni) / uni  # è maggiore di 1
        # diz = {c: w for c, w in enumerate(positive_weight)}
        # torch.tensor([diz[y] for y in y_true])
        # visto che sono 1 e 0, moltiplico tutto l'array per positive_w-1, e poi sommo 1.
        return (y_true * (positive_weight - 1)) + 1

    def encode_decode_inputadj(self, data, i, batch_size, num_nodes_batch):
        # encoding su tutti i grafi del batch, tutte le edges di ciascun grafo:
        total_batch_z = self.model.encode(data.x, data.edge_index, data.batch)
        # total batch ha dimensione:  ( (nodipergrafo * grafi del batch), dim_embedding )
        # plt.hist(total_batch_z.detach().cpu().numpy().squeeze(), bins=50);
        # plt.show()
        # print(f"x shape: {data.x.shape}\t nodi per grafo: {self.gg.num_nodes_per_graph}")

        # z è l'embedding di ciascun nodo (tutti i nodi del batch)
        # il decoder Innerproduct calcola anche la matrice di adiacenza con il forward_all
        # out = self.model.forward_all(z, sigmoid=False)  effettua un prod scalare per z = total_z[i:i+n]
        # quindi è già separato per grafi
        adjusted_pred_adj = self.calc_decoder_for_batches(total_batch_z, num_nodes_batch[i:i + batch_size])

        # (tolto perché prendeva troppa memoria anche la sigmoid)
        # out = self.model.forward_all(z)
        # adjusted_pred_adj = self.gestisci_batch(out, data.batch, self.num_nodes_per_graph)

        # ottieni la matrice di adiacenza dalle edge indexes
        input_adj = to_dense_adj(data.edge_index, data.batch)
        # print(f"adjusted_pred_adj shape {adjusted_pred_adj.shape}, input adj shape: {input_adj.shape}")

        return total_batch_z, adjusted_pred_adj, input_adj

    def check_nans(self, input_array, embedding_z):
        nans = torch.isnan(input_array).any()
        if nans:
            print("NANNNNNNS")
        non_finiti = torch.logical_not(torch.isfinite(input_array))
        if non_finiti.any():
            print("non finiti")
            print(input_array)
            print(f"Embedding z chi erano  {embedding_z}")
            print(non_finiti.sum().item() / input_array.shape[0])
            # adjusted_pred_adj[adjusted_pred_adj != adjusted_pred_adj] = 0.5
            # print(adjusted_pred_adj)
            print("\n")

    def train(self):
        self.model.train()
        running_loss = 0
        num_nodes_batch = [len(data.x) for data in self.dataset.train_loader.dataset]
        i = 0
        num_batches = 0
        for data in self.dataset.train_loader:
            total_batch_z, adjusted_pred_adj, input_adj = self.encode_decode_inputadj(
                data, i, self.dataset.train_loader.batch_size, num_nodes_batch)

            adjusted_pred_adj_r = adjusted_pred_adj.ravel()
            input_adj_r = input_adj.ravel()
            ##print(f"adjusted_pred_adj shape: {adjusted_pred_adj_r.shape}")
            ##print(f"input_adj shape: {input_adj_r.shape}")

            # quindi la loss è calcolata come l'errore rispetto alla riconstruzione di edges
            # devo usare la binary cross entropy perché devo pesare diversamente le edges,
            # essendo un problema sbilanciato (la matrice è sparsa!)

            #criterion = self.get_weighted_criterion(input_adj_r)  # .detach().cpu().numpy()
            #input_adj_t = self.transf_complementare(input_adj_r.unsqueeze(1))
            #adjusted_pred_adj_t = self.transf_complementare(adjusted_pred_adj_r.unsqueeze(1))

            ##print(f"adjusted_pred_adj_t shape: {adjusted_pred_adj_t.shape}")
            ##print(f"input_adj_t shape_t: {input_adj_t.shape}")
            loss = self.calc_loss(adjusted_pred_adj_r, input_adj_r)

            ##view_weights_gradients(self.model)
            loss.backward()  # Derive gradients.
            # check che i gradienti siano non nulli

            #self.check_zero_gradients(loss)

            ##print(gradients)
            ##plt.hist([g.detach().cpu().numpy().ravel() for g in gradients])
            ##plt.show()

            # se volessi aggiungere del rumore al gradiente per (o ai pesi)
            # con lo scopo di uscire da eventuali minimi locali
            #model.layer.weight.grad = model.layer.weight.grad + torch.randn_like(model.layer.weight.grad)

            
            self.optimizer.step()  # Update parameters based on gradients.
            self.optimizer.zero_grad()  # Clear gradients.
            # self.scheduler.step()
            running_loss += loss.item()
            i += self.dataset.train_loader.batch_size
            num_batches += 1

        return running_loss / num_batches

    def check_zero_gradients(self, loss):
        gradients = view_weights_gradients(self.model)
        for g in gradients:
            if torch.count_nonzero(g).item() == 0:
                pars = get_parameters(self.model.convs)
                for p in pars:
                    print(f"loss: {loss.item()}")
                    print(f" max: {max(p)}, min: {min(p)}")
                raise Exception("Problema con i gradienti: sono tutti ZER000")

    def test(self, loader):
        self.model.eval()
        running_loss = 0
        num_nodes_batch = [len(data.x) for data in loader.dataset]

        i = 0
        num_batches = 0
        with torch.no_grad():
            for data in loader:
                total_batch_z, adjusted_pred_adj, input_adj = self.encode_decode_inputadj(
                    data, i, loader.batch_size, num_nodes_batch)

                adjusted_pred_adj_r = adjusted_pred_adj.ravel()
                input_adj_r = input_adj.ravel()

                loss = self.calc_loss(adjusted_pred_adj_r, input_adj_r)

                running_loss += loss.item()
                i += loader.batch_size
                num_batches += 1

        return running_loss / num_batches

    def calc_loss(self, adjusted_pred_adj_r, input_adj_r):
        """
        :param adjusted_pred_adj_r:  deve essere sempre un vettore 1D quello rispetto al quale si calcola la loss
        :param input_adj_r:  idem. Altrimenti calcola tante loss, una per ogni vettore lungo l'ultima dimensione
        :param is_weighted:
        :return:
        """
        #pred_activated = self.loss_activation(adjusted_pred_adj_r)
        if self.is_weighted:
            unreduced_loss = self.criterion(adjusted_pred_adj_r, input_adj_r)
            all_weights = self.get_all_weights_for_loss(input_adj_r)
            loss = (unreduced_loss.squeeze() * all_weights).mean()
        else:
            loss = self.criterion(adjusted_pred_adj_r, input_adj_r)
        return loss

    def calc_loss_input_dataset(self, loader):
        coppie = self.dataset.get_coppie_from_dataset(loader)
        shape = coppie.shape[-1]

        start_out_pred = torch.empty((1, shape, shape))
        start_out_true = torch.empty((1, shape, shape))
        for a1, a2 in coppie:
            start_out_pred = torch.cat((start_out_pred, torch.tensor(a1).unsqueeze(0)))
            start_out_true = torch.cat((start_out_true, torch.tensor(a2).unsqueeze(0)))
        out1 = start_out_pred[1:].ravel()
        out2 = start_out_true[1:].ravel()

        final_loss = self.calc_loss(start_out_pred, start_out_true).item()
        return final_loss

    def calc_loss_input_dataset_ER(self, loader):
        """
        Voglio calcolare la loss di una matrice con valori costanti = prob. ER rispetto a ogni matrice input
        Quest matrice costante è usata nel
        """
        #if self.config_class.graphtype == GraphType.ER:
        Adjs = self.dataset.get_concatenated_input_adjs(loader)
        cost_matrix = self.dataset.get_concatenated_constant_matrix(loader)
        #print(cost_matrix.shape, Adjs.shape)

        final_loss = self.calc_loss(cost_matrix.ravel(), Adjs.ravel()).item()
        return final_loss

    # def get_embedding(self, loader, type_embedding='both'):
    #     self.model.eval()
    #     graph_embeddings_array = []
    #     node_embeddings_array = []
    #     node_embeddings_array_id = []
    #     final_output = []
    #
    #     with torch.no_grad():
    #         for data in loader:
    #             if type_embedding == 'graph':
    #                 out = self.model.encode(data.x, data.edge_index, data.batch, graph_embedding=True)
    #                 to = out.detach().cpu().numpy()
    #                 graph_embeddings_array.extend(to)
    #             elif type_embedding == 'node':
    #                 out = self.model.encode(data.x, data.edge_index, data.batch, node_embedding=True)
    #                 to = out.detach().cpu().numpy()
    #                 node_embeddings_array.extend(to)
    #                 #node_embeddings_array_id.extend(data.id)
    #             elif type_embedding == 'both':  # quì ho garantito che i graph embedding sono ordinati come i node_embedding
    #                 node_out = self.model.encode(data.x, data.edge_index, data.batch, node_embedding=True)
    #                 to = node_out.detach().cpu().numpy()
    #                 #print(f"node emb size: {to.nbytes}")
    #                 node_embeddings_array.extend(to)
    #                 #node_embeddings_array_id.extend(data.id)
    #                 graph_out = self.model.encode(data.x, data.edge_index, data.batch, graph_embedding=True)
    #                 to = graph_out.detach().cpu().numpy()
    #                 #print(f"graph emb size: {to.nbytes}")
    #                 graph_embeddings_array.extend(to)
    #
    #                 final_output.extend([None]*len(to))
    #
    #     graph_embeddings_array = np.array(graph_embeddings_array)
    #     node_embeddings_array = np.array(node_embeddings_array)
    #     return graph_embeddings_array, node_embeddings_array, node_embeddings_array_id, final_output
    
    def get_recon_adjs(self, loader):
        # restituisce tutte le matrici di adiacenza e le feature del dataset
        self.model.eval()
        adjs_list = []
        feats = []
        with torch.no_grad():
            for data in loader:
                total_batch_z = self.model.encode(data.x, data.edge_index, data.batch) 
                adjusted_pred_adj = self.calc_decoder_for_batches(total_batch_z, self.num_nodes_per_graph)
                recon_adjs = adjusted_pred_adj.detach().cpu().numpy()
                adjs_list.extend(recon_adjs)
                
                #anche per le feature x
                for i in range(0, len(data.x), self.num_nodes_per_graph):
                    x = data.x[i:i+self.num_nodes_per_graph]
                    x = x.detach().cpu().numpy()
                    feats.append(x)
                
        return np.array(adjs_list), np.array(feats)

    def get_embedding_autoencoder(self, loader):
        """
        Funzione che si occupa di prendere l'embedding nel caso di autoencoding,
        è l'equivalente di get_embedding nel trainer base, ma istanzia anche la classe
        Embedding_autoencoder
        :param loader:
        :return:
        """
        self.model.eval()
        graph_embeddings_array = []
        node_embeddings_array = []
        node_embeddings_array_id = []
        final_output = []
        embeddings_per_graph = []

        num_nodes_batch = [len(data.x) for data in loader.dataset]
        #print(f"num_node_batch = {num_nodes_batch}, batch_size = {loader.batch_size}")
        with torch.no_grad():
            i = 0
            for data in loader:
                total_batch_z, adjusted_pred_adj, input_adj = self.encode_decode_inputadj(
                    data, i, loader.batch_size, num_nodes_batch)

                # calcolo l'emb per graph
                # print(f"shape z  {total_batch_z.shape},  shape input_adj {input_adj.shape}")
                epg = self.separate_per_graph_from_batch(
                    total_batch_z, num_nodes_batch[i:i + loader.batch_size],
                    input_adj)
                # print("verifico decoder output")
                # print(all([np.allclose(e.decoder_output, adjusted_pred_adj[i].detach().cpu().numpy()) for i,e in enumerate(epg)]))
                embeddings_per_graph.extend(epg)
                i += loader.batch_size
                # verifico che siano lo stessa cosa...qunte volte lo faccio?
                # print(f"\n i: {i} ")
                # print( all([torch.all(torch.eq(a, e.decoder_output)) for a,e in zip(adjusted_pred_adj, epg)]))

        # devo calcolare subito l'output perché deve passare per le funzioni di torch e poi
        # posso fare detach
        # TODO: però vorrei ricontrollare se posso trovare un altro modo per posticipare queste operazioni
        #  e farle fare in parallelo mentre il training va avanti
        # calcolo il z.zT e la matrice di adiacenza binaria


        for e in embeddings_per_graph:
            if self.conf['model']['autoencoder']:
                e.calc_decoder_output(self.model.forward_all, sigmoid=False)  #, activation_func=self.loss_activation)
                #e.decoder_output = torch.nn.Hardtanh(0,1)(e.decoder_output)
            else:
                e.calc_decoder_output(self.model.forward_all, sigmoid=False)
            e.to_cpu()
            e.calc_degree_sequence()
            e.sample_without_threshold()


        return embeddings_per_graph


    def calc_metric(self, loader):
        """
        Calcola una AUC per la ricostruzione dell'intera matrice
        :param loader:
        :return: un oggetto Metric contenente embeddings_per_graph (Embedding_autoencoder class)
        """
        self.model.eval()
        num_nodes_batch = [len(data.x) for data in loader.dataset]
        with torch.no_grad():
            i = 0
            inputs = []
            predictions = []
            for data in loader:
                total_batch_z, adjusted_pred_adj, input_adj = self.encode_decode_inputadj(
                    data, i, loader.batch_size, num_nodes_batch)
                input_adj_flat = input_adj.detach().cpu().numpy().ravel()
                pred_adj_flat = adjusted_pred_adj.detach().cpu().numpy().ravel()
                #non posso sommare o mediare le metriche, devo accumumlare gli array
                predictions.append(pred_adj_flat)
                inputs.append(input_adj_flat)
                i += loader.batch_size

            inputs = np.concatenate(inputs)
            predictions = np.concatenate(predictions)

            try:
                auc = roc_auc_score(inputs, predictions)
                # average_precision è la PR_AUC
                average_precision = average_precision_score(inputs, predictions)

                # f1 score per il calcolo della threshold migliore
                #thresholds = np.linspace(0.05, 0.95, 10)
                #start = time.time()
                #f1scores = [f1_score(inputs, (predictions >= t).astype('int'), average="binary") for t in thresholds]
                #end = time.time()
                #ix = np.argmax(f1scores)
                #best_threshold, f1 = thresholds[ix], f1scores[ix]
                #print(f"durata calcolo soglie: {round(end - start, 2)} - valore: {best_threshold}")
                #best_threshold = 0.5  # TODO: la impongo io e poi cambierò quello che devo cambiare
                #single_f1_score = f1_score(inputs, (predictions >= best_threshold).astype('int'), average="binary")

                pred_t = torch.tensor(predictions)
                inpt_t = torch.tensor(inputs, dtype=torch.uint8)
                #hamming_dist = self.HD(pred_t, inpt_t)
                euclid_dist = (pred_t.ravel() - inpt_t.ravel()).pow(2).sum().sqrt()
                euclid_dist = euclid_dist / inpt_t.shape[0]  # deve essere uguale anche a pred shape
                #print(hamming_dist)
            except Exception as e:
                auc = -1
                average_precision = -1
                f1 = -1
                euclid_dist = -1
                print(f"Eccezione data dalle metriche...")
                print(e)
                print(''.join(traceback.format_exception(etype=type(e), value=e, tb=e.__traceback__)))
                #print(f"nan elements: {np.count_nonzero(np.isnan(input_adj_flat))} su totale: {input_adj_flat.size}")
                print(f"nan elements: {np.count_nonzero(np.isnan(predictions))} su totale: {predictions.size}")

                #print(f"auc_{i}: {auc}", end=', ')

            metriche = Metrics(auc=auc, pr_auc=average_precision,   euclid=euclid_dist)
            # f1_score=single_f1_score, soglia=best_threshold,¨

        return metriche