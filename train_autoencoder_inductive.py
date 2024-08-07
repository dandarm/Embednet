import traceback
import time
import numpy as np
import matplotlib.pyplot as plt
#from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.utils import class_weight, compute_class_weight

import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_adj
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d, LeakyReLU, Hardsigmoid, Tanh, ELU, Hardtanh, ReLU

from train import Trainer
from models import GCN, AutoencoderGCN, ConfModelDecoder, MLPDecoder, MLPCMDecoder
from models import view_parameters, get_parameters, new_parameters, modify_parameters, Inits, modify_parameters_linear, view_weights_gradients
from embedding import Embedding_autoencoder_per_graph, Embedding_autoencoder
from Metrics import Metrics

from grid_automate import calc_media_scarti
from torchmetrics import HammingDistance

class Trainer_Autoencoder(Trainer):
    def __init__(self, config_class, verbose=False, rootsave="."):
        super().__init__(config_class, verbose, rootsave)
        #self.name_of_metric = ["auc", "pr_auc", "f1_score", "euclid"]
        self.name_of_metric = ["diff_deg_seq", "MSE"]  # "auc",  "f1_score" pr_auc euclid

        #self.HD = HammingDistance(task="binary")

        self.HardTanh = Hardtanh(0,1)

        self.total_dataset_reference_loss = None
        self.mse = torch.nn.MSELoss()

    def init_GCN(self, init_weights_gcn=None, init_weights_lin=None, verbose=False, dataset_degree_prob_infos=None):
        """
        Returns the GCN model given the class of configurations
        :param config_class:
        :param verbose:
        :return:
        """
        if verbose: print("Initialize model")

        encoder = GCN(self.config_class, dataset_degree_prob_infos=dataset_degree_prob_infos)
        model = AutoencoderGCN.from_parent_instance(dic_attr="dict_attr", parent_instance=encoder)
        self.init_decoder(encoder, model)

        model.to(self.device)
        if init_weights_gcn is not None:
            modify_parameters(model, init_weights_gcn)
        if init_weights_lin is not None:
            modify_parameters_linear(model, init_weights_lin)
        if verbose:
            print(model)
        return model

    def init_decoder(self, encoder, model):
        if self.config_class.conf['model']['autoencoder']:
            model.set_decoder(encoder)
        elif self.config_class.conf['model'].get('autoencoder_confmodel'):
            model.set_decoder(encoder, ConfModelDecoder())
        elif self.config_class.conf['model'].get('autoencoder_mlpdecoder'):
            # quì suppongo che i nodi siano gli stessi per tutti i grafi
            out_dim = self.config_class.conf['graph_dataset']['Num_nodes'][0]  # numero di nodi per grafo
            in_dim = encoder.convs[-1].out_channels  # dimensione dell'embedding: ultimo layer convolutional
            model.set_decoder(encoder, MLPDecoder(self.conf, in_dim, out_dim))
        elif self.config_class.conf['model'].get('autoencoder_MLPCM'):
            in_dim = encoder.convs[-1].out_channels  # dimensione dell'embedding: ultimo layer convolution
            out_dim = self.config_class.conf['model'].get('neurons_last_linear')[-1]
            model.set_decoder(encoder, MLPCMDecoder(self.conf, in_dim, out_dim))


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
            out = self.model.forward_all(z, sigmoid=False)
            if self.conf['model']['autoencoder']:  # solo nel caso di autoencoder più semplice
                out = self.HardTanh(out)
                #print(np.histogram(out.clone().detach().cpu().numpy().ravel(), bins=20))


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

    def separate_per_graph_from_batch(self, batch_embedding, numnodes_list, input_adj=None, pred_adj_mat=None):
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
                assert ia.dim() == 2
            else:
                ia = None
            if pred_adj_mat is not None:
                pa = pred_adj_mat[j]
                assert pa.dim() == 2
            else:
                pa = None
            emb_auto = Embedding_autoencoder_per_graph(z, input_adj_mat=ia, pred_adj_mat=pa)
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


    def calc_decoder_for_batches_v2(self, total_z, batch, batch_size, num_nodes_per_graph):
        # Create the mask for each graph in the batch
        mask = batch.unsqueeze(0) == batch.unsqueeze(1)

        # Compute the full inner product matrix for the entire batch
        recon_adj_full = self.model.forward_all(total_z)
        if self.conf['model']['autoencoder']:  # solo nel caso di autoencoder più semplice
            recon_adj_full = self.HardTanh(recon_adj_full)

        # Apply the mask to zero out the connections between nodes from different graphs
        recon_adj = recon_adj_full * mask.float()

        # Reshape recon_adj into a batch of N x N matrices
        recon_adj = recon_adj.view(batch_size, num_nodes_per_graph, batch_size, num_nodes_per_graph)
        recon_adj = recon_adj.permute(0, 2, 1, 3).contiguous()
        recon_adj = recon_adj.view(batch_size * batch_size, num_nodes_per_graph, num_nodes_per_graph)
        recon_adj = recon_adj[::batch_size+1]

        return recon_adj

    def encode_decode_inputadj(self, data, i, batch_size, num_nodes_batch):
        # encoding su tutti i grafi del batch, tutte le edges di ciascun grafo:
        if self.conf['model'].get('my_normalization_adj'):
            total_batch_z = self.model.encode(data.x, data.edge_index, data.batch, edge_weight_normalized=data.edge_weight_normalized)
        else:
            total_batch_z = self.model.encode(data.x, data.edge_index, data.batch)
        # total batch ha dimensione:  ( (nodipergrafo * grafi del batch), dim_embedding )
        # plt.hist(total_batch_z.detach().cpu().numpy().squeeze(), bins=50);
        # plt.show()
        # print(f"x shape: {data.x.shape}\t nodi per grafo: {self.gg.num_nodes_per_graph}")

        # z è l'embedding di ciascun nodo (tutti i nodi del batch)
        # il decoder Innerproduct calcola anche la matrice di adiacenza con il forward_all
        # out = self.model.forward_all(z, sigmoid=False)  effettua un prod scalare per z = total_z[i:i+n]
        # quindi è già separato per grafi
        #adjusted_pred_adj = self.calc_decoder_for_batches(total_batch_z, num_nodes_batch[i:i + batch_size])

        # funzione di decoder del batch senza ciclo for
        # devo usare per ora l'assunzione che tutti i grafi abbiano lo stesso numero di nodi
        num_nodes_const = num_nodes_batch[0]
        batch_size_v2 = data.batch.max().item() + 1
        adjusted_pred_adj_v2 = self.calc_decoder_for_batches_v2(total_batch_z, data.batch, batch_size_v2, num_nodes_const)
        #assert torch.allclose(adjusted_pred_adj, adjusted_pred_adj_v2, rtol=1e-04, atol=1e-06)
        # (tolto perché prendeva troppa memoria anche la sigmoid)
        # out = self.model.forward_all(z)
        # adjusted_pred_adj = self.gestisci_batch(out, data.batch, self.num_nodes_per_graph)

        # ottieni la matrice di adiacenza dalle edge indexes
        input_adj = to_dense_adj(data.edge_index, data.batch)
        # print(f"adjusted_pred_adj shape {adjusted_pred_adj.shape}, input adj shape: {input_adj.shape}")

        return total_batch_z, adjusted_pred_adj_v2, input_adj

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
            loss = self.calc_loss(adjusted_pred_adj, input_adj)

            ##view_weights_gradients(self.model)
            loss.backward()  # Derive gradients.

            # check che i gradienti siano non nulli
            self.is_all_zero_gradient = self.check_zero_gradients(loss)
            #if self.is_all_zero_gradient:
            #    raise ZeroGradientsException("I gradienti sono tutti 0!")

            ##print(gradients)
            ##plt.hist([g.detach().cpu().numpy().ravel() for g in gradients])
            ##plt.show()

            # se volessi aggiungere del rumore al gradiente per (o ai pesi)
            # con lo scopo di uscire da eventuali minimi locali
            #model.layer.weight.grad = model.layer.weight.grad + torch.randn_like(model.layer.weight.grad)

            
            self.optimizer.step()  # Update parameters based on gradients.
            self.optimizer.zero_grad(set_to_none=True)  # Clear gradients.
            # self.scheduler.step()
            running_loss += loss.item()
            i += self.dataset.train_loader.batch_size
            num_batches += 1

        return running_loss / num_batches

    def check_zero_gradients(self, loss):
        gradients = view_weights_gradients(self.model)
        #print(f"quanti sono i gradienti: {len(gradients)}")
        all_zeros = all(torch.count_nonzero(g).item() == 0 for g in gradients)
        return all_zeros
                #pars = get_parameters(self.model.convs)
                #for p in pars:
                    #print(f"loss: {loss.item()}")
                    #print(f" max: {max(p)}, min: {min(p)}")


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

                loss = self.calc_loss(adjusted_pred_adj, input_adj)

                running_loss += loss.item()
                i += loader.batch_size
                num_batches += 1

        return running_loss / num_batches

    def debug_ATen(self, adjusted_pred_adj, input_adj):
        predmax = adjusted_pred_adj.ravel().max()
        predmin = adjusted_pred_adj.ravel().min()
        print(f"predmax: {predmax}\tpredmin: {predmin}")
    def calc_loss(self, adjusted_pred_adj, input_adj):
        """
        :param adjusted_pred_adj_r:  deve essere sempre un vettore 1D quello rispetto al quale si calcola la loss
        :param input_adj_r:  idem. Altrimenti calcola tante loss, una per ogni vettore lungo l'ultima dimensione
        :param is_weighted:
        :return:
        """
        adjusted_pred_adj_r = adjusted_pred_adj.ravel()
        input_adj_r = input_adj.ravel()
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

        final_loss = self.calc_loss(cost_matrix, Adjs).item()
        return final_loss

    def calc_loss_input_dataset_CM(self, loader):
        Adjs = self.dataset.get_concatenated_input_adjs(loader)
        starting_matrix_pij = self.dataset.get_concatenated_starting_matrix(loader)
        final_loss = self.calc_loss(starting_matrix_pij.ravel(), Adjs.ravel()).item()
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
        a partire da un dataloader
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
                    input_adj, adjusted_pred_adj)
                # print("verifico decoder output")
                # print(all([np.allclose(e.pred_adj_mat, adjusted_pred_adj[i].detach().cpu().numpy()) for i,e in enumerate(epg)]))
                embeddings_per_graph.extend(epg)
                i += loader.batch_size
                # verifico che siano lo stessa cosa...qunte volte lo faccio?  ora l'ho risolto in pred_adj_mat
                # print(f"\n i: {i} ")
                # print( all([torch.all(torch.eq(a, e.pred_adj_mat)) for a,e in zip(adjusted_pred_adj, epg)]))

        # devo calcolare subito l'output perché deve passare per le funzioni di torch e poi
        # posso fare detach
        # TODO: però vorrei ricontrollare se posso trovare un altro modo per posticipare queste operazioni
        #  e farle fare in parallelo mentre il training va avanti
        # calcolo il z.zT e la matrice di adiacenza binaria


        for e in embeddings_per_graph:
            #if self.conf['model']['autoencoder']:
                # TODO: ridurre se non serve
                #  TODO2!!!! VERIFICARE CHE IL DECODER_OUTPUT NON SIA UGUALE A adjusted_pred_adj
                #e.calc_decoder_output(self.model.forward_all, sigmoid=False)  #, activation_func=self.loss_activation)
                #e.decoder_output = torch.nn.Hardtanh(0,1)(e.decoder_output)
            #else:
                #e.calc_decoder_output(self.model.forward_all, sigmoid=False)
            e.to_cpu()
            e.calc_degree_sequence()
            #e.calc_clustering_coeff()
            #e.calc_knn()
            e.sample_without_threshold()


        return embeddings_per_graph


    def calc_metric(self, actual_node_class, embeddings):
        # calcola soltanto la differenza della sequenza di grado
        input_seq = np.array(actual_node_class).ravel()
        pred_seq = np.array([g.out_degree_seq for g in embeddings]).ravel().squeeze()

        msevalue = self.mse(torch.tensor(pred_seq), torch.tensor(input_seq)).item()

        l = len(pred_seq)
        assert len(input_seq) == l

        diff = (pred_seq - input_seq)  # / input_seq  # calcolo la differenza relativa tra la seq grado predetta e quella di input
        #print(f"sum(diff): {np.abs(diff).sum()}")
        #stats = calc_media_scarti(input_seq, diff)

        #x_vals = list(stats.keys())  # contiene i valori unici del grado, mentre input_seq sono tutti i gradi di tutti i nodi (anche ripetuti)
        #y_vals = [stats[k]['media_assoluta'] for k in x_vals]
        #y_val_abs = [stats[k]['somma_assoluta'] for k in x_vals]

        #errore_medio_assoluto_per_grado = sum(y_vals) / l
        errore_totale_assoluto_per_nodo = np.abs(diff).sum() / l
        metriche = Metrics(diff_deg_seq=errore_totale_assoluto_per_nodo, MSE=msevalue)
        #print(f"debug diff: sum(y_vals) = {sum(y_val_abs)} \t len sequence: {l} \t valore plot: {errore_medio_assoluto_per_grado} ")
        return metriche

    def calc_metric_prauc_euclid(self, loader):
        """  AL MOMENTO NON LA USO, SE MI SERVE VERIFICARE SKLEARN NELL'AMBIENTE
        Calcola una AUC per la ricostruzione dell'intera matrice
        :param loader:
        :return: un oggetto Metric contenente embeddings_per_graph (Embedding_autoencoder class)
        """
        self.model.eval()
        num_nodes_batch = [len(data.x) for data in loader.dataset]
        with torch.no_grad():
            i = 0
            num_grafi = 0
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
                num_grafi += len(data)

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
                euclid_dist = (pred_t.ravel() - inpt_t.ravel()).pow(2).sum().sqrt().item()
                euclid_dist = euclid_dist / num_grafi  # deve essere uguale anche a pred shape
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

