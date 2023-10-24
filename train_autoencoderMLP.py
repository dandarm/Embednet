import traceback
import os
from pathlib import Path
import multiprocessing
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import torch
from train import Trainer
from models import AutoencoderMLP
from Dataset_ae_mlp import Dataset_for_ae_mlp
from Metrics import Metrics
from plot_funcs import DataAutoenc2Plot, plot_metrics
from Data_AEMLP_2Plot import DataAEMLP2Plot
from embedding import Embedding_AEMLP_per_graph

class Trainer_AutoencoderMLP(Trainer):
    def __init__(self, config_class, verbose=False, rootsave="."):
        super().__init__(config_class, verbose, rootsave)
        self.name_of_metric = ["pr_auc", "euclid"]   # "auc" la tengo fuori al momento

    def init_all(self, parallel=True, verbose=False):
        """
        Inizializza modello e datasest
        :param parallel:
        :param verbose: se True ritorna il plot object del model
        :return:
        """
        model = self.initAutoencoderMLP(verbose=verbose)
        self.load_model(model)

        self.init_dataset(parallel=parallel, verbose=verbose)
        self.load_dataset(self.gg.dataset, parallel=False)

        self.create_runpath_dir()

    def initAutoencoderMLP(self, verbose=False):
        model = AutoencoderMLP(self.config_class)

        if verbose:
            print(model)

        return model

    def load_dataset(self, dataset, parallel=False):
        print("Loading Dataset...")
        if not self.config_class.conf['graph_dataset'].get('real_dataset'):
            self.dataset = Dataset_for_ae_mlp(self.config_class, super_instance=dataset, verbose=self.verbose)
            self.dataset.prepare(self.shuffle_dataset, parallel)


    def train(self):
        self.model.train()
        running_loss = 0
        num_batches = 0

        for data in self.dataset.train_loader:
            batch_data = data[0]
            output = self.model(batch_data)
            loss = self.criterion(output, batch_data)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()  # * batch_data.size(0)
            num_batches += 1

        return running_loss / num_batches

    def test(self, loader):
        self.model.eval()
        running_loss = 0
        num_batches = 0

        with torch.no_grad():
            for data in loader:
                batch_data = data[0]
                output = self.model(batch_data)
                loss = self.criterion(output, batch_data)
                running_loss += loss.item()
                num_batches += 1

        return running_loss / num_batches

    # def separate_per_graph_from_batch(self, input_batch_data, output_batch_data, numnodes_list):
    #     """
    #     :param input_batch_data:  matrici di adiacenza complessive del batch
    #     :param numnodes_list: lista che rappresenta i nodi per ciascun grafo
    #     :return: una lista di Embedding_autoencoder (per graph)
    #     """
    #     embs = []
    #     i = 0
    #     for j, n in enumerate(numnodes_list):
    #         z = input_batch_data[i:i + n]
    #         if input_adj is not None:
    #             ia = input_adj[j]
    #         else:
    #             ia = None
    #         emb_auto = Embedding_autoencoder_per_graph(z, input_adj_mat=ia)
    #         embs.append(emb_auto)
    #         i += n
    #     return embs

    def calc_metric(self, loader):
        """
        Calcola una AUC per la ricostruzione dell'intera matrice
        :param loader:
        :return: un oggetto Metric contenente embeddings_per_graph (Embedding_autoencoder class)
        """
        self.model.eval()
        with torch.no_grad():
            i = 0
            inputs = []
            predictions = []
            for data in loader:
                batch_data = data[0]
                output = self.model(batch_data)

                input_adj_flat = batch_data.detach().cpu().numpy().ravel()
                pred_adj_flat = output.detach().cpu().numpy().ravel()
                # non posso sommare o mediare le metriche, devo accumumlare gli array
                predictions.append(pred_adj_flat)
                inputs.append(input_adj_flat)
                i += len(batch_data)

            inputs = np.concatenate(inputs)
            predictions = np.concatenate(predictions)

            try:
                auc = roc_auc_score(inputs, predictions)
                # average_precision è la PR_AUC
                average_precision = average_precision_score(inputs, predictions)

                pred_t = torch.tensor(predictions)
                inpt_t = torch.tensor(inputs, dtype=torch.uint8)
                euclid_dist = (pred_t.ravel() - inpt_t.ravel()).pow(2).sum().sqrt()
                # divido per il numero totale di nodi nel dataloader
                euclid_dist = euclid_dist / (i)  #  NON divido anche per i nodi * self.conf['graph_dataset']['Num_nodes'][0]
                #me lo tengo PER GAFO

            except Exception as e:
                auc = -1
                average_precision = -1
                f1 = -1
                euclid_dist = -1
                print(f"Eccezione data dalle metriche...")
                print(e)
                print(''.join(traceback.format_exception(etype=type(e), value=e, tb=e.__traceback__)))
                # print(f"nan elements: {np.count_nonzero(np.isnan(input_adj_flat))} su totale: {input_adj_flat.size}")
                print(f"nan elements: {np.count_nonzero(np.isnan(predictions))} su totale: {predictions.size}")

                # print(f"auc_{i}: {auc}", end=', ')

            metriche = Metrics(auc=auc, pr_auc=average_precision, euclid=euclid_dist)
            # f1_score=single_f1_score, soglia=best_threshold,¨

        return metriche


    def produce_traning_snapshot(self, epoch, parallel, parallel_processes_save_images, **kwargs):
        if self.conf['training'].get('calculate_metrics'):
            #metric_object = self.calc_metric(self.dataset.all_data_loader)
            metric_object_train = self.calc_metric(self.dataset.train_loader)
            metric_object_test = self.calc_metric(self.dataset.test_loader)
            self.metric_obj_list_train.append(metric_object_train)
            self.metric_obj_list_test.append(metric_object_test)

        emb_pergraph_train = self.get_embedding_MLP(self.dataset.train_loader)
        emb_pergraph_test = self.get_embedding_MLP(self.dataset.test_loader)


        if parallel:
            p = multiprocessing.Process(target=self.save_image_at_epoch, #args=(epoch, last_epoch, self.run_path),
                                         kwargs={**{
                                            "emb_pergraph_train":emb_pergraph_train,
                                            "emb_pergraph_test":emb_pergraph_test,
                                            "epoch":epoch,
                                            "path":self.run_path}, **kwargs})
            p.start()
            parallel_processes_save_images.append(p)
        else:
            self.save_image_at_epoch(epoch=epoch,
                                     emb_pergraph_train=emb_pergraph_train,
                                     emb_pergraph_test=emb_pergraph_test,
                                     path=self.run_path, **kwargs)
        return metric_object_train, metric_object_test

    def get_embedding_MLP(self, loader):
        self.model.eval()
        embeddings_per_graph = []
        with torch.no_grad():
            for data in loader:
                batch_data = data[0]  # TODO: che shape avrà questo batch?
                output = self.model(batch_data)
                # Embedding-Per-Graph ma quì contiene solo le matrici (al momento)
                for i, input_m in enumerate(batch_data):
                    output_m = output[i]
                    epg = Embedding_AEMLP_per_graph(input_m, output_m)
                    embeddings_per_graph.append(epg)

        return embeddings_per_graph


    def save_image_at_epoch(self, epoch, model_weights_and_labels=None,
                            emb_pergraph_train=None, emb_pergraph_test=None,
                            path=".", **kwargs):


        data = DataAEMLP2Plot(config_class=self.config_class,
                                emb_pergraph_train=emb_pergraph_train,
                                emb_pergraph_test=emb_pergraph_test)
        path = Path(path)
        try:
            metric_epoch_list = self.epochs_list[:np.where(self.epochs_list == epoch)[0][0]+1]
            if not kwargs.get("unico_plot"):
                testll = self.test_loss_list
                trainll = self.train_loss_list
                all_range_epochs_list = range(epoch+1)
                metric_obj_list_train = self.metric_obj_list_train
                metric_obj_list_test = self.metric_obj_list_test

                if kwargs.get("last_plot"):
                    all_range_epochs_list = range(epoch)
                    metric_epoch_list = [0, self.epochs_list[-1]]
            else:  # quando voglio fare un solo plot, cioè quando carico un modello preaddestrato
                testll = [0]
                trainll = [0]
                all_range_epochs_list = [0]
                #metric_epoch_list = [1]
                metric_obj_list_train = [self.metric_obj_list_train[0]]
                metric_obj_list_test = [self.metric_obj_list_test[0]]
                total_node_emb_dim = [self.total_node_emb_dim[0]]
                total_graph_emb_dim = [self.total_graph_emb_dim[0]]


            fig = plot_metrics(data, self.embedding_dimension,
                               testll, all_range_epochs_list,
                               sequential_colors=True,
                               showplot=False, last_epoch=self.epochs_list[-1], metric_name=self.name_of_metric,
                               long_string_experiment=self.config_class.long_string_experiment,
                               metric_obj_list_train=metric_obj_list_train,
                               metric_obj_list_test=metric_obj_list_test,
                               train_loss_list=trainll,
                               x_axis_log=self.conf.get("plot").get("x_axis_log"),
                               metric_epoch_list=metric_epoch_list,
                               **kwargs)

            if kwargs.get("notsave"):
                fig.show()
            else:
                file_name = path / f"_epoch{epoch}"
                plt.savefig(file_name)
                #fig.clf()
                #plt.cla()
                #plt.clf()
                if not self.conf['training']['every_epoch_embedding']: # quì ho soltanto una immagine all'inizio e una allafine
                    # salvo solo lultima immagine e la rinomino:
                    os.rename(f"{file_name}.png", path / f"{self.unique_train_name}.png")

        except Exception as e:
            print(f"Immagine {epoch} non completata")
            print(e)
            # traceback.print_stack()
            # print(traceback.format_exc())
            print(''.join(traceback.format_exception(etype=type(e), value=e, tb=e.__traceback__)))

        return

    def save_all_animations(self, animation_files, epochs_list):
        nomefile = self.run_path / str(self.unique_train_name)

        self.save_gif_snapshots(animation_files, nomefile)

        new_files = self.save_mp4_snapshots(animation_files, epochs_list, nomefile)

        # ora cancello le singole snapshots
        self.delete_list_of_files(new_files)



