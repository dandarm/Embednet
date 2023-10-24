from plot_funcs import DataAutoenc2Plot
import numpy as np
import matplotlib.pyplot as plt

class DataAEMLP2Plot(DataAutoenc2Plot):
    def __init__(self, emb_pergraph_train, emb_pergraph_test,  config_class=None, sequential_colors=False, **kwargs):
        self.fullMLP = True
        self.emb_pergraph_train = emb_pergraph_train
        self.emb_pergraph_test = emb_pergraph_test
        self.config_class = config_class
        #self.wrapper_obj = wrapper_obj

    def set_data(self, type):
        if type == 'adj_entries':
            pij_void = np.array([0])
            zipped_list = [(emb.input_adj_mat, emb.output_adj_mat) for emb in self.emb_pergraph_train]
            inputs_train, outputs_train = zip(*zipped_list)
            self.array2plot_train = ( np.array(inputs_train).ravel(), pij_void, np.array(outputs_train).ravel() )

            zipped_list = [(emb.input_adj_mat, emb.output_adj_mat) for emb in self.emb_pergraph_test]
            inputs_test, outputs_test = zip(*zipped_list)
            self.array2plot_test = (np.array(inputs_test).ravel(), pij_void, np.array(outputs_test).ravel())


    def plot_output_degree_sequence(self, ax):
        # voglio plottare anche altre cose tipo la sequenza di grado
        # prendo la seq grado delloutput
        pred_degrees_train = np.array([g.out_degree_seq for g in self.emb_pergraph_train]).ravel().squeeze()
        input_degree_train = np.array([g.input_degree_seq for g in self.emb_pergraph_train]).ravel().squeeze()

        pred_degrees_test = np.array([g.out_degree_seq for g in self.emb_pergraph_test]).ravel().squeeze()
        input_degree_test = np.array([g.input_degree_seq for g in self.emb_pergraph_test]).ravel().squeeze()

        #input_degree = np.concatenate((input_degree_train, input_degree_test), axis=0)

        ax.scatter(input_degree_train, input_degree_train, label="Input Train", color='orangered', alpha=1.0)
        ax.scatter(input_degree_test, input_degree_test, label="Input Test", color='crimson', alpha=1.0)

        ax.scatter(input_degree_test, pred_degrees_test, label="Predicted Test", color='mediumblue', alpha=0.3)
        ax.scatter(input_degree_train, pred_degrees_train, label="Predicted Train", color='cornflowerblue', alpha=0.3)


        minimo = min(min(input_degree_train), min(input_degree_test))
        massimo = max(max(input_degree_train), max(input_degree_test))
        ax.set_ylim(minimo-1, massimo+1)
        ax.legend()

    def plot_output_clust_coeff(self, ax):
        pred_cc_train = np.array([g.out_clust_coeff for g in self.emb_pergraph_train]).ravel().squeeze()
        input_cc_train = np.array([g.input_clust_coeff for g in self.emb_pergraph_train]).ravel().squeeze()

        pred_cc_test = np.array([g.out_clust_coeff for g in self.emb_pergraph_test]).ravel().squeeze()
        input_cc_test = np.array([g.input_clust_coeff for g in self.emb_pergraph_test]).ravel().squeeze()

        #input_degree = np.concatenate((input_degree_train, input_degree_test), axis=0)

        ax.scatter(input_cc_train, input_cc_train, label="Input Train", color='orangered', alpha=1.0)
        ax.scatter(input_cc_test, input_cc_test, label="Input Test", color='crimson', alpha=1.0)

        ax.scatter(input_cc_test, pred_cc_test, label="Predicted Test", color='mediumblue', alpha=0.3)
        ax.scatter(input_cc_train, pred_cc_train, label="Predicted Train", color='cornflowerblue', alpha=0.3)


        #minimo = min(min(input_cc_train), min(input_cc_test))
        #massimo = max(max(input_cc_train), max(input_cc_test))
        ax.set_ylim(-0.01, 1.01)
        ax.set_xlim(-0.01, 1.01)
        #ax.set_ylim(minimo-0.01, massimo+0.01)
        ax.legend()
