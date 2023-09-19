import numpy as np
import networkx as nx
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader
from Dataset import GeneralDataset


#def get_Dataset_for_ae_mlp_from_super_istance(config_class, general_dataset):

# __init__ first argument, self, holds the new instance that results from calling .__new__()

def __init__(self, config_class, **kwarg):
    super().__init__(**kwarg)

class Dataset_for_ae_mlp(GeneralDataset):
    #def __new__(cls, *args, **kwargs):
        #super_instance = kwargs.get('dataset')
        #instance = super().__new__(cls, **super_instance.__dict__)

        # aggiungo ulteriori attributi che avrei inserito in __init__  ??????



        #return super_instance

    def __init__(self, config_class, super_instance, **kwarg):
        """
        :param config_class:
        :param kwarg: è il __dict__ di GeneralDataset
        """
        super().__init__(**super_instance.__dict__)

        self.config_class = config_class
        self.conf = config_class.conf
        self.percentage_train = self.conf['training']['percentage_train']
        self.prc_test = round(1 - self.percentage_train, 2)
        self.bs = self.conf['training']['batch_size']
        self.device = config_class.device

        self.verbose = kwarg.get('verbose')

        self.train_loader = None
        self.test_loader = None

    # @classmethod
    # def from_super_instance(cls, config_class, super_instance, verbose):
    #     return cls(config_class, **super_instance.__dict__, verbose=verbose)

    def prepare(self, shuffle=True, parallel=False):
        dataset_numpy = []
        for graph_nx in self.dataset_list:
            array = nx.to_numpy_array(graph_nx)
            dataset_numpy.append(array)

        dataset_numpy = np.array(dataset_numpy)
        train_data, test_data = train_test_split(dataset_numpy, test_size=self.prc_test, random_state=42)
        train_data, test_data = torch.Tensor(train_data), torch.Tensor(test_data)
        #print(f" %test {self.prc_test}\t\t dataset intero  {dataset_numpy.shape}")

        # sposto tutti i dati prima, se ho abbastanza memoria nella GPU per contenere tutti i dati
        train_data = train_data.to(self.device)
        test_data = test_data.to(self.device)

        # Reshape dei dati bidimensionali in un vettore
        train_data = train_data.reshape(train_data.shape[0], -1)
        #print(f"train_data shape  {train_data.shape}")
        train_dataset = TensorDataset(train_data)
        self.train_loader = DataLoader(train_dataset, batch_size=self.bs, shuffle=True)  # , pin_memory=True)

        test_data = test_data.reshape(test_data.shape[0], -1)
        print(f"test_data shape  {test_data.shape}")
        test_dataset = TensorDataset(test_data)
        self.test_loader = DataLoader(test_dataset, batch_size=self.bs, shuffle=False)  # , pin_memory=True)

