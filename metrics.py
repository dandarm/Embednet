import numpy as np
from torchmetrics import Metric
import torch
from TorchPCA import PCA

class ExplainedVarianceMetric(Metric):
    def __init__(self, dimension):
        super().__init__()
        #self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("ExpVar", default=torch.tensor([0] * dimension, dtype=torch.float), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        #preds, target = self._input_format(preds, target)
        #assert preds.shape == target.shape
        #print(type(preds), preds.shape)
        #self.correct += torch.sum(preds == target)
        self.total += target.numel()
        #print(f'preds shape: {preds.shape}')
        obj = PCA(preds)
        var_exp, _, _ = obj.get_ex_var()
        print(f'var_exp: {var_exp}')
        var_exp = torch.as_tensor(np.array(var_exp))
        #print(f'{self.ExpVar.shape}, {var_exp.shape}')
        self.ExpVar = var_exp  #[0]

    def compute(self):
        #return self.correct.float() / self.total
        #print(self.ExpVar.shape, self.total.shape)
        return self.ExpVar# / self.total