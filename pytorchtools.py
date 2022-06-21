import numpy as np
import torch

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, initial_delta=0, minvalue=1, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.start_counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.initial_score = None
        self.started = False
        self.initial_delta = initial_delta
        self.minvalue = minvalue  #  soglia sotto la quale la loss è bassa e il training sembra essere andato bene
        
    def __call__(self, val_loss, model=None):

        score = -val_loss
        self.start_counter += 1
        
        if self.initial_score is None:
            self.initial_score = score
        else:
            if self.start_counter > 30 and score > self.initial_score + self.initial_delta:
                self.started = True

        if self.best_score is None:
            self.best_score = score
            return
            #self.save_checkpoint(val_loss, model)

        # SE lo score non è migliorato, e SE è iniziato a scendere e SE il valore è sotto la soglia
        if (score < self.best_score + self.delta) and self.started and val_loss < self.minvalue:
            #if self.counter == 0:
            #    print("Start patience for early stopping")
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            #self.save_checkpoint(val_loss, model)
            #if self.counter > 0:
            #    print("Reset patience")
            self.counter = 0  # resetta e ricomincia a contare


    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss