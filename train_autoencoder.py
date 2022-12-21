from train import Trainer

class Trainer_Autoencoder(Trainer):
    def __init__(self, model, config):
        super().__init__(config)

    def train(self):
        self.model.train()
        running_loss = 0
        for data in self.dataset.train_loader:
            z = self.model.encode(data.x, data.edge_index, data.batch)
            loss = self.model.recon_loss(z, data.pos_edge_label_index)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            running_loss += loss.item()

        return running_loss / self.dataset.train_len

    def test(self, loader):
        self.model.eval()
        running_loss = 0
        for data in self.dataset.test_loader:
            z = self.model.encode(data.x, data.edge_index, data.batch)
            loss = self.model.recon_loss(z, data.pos_edge_label_index)
            running_loss += loss.item()

        return running_loss / self.dataset.test_len

    def accuracy(self, loader):
        return None

    def get_embedding(self, loader):
        self.model.eval()
        embeddings_array = []
        for data in loader:  # Iterate in batches over the training dataset.
            z = self.model.encode(data.x, data.edge_index, data.batch)
            embeddings_array.extend(z)
        return embeddings_array