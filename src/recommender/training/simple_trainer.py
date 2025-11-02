import tqdm

class SimpleTrainer:

    def __init__(self, num_epochs: int = 20):
        self.num_epochs = num_epochs

    def fit(self, model, data):
        for epoch in tqdm.tqdm(range(self.num_epochs)):
            pass

        return model