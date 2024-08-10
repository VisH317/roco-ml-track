import torch
from torch import Tensor

class KNN:
    def __init__(self, k: int, data_points: list[tuple[Tensor, int]]):
        # data matrix, take the list of features for each data point and convert to one big tensor
        self.k = k
        self.data = torch.stack([data[0] for data in data_points], dim=0).unsqueeze(-1)
        self.classes = torch.tensor([data[1] for data in data_points], dtype=torch.int64)

    def predict(self, point: Tensor):
        distances = torch.dist(point.unsqueeze(0), self.data)
        top_dist = torch.topk(distances.squeeze(), self.k).indices.detach().tolist()

        classes = self.classes[top_dist]

        return torch.mode(classes).values



    