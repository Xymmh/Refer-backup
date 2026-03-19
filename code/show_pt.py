import torch

data = torch.load("data/yelp/data_trn.pt", weights_only=False)

print(data)