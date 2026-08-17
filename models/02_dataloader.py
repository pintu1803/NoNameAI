from torch.utils.data import TensorDataset, DataLoader
import torch


"""
batch_size=n -> each batch consists of n samples
shuffle=True -> indices of the samples get shuffled, so batches are shuffled (not in order)
drop_last=True -> drop the last incomplete batch (partial batch)

if sample shape = (d1, d2)
then batch shape = (batch_size, d1, d2)

Generally, labels are scalar, but after batch creation,
label becomes a vector (group of scalars)
"""

X = torch.arange(20, dtype=torch.float32).reshape(10, -1)

Y = torch.arange(10)

dataset = TensorDataset(X, Y)

print("\nDataset prepared using samples-X and labels-Y")
print("Size of dataset : ", len(dataset))
print("Get index-0 of dataset : ", dataset.__getitem__(0))

loader = DataLoader(dataset=dataset, batch_size=3, shuffle=False, drop_last=False)

print("\nLoader prepared from dataset")

for index, (samples, labels) in enumerate(loader):
    print("\nBatch samples-size : ", samples.shape)
    print("Batch labels-size : ", labels.shape)
    print("Batch index : ", index)
    print("Samples : ", samples)
    print("Labels : ", labels)
    print("-"*30)
