import torch
import torchvision
import os
import random

root = os.path.dirname(os.path.abspath(__file__))


class FashionMNIST:
    def __init__(self, device, batch_size, max_per_class=100, seed=None, group_size=None):
        print("prepare dataset")
        self.batch_size = batch_size
        self.dataset = torchvision.datasets.FashionMNIST(os.path.join(root, "data"), train=True, download=True, transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]))

        self.label_descriptions = {
            0: 'Top', 1: 'Trouser', 2: 'Pullover', 3: 'Dress', 4: 'Coat', 5: 'Sandal', 6: 'Shirt', 7: 'Sneaker', 8: 'Bag', 9: 'Boot'
        }

        self.total = max_per_class * 10
        self.group_size = group_size if group_size is not None else max_per_class

        indices = list(range(len(self.dataset)))
        random.Random(seed).shuffle(indices)

        class_dict = {}
        for c in range(10):
            class_dict[c] = []

        count = 0
        for i in range(len(indices)):
            label = self.dataset[indices[i]][1]
            if len(class_dict[label]) < max_per_class:
                class_dict[label].append(indices[i])
                count = count + 1
                if count >= self.total:
                    break

        self.pointer = []
        for i in range(max_per_class // self.group_size):
            for c in range(10):
                for g in range(self.group_size):
                    self.pointer.append(class_dict[c].pop())

    def readout(self, label):
        np_label = label.numpy()
        out = []
        for i in range(np_label.shape[0]):
            out.append(self.label_descriptions[np_label[i]])
        return out

    def __len__(self):
        return self.total // self.batch_size

    def __iter__(self):
        self.index_iterator = 0
        return self

    def __next__(self):

        data = []
        labels = []
        for i in range(self.batch_size):
            if self.index_iterator >= self.total:
                raise StopIteration()

            datum, label = self.dataset[self.pointer[self.index_iterator]]
            self.index_iterator = self.index_iterator + 1
            data.append(datum)
            labels.append(label)

        tensor_data = torch.stack(data, dim=0)
        tensor_labels = torch.tensor(labels, dtype=torch.int64)

        return tensor_data, tensor_labels


if __name__ == '__main__':
    print("assert dataset has fixed permutation")

    dtype = torch.float
    device = torch.device("cuda:0")

    dataset = FashionMNIST(device, batch_size=2, max_per_class=4, seed=100, group_size=2)

    for i, (data, label) in enumerate(dataset):
        print(data.shape, label.shape)
        print(dataset.readout(label))
