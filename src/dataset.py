"""
This dataset is an extended version of the dataset used in the SquiggleNet project, see
https://github.com/welch-lab/SquiggleNet/blob/master/dataset.py
"""

import torch

from torch.utils.data.dataset import Dataset
from torchvision import transforms

from random import shuffle as randFunct
from random import randint
import numpy as np

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, filenames, pos_ids, current_pos_idx, neg_ids, current_neg_idx, transform=False):
        p = torch.load(filenames[0])
        n = torch.load(filenames[1])
        self.data = torch.cat((p, n))

        print(filenames[0])
        print(filenames[1])
        # extract number of reads per class (for labels and ID list iteration)
        self.n_pos_reads = p.shape[0]
        self.n_neg_reads = n.shape[0]
        stride = 1
        winLength = 32
        seqLength = 2000

        # assign labels (plasmids: 0, chromosomes: 1)
        self.labels = torch.cat((torch.zeros(self.n_pos_reads), torch.ones(self.n_neg_reads)))

        self.do_transform = transform
        self.transform = transforms.Compose([
					transforms.Lambda(lambda x: self.startMove_transform(x)),
					transforms.Lambda(lambda x: self.differences_transform(x)),
					transforms.Lambda(lambda x: self.cutToWindows_transform(x, seqLength, stride, winLength)),
					transforms.Lambda(lambda x: self.noise_transform(x)),
				])

        # store read IDs for evaluation of validation data
        # if no read ID file is parsed, no evaluation data is stored during training
        if pos_ids is not None and neg_ids is not None:
            self.ids = pos_ids[current_pos_idx: (current_pos_idx + self.n_pos_reads)] \
                       + neg_ids[current_neg_idx: (current_neg_idx + self.n_neg_reads)]
        else:
            self.ids = None

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        X = self.data[index]
        y = self.labels[index]

        if self.do_transform:
            X = self.transform(X)

        if self.ids is not None:
            read_id = self.ids[index]
            return X, y, read_id
        else:
            return X, y, ""

    def get_n_pos_reads(self):
        return self.n_pos_reads

    def get_n_neg_reads(self):
        return self.n_neg_reads
    
    def differences_transform(self, signal):
        return np.diff(signal)

    def startMove_transform(self, signal):
        startPosModification = randint(0, 999)
        return signal[startPosModification: -1000 + startPosModification]

    def cutToWindows_transform(self, signal, seqLength, stride, winLength):
        splitInput = np.zeros((seqLength, winLength), dtype="float32")
        for i in range(seqLength):
            splitInput[i, :] = signal[(i*stride):(i*stride)+winLength]
        return splitInput

    def noise_transform(self, signal):
        shape = tuple(signal.shape)
        noise = np.random.normal(0,5, size = shape)
        return signal + noise.astype("float32")
