from collections import defaultdict

import numpy as np
from torch.utils.data import Sampler

__all__ = ['StratifiedSampler']


class StratifiedSampler(Sampler):
    def __init__(self, labels, num_samples_per_class):
        self.num_samples_per_class = num_samples_per_class
        self.label_to_indices = defaultdict(list)
        for index, label in enumerate(labels):
            self.label_to_indices[label].append(index)
            
    def __iter__(self):
        samples = []
        for indices in self.label_to_indices.values():
            samples.extend(np.random.choice(indices, size=self.num_samples_per_class, replace=True))
        np.random.shuffle(samples)
        return iter(samples)

    def __len__(self):
        return self.num_samples_per_class * len(self.label_to_indices)
