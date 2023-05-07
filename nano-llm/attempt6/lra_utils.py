from jax import lax, numpy as np
from typing import NamedTuple
from pathlib import Path
from s5.dataloading import Datasets, DataLoader


class LRABatchConfig(NamedTuple):
    block_size: int
    batch_size: int
    s5_dataloaders: DataLoader
    train_size: int
    n_classes_in: int
    n_classes_out: int

    @classmethod
    def from_s5(cls, batch_size: int, cache_path: Path, dataset_name: str, seed: int = 0):
        create_dataset_fn = Datasets[dataset_name]
        trainloader, valloader, testloader, aux_dataloaders, n_classes, seq_len, in_dim, train_size = create_dataset_fn(
            cache_path, seed=seed, bsz=batch_size)
        return cls(block_size=seq_len, batch_size=batch_size,
                   s5_dataloaders={'train': trainloader, 'val': valloader, 'test': testloader}, train_size=train_size,
                     n_classes_in=in_dim, n_classes_out=n_classes)

    def get_dataloader(self, name):
        # name = 'train', 'val', or 'test'
        # output a data_generator function
        def _get_dataloader(loader: DataLoader):
            def data_generator():
                for x, y, l in loader:
                    x = trim_or_pad(np.array(x), self.block_size)
                    yield x, np.array(y), np.array(l['lengths'])
            return data_generator()
        return {k: _get_dataloader(v) for k, v in self.s5_dataloaders.items()}[name]

def trim_or_pad(x, max_length):
    if x.shape[-1] >= max_length:
        return x[..., :max_length]
    else:
        return lax.pad(x, 0, ((0,0,0),(0,max_length-x.shape[-1],0)))

# %%
# lra = LRABatchConfig.from_s5(24, "lra_benchmarks", "listops-classification")
# train_loader = lra.get_dataloader('train')
# for batch in train_loader:
#     print(batch)
#     break
