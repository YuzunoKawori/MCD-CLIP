import numpy as np
import elasticdeform
import torch
import torch.utils
import h5py
datasets = {}
def register_dataset(cls):
    datasets[cls.__name__] = cls
    return cls

@register_dataset
class CheXpertDataset(torch.utils.data.Dataset):
    LABEL_MAP = {99: 99, -1: 1, 0: 0, 1: 1}

    def __init__(self, datafile, task='No Finding', augment=False, dtype=float, dtype_x=None, dtype_y=None, views=['frontal', 'lateral']):
        super().__init__()
        self.datafile = datafile
        self.task = task
        self.augment = augment
        self.dtype_x = dtype_x or dtype
        self.dtype_y = dtype_y or dtype
        self.views = views

        ds = h5py.File(self.datafile, 'r')
        self.scan_ids = [
            scan_id for scan_id in ds['scans']
            if self.LABEL_MAP[ds['scans'][scan_id].attrs['score %s' % self.task]] != 99
        ]
        self.num_scans = len(self.scan_ids)

    def __getitem__(self, i):
        ds = h5py.File(self.datafile, 'r')
        scan = ds['scans'][self.scan_ids[i]]
        x_i = [scan[view]['image'][:] for view in self.views]
        y_i = self.LABEL_MAP[scan.attrs['score %s' % self.task]]
        if self.augment:
            coflip = np.random.randint(4)
            x_i = [self.augment_image(x_i_v, coflip=coflip) for x_i_v in x_i]
        x_i = [torch.tensor(x_i_v[None, :, :], dtype=self.dtype_x) for x_i_v in x_i]

        y_i = torch.tensor(y_i, dtype=self.dtype_y)
        return x_i + [y_i]

    def augment_image(self, x_i, coflip=None):
        if self.augment:
            x_i = x_i.astype(float)
            if 'flip' in self.augment or 'coflip' in self.augment:
                t = coflip if 'coflip' in self.augment else np.random.randint(4)
                if t == 1:
                    x_i = x_i[::-1, :]
                elif t == 2:
                    x_i = x_i[:, ::-1]
                elif t == 3:
                    x_i = x_i[::-1, ::-1]
                    

            if 'elastic' in self.augment:
                t = np.random.randint(2)
                if t == 1:
                    zoom = np.random.uniform(0.9, 1.1)
                    rotate = np.random.uniform(-30, 30)
                    x_i = elasticdeform.deform_random_grid(x_i, sigma=5, points=5, zoom=zoom, rotate=rotate)

            if 'crop20' in self.augment:
                offset_x = np.random.randint(40)
                offset_y = np.random.randint(40)
                x_i = x_i[offset_y:-(40 - offset_y), offset_x:-(40 - offset_x)]

            if 'gaussiannoise' in self.augment:
                x_i = np.random.normal(x_i, 0.01)

            x_i = np.ascontiguousarray(x_i)
        
        return x_i

    def __len__(self):
        return self.num_scans
