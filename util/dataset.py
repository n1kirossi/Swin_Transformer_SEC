import os
import glob
import numpy as np
import random
import torch
import torch.utils.data as data
import h5py
from tqdm import tqdm


class SeismicHDF5Dataset(data.Dataset):
    def __init__(self, path):
        self.path = path
        self.file = None
        with h5py.File(path, 'r') as f:
            self.length = len(f['image'])
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        if self.file is None:
            self.file = h5py.File(self.path, 'r')
        
        image = self.file['image'][idx].astype(np.float32)
        label = self.file['label'][idx].astype(np.float32)

        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).float()

        return image, label


class SeismicPatchDataset(data.Dataset):
    def __init__(self, thinned_dir, clean_dir, patch_size=128, stride=128, flip_prob=0.5, normalize=True):
        self.thinned_files = glob.glob(os.path.join(thinned_dir, '*.npy'))
        self.clean_files = glob.glob(os.path.join(clean_dir, '*.npy'))
        assert all(os.path.basename(a) == os.path.basename(b) for a, b in zip(self.thinned_files, self.clean_files))
        self.patch_size = patch_size
        self.stride = stride
        self.flip_prob = flip_prob
        self.norm = normalize

        self.patches = []
        for file_indx, (th_path, _) in tqdm(enumerate(zip(self.thinned_files, self.clean_files)), desc='Просмотрено файлов для нарезки на патчи'):
            sample = np.load(th_path)
            h, w = sample.shape
            for i in range(0, h - patch_size + 1, stride):
                for j in range(0, w - patch_size + 1, stride):
                    self.patches.append((file_indx, i, j))
        print(f'Всего сгенерировано патчей: {len(self.patches)}')
    
    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, indx):
        file_indx, i, j = self.patches[indx]

        thinned = np.load(self.thinned_files[file_indx]).astype(np.float32)
        clean = np.load(self.clean_files[file_indx]).astype(np.float32)

        if self.norm:
            maximum = max(abs(thinned).max(), abs(clean).max())

            thinned /= maximum
            clean /= maximum
            
            # z-score:
            # th_mean, th_std = thinned.mean(), thinned.std()
            # cl_mean, cl_std = clean.mean(), clean.std()

            # thinned = np.clip((thinned - th_mean) / (5 * (th_std + 1e-10)), -1, 1)
            # clean = np.clip((clean - cl_mean) / (5 * (cl_std + 1e-10)), -1, 1)

            # минимакс
            # max_th, min_th = np.max(thinned), np.min(thinned)
            # max_cl, min_cl = np.max(clean), np.min(clean)
            
            # thinned = 2 * (thinned - min_th) / (max_th - min_th + 1e-7) - 1
            # clean = 2 * (clean - min_cl) / (max_cl - min_cl + 1e-7) - 1
        
        thinned_patch = thinned[i:i + self.patch_size, j: j + self.patch_size]
        clean_patch = clean[i:i + self.patch_size, j: j + self.patch_size]

        if random.random() < self.flip_prob:
            thinned_patch = np.fliplr(thinned_patch).copy()
            clean_patch = np.fliplr(clean_patch).copy()
        
        thinned_patch = torch.from_numpy(thinned_patch).unsqueeze(0).float()
        clean_patch = torch.from_numpy(clean_patch).unsqueeze(0).float()

        return thinned_patch, clean_patch
