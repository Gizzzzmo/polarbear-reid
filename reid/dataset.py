from typing import Optional, Tuple, Union, Sequence
import numpy as np

from .utils import read_json
from os import path as osp
from PIL import Image

def pluck(tracklets, indices, relabel=False, pid_remap=None, label_offset=0):
    ret = []
    if relabel:
        if pid_remap is None:
            pid_remap = dict()
    else:
        pid_remap = None
    for tid in indices:
        tid_images = tracklets[tid]
        for fname in tid_images:
            name = osp.splitext(fname)[0]
            pid, camid, tracklet, frame = map(int, name.split('_'))
            
            if relabel:
                if pid not in pid_remap:
                    pid_remap[pid] = len(pid_remap)
                ret.append((fname, pid_remap[pid] + label_offset, (camid, tid, frame)))
            else:
                ret.append((fname, pid + label_offset, (camid, tid, frame)))
    return ret, pid_remap

class Dataset(object):
    def __init__(
        self,
        root: str,
        split_id: int = 0,
        validation_folds: Union[Union[int, str], Sequence[Union[int, str]]] = 1,
        galleries_per_fold: Optional[int] = None,
        label_offset: int = 0,
        classifier_idx: int = 0,
        name: Optional[str] = None
        ):
        
        self.root = root
        self.label_offset = label_offset
        self.split_id = split_id
        self.validation_folds = (validation_folds,) if not isinstance(validation_folds, Sequence) else validation_folds
        self.validation_folds = tuple(str(fold) for fold in self.validation_folds)
        self.galleries_per_fold = galleries_per_fold
        self.meta = None
        self.split = None
        self.train, self.val = [], []
        self.train_loader, self.val_loader = None, None
        self.qg_masks = None
        self.num_train_ids, self.num_val_ids = 0, 0
        self.classifer_idx = classifier_idx
        self.logger = None
        self.name = root if name is None else name

    @property
    def loaded(self):
        return self.meta is not None

    @property
    def images_dir(self):
        return osp.join(self.root, 'images')
    
    def load(self, relabel=False):
        splits = read_json(osp.join(self.root, 'splits.json'))
        if self.split_id >= len(splits):
            raise ValueError('split_id exceeds the number of splits')
        self.split = splits[self.split_id]
        
        self.val_tracklets = []
        self.train_tracklets = []
        self.gallery_sizes = None
        galleries_per_size = None
        
        for fold in self.split:
            if fold in self.validation_folds:
                fold = self.split[fold]
                self.val_tracklets.extend(fold['set'])
                if self.gallery_sizes == None:
                    self.gallery_sizes = list(fold['galleries'].keys())
                else:
                    for size_1, size_2 in zip(self.gallery_sizes, fold['galleries'].keys()):
                        assert size_1 == size_2
                for size in self.gallery_sizes:
                    if galleries_per_size == None:
                        galleries_per_size = len(fold['galleries'][size])
                    else:
                        assert galleries_per_size == len(fold['galleries'][size])
            else: 
                fold = self.split[fold]
                self.train_tracklets.extend(fold['set'])

        self.meta = read_json(osp.join(self.root, 'meta.json'))
        tracklet_files = self.meta['tracklets']
        self.train, pid_map = pluck(tracklet_files, self.train_tracklets, relabel=relabel, label_offset=self.label_offset)
        self.val, pid_map = pluck(tracklet_files, self.val_tracklets, relabel=relabel, label_offset=self.label_offset, pid_remap=pid_map)
        self.pid_map = pid_map
        
        self.num_val_ids = len(np.unique([pid for _, pid, _ in self.val]))
        self.num_train_ids = len(np.unique([pid for _, pid, _ in self.train]))

        num_galleries = min(self.galleries_per_fold, galleries_per_size) if self.galleries_per_fold is not None else galleries_per_size
        val_tracklets_per_image = [tracklet for (_, _, (_, tracklet, _)) in self.val]
        
        self.qg_masks = np.zeros(
            (
                len(self.gallery_sizes),
                num_galleries * len(self.validation_folds),
                2, len(self.val)
            ),
            dtype=np.bool
        )
        
        offset = 0
        for fold in self.validation_folds:
            if fold not in self.split:
                raise ValueError('validation_folds contains a fold that is not in the split')
            
            for gallery_size in self.gallery_sizes:
                for i, gallery in enumerate(self.split[fold]['galleries'][gallery_size]):
                    if i >= num_galleries:
                        break
                    
                    self.qg_masks[
                        self.gallery_sizes.index(gallery_size),
                        i+offset,
                        0,
                    ] = np.isin(val_tracklets_per_image, gallery)
            
            offset += num_galleries
        
        self.qg_masks[:, :, 1, :] = np.logical_not(self.qg_masks[:, :, 0, :])
        
    
    def _check_integrity(self):
        return osp.isdir(osp.join(self.root, 'images')) and \
               osp.isfile(osp.join(self.root, 'meta.json')) and \
               osp.isfile(osp.join(self.root, 'splits.json'))

class Preprocessor(object):
    def __init__(self, dataset, root=None, transform=None):
        super(Preprocessor, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        if isinstance(indices, (tuple, list)):
            return [self._get_single_item(index) for index in indices]
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        fname, pid, camid = self.dataset[index]
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)
        img = Image.open(fpath).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, fname, pid, camid

class PolarBears(Dataset):
    def __init__(
        self,
        root: str,
        split_id: int = 0,
        validation_folds: Union[Union[int, str], Tuple[Union[int, str]]] = 1,
        galleries_per_fold: Optional[int] = None,
        label_offset: int = 0,
        classifier_idx: int = 0,
        name: str = None,
        ):
        super().__init__(root, split_id, validation_folds, galleries_per_fold, label_offset, classifier_idx, name)
        
        self.load(relabel=True)
        
__factory = {
    '1_NBapril_back': PolarBears, 
    '2_NBapril_pad': PolarBears, 
    '3_berlin_back': PolarBears,
    '4_berlin_pad': PolarBears,
    '5_vienna_back': PolarBears,
    '6_vienna_pad': PolarBears, 
    '7_mulhouse_back': PolarBears,
    '8_mulhouse_pad': PolarBears
}

def names():
    return sorted(__factory.keys())

def create(name, root, *args, **kwargs):
    """
    Create a dataset instance.

    Parameters
    ----------
    name : str
        The dataset name. Can be one of 'viper', 'cuhk01', 'cuhk03',
        'market1501', and 'dukemtmc'.
    root : str
        The path to the dataset directory.
    split_id : int, optional
        The index of data split. Default: 0
    """
    if name not in __factory:
        raise KeyError("Unknown dataset:", name)
    return __factory[name](root, *args, name=name, **kwargs)