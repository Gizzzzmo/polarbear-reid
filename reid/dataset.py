from typing import Optional, Tuple, Union, Sequence
import numpy as np

from .utils import read_json
from .metric_logging import MetricsLogger
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
        validation_fold: int = 1,
        invert_split: bool = False,
        galleries_per_fold: Optional[int] = None,
        label_offset: int = 0,
        classifier_idx: int = 0,
        name: Optional[str] = None
        ):
        
        self.root = root
        self.label_offset = label_offset
        self.split_id = split_id
        self.invert_split = invert_split
        self.validation_fold = validation_fold
        self.galleries_per_fold = galleries_per_fold
        self.meta = None
        self.split = None
        self.train, self.val = [], []
        self.train_loader, self.val_loader = None, None
        self.qg_masks = None
        self.num_train_ids, self.num_val_ids = 0, 0
        self.classifer_idx = classifier_idx
        self.logger: Optional[MetricsLogger] = None
        self.name = root if name is None else name

    @property
    def loaded(self):
        return self.meta is not None

    @property
    def images_dir(self):
        return osp.join(self.root, 'images')
    
    def print_stats(self):
        assert self.loaded
        print(
            f"Dataset {self.name} opened with split {self.split_id}. "
            f"The split has {len(self.split)} folds."
        )
        if self.invert_split:
            print(f"Folds {', '.join([fold for fold in self.split if fold != self.validation_fold])} are loaded for validation.")
            print(f"Fold {self.validation_fold} is loaded for training.")
        else:
            print(f"Fold {self.validation_fold} is loaded for validation.")
            print(f"Folds {', '.join([fold for fold in self.split if fold != self.validation_fold])} are loaded for training.")
            
        reid_ids = 0
        total_ids = self.num_train_ids + reid_ids
        cameras = self.meta['num_cameras']
        
        train_tracklets_by_id = np.zeros(self.num_train_ids)
        train_tracklets_by_camera = np.zeros(cameras)
        train_frames_by_id = np.zeros(self.num_train_ids)
        train_frames_by_camera = np.zeros(cameras)
        train_frames_by_tracklet = np.zeros(len(self.train_tracklets) + len(self.val_tracklets))
        checked_train_tids = set()
        for _, id, (camid, tid, frame) in self.train:
            train_frames_by_id[id] += 1
            train_frames_by_camera[camid] += 1
            train_frames_by_tracklet[tid] += 1
            if tid not in checked_train_tids:
                train_tracklets_by_id[id] +=1
                train_tracklets_by_camera[camid] +=1
                checked_train_tids.add(tid)
                
        val_tracklets_by_id = np.zeros(self.num_train_ids)
        val_tracklets_by_camera = np.zeros(cameras)
        val_frames_by_id = np.zeros(self.num_train_ids)
        val_frames_by_camera = np.zeros(cameras)
        val_frames_by_tracklet = np.zeros(len(self.train_tracklets) + len(self.val_tracklets))
        checked_tids = set()
        for _, id, (camid, tid, frame) in self.val:
            val_frames_by_id[id] += 1
            val_frames_by_camera[camid] += 1
            val_frames_by_tracklet[tid] += 1
            if tid not in checked_tids:
                val_tracklets_by_id[id] +=1
                val_tracklets_by_camera[camid] +=1
                checked_tids.add(tid)
                if tid in checked_train_tids:
                    print("warning: overlap")
        
        
        train_tracklets = len(self.train_tracklets)
        avg_train_tr_id = np.mean(train_tracklets_by_id)
        std_train_tr_id = np.std(train_tracklets_by_id)
        avg_train_tr_ca = np.mean(train_tracklets_by_camera)
        std_train_tr_ca = np.std(train_tracklets_by_camera)
        idval_tracklets = len(self.val_tracklets)
        avg_val_tr_id = np.mean(val_tracklets_by_id)
        std_val_tr_id = np.std(val_tracklets_by_id)
        avg_val_tr_ca = np.mean(val_tracklets_by_camera)
        std_val_tr_ca = np.std(val_tracklets_by_camera)
        id_tracklets = train_tracklets + idval_tracklets
        avg_tr_id = np.mean(val_tracklets_by_id + train_tracklets_by_id)
        std_tr_id = np.std(val_tracklets_by_id + train_tracklets_by_id)
        avg_tr_ca = np.mean(val_tracklets_by_camera + train_tracklets_by_camera)
        std_tr_ca = np.std(val_tracklets_by_camera + train_tracklets_by_camera)
        reid_tracklets = 0
        total_tracklets = id_tracklets + reid_tracklets
        
        train_frames = len(self.train)
        avg_train_fr_id = np.mean(train_frames_by_id)
        std_train_fr_id = np.std(train_frames_by_id)
        avg_train_fr_tr = np.mean(train_frames_by_tracklet[train_frames_by_tracklet != 0])
        std_train_fr_tr = np.std(train_frames_by_tracklet[train_frames_by_tracklet != 0])
        avg_train_fr_ca = np.mean(train_frames_by_camera)
        std_train_fr_ca = np.std(train_frames_by_camera)
        
        idval_frames = len(self.val)
        avg_val_fr_id = np.mean(val_frames_by_id)
        std_val_fr_id = np.std(val_frames_by_id)
        avg_val_fr_tr = np.mean(val_frames_by_tracklet[val_frames_by_tracklet != 0])
        std_val_fr_tr = np.std(val_frames_by_tracklet[val_frames_by_tracklet != 0])
        avg_val_fr_ca = np.mean(val_frames_by_camera)
        std_val_fr_ca = np.std(val_frames_by_camera)
        id_frames = train_frames + idval_frames
        
        avg_fr_id = np.mean(val_frames_by_id + train_frames_by_id)
        std_fr_id = np.std(val_frames_by_id + train_frames_by_id)
        avg_fr_tr = np.mean(val_frames_by_tracklet + train_frames_by_tracklet)
        std_fr_tr = np.std(val_frames_by_tracklet + train_frames_by_tracklet)
        avg_fr_ca = np.mean(val_frames_by_camera + train_frames_by_camera)
        std_fr_ca = np.std(val_frames_by_camera + train_frames_by_camera)
        reid_frames = 0
        total_frames = id_frames + reid_frames
        
        
        print(
            f"                     | {'identification'                                                      :<40}|| {'reid val'                                  :<26}|| {'combined'     :<12}\n"
            f" identities          | {self.num_train_ids                                                    :<40}|| {reid_ids                                    :<26}|| {total_ids      :<12}\n"
            f" cameras             | {cameras                                                               :<40}|| {cameras                                     :<26}|| {cameras        :<12}\n"
            f"----------------------------------------------------------------------------------------------------\n"
            f"                     | {'train'           :<12}| {'val'             :<12}| {'combined'        :<12}|| {'avg query'       :<12}| {'avg gallery'     :<12}||\n"
            f" tracklets           | {''                :<12}| {''                :<12}| {''                :<12}|| {''                :<12}| {''                :<12}||\n"
            f"     total           | {train_tracklets   :<12}| {idval_tracklets   :<12}| {id_tracklets      :<12}|| {reid_tracklets :<12.2f}| {reid_tracklets :<12.2f}|| {total_tracklets:<12}\n"
            f"     by id           | {avg_train_tr_id:<12.2f}| {avg_val_tr_id  :<12.2f}| {avg_tr_id      :<12.2f}|| {reid_tracklets :<12.2f}| {reid_tracklets :<12.2f}|| {avg_tr_id:<12.2f}\n"
            f"         +/-         | {std_train_tr_id:<12.2f}| {std_val_tr_id  :<12.2f}| {std_tr_id      :<12.2f}|| {reid_tracklets :<12.2f}| {reid_tracklets :<12.2f}|| {std_tr_id:<12.2f}\n"
            f"     by camera       | {avg_train_tr_ca:<12.2f}| {avg_val_tr_ca  :<12.2f}| {avg_tr_ca      :<12.2f}|| {reid_tracklets :<12.2f}| {reid_tracklets :<12.2f}|| {avg_tr_ca:<12.2f}\n"
            f"         +/-         | {std_train_tr_ca:<12.2f}| {std_val_tr_ca  :<12.2f}| {std_tr_ca      :<12.2f}|| {reid_tracklets :<12.2f}| {reid_tracklets :<12.2f}|| {std_tr_ca:<12.2f}"
        )
        print(
            f" frames              | {''                :<12}| {''                :<12}| {''                :<12}|| {''                :<12}| {''                :<12}||\n"
            f"     total           | {train_frames      :<12}| {idval_frames      :<12}| {id_frames         :<12}|| {reid_frames    :<12.2f}| {reid_frames    :<12.2f}|| {total_frames:<12}\n"
            f"     by id           | {avg_train_fr_id:<12.2f}| {avg_val_fr_id  :<12.2f}| {avg_fr_id      :<12.2f}|| {reid_tracklets :<12.2f}| {reid_tracklets :<12.2f}|| {avg_fr_id:<12.2f}\n"
            f"         +/-         | {std_train_fr_id:<12.2f}| {std_val_fr_id  :<12.2f}| {std_fr_id      :<12.2f}|| {reid_tracklets :<12.2f}| {reid_tracklets :<12.2f}|| {std_fr_id:<12.2f}"
        )
        print(
            f"     by camera       | {avg_train_fr_ca:<12.2f}| {avg_val_fr_ca  :<12.2f}| {avg_fr_ca      :<12.2f}|| {reid_tracklets :<12.2f}| {reid_tracklets :<12.2f}|| {avg_tr_ca:<12.2f}\n"
            f"         +/-         | {std_train_fr_ca:<12.2f}| {std_val_fr_ca  :<12.2f}| {std_fr_ca      :<12.2f}|| {reid_tracklets :<12.2f}| {reid_tracklets :<12.2f}|| {std_tr_ca:<12.2f}\n"
            f"     by tracklet     | {avg_train_fr_tr:<12.2f}| {avg_val_fr_tr  :<12.2f}| {avg_fr_tr      :<12.2f}|| {reid_tracklets :<12.2f}| {reid_tracklets :<12.2f}|| {avg_fr_tr:<12.2f}\n"
            f"         +/-         | {std_train_fr_tr:<12.2f}| {std_val_fr_tr  :<12.2f}| {std_fr_tr      :<12.2f}|| {reid_tracklets :<12.2f}| {reid_tracklets :<12.2f}|| {std_fr_tr:<12.2f}\n"
        )
    
    def load(self, relabel=False):
        splits = read_json(osp.join(self.root, 'splits.json'))
        if self.split_id >= len(splits):
            raise ValueError('split_id exceeds the number of splits')
        self.split = splits[self.split_id]
        
        self.val_tracklets = []
        self.train_tracklets = []
        self.gallery_sizes = None
        galleries_per_size = None
        
        galleries_key = 'galleries_complement' if self.invert_split else 'galleries'
        
        for validation_fold in self.split:
            if validation_fold == self.validation_fold:
                validation_fold = self.split[validation_fold]
                self.val_tracklets.extend(validation_fold['set'])
                if self.gallery_sizes == None:
                    self.gallery_sizes = list(validation_fold[galleries_key].keys())
                else:
                    for size_1, size_2 in zip(self.gallery_sizes, validation_fold[galleries_key].keys()):
                        assert size_1 == size_2
                for size in self.gallery_sizes:
                    if galleries_per_size == None:
                        galleries_per_size = len(validation_fold[galleries_key][size])
                    else:
                        assert galleries_per_size == len(validation_fold[galleries_key][size])
            else: 
                validation_fold = self.split[validation_fold]
                self.train_tracklets.extend(validation_fold['set'])
                
        if self.invert_split:
            temp = self.train_tracklets
            self.train_tracklets = self.val_tracklets
            self.val_tracklets = temp

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
                num_galleries,
                2, len(self.val)
            ),
            dtype=np.bool
        )
        
        if self.validation_fold not in self.split:
            raise ValueError('given validation_fold is not in the split')
        
        for gallery_size in self.gallery_sizes:
            for i, gallery in enumerate(self.split[self.validation_fold][galleries_key][gallery_size]):
                if i >= num_galleries:
                    break
                
                self.qg_masks[
                    self.gallery_sizes.index(gallery_size),
                    i,
                    0,
                ] = np.isin(val_tracklets_per_image, gallery)
        
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
        validation_fold: int = 1,
        invert_split: bool = False,
        galleries_per_fold: Optional[int] = None,
        label_offset: int = 0,
        classifier_idx: int = 0,
        name: str = None,
        relabel: bool = True,
        ):
        super().__init__(root, split_id, validation_fold, invert_split, galleries_per_fold, label_offset, classifier_idx, name)
        
        self.load(relabel=relabel)
        
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