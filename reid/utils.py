import tqdm
import numpy as np
import torch
import os
import errno
import json
import yaml
from os import path as osp
import shutil
from glob import glob


def mkdir_if_missing(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def read_json(fpath):
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj

def read_yaml(fpath):
    with open(fpath, 'r') as f:
        obj = yaml.full_load(f)
    return obj

def write_yaml(obj, fpth):
    with open(fpth, 'w') as f:
        yaml.dump(obj, f)

def write_json(obj, fpath):
    mkdir_if_missing(osp.dirname(fpath))
    with open(fpath, 'w') as f:
        json.dump(obj, f, indent=4, separators=(',', ': '))

def sample_gallery(labels, tracklets, gallery_sizes, imgs_per_tracklet=None, n=1):
    assert len(labels) == len(tracklets)
    for gallery_size in gallery_sizes:
        assert len(labels) > gallery_size
    
    #print(f'{labels=}')
    unique_labels = np.unique(labels)
    unique_tracklets = {pid:np.unique(tracklets[labels == pid]) for pid in unique_labels}
    
    masks = np.zeros((len(gallery_sizes), n, 2, len(labels)), dtype=bool)
    
    for i, gallery_size in enumerate(gallery_sizes):
        for j in range(n):
            gallery_images = []
            gallery_tracklets = []
            for pid in unique_labels:
                for tracklet in np.random.choice(unique_tracklets[pid], gallery_size, replace=False):
                    gallery_tracklets.append(tracklet)
                    indices = np.arange(len(labels))[tracklets == tracklet]
                    if imgs_per_tracklet is not None and len(indices) > imgs_per_tracklet > 0:
                        indices = np.random.choice(indices, imgs_per_tracklet, replace=False)
                    gallery_images.extend(indices)
                
            masks[i, j, 0][gallery_images] = True
            masks[i, j, 1] = np.logical_not(np.isin(tracklets, gallery_tracklets))
    
    return masks
    
def has_func(obj, name):
    attr = getattr(obj, name, None)
    return callable(attr)
    

def extract_output(model, loader, target_classifier, to_cpu=True, input_key='imgs'):
    model.eval()
    embeddings = []
    classifications = []
    allpids = []
    allcams = []
    alltracklets = []
    allframes = []
    
    for imgs, fnames, pids, (cams, tracklets, frames) in tqdm.tqdm(loader):
        output = model({
            input_key: imgs
        }, target_classifier)
        
        embeddings.append(output['embedding'].detach().cpu() if to_cpu else output['embeddings'].detach())
        outt = output['classifications'].detach()
        classifications.append(outt.cpu() if to_cpu else outt)
        
        allpids.extend(pids)
        allcams.extend(cams)
        alltracklets.extend(tracklets)
        allframes.extend(frames)

    return (
        torch.cat(embeddings),
        torch.cat(classifications),
        np.array(allpids),
        np.array(allcams),
        np.array(alltracklets),
        np.array(allframes)
    )

def load_embedding_sequence(path, dataset, to_cpu=True, embedding_dim=2048):
    if to_cpu:
        labels = torch.load(osp.join(path, f'{dataset}_labels.pth.tar'), map_location=torch.device('cpu'))
        cams = torch.load(osp.join(path, f'{dataset}_cams.pth.tar'), map_location=torch.device('cpu'))
        tracklets = torch.load(osp.join(path, f'{dataset}_tracklets.pth.tar'), map_location=torch.device('cpu'))
        frames = torch.load(osp.join(path, f'{dataset}_frames.pth.tar'), map_location=torch.device('cpu'))
    else:
        labels = torch.load(osp.join(path, f'{dataset}_labels.pth.tar'))
        cams = torch.load(osp.join(path, f'{dataset}_cams.pth.tar'))
        tracklets = torch.load(osp.join(path, f'{dataset}_tracklets.pth.tar'))
        frames = torch.load(osp.join(path, f'{dataset}_frames.pth.tar'))

    identifiers = torch.stack(tuple(map(torch.from_numpy, [labels, cams, tracklets, frames])))
    
    sequence_files = sorted(glob(osp.join(path, f'{dataset}_??_????_features.pth.tar')))
    sequence = torch.empty((len(sequence_files), len(labels), embedding_dim))
    for i, file in enumerate(sequence_files):
        sequence[i] = torch.load(file) if not to_cpu else torch.load(file, map_location=torch.device('cpu'))

    return sequence, identifiers

def load_checkpoint(fpath, cpu=False):
    """Load a saved model checkpoint into memory.

    Keyword arguments:
    fpath -- the path to the .pth.tar file containing the state dict information
    load_classifier -- if set to True will load the model as it was saved, otherwise the final classification layer (if it exists) will be discarded (default False)
    cpu -- if set to True the model will not be loaded onto an available gpu, but into main memory instead (default False)

    Returns:
    checkpoint -- the state_dict saved in the file pointed to by the given *fpath*
    """
    if osp.isfile(fpath):
        if cpu:
            checkpoint = torch.load(fpath, map_location=torch.device('cpu'))['state_dict']
        else:
            checkpoint = torch.load(fpath)['state_dict']
        print("=> Loaded checkpoint '{}'".format(fpath))
        return checkpoint
    else:
            raise ValueError("=> No checkpoint found at '{}'".format(fpath))

class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count