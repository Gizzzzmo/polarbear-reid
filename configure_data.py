import argparse
from glob import glob
import json
from os import path as osp
from shutil import copyfile
import tqdm
import csv
import numpy as np
from reid.utils import mkdir_if_missing


def read_json(fpath):
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj


def write_json(obj, fpath):
    mkdir_if_missing(osp.dirname(fpath))
    with open(fpath, 'w') as f:
        json.dump(obj, f, indent=4, separators=(',', ': '))


datasets = ['1_NBapril_back', '2_NBapril_pad', '3_berlin_back', '4_berlin_pad', '5_vienna_back', '6_vienna_pad', '7_mulhouse_back', '8_mulhouse_pad', '9_all_back', '10_all_pad']
translation = dict()
for entry in datasets:
    ind, zoo, mode = entry.split('_')
    name = '_'.join((zoo, mode))
    translation[ind] = entry
    translation[entry] = entry
    translation[name] = entry

def main(args):
    meta = {'name': args.dataset, 'shot': 'multiple'}

    pid_of_tracklet = dict()
    tracklets = dict()
    cameras = dict()
    pids = dict()
    target_directory = osp.join(args.dataset, 'images')
    mkdir_if_missing(target_directory)
    print(args.exclude_discarded)
    source1 = glob(osp.join('raw', args.dataset, '???', '*.jpg'))
    source2 = [] if args.exclude_discarded else glob(osp.join('raw', 'Aussortiert', args.dataset, '*', '*.jpg'))

    for fname in tqdm.tqdm(source1 + source2):
        filemeta = fname.split('/')[-1]
        # print(filemeta)
        pid, cam, tracklet, frame = int(filemeta[:3]), int(filemeta[4:6]), int(filemeta[7:10]), int(filemeta[11:14])

        if pid not in pids:
            pids[pid] = len(pids)
        if cam not in cameras:
            cameras[cam] = len(cameras)
        #print(f'{pid=} {cam=} {tracklet=} {frame=}')
        if (pid, tracklet) not in tracklets:
            tracklets[(pid, tracklet)] = ([], len(tracklets))
            pid_of_tracklet[len(tracklets)-1] = pids[pid]

        new_cam_id = cameras[cam]
        new_pid = pids[pid]
        unique_tracklet_id = tracklets[(pid, tracklet)][1]

        new_filename = f'{str(new_pid).zfill(2)}_{str(new_cam_id).zfill(2)}_{str(unique_tracklet_id).zfill(4)}_{str(frame).zfill(8)}.jpg'
        copyfile(fname, osp.join(target_directory, new_filename))

        #print(new_filename)

        tracklets[(pid, tracklet)][0].append(new_filename)

    print(len(tracklets))
    meta['num_cameras'] = len(cameras)
    meta['tracklets'] = [tracklets[k][0] for k in sorted(tracklets)]
    write_json(meta, osp.join(args.dataset, 'meta.json'))

    fold_files = sorted(glob(osp.join(args.folds_directory, '*track_fold_info_*.csv')))
    if len(fold_files) == 0:
        splits = [{0: {'set': [i for i in range(len(tracklets))], 'query': [], 'gallery': []}}]
    else:
        prefixes = dict()
        for fold_file in fold_files:
            prefix, tail = fold_file.split('track_fold_info_')
            fold = int(tail.split('.')[0])
            if prefix not in prefixes:
                prefixes[prefix] = dict()
            with open(fold_file, 'r') as f:
                reader = csv.reader(f)
                next(reader, None)
                foldset = []
                for pid, _, tracklet, start, end in reader:
                    pid = int(pid)
                    if pid in pids:
                        foldset.append(tracklets[(pid, int(tracklet))][1])
                        #print(int(end), int(start), len(tracklets[(int(pid), int(tracklet))][0]))
                        #assert int(end) - int(start) + 1 == len(tracklets[(int(pid), int(tracklet))][0])
                prefixes[prefix][fold] = {'set': foldset, 'query': [], 'gallery': []}
    
        splits = list(prefixes.values())
        
    for split in splits:
        for fold_id in split:
            fold = split[fold_id]
            fold['galleries_complement'] = dict()
            fold['galleries'] = dict()
            fold_tracklets = fold['set']
            print(len(fold_tracklets))
            complement_tracklets = [i for i in range(len(tracklets)) if i not in fold_tracklets]
            for g_size in args.gallery_sizes:
                fold['galleries'][g_size] = []
                fold['galleries_complement'][g_size] = []
                for _ in range(args.galleries_per_fold):
                    gallery = []
                    complement_gallery = []
                    for pid in pids:
                        pid = pids[pid]
                        gallery.extend(
                            np.random.choice(
                                [tr for tr in fold_tracklets if pid == pid_of_tracklet[tr]], g_size, replace=False
                            )
                        )
                        complement_gallery.extend(
                            np.random.choice(
                                [tr for tr in complement_tracklets if pid == pid_of_tracklet[tr]], g_size, replace=False
                            )
                        )
                    fold['galleries'][g_size].append([int(tr) for tr in gallery])
                    fold['galleries_complement'][g_size].append([int(tr) for tr in complement_gallery])
    
    write_json(splits, osp.join(args.dataset, 'splits.json'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Configure Metadata for Icebear Datasets")

    parser.add_argument('dataset', type=str, choices=translation.keys())
    parser.add_argument('--exclude-discarded', action='store_true')
    parser.add_argument('--folds-directory', type=str, default='.')
    parser.add_argument('--generate-galleries', action='store_true')
    parser.add_argument('--gallery-sizes', type=int, nargs='+', default=[1, 2, 4, 8])
    parser.add_argument('--galleries-per-fold', type=int, default=10)
    args = parser.parse_args()
    args.dataset = translation[args.dataset]

    main(args)