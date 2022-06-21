from typing import Tuple
import csv
import os
import glob
import pandas as pd
import numpy as np
from .utils import read_yaml

class MetricsLogger():
    def __init__(self, metrics: Tuple[str], dest_path: str, overwrite: bool = False):
        self.dest_path = dest_path
        self.metrics = metrics
        self.current_row_data = {}
        if overwrite or not os.path.exists(dest_path):
            with open(dest_path, 'w') as f:
                csvwriter = csv.writer(f)
                csvwriter.writerow(self.metrics)
            
    def __getitem__(self, key: str):
        if key not in self.metrics:
            raise KeyError(f'{key} is not in {self.metrics}')
        
        return self.current_row_data.get(key, None)
        
    def __setitem__(self, key: str, value):
        if key not in self.metrics:
            raise KeyError(f'{key} is not in {self.metrics}')
        if key in self.current_row_data:
            print(f'Warning: {key} is already set. Overwriting.')
            
        self.current_row_data[key] = value
        
    def flush(self):
        self.current_row_data = {}
        
    def write_row(self):
        with open(self.dest_path, 'a') as f:
            csvwriter = csv.writer(f)
            csvwriter.writerow([self.current_row_data.get(key, '') for key in self.metrics])
        self.flush()
        
def load_run_data(path: str) -> pd.DataFrame:
    dataset_metrics = glob.glob(os.path.join(path, '*_train_metrics.csv'))
    dataset_metrics = dict(map(
        lambda x: (
            x.split('/')[-1].split('_train_metrics.csv')[0],
            x
        ),
        dataset_metrics
    ))
    train_metrics = pd.read_csv(os.path.join(path, 'train_log.csv'))
    
    conf = read_yaml(os.path.join(path, 'starting_config.yaml'))
    model_conf = read_yaml(conf['MODEL_CONFIG'])
    train_conf = read_yaml(conf['TRAIN_CONFIG'])
    train_sets = list(map(lambda x: tuple(x.split(':')), conf['DATASETS']))
    monitor_sets = list(map(lambda x: tuple(x.split(':')), conf['MONITOR_DATASETS']))
    pretrained = model_conf['BACKBONE']['KWARGS']['pretrained']
    single_head = train_conf['SINGLE_HEAD']
    
    return dataset_metrics, train_metrics, train_sets, monitor_sets, pretrained, single_head

def view_all_runs(logs_dir: str):
    for tgt in sorted(glob.glob(os.path.join(logs_dir, '*')), key=lambda x: int(x.split('_')[-1])):
        dm, tm, ts, ms, pre, sh = load_run_data(tgt)
        print(tgt)
        print(f'single head: {sh}, pretrained: {pre}')
        print(np.max(tm.loc[:, 'epoch']))
        print(ts, '->', ms)