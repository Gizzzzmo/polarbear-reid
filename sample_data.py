import torch
from torchvision import transforms as T
import numpy as np
import argparse
from reid.utils import mkdir_if_missing, read_yaml, write_yaml
from reid.configuration import setup_data_sample_logging, setup_model, setup_datasets
import os


def main(args):
    if args.gpu_devices is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
        
    sample_log_dir = setup_data_sample_logging(args.logs_dir, False)
    mkdir_if_missing(os.path.join(sample_log_dir, 'train_samples'))
    
    if args.training_config is None:
        data_config = {
            'BATCH_SIZE': 1, 'IDS_PER_BATCH': 1, 'WIDTH': 256, 'HEIGHT': 128
        }
        single_head = False
    else:
        data_config = args.training_config['DATA']
        single_head = args.training_config['SINGLE_HEAD']
    
    datasets, _ = setup_datasets(
        args.datasets,
        [],
        single_head,
        data_config,
        args.data_dir,
        sample_log_dir,
        normalize_train_data=False,
        normalize_monitor_data=True,
        create_logger=False
    )
    for dataset in datasets:
        dataset.print_stats()
    
    if args.training_config is None:
        return
    
    write_yaml(args.resume_config, os.path.join(sample_log_dir, 'sampling_config.yaml'))
    
    toimg = T.ToPILImage()
    
    for dataset in datasets:
        for i, (imgs, blibs, pids, (camids, tids, frames)) in enumerate(dataset.train_loader):
            if i == 2:
                break
            for img_tensor, blib, pid, camid, tid, frame in zip(imgs, blibs, pids, camids, tids, frames):
                img = toimg(img_tensor)
                img.save(os.path.join(sample_log_dir, 'train_samples', f'{dataset.name}_{pid}_{camid}_{tid}_{frame}.jpg'))
            
    if args.model_config is None:
        return
    
    model = setup_model(
        args.model_config,
        datasets,
        False,
        args.resume_config
    )
    
    model = torch.nn.DataParallel(model)
    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Re-ID Training Script")
    
    parser.add_argument(
        '--model-config',
        type=str, 
        metavar='CONFIG_FILE',
        help="""
            The .yaml file to configure the model.
            Format:
                BACKBONE:
                    ARCHITECTURE: *name from model_zoo in models.py*
                    ARGS:
                        - arg1
                        - arg2
                        ...
                    KWARGS:
                        KEY1: arg1
                        KEY2: arg2
                        ...
                EMBEDDER:
                    ARCHITECTURE: *name from model_zoo in models.py*
                    ARGS:
                        - arg1
                        - arg2
                        ...
                    KWARGS:
                        KEY1: arg1
                        KEY2: arg2
                        ...
                CLASSIFIER:
                    ARCHITECTURE: *name from model_zoo in models.py*
                    ARGS:
                        - arg1
                        - arg2
                        ...
                    KWARGS:
                        KEY1: arg1
                        KEY2: arg2
                        ...
        """
    )
    
    parser.add_argument(
        '--training-config',
        type=str,
        metavar='CONFIG_FILE',
        help="""
            The .yaml file to configure the training.
            Format:
                OPTIMIZER:
                    LEARN_RATE: *float*
                    BACKBONE_LR_MULT: *float*
                    EMBEDDER_LR_MULT: *float*
                    CLASSIFIER_LR_MULT: *float*
                    WEIGHT_DECAY: *float*
                DATA:
                    BATCH_SIZE: *int*
                    IDS_PER_BACTCH: *int*
                    WIDTH: *int*
                    HEIGHT: *int*
                SINGLE_HEAD: *bool*
                NETWORK_INPUT_KEY: *str - the key to which the network inputs are fed*
                CLASSIFICATION_LOSS:
                    NAME: *name from losses in loss.py*
                    ARGS:
                        - arg1
                        - arg2
                        ...
                    KWARGS:
                        KEY1: kwarg1
                        KEY2: kwarg2
                        ...
                TRIPLET_LOSS:
                    NAME: *name from losses in loss.py*
                    ARGS:
                        - arg1
                        - arg2
                        ...
                    KWARGS:
                        KEY1: kwarg1
                        KEY2: kwarg2
                        ...
        """
    )
    
    parser.add_argument(
        '--resume-config',
        type=str,
        metavar='CONFIG_FILE',
        help="""
            The .yaml config file for resuming training.
            Format:
                BACKBONE_WEIGHTS: *path to weights file*
                EMBEDDER_WEIGHTS: *path to weights file*
                CLASSIFIERS: 
                    - CLASSIFIER_WEIGHTS: *path to weights file*
                      DATASETS: 
                        - *dataset name*
                        ...
                STARTING_EPOCH: *int, default: 1*
                EPOCHS: *int, default 50*
                TRAIN_CONFIG: *path to config file*
                MODEL_CONFIG: *path to config file*
                DATASETS:
                    - *training dataset 1*
                    - *training dataset 2*
                    ...
                MONITOR_DATASETS:
                    - *monitoring dataset 1*
                    - *monitoring dataset 2*
                    ...
                DATA_DIR: *path to data directory, default: data*
                LOGS_DIR: *path to logs directory, default: logs*
                PRINT_FREQ: *int, default: 50*
                RNG_STATE? -- TODO
        """
    )       
    
    parser.add_argument('--datasets', type=str, metavar='DATASET', nargs='+')
    
    parser.add_argument('--logs-dir', type=str, metavar='LOGS_DIRECTORY')
    parser.add_argument('--data-dir', type=str, metavar='DATA_DIRECTORY')    
    
    parser.add_argument('--gpu-devices', metavar='GPU_DEVICE_IDS', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')
    
    args = parser.parse_args()
    resume_config = read_yaml(args.resume_config) if args.resume_config is not None else {}
    failed = False
    if args.datasets is None:
        if ('DATASETS' not in resume_config or len(resume_config['DATASETS']) == 0) and \
           ('MONITOR_DATASETS' not in resume_config or len(resume_config['MONITOR_DATASETS']) == 0):
            print('No datasets specified. Please specify at least one dataset to sample data from.')
            failed = True
        else:
            args.datasets = resume_config['DATASETS'] + resume_config['MONITOR_DATASETS']
            
    resume_config['DATASETS'] = args.datasets
    if 'MONITOR_DATASETS' in resume_config:
        del resume_config['MONITOR_DATASETS']
    
    if failed:
        exit(1)
    
    
    if args.training_config is None:
        if 'TRAIN_CONFIG' not in resume_config:
            print('No training config specified. No training images will be sampled.')
        else:
            args.training_config = resume_config['TRAIN_CONFIG']
    
    if args.training_config is not None:
        args.training_config = read_yaml(args.training_config)
            
    if args.model_config is None:
        if 'MODEL_CONFIG' in resume_config:
            args.model_config = resume_config['MODEL_CONFIG']
            print('No model config specified. No embeddings will be compueted.')
    elif args.model_config == 'None':
        args.model_config = None
    
    if args.model_config is not None:
        args.model_config = read_yaml(args.model_config)
    
    if args.data_dir is None:
        if 'DATA_DIR' in resume_config:
            args.data_dir = resume_config['DATA_DIR']
        else:
            args.data_dir = 'data'
    resume_config['DATA_DIR'] = os.path.abspath(args.data_dir)
        
    if args.logs_dir is None:
        if 'LOGS_DIR' in resume_config:
            args.logs_dir = resume_config['LOGS_DIR']
        else:
            args.logs_dir = 'logs'
    resume_config['LOGS_DIR'] = os.path.abspath(args.logs_dir)
    
    args.resume_config = resume_config
    
    main(args)