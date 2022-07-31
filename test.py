import torch
from torchvision import transforms as T
import numpy as np
import argparse
from reid.utils import read_yaml, write_yaml
from reid.configuration import setup_test_logging, setup_model, setup_test_sets
from reid.metric_logging import MetricsLogger
import os
from train import evaluate



def main(args):
    if args.gpu_devices is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
        
    test_log_dir = setup_test_logging(args.logs_dir, args.output_to_file)
    
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]) # TODO make configurable
    
    transformer = T.Compose([
        T.Resize((128, 256)), # TODO un-hardcode
        T.ToTensor(),
        normalizer,
    ])
    
    datasets = setup_test_sets(
        args.datasets,
        args.batch_size,
        transformer,
        args.data_dir
    )
    
    # TODO combine this block with the one in train.py
    cluster_metrics = ('k_means_accuracy', 'k_means_kappa', 'k_means_f1', 'k_means_mAP', 'k_means_rand', 'k_means_adj_rand')
    query_gallery_metrics = (
        '1nn_accuracy', '1nn_kappa', '1nn_f1', '1nn_mAP',
        'logreg_accuracy', 'logreg_kappa', 'logreg_f1', 'logreg_mAP',
        'svm_accuracy', 'svm_kappa', 'svm_f1', 'svm_mAP'
    )
    query_gallery_metrics = query_gallery_metrics + tuple(f'{metric}_tracklet' for metric in query_gallery_metrics)
    
    for dataset in datasets:
        composed_query_gallery_metrics = tuple(
            f'{qgm}_{size}_{j}'
                for qgm in query_gallery_metrics
                for size in dataset.gallery_sizes
                for j in range(dataset.qg_masks.shape[1])
        )
        metrics = ('epoch',) + cluster_metrics + composed_query_gallery_metrics
        
        dataset.logger = MetricsLogger(
            metrics,
            os.path.join(test_log_dir, f'{dataset.name}_test_metrics.csv')
        )
    
    print(args.resume_config)
    write_yaml(args.resume_config, os.path.join(test_log_dir, 'test_config.yaml'))
    
    model = setup_model(
        args.model_config,
        datasets,
        False,
        args.resume_config,
        load_classifier_weights=False
    )
    
    model = torch.nn.DataParallel(model)
    
    evaluate(
        epoch=0,
        model=model,
        train_sets=[],
        monitor_sets=datasets,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Re-ID Testing Script")
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
    
    parser.add_argument('--batch-size', type=int, metavar='N', default=64)
    
    parser.add_argument('--data-dir', type=str, metavar='DATA_DIRECTORY')
    parser.add_argument('--logs-dir', type=str, metavar='LOGS_DIRECTORY')
    
    parser.add_argument('--copy-weights', action='store_true', help='When set, save the used weights in the test directory')
    
    parser.add_argument('--output-to-file', action='store_true')
    parser.add_argument('--gpu-devices', metavar='GPU_DEVICE_IDS', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')
    
    args = parser.parse_args()
    resume_config = read_yaml(args.resume_config) if args.resume_config is not None else {}
    failed = False
    if args.datasets is None:
        if 'MONITOR_DATASETS' not in resume_config or len(resume_config['MONITOR_DATASETS']) == 0:
            print('No datasets specified. Please specify at least one dataset to test the model on, either by providing a --datasets argument or through the resume config file.')
            failed = True
        else:
            args.datasets = resume_config['MONITOR_DATASETS']
    resume_config['MONITOR_DATASETS'] = args.datasets
    
    if args.model_config is None:
        if 'MODEL_CONFIG' not in resume_config:
            print('No model config specified. Please specify a model config file through the --model-config argument or through the resume config file.')
            failed = True
        else:
            args.model_config = resume_config['MODEL_CONFIG']
    resume_config['MODEL_CONFIG'] = args.model_config
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