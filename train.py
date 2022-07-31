import datetime
import argparse
import os
import numpy as np
import torch
from typing import Sequence

from reid.clustering import k_means
from reid.utils import mkdir_if_missing, read_yaml, write_yaml, extract_output
from reid.dataset import Dataset
from reid.models import MetricLearningNetwork
from reid.configuration import setup_train_logging, setup_model, setup_trainer, setup_datasets
from reid import evaluation

def save_starting_config(run_log_dir: str, resume_config: dict, training_config: dict, model_config: dict):
    training_config_path = os.path.abspath(os.path.join(run_log_dir, 'training_config.yaml'))
    model_config_path = os.path.abspath(os.path.join(run_log_dir, 'model_config.yaml'))
    
    write_yaml(training_config, training_config_path)
    write_yaml(model_config, model_config_path)
    
    resume_config['TRAIN_CONFIG'] = training_config_path
    resume_config['MODEL_CONFIG'] = model_config_path
    
    write_yaml(resume_config, os.path.join(run_log_dir, 'starting_config.yaml'))
        
    resume_config['RUN_LOG_DIR'] = run_log_dir


def configure_dataset_loader_starting_point(
    train_sets: Sequence[Dataset],
    monitor_sets: Sequence[Dataset],
    network_input_key: str
    ):
    
    # TODO: implement
    pass

def log_qg_metrics(
    dataset: Dataset,
    technique: str,
    accuracies: np.array,
    f1s: np.array,
    kappas: np.array, 
    mAPs: np.array
    ):
    
    for i, size in enumerate(dataset.gallery_sizes):
        for j in range(dataset.qg_masks.shape[1]):
            for k, tracklet_corrected in enumerate(['', '_tracklet']):
                dataset.logger[f'{technique}_accuracy{tracklet_corrected}_{size}_{j}'] = accuracies[i, j, k]
                dataset.logger[f'{technique}_f1{tracklet_corrected}_{size}_{j}'] = f1s[i, j, k]
                dataset.logger[f'{technique}_kappa{tracklet_corrected}_{size}_{j}'] = kappas[i, j, k]
                dataset.logger[f'{technique}_mAP{tracklet_corrected}_{size}_{j}'] = mAPs[i, j, k]
                
    print(f'\n\nQuery-gallery evaluation with {technique}:\n')
    print(f'Gallery size:       ', ','.join(f'    {size:<6}' for size in dataset.gallery_sizes))
    print(f'                    {accuracies[:, :, 0].mean(axis=1)} +/- {accuracies[:, :, 0].std(axis=1)} (accuracies)')
    print(f'                    {mAPs[:, :, 0].mean(axis=1)      } +/- {mAPs[:, :, 0].std(axis=1)      } (mean-AP)')
    print(f'Image based:        {f1s[:, :, 0].mean(axis=1)       } +/- {f1s[:, :, 0].std(axis=1)       } (f1-scores)')
    print(f'                    {kappas[:, :, 0].mean(axis=1)    } +/- {kappas[:, :, 0].std(axis=1)    } (kappa-scores)')
    print('\n')
    print(f'                    {accuracies[:, :, 1].mean(axis=1)} +/- {accuracies[:, :, 1].std(axis=1)} (accuracies)')
    print(f'                    {mAPs[:, :, 1].mean(axis=1)      } +/- {mAPs[:, :, 1].std(axis=1)      } (mean-AP)')
    print(f'Tracklet corrected: {f1s[:, :, 1].mean(axis=1)       } +/- {f1s[:, :, 1].std(axis=1)       } (f1-scores)')
    print(f'                    {kappas[:, :, 1].mean(axis=1)    } +/- {kappas[:, :, 1].std(axis=1)    } (kappa-scores)')

def evaluate(
    epoch: int,
    model: MetricLearningNetwork,
    train_sets: Sequence[Dataset],
    monitor_sets: Sequence[Dataset]
    ):
    
    model.eval()
    
    critical_accuracy = 0
    
    all_pids = []
    all_classifications = []
    
    for dataset in train_sets:
        logger = dataset.logger
        logger['epoch'] = epoch
        
        features, classifications, pids, cams, tracklets, frames = extract_output(
            model,
            dataset.val_loader,
            dataset.classifer_idx
        )
        print(pids.shape, classifications.shape)
        pred = evaluation.predictions(classifications)
        all_pids.append(pids)
        all_classifications.append(pred)
        
        print(f'Evaluating model on {dataset.name}:\n')
        
        accuracy, f1, kappa = evaluation.accuracy(classifications, pids)
        critical_accuracy += accuracy
        
        print(f'Accuracy: {accuracy}')
        print(f'F1: {f1}')
        print(f'Kappa: {kappa}')
        logger['accuracy'] = accuracy
        logger['f1'] = f1
        logger['kappa'] = kappa
        
        accuracy, f1, kappa = evaluation.accuracy(classifications, pids, tracklets)
        print('\nTracklet corrected:')
        print(f'Accuracy: {accuracy}')
        print(f'F1: {f1}')
        print(f'Kappa: {kappa}')
        logger['accuracy_tracklet'] = accuracy
        logger['f1_tracklet'] = f1
        logger['kappa_tracklet'] = kappa
        
        pids = pids - dataset.label_offset
        
        predictions = k_means(features, dataset.num_val_ids)
        print(f'\nKmeans cluster evaluation:\n')
        _, _, acc, f1, kappa, mAP = evaluation.evaluate_cluster(predictions, pids)
        adj_rand_score, rand_score = evaluation.evaluate_rand_index(predictions, pids)
        logger['k_means_rand'] = rand_score
        logger['k_means_adj_rand'] = adj_rand_score
        logger['k_means_accuracy'] = acc
        logger['k_means_f1'] = f1
        logger['k_means_kappa'] = kappa
        logger['k_means_mAP'] = mAP
        print(f'Rand score:          {rand_score}')
        print(f'Adjusted Rand score: {adj_rand_score}')
        
        accuracies, kappas, f1s, mAPs = evaluation.query_gallery_1nn(features, dataset.qg_masks, pids, tracklets)
        log_qg_metrics(dataset, '1nn', accuracies, f1s, kappas, mAPs)
        
        accuracies, kappas, f1s, mAPs = evaluation.query_gallery_logreg(features, dataset.qg_masks, pids, tracklets)
        log_qg_metrics(dataset, 'logreg', accuracies, f1s, kappas, mAPs)
        
        accuracies, kappas, f1s, mAPs = evaluation.query_gallery_svm(features, dataset.qg_masks, pids, tracklets)
        log_qg_metrics(dataset, 'svm', accuracies, f1s, kappas, mAPs)
        
        logger.write_row()
    
    if len(all_classifications) > 0:
        all_classifications = np.concatenate(all_classifications, axis=0)
        all_pids = np.concatenate(all_pids, axis=0)    
        
        confusion = evaluation.confusion_matrix(all_classifications, all_pids)
        print('\nConfusion matrix:')
        print(confusion)
        
        np.save(
            '/'.join(logger.dest_path.split('/')[:-1]) + f'/confusion.npy',
            confusion
        )
    
    for dataset in monitor_sets:
        logger = dataset.logger
        logger['epoch'] = epoch
        
        features, classifications, pids, cams, tracklets, frames = extract_output(
            model,
            dataset.val_loader,
            dataset.classifer_idx
        )
        print(f'Evaluating model on {dataset.name}:\n')
        
        predictions = k_means(features, dataset.num_val_ids)
        print(f'Kmeans cluster evaluation:\n')
        _, _, acc, f1, kappa, mAP = evaluation.evaluate_cluster(predictions, pids)
        adj_rand_score, rand_score = evaluation.evaluate_rand_index(predictions, pids)
        logger['k_means_rand'] = rand_score
        logger['k_means_adj_rand'] = adj_rand_score
        logger['k_means_accuracy'] = acc
        logger['k_means_f1'] = f1
        logger['k_means_kappa'] = kappa
        logger['k_means_mAP'] = mAP
        print(f'Rand score:          {rand_score}')
        print(f'Adjusted Rand score: {adj_rand_score}')
        
        accuracies, kappas, f1s, mAPs = evaluation.query_gallery_1nn(features, dataset.qg_masks, pids, tracklets)
        log_qg_metrics(dataset, '1nn', accuracies, f1s, kappas, mAPs)
        
        accuracies, kappas, f1s, mAPs = evaluation.query_gallery_logreg(features, dataset.qg_masks, pids, tracklets)
        log_qg_metrics(dataset, 'logreg', accuracies, f1s, kappas, mAPs)
        
        accuracies, kappas, f1s, mAPs = evaluation.query_gallery_svm(features, dataset.qg_masks, pids, tracklets)
        log_qg_metrics(dataset, 'svm', accuracies, f1s, kappas, mAPs)
        
        logger.write_row()
        
        
    return float(critical_accuracy/len(train_sets)) if len(train_sets) > 0 else float('nan')

def save_checkpoint(
    epoch: int,
    final_epoch: int,
    model: MetricLearningNetwork,
    optimizer: torch.optim.Optimizer,
    datasets: Sequence[Dataset],
    resume_config: dict,
    run_log_dir: str,
    sub_dir: str
    ):
    
    dest_dir = os.path.join(run_log_dir, sub_dir)
    mkdir_if_missing(dest_dir)
    
    optimizer_path = os.path.abspath(os.path.join(dest_dir, 'optimizer.pth'))
    torch.save(optimizer.state_dict(), optimizer_path)
    
    backbone_path = os.path.abspath(os.path.join(dest_dir, 'backbone.pth'))
    model.save_backbone(backbone_path)
    
    embedder_path = os.path.abspath(os.path.join(dest_dir, 'embedder.pth'))
    model.save_metric_embedder(embedder_path)
    
    classifiers = [{} for _ in model.classifiers]
    for classifier_idx in range(len(model.classifiers)):
        classifier_path = os.path.abspath(os.path.join(dest_dir, f'classifier_{classifier_idx}.pth'))
        model.save_classifier(classifier_path, classifier_idx)
        classifiers[classifier_idx]['CLASSIFIER_WEIGHTS'] = classifier_path
        classifiers[classifier_idx]['DATASETS'] = [
            dataset.name for dataset in datasets if dataset.classifer_idx == classifier_idx
        ]
        
        
    resume_config['STARTING_EPOCH'] = epoch + 1
    resume_config['EPOCHS'] = final_epoch - resume_config['STARTING_EPOCH']
    
    resume_config['OPTIMIZER'] = optimizer_path
    resume_config['BACKBONE_WEIGHTS'] = backbone_path
    resume_config['EMBEDDER_WEIGHTS'] = embedder_path
    resume_config['CLASSIFIERS'] = classifiers
    
    resume_config_path = os.path.join(dest_dir, 'resume.yaml')
    write_yaml(resume_config, resume_config_path)   


def main(args):
    if args.gpu_devices is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
        
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    run_log_dir, train_logger, lock = setup_train_logging(args.logs_dir, args.resume_config, args.output_to_file)
    
    if 'RUN_LOG_DIR' not in args.resume_config:
        save_starting_config(
            run_log_dir,
            args.resume_config,
            args.training_config,
            args.model_config
        )
    
    train_sets, monitor_sets = setup_datasets(
        args.datasets,
        args.monitor_datasets,
        args.training_config['SINGLE_HEAD'],
        args.training_config['DATA'],
        args.data_dir,
        run_log_dir
    )
    
    model = setup_model(
        args.model_config,
        train_sets,
        args.training_config['SINGLE_HEAD'],
        args.resume_config
    )
    
    starting_epoch = args.starting_epoch
    
    model = torch.nn.DataParallel(model).cuda()
    
    trainer = setup_trainer(
        args.training_config,
        model,
        train_logger,
        args.print_freq,
        args.resume_config,
        run_log_dir
    )
    
    best_accuracy = evaluate(starting_epoch-1, model, train_sets, monitor_sets)
    if 'RUN_LOG_DIR' in args.resume_config:
        resume_config['BEST_ACCURACY'] = best_accuracy
    print(f'Training process starts @ {datetime.datetime.now():%Y-%m-%d %H:%M:%S}')
    
    for i in range(starting_epoch, starting_epoch + args.epochs):
        
        if args.create_embedding_sequence_upto is None or i >= args.create_embedding_sequence_upto >= 0:
            save_embeddings_freq = 0
        else:
            save_embeddings_freq = args.save_embeddings_freq
        
        trainer.train(
            i,
            train_sets,
            monitor_sets,
            save_embeddings_freq
        )
        
        critical_accuracy = evaluate(i, model, train_sets, monitor_sets)
        
        if critical_accuracy > best_accuracy:
            print('\nNew best model!')
            best_accuracy = critical_accuracy
            resume_config['BEST_ACCURACY'] = best_accuracy
            
            temp = resume_config['RUN_LOG_DIR']
            del resume_config['RUN_LOG_DIR']
            save_checkpoint(
                i,
                args.epochs + starting_epoch,
                model.module,
                trainer.optimizer,
                train_sets,
                resume_config,
                run_log_dir,
                'best'
            )
            resume_config['RUN_LOG_DIR'] = temp
        
        save_checkpoint(
            i,
            args.epochs + starting_epoch,
            model.module,
            trainer.optimizer,
            train_sets,
            resume_config,
            run_log_dir,
            'checkpoint'
        )
            
        print(f'\n-------------------------------\n')
        print(f'  Epoch {i} complete @ {datetime.datetime.now():%Y-%m-%d %H:%M:%S}')
        print(f'  Avg Accuracy:  {critical_accuracy:.4f}')
        print(f'  Best Accuracy: {best_accuracy    :.4f}')
        print(f'\n-------------------------------\n')
    
    lock.close()
    

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
    parser.add_argument('--monitor-datasets', type=str, metavar='DATASET', nargs='+')
    
    parser.add_argument('--logs-dir', type=str, metavar='LOGS_DIRECTORY')
    parser.add_argument('--data-dir', type=str, metavar='DATA_DIRECTORY')    
    
    parser.add_argument('--print-freq', type=int, metavar='PRINT_FREQUENCY')
    
    parser.add_argument(
        '--epochs',
        type=int,
        metavar='EPOCH',
        help="""
            Epochs to train for.
        """
    )
    parser.add_argument('--starting-epoch', type=int, metavar='STARTING_EPOCH')
    
    parser.add_argument('--seed', type=int, metavar='RANDOM_SEED', default=1)
    parser.add_argument('--new-run', action='store_true')
    parser.add_argument('--output-to-file', action='store_true')
    
    parser.add_argument(
        '--create-embedding-sequence-upto',
        metavar='EPOCH',
        type=int,
        help='The epoch up to which the validation set embeddings of the network are saved. Specify -1 to save all embeddings. If not given, no embeddings are saved.'
    )
    parser.add_argument(
        '--embedding-sequence-freq',
        metavar='ITERATIONS',
        type=int
    )
    
    parser.add_argument('--gpu-devices', metavar='GPU_DEVICE_IDS', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')
    
    args = parser.parse_args()
    resume_config = read_yaml(args.resume_config) if args.resume_config is not None else {}
    failed = False
    if args.datasets is None:
        if 'DATASETS' not in resume_config or len(resume_config['DATASETS']) == 0:
            print('No datasets specified. Please specify at least one dataset to train on, either through the --datasets argument or through the resume config file.')
            failed = True
        else:
            args.datasets = resume_config['DATASETS']
    resume_config['DATASETS'] = args.datasets
    
    if args.training_config is None:
        if 'TRAIN_CONFIG' not in resume_config:
            print('No training config specified. Please specify a training config file through the --train-config argument or through the resume config file.')
            failed = True
        else:
            args.training_config = resume_config['TRAIN_CONFIG']
    args.training_config = read_yaml(args.training_config)
            
    if args.model_config is None:
        if 'MODEL_CONFIG' not in resume_config:
            print('No model config specified. Please specify a model config file through the --model-config argument or through the resume config file.')
            failed = True
        else:
            args.model_config = resume_config['MODEL_CONFIG']
    args.model_config = read_yaml(args.model_config)
            
    if failed:
        exit(1)
        
    if args.monitor_datasets is None:
        if 'MONITOR_DATASETS' in resume_config:
            args.monitor_datasets = resume_config['MONITOR_DATASETS']
        else:
            args.monitor_datasets = []
    resume_config['MONITOR_DATASETS'] = args.monitor_datasets
        
    if args.starting_epoch is None:
        if 'STARTING_EPOCH' in resume_config:
            args.starting_epoch = resume_config['STARTING_EPOCH']
        else:
            args.starting_epoch = 1
    resume_config['STARTING_EPOCH'] = args.starting_epoch
    
    if args.epochs is None:
        if 'EPOCHS' in resume_config:
            args.epochs = resume_config['EPOCHS']
        else:
            args.epochs = 50
    resume_config['EPOCHS'] = args.epochs
    
    if args.print_freq is None:
        if 'PRINT_FREQ' in resume_config:
            args.print_freq = resume_config['PRINT_FREQ']
        else:
            args.print_freq = 50
    resume_config['PRINT_FREQ'] = args.print_freq
    
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
    
    if args.new_run:
        del resume_config['RUN_LOG_DIR']
    
    args.resume_config = resume_config
    
    main(args)
    