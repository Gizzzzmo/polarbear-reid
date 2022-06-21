import datetime
import argparse
from io import TextIOWrapper
import os
import sys
import fcntl
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T
from itertools import chain
from typing import List, Sequence, Tuple

from reid.clustering import k_means
from reid.utils import mkdir_if_missing, read_yaml, write_yaml, extract_output
from reid.dataset import Dataset, create, Preprocessor
from reid.training import Trainer
from reid.models import MetricLearningNetwork, build_metric_learning_network
from reid.loss import losses
from reid.logging import MetricsLogger
from reid import evaluation

def setup_logging(logs_dir: str, resume_config: dict, output_to_file: bool) -> Tuple[str, MetricsLogger, TextIOWrapper]:
    
    mkdir_if_missing(logs_dir)
    
    if 'RUN_LOG_DIR' in resume_config:
        run_dir = resume_config['RUN_LOG_DIR']
        mkdir_if_missing(run_dir)
    else:
        with open(os.path.join(logs_dir, '.lock'), 'w') as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            
            dirs = next(os.walk(logs_dir))[1]
            i = 1
            while f'Run_{i}' in dirs:
                i += 1
                
            run_dir = os.path.join(logs_dir, f'Run_{i}')
            mkdir_if_missing(run_dir)
            
    lock = open(os.path.join(run_dir, '.lock'), 'w')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except:
        print('Can\'t resume run inplace, since training is still ongoing')
    
    if output_to_file:
        sys.stdout = open(os.path.join(run_dir, 'stdout.txt'), 'a')
    
    print(f'Logging to {run_dir}')
    
    train_logger = MetricsLogger(
        metrics=('epoch', 'loss', 'classification_loss', 'triplet_loss', 'accuracy'),
        dest_path=os.path.join(run_dir, 'train_log.csv')
    )
        
    return run_dir, train_logger, lock

def save_starting_config(run_log_dir: str, resume_config: dict, training_config: dict, model_config: dict):
    training_config_path = os.path.abspath(os.path.join(run_log_dir, 'training_config.yaml'))
    model_config_path = os.path.abspath(os.path.join(run_log_dir, 'model_config.yaml'))
    
    write_yaml(training_config, training_config_path)
    write_yaml(model_config, model_config_path)
    
    resume_config['TRAIN_CONFIG'] = training_config_path
    resume_config['MODEL_CONFIG'] = model_config_path
    
    write_yaml(resume_config, os.path.join(run_log_dir, 'starting_config.yaml'))
        
    resume_config['RUN_LOG_DIR'] = run_log_dir
    

def setup_datasets(
    dataset_names: Sequence[str],
    monitor_dataset_names: Sequence[str],
    single_head: bool,
    data_config: dict,
    data_dir: str,
    run_log_dir: str
    ) -> Tuple[Sequence[Dataset], Sequence[Dataset]]:
    
    batch_size = data_config['BATCH_SIZE']
    ids_per_batch = data_config['IDS_PER_BATCH']
    width = data_config['WIDTH']
    height = data_config['HEIGHT']    
    
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    
    train_transformer = T.Compose([
        T.Resize((height, width)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalizer,
        T.RandomErasing(p=0.5, scale=(0.2, 0.2), ratio=(0.3, 0.3))
    ])
    
    test_transformer = T.Compose([
        T.Resize((height, width)),
        T.ToTensor(),
        normalizer,
    ])
    
    train_sets, monitor_sets = [], []
    
    classification_metrics = ('accuracy', 'kappa', 'f1')
    classification_metrics = classification_metrics + tuple(f'{metric}_tracklet' for metric in classification_metrics)
    cluster_metrics = ('k_means_accuracy', 'k_means_kappa', 'k_means_f1', 'k_means_rand', 'k_means_adj_rand')
    query_gallery_metrics = (
        '1nn_accuracy', '1nn_kappa', '1nn_f1', '1nn_mAP',
        'logreg_accuracy', 'logreg_kappa', 'logreg_f1', 'logreg_mAP',
        'svm_accuracy', 'svm_kappa', 'svm_f1', 'svm_mAP'
    )
    query_gallery_metrics = query_gallery_metrics + tuple(f'{metric}_tracklet' for metric in query_gallery_metrics)
    
    label_offset = 0
    for i, name in enumerate(dataset_names):
        name, val_folds = name.split(':')
        
        root = os.path.join(data_dir, name)
        dataset = create(
            name,
            root,
            classifier_idx=0 if single_head else i, validation_folds=val_folds.split(','),
            label_offset=label_offset
        )
        
        dataset.train_loader = DataLoader(
            Preprocessor(
                dataset.train,
                root=dataset.images_dir,
                transform=train_transformer
            ),
            batch_size=batch_size, num_workers=1,
            shuffle=True,
            pin_memory=True, drop_last=True
        )

        dataset.val_loader = DataLoader(
            Preprocessor(
                dataset.val,
                root=dataset.images_dir,
                transform=test_transformer
            ),
            batch_size=batch_size, num_workers=1,
            shuffle=False, pin_memory=True
        )
        
        composed_query_gallery_metrics = tuple(
            f'{qgm}_{size}_{j}'
                for qgm in query_gallery_metrics
                for size in dataset.gallery_sizes
                for j in range(dataset.qg_masks.shape[1])
        )
        metrics = ('epoch',) + classification_metrics + cluster_metrics + composed_query_gallery_metrics
        
        dataset.logger = MetricsLogger(
            metrics,
            os.path.join(run_log_dir, f'{name}_train_metrics.csv')
        )
        
        train_sets.append(dataset)
        if single_head:
            label_offset += dataset.num_train_ids
    
    for name in monitor_dataset_names:
        name, val_folds = name.split(':')
        root = os.path.join(data_dir, name)
        dataset = create(name, root, validation_folds=val_folds.split(','))
        
        dataset.val_loader = DataLoader(
            Preprocessor(
                dataset.val,
                root=dataset.images_dir,
                transform=test_transformer
            ),
            batch_size=batch_size, num_workers=1,
            shuffle=False, pin_memory=True
        )
        
        composed_query_gallery_metrics = tuple(
            f'{qgm}_{size}_{j}'
                for qgm in query_gallery_metrics
                for size in dataset.gallery_sizes
                for j in range(dataset.qg_masks.shape[1])
        )
        metrics = ('epoch',) + cluster_metrics + composed_query_gallery_metrics
        
        dataset.logger = MetricsLogger(
            metrics,
            os.path.join(run_log_dir, f'{name}_train_metrics.csv')
        )
        
        monitor_sets.append(dataset)
    
    return tuple(train_sets), tuple(monitor_sets)

def setup_model(
    model_config: str,
    train_sets: Sequence[Dataset],
    single_head: bool,
    resume_config: dict,
    run_log_dir: str
    ) -> MetricLearningNetwork:
    
    if single_head:
        nums_classes = [sum(dataset.num_train_ids for dataset in train_sets)]
    else:
        nums_classes = [dataset.num_train_ids for dataset in train_sets]
        
    model = build_metric_learning_network(model_config, nums_classes=nums_classes)
    if 'BACKBONE_WEIGHTS' in resume_config:
        model.load_backbone(resume_config['BACKBONE_WEIGHTS'])
    if 'EMBEDDER_WEIGHTS' in resume_config:
        model.load_embedder(resume_config['EMBEDDER_WEIGHTS'])
    
    if 'CLASSIFIERS' in resume_config and len(resume_config['CLASSIFIERS']) != 0:
        if single_head:
            model.load_classifier(resume_config['CLASSIFIERS'][0]['CLASSIFIER_WEIGHTS'], 0)
            pass
        else:
            for i, dataset in enumerate(train_sets):
                for classifier in resume_config['CLASSIFIERS']:
                    names = classifier['DATASETS']
                    if dataset.name in names:
                        model.load_classifier(classifier['CLASSIFIER_WEIGHTS'], i)
                        break
                
    return model.cuda()

def configure_dataset_loader_starting_point(
    train_sets: Sequence[Dataset],
    monitor_sets: Sequence[Dataset],
    network_input_key: str
    ):
    
    # TODO: implement
    pass

def setup_trainer(
    training_config: dict,
    model: MetricLearningNetwork,
    logger: MetricsLogger,
    print_freq: int,
    resume_config: dict,
    run_log_dir: str
    ) -> Trainer:
    
    classification_loss_config = training_config['CLASSIFICATION_LOSS']
    args = classification_loss_config.get('ARGS', [])
    kwargs = classification_loss_config.get('KWARGS', {})
    classification_loss = losses[classification_loss_config['NAME']](*args, **kwargs).cuda()
    
    triplet_loss_config = training_config['TRIPLET_LOSS']
    args = triplet_loss_config.get('ARGS', [])
    kwargs = triplet_loss_config.get('KWARGS', {})
    triplet_loss = losses[triplet_loss_config['NAME']](*args, **kwargs).cuda()
    
    triplet_weight = training_config['TRIPLET_WEIGHT']
    global_triplet_loss = training_config['GLOBAL_TRIPLET_LOSS']
    network_input_key = training_config['NETWORK_INPUT_KEY']
    
    optimier_config = training_config['OPTIMIZER']
    
    param_groups = [
        {
            'params': model.module.backbone.parameters(),
            'lr': optimier_config['BACKBONE_LR_MULT'] * optimier_config['LEARN_RATE']
        },
        {
            'params': model.module.metric_embedder.parameters(),
            'lr': optimier_config['EMBEDDER_LR_MULT'] * optimier_config['LEARN_RATE']
        },
        {
            'params': chain(*(classifier.parameters() for classifier in model.module.classifiers)),
            'lr': optimier_config['CLASSIFIER_LR_MULT'] * optimier_config['LEARN_RATE']
        }
    ]
    
    optimizer = torch.optim.Adam(
        param_groups,
        lr=optimier_config['LEARN_RATE'],
        weight_decay=optimier_config['WEIGHT_DECAY']
    )
    
    if 'OPTIMIZER' in resume_config:
        optimizer.load_state_dict(torch.load(resume_config['OPTIMIZER']))
    
    trainer = Trainer(
        model,
        optimizer,
        classification_loss,
        triplet_loss,
        triplet_weight,
        global_triplet_loss,
        network_input_key,
        logger,
        print_freq,
        run_log_dir
    )
    
    return trainer

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
    print('Gallery size:       ', ','.join(f'    {size:<6}' for size in dataset.gallery_sizes))
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
    
    for dataset in train_sets:
        logger = dataset.logger
        logger['epoch'] = epoch
        
        features, classifications, pids, cams, tracklets, frames = extract_output(
            model,
            dataset.val_loader,
            dataset.classifer_idx
        )
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
        _, _, acc = evaluation.evaluate_cluster(predictions, pids)
        adj_rand_score, rand_score = evaluation.evaluate_rand_index(predictions, pids)
        logger['k_means_rand'] = rand_score
        logger['k_means_adj_rand'] = adj_rand_score
        logger['k_means_accuracy'] = acc
        print(f'Rand score:          {rand_score}')
        print(f'Adjusted Rand score: {adj_rand_score}')
        
        accuracies, kappas, f1s, mAPs = evaluation.query_gallery_1nn(features, dataset.qg_masks, pids, tracklets)
        log_qg_metrics(dataset, '1nn', accuracies, f1s, kappas, mAPs)
        
        accuracies, kappas, f1s, mAPs = evaluation.query_gallery_logreg(features, dataset.qg_masks, pids, tracklets)
        log_qg_metrics(dataset, 'logreg', accuracies, f1s, kappas, mAPs)
        
        accuracies, kappas, f1s, mAPs = evaluation.query_gallery_svm(features, dataset.qg_masks, pids, tracklets)
        log_qg_metrics(dataset, 'svm', accuracies, f1s, kappas, mAPs)
        
        logger.write_row()
        
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
        _, _, acc = evaluation.evaluate_cluster(predictions, pids)
        adj_rand_score, rand_score = evaluation.evaluate_rand_index(predictions, pids)
        logger['k_means_rand'] = rand_score
        logger['k_means_adj_rand'] = adj_rand_score
        logger['k_means_accuracy'] = acc
        print(f'Rand score:          {rand_score}')
        print(f'Adjusted Rand score: {adj_rand_score}')
        
        accuracies, kappas, f1s, mAPs = evaluation.query_gallery_1nn(features, dataset.qg_masks, pids, tracklets)
        log_qg_metrics(dataset, '1nn', accuracies, f1s, kappas, mAPs)
        
        accuracies, kappas, f1s, mAPs = evaluation.query_gallery_logreg(features, dataset.qg_masks, pids, tracklets)
        log_qg_metrics(dataset, 'logreg', accuracies, f1s, kappas, mAPs)
        
        accuracies, kappas, f1s, mAPs = evaluation.query_gallery_svm(features, dataset.qg_masks, pids, tracklets)
        log_qg_metrics(dataset, 'svm', accuracies, f1s, kappas, mAPs)
        
        logger.write_row()
        
        
    return float(critical_accuracy/len(train_sets))

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
    
    run_log_dir, train_logger, _ = setup_logging(args.logs_dir, args.resume_config, args.output_to_file)
    
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
        args.resume_config,
        run_log_dir
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
        print('hi')
        
        if args.create_embedding_sequence_upto is None or i >= args.create_embedding_sequence_upto >= 0:
            save_embeddings_freq = 0
        else:
            save_embeddings_freq = args.save_embeddings_freq
        print('hihi')
        
        trainer.train(
            i,
            train_sets,
            monitor_sets,
            save_embeddings_freq
        )
        print('hihihi')
        
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
    