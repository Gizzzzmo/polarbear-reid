import os
import sys
import fcntl
import torch
from io import TextIOWrapper
from torch.utils.data import DataLoader
from torchvision import transforms as T
from itertools import chain
from typing import Sequence, Tuple, Union

from .dataset import Dataset
from .models import MetricLearningNetwork, build_metric_learning_network
from .metric_logging import MetricsLogger
from .training import Trainer
from .loss import losses
from .dataset import Dataset, create, Preprocessor
from .utils import mkdir_if_missing


def setup_train_logging(logs_dir: str, resume_config: dict, output_to_file: bool) -> Tuple[str, MetricsLogger, TextIOWrapper]:
    
    mkdir_if_missing(logs_dir)
    
    if 'RUN_LOG_DIR' in resume_config:
        run_dir = resume_config['RUN_LOG_DIR']
        mkdir_if_missing(run_dir)
    else:
        with open(os.path.join(logs_dir, '.trainlock'), 'w') as f:
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

def setup_test_logging(logs_dir: str, output_to_file: bool) -> str:
    
    mkdir_if_missing(logs_dir)
    
    with open(os.path.join(logs_dir, '.testlock'), 'w') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        
        dirs = next(os.walk(logs_dir))[1]
        i = 1
        while f'Test_{i}' in dirs:
            i += 1
            
        test_dir = os.path.join(logs_dir, f'Test_{i}')
        mkdir_if_missing(test_dir)
    
    if output_to_file:
        sys.stdout = open(os.path.join(test_dir, 'stdout.txt'), 'a')
    
    print(f'Logging to {test_dir}')
        
    return test_dir


def setup_data_sample_logging(logs_dir: str, output_to_file: bool) -> str:
    
    mkdir_if_missing(logs_dir)
    
    with open(os.path.join(logs_dir, '.samplelock'), 'w') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        
        dirs = next(os.walk(logs_dir))[1]
        i = 1
        while f'Data_Samples_{i}' in dirs:
            i += 1
            
        data_sample_dir = os.path.join(logs_dir, f'Data_Samples_{i}')
        mkdir_if_missing(data_sample_dir)
    
    if output_to_file:
        sys.stdout = open(os.path.join(data_sample_dir, 'stdout.txt'), 'a')
    
    print(f'Logging to {data_sample_dir}')
        
    return data_sample_dir



def setup_trainer(
    training_config: dict,
    model: Union[MetricLearningNetwork, torch.nn.DataParallel],
    logger: MetricsLogger,
    print_freq: int,
    resume_config: dict,
    run_log_dir: str
    ) -> Trainer:
    
    if type(model) is torch.nn.DataParallel:
        module = model.module
    else:
        module = model
    
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
            'params': module.backbone.parameters(),
            'lr': optimier_config['BACKBONE_LR_MULT'] * optimier_config['LEARN_RATE']
        },
        {
            'params': module.metric_embedder.parameters(),
            'lr': optimier_config['EMBEDDER_LR_MULT'] * optimier_config['LEARN_RATE']
        },
        {
            'params': chain(*(classifier.parameters() for classifier in module.classifiers)),
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

def setup_train_sets(
    dataset_names: Sequence[str],
    single_head: bool,
    batch_size: int,
    ids_per_batch: int,
    train_transformer,
    val_transformer,
    data_dir: str
    ) -> Sequence[Dataset]:
    
    train_sets = []
    
    label_offset = 0
    for i, name in enumerate(dataset_names):
        name, val_fold = name.split(':')
        invert_split = val_fold.startswith('~')
        val_fold = val_fold.split('~')[-1]
        
        root = os.path.join(data_dir, name)
        dataset = create(
            name,
            root,
            classifier_idx=0 if single_head else i,
            validation_fold=val_fold,
            invert_split=invert_split,
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
                transform=val_transformer
            ),
            batch_size=batch_size, num_workers=1,
            shuffle=False, pin_memory=True
        )
        
        train_sets.append(dataset)
        if single_head:
            label_offset += dataset.num_train_ids
        
    return train_sets

def setup_model(
    model_config: str,
    train_sets: Sequence[Dataset],
    single_head: bool,
    resume_config: dict,
    load_classifier_weights: bool = True,
    ):
    
    if single_head:
        nums_classes = [sum(dataset.num_train_ids for dataset in train_sets)]
    else:
        nums_classes = [dataset.num_train_ids for dataset in train_sets]
        
    
    model = build_metric_learning_network(model_config, nums_classes=nums_classes)
    
    if 'BACKBONE_WEIGHTS' in resume_config:
        model.load_backbone(resume_config['BACKBONE_WEIGHTS'])
        print('hi?')
    if 'EMBEDDER_WEIGHTS' in resume_config:
        model.load_embedder(resume_config['EMBEDDER_WEIGHTS'])
        print('hi?')
    
    if 'CLASSIFIERS' in resume_config and len(resume_config['CLASSIFIERS']) != 0 and load_classifier_weights:
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

def setup_test_sets(
    dataset_names: Sequence[str],
    batch_size: int,
    transformer,
    data_dir: str
    ):
    
    test_sets = []
    
    for name in dataset_names:
        name, val_fold = name.split(':')
        invert_split = val_fold.startswith('~')
        val_fold = val_fold.split('~')[-1]
        
        root = os.path.join(data_dir, name)
        dataset = create(name, root, validation_fold=val_fold, invert_split=invert_split)
        
        dataset.val_loader = DataLoader(
            Preprocessor(
                dataset.val,
                root=dataset.images_dir,
                transform=transformer
            ),
            batch_size=batch_size,
            num_workers=1,
            shuffle=False,
            pin_memory=True
        )
        
        test_sets.append(dataset)
        
    return test_sets
    


def setup_datasets(
    dataset_names: Sequence[str],
    monitor_dataset_names: Sequence[str],
    single_head: bool,
    data_config: dict,
    data_dir: str,
    run_log_dir: str,
    normalize_train_data: bool = True,
    normalize_monitor_data: bool = True,
    create_logger: bool = True
    ) -> Tuple[Sequence[Dataset], Sequence[Dataset]]:
    
    batch_size = data_config['BATCH_SIZE']
    ids_per_batch = data_config['IDS_PER_BATCH']
    width = data_config['WIDTH']
    height = data_config['HEIGHT']
    
    train_transformer = T.Compose([
        T.Resize((height, width)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.RandomErasing(
            p=0.5,
            scale=(0.2, 0.2),
            ratio=(0.3, 3.3)
        )
    ])
    
    
    test_transformer = T.Compose([
        T.Resize((height, width)),
        T.ToTensor(),
    ])
    
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    if normalize_train_data:
        train_transformer = T.Compose([train_transformer, normalizer])
    if normalize_monitor_data:
        test_transformer = T.Compose([test_transformer, normalizer])
    
    train_sets = setup_train_sets(
        dataset_names,
        single_head,
        batch_size,
        ids_per_batch,
        train_transformer,
        test_transformer,
        data_dir
    )
    
    monitor_sets = setup_test_sets(
        monitor_dataset_names,
        batch_size,
        test_transformer,
        data_dir
    )
    
    classification_metrics = ('accuracy', 'kappa', 'f1')
    classification_metrics = classification_metrics + tuple(f'{metric}_tracklet' for metric in classification_metrics)
    cluster_metrics = ('k_means_accuracy', 'k_means_kappa', 'k_means_f1', 'k_means_mAP', 'k_means_rand', 'k_means_adj_rand')
    query_gallery_metrics = (
        '1nn_accuracy', '1nn_kappa', '1nn_f1', '1nn_mAP',
        'logreg_accuracy', 'logreg_kappa', 'logreg_f1', 'logreg_mAP',
        'svm_accuracy', 'svm_kappa', 'svm_f1', 'svm_mAP'
    )
    query_gallery_metrics = query_gallery_metrics + tuple(f'{metric}_tracklet' for metric in query_gallery_metrics)
    
    if create_logger:
        for dataset in train_sets:
            composed_query_gallery_metrics = tuple(
                f'{qgm}_{size}_{j}'
                    for qgm in query_gallery_metrics
                    for size in dataset.gallery_sizes
                    for j in range(dataset.qg_masks.shape[1])
            )
            metrics = ('epoch',) + classification_metrics + cluster_metrics + composed_query_gallery_metrics
            
            dataset.logger = MetricsLogger(
                metrics,
                os.path.join(run_log_dir, f'{dataset.name}_train_metrics.csv')
            )
        
        for dataset in monitor_sets:
            composed_query_gallery_metrics = tuple(
                f'{qgm}_{size}_{j}'
                    for qgm in query_gallery_metrics
                    for size in dataset.gallery_sizes
                    for j in range(dataset.qg_masks.shape[1])
            )
            metrics = ('epoch',) + cluster_metrics + composed_query_gallery_metrics
            
            dataset.logger = MetricsLogger(
                metrics,
                os.path.join(run_log_dir, f'{dataset.name}_train_metrics.csv')
            )
    
    return tuple(train_sets), tuple(monitor_sets)
