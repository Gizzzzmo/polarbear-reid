import time
import torch
from os import path as osp
from torch.autograd import Variable
from torch.optim import Optimizer
from typing import Optional, Sequence, Tuple, Union

from .metric_logging import MetricsLogger
from .utils import mkdir_if_missing, AverageMeter
from .models import MetricLearningNetwork
from .dataset import Dataset
from .evaluation import accuracy


class Trainer:
    def __init__(
        self,
        model: Union[MetricLearningNetwork, torch.nn.DataParallel],
        optimizer: Optimizer,
        classification_loss: torch.nn.Module,
        triplet_loss: torch.nn.Module,
        triplet_weight: int = 1.0,
        global_triplet_loss: bool = False,
        network_input_key: str = 'imgs',
        logger: Optional[MetricsLogger] = None,
        print_freq: int = 10,
        log_dir: str = '.'
        ):
        
        super(Trainer, self).__init__()
        self.model = model
        self.classification_loss = classification_loss
        self.triplet_loss = triplet_loss
        self.global_triplet_loss = global_triplet_loss
        self.network_input_key = network_input_key
        self.print_freq = print_freq
        self.logger = logger
        
        self.sequence_dir = osp.join(log_dir, 'sequence')
        self.optimizer = optimizer
        self.triplet_weight = triplet_weight
        mkdir_if_missing(self.sequence_dir)

    def train(
        self,
        epoch: int,
        train_sets: Sequence[Dataset],
        monitor_datasets: Sequence[Dataset] = tuple(),
        save_embedding_freq: int = 0,
        ) -> Tuple[float, float, float, float]:
        
        print(f'Start training epoch {epoch}')
        self.model.train()
                
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        cls_losses = AverageMeter()
        trp_losses = AverageMeter()
        accuracies = AverageMeter()

        end = time.time()
        
        data_loaders = [iter(train_set.train_loader) for train_set in train_sets]
        
        i = 0
        done = False
        
        while True:
            overall_classification_loss = AverageMeter()
            overall_accuracy = AverageMeter()

            data_time.update(time.time() - end)
            
            if self.global_triplet_loss:
                overall_features = []
            else:
                overall_triplet_loss = AverageMeter()
            overall_targets = []
            label_offset = 0
            
            full_batch_size = 0
            
            for data_loader, train_set in zip(data_loaders, train_sets):
                try:
                    inputs = next(data_loader)
                except StopIteration:
                    done = True
                    break

                inputs, targets = self._parse_data(inputs)
                features, classifications = self._forward(inputs, train_set.classifer_idx)
                batch_size = targets.size(0)
                full_batch_size += batch_size
                
                classification_loss = self.classification_loss(classifications, targets)
                acc1, _, _ = accuracy(classifications.data, targets.cpu().data)
                
                if self.global_triplet_loss:
                    overall_features.append(features)
                    overall_targets.append(targets + label_offset - train_set.label_offset)
                    label_offset += train_set.num_train_ids
                else:
                    triplet_loss = self.triplet_loss(features, targets)
                    overall_triplet_loss.update(triplet_loss, batch_size)
                
                overall_classification_loss.update(classification_loss, batch_size)
                overall_accuracy.update(acc1.item(), batch_size)

            if done:
                break
            
            
            if self.global_triplet_loss:
                overall_features = torch.cat(overall_features, dim=0)
                overall_targets = torch.cat(overall_targets, dim=0)
                triplet_loss = self.triplet_loss(overall_features, overall_targets)
            else:
                triplet_loss = overall_triplet_loss.avg
            
            classification_loss = overall_classification_loss.avg
            
            overall_loss = classification_loss + triplet_loss * self.triplet_weight
            
            self.optimizer.zero_grad()
            overall_loss.backward()
            self.optimizer.step()
            
            losses.update(overall_loss.data.item(), full_batch_size)
            cls_losses.update(classification_loss.data.item(), full_batch_size)
            trp_losses.update(triplet_loss.data.item(), full_batch_size)
            accuracies.update(overall_accuracy.avg, full_batch_size)

            batch_time.update(time.time() - end)
            end = time.time()

            if save_embedding_freq != 0 and (i + 1) % save_embedding_freq == 0:
                # TODO -- implement save embedding
                pass
            
            if ((i + 1) % self.print_freq) == 0:
                print(
                    f'Epoch: [{epoch}][{i+1}/{min(len(dataset.train_loader) for dataset in train_sets)}]\t'
                    f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    f'Data {data_time.val :.3f} ({data_time.avg :.3f})\t'
                    f'Loss {losses.val    :.3f} ({losses.avg    :.3f})'
                    f' cls {cls_losses.val:.3f} ({cls_losses.avg:.3f})'
                    f' trp {trp_losses.val:.3f} ({trp_losses.avg:.3f})\t'
                    f'Accu {accuracies.val:.2%} ({accuracies.avg:.2%})\t'
                )
            i += 1
        
        if self.logger is not None:
            self.logger['epoch'] = epoch
            self.logger['loss'] = losses.avg
            self.logger['classification_loss'] = cls_losses.avg
            self.logger['triplet_loss'] = trp_losses.avg
            self.logger['accuracy'] = accuracies.avg
            self.logger.write_row()
            
        return losses.avg, cls_losses.avg, trp_losses.avg, accuracies.avg

    def _parse_data(self, inputs):
        imgs, _, pids, _ = inputs
        inputs = Variable(imgs)
        targets = Variable(pids.cuda())
        return inputs, targets

    def _forward(self, inputs, classifier_idx):
        output = self.model({
            self.network_input_key: inputs
        }, classifier_idx)
        
        classifications = output['classifications']
        features = output['embedding']

        return features, classifications
