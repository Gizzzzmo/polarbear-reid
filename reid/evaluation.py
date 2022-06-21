from typing import Optional
from .utils import extract_output
import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.semi_supervised import LabelPropagation
from sklearn.svm import SVC
from sklearn.metrics import average_precision_score, cohen_kappa_score, f1_score, accuracy_score, adjusted_rand_score, rand_score


def accuracy(output: torch.Tensor, target: np.array, tracklets: Optional[np.array] = None):
    _, pred = output.topk(1, 1)
    pred = pred[:, 0].cpu().numpy()
    
    if tracklets is None:
        acc = accuracy_score(target, pred)
        f1 = f1_score(target, pred, average='weighted')
        kappa = cohen_kappa_score(target, pred)
        
        return acc, f1, kappa
    
    pred = majority_image_to_tracklet_labels(pred, tracklets)
    
    acc = accuracy_score(target, pred)
    f1 = f1_score(target, pred, average='weighted')
    kappa = cohen_kappa_score(target, pred)
    return acc, f1, kappa


def best_matching_accuracy(predicted_clusters, gt_clusters):
    num_samples = len(predicted_clusters)
    mask = predicted_clusters != -1
    predicted_clusters = predicted_clusters[mask]

    unique_pred_clusters = np.unique(predicted_clusters)
    unique_gt_clusters = np.unique(gt_clusters)
    num_predicted_clusters = np.size(unique_pred_clusters)
    num_gt_clusters = np.size(unique_gt_clusters)
    #print(unique_gt_clusters, np.arange(num_gt_clusters))
    if not np.all(np.arange(num_predicted_clusters) == unique_pred_clusters):
        for i, cluster in enumerate(unique_pred_clusters):
            predicted_clusters[predicted_clusters == cluster] = i
    unique_pred_clusters = np.unique(predicted_clusters)
    assert np.all(np.arange(num_predicted_clusters) == unique_pred_clusters)

    if not np.all(np.arange(num_gt_clusters) == unique_gt_clusters):
        for i, cluster in enumerate(unique_gt_clusters):
            gt_clusters[gt_clusters == cluster] = i
    unique_gt_clusters = np.unique(gt_clusters)
    assert np.all(np.arange(num_gt_clusters) == unique_gt_clusters)

    gt_clusters = gt_clusters[mask]

    weight_matrix = np.empty((num_predicted_clusters, num_gt_clusters))
    for cluster_id in unique_pred_clusters:
        for gt_cluster_id in unique_gt_clusters:
            weight_matrix[cluster_id, gt_cluster_id] = np.sum((predicted_clusters == cluster_id) & (gt_clusters == gt_cluster_id))

    pred_indices, gt_indices = linear_sum_assignment(weight_matrix, maximize=True)
    return pred_indices, gt_indices, weight_matrix[pred_indices, gt_indices].sum()/num_samples

def evaluate_cluster(predictions, labels):
    non_noise = np.sum(predictions != -1)/len(predictions)
    map1, map2, acc = best_matching_accuracy(predictions, labels)
    acc = acc * non_noise
    
    return map1, map2, acc

def evaluate_rand_index(predictions, labels):
    assert np.all(np.array(predictions.shape) == np.array(labels.shape))
    
    return adjusted_rand_score(labels, predictions), rand_score(labels, predictions)

def query_gallery_1nn(samples, masks, labels, tracklets):
    accuracies = np.empty((*masks.shape[:2], 2))
    kappas = np.empty((*masks.shape[:2], 2))
    f1s = np.empty((*masks.shape[:2], 2))
    mAPs = np.empty((*masks.shape[:2], 2))
    for i in range(masks.shape[0]):
        for j in range(masks.shape[1]):                
            gallery_mask = masks[i, j, 0]
            query_mask = masks[i, j, 1]
            
            query_vectors = samples[query_mask]
            gallery_vectors = samples[gallery_mask]
            gallery_vector_labels = labels[gallery_mask]
            query_vector_labels = labels[query_mask]
            query_vector_tracklets = tracklets[query_mask]
            
            dist_mat = torch.cdist(query_vectors, gallery_vectors)
            closest = np.argmin(dist_mat, axis=1)
            predictions = gallery_vector_labels[closest]
            
            accuracies[i, j, 0] = accuracy_score(query_vector_labels, predictions)
            kappas[i, j, 0] = cohen_kappa_score(predictions, query_vector_labels)
            f1s[i, j, 0] = f1_score(predictions, query_vector_labels, average='weighted')
            mAPs[i, j, 0] = average_precision_score(query_vector_labels, predictions, average='weighted')
            predictions = majority_image_to_tracklet_labels(predictions, query_vector_tracklets)
            accuracies[i, j, 1] = accuracy_score(query_vector_labels, predictions)
            kappas[i, j, 1] = cohen_kappa_score(predictions, query_vector_labels)
            f1s[i, j, 1] = f1_score(predictions, query_vector_labels, average='weighted')
            mAPs[i, j, 1] = average_precision_score(query_vector_labels, predictions, average='weighted')
            
    return accuracies, kappas, f1s, mAPs
           
def query_gallery_logreg(samples, masks, labels, tracklets):
    accuracies = np.empty((*masks.shape[:2], 2))
    kappas = np.empty((*masks.shape[:2], 2))
    f1s = np.empty((*masks.shape[:2], 2))
    mAPs = np.empty((*masks.shape[:2], 2))
    for i in range(masks.shape[0]):
        for j in range(masks.shape[1]):
            gallery_mask = masks[i, j, 0]
            query_mask = masks[i, j, 1]
            
            query_vectors = samples[query_mask]
            gallery_vectors = samples[gallery_mask]
            gallery_vector_labels = labels[gallery_mask]
            query_vector_labels = labels[query_mask]
            query_vector_tracklets = tracklets[query_mask]
            
            logreg = LogisticRegression(max_iter=1000)
            logreg.fit(gallery_vectors, gallery_vector_labels)
            predictions = logreg.predict(query_vectors)
            
            accuracies[i, j, 0] = accuracy_score(query_vector_labels, predictions)
            kappas[i, j, 0] = cohen_kappa_score(predictions, query_vector_labels)
            f1s[i, j, 0] = f1_score(predictions, query_vector_labels, average='weighted')
            mAPs[i, j, 0] = average_precision_score(query_vector_labels, predictions, average='weighted')
            predictions = majority_image_to_tracklet_labels(predictions, query_vector_tracklets)
            accuracies[i, j, 1] = accuracy_score(query_vector_labels, predictions)
            kappas[i, j, 1] = cohen_kappa_score(predictions, query_vector_labels)
            f1s[i, j, 1] = f1_score(predictions, query_vector_labels, average='weighted')
            mAPs[i, j, 1] = average_precision_score(query_vector_labels, predictions, average='weighted')
            
    return accuracies, kappas, f1s, mAPs
 
def query_gallery_svm(samples, masks, labels, tracklets):
    accuracies = np.empty((*masks.shape[:2], 2))
    kappas = np.empty((*masks.shape[:2], 2))
    f1s = np.empty((*masks.shape[:2], 2))
    mAPs = np.empty((*masks.shape[:2], 2))
    for i in range(masks.shape[0]):
        for j in range(masks.shape[1]):
            gallery_mask = masks[i, j, 0]
            query_mask = masks[i, j, 1]
            
            query_vectors = samples[query_mask]
            gallery_vectors = samples[gallery_mask]
            gallery_vector_labels = labels[gallery_mask]
            query_vector_labels = labels[query_mask]
            query_vector_tracklets = tracklets[query_mask]
            
            svm = SVC()
            svm.fit(gallery_vectors, gallery_vector_labels)
            predictions = svm.predict(query_vectors)
            
            accuracies[i, j, 0] = accuracy_score(query_vector_labels, predictions)
            kappas[i, j, 0] = cohen_kappa_score(predictions, query_vector_labels)
            f1s[i, j, 0] = f1_score(predictions, query_vector_labels, average='weighted')
            mAPs[i, j, 0] = average_precision_score(query_vector_labels, predictions, average='weighted')
            predictions = majority_image_to_tracklet_labels(predictions, query_vector_tracklets)
            accuracies[i, j, 1] = accuracy_score(query_vector_labels, predictions)
            kappas[i, j, 1] = cohen_kappa_score(predictions, query_vector_labels)
            f1s[i, j, 1] = f1_score(predictions, query_vector_labels, average='weighted')
            mAPs[i, j, 1] = average_precision_score(query_vector_labels, predictions, average='weighted')
            
    return accuracies, kappas, f1s, mAPs

def query_gallery_mlp(samples, masks, labels, tracklets):
    accuracies = np.empty((*masks.shape[:2], 2))
    kappas = np.empty((*masks.shape[:2], 2))
    f1s = np.empty((*masks.shape[:2], 2))
    mAPs = np.empty((*masks.shape[:2], 2))
    for i in range(masks.shape[0]):
        for j in range(masks.shape[1]):
            gallery_mask = masks[i, j, 0]
            query_mask = masks[i, j, 1]
            
            query_vectors = samples[query_mask]
            gallery_vectors = samples[gallery_mask]
            gallery_vector_labels = labels[gallery_mask]
            query_vector_labels = labels[query_mask]
            query_vector_tracklets = tracklets[query_mask]
            
            mlp = MLPClassifier(hidden_layer_sizes=(2048,), max_iter=50, solver='lbfgs')
            mlp.fit(gallery_vectors, gallery_vector_labels)
            predictions = mlp.predict(query_vectors)
            
            accuracies[i, j, 0] = accuracy_score(query_vector_labels, predictions)
            kappas[i, j, 0] = cohen_kappa_score(predictions, query_vector_labels)
            f1s[i, j, 0] = f1_score(predictions, query_vector_labels, average='weighted')
            mAPs[i, j, 0] = average_precision_score(query_vector_labels, predictions, average='weighted')
            predictions = majority_image_to_tracklet_labels(predictions, query_vector_tracklets)
            accuracies[i, j, 1] = accuracy_score(query_vector_labels, predictions)
            kappas[i, j, 1] = cohen_kappa_score(predictions, query_vector_labels)
            f1s[i, j, 1] = f1_score(predictions, query_vector_labels, average='weighted')
            mAPs[i, j, 1] = average_precision_score(query_vector_labels, predictions, average='weighted')
            
    return accuracies, kappas, f1s, mAPs


def query_gallery_labelprop(samples, masks, labels, tracklets):
    accuracies = np.empty((*masks.shape[:2], 2))
    kappas = np.empty((*masks.shape[:2], 2))
    f1s = np.empty((*masks.shape[:2], 2))
    mAPs = np.empty((*masks.shape[:2], 2))
    for i in range(masks.shape[0]):
        for j in range(masks.shape[1]):
            gallery_mask = masks[i, j, 0]
            query_mask = masks[i, j, 1]
            
            query_vectors = samples[query_mask]
            query_vector_labels = labels[query_mask]
            query_vector_tracklets = tracklets[query_mask]
            sublabels = labels.copy()
            sublabels[~gallery_mask] = -1
            
            labelprop = LabelPropagation()
            labelprop.fit(samples, sublabels)
            predictions = labelprop.predict(query_vectors)
            
            accuracies[i, j, 0] = accuracy_score(query_vector_labels, predictions)
            kappas[i, j, 0] = cohen_kappa_score(predictions, query_vector_labels)
            f1s[i, j, 0] = f1_score(predictions, query_vector_labels, average='weighted')
            mAPs[i, j, 0] = average_precision_score(query_vector_labels, predictions, average='weighted')
            predictions = majority_image_to_tracklet_labels(predictions, query_vector_tracklets)
            accuracies[i, j, 1] = accuracy_score(query_vector_labels, predictions)
            kappas[i, j, 1] = cohen_kappa_score(predictions, query_vector_labels)
            f1s[i, j, 1] = f1_score(predictions, query_vector_labels, average='weighted')
            mAPs[i, j, 1] = average_precision_score(query_vector_labels, predictions, average='weighted')
            
    return accuracies, kappas, f1s, mAPs
            
def majority_image_to_tracklet_labels(labels, tracklets):
    new_labels = np.empty_like(labels)
    for tracklet in np.unique(tracklets):
        new_labels[tracklets == tracklet] = np.bincount(labels[tracklets == tracklet]).argmax()
    return new_labels
        
            
def evaluate_query_gallery_accuracy(samples, labels, tracklets):
    tracklets_sim = np.tile(tracklets, (tracklets.shape[0], 1)) == tracklets.reshape(-1, 1)
    dist_mat = torch.cdist(samples, samples, p=2).cpu().numpy()
    dist_mat[tracklets_sim] = np.inf
    closest = np.argmin(dist_mat, axis=1)

    return np.sum(labels[closest] == labels)/len(labels)
