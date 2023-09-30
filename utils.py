import os
import torch
import torchvision.utils as tvutils
import numpy as np
import cv2
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sn
import torch.nn as nn

def confidenc_intervals(score_box):
    alpha = 0.95
    p = ((1.0 - alpha)/2.0) * 100
    lower_score = max(0.0, np.percentile(score_box, p))
    p = (alpha+((1.0-alpha)/2.0)) * 100
    upper_score = min(1.0, np.percentile(score_box, p))
    return sum(score_box)/len(score_box), lower_score, upper_score


"""
multi
"""
def plot_multiclass_roc_curve(fpr, tpr, roc_auc_score, filename):
    save_path = './roc_auc_curve/'
    Path(save_path).mkdir(parents=True, exist_ok=True)

    lw = 2
    plt.plot(fpr["macro"], tpr["macro"], color='orange', lw=lw, label='Macro-average ROC curve (area = %.4f)' % roc_auc_score[3],)
    plt.plot(fpr[0], tpr[0], color='red', lw=lw, label='ROC curve of normal (area = %.4f)' % roc_auc_score[0],)
    plt.plot(fpr[1], tpr[1], color='blue', lw=lw, label='ROC curve of abnormal1 (area = %.4f)' % roc_auc_score[1],)
    plt.plot(fpr[2], tpr[2], color='green', lw=lw, label='ROC curve of abnormal2 (area = %.4f)' % roc_auc_score[2],)
    
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title("Receiver operating characteristic")
    plt.legend(loc='lower right')
    plt.savefig(save_path + filename)
    plt.close()


##################################################################################################
def plot_multiclass_roc_curve(fpr, tpr, roc_auc_score, filename):
    save_path = './roc_auc_curve/'
    Path(save_path).mkdir(parents=True, exist_ok=True)

    lw = 2
    plt.plot(fpr["macro"], tpr["macro"], color='orange', lw=lw, label='Macro-average ROC curve (area = %.4f)' % roc_auc_score[3],)
    plt.plot(fpr[0], tpr[0], color='red', lw=lw, label='ROC curve of normal (area = %.4f)' % roc_auc_score[0],)
    plt.plot(fpr[1], tpr[1], color='blue', lw=lw, label='ROC curve of abnormal1 (area = %.4f)' % roc_auc_score[1],)
    plt.plot(fpr[2], tpr[2], color='green', lw=lw, label='ROC curve of abnormal2 (area = %.4f)' % roc_auc_score[2],)
    
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title("Receiver operating characteristic")
    plt.legend(loc='lower right')
    plt.savefig(save_path + filename)
    plt.close()


def evaludation_multi_auc(y, y_pred, n_classes):
    """
    y : onehot
    y_pred : predict probability
    """
    fpr, tpr, auc = dict(), dict(), dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y[:, i], pred[:, i])
        auc[i] = auc(fpr[i], tpr[i])
    
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes

    fpr["macro"], tpr["macro"] = all_fpr, mean_tpr
    auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    return auc["macro"]

def multi_confusion_matrix(all_labels, all_preds, number):
    """
    all_labels : 1-dimention with class info (0 or 1)
    all_preds : 1-dimention with class info (0 or 1)
    """
    Path('./confusion_matrix/').mkdir(parents=True, exist_ok=True)

    classes = ('Normal', 'Abnormal0', 'Abnormal1')
    cf_matrix = confusion_matrix(all_labels.cpu(), all_preds.cpu())
    df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix), index= [i for i in classes], columns=[i for i in classes])
    sn.heatmap(df_cm, annot=True)
    plt.savefig('./confusion_matrix/' + str(number) + '_result_cf_matrix.jpg')
    plt.close()

def evaluation_per_multiclass(y, y_pred, n_classes):
    """
    taget, pred info : target, predict class info (index)
    target, pred shape: (B,)
    """
    precision, sensitivity, specificity, accuracy = 0, 0, 0, 0
   
    for y_class in range(n_classes):
        y = np.where(y == y_class, 1, 0)
        y_pred = np.where(y_pred == y_class, 1, 0)
        cfx = confusion_matrix(y, y_pred)
        tp, fn, fp, tn = cfx[0, 0], cfx[0, 1], cfx[1, 0], cfx[1, 1]

        precision += (tp / (tp + fp))
        sensitivity += (tp / (tp + fn))
        specificity += (tn / (tn + fp))
        accuracy += ((tp + tn) / (tp + fn + fp +tn))
    
    precision /= n_classes
    sensitivity /= n_classes
    specificity /= n_classes
    accuracy /= n_classes

    return precision, sensitivity, specificity, accuracy

def plot_all_binaryclass_roc_curve(fpr, tpr, auc_score, label_name, no):
    save_path = './roc_auc_curve/'
    Path(save_path).mkdir(parents=True, exist_ok=True)

    lw = 2
    color = ['orange', 'red', 'green', 'blue']
    
    plt.plot(fpr, tpr, color=color[no], lw=lw, label= label_name[no] +' ROC curve (area = %.4f)' % auc_score)
    
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title("Receiver operating characteristic")
    plt.legend(loc='lower right')
    plt.savefig(save_path + 'all_roc_auc_curve.png')
    plt.close()

def plot_binaryclass_roc_curve(fpr, tpr, auc_score, filename):
    save_path = './roc_auc_curve/'
    Path(save_path).mkdir(parents=True, exist_ok=True)

    lw = 2
    plt.plot(fpr, tpr, color='orange', lw=lw, label='Macro-average ROC curve (area = %.4f)' % auc_score)
    
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title("Receiver operating characteristic")
    plt.legend(loc='lower right')
    plt.savefig(save_path + filename)
    plt.close()

def binary_confusion_matrix(all_labels, all_preds, number):
    """
    all_labels : 1-dimention with class info (0 or 1)
    all_preds : 1-dimention with class info (0 or 1)
    """
    Path('./confusion_matrix/').mkdir(parents=True, exist_ok=True)

    classes = ('Normal', 'Abnormal')
    cf_matrix = confusion_matrix(all_labels.cpu(), all_preds.cpu())
    df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix), index= [i for i in classes], columns=[i for i in classes])
    sn.heatmap(df_cm, annot=True)
    plt.savefig('./confusion_matrix/' + str(number) + '_result_cf_matrix.jpg')
    plt.close()

def evaluation_per_binayclass(y, y_pred):
    """
    taget, pred info : target, predict class info (index)
    target, pred shape: (B,)
    """
    precision, sensitivity, specificity, accuracy = 0, 0, 0, 0
    y_class = 1

    y = np.where(y == y_class, 1, 0)
    y_pred = np.where(y_pred == y_class, 1, 0)
    cfx = confusion_matrix(y, y_pred)
    tp, fn, fp, tn = cfx[0, 0], cfx[0, 1], cfx[1, 0], cfx[1, 1]

    precision = (tp / (tp + fp))
    sensitivity = (tp / (tp + fn))
    specificity = (tn / (tn + fp))
    accuracy = ((tp + tn) / (tp + fn + fp +tn))
    
    return precision, sensitivity, specificity, accuracy

####################################################################################

class Distance_euclidean(nn.Module):
    def __init__(self):
        super(Distance_euclidean, self).__init__()
    def calc_euclidean(self, x1, x2):
        return (x1 - x2).pow(2).sum(1)
    def forward(self, anchor: torch.Tensor, another: torch.Tensor) -> torch.Tensor:
        distance = self.calc_euclidean(anchor, another)
        return distance

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
    def calc_euclidean(self, x1, x2):
        return (x1 - x2).pow(2).sum(1)
    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        distance_positive = self.calc_euclidean(anchor, positive)
        distance_negative = self.calc_euclidean(anchor, negative)
        losses = torch.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()


def binary_confusion_matrix(all_labels, all_preds, number):
    """
    all_labels : 1-dimention with class info (0 or 1)
    all_preds : 1-dimention with class info (0 or 1)
    """
    Path('./confusion_matrix/').mkdir(parents=True, exist_ok=True)

    classes = ('Normal', 'Abnormal')
    cf_matrix = confusion_matrix(all_labels.cpu(), all_preds.cpu())
    cmn = cf_matrix.astype('float') / cf_matrix.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(figsize=(10, 10))
    sn.heatmap(cmn, annot=True, xticklabels=classes, yticklabels=classes)

    # df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix), index= [i for i in classes], columns=[i for i in classes])
    # sn.heatmap(df_cm, annot=True)
    plt.savefig('./confusion_matrix/' + str(number) + '_result_cf_matrix.jpg')
    plt.close()



def model_ckpt_save(model, filename):
    model_save_dir = 'model_save_dir/'
    if not os.path.exists(model_save_dir):
        os.mkdir(model_save_dir)
    model_save_path = os.path.join(model_save_dir, '{}-epo.ckpt'.format(filename))
    torch.save(model.state_dict(), model_save_path)
    print('Save Model checkpoint into {}...'.format(model_save_dir))

def img_save(result, class_info, step):
    save_path = './cam/' + str(class_info) + '/'
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    cv2.imwrite(save_path + '{}.jpg'.format(step), result)
    # tvutils.save_image(result, save_path + '{}.jpg'.format(step))

def returnCAM(feature_conv, weight_softmax, class_idx, IMAGE_SIZE):
    size_upsample = (IMAGE_SIZE, IMAGE_SIZE)
    _, nc, h, w = feature_conv.shape
    output_cam = []
    
    cam = weight_softmax[class_idx].dot(feature_conv.reshape((nc, h * w)))
    cam = cam.reshape(h, w)
    cam = cam - np.min(cam)
    cam_img = cam / np.max(cam)
    cam_img = np.uint8(255 * cam_img)
    output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam