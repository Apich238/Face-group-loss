import os
import numpy as np
from sklearn.model_selection import KFold
from scipy import interpolate
import math


def detect_image_format(folder):
    f1 = [os.path.join(folder, f) for f in os.listdir(folder) if os.path.isdir(os.path.join(folder, f))][0]
    f2 = [f.split('.')[-1] for f in os.listdir(f1) if f.split('.')[-1] in ['jpg', 'png', 'bmp', 'jpeg']][0]
    return f2


def read_pairs(folder, pairs_file):
    with open(pairs_file, 'r') as f:
        flines = f.readlines()
    fformat = detect_image_format(folder)
    parsed = [line.rstrip('\n').split('\t') for line in flines if len(line.split('\t')) > 2]
    fnames = []
    for line in parsed:
        if len(line) == 3:
            name1 = line[0]
            name2 = name1
            n1 = int(line[1])
            n2 = int(line[2])
            same = True
        else:
            name1 = line[0]
            n1 = int(line[1])
            name2 = line[2]
            n2 = int(line[3])
            same = False
        f1 = os.path.join(folder, name1, '{}_{:0>4}.{}'.format(name1, n1, fformat))
        f2 = os.path.join(folder, name2, '{}_{:0>4}.{}'.format(name2, n2, fformat))
        fnames.append((f1, f2, same))
    return fnames


def eval_accuracy(embs, issame, target_far):
    # обработка исходных данных
    emb_a = embs[:, 0, :]
    emb_b = embs[:, 1, :]
    dists = np.arccos(np.sum(np.multiply(emb_a, emb_b), 1).clip(-1., 1.))
    folds_cnt = 10
    thresholds = np.arange(0, math.pi + 0.01, 0.01)
    # roc - кривая и точность
    tpr, fpr, acc_mean, acc_std = eval_roc(dists, issame, thresholds, folds_cnt)
    # точность при различных частотах ложных допусков far
    thresholds = np.arange(0, math.pi + 0.001, 0.001)
    val_mean, val_std, far = eval_val(dists, issame, target_far, folds_cnt, thresholds)
    return tpr, fpr, acc_mean, acc_std, val_mean, val_std, far


def eval_acc(dists, issame, threshold):
    '''
    вычисляет точность на тестовой выборке при заданном пороге
    :param dists: 
    :param issame: 
    :param threshold: 
    :return: 
    '''
    predict_issame = np.less(dists, threshold)
    tp = np.sum(np.logical_and(predict_issame, issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), issame))

    tpr = 0 if (tp + fn == 0) else (tp / (tp + fn))
    fpr = 0 if (fp + tn == 0) else (fp / (fp + tn))
    acc = (tp + tn) / dists.size
    return tpr, fpr, acc


def eval_roc(dists, issame, thresholds, folds_cnt):
    '''
    вычисляет roc - кривую и точность
    :param dists:
    :param issame:
    :param thresholds:
    :param folds_cnt:
    :return:
    '''
    # разбиение для перекрёстной валидации
    k_fold = KFold(n_splits=folds_cnt, shuffle=False)
    # частоты верно - положительных и ложно - положительных обнаружений
    # на тестовом множестве на каждом разбиении и при каждом пороге
    tps = np.zeros((folds_cnt, thresholds.shape[0]), dtype=float)
    fps = np.zeros((folds_cnt, thresholds.shape[0]), dtype=float)
    # точность на каждом разбиении
    acc = np.zeros((folds_cnt), dtype=float)
    # разбиваемые индексы
    indices = np.arange(0, dists.shape[0], 1, dtype=int)
    # разбиение
    split = k_fold.split(indices)
    for fold_id, (train, test) in enumerate(split):
        # находим лучший порог принятия решения
        # для этого вычисляем точности для всех порогов на обучающем подмножестве
        acc_train = np.zeros((thresholds.shape[0]), dtype=float)
        for threshold_id, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_id] = eval_acc(dists[train], issame[train], threshold)
        best_threshold_id = np.argmax(acc_train)
        # вычисляем частоты для всех порогов на тестовом множестве
        for threshold_id, threshold in enumerate(thresholds):
            tps[fold_id, threshold_id], fps[fold_id, threshold_id], _ = \
                eval_acc(dists[test], issame[test], threshold)
        # вычисляем точность при лучшем пороге
        _, _, acc[fold_id] = eval_acc(dists[test], issame[test], thresholds[best_threshold_id])
    # всё усредняется по разбиениям
    tpr = np.mean(tps, 0)
    fpr = np.mean(fps, 0)
    acc_mean = np.mean(acc)
    acc_std = np.std(acc)
    return tpr, fpr, acc_mean, acc_std


def eval_val(dists, issame, target_far, folds_cnt, thresholds):
    '''
    вычисляет частоту верных допусков при заданной частоте неверных
    :param dists:
    :param issame:
    :param target_far:
    :param folds_cnt:
    :param thresholds:
    :return:
    '''
    # разбиения
    k_fold = KFold(n_splits=folds_cnt, shuffle=False)
    indices = np.arange(0, dists.shape[0], 1, dtype=int)
    split = k_fold.split(indices)
    val = np.zeros(folds_cnt)
    far = np.zeros(folds_cnt)
    for fold_id, (train, test) in enumerate(split):
        # частоты ложных допусков на обучающем множестве при разных порогах
        far_train = np.zeros(thresholds.shape[0])
        for threshold_idx, threshold in enumerate(thresholds):
            _, far_train[threshold_idx] = eval_val_far(dists[train], issame[train], threshold)
        if np.max(far_train) >= target_far:
            # постройка графика зависимости порога от частоты ложных допусков
            f = interpolate.interp1d(far_train, thresholds, kind='slinear')
            # выбор порога с требуемой частотой
            threshold = f(target_far)
        else:
            # елси порог не найден, выбираем порог, не допускающий допусков
            threshold = 0.0
        # вычисляем частоты на тестовом множестве с полученным порогом
        val[fold_id], far[fold_id] = eval_val_far(dists[test], issame[test], threshold)
    # усредняем по разбиениям
    val_mean = np.mean(val)
    val_std = np.std(val)
    far_mean = np.mean(far)
    return val_mean, val_std, far_mean


def eval_val_far(dist, issame, threshold):
    predicted_same = np.less(dist, threshold)
    # количество верных допусков
    true_accept = np.sum(np.logical_and(predicted_same, issame))
    # количество ложных допусков
    false_accept = np.sum(np.logical_and(predicted_same, np.logical_not(issame)))
    n_same = np.sum(issame)
    n_diff = np.sum(np.logical_not(issame))
    # частота верных допусков
    val = true_accept / n_same  # must be float
    # частота ложных допусков
    far = false_accept / n_diff
    return val, far
