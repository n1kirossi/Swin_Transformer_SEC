# -*- coding: utf-8 -*-
import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from skimage.metrics import structural_similarity as compare_ssim


def snr_(arr1: np.ndarray, arr2: np.ndarray) -> float:
    """
    Вычисление отношения сигнал/шум (SNR).
    :param arr1: денойзированные данные
    :param arr2: чистые данные
    :return: SNR
    """
    snr = 10 * np.log10(np.sum(arr2 ** 2) / np.sum((arr2 - arr1) ** 2))
    return snr


def ssim_(imageA, imageB, data_range=1.0):
    """
    Вычисление структурного сходства (SSIM) для данных в диапазоне [-1, 1].
    :param imageA: изображение 1 (2D)
    :param imageB: изображение 2 (2D)
    :param data_range: диапазон значений данных (по умолчанию 1.0)
    :return: коэффициент структурного сходства
    """
    if imageA.ndim != 2 or imageB.ndim != 2:
        raise ValueError("SSIM ожидает 2D массивы")
    
    return compare_ssim(imageA, imageB, win_size=7, data_range=data_range)


def log(*args, **kwargs):
    """
    Логирование с префиксом времени.
    Пример: 2022-02-18 14:34:23: сообщение
    """
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S:"), *args, **kwargs)


def findLastCheckpoint(save_dir):
    """
    Получение последней сохранённой эпохи модели.
    :param save_dir: директория с моделями
    :return: номер последней эпохи или 0, если моделей нет
    """
    file_list = glob.glob(os.path.join(save_dir, 'model_*.pth'))
    if file_list:
        epochs_exist = []
        for file_ in file_list:
            result = re.findall(".*model_(.*).pth.*", file_)
            epochs_exist.append(int(result[0]))
        initial_epoch = max(epochs_exist)
    else:
        initial_epoch = 0
    return initial_epoch


def produce_csv(file_name):
    """Создание CSV-файла с заголовками для обучения."""
    df = pd.DataFrame(columns=['time', 'step', 'train_Loss', 'val_loss', 'time'])
    df.to_csv(file_name, index=False)


def save_csv(file_name, epoch, train_loss, val_loss, time):
    """Сохранение результатов обучения в CSV."""
    time = "%s" % datetime.now()
    step = "Step[%d]" % epoch
    train_l = "%f" % train_loss
    val_l = "%g" % val_loss

    row = [time, step, train_l, val_l]
    data = pd.DataFrame([row])
    data.to_csv(file_name, mode='TestData11_picture', header=False, index=False)


def produce_csv1(file_name):
    """Создание CSV-файла с заголовками для метрик (SNR, SSIM)."""
    df = pd.DataFrame(columns=['step', 'pre_snr', 'snr', 'pre_ssmi', 'ssmi', 'time'])
    df.to_csv(file_name, index=False)


def save_csv1(file_name, epoch, pre_snr, snr, pre_ssmi, ssmi, time):
    """Сохранение значений метрик (SNR, SSIM) в CSV."""
    step = "Step[%d]" % epoch
    pre_snr_l = "%f" % pre_snr
    snr_l = "%f" % snr
    pre_ssmi_l = "%f" % pre_ssmi
    ssmi_l = "%f" % ssmi
    time = "%s" % time

    row = [step, pre_snr_l, snr_l, pre_ssmi_l, ssmi_l, time]
    data = pd.DataFrame([row])
    data.to_csv(file_name, mode='TestData11_picture', header=False, index=False)


def show_csv(file_path, name_1, name_2, epoches):
    """
    Построение графика train_loss и val_loss по CSV.
    :param file_path: путь к файлу
    :param name_1: название колонки train_loss
    :param name_2: название колонки val_loss
    :param epoches: количество эпох
    """
    data = pd.read_csv(file_path)
    train_loss = data[[name_1]]
    val_loss = data[[name_2]]
    x = np.arange(0, epoches)
    y1 = np.array(train_loss)
    y2 = np.array(val_loss)

    plt.plot(x, y1, label="train_loss")
    plt.plot(x, y2, label="val_loss")
    plt.title("Loss")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.legend()
    plt.show()


def show_loss_(epoch, train_loss, val_loss):
    """График динамики train_loss и val_loss."""
    plt.plot(epoch, train_loss, label="Обучающая выборка")
    plt.plot(epoch, val_loss, label="Валидационная выборка")
    plt.legend(loc="best")
    plt.ylabel("Значение функции потерь")
    plt.xlabel("Эпоха")
    plt.title("Сравнение потерь на обучении и валидации")
    plt.show()


def show_snr_(train_snr, val_snr):
    """График динамики SNR на обучении и валидации."""
    plt.plot(train_snr, label="Обучающая выборка (SNR)")
    plt.plot(val_snr, label="Валидационная выборка (SNR)")
    plt.legend(loc="best")
    plt.ylabel("SNR")
    plt.xlabel("Эпоха")
    plt.title("Сравнение SNR на обучении и валидации")
    plt.show()


def show_ssmi_(train_ssmi, val_ssmi):
    """График динамики SSIM на обучении и валидации."""
    plt.plot(train_ssmi, label="Обучающая выборка (SSIM)")
    plt.plot(val_ssmi, label="Валидационная выборка (SSIM)")
    plt.legend(loc="best")
    plt.ylabel("SSIM")
    plt.xlabel("Эпоха")
    plt.title("Сравнение SSIM на обучении и валидации")
    plt.show()


def show_x_y(x, y):
    """
    Визуализация чистых и зашумлённых данных.
    :param x: чистые данные
    :param y: зашумлённые данные
    """
    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    plt.imshow(x, cmap=plt.cm.seismic, interpolation="nearest",
               aspect="auto", vmin=-0.5, vmax=0.5)
    plt.subplot(122)
    plt.imshow(y, cmap=plt.cm.seismic, interpolation="nearest",
               aspect="auto", vmin=-0.5, vmax=0.5)
    plt.show()


def show_x1_n(x1, y1):
    """
    Визуализация денойзированных данных и шума.
    :param x1: восстановленные данные
    :param y1: оставшийся шум
    """
    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    plt.imshow(x1, cmap=plt.cm.seismic, interpolation="nearest",
               aspect="auto", vmin=-1, vmax=1)
    plt.colorbar(shrink=0.5)

    plt.subplot(122)
    plt.imshow(y1, cmap=plt.cm.seismic, interpolation="nearest",
               aspect="auto", vmin=-1, vmax=1)
    plt.colorbar(shrink=0.5)
    plt.show()
