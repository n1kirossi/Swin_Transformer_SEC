import os
import numpy as np

# Функция для предварительной обработки данных

def preprocess_seismic_data(seismic_data, nsub, norm):
    """
    Preprocess seismic data by adding noise and subsampling traces.

    Parameters:
        seismic_data (np.ndarray): Original seismic data array.
        nsub (int): Subsampling factor. From every 'nsub' traces, only one is kept.
        noise_level (float): Standard deviation of the Gaussian noise to be added.

    Returns:
        np.ndarray: Preprocessed seismic data.
    """
    data = seismic_data.astype(np.float32)

    if norm:
        mean, std = data.mean(), data.std()
        data = np.clip((data - mean) / (5 * (std + 1e-10)), -1, 1)

    mask = np.zeros(data.shape[1], dtype=np.float32)
    mask[::nsub] = 1.0
    return mask * data

if __name__ == '__main__':
    # Загрузка данных
    CLEAN_DIR = 'inputs/data/clean_data'
    THINNED_DIR = 'inputs/data/regular_thinned_data'
    input_folder = os.path.join(CLEAN_DIR)
    output_folder = os.path.join(THINNED_DIR)

    # Определяем параметры
    nsub = 2  # Каждая nsub-я трасса сохраняется
    norm = False

    # Обрабатываем каждый файл в папке с исходными данными
    for filename in os.listdir(input_folder):
        if filename.endswith(".npy"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # Загружаем данные
            seismic_data = np.load(input_path).astype(np.float64)

            # Прореживаем и добавляем шум
            subsampled_data = preprocess_seismic_data(seismic_data, nsub, norm)

            # Сохраняем прореженные данные
            np.save(output_path, subsampled_data)

            print(f"[OK] {filename:20s}")

    print(f"Прореженные данные сохранены в папке {output_folder}.")