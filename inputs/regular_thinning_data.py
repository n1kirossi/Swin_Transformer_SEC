import os
import numpy as np


def preprocess_seismic_data(seismic_data, nsub, norm):
    data = seismic_data.astype(np.float32)

    if norm:
        mean, std = data.mean(), data.std()
        data = np.clip((data - mean) / (5 * (std + 1e-10)), -1, 1)

    mask = np.zeros(data.shape[1], dtype=np.float32)
    mask[::nsub] = 1.0
    return mask * data


if __name__ == '__main__':
    CLEAN_DIR = 'inputs/data/clean_data'
    THINNED_DIR = 'inputs/data/regular_thinned_data'
    input_folder = os.path.join(CLEAN_DIR)
    output_folder = os.path.join(THINNED_DIR)

    nsub = 2
    norm = False

    for filename in os.listdir(input_folder):
        if filename.endswith(".npy"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            seismic_data = np.load(input_path).astype(np.float64)

            subsampled_data = preprocess_seismic_data(seismic_data, nsub, norm)

            np.save(output_path, subsampled_data)

            print(f"{filename:20s}")

    print(f"Прореженные данные сохранены в папке {output_folder}.")