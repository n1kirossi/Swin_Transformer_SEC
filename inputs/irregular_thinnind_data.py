import json
from datetime import datetime
import os
import numpy as np

INPUT_FOLDER = "inputs/data/clean_data"
OUTPUT_FOLDER = "inputs/data/irregular_thinned_data"
SAVE_MASK = False
SEED = None
DROP_RATES = (0.02, 0.08, 0.18, 0.28, 0.38)


def preprocess_seismic_data_scrn(
        seismic_data: np.ndarray,
        drop_rates=(0.02, 0.08, 0.18, 0.28, 0.38),
        snr_db_choices=(-2, -1, 1, 5, 10),
        rng: np.random.Generator | None = None,
        norm=False,
):
    if rng is None:
        rng = np.random.default_rng()

    data = seismic_data.astype(np.float32)

    if norm:
        mean, std = data.mean(), data.std()
        data = np.clip((data - mean) / (5 * (std + 1e-10)), -1, 1)
    
    _, n_traces = data.shape

    drop_rate = float(rng.choice(drop_rates))
    n_drop = int(drop_rate * n_traces)

    mask = np.ones(n_traces, dtype=np.float32)
    if n_drop > 0:
        mask[:n_drop] = 0.0
        rng.shuffle(mask)

    corrupted = data * mask

    snr_db = None

    return corrupted.astype(seismic_data.dtype), mask, snr_db, drop_rate


def main() -> None:
    rng = np.random.default_rng(SEED)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    meta_log = []

    for fname in sorted(os.listdir(INPUT_FOLDER)):
        if not fname.endswith(".npy"):
            continue

        in_path  = os.path.join(INPUT_FOLDER, fname)
        out_path = os.path.join(OUTPUT_FOLDER, fname)

        data = np.load(in_path).astype(np.float32)

        y, mask, _, rate = preprocess_seismic_data_scrn(
            data,
            drop_rates=DROP_RATES,
            rng=rng,
            norm=False
        )

        np.save(out_path, y)
        if SAVE_MASK:
            mask_path = os.path.join(
                OUTPUT_FOLDER,
                f"{os.path.splitext(fname)[0]}_mask.npy",
            )

        meta_log.append(
            dict(
                file = fname,
                drop_rate = float(rate),
                n_traces = data.shape[1],
                n_zeroed = int(rate * data.shape[1]),
            )
        )

        print(f"[OK] {fname:20s}  drop={rate:>5.1%}")

    summary_path = os.path.join(OUTPUT_FOLDER, "preprocess_log.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            dict(
                created = datetime.now().isoformat(timespec="seconds"),
                seed = SEED,
                drop_rates = DROP_RATES,
                files = meta_log,
            ),
            f,
            indent = 2,
            ensure_ascii = False,
        )

    print(f"\nГотово! Результаты и лог сохранены в '{OUTPUT_FOLDER}'.")
    print(f"Всего обработано файлов: {len(meta_log)}")


if __name__ == "__main__":
    main()
