import json
from datetime import datetime
import os
import numpy as np


def preprocess_seismic_data_scrn(
        seismic_data: np.ndarray,
        drop_rates=(0.02, 0.08, 0.18, 0.28, 0.38),
        snr_db_choices=(-2, -1, 1, 5, 10),  # оставлен для совместимости, но не используется
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

    # ---------- 1. Irregular down-sampling (column drop) ----------
    drop_rate = float(rng.choice(drop_rates))
    n_drop = int(drop_rate * n_traces)

    mask = np.ones(n_traces, dtype=np.float32)
    if n_drop > 0:
        mask[:n_drop] = 0.0
        rng.shuffle(mask)  # in-place permutation

    # Broadcast mask to all time samples
    corrupted = data * mask

    # ---------- 2. Шум не добавляем ----------
    snr_db = None

    return corrupted.astype(seismic_data.dtype), mask, snr_db, drop_rate


# ——— пользовательские параметры ——————————————————————————————
INPUT_FOLDER = "inputs/data/clean_data"          # исходные *.npy*
OUTPUT_FOLDER = "inputs/data/irregular_thinned_data"   # куда кладём результаты
SAVE_MASK     = False           # сохранять ли файл "xxx_mask.npy"
SEED          = None             # None = полностью случайно
DROP_RATES    = (0.02, 0.08, 0.18, 0.28, 0.38)
# ---------------------------------------------------------------------------


def main() -> None:
    rng = np.random.default_rng(SEED)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    meta_log = []  # сюда сложим информацию по каждому файлу

    for fname in sorted(os.listdir(INPUT_FOLDER)):
        if not fname.endswith(".npy"):
            continue

        in_path  = os.path.join(INPUT_FOLDER, fname)
        out_path = os.path.join(OUTPUT_FOLDER, fname)

        # ---------- load -----------------------------------------------------
        data = np.load(in_path).astype(np.float32)   # (n_samples, n_traces)

        # ---------- corrupt (теперь только прореживание) ---------------------

        y, mask, _, rate = preprocess_seismic_data_scrn(
            data,
            drop_rates=DROP_RATES,
            rng=rng,
            norm=False
        )

        # ---------- save -----------------------------------------------------
        np.save(out_path, y)
        if SAVE_MASK:
            mask_path = os.path.join(
                OUTPUT_FOLDER,
                f"{os.path.splitext(fname)[0]}_mask.npy",
            )
            # сохраняем маску, если хочешь включить — просто раскомментируй строку ниже
            # np.save(mask_path, mask.astype(np.uint8))

        # ---------- log ------------------------------------------------------
        meta_log.append(
            dict(
                file      = fname,
                drop_rate = float(rate),
                n_traces  = data.shape[1],
                n_zeroed  = int(rate * data.shape[1]),
            )
        )

        print(f"[OK] {fname:20s}  drop={rate:>5.1%}")

    # ---------- save JSON summary -------------------------------------------
    summary_path = os.path.join(OUTPUT_FOLDER, "preprocess_log.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            dict(
                created    = datetime.now().isoformat(timespec="seconds"),
                seed       = SEED,
                drop_rates = DROP_RATES,
                files      = meta_log,
            ),
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\nГотово! Результаты и лог сохранены в '{OUTPUT_FOLDER}'.")
    print(f"Всего обработано файлов: {len(meta_log)}")


if __name__ == "__main__":
    main()