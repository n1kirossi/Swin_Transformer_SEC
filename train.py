# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.distributed as dist
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from util.dataset import SeismicHDF5Dataset


def setup(device, cnt_device):
    dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:29500', rank=device, world_size=cnt_device)
    torch.cuda.set_device(device)


def cleanup():
    dist.destroy_process_group()


def main(device, cnt_device):
    setup(device, cnt_device)
    TRAIN_PATH = 'inputs/data/train-4-128.hdf5'
    VAL_PATH = 'inputs/data/val-4-128.hdf5'
    MODEL_SAVE_PATH = 'trained_model/models/train_model_second_data_result.pth'

    global_batch_size = 96
    local_batch_size = global_batch_size // cnt_device
    epochs = 200
    learning_rate = 1e-4

    d_train = SeismicHDF5Dataset(TRAIN_PATH)
    d_val = SeismicHDF5Dataset(VAL_PATH)

    train_sampler = DistributedSampler(d_train, num_replicas=cnt_device, rank=device, shuffle=True)
    val_sampler = DistributedSampler(d_val, num_replicas=cnt_device, rank=device, shuffle=False)

    train_data = data.DataLoader(d_train, batch_size=local_batch_size, sampler=train_sampler, num_workers=2, pin_memory=True)
    val_data = data.DataLoader(d_val, batch_size=local_batch_size, sampler=val_sampler, num_workers=2, pin_memory=True)

    from model.SCRN import SCRN
    model = SCRN().to(device)
    model = DDP(model, device_ids=[device], output_device=device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 80, 120, 160], gamma=0.2)
    loss_func = nn.MSELoss(reduction='sum')

    # patience = 14
    # best_val_loss = float('inf')
    # epochs_no_improve = 0

    loss_train, loss_val = [], []
    for _e in range(epochs):
        train_sampler.set_epoch(_e)

        model.train()
        epoch_loss = 0
        n_batches = 0

        if device == 0:
            pbar = tqdm(train_data, desc=f'Epoch {_e + 1}/{epochs}')
        else:
            pbar = train_data
        
        for x, y in pbar:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            pred = model(x)
            loss = loss_func(pred, y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1
            if device == 0:
                pbar.set_postfix(loss=loss.item())
    
        train_loss = epoch_loss / len(d_train)
        loss_train.append(train_loss)

        model.eval()
        epoch_loss = 0
        with torch.no_grad():
            for x, y in val_data:
                x, y = x.to(device), y.to(device)

                pred = model(x)
                loss = loss_func(pred, y)

                epoch_loss += loss.item()

        val_loss = epoch_loss / len(d_val)
        loss_val.append(val_loss)

        scheduler.step()

        if device == 0:
            print(f'Epoch {_e + 1}/{epochs}  |  Train loss: {train_loss:.6f}  |  Val loss: {val_loss:.6f}')
            print()

            # if val_loss < best_val_loss:
                # best_val_loss = val_loss
                # epochs_no_improve = 0

            EVERY_MODEL_SAVE_PATH = f'trained_model/models/train_model_second_data_{_e + 1}.pth'
            torch.save(model.module.state_dict(), EVERY_MODEL_SAVE_PATH)
            print(f'Модель на шаге {_e + 1} сохранена в {EVERY_MODEL_SAVE_PATH}')
            # else:
            #     epochs_no_improve += 1

        # stop = torch.tensor(epochs_no_improve >= patience, device=device)
        # dist.broadcast(stop, src=0)
        # if stop.item():
        #     if device == 0:
        #         print(f'Ранняя остановка произошла на {_e + 1} эпохе. Валидационная ошибка не менялась {patience} эпох.')
        #     break
    
    if device == 0:
        torch.save(model.module.state_dict(), MODEL_SAVE_PATH)
        print(f'Модель сохранена в {MODEL_SAVE_PATH}')

        plt.figure()
        plt.plot(loss_train, label='Train')
        plt.plot(loss_val, label='Val')
        plt.xlabel('Epoch')
        plt.ylabel('MSE / N')
        plt.legend()
        plt.grid(True)
        plt.savefig('train_val_loss.png')
    
    cleanup()


if __name__ == '__main__':
    cnt_device = torch.cuda.device_count()

    if cnt_device < 2:
        print('Для DDP нужно минимум 2 GPU. Ошибка')
        exit(1)
    
    print(f'Запуск DDP на {cnt_device} GPU')

    import torch.multiprocessing as mp
    mp.spawn(main, args=(cnt_device, ), nprocs=cnt_device, join=True)
