from perf import *

import os
import heapq
import math
import time
from functools import partial
from datetime import datetime
from collections import OrderedDict
from argparse import ArgumentParser

import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch import multiprocessing as mp

import utils
from neumf import NeuMF
from dataset import CFTrainDataset, load_test_ratings, load_test_negs
from convert import (TEST_NEG_FILENAME, TEST_RATINGS_FILENAME,
                     TRAIN_RATINGS_FILENAME)


def parse_args():
    parser = parser_base(description="Train a Nerual Collaborative Filtering model")

    parser.add_argument('data', type=str,
                        help='path to test and training data files')

    parser.add_argument('-f', '--factors', type=int, default=8,
                        help='number of predictive factors')

    parser.add_argument('--layers', nargs='+', type=int, default=[64, 32, 16, 8],
                        help='size of hidden layers for MLP')

    parser.add_argument('-n', '--negative-samples', type=int, default=4,
                        help='number of negative examples per interaction')

    parser.add_argument('-l', '--learning-rate', type=float, default=0.001,
                        help='learning rate for optimizer')

    parser.add_argument('-k', '--topk', type=int, default=10,
                        help='rank for test examples to be considered a hit')

    parser.add_argument('--threshold', '-t', type=float,
                        help='stop training early at threshold')

    parser.add_argument('--processes', '-p', type=int, default=1,
                        help='Number of processes for evaluating model')

    return parser


def predict(model, users, items, batch_size=1024, use_cuda=True):
    batches = [(users[i:i + batch_size], items[i:i + batch_size])
               for i in range(0, len(users), batch_size)]
    preds = []
    for user, item in batches:
        def proc(x):
            x = np.array(x)
            x = torch.from_numpy(x)
            if use_cuda:
                x = x.cuda(non_blocking=True)
            return torch.autograd.Variable(x)
        outp = model(proc(user), proc(item), sigmoid=True)
        outp = outp.data.cpu().numpy()
        preds += list(outp.flatten())
    return preds


def _calculate_hit(ranked, test_item):
    return int(test_item in ranked)


def _calculate_ndcg(ranked, test_item):
    for i, item in enumerate(ranked):
        if item == test_item:
            return math.log(2) / math.log(i + 2)
    return 0.


def eval_one(rating, items, model, K, use_cuda=True):
    user = rating[0]
    test_item = rating[1]
    items.append(test_item)
    users = [user] * len(items)
    predictions = predict(model, users, items, use_cuda=use_cuda)

    map_item_score = {item: pred for item, pred in zip(items, predictions)}
    ranked = heapq.nlargest(K, map_item_score, key=map_item_score.get)

    hit = _calculate_hit(ranked, test_item)
    ndcg = _calculate_ndcg(ranked, test_item)
    return hit, ndcg, len(predictions)


def main():
    # Note: The run start is in convert.py

    exp = Experiment(__file__)
    args = exp.get_arguments(parse_args(), show=True)
    device = exp.get_device()
    chrono = exp.chrono()

    # Save configuration to file
    config = {k: v for k, v in args.__dict__.items()}
    config['timestamp'] = "{:.0f}".format(datetime.utcnow().timestamp())
    config['local_timestamp'] = str(datetime.now())

    tmp = os.environ['TEMP_DIRECTORY']
    run_dir = "{}/run/neumf/{}".format(tmp, config['timestamp'])

    print("Saving config and results to {}".format(run_dir))

    if run_dir != '':
        os.makedirs(run_dir, exist_ok=True)

    utils.save_config(config, run_dir)

    # Load Data
    # ------------------------------------------------------------------------------------------------------------------
    print('Loading data')
    with chrono.time('loading_data', skip_obs=0):
        t1 = time.time()

        train_dataset = CFTrainDataset(os.path.join(args.data, TRAIN_RATINGS_FILENAME), args.negative_samples)

        # mlperf_log.ncf_print(key=# mlperf_log.INPUT_BATCH_SIZE, value=args.batch_size)
        # mlperf_log.ncf_print(key=# mlperf_log.INPUT_ORDER)  # set shuffle=True in DataLoader
        train_dataloader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=True)

        nb_users, nb_items = train_dataset.nb_users, train_dataset.nb_items

        print('Load data done [%.1f s]. #user=%d, #item=%d, #train=%d'
              % (time.time()-t1, nb_users, nb_items, train_dataset.mat.nnz))
    # ------------------------------------------------------------------------------------------------------------------

    # Create model
    model = NeuMF(nb_users, nb_items,
                  mf_dim=args.factors, mf_reg=0.,
                  mlp_layer_sizes=args.layers,
                  mlp_layer_regs=[0. for i in args.layers]).to(device)
    print(model)
    print("{} parameters".format(utils.count_parameters(model)))

    # Save model text description
    with open(os.path.join(run_dir, 'model.txt'), 'w') as file:
        file.write(str(model))

    # Add optimizer and loss to graph
    # mlperf_log.ncf_print(key=# mlperf_log.OPT_LR, value=args.learning_rate)
    beta1, beta2, epsilon = 0.9, 0.999, 1e-8

    optimizer = torch.optim.Adam(
        model.parameters(),
        betas=(beta1, beta2),
        lr=args.learning_rate, eps=epsilon)

    # mlperf_log.ncf_print(key=# mlperf_log.MODEL_HP_LOSS_FN, value=# mlperf_log.BCE)
    criterion = nn.BCEWithLogitsLoss().to(device)

    model.train()

    for epoch in range(args.repeat):
        losses = utils.AverageMeter()

        with chrono.time('train') as t:

            for batch_index, (user, item, label) in enumerate(train_dataloader):
                if batch_index > args.number:
                    break

                user = torch.autograd.Variable(user, requires_grad=False).to(device)
                item = torch.autograd.Variable(item, requires_grad=False).to(device)
                label = torch.autograd.Variable(label, requires_grad=False).to(device)

                outputs = model(user, item)
                loss = criterion(outputs, label)
                exp.log_batch_loss(loss.item())
                losses.update(loss.item(), user.size(0))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        exp.log_epoch_loss(losses.sum)

        # Save stats to file
        exp.show_eta(epoch, t)

    exp.report()


if __name__ == '__main__':
    main()
