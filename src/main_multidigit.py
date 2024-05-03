import numpy as np
import argparse
import random
import os
import logging
from datetime import datetime
from tqdm import tqdm
import torch
import torch.nn.functional as F

from error_free_dsf import DiffSortNet

from datasets.dataset import MultiDigitSplits
from models import models_cnn
from models import models_transformer
import utils


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--num_compare', type=int, default=5)
    parser.add_argument('--num_steps', type=int, default=100_000)
    parser.add_argument('--eval_freq', type=int, default=1_000)
    parser.add_argument('--method', type=str, default='odd_even', choices=['odd_even', 'bitonic'])
    parser.add_argument('--distribution', type=str, default='optimal', choices=[
        'cauchy',
        'reciprocal',
        'optimal',
        'gaussian',
        'logistic',
        'logistic_phi',
    ])
    parser.add_argument('--steepness', type=float, default=10)
    parser.add_argument('--art_lambda', type=float, default=0.25)
    parser.add_argument('--dataset', type=str, default='mnist_cnn', choices=[
        'mnist_cnn',
        'svhn_cnn',
        'mnist_transformer',
        'svhn_transformer',
        'mnist_transformer_large',
        'svhn_transformer_large',
    ])
    parser.add_argument('--nloglr', type=float, default=3.5)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--use_ste', action='store_true')
    parser.add_argument('--use_soft', action='store_true')
    parser.add_argument('--loss_weight', type=float, default=1.0)

    args = parser.parse_args()
    str_time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    if not os.path.exists('../logs'):
        os.mkdir('../logs')

    if args.use_ste:
        str_use_ste = 'ste'

        if args.use_soft:
            str_use_ste += '_also_soft'
    else:
        str_use_ste = 'no_ste'

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(f'../logs/log_{args.dataset}_{args.num_compare}_{str_use_ste}_{args.seed}_{str_time}.log'),
            logging.StreamHandler()
        ]
    )

    ##
    logging.info('=============================')
    logging.info('config')
    logging.info('=============================')
    logging.info(f'batch_size: {args.batch_size}')
    logging.info(f'num_compare: {args.num_compare}')
    logging.info(f'num_steps: {args.num_steps}')
    logging.info(f'eval_freq: {args.eval_freq}')
    logging.info(f'method: {args.method}')
    logging.info(f'distribution: {args.distribution}')
    logging.info(f'steepness: {args.steepness}')
    logging.info(f'art_lambda: {args.art_lambda}')
    logging.info(f'dataset: {args.dataset}')
    logging.info(f'nloglr: {args.nloglr}')
    logging.info(f'device: {args.device}')
    logging.info(f'seed: {args.seed}')
    logging.info(f'gpu: {args.gpu}')
    logging.info(f'use_ste: {args.use_ste}')
    logging.info(f'use_soft: {args.use_soft}')
    logging.info(f'loss_weight: {args.loss_weight}')
    logging.info('=============================')
    ##

    if args.device == 'cuda':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    best_valid_acc = 0.0

    list_acc_em = []
    list_acc_ew = []

    if args.device == 'cuda' and not torch.cuda.is_available():
        logging.info('----------------------------------------------------')
        logging.info('--- WARNING: No GPU detected, running on CPU ... ---')
        logging.info('----------------------------------------------------')
        args.device = 'cpu'

    if args.dataset == 'mnist_cnn':
        model = models_cnn.MultiDigitMNISTNet().to(args.device)
        dummy_inputs = torch.randn((1, args.num_compare, 1, 28 * 4, 28)).to(args.device)
        args.dataset = 'mnist'
    elif args.dataset == 'svhn_cnn':
        model = models_cnn.SVHNConvNet().to(args.device)
        dummy_inputs = torch.randn((1, args.num_compare, 3, 54, 54)).to(args.device)
        args.dataset = 'svhn'
    elif args.dataset == 'mnist_transformer':
        num_last = 16
        model = models_transformer.MultiDigitMNISTNet(num_last=num_last).to(args.device)
        dummy_inputs = torch.randn((1, args.num_compare, 1, 28 * 4, 28)).to(args.device)
        args.dataset = 'mnist'
    elif args.dataset == 'svhn_transformer':
        num_last = 32
        model = models_transformer.SVHNNet(num_last=num_last).to(args.device)
        dummy_inputs = torch.randn((1, args.num_compare, 3, 54, 54)).to(args.device)
        args.dataset = 'svhn'
    elif args.dataset == 'mnist_transformer_large':
        num_last = 64
        num_layers = 8
        model = models_transformer.MultiDigitMNISTNet(num_last=num_last, num_layers=num_layers).to(args.device)
        dummy_inputs = torch.randn((1, args.num_compare, 1, 28 * 4, 28)).to(args.device)
        args.dataset = 'mnist'
    elif args.dataset == 'svhn_transformer_large':
        num_last = 64
        num_layers = 8
        is_large = True
        model = models_transformer.SVHNNet(num_last=num_last, num_layers=num_layers, is_large=is_large).to(args.device)
        dummy_inputs = torch.randn((1, args.num_compare, 3, 54, 54)).to(args.device)
        args.dataset = 'svhn'
    else:
        raise ValueError(args.dataset)

    splits = MultiDigitSplits(dataset=args.dataset, num_compare=args.num_compare, seed=args.seed)

    # drop_last needs to be true, otherwise error with testing for SVHN
    loader_kwargs = dict(batch_size=args.batch_size, drop_last=True)

    if args.dataset == 'svhn':
        loader_kwargs['batch_size'] = args.batch_size * args.num_compare

        def collate_fn(batch):
            data, targets = zip(*batch)
            data = torch.stack(data)
            targets = torch.tensor(targets)  # targets is a tuple of int
            data = data.reshape(args.batch_size, args.num_compare, *data.shape[1:])
            targets = targets.reshape(args.batch_size, args.num_compare, *targets.shape[1:])
            return data, targets

        loader_kwargs['collate_fn'] = collate_fn

    data_loader_train = splits.get_train_loader(**loader_kwargs)
    data_loader_valid = splits.get_valid_loader(**loader_kwargs)
    data_loader_test = splits.get_test_loader(**loader_kwargs)

    optim = torch.optim.AdamW(model.parameters(), lr=10**(-args.nloglr))

    sorter = DiffSortNet(
        sorting_network_type=args.method,
        size=args.num_compare,
        device=args.device,
        steepness=args.steepness,
        art_lambda=args.art_lambda,
        use_ste=args.use_ste,
    )

    if args.use_ste:
        sorter_soft = DiffSortNet(
            sorting_network_type=args.method,
            size=args.num_compare,
            device=args.device,
            steepness=args.steepness,
            art_lambda=args.art_lambda,
            use_ste=False,
        )

    valid_accs = []
    test_acc = None

    for iter_idx, (data, targets) in tqdm(
        enumerate(utils.load_n(data_loader_train, args.num_steps)),
        desc="Training steps",
        total=args.num_steps,
    ):
        model.train()

        data = data.to(args.device) # (bs, n, c, h, w)
        targets = targets.to(args.device)
        outputs = model(data).squeeze(2)

        perm_ground_truth = F.one_hot(torch.argsort(targets, dim=-1)).transpose(-2, -1).float()

        if not args.use_ste:
            _, perm_prediction_soft = sorter(outputs)

            loss = torch.nn.BCELoss()(perm_prediction_soft, perm_ground_truth)
        else:
            loss = 0.0

            if args.use_soft:
                _, perm_prediction_soft = sorter_soft(outputs)
                loss_diffsort = torch.nn.BCELoss()(perm_prediction_soft, perm_ground_truth)
                loss += loss_diffsort

            _, perm_prediction_hard = sorter(outputs)

            data_vectorized = torch.reshape(data, (data.shape[0], data.shape[1], -1))

            gap = 4
            indices_split = torch.arange(0, args.num_compare, gap)
            loss_ours = 0.0

            for ind in indices_split:
                _gap = gap + (args.num_compare == (ind + gap + 1))

                # TODO: remove the case with size 1
                _perm_gt = perm_ground_truth[:, ind:ind+_gap, :]
                shape_perm_gt = _perm_gt.shape[1]
                _perm_gt = _perm_gt[torch.tile(torch.abs(torch.sum(_perm_gt, dim=1).unsqueeze(1) - 1) < 1e-2, (1, shape_perm_gt, 1))]
                _perm_gt = torch.reshape(_perm_gt, (args.batch_size, shape_perm_gt, shape_perm_gt))

                _perm_preds = perm_prediction_hard[:, ind:ind+_gap, :]
                shape_perm_preds = _perm_preds.shape[1]
                _perm_preds = _perm_preds[torch.tile(torch.abs(torch.sum(_perm_preds, dim=1).unsqueeze(1) - 1) < 1e-2, (1, shape_perm_preds, 1))]
                _perm_preds = torch.reshape(_perm_preds, (args.batch_size, shape_perm_preds, shape_perm_preds))

                assert np.all(_perm_gt.shape == _perm_preds.shape)

                res_gt = torch.bmm(_perm_gt.transpose(-2, -1), data_vectorized[:, ind:ind+_gap]).squeeze(2)
                res_preds = torch.bmm(_perm_preds.transpose(-2, -1), data_vectorized[:, ind:ind+_gap]).squeeze(2)

                loss_ours += torch.nn.MSELoss()(res_preds, res_gt)

            loss_ours *= args.loss_weight
            loss += loss_ours

        optim.zero_grad()
        loss.backward()
        optim.step()

        if (iter_idx + 1) % 50000 == 0:
            for g in optim.param_groups:
                g['lr'] = g['lr'] / 2.0

        if (iter_idx + 1) % args.eval_freq == 0:
            model.eval()

            current_valid_accs = []
            for data, targets in data_loader_valid:
                data, targets = data.to(args.device), targets.to(args.device)
                current_valid_accs.append(utils.ranking_accuracy(model, data, targets))
            valid_accs.append(utils.avg_list_of_dicts(current_valid_accs))

            logging.info(f'STEP {iter_idx+1} VALID: {valid_accs[-1]}')
            logging.info(f'STEP {iter_idx+1} VALID: acc_em {valid_accs[-1]["acc_em"]:.4f} acc_ew {valid_accs[-1]["acc_ew"]:.4f} acc_em5 {valid_accs[-1]["acc_em5"]:.4f}')

            if valid_accs[-1]['acc_em5'] > best_valid_acc:
                best_valid_acc = valid_accs[-1]['acc_em5']

                current_test_accs = []
                for data, targets in data_loader_test:
                    data, targets = data.to(args.device), targets.to(args.device)
                    current_test_accs.append(utils.ranking_accuracy(model, data, targets))
                test_acc = utils.avg_list_of_dicts(current_test_accs)

                list_acc_em.append(test_acc['acc_em'])
                list_acc_ew.append(test_acc['acc_ew'])

                logging.info(f'STEP {iter_idx+1} TEST: {test_acc}')
                logging.info(f'STEP {iter_idx+1} TEST: acc_em {test_acc["acc_em"]:.4f} acc_ew {test_acc["acc_ew"]:.4f} acc_em5 {test_acc["acc_em5"]:.4f}')

                ind_max = np.argmax(list_acc_em)
                logging.info(f'STEP {iter_idx+1} BEST TEST: acc_em {list_acc_em[ind_max]:.4f} acc_ew {list_acc_ew[ind_max]:.4f}')

    logging.info(f'FINAL TEST: acc_em {test_acc["acc_em"]:.4f} acc_ew {test_acc["acc_ew"]:.4f} acc_em5 {test_acc["acc_em5"]:.4f}')

    ind_max = np.argmax(list_acc_em)
    logging.info(f'BEST TEST: acc_em {list_acc_em[ind_max]:.4f} acc_ew {list_acc_ew[ind_max]:.4f}')
