import argparse, collections, copy, datetime, functools, itertools
import json, logging, os, pathlib, sys, time, typing
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt; plt.style.use('bmh')
import torch
import torchvision

from dataset import Dataset
from plot import confusion_matrix
from trainer import Trainer
from metrics import auroc_metric, sensitivity_metric
from utils import first
from plot import plot_curve
import resnet

if __name__ == '__main__':
    START = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-a', '--arch', default='resnet34',
        help='torchvision network architecture',
    )
    parser.add_argument(
        '-b', '--base',
        #default='/mnt/dfs/qtim/clu/data/mammo_density_dmist/',
        default='/mnt/dfs/qtim/clu/data/skin/data',
        help='path to image data directory',
    )
    parser.add_argument(
        '-bs', '--batch_size', default=16,
        type=int, help='batch size to use',
    )
    parser.add_argument(
        '-c', '--csv',
        #default='/mnt/dfs/qtim/clu/data/mammo_density_dmist/dmist_info.csv',
        #default='/mnt/dfs/qtim/clu/data/skin/skin-info.csv',
        default='/mnt/dfs/qtim/clu/data/skin/skin-info_without_excluded_images.csv',
        help='path to csv file',
    )
    parser.add_argument(
        '-d', '--debug', action='store_true', help='for debugging purposes',
    )
    parser.add_argument(
        '-dr', '--dropout_rate', default=0.1, type=float, help='rate of dropout probability',
    )
    parser.add_argument(
        '-e', '--epochs', default=100, type=int, help='number of epochs to use',
    )
    parser.add_argument(
        '-es', '--early_stop', default=5, type=int,
        help='patience for early stopping'
    )
    parser.add_argument(
        '-eq', '--equal_sampling', action='store_true',
        help='whether to use equal sampling based on subgroup'
    )
    parser.add_argument(
        '-gpu', '--multi_gpu', action='store_true',
        help='whether to use multiple GPUs'
    )
    parser.add_argument(
        '-l', '--label',
        #default='density',
        #default='nine_partition_label',
        default='raw_label',
        help='label to use (should be column in dataframe)'
    )
    parser.add_argument(
        '-lr', '--learning_rate', default=1e-4, type=float, help='learning rate to use',
    )
    parser.add_argument(
        '-mc', '--monte_carlo', default=30, type=int, help='number of mc runs',
    )
    parser.add_argument(
        '-nw', '--num_workers', default=8, type=int,
        help='number of workers to use in dataloader'
    )
    parser.add_argument(
        '-o', '--output', default='.', help='path to write out all outputs'
    )
    #parser.add_argument(
        #'-p', '--pretrained', action='store_true',
        #help='whether to use pretrained model on imagenet'
    #)
    parser.add_argument(
        '-s', '--subgroup',
        #default='race',
        default='fitzpatrick',
        help='subgroup to use (should be column in dataframe)'
    )
    parser.add_argument(
        '-t', '--tag', default=None, help='append tag to output directory name'
    )
    parser.add_argument(
        '-r', '--runs', default=10, type=int, help='number of runs'
    )
    args = parser.parse_args()

    date = datetime.datetime.now().strftime('%Y_%b_%d_%H_%M_%S').lower()
    dir_name =  'debug' if args.debug else f'run_{date}'
    if args.tag is not None:
        dir_name = f'{args.tag}-' + dir_name

    run_dir = pathlib.Path(args.output) / dir_name
    run_dir.mkdir(exist_ok=True, parents=True)

    with open(run_dir / 'args.json', 'w') as f:
        json.dump(args.__dict__, f, indent=4)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(run_dir / f'log.txt')
    file_handler.setFormatter(
        logging.Formatter(
            '%(asctime)s:%(levelname)s %(message)s',
            datefmt='%H:%M:%S'
        )
     )
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    logger.addHandler(console_handler)

    df = pd.read_csv(args.csv)
    classes = sorted(getattr(df, args.label).unique())
    class_map = dict(zip(classes, itertools.count()))
    groups = sorted(getattr(df, args.subgroup).unique())
    group_map = dict(zip(groups, itertools.count()))
    logger.info(f'--- class_map: {class_map}')
    logger.info(f'--- group_map: {group_map}')
    df['label'] = df.apply(lambda row: class_map[getattr(row, args.label)], axis=1)
    df['subgroup'] = df.apply(lambda row: group_map[getattr(row, args.subgroup)], axis=1)

    dev_df = df.sample(frac=0.5)
    test_df = df[~df.index.isin(dev_df.index)]
    train_df = dev_df.sample(frac=0.7)
    valid_df = dev_df[~dev_df.index.isin(train_df.index)]

    if args.debug:
        n = 64
        train_df = train_df.sample(n=n)
        valid_df = valid_df.sample(n=n)
        test_df = test_df.sample(n=n)

    logger.info(f'--- number training:   {len(train_df)}')
    logger.info(f'--- number validation: {len(valid_df)}')
    logger.info(f'--- number testing:    {len(test_df)}')

    for rnd in range(2 if args.debug else args.runs):
        logger.info(f'-- Random state: {rnd}')
        rnd_dir = run_dir / f'seed_{rnd}'
        rnd_dir.mkdir(exist_ok=True, parents=True)
        np.random.seed(rnd)
        torch.manual_seed(rnd)

        # =========================== DATA =========================== #
        columns = ['image', 'label', 'subgroup']
        train_ds = Dataset(
            data=train_df[columns].to_dict(orient='records'),
            base=args.base,
            augment=True,
        )
        valid_ds = Dataset(
            data=valid_df[columns].to_dict(orient='records'),
            base=args.base,
        )
        test_ds = Dataset(
            data=test_df[columns].to_dict(orient='records'),
            base=args.base,
        )
        if args.equal_sampling:
            group_counts = train_df.subgroup.value_counts(normalize=True).to_dict()

            weights = np.array([1 / v for k, v in sorted(group_counts.items())])

            batch_sampler = get_weighted_sampler(
                weights[train_df.subgroup.values],
                batch_size=args.batch_size,
            )
            train_dl = torch.utils.data.DataLoader(
                train_ds,
                batch_sampler=batch_sampler,
                shuffle=True,
                pin_memory=True,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            )
        else:
            train_dl = torch.utils.data.DataLoader(
                train_ds,
                shuffle=True,
                pin_memory=True,
                batch_size=args.batch_size,
                num_workers=args.num_workers
            )
        valid_dl = torch.utils.data.DataLoader(
            valid_ds,
            shuffle=True,
            pin_memory=True,
            num_workers=args.num_workers,
            batch_size=args.batch_size,
        )
        valid_dl = torch.utils.data.DataLoader(
            valid_ds,
            shuffle=True,
            pin_memory=True,
            num_workers=args.num_workers,
            batch_size=args.batch_size,
        )
        test_dl = torch.utils.data.DataLoader(
            test_ds,
            shuffle=True,
            pin_memory=True,
            num_workers=args.num_workers,
            batch_size=args.batch_size,
        )

        # ========================== MODEL =========================== #
        arch = getattr(resnet, args.arch)
        model = arch(pretrained=True, dropout_rate=args.dropout_rate)
        if hasattr(model, 'fc'):
            model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
        else:
            raise ValueError('what is last layer called')
        model = model.to('cuda')
        if args.multi_gpu:
            model = torch.nn.DataParallel(model)

        # =========================== LOSS =========================== #
        #  weight = torch.tensor(param['loss_weight'], dtype=torch.float).cuda()
        criterion = torch.nn.CrossEntropyLoss(reduction='mean', weight=None)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            verbose=True,
        )

        # =========================== TRAIN ========================= #
        trainer = Trainer(
            model=model,
            train_loader=train_dl,
            valid_loader=valid_dl,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            metrics={
                'auroc': auroc_metric,
                'sensitivity': sensitivity_metric,
            },
        )

        best_loss, best_epoch, patience = float('inf'), 0, 0
        epoch_train_loss, epoch_valid_loss = [], []
        epoch_train_metric, epoch_valid_metric = [], []
        for e in range(args.epochs if not args.debug else 1):
            train_res = trainer.train()
            train_loss = train_res['loss']
            train_metrics = train_res['metrics']

            valid_res = trainer.validate()
            valid_loss = valid_res['loss']
            valid_metrics = valid_res['metrics']

            epoch_train_loss.append(train_loss)
            epoch_valid_loss.append(valid_loss)

            logger.info(train_loss)
            logger.info(valid_loss)

            if valid_loss < best_loss:
                best_loss = valid_loss
                best_epoch = e
                patience = 0
                logger.info(f'    Saved checkpoint ==> {best_loss}')
                model_name = f'checkpoint-{round(best_loss, 4)}-{str(e).zfill(3)}.pth'
                trainer.save(rnd_dir / model_name, epoch=e)
            else:
                patience += 1
            if args.early_stop is not None and patience > args.early_stop:
                logger.info(f'    Stopped at {e}')
                break

        logger.info('    Finished training')
        logger.info(f'    Best checkpoint epoch {best_epoch} - {best_loss}')


        # =========================== TEST =========================== #
        valid_dl.dataset.return_meta_data = True
        if args.monte_carlo is not None:
            valid_res = trainer.monte_carlo_inference(
                test_loader=valid_dl,
                num_iter=args.monte_carlo,
            )
        else:
            valid_res = trainer.evaluate(test_loader=valid_dl)

        with open(rnd_dir / f'valid-res.json', 'w') as f:
            json.dump(valid_res, f, indent=4, default=int)

        test_dl.dataset.return_meta_data = True
        if args.monte_carlo is not None:
            test_res = trainer.monte_carlo_inference(
                test_loader=test_dl,
                num_iter=args.monte_carlo,
            )
        else:
            test_res = trainer.evaluate(test_loader=test_dl)

        with open(rnd_dir / f'test-res.json', 'w') as f:
            json.dump(test_res, f, indent=4, default=int)

        # =========================== PLOT =========================== #
        plot_curve(
            train_loss, valid_loss,
            train_label='train loss',
            valid_label='valid loss',
            save_path = rnd_dir / 'loss_curve.png',
        )

    logger.info(f'=-=-= total runtime: {time.perf_counter() - START:.0f}s =-=-=')

