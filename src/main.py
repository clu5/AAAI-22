# ==================================================================== #
#                                                                      #
#                          ENTRY  POINT                               #
#                                                                      #
# ==================================================================== #

import argparse, collections, copy, datetime, functools, itertools
import json, logging, os, pathlib, sys, time, typing
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch 
import torchvision
import monai
import matplotlib.pyplot as plt; plt.style.use('bmh')
import matplotlib
matplotlib.use('Agg')

from data import MONAI_DS, get_weighted_sampler
from train import Trainer
from utils import first, get_model, auroc_metric, epoch_loop, plot_curve

if __name__ == '__main__':
    START = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-o', '--output', default='.',
        help='path to write out all outputs'
    )
    parser.add_argument(
        '-t', '--tag', default=None,
        help='append tag to output directory name'
    )
    parser.add_argument(
        '-p', '--parameters', 
        default='/mnt/dfs/qtim/clu/fed-learn/params/rop-base.json',
        help='path to param file',
    )
    parser.add_argument(
        '-c', '--csv', 
        default='/mnt/dfs/qtim/clu/files/rop-redo-2021-11-12.csv',
        help='path to csv file',
    )
    parser.add_argument(
        '-s', '--site', default=None,
        help='only use data from specific site (default all sites)'
    )
    parser.add_argument(
        '-d', '--debug', action='store_true',
        help='for debugging purposes'
    )
    parser.add_argument(
        '-clplus', action='store_true',
        help='use clplus instead of RSD as target label for training'
    )
    parser.add_argument('-r', '--runs', default=1, type=int,
                        help='number of runs for variance estimate'
                       )
    parser.add_argument(
        '-f', '--fed', action='store_true',
        help='Federated model averaging'
    )
    args = parser.parse_args()
    
    param = json.load(open(args.parameters))
    loader_param = {
#         'batch_size': 4,
#         'num_workers': 4,
        'pin_memory': True,
    }
    model_param = {
        'arch': 'resnet18',
        'pretrained': True,
        'in_channels': 3,
        'out_channels': 3,
        'spatial_dims': 2,
    }
    train_param = {
#         'loss_weight': [1, 1, 1],
#         'loss_weight': [0.1, 0.7, 1],
        'loss_weight': [0.2, 0.5, 1.2],
#         'learn_rate': 1e-4,
#         'epochs': 100,
#         'early_stop': 20,
    }
    fed_param = {
#         'rounds': 2,
#         'epochs_per_round': 50,
        'rounds': 5,
        'epochs_per_round': 20,
#         'rounds': 10,
#         'epochs_per_round': 10,
#         'rounds': 20,
#         'epochs_per_round': 5,
    }
    for k, v in loader_param.items(): param.setdefault(k, v)
    for k, v in model_param.items(): param.setdefault(k, v)
    for k, v in train_param.items(): param.setdefault(k, v)
    if args.fed:
        for k, v in fed_param.items(): param.setdefault(k, v)
    
    date = datetime.datetime.now().strftime('%Y_%b_%d_%H_%M_%S').lower()
    dir_name =  'debug' if args.debug else f'run_{date}'
    if args.tag is not None:
        dir_name = f'{args.tag}-' + dir_name
    run_dir = pathlib.Path(args.output) / dir_name
    run_dir.mkdir(exist_ok=True, parents=True)

    with open(run_dir / 'args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=4)

    with open(run_dir / 'params.json', 'w') as f:
        f.write(json.dumps(param, default=int, indent=4))
    
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
    df = df[df.split != 'exclude']
    
    if args.fed:
        logger.info(f'=== Using federated averaging')
        fed_df = {
            site: df[df.site == site] for site in df.site.unique()
        }
        
    test_df = df[df.split == 'test']
    
    if args.site is not None:
        logger.info(f'Training on site: {args.site}')
        df = df[df.site == args.site.upper()]
    
    train_df = df[df.split == 'train']
    valid_df = df[df.split == 'valid']
    
    if args.debug:
        n = 128 
        train_df = train_df.sample(n=min(len(train_df), n))
        valid_df = valid_df.sample(n=min(len(valid_df), n))
        test_df = test_df.sample(n=min(len(test_df), n))
    
    logger.info(f'--- number training:   {len(train_df)}')
    logger.info(f'--- number validation: {len(valid_df)}')
    logger.info(f'--- number testing:    {len(test_df)}')
    
    
    for rnd in range(2 if args.debug else args.runs):
        logger.info(f'-- Random state: {rnd}')
        rnd_dir = run_dir / f'seed_{rnd}'
        rnd_dir.mkdir(exist_ok=True, parents=True)
        np.random.seed(rnd)
        torch.manual_seed(rnd)
        target = 'clplus' if args.clplus else 'rsd'
        if args.fed:
            fed_ds = {
                site: {
                    'train': MONAI_DS(
                        site_df[site_df.split == 'train'],
                        example='segment',
                        augment=True,
                        target=target,
                    ),
                    'valid': MONAI_DS(
                        site_df[site_df.split == 'valid'],
                        example='segment',
                        augment=False,
                        target=target,
                    ),
                } for site, site_df in fed_df.items()
            }
        else:
            train_ds = MONAI_DS(train_df, example='segment', augment=True, target=target)
            valid_ds = MONAI_DS(valid_df, example='segment', augment=False, target=target)

        dl_kwargs = {
            'batch_size': param['batch_size'],
            'num_workers': param['num_workers'],
            'pin_memory': param['pin_memory'],
        }
        trn_dl_kwargs = {k: v for k, v in dl_kwargs.items() if k != 'batch_size'}

        # weight sampling by class
        class_prop = []
        for c in {0, 1, 2}:
            if args.clplus:
                class_prop.append(1 / max(len(train_df.clplus == c), 1e-5))
            else:
                class_prop.append(1 / max(len(train_df.rsd == c), 1e-5))
        class_prop = np.array(class_prop)
        if args.fed:
            if args.clplus:
                fed_weights = {
                    site: class_prop[site_df[site_df.split == 'train'].clplus.values]
                    for site, site_df in  fed_df.items()
                }
            else:
                fed_weights = {
                    site: class_prop[site_df[site_df.split == 'train'].rsd.values]
                    for site, site_df in  fed_df.items()
                }
            fed_batch_sampler = {
                site: get_weighted_sampler(
                    site_weights, 
                    batch_size=dl_kwargs['batch_size'],
                ) for site, site_weights in fed_weights.items()
            }
            site_dl = {
                site: {
                    'train': torch.utils.data.DataLoader(
                        site_ds['train'],
                        batch_sampler=fed_batch_sampler[site],
                        **trn_dl_kwargs,
                    ),
                    'valid': torch.utils.data.DataLoader(
                        site_ds['valid'], 
                        **dl_kwargs
                    ),
                } for site, site_ds in fed_ds.items()
            }
        else:
            if args.clplus:
                weights = class_prop[train_df.clplus.values]
            else:
                weights = class_prop[train_df.rsd.values]
            batch_sampler = get_weighted_sampler(
                weights, 
                batch_size=dl_kwargs['batch_size'],
            )
            train_dl = torch.utils.data.DataLoader(
                train_ds,
                batch_sampler=batch_sampler,
                **trn_dl_kwargs,
            )
            valid_dl = torch.utils.data.DataLoader(valid_ds, **dl_kwargs)

        if args.fed:
            global_model = get_model(arch=param['arch'], 
                                     pretrained=param['pretrained'],
                                     in_channels=param['in_channels'],
                                     out_channels=param['out_channels'],
                                     spatial_dims=param['spatial_dims'],
                                    ).cpu()
            for r in range(2 if args.debug else param['rounds']):
                global_state = copy.deepcopy(global_model.state_dict())
                local_state = []
                for site, dl in site_dl.items():
                    site_dir = rnd_dir / site
                    site_dir.mkdir(exist_ok=True, parents=True)
                    logger.info(f'=== Round: {r} --- site: {site}')
                    site_model = get_model(arch=param['arch'], 
                                           pretrained=param['pretrained'],
                                           in_channels=param['in_channels'],
                                           out_channels=param['out_channels'],
                                           spatial_dims=param['spatial_dims'],
                                          )
                    site_model.load_state_dict(global_state)
                    site_model = torch.nn.DataParallel(site_model.to('cuda'))
                    site_optimizer = torch.optim.AdamW(
                        site_model.parameters(), 
                        lr=param['learn_rate'],
                    )
                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        site_optimizer, 
                        mode='min',
                        factor=0.5,
                        patience=10,
                        verbose=True,
                    )
                    weight = torch.tensor(param['loss_weight'], dtype=torch.float).cuda()
                    criterion = torch.nn.CrossEntropyLoss(reduction='mean',
                                                          weight=weight,
                                                         )
                    #metric = lambda t, p: auroc_metric(t, p, labels=[0, 1, 2])
                    metric = lambda t, p: auroc_metric(t, p)
                    site_trainer = Trainer(model=site_model, 
                                           train_loader=site_dl[site]['train'], 
                                           valid_loader=site_dl[site]['valid'], 
                                           optimizer=site_optimizer, 
                                           scheduler=scheduler,
                                           criterion=criterion, 
                                           metric=metric, 
                                          )
                    epochs = 1 if args.debug else param['epochs_per_round']
                    site_epoch_res = epoch_loop(trainer=site_trainer,
                                                param=param, 
                                                run_dir=site_dir,
                                                logger=logger,
                                                epochs=epochs,
                                               )
                    site_train_loss = site_epoch_res['train_loss']
                    site_valid_loss = site_epoch_res['valid_loss']
                    site_train_metric = site_epoch_res['train_metric']
                    site_valid_metric = site_epoch_res['valid_metric']
                    plot_curve(site_train_loss, site_valid_loss, 
                               train_label=f'{site} train loss',
                               valid_label=f'{site} valid loss',
                               save_path = rnd_dir / f'round-{r}-{site}-loss_curve.png',
                              )
                    plot_curve(site_train_metric, site_valid_metric, 
                               train_label=f'{site} train metric',
                               valid_label=f'{site} valid metric',
                               ylim=(0, 1),
                               save_path = rnd_dir / f'round-{r}-{site}-metric_curve.png',
                              )

                    site_checkpoints = list(site_dir.glob('*.pth'))
                    best_checkpoint = sorted(
                        site_checkpoints,
                        key=lambda p: pathlib.Path(p).stem.split('-')[-2:],
                        reverse=True)[0]

                    with open(site_dir / f'round-{r}-site-{site}.json', 'w') as f:
                        json.dump(site_epoch_res, f, indent=4, default=int)

                    logger.info(f'    Averaging checkpoint: {best_checkpoint}')

                    #print(len(local_state))
                    local_state.append(torch.load(best_checkpoint, map_location='cpu')['state_dict'])

                    del site_model, site_trainer, site_epoch_res
                    torch.cuda.empty_cache()

                for k in global_state.keys():
                    for l in local_state:
                        global_state[k] += l[k]
                    # +1 because for global model 
                    global_state[k] = torch.true_divide(global_state[k], len(local_state) + 1)
                global_model.load_state_dict(global_state)
                torch.save({
                    'state_dict': global_model.state_dict(),
                    'round': r,
                }, str(rnd_dir / f'global_model_{r}.pth'))

            global_model = torch.nn.DataParallel(global_model.to('cuda'))
            global_trainer = Trainer(model=global_model, 
                                     optimizer=site_optimizer, 
                                     scheduler=scheduler,
                                     criterion=criterion, 
                                     metric=metric, 
                                    )
            logger.info(f'=== Testing on each site ')
            test_dl = {
                site: torch.utils.data.DataLoader(
                    MONAI_DS(
                        test_df.query(f'site == "{site}"'),
                        example='segment',
                        augment=False,
                        target='rsd',
                    ),
                    **dl_kwargs,
                )
                for site in df.site.unique()
            }
            for site, site_dl in test_dl.items():
                site_res = global_trainer.evaluate(test_loader=site_dl)
                logger.info(f'    -- {site} -- {round(site_res["metric"], 4)}')
                with open(rnd_dir / f'{site}-test-rsd-{rnd}.json', 'w') as f:
                    json.dump(site_res, f, indent=4, default=int)
            logger.info(f'=== Testing on each site ')
            test_dl = {
                site: torch.utils.data.DataLoader(
                    MONAI_DS(
                        test_df.query(f'site == "{site}"'),
                        example='segment',
                        augment=False,
                        target='clplus',
                    ),
                    **dl_kwargs,
                )
                for site in df.site.unique()
            }
            for site, site_dl in test_dl.items():
                site_res = global_trainer.evaluate(test_loader=site_dl)
                logger.info(f'    -- {site} -- {round(site_res["metric"], 4)}')
                with open(rnd_dir / f'{site}-test-clplus-{rnd}.json', 'w') as f:
                    json.dump(site_res, f, indent=4, default=int)
        else:
            model = get_model(arch=param['arch'], 
                              pretrained=param['pretrained'],
                              in_channels=param['in_channels'],
                              out_channels=param['out_channels'],
                              spatial_dims=param['spatial_dims'],
                             )
            model = model.to('cuda')
            model = torch.nn.DataParallel(model)
            metric = lambda t, p: auroc_metric(t, p)
            weight = torch.tensor(param['loss_weight'], dtype=torch.float).cuda()
            criterion = torch.nn.CrossEntropyLoss(reduction='mean', weight=weight)
            optimizer = torch.optim.AdamW(
                model.parameters(), 
                lr=param['learn_rate'],
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode='min',
                factor=0.5,
                patience=10,
                verbose=True,
            )
            trainer = Trainer(model=model, 
                              train_loader=train_dl, 
                              valid_loader=valid_dl, 
                              optimizer=optimizer, 
                              scheduler=scheduler,
                              criterion=criterion, 
                              metric=metric, 
                             )
            epoch_res = epoch_loop(trainer=trainer,
                                   param=param, 
                                   run_dir=rnd_dir,
                                   logger=logger,
                                   epochs=5 if args.debug else None,
                                  )

            train_loss = epoch_res['train_loss']
            valid_loss = epoch_res['valid_loss']
            train_metric = epoch_res['train_metric']
            valid_metric = epoch_res['valid_metric']

            plot_curve(train_loss, valid_loss, 
                       train_label='train loss',
                       valid_label='valid loss',
                       save_path = rnd_dir / 'loss_curve.png',
                      )
            plot_curve(train_metric, valid_metric, 
                       train_label='train metric',
                       valid_label='valid metric',
                       ylim=(0, 1),
                       save_path = rnd_dir / 'metric_curve.png',
                      )

            # test on each site using RSD as label
            logger.info(f'=== Testing on each site')
            test_dl = {
                site: torch.utils.data.DataLoader(
                    MONAI_DS(
                        test_df.query(f'site == "{site}"'),
                        example='segment',
                        augment=False,
                        target='rsd',
                    ),
                    **dl_kwargs,
                )
                for site in test_df.site.unique()
            }
            for site, site_dl in test_dl.items():
                site_res = trainer.evaluate(test_loader=site_dl)
                logger.info(f'    -- {site} -- {round(site_res["metric"], 4)}')
                with open(rnd_dir / f'{site}-test-rsd-{rnd}.json', 'w') as f:
                    json.dump(site_res, f, indent=4, default=int)

            # test on each site using clplus as label
            logger.info(f'=== Testing on each site')
            test_dl = {
                site: torch.utils.data.DataLoader(
                    MONAI_DS(
                        test_df.query(f'site == "{site}"'),
                        example='segment',
                        augment=False,
                        target='clplus',
                    ),
                    **dl_kwargs,
                )
                for site in test_df.site.unique()
            }
            for site, site_dl in test_dl.items():
                site_res = trainer.evaluate(test_loader=site_dl)
                logger.info(f'    -- {site} -- {round(site_res["metric"], 4)}')
                with open(rnd_dir / f'{site}-test-clplus-{rnd}.json', 'w') as f:
                    json.dump(site_res, f, indent=4, default=int)

    logger.info(f'=-=-= total runtime: {time.perf_counter() - START:.0f}s =-=-=')