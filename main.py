# main.py

import os
import sys
import traceback
import random
import config
import utils
from model import Model
from dataloader import Dataloader
from checkpoints import Checkpoints
import torch

args, config_file = config.parse_args()
# Data Loading    
if args.train == 'face_cls':
    from test_cls import Tester
    from train_cls import Trainer

if args.train == 'rfw':
    from test_cls import Tester
    from train_RFW import Trainer

if args.dataset_train == 'ClassSamplesDataLoader':
    from train_classload import Trainer


def main():
    # parse the arguments
    random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)
    if args.save_results:
        utils.saveargs(args, config_file)

    # initialize the checkpoint class
    checkpoints = Checkpoints(args)

    # Create Model
    models = Model(args)
    model, model_dict, evaluation = models.setup(checkpoints)

    print('Model:\n\t{model}\nTotal params:\n\t{npar:.2f}M'.format(
          model=args.model_type,
          npar=sum(p.numel() for p in model['feat'].parameters()) / 1000000.0))

    # The trainer handles the training loop
    trainer = Trainer(args, model, model_dict['loss'], evaluation)
    # The trainer handles the evaluation on validation set
    tester = Tester(args, model, model_dict['loss'], evaluation)

    test_freq = 1

    dataloader = Dataloader(args)

    if args.extract_feat:
        loaders  = dataloader.create(flag='Test')
        tester.extract_features(loaders)
        # tester.extract_features_h5py(loaders, len(dataloader.dataset_test))
    elif args.just_test:
        loaders  = dataloader.create(flag='Test')
        acc_test = tester.test(args.epoch_number, loaders)
        print(acc_test)
    else:

        loaders  = dataloader.create()
        if args.dataset_train == 'ClassSamplesDataLoader':
            loaders['train'] = dataloader.dataset_train

        # start training !!!
        acc_best = 0
        loss_best = 999
        stored_models = {}

        for epoch in range(args.nepochs-args.epoch_number):
            epoch += args.epoch_number
            print('\nEpoch %d/%d\n' % (epoch + 1, args.nepochs))

            # train for a single epoch
            # loss_train = 3.0
            loss_train = trainer.train(epoch, loaders, checkpoints, acc_best)
            if float(epoch) % test_freq == 0:
                acc_test = tester.test(epoch, loaders)

            if loss_best > loss_train:
                model_best = True
                loss_best = loss_train
                acc_best = acc_test
                if args.save_results:
                    stored_models['model'] = model
                    stored_models['loss'] = trainer.criterion
                    checkpoints.save(acc_best, stored_models, epoch, 'final', model_best)


if __name__ == "__main__":
    utils.setup_graceful_exit()
    try:
        main()
    except (KeyboardInterrupt, SystemExit):
        # do not print stack trace when ctrl-c is pressed
        pass
    except Exception:
        traceback.print_exc(file=sys.stdout)
    finally:
        traceback.print_exc(file=sys.stdout)
        utils.cleanup()
