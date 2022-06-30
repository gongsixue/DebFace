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

def get_demog_cohorts(demog_type):
    demog_dict = {'gender':[6,4], 'age':[2,4], 'race':[2,6]}
    fact1 = demog_dict[demog_type][0]
    fact2 = demog_dict[demog_type][1]
    demog1 = list(range(fact1))
    demog2 = list(range(fact2))
    gdemog = []
    for x in demog1:
        for y in demog2:
            gdemog.append(str(x)+str(y))
    
    gdemog.sort()
    return gdemog

def main():
    demog_type = 'race'
    demog_target = {'gender':1, 'age':2, 'race':3}
    demog_refer = {'gender':[2,3], 'age':[1,3], 'race':[1,2]}

    # initialize the checkpoint class
    checkpoints = Checkpoints(args)

    # Create Model
    models = Model(args)
    model, model_dict, evaluation = models.setup(checkpoints)

    print('Model:\n\t{model}\nTotal params:\n\t{npar:.2f}M'.format(
          model=args.model_type,
          npar=sum(p.numel() for p in model['feat'].parameters()) / 1000000.0))

    # The trainer handles the evaluation on validation set
    tester = Tester(args, model, model_dict['loss'], evaluation)

    test_freq = 1

    dataloader = Dataloader(args)
    dataset_options_test = args.dataset_options_test

    resfilename = '/research/prip-gongsixu/codes/biasface/results/evaluation/demogbias/race.txt'
    gdemog = get_demog_cohorts(demog_type)
    with open(resfilename, 'w') as f:
        for demog_group in gdemog:
            dataset_options_test['target_ind'] = demog_target[demog_type]
            dataset_options_test['refer_ind'] = demog_refer[demog_type]
            dataset_options_test['demog_group'] = demog_group
            args.dataset_options_test = dataset_options_test
            loaders  = dataloader.create(flag='Test')
            acc_test = tester.test_demog(demog_type, loaders)
            f.write(demog_group+'\t'+str(acc_test)+'\n')
            print(acc_test)


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