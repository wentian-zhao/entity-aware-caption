import copy

import fairseq_cli.train
from fairseq.criterions.label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterion
from fairseq_cli.train import cli_main, validate
import fairseq.trainer

from main import *
from util import *
from model import *

# monkey patch
_original_validate = fairseq_cli.train.validate


def _validate(args, trainer, task, epoch_itr, subsets):
    print('================ replaced _validate ================', hasattr(trainer.task, 'end_valid_epoch'))
    _ret = _original_validate(args, trainer, task, epoch_itr, subsets)
    if hasattr(trainer.task, 'end_valid_epoch'):
        metrics = trainer.task.end_valid_epoch(epoch=epoch_itr.epoch, num_updates=trainer.get_num_updates())

        # use CIDEr as best_checkpoint_metric here
        print('CIDEr:', metrics['CIDEr'])
        _ret = [metrics['CIDEr']]

    return _ret


fairseq_cli.train.validate = _validate


"""
--tensorboard-logdir	    path to save logs for tensorboard,

--instances-per-epoch       training samples per epoch (set to -1 to disable; can be replaced with setting --validate-interval-updates and --save-interval-updates) 

--maximize-best-checkpoint-metric       select the largest metric value for saving “best” checkpoints

"""

if __name__ == '__main__':
    print('cli_main')
    cli_main()