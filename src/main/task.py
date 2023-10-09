import os
import sys
import json
import time
import zipfile
from abc import abstractmethod

from fairseq.criterions.adaptive_loss import AdaptiveLoss
from fairseq.data import Dictionary, data_utils
from fairseq.modules import AdaptiveSoftmax
from fairseq.tasks import FairseqTask, register_task
from fairseq.logging import metrics

from config import *
from util.evaluate import COCOResultGenerator, evaluate, save_metrics, evaluate_news
from . import data
from util.tokenizer import RobertaTokenizer
from util.optimization import *


def zip_source_code(folder, target_zip_file):
    f_zip = zipfile.ZipFile(target_zip_file, 'w', zipfile.ZIP_STORED)
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith('.py'):
                f_zip.write(os.path.join(root, file))
    f_zip.close()


@register_task('caption')
class CaptioningTask(FairseqTask):
    @staticmethod
    def add_args(parser):
        parser.add_argument('--dataset', default='goodnews', type=str)
        parser.add_argument('--max-source-positions', default=512, type=int, metavar='N',
                            help='max number of objects in the source sequence')
        # parser.add_argument('--max-target-positions', default=50, type=int, metavar='N',
        #                     help='max number of tokens in the target sequence')
        parser.add_argument('--max-target-positions', default=512, type=int, metavar='N',
                            help='max number of tokens in the target sequence')
        parser.add_argument('--instances-per-epoch', default=65536, type=int, metavar='N',
                            help='training samples per epoch (>0 to enable, set to -1 to disable; can be replaced with setting --validate-interval-updates and --save-interval-updates) ')

        # parser.add_argument('--beam', default=5, help='validation | beam size')
        # parser.add_argument('--max-len-a', default=0, help='validation | max-len-a')
        # parser.add_argument('--max-len-b', default=200, help='validation | max-len-b')

        parser.add_argument('--use-image', type=int, default=1)
        parser.add_argument('--use-text', type=int, default=1)

        parser.add_argument('--use-text-graph', type=int, default=0)
        parser.add_argument('--text-graph-max-size', type=int, default=150)
        parser.add_argument('--use-image-graph', type=int, default=1)
        parser.add_argument('--image-graph-max-size', type=int, default=20)

        parser.add_argument('--gpt2-encoder-json', type=str, default="../pretrained_model/gpt2_bpe/encoder.json", help='path to encoder.json')
        parser.add_argument('--gpt2-vocab-bpe', type=str, default="../pretrained_model/gpt2_bpe/vocab.bpe", help='path to vocab.bpe')

    @classmethod
    def setup_task(cls, args, **kwargs):
        return CaptioningTask(args)

    def __init__(self, args):
        super().__init__(args)

        if hasattr(args, 'save_dir'):
            self.save_dir = args.save_dir
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
            with open(os.path.join(self.save_dir, 'args'), 'w') as f:
                f.write('cwd: ' + os.getcwd() + '\n')
                f.write('cmd: ' + ' '.join(sys.argv) + '\n')
                f.write('args: ' + json.dumps(self.args.__dict__, indent=4) + '\n')
            zip_source_code(folder='.', target_zip_file=os.path.join(self.save_dir, 'src.zip'))

        dictionary = Dictionary.load(os.path.join(data_dir, 'roberta.large.dictionary'))
        tokenizer = RobertaTokenizer(
            encoder_json=os.path.join(data_dir, 'encoder.json'),
            vocab_bpe=os.path.join(data_dir, 'vocab.bpe'),
            dictionary_path=os.path.join(data_dir, 'roberta.large.dictionary')
        )

        # self.args = args          # in super().__init__()
        self.dictionary = dictionary
        self.tokenizer = tokenizer
        self._base_dataset = data.NewsCaptionDataset(args, self.args.dataset, dictionary=self.dictionary, tokenizer=tokenizer)

    def _init_generator(self, models):
        self.generator = self.build_generator(models=models, args=self.args)

    def load_dataset(self, split, **kwargs):
        _split = {'train': 'train', 'valid': 'val', 'test': 'test', 'test-small': 'test-small'}[split]
        _dataset = data.FairseqDatasetSubset(self._base_dataset, self._base_dataset.get_split_index(_split))
        print(f'{split} length:', len(_dataset))
        self.datasets[split] = _dataset

    def max_positions(self):
        return self.args.max_source_positions, self.args.max_target_positions

    @property
    def source_dictionary(self):
        return self.dictionary

    @property
    def target_dictionary(self):
        return self.dictionary

    @metrics.aggregate("train")
    def train_step(self, sample, model, criterion, optimizer, update_num, ignore_grad=False):
        # loss, sample_size, logging_output = \
        #     super().train_step(sample, model, criterion, optimizer, update_num, ignore_grad=False)

        last_call_time = getattr(self, 'last_call_time', None)
        t0 = time.time()
        if last_call_time is not None:
            step_time = t0 - last_call_time        # time interval between two steps
        else:
            step_time = 0.
        setattr(self, 'last_call_time', t0)

        model.train()
        model.set_num_updates(update_num)
        with torch.autograd.profiler.record_function("forward"):
            loss, sample_size, logging_output = criterion(model, sample)

        # torch.cuda.synchronize()
        t1 = time.time(); time_forward = t1 - t0

        if ignore_grad:
            loss *= 0
        with torch.autograd.profiler.record_function("backward"):
            optimizer.backward(loss)

        # torch.cuda.synchronize()
        t2 = time.time(); time_backward = t2 - t1
        time_train_step = t2 - t0

        if update_num % 5 == 0:
            metrics.log_scalar('time_forward', time_forward)
            metrics.log_scalar('time_backward', time_backward)
            metrics.log_scalar('train_step_call_interval', step_time)
            metrics.log_scalar('train_step_inner', time_train_step)

        return loss, sample_size, logging_output

    def begin_valid_epoch(self, epoch, model):
        super().begin_valid_epoch(epoch, model)
        print('begin_valid_epoch {}'.format(epoch))
        self.result_generator = COCOResultGenerator()

    def valid_step(self, sample, model, criterion):
        if not hasattr(self, 'generator'):
            self._init_generator(models=[model])

        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)

        model.eval()
        hypos = self.inference_step(generator=self.generator, models=[model], sample=sample, )

        batch_size = len(hypos)
        for i in range(batch_size):
            # index = sample['id'][i]
            image_id = sample['image_id'][i]
            # gen = self.dictionary.string(hypos[i][0]['tokens'])
            tokens = hypos[i][0]['tokens'].detach().cpu()
            gen = self.tokenizer.decode(tokens)

            gt_sent = self._base_dataset.get_gt_sent_by_image_id(image_id)
            self.result_generator.add_annotation(image_id, gt_sent)

            if not self.result_generator.has_output(image_id):
                self.result_generator.add_output(image_id, gen)

        return loss, sample_size, logging_output

    def end_valid_epoch(self, epoch, num_updates):
        print('end_valid_epoch {}, {} updates'.format(epoch, num_updates))

        result_dir = os.path.join(self.args.save_dir, 'validation_results_{}'.format(self.args.valid_subset))
        print('beam size:', self.generator.beam_size)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        annotation_file = os.path.join(result_dir, 'annotation.json')
        result_file = os.path.join(result_dir, 'result_{}_{}.json'.format(epoch, num_updates))
        metric_file = os.path.join(result_dir, 'metric.csv')
        self.result_generator.dump_annotation_and_output(annotation_file, result_file)

        # metrics, _img_score = evaluate(annotation_file, result_file, return_imgscores=True, use_scorers=['Bleu', 'METEOR', 'ROUGE_L', 'CIDEr'])
        metrics = evaluate_news(annotation_file, result_file)
        save_metrics(metric_file, metrics, epoch=epoch, global_step=num_updates)
        return metrics