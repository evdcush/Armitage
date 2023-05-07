"""
+-------+
|Config |
+-------+-----------+
|                   |
| model             |
| train_dataloader  |
|   dataset         |
|   sampler         |
| train_cfg         |
| optim_wrapper     |
|   optimizer       |
| param_scheduler   |
| val_dataloader    |
|   dataset         |
|   sampler         |
| val_cfg           |
| val_evaluator     |
| default_hooks     |
| launcher          |
| env_cfg           |
|   backend         |
|   mp_cfg          |
| log_level         |
| load_from         |
| resume            |
|                   |
+-------------------+

- Build Runner from config
- Run

"""

from argparse import ArgumentParser

from mmengine.registry import DefaultScope

from armitage.infer import CustomInferencer


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--out-file', default='result', help='Path to output file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    args = parser.parse_args()
    return args


def main(args):
    DefaultScope.get_instance(
        name='armitage', scope_name='armitage')
    inferencer = CustomInferencer(
        args.config, args.checkpoint, save_path=args.out_file)
    inferencer(args.img, vis_thresh=0.8)


if __name__ == '__main__':
    args = parse_args()
    main(args)
