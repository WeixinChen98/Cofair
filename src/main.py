import os
import argparse
from utils.quick_start import quick_start
os.environ['NUMEXPR_MAX_THREADS'] = '48'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--recommendation_model', '-m', type=str, default='VBPR', help='name of recommendation models')
    parser.add_argument('--dataset', '-d', type=str, default='ml1m', help='name of datasets')
    parser.add_argument('--gpu_id', '-g', type=int, default=0, help='gpu_id')
    parser.add_argument('--epochs', '-e', type=int, default=1000, help='epoch')
    parser.add_argument('--fairness_model', '-fm', type=str, default=None, help='name of fairness model')
    parser.add_argument('--disc_epoch', '-de', type=int, default=500, help='discriminator epoch')
    parser.add_argument('--model_load_path', type=str, default=None, help='the path of the trained model')
    parser.add_argument('--save_recommended_topk', action='store_true', help='flag to save recommended top-k results')

    # parser.add_argument('--d_steps', '-ds', type=int, default=10, help='discriminator update steps')
    # parser.add_argument('--disc_reg_weight', type=float, default=0.1)


    config_dict = {
        'gpu_id': 0,
    }

    args = parser.parse_args()

    config_dict.update(vars(args))

    quick_start(recommendation_model=args.recommendation_model, dataset=args.dataset, config_dict=config_dict, save_model=True)



