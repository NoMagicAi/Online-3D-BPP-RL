# acktr/arguments.py
import argparse
import time
import torch

def get_args():
    parser = argparse.ArgumentParser(description='RL')
    
    # Corrected definitions with proper nargs and type
    parser.add_argument(
        '--container_size', default=[10, 10, 10], type=int, nargs=3, help='container size (width, length, height)')
    parser.add_argument(
        '--item-size-range', default=[2,2,2,5,5,5], type=int, nargs=6, help='item size range (min_w, min_l, min_h, max_w, max_l, max_h)')
    
    parser.add_argument('--mode', default='train')
    parser.add_argument('--env_name', default='Bpp-v0')
    parser.add_argument('--load-model', action='store_true', default=False)
    parser.add_argument('--use-cuda', action='store_true', default=False)
    parser.add_argument('--tensorboard', action='store_true', default=False)
    parser.add_argument('--preview', default=1, type=int)
    parser.add_argument('--item-seq', default='cut1')
    parser.add_argument('--algorithm', default='acktr', type=str)
    parser.add_argument('--gamma', default=1.0, type=float)
    parser.add_argument('--use-gae', action='store_true', default=False)
    parser.add_argument('--gae-lambda', type=float, default=0.95)
    parser.add_argument('--entropy_coef', default=0.01, type=float)
    parser.add_argument('--value_loss_coef', default=0.5, type=float)
    parser.add_argument('--invalid_coef', default=0.01, type=float)
    parser.add_argument('--hidden_size', default=256, type=int)
    parser.add_argument('--num_processes', default=16, type=int)
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--save_interval', default=100, type=int)
    parser.add_argument('--log_interval', default=10, type=int)
    parser.add_argument('--save_model', action='store_true', default=False)
    parser.add_argument('--num_env_steps', type=int, default=10e6)
    parser.add_argument('--num_steps', default=128, type=int)
    parser.add_argument('--enable_rotation', action='store_true', default=False)
    parser.add_argument('--data_name', default='cut_2.pt')
    parser.add_argument('--load_name', default='default_cut_2.pt')
    parser.add_argument('--load_dir', default='./pretrained_models/')
    parser.add_argument('--save_dir', default='./saved_models/')
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--use-popart', action='store_true', default=False,
                        help='use PopArt to normalize rewards')
    parser.add_argument('--popart-beta', type=float, default=1e-4,
                        help='beta for running mean and std in PopArt (decay factor)')
    parser.add_argument(
        '--visual-log-interval',
        type=int,
        default=500,
        help='Log a visual of the packing state every N updates (default: 500)')
    parser.add_argument('--project_name', default='rl_planner', type=str, help='project name for ClearML')
    parser.add_argument('--experiment_name', default='acktr_experiment_' + time.strftime("%Y%m%d-%H%M%S"), type=str, help='experiment name for ClearML')
    args = parser.parse_args()

    args.device = "cuda:" + str(args.device) if args.use_cuda and torch.cuda.is_available() else "cpu"
    args.bin_size = (args.container_size[0], args.container_size[1], args.container_size[2])
    args.pallet_size = args.container_size[0]
    args.channel = 6 
    args.data_type = args.item_seq
    args.test = (args.mode == 'test')

    box_range = args.item_size_range
    box_size_set = []
    for i in range(box_range[0], box_range[3] + 1):
        for j in range(box_range[1], box_range[4] + 1):
            for k in range(box_range[2], box_range[5] + 1):
                box_size_set.append((i, j, k))
    args.box_size_set = box_size_set
    
    return args