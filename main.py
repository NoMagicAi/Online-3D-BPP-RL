import sys
import os
import time
import argparse
from collections import deque
import numpy as np
import torch
import csv
from shutil import copyfile

import envs  # Import to register the environment
from acktr import algo, utils
from acktr.envs import make_vec_envs
from acktr.arguments import get_args
from acktr.model import Policy
from acktr.storage import RolloutStorage
from tensorboardX import SummaryWriter
from gym.envs.registration import register
#from clearml import Task


# --- ADDED: Helper function to clean the model's state_dict ---
def get_cleaned_state_dict(state_dict_to_clean: dict) -> dict:
    """
    Cleans a model's state_dict to handle common issues from training wrappers
    like DataParallel or K-FAC, making it suitable for loading.
    """
    if not isinstance(state_dict_to_clean, dict):
        raise TypeError(f"Expected state_dict to be a dictionary, but got {type(state_dict_to_clean)}.")

    cleaned_state_dict = {}
    print("Inspecting and cleaning state_dict keys for robust loading...")

    for key, value in state_dict_to_clean.items():
        new_key = key
        new_value = value

        # Skip 'num_batches_tracked' buffers, common in BatchNorm layers.
        # This improves robustness across different PyTorch versions.
        if "num_batches_tracked" in new_key:
            continue

        # Handle '.module' prefix from torch.nn.DataParallel
        if new_key.startswith("module."):
            new_key = new_key[7:]

        # Handle K-FAC's SplitBias wrapper on linear layers
        if new_key.endswith(".module.weight"):
            new_key = new_key.replace(".module.weight", ".weight")
        elif new_key.endswith(".add_bias._bias"):
            new_key = new_key.replace(".add_bias._bias", ".bias")
            # Correct the bias tensor shape mismatch by removing the extra dimension
            new_value = value.squeeze()

        # Handle BatchNorm running stats that might also be wrapped by K-FAC
        if ".module.running_mean" in new_key:
            new_key = new_key.replace(".module.running_mean", ".running_mean")
        if ".module.running_var" in new_key:
            new_key = new_key.replace(".module.running_var", ".running_var")

        cleaned_state_dict[new_key] = new_value
        
    print("State_dict cleaning complete.")
    return cleaned_state_dict


def main(args):
    if args.test:
        print("Test mode not implemented in this refactored script.")
        pass
    else:
        train_model(args)


def train_model(args):
    custom = "training-at-grace-robot"
    '''
    task = Task.init(
        project_name=f'ACKTR/{args.env_name}',
        task_name=custom,
        output_uri=True
    )
    task.connect(args)
    '''

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    save_path = args.save_dir
    if not os.path.exists(save_path): os.makedirs(save_path)
    data_path = os.path.join(save_path, custom)
    if not os.path.exists(data_path): os.makedirs(data_path)

    torch.set_num_threads(1)
    device = torch.device(args.device)

    envs = make_vec_envs(
        args.env_name, args.seed, args.num_processes, args.gamma,
        None, device, False, args=args,
    )

    actor_critic = Policy(
        envs.observation_space.shape,
        envs.action_space,
        base_kwargs={"recurrent": False, "hidden_size": args.hidden_size},
    )
    actor_critic.to(device)

    agent = algo.ACKTR(
        actor_critic, args.value_loss_coef, args.entropy_coef,
        args.invalid_coef, acktr=True, args=args,
    )

    rollouts = RolloutStorage(
        args.num_steps, args.num_processes, envs.observation_space.shape,
        envs.action_space, actor_critic.recurrent_hidden_state_size,
        can_give_up=False, enable_rotation=args.enable_rotation,
        pallet_size=args.container_size[0],
    )

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards_summary = deque(maxlen=10)
    episode_ratio_summary = deque(maxlen=10)
    episode_items_summary = deque(maxlen=10)
    current_episode_rewards = [0] * args.num_processes

    writer = None
    if args.tensorboard:
        tbx_dir = os.path.join("./runs", args.env_name, custom)
        if not os.path.exists(tbx_dir): os.makedirs(tbx_dir)
        writer = SummaryWriter(logdir=tbx_dir)

    # --- MODIFIED: Resume logic now uses the state_dict cleaning helper function ---
    j = 0
    best_mean_ratio = 0.0

    if args.load_model:
        load_path = os.path.join(args.load_dir, args.load_name)
        if os.path.exists(load_path):
            try:
                print(f"🚀 Loading checkpoint from: {load_path}")
                checkpoint = torch.load(load_path, map_location=device)

                # Step 1: Extract the raw state_dict for the model
                # This handles both our new dictionary format and older formats.
                if isinstance(checkpoint, dict) and 'actor_critic_state_dict' in checkpoint:
                    model_state_dict = checkpoint['actor_critic_state_dict']
                elif isinstance(checkpoint, list):
                    model_state_dict = checkpoint[0] # For older format [state_dict, ob_rms]
                else:
                    model_state_dict = checkpoint # Assumes the file is just the state_dict

                # Step 2: Clean the state_dict using the helper function
                cleaned_model_state_dict = get_cleaned_state_dict(model_state_dict)
                actor_critic.load_state_dict(cleaned_model_state_dict, strict=True)
                print("[SUCCESS] Actor-critic model state loaded successfully!")

                # Step 3: Load optimizer, ob_rms, and training progress from the full checkpoint
                if isinstance(checkpoint, dict):
                    if 'optimizer_state_dict' in checkpoint:
                        agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                        print("Optimizer state loaded successfully.")

                    ob_rms = checkpoint.get('ob_rms')
                    vec_norm = utils.get_vec_normalize(envs)
                    if vec_norm is not None and ob_rms is not None:
                        vec_norm.ob_rms = ob_rms
                        print("Observation normalization stats (ob_rms) loaded successfully.")

                    j = checkpoint.get('update_step', 0)
                    best_mean_ratio = checkpoint.get('best_mean_ratio', 0.0)
                    print(f"Training progress restored. Resuming from update #{j + 1}.")
                    print(f"Previous best mean ratio was: {best_mean_ratio:.4f}")

            except Exception as e:
                print(f"🔥 FATAL: Error loading checkpoint: {e}. Starting training from scratch.")
        else:
            print(f"⚠️ Warning: Checkpoint file '{load_path}' not found. Starting from scratch.")
    # ---

    # --- ADDED: CSV Logging Setup ---
    log_file_path = os.path.join(data_path, "training_log.csv")
    log_file_exists = os.path.isfile(log_file_path)

    log_file = open(log_file_path, 'a', newline='')
    csv_writer = csv.writer(log_file)

    # Write header only if the file is new
    if not log_file_exists:
        header = [
            'update_step', 'total_timesteps', 'mean_reward', 'median_reward', 
            'mean_space_ratio', 'mean_items_packed', 'entropy_loss', 
            'value_loss', 'action_loss', 'infeasibility_loss'
        ]
        csv_writer.writerow(header)
    # --- END OF ADDED CODE ---

    start = time.time()
    while True:
        j += 1
        for step in range(args.num_steps):
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step]
                )
            obs, reward, done, infos = envs.step(action)
            for i, info in enumerate(infos):
                current_episode_rewards[i] += reward[i].item()
                if done[i]:
                    episode_rewards_summary.append(current_episode_rewards[i])
                    episode_ratio_summary.append(info.get("ratio", 0))
                    episode_items_summary.append(info.get("counter", 0))
                    current_episode_rewards[i] = 0

            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if "bad_transition" in info.keys() else [1.0] for info in infos]
            )
            rollouts.insert(
                obs, recurrent_hidden_states, action, action_log_prob, value,
                reward, masks, bad_masks, torch.ones(args.num_processes, 1)
            )

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]
            ).detach()
            next_value = agent.de_normalize_value(next_value)

        rollouts.compute_returns(
            next_value, True, args.gamma, 0.95, use_proper_time_limits=True
        )

        value_loss, action_loss, dist_entropy, infeasibility_loss = agent.update(rollouts)
        rollouts.after_update()

        if j % args.log_interval == 0 and len(episode_rewards_summary) > 1:
            total_num_steps = j * args.num_processes * args.num_steps
            end = time.time()
            popart_mean = agent.actor_critic.popart_mean.item()
            popart_std = torch.sqrt(agent.actor_critic.popart_mean_sq - agent.actor_critic.popart_mean.pow(2)).item()
            current_mean_ratio = np.mean(episode_ratio_summary)

            print(
                f"\nUpdates {j}, Timesteps {total_num_steps}, FPS {int(total_num_steps / (end - start))}\n"
                f"Last {len(episode_rewards_summary)} episodes: "
                f"reward mean/median {np.mean(episode_rewards_summary):.2f}/{np.median(episode_rewards_summary):.2f}, "
                f"space ratio {current_mean_ratio:.3f}, "
                f"items packed {np.mean(episode_items_summary):.2f}\n"
                f"Losses: entropy {dist_entropy:.4f}, value {value_loss:.4f}, action {action_loss:.4f}, infeasibility {infeasibility_loss:.4f}\n"
                f"PopArt stats: mean {popart_mean:.3f}, std {popart_std:.3f}\n"
            )

            if args.save_model and (current_mean_ratio > best_mean_ratio):
                print(f"🚀 New best model! Ratio improved from {best_mean_ratio:.4f} to {current_mean_ratio:.4f}. Saving checkpoint...")
                best_mean_ratio = current_mean_ratio
                save_file_path = os.path.join(data_path, "best_model.pt")

                # The saving logic remains the same, creating the comprehensive checkpoint
                torch.save(
                    {
                        'update_step': j,
                        'best_mean_ratio': best_mean_ratio,
                        'actor_critic_state_dict': actor_critic.state_dict(),
                        'optimizer_state_dict': agent.optimizer.state_dict(),
                        'ob_rms': getattr(utils.get_vec_normalize(envs), "ob_rms", None),
                    },
                    save_file_path,
                )

                #task.upload_artifact(name='best_model', artifact_object=save_file_path)

            if writer:
                writer.add_scalar("rewards/mean_episode_reward", np.mean(episode_rewards_summary), j)
                writer.add_scalar("metrics/mean_space_ratio", np.mean(episode_ratio_summary), j)
                writer.add_scalar("metrics/mean_items_packed", np.mean(episode_items_summary), j)
                writer.add_scalar("losses/entropy", dist_entropy, j)
                writer.add_scalar("losses/value_loss", value_loss, j)
                writer.add_scalar("losses/action_loss", action_loss, j)
                writer.add_scalar("losses/infeasibility_loss", infeasibility_loss, j)
                writer.add_scalar("popart/mean", popart_mean, j)
                writer.add_scalar("popart/std", popart_std, j)
            
            # --- ADDED: Write metrics to CSV file ---
            log_data = [
                j, total_num_steps, np.mean(episode_rewards_summary), 
                np.median(episode_rewards_summary), current_mean_ratio, 
                np.mean(episode_items_summary), dist_entropy, value_loss,
                action_loss, infeasibility_loss
            ]
            csv_writer.writerow(log_data)
            log_file.flush() # Ensure data is written to disk immediately
            # --- END OF ADDED CODE ---


def registration_envs():
    register(
        id='Bpp-v0',
        entry_point='envs.bpp0:PackingGame',
    )

if __name__ == "__main__":
    time.sleep(10)
    registration_envs()
    args = get_args()
    main(args)