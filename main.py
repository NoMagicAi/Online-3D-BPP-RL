import sys
import os
import time
import argparse
from collections import deque
import numpy as np
import torch
import csv
from shutil import copyfile
import colorsys

import envs  # Import to register the environment
from acktr import algo, utils
from acktr.envs import make_vec_envs
from acktr.arguments import get_args
from acktr.model import Policy
from acktr.storage import RolloutStorage
from tensorboardX import SummaryWriter
from gym.envs.registration import register
# --- ADDED: Import ClearML and Matplotlib ---
from clearml import Task
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

def log_final_heatmap(logger, heightmap, container_size, iteration):
    """Logs a 2D heatmap of the final heightmap to ClearML."""
    if heightmap is None:
        return
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(heightmap.T, cmap='viridis', origin='lower', vmin=0, vmax=container_size[2])
    ax.set_title(f'Final Heightmap at Update {iteration}')
    ax.set_xlabel('Container Width')
    ax.set_ylabel('Container Length')
    fig.colorbar(im, ax=ax, label='Packed Height')
    logger.report_matplotlib_figure(
        title="Final Packing State",
        series="2D Heightmap",
        iteration=iteration,
        figure=fig
    )
    plt.close(fig)

# In your main training script

def log_3d_render(logger, boxes, container_size, iteration, failed_box_info=None, final_heightmap=None):
    """
    Logs a 3D voxel render, including a failed box with a visual hint
    for the reason of episode termination.
    """
    if not boxes and not failed_box_info:
        return

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    base_cmap = plt.colormaps.get_cmap('tab20')
    num_colors = len(boxes) + 1 if boxes else 1
    cmap = base_cmap.resampled(num_colors)
    
    facecolors_array = np.full(tuple(container_size) + (4,), [0, 0, 0, 0.0], dtype=float)

    # 1. Render successfully packed boxes (no changes here)
    if boxes:
        for i, box in enumerate(boxes):
            x, y, z = int(box.x), int(box.y), int(box.z)
            w, l, h = int(box.lx), int(box.ly), int(box.lz)
            box_color = cmap(i + 1)
            
            if (0 <= x < container_size[0] and 0 <= y < container_size[1] and 0 <= z < container_size[2]):
                    facecolors_array[x:x+w, y:y+l, z:z+h] = box_color

    # --- MODIFICATION START: Add visual hints for failure reason ---
    title_text = f'3D Render (Update {iteration})' # Default title

    if failed_box_info and 'dims' in failed_box_info and final_heightmap is not None:
        dims = failed_box_info['dims']
        pos = failed_box_info.get('pos')
        
        w, l, h = int(dims[0]), int(dims[1]), int(dims[2])
        
        # Default to a black box for a standard invalid move
        failed_box_color = [0, 0, 0, 0.8] 

        if pos is not None:
            # Case 1: Episode ended due to an INVALID ACTION at a specific (x, y)
            title_text = (
                f'Invalid Action at Update {iteration}\n'
                f'Failed Item Dims: {w}x{l}x{h} at ({int(pos[0])}, {int(pos[1])})'
            )
            x, y = int(pos[0]), int(pos[1])
            footprint = final_heightmap[x:min(x + w, container_size[0]), y:min(y + l, container_size[1])]
            z_attempt = int(np.max(footprint)) if footprint.size > 0 else 0
        else:
            # Case 2: Episode ended because NO VALID MOVES were left
            title_text = (
                f'No Valid Moves Left at Update {iteration}\n'
                f'Next Item Dims: {w}x{l}x{h}'
            )
            # Use a distinct color (RED) to indicate this specific failure type
            failed_box_color = [1, 0, 0, 0.8] # Red, semi-transparent
            x, y = 0, 0 # Place at a default corner for visualization
            z_attempt = int(np.max(final_heightmap)) if final_heightmap.size > 0 else 0

        # The clamping fix remains the same, it's essential for both cases
        z = min(z_attempt, container_size[2] - h)
        z = max(z, 0)

        # Clip dimensions and draw the failed box with the chosen color
        x_end = min(x + w, container_size[0])
        y_end = min(y + l, container_size[1])
        z_end = min(z + h, container_size[2])
        if x < x_end and y < y_end and z < z_end:
            facecolors_array[x:x_end, y:y_end, z:z_end] = failed_box_color
    # --- MODIFICATION END ---

    filled = np.any(facecolors_array[..., :3] != [0, 0, 0], axis=-1)
    ax.voxels(filled, facecolors=facecolors_array, edgecolor='k', linewidth=0.5)

    ax.set_xlabel('Width')
    ax.set_ylabel('Length')
    ax.set_zlabel('Height')
    ax.set_title(title_text) # Use the new dynamic title
    ax.set_xlim(0, container_size[0])
    ax.set_ylim(0, container_size[1])
    ax.set_zlim(0, container_size[2])
    
    logger.report_matplotlib_figure(
        title="Final Packing State",
        series="3D Voxel Render",
        iteration=iteration,
        figure=fig
    )
    plt.close(fig)

def train_model(args):
    # --- MODIFIED: Activated ClearML Task ---
    task = Task.init(
        project_name=f'{args.project_name}',
        task_name=args.experiment_name,
        output_uri=True
    )
    task.connect(args)
    # --- ADDED: Get ClearML logger ---
    logger = task.get_logger()


    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    save_path = args.save_dir
    if not os.path.exists(save_path): os.makedirs(save_path)
    data_path = os.path.join(save_path, args.experiment_name)
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
        tbx_dir = os.path.join("./runs", args.env_name, args.experiment_name)
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

    # --- ADDED: History tracking for ClearML plots ---
    plot_history = {
        'updates': [], 'mean_reward': [], 'median_reward': [],
        'mean_space_ratio': [], 'mean_items_packed': [],
        'entropy_loss': [], 'value_loss': [], 'action_loss': [],
        'infeasibility_loss': []
    }

    start = time.time()
    while True:
        j += 1
        logged_visual_this_step = False
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

                    # MODIFICATION START: Extract failure info and pass to the logger
                    if j % args.visual_log_interval == 0 and not logged_visual_this_step:
                        final_heightmap = info.get('final_heightmap')
                        final_boxes = info.get('final_boxes')
                        
                        failed_box_info = None
                        if 'failed_box_dims' in info:
                            failed_box_info = {
                                'dims': info['failed_box_dims'],
                                'pos': info.get('failed_box_pos') # .get() safely handles if 'pos' is missing
                            }
                        
                        if final_heightmap is not None:
                            log_final_heatmap(logger, final_heightmap, args.container_size, j)
                        
                        # We can render even if no boxes were packed, as long as there was a failed attempt
                        if final_boxes is not None or failed_box_info:
                            log_3d_render(
                                logger=logger, 
                                boxes=(final_boxes or []), # Use empty list if final_boxes is None
                                container_size=args.container_size, 
                                iteration=j, 
                                failed_box_info=failed_box_info, 
                                final_heightmap=final_heightmap
                            )
                        
                        if final_heightmap is not None or final_boxes is not None:
                            logged_visual_this_step = True
                    # MODIFICATION END

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

                task.upload_artifact(name='best_model', artifact_object=save_file_path)

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

            # --- START: ADDED ClearML Plot Logging ---
            # 1. Update history with current metrics
            plot_history['updates'].append(j)
            plot_history['mean_reward'].append(np.mean(episode_rewards_summary))
            plot_history['median_reward'].append(np.median(episode_rewards_summary))
            plot_history['mean_space_ratio'].append(current_mean_ratio)
            plot_history['mean_items_packed'].append(np.mean(episode_items_summary))
            plot_history['entropy_loss'].append(dist_entropy)
            plot_history['value_loss'].append(value_loss)
            plot_history['action_loss'].append(action_loss)
            plot_history['infeasibility_loss'].append(infeasibility_loss)

            # 2. Generate and log 'Rewards' plot
            plt.figure(figsize=(10, 6))
            plt.plot(plot_history['updates'], plot_history['mean_reward'], label='Mean Reward', marker='o', linestyle='-')
            plt.plot(plot_history['updates'], plot_history['median_reward'], label='Median Reward', marker='x', linestyle='--')
            plt.title('Episode Rewards over Time')
            plt.xlabel('Update Step')
            plt.ylabel('Reward')
            plt.legend()
            plt.grid(True)
            logger.report_matplotlib_figure(
                title="Episode Rewards",
                series="Rewards Plot",
                iteration=j,
                figure=plt
            )
            plt.close()

            # 3. Generate and log 'Performance Metrics' plot
            fig, ax1 = plt.subplots(figsize=(10, 6))
            ax1.set_xlabel('Update Step')
            ax1.set_ylabel('Space Ratio', color='tab:blue')
            ax1.plot(plot_history['updates'], plot_history['mean_space_ratio'], label='Mean Space Ratio', color='tab:blue', marker='o')
            ax1.tick_params(axis='y', labelcolor='tab:blue')
            ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
            ax2.set_ylabel('Items Packed', color='tab:orange')
            ax2.plot(plot_history['updates'], plot_history['mean_items_packed'], label='Mean Items Packed', color='tab:orange', marker='x', linestyle='--')
            ax2.tick_params(axis='y', labelcolor='tab:orange')
            fig.tight_layout()
            plt.title('Performance Metrics over Time')
            plt.grid(True)
            logger.report_matplotlib_figure(
                title="Performance Metrics",
                series="Ratio and Items Plot",
                iteration=j,
                figure=plt
            )
            plt.close()

            # 4. Generate and log 'Losses' plot
            plt.figure(figsize=(10, 6))
            plt.plot(plot_history['updates'], plot_history['entropy_loss'], label='Entropy', marker='.')
            plt.plot(plot_history['updates'], plot_history['value_loss'], label='Value', marker='.')
            plt.plot(plot_history['updates'], plot_history['action_loss'], label='Action', marker='.')
            plt.plot(plot_history['updates'], plot_history['infeasibility_loss'], label='Infeasibility', marker='.')
            plt.title('Training Losses over Time')
            plt.xlabel('Update Step')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            logger.report_matplotlib_figure(
                title="Training Losses",
                series="Losses Plot",
                iteration=j,
                figure=plt
            )
            plt.close()
            # --- END: ADDED ClearML Plot Logging ---


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