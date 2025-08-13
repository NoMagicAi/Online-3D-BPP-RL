import sys
import os
import time
from collections import deque
import numpy as np
import torch
from shutil import copyfile

import envs  # Import to register the environment
from acktr import algo, utils
from acktr.envs import make_vec_envs
from acktr.arguments import get_args
from acktr.model import Policy
from acktr.storage import RolloutStorage
from tensorboardX import SummaryWriter
from gym.envs.registration import register
from clearml import Task # <-- ADDED: Import ClearML


def main(args):
    # input arguments about environment
    if args.test:
        # The test_model function is assumed to be in another file (e.g., evaluation.py)
        # and is not part of this refactoring.
        print("Test mode not implemented in this refactored script.")
        pass
    else:
        train_model(args)


def train_model(args):
    custom = input("please input the test name: ")

    # --- ADDED: ClearML Initialize Task ---
    task = Task.init(
        project_name=f'ACKTR/{args.env_name}',
        task_name=custom,
        output_uri=True
    )
    # Connect your arguments for hyperparameter tracking
    task.connect(args)
    # ---

    time_now = time.strftime("%Y.%m.%d-%H-%M", time.localtime(time.time()))
    env_name = args.env_name

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    save_path = args.save_dir
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    data_path = os.path.join(save_path, custom)
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    torch.set_num_threads(1)
    device = torch.device(args.device)

    # Create vectorized environments WITHOUT the Monitor wrapper
    envs = make_vec_envs(
        env_name,
        args.seed,
        args.num_processes,
        args.gamma,
        None,
        device,
        False,
        args=args,
    )

    actor_critic = Policy(
        envs.observation_space.shape,
        envs.action_space,
        base_kwargs={"recurrent": False, "hidden_size": args.hidden_size},
    )
    #actor_critic = torch.compile(actor_critic)
    actor_critic.to(device)

    agent = algo.ACKTR(
        actor_critic,
        args.value_loss_coef,
        args.entropy_coef,
        args.invalid_coef,
        acktr=True,
        args=args,
    )

    rollouts = RolloutStorage(
        args.num_steps,
        args.num_processes,
        envs.observation_space.shape,
        envs.action_space,
        actor_critic.recurrent_hidden_state_size,
        can_give_up=False,
        enable_rotation=args.enable_rotation,
        pallet_size=args.container_size[0],
    )

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    # --- Manual tracking deques and per-process accumulators ---
    episode_rewards_summary = deque(maxlen=10)
    episode_ratio_summary = deque(maxlen=10)
    episode_items_summary = deque(maxlen=10)

    current_episode_rewards = [0] * args.num_processes
    # ---

    start = time.time()

    writer = None
    if args.tensorboard:
        tbx_dir = os.path.join("./runs", env_name, custom)
        if not os.path.exists(tbx_dir):
            os.makedirs(tbx_dir)
        writer = SummaryWriter(logdir=tbx_dir)
    
    # --- ADDED: Initialize a tracker for the best performance ---
    best_mean_ratio = 0.0

    j = 0
    while True:
        j += 1
        for step in range(args.num_steps):
            #start_time_1 = time.perf_counter()
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = (
                    actor_critic.act(
                        rollouts.obs[step],
                        rollouts.recurrent_hidden_states[step],
                        rollouts.masks[step],
                    )
                )
            #end_time_1 = time.perf_counter()
            #print(f"actor critic act: {end_time_1 - start_time_1} seconds")
            #start_time_2 = time.perf_counter()
            obs, reward, done, infos = envs.step(action)
            #end_time_2 = time.perf_counter()
            #print(f"env step: {end_time_2 - start_time_2} seconds")



            # --- Manually update logs for each parallel environment ---
            for i, info in enumerate(infos):
                current_episode_rewards[i] += reward[i].item()
                if done[i]:
                    episode_rewards_summary.append(current_episode_rewards[i])
                    episode_ratio_summary.append(info.get("ratio", 0))
                    episode_items_summary.append(info.get("counter", 0))
                    current_episode_rewards[i] = 0  # Reset for the next episode
            # ---

            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if "bad_transition" in info.keys() else [1.0] for info in infos]
            )
            rollouts.insert(
                obs,
                recurrent_hidden_states,
                action,
                action_log_prob,
                value,
                reward,
                masks,
                bad_masks,
                torch.ones(args.num_processes, 1),
            )

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1],
                rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1],
            ).detach()

            ## --- POP-ART MODIFICATION ---
            # De-normalize the value before using it for bootstrapping.
            next_value = agent.de_normalize_value(next_value)
            ## --------------------------

        #start_time = time.perf_counter()
        # NOTE: GAE is enabled here by default. Add args.use_gae if you make it configurable.
        rollouts.compute_returns(
            next_value, True, args.gamma, 0.95, use_proper_time_limits=True
        )
        #end_time = time.perf_counter()
        #print(f"Rollout compute returns: {end_time - start_time} seconds") 

        value_loss, action_loss, dist_entropy, infeasibility_loss = agent.update(
            rollouts
        )
        rollouts.after_update()

        # --- REMOVED: The old periodic saving block is gone ---
        # if args.save_model and (j % args.save_interval == 0):
        #    ...

        if j % args.log_interval == 0 and len(episode_rewards_summary) > 1:
            total_num_steps = j * args.num_processes * args.num_steps
            end = time.time()
            popart_mean = agent.actor_critic.popart_mean.item()
            popart_std = torch.sqrt(agent.actor_critic.popart_mean_sq - agent.actor_critic.popart_mean.pow(2)).item()
            
            # --- ADDED: Calculate current performance ---
            current_mean_ratio = np.mean(episode_ratio_summary)

            print(
                f"Updates {j}, num timesteps {total_num_steps}, FPS {int(total_num_steps / (end - start))}\n"
                f"Last {len(episode_rewards_summary)} training episodes:\n"
                f"  Mean/median reward: {np.mean(episode_rewards_summary):.2f}/{np.median(episode_rewards_summary):.2f}\n"
                f"  Mean space ratio: {current_mean_ratio:.3f}\n"
                f"  Mean items packed: {np.mean(episode_items_summary):.2f}\n"
                f"Losses:\n"
                f"  entropy: {dist_entropy:.4f}, value: {value_loss:.4f}, action: {action_loss:.4f}, infeasibility: {infeasibility_loss:.4f}\n"
                f"popart/mean: {popart_mean:.3f}\n"
                f"popart/std: {popart_std:.3f}\n"
            )

            # --- ADDED: Check for new best performance and save the model ---
            if args.save_model and (current_mean_ratio > best_mean_ratio):
                print(f"🚀 New best model found! Ratio improved from {best_mean_ratio:.4f} to {current_mean_ratio:.4f}. Saving model...")
                best_mean_ratio = current_mean_ratio
                
                # Define the save path with a consistent filename
                save_file_path = os.path.join(data_path, "best_model.pt")

                torch.save(
                    [
                        actor_critic.state_dict(),
                        getattr(utils.get_vec_normalize(envs), "ob_rms", None),
                    ],
                    save_file_path,
                )
                
                # --- ClearML: Upload the new best model ---
                task.upload_artifact(
                    name='best_model', 
                    artifact_object=save_file_path,
                )


            if writer:
                writer.add_scalar(
                    "rewards/mean_episode_reward", np.mean(episode_rewards_summary), j
                )
                writer.add_scalar(
                    "metrics/mean_space_ratio", np.mean(episode_ratio_summary), j
                )
                writer.add_scalar(
                    "metrics/mean_items_packed", np.mean(episode_items_summary), j
                )
                writer.add_scalar("losses/entropy", dist_entropy, j)
                writer.add_scalar("losses/value_loss", value_loss, j)
                writer.add_scalar("losses/action_loss", action_loss, j)
                writer.add_scalar("losses/infeasibility_loss", infeasibility_loss, j)
                writer.add_scalar("popart/mean", popart_mean, j)
                writer.add_scalar("popart/std", popart_std, j)


def registration_envs():
    register(
        id='Bpp-v0',                                  # Format should be xxx-v0, xxx-v1
        entry_point='envs.bpp0:PackingGame',    # Expalined in envs/__init__.py
    )

if __name__ == "__main__":
    registration_envs()
    args = get_args()
    main(args)
