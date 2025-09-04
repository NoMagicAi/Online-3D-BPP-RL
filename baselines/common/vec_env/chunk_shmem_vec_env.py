"""
An interface for asynchronous vectorized environments, with a chunked version
that runs multiple environments sequentially per process.
"""
import multiprocessing as mp
import numpy as np
from .vec_env import VecEnv, CloudpickleWrapper, clear_mpi_env_vars
import ctypes
from baselines import logger
import math # Import math for ceiling division

from .util import dict_to_obs, obs_space_info, obs_to_dict

_NP_TO_CT = {np.float32: ctypes.c_float,
             np.int32: ctypes.c_int32,
             np.int8: ctypes.c_int8,
             np.uint8: ctypes.c_char,
             bool: ctypes.c_bool}


# THIS IS THE NEW WORKER FUNCTION
def _subproc_worker_chunked(pipe, parent_pipe, env_fn_wrappers, obs_bufs, obs_shapes, obs_dtypes, keys):
    """
    Control a chunk of environments sequentially within a single process.
    """
    def _write_obs(env_idx, maybe_dict_obs):
        flatdict = obs_to_dict(maybe_dict_obs)
        for k in keys:
            dst = obs_bufs[env_idx][k].get_obj()
            dst_np = np.frombuffer(dst, dtype=obs_dtypes[k]).reshape(obs_shapes[k])
            np.copyto(dst_np, flatdict[k])

    envs = [fn_wrapper.x() for fn_wrapper in env_fn_wrappers]
    parent_pipe.close()
    try:
        while True:
            cmd, data = pipe.recv()
            if cmd == 'reset':
                for i, env in enumerate(envs):
                    _write_obs(i, env.reset())
                pipe.send(None)
            elif cmd == 'step':
                # Data is a list/array of actions for the envs in this process
                rewards, dones, infos = [], [], []
                for i, env in enumerate(envs):
                    obs, reward, done, info = env.step(data[i])
                    if done:
                        obs = env.reset()
                    _write_obs(i, obs)
                    rewards.append(reward)
                    dones.append(done)
                    infos.append(info)
                pipe.send((rewards, dones, infos))
            elif cmd == 'render':
                pipe.send([env.render(mode='rgb_array') for env in envs])
            elif cmd == 'close':
                pipe.send(None)
                break
            else:
                raise RuntimeError(f'Got unrecognized cmd {cmd}')
    except KeyboardInterrupt:
        print('ChunkedShmemVecEnv worker: got KeyboardInterrupt')
    finally:
        for env in envs:
            env.close()

# THIS IS THE NEW VECENV CLASS
class ChunkedShmemVecEnv(VecEnv):
    """
    An optimized version of SubprocVecEnv that uses shared variables to communicate observations.
    This version runs a specified number of environments sequentially in each process.

    Args:
        env_fns (list[callable]): A list of functions that create the environments.
        spaces (tuple, optional): A tuple of (observation_space, action_space). If None,
            it's inferred from a dummy environment.
        context (str, optional): The multiprocessing context to use ('spawn', 'fork').
        envs_per_proc (int, optional): The number of environments to run sequentially
            in each worker process. Set to 8 or 16 as requested.
    """
    def __init__(self, env_fns, spaces=None, context='spawn', envs_per_proc=8):
        ctx = mp.get_context(context)
        if spaces:
            observation_space, action_space = spaces
        else:
            logger.log('Creating dummy env object to get spaces')
            with logger.scoped_configure(format_strs=[]):
                dummy = env_fns[0]()
                observation_space, action_space = dummy.observation_space, dummy.action_space
                dummy.close()
                del dummy
        
        num_envs = len(env_fns)
        VecEnv.__init__(self, num_envs, observation_space, action_space)

        self.envs_per_proc = envs_per_proc
        self.num_procs = math.ceil(num_envs / self.envs_per_proc)
        logger.log(f'Creating {self.num_procs} processes for {num_envs} environments ({self.envs_per_proc} envs per process).')

        self.obs_keys, self.obs_shapes, self.obs_dtypes = obs_space_info(observation_space)
        self.obs_bufs = [
            {k: ctx.Array(_NP_TO_CT[self.obs_dtypes[k].type], int(np.prod(self.obs_shapes[k]))) for k in self.obs_keys}
            for _ in range(num_envs)
        ]
        
        self.parent_pipes, self.procs = [], []
        
        # Split env_fns and obs_bufs into chunks for each process
        env_fn_chunks = [env_fns[i:i + self.envs_per_proc] for i in range(0, num_envs, self.envs_per_proc)]
        obs_buf_chunks = [self.obs_bufs[i:i + self.envs_per_proc] for i in range(0, num_envs, self.envs_per_proc)]

        with clear_mpi_env_vars():
            for env_fn_chunk, obs_buf_chunk in zip(env_fn_chunks, obs_buf_chunks):
                wrapped_fns = [CloudpickleWrapper(fn) for fn in env_fn_chunk]
                parent_pipe, child_pipe = ctx.Pipe()
                proc = ctx.Process(target=_subproc_worker_chunked,
                                   args=(child_pipe, parent_pipe, wrapped_fns, obs_buf_chunk,
                                         self.obs_shapes, self.obs_dtypes, self.obs_keys))
                proc.daemon = True
                self.procs.append(proc)
                self.parent_pipes.append(parent_pipe)
                proc.start()
                child_pipe.close()

        self.waiting_step = False
        self.viewer = None

    def reset(self):
        if self.waiting_step:
            logger.warn('Called reset() while waiting for the step to complete')
            self.step_wait()
        for pipe in self.parent_pipes:
            pipe.send(('reset', None))
        # Wait for all workers to finish resetting
        [pipe.recv() for pipe in self.parent_pipes]
        return self._decode_obses()

    def step_async(self, actions):
        # Split actions into chunks for each process
        action_chunks = np.array_split(actions, self.num_procs)
        for pipe, act_chunk in zip(self.parent_pipes, action_chunks):
            pipe.send(('step', act_chunk))
        self.waiting_step = True

    def step_wait(self):
        outs = [pipe.recv() for pipe in self.parent_pipes]
        self.waiting_step = False
        
        # Unpack and flatten the results from each process
        rews_chunks, dones_chunks, infos_chunks = zip(*outs)
        rews = np.concatenate(rews_chunks)
        dones = np.concatenate(dones_chunks)
        infos = [info for chunk in infos_chunks for info in chunk]

        return self._decode_obses(), rews, dones, infos

    def close_extras(self):
        if self.waiting_step:
            self.step_wait()
        for pipe in self.parent_pipes:
            pipe.send(('close', None))
        for pipe in self.parent_pipes:
            pipe.recv()
            pipe.close()
        for proc in self.procs:
            proc.join()

    def get_images(self, mode='human'):
        for pipe in self.parent_pipes:
            pipe.send(('render', None))
        # Images will be a list of lists, so we flatten it
        img_chunks = [pipe.recv() for pipe in self.parent_pipes]
        return [img for chunk in img_chunks for img in chunk]

    def _decode_obses(self):
        result = {}
        for k in self.obs_keys:
            bufs = [b[k] for b in self.obs_bufs]
            o = [np.frombuffer(b.get_obj(), dtype=self.obs_dtypes[k]).reshape(self.obs_shapes[k]) for b in bufs]
            result[k] = np.array(o)
        return dict_to_obs(result)