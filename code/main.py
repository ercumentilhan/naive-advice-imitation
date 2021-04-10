import argparse
from executor import Executor
import gym
import frame_stack

import atari_preprocessing

from constants import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--run-id', type=str, default='0')  # Unique identifier
    parser.add_argument('--process-index', type=int, default=0)
    parser.add_argument('--machine-name', type=str, default='DESKTOP')

    parser.add_argument('--n-training-frames', type=int, default=3000000)
    parser.add_argument('--n-evaluation-trials', type=int, default=10)
    parser.add_argument('--evaluation-period', type=int, default=25000)
    parser.add_argument('--evaluation-visualization-period', type=int, default=10)

    # ------------------------------------------------------------------------------------------------------------------
    # DQN
    parser.add_argument('--dqn-gamma', type=float, default=0.99)
    parser.add_argument('--dqn-rm-type', type=str, default='uniform')
    parser.add_argument('--dqn-per-ims', action='store_true', default=False)
    parser.add_argument('--dqn-per-alpha', type=float, default=0.4)
    parser.add_argument('--dqn-per-beta', type=float, default=0.6)
    parser.add_argument('--dqn-rm-init', type=int, default=50000)
    parser.add_argument('--dqn-rm-max', type=int, default=500000)
    parser.add_argument('--dqn-target-update', type=int, default=7500)
    parser.add_argument('--dqn-batch-size', type=int, default=32)
    parser.add_argument('--dqn-learning-rate', type=float, default=0.0000625)
    parser.add_argument('--dqn-train-per-step', type=int, default=1)
    parser.add_argument('--dqn-train-period', type=int, default=4)
    parser.add_argument('--dqn-adam-eps', type=float, default=0.00015)
    parser.add_argument('--dqn-eps-start', type=float, default=1.0)
    parser.add_argument('--dqn-eps-final', type=float, default=0.01)
    parser.add_argument('--dqn-eps-steps', type=int, default=500000)
    parser.add_argument('--dqn-huber-loss-delta', type=float, default=1.0)
    parser.add_argument('--dqn-hidden-size', type=int, default=512)

    # ------------------------------------------------------------------------------------------------------------------
    # Action Advising

    parser.add_argument('--action-advising-method', type=str, default='none')
    parser.add_argument('--action-advising-budget', type=int, default=10000)
    parser.add_argument('--bc-uc-threshold', type=float, default=0.01)
    parser.add_argument('--bc-batch-size', type=int, default=32)
    parser.add_argument('--bc-learning-rate', type=float, default=0.0001)
    parser.add_argument('--bc-adam-eps', type=float, default=0.00015)
    parser.add_argument('--bc-dropout-rate', type=float, default=0.2)
    parser.add_argument('--bc-training-iters', type=int, default=50000)
    parser.add_argument('--bc-hidden-size', type=int, default=512)
    parser.add_argument('--bc-uc-ensembles', type=int, default=100)

    # ------------------------------------------------------------------------------------------------------------------

    parser.add_argument('--use-gpu', action='store_true', default=False)
    parser.add_argument('--save-models', action='store_true', default=False)
    parser.add_argument('--visualization-period', type=int, default=100)
    parser.add_argument('--model-save-period', type=int, default=100000)
    parser.add_argument('--env-name', type=str, default='')
    parser.add_argument('--env-training-seed', type=int, default=0)
    parser.add_argument('--env-evaluation-seed', type=int, default=1)
    parser.add_argument('--seed', type=int, default=100)

    # ------------------------------------------------------------------------------------------------------------------

    config = vars(parser.parse_args())

    env_info = ENV_INFO[config['env_name']]
    env_name = env_info[4]

    env = gym.make(env_name)
    env = atari_preprocessing.AtariPreprocessing(env)
    env = frame_stack.FrameStack(env, num_stack=4)
    env.seed(config['env_training_seed'])

    eval_env = gym.make(env_name)
    eval_env = atari_preprocessing.AtariPreprocessing(eval_env)
    eval_env = frame_stack.FrameStack(eval_env, num_stack=4)
    eval_env.seed(config['env_training_seed'])

    executor = Executor(config, env, eval_env)
    executor.run()
