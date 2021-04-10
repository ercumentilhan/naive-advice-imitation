import os
import random
import pathlib
import numpy as np
import tensorflow as tf

tf.compat.v1.disable_eager_execution()
tf.compat.v1.disable_v2_behavior()

from dqn_egreedy import EpsilonGreedyDQN
from run_statistics import Statistics

os.environ['TF_CPP_MIN_LONG_LEVEL'] = '2'

from constants import *
import video_recorder

from behavioural_cloning import BehaviouralCloning

class Executor:
    def __init__(self, config, env, eval_env):
        self.config = config
        self.env = env
        self.eval_env = eval_env

        self.stats = None
        self.student_agent = None

        self.episode_duration = 0
        self.episode_reward = 0.0
        self.steps_reward = 0.0

        self.run_id = None

        self.session = None
        self.summary_writer = None
        self.saver = None

        self.teacher_agent = None

        self.action_advising_budget = None

        self.video_recorder = None
        self.save_videos_path = None

        self.bc_net = None
        self.bc_reuse_allowed = False
        self.bc_net_is_trained = False

    # ==================================================================================================================

    def run(self):
        os.environ['PYTHONHASHSEED'] = str(self.config['seed'])
        random.seed(self.config['seed'])
        np.random.seed(self.config['seed'])
        tf.compat.v1.set_random_seed(self.config['seed'])
        tf.random.set_seed(self.config['seed'])

        self.run_id = self.config['run_id']
        self.seed_id = str(self.config['seed'])

        print('Run ID: {}'.format(self.run_id))

        # --------------------------------------------------------------------------------------------------------------

        scripts_dir = os.path.dirname(os.path.abspath(__file__))
        workspace_dir = os.path.join(str(pathlib.Path(scripts_dir).parent))

        print('{} (Code directory)'.format(scripts_dir))
        print('{} (Workspace directory)'.format(workspace_dir))

        summaries_dir = os.path.join(workspace_dir, 'summaries')
        os.makedirs(summaries_dir, exist_ok=True)

        checkpoints_dir = os.path.join(workspace_dir, 'checkpoints')
        os.makedirs(checkpoints_dir, exist_ok=True)

        videos_dir = os.path.join(workspace_dir, 'videos')
        os.makedirs(videos_dir, exist_ok=True)

        save_summary_path = os.path.join(summaries_dir, self.run_id, self.seed_id)
        save_model_path = os.path.join(checkpoints_dir, self.run_id, self.seed_id)
        self.save_videos_path = os.path.join(videos_dir, self.run_id, self.seed_id)

        if self.config['save_models']:
            os.makedirs(save_model_path, exist_ok=True)

        os.makedirs(self.save_videos_path, exist_ok=True)

        # --------------------------------------------------------------------------------------------------------------

        if self.config['use_gpu']:
            print('Using GPU.')
            session_config = tf.compat.v1.ConfigProto(
                #intra_op_parallelism_threads=1,
                #inter_op_parallelism_threads=1
                )
            session_config.gpu_options.allow_growth = True
        else:
            print('Using CPU.')
            session_config = tf.compat.v1.ConfigProto(
                intra_op_parallelism_threads=1,
                inter_op_parallelism_threads=1,
                allow_soft_placement=True,
                device_count={'CPU': 1, 'GPU': 0})

        self.session = tf.compat.v1.InteractiveSession(graph=tf.compat.v1.get_default_graph(), config=session_config)

        self.summary_writer = tf.compat.v1.summary.FileWriter(save_summary_path, self.session.graph)

        self.stats = Statistics(self.summary_writer, self.session)
        #self.teacher_stats = Statistics(self.summary_writer, self.session)

        # --------------------------------------------------------------------------------------------------------------

        self.env_info = {}

        env_info = ENV_INFO[self.config['env_name']]
        self.env_info['max_timesteps'] = env_info[8]

        self.config['env_type'] = env_info[1]
        self.config['env_obs_form'] = env_info[2]
        self.config['env_states_are_countable'] = env_info[3]

        self.config['env_obs_dims'] = self.env.observation_space.shape
        self.config['env_n_actions'] = self.env.action_space.n
        self.config['env_obs_dims'] = (84, 84, 4)  # Lazy frames are enabled

        self.config['rm_extra_content'] = ['source', 'expert_action']

        # --------------------------------------------------------------------------------------------------------------
        # Setup student agent
        self.config['actor_id'] = self.run_id
        self.student_agent = EpsilonGreedyDQN(self.config['actor_id'], self.config, self.session,
                                              self.config['dqn_eps_start'],
                                              self.config['dqn_eps_final'],
                                              self.config['dqn_eps_steps'], self.stats)

        self.config['actor_id'] = self.student_agent.id

        print('Student ID: {}'.format(self.student_agent.id))

        # --------------------------------------------------------------------------------------------------------------
        # Initialise the teacher agent
        if self.config['action_advising_method'] != 'none':
            teacher_info = TEACHER[self.config['env_name']]
            self.config['teacher_id'] = teacher_info[0]
            self.teacher_agent = EpsilonGreedyDQN(self.config['teacher_id'], self.config, self.session, 0.0, 0.0, 1,
                                              self.stats)
        # --------------------------------------------------------------------------------------------------------------

        if self.config['action_advising_method'] == 'ai':
            self.bc_net = BehaviouralCloning('BHC', self.config, self.session, None)

        # --------------------------------------------------------------------------------------------------------------

        total_parameters = 0
        for variable in tf.compat.v1.trainable_variables():
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        print('Number of parameters: {}'.format(total_parameters))

        self.saver = tf.compat.v1.train.Saver(max_to_keep=None)
        self.session.run(tf.compat.v1.global_variables_initializer())

        # --------------------------------------------------------------------------------------------------------------
        # Restore the teacher agent
        if self.config['action_advising_method'] != 'none':
            print('Restoring the teacher...')
            teacher_info = TEACHER[self.config['env_name']]
            self.teacher_agent.restore(checkpoints_dir, teacher_info[0] + '/' + teacher_info[1], teacher_info[2])
            print('done.')

        self.action_advising_budget = self.config['action_advising_budget']

        # --------------------------------------------------------------------------------------------------------------

        if not self.config['save_models']:
            tf.compat.v1.get_default_graph().finalize()

        self.evaluate_student_agent()
        obs, render = self.reset_env()

        while True:
            if self.config['action_advising_method'] == 'ai':
                if (not self.bc_net_is_trained) and \
                        self.bc_net.replay_memory.__len__() == self.config['action_advising_budget']:

                    print('Initial BC training.')
                    print(self.stats.n_env_steps, self.bc_net.replay_memory.__len__())

                    for _ in range(self.config['bc_training_iters']):
                        self.bc_net.feedback_learn()

                    self.bc_net_is_trained = True

            # ----------------------------------------------------------------------------------------------------------
            action = None

            self_action, action_is_explorative = self.student_agent.get_action(obs)
            #self_action, action_is_explorative = self.actor_agent.get_random_action()

            if action_is_explorative:
                self.stats.exploration_steps_taken += 1
                self.stats.exploration_steps_taken_cum += 1

            # ----------------------------------------------------------------------------------------------------------

            if self.config['action_advising_method'] != 'none':
                if self.config['action_advising_method'] == 'ai':

                    bc_collection_occurred = False

                    # Collection
                    if self.action_advising_budget > 0 and action_is_explorative:
                        teacher_action = self.teacher_agent.get_greedy_action(obs)
                        self.bc_net.feedback_observe(obs, teacher_action)
                        bc_collection_occurred = True
                        action = teacher_action
                        self.action_advising_budget -= 1
                        self.stats.advices_taken += 1
                        self.stats.advices_taken_cum += 1

                    # Reuse
                    if self.bc_net_is_trained and self.bc_reuse_allowed and (not bc_collection_occurred):
                        if action_is_explorative:
                            bc_uncertainty = self.bc_net.get_uncertainty(obs, self.config['bc_uc_ensembles'])
                            bc_action = np.argmax(self.bc_net.get_action_probs(obs))
                            teacher_action = self.teacher_agent.get_greedy_action(obs)  # Oracle, for measuring accuracy

                            if bc_uncertainty < self.config['bc_uc_threshold']:
                                action = bc_action
                                self.stats.advices_reused += 1
                                self.stats.advices_reused_cum += 1

                                if bc_action == teacher_action:
                                    self.stats.advices_reused_correct += 1
                                    self.stats.advices_reused_correct_cum += 1

                elif self.config['action_advising_method'] == 'early':
                    if self.action_advising_budget > 0 and action_is_explorative:
                        action = self.teacher_agent.get_greedy_action(obs)
                        self.action_advising_budget -= 1
                        self.stats.advices_taken += 1
                        self.stats.advices_taken_cum += 1

            # ----------------------------------------------------------------------------------------------------------

            if action is not None:
                source = 1
                self.stats.advices_used += 1
                self.stats.advices_used_cum += 1
            else:
                source = 0
                action = self_action

            # ----------------------------------------------------------------------------------------------------------
            # Execute action
            obs_next, reward, done, info, _ = self.env.step(action)

            transition = {
                'obs': obs,
                'action': action,
                'reward': reward,
                'obs_next': obs_next,
                'done': done,
                'source': source,
                'expert_action': None  # self.teacher_agent.get_greedy_action(obs)
            }

            if render:
                self.video_recorder.capture_frame()

            self.episode_reward += reward
            self.episode_duration += 1

            self.steps_reward += reward
            self.stats.n_env_steps += 1

            # ----------------------------------------------------------------------------------------------------------
            # Feedback
            self.student_agent.feedback_observe(transition)

            # ----------------------------------------------------------------------------------------------------------

            td_error_batch, loss = self.student_agent.feedback_learn()

            self.stats.loss += loss
            obs = obs_next

            done = done or self.episode_duration >= self.env_info['max_timesteps']

            if done:
                self.stats.n_episodes += 1
                self.stats.episode_reward_auc += np.trapz([self.stats.episode_reward_last, self.episode_reward])
                self.stats.episode_reward_last = self.episode_reward

                self.stats.update_summary_episode(self.episode_reward, self.stats.episode_reward_auc,
                                                  self.episode_duration, 0.0, 0.0)

                print('{}'.format(self.stats.n_episodes), end=' | ')
                print('{:.1f}'.format(self.episode_reward), end=' | ')
                print('{}'.format(self.episode_duration), end=' | ')
                print('{}'.format(self.stats.n_env_steps), end=' | ')
                print(self.bc_reuse_allowed)

                if render:
                    self.video_recorder.close()
                    self.video_recorder.enabled = False

                obs, render = self.reset_env()

            # Per N steps summary update
            if self.stats.n_env_steps % self.stats.n_steps_per_update == 0:
                self.stats.steps_reward_auc += np.trapz([self.stats.steps_reward_last, self.steps_reward])
                self.stats.steps_reward_last = self.steps_reward
                self.stats.epsilon = self.student_agent.eps

                self.stats.update_summary_steps(self.steps_reward, self.stats.steps_reward_auc, 0.0, 0.0)

                self.stats.exploration_steps_taken = 0

                self.stats.advices_taken = 0
                self.stats.advices_used = 0
                self.stats.advices_reused = 0
                self.stats.advices_reused_correct = 0

                self.steps_reward = 0.0

            if self.stats.n_env_steps % self.config['evaluation_period'] == 0:
                self.evaluate_student_agent()

            if self.config['save_models'] and \
                    (self.stats.n_env_steps % self.config['model_save_period'] == 0 or
                     self.stats.n_env_steps >= self.config['n_training_frames']):
                self.save_model(save_model_path)

            if self.stats.n_env_steps >= self.config['n_training_frames']:
                break

        print('Env steps: {}'.format(self.stats.n_env_steps))

        self.session.close()

    # ==================================================================================================================

    def reset_env(self):

        self.episode_duration = 0
        self.episode_reward = 0.0

        render = self.stats.n_episodes % self.config['visualization_period'] == 0
        if render:
            self.video_recorder = video_recorder.\
                VideoRecorder(self.env,
                              base_path=os.path.join(self.save_videos_path, '{}_{}'.format(str(self.stats.n_episodes),
                                                                                      str(self.stats.n_env_steps))))

        obs = self.env.reset()
        if render:
            self.video_recorder.capture_frame()

        self.bc_reuse_allowed = (self.bc_net_is_trained and random.random() < 0.5)

        return obs, render

    # ==================================================================================================================

    def evaluate_student_agent(self):
        eval_render = self.stats.n_evaluations % self.config['evaluation_visualization_period'] == 0

        eval_total_reward = 0.0
        eval_duration = 0

        self.eval_env.seed(self.config['env_evaluation_seed'])

        if eval_render:
            video_capture_eval = video_recorder.\
                VideoRecorder(self.eval_env,
                              base_path=os.path.join(self.save_videos_path, 'E_{}_{}'.format(str(self.stats.n_episodes),
                                                                                      str(self.stats.n_env_steps))))

        for i_eval_trial in range(self.config['n_evaluation_trials']):
            eval_obs = self.eval_env.reset()

            eval_episode_reward = 0.0
            eval_episode_duration = 0

            while True:
                if eval_render:
                    video_capture_eval.capture_frame()

                eval_action = self.student_agent.get_greedy_action(eval_obs)
                eval_obs_next, eval_reward, eval_done, eval_info, eval_actual_reward \
                    = self.eval_env.step(eval_action)

                eval_episode_reward += eval_actual_reward
                eval_duration += 1
                eval_episode_duration += 1
                eval_obs = eval_obs_next

                eval_done = eval_done or eval_episode_duration >= self.env_info['max_timesteps']

                if eval_done:
                    if eval_render:
                        video_capture_eval.capture_frame()
                        video_capture_eval.close()
                        video_capture_eval.enabled = False

                        eval_render = False
                    eval_total_reward += eval_episode_reward
                    break

        eval_mean_reward = eval_total_reward / float(self.config['n_evaluation_trials'])

        self.stats.evaluation_reward_auc += np.trapz([self.stats.evaluation_reward_last, eval_mean_reward])
        self.stats.evaluation_reward_last = eval_mean_reward

        self.stats.n_evaluations += 1

        self.stats.update_summary_evaluation(eval_mean_reward, eval_duration, self.stats.evaluation_reward_auc)

        print('Evaluation @ {} | {}'.format(self.stats.n_env_steps, eval_mean_reward))

        return eval_mean_reward

    # ==================================================================================================================

    def save_model(self, save_model_path):
        model_path = os.path.join(os.path.join(save_model_path), 'model-{}.ckpt').format(
            self.stats.n_env_steps)
        print('[{}] Saving model... {}'.format(self.stats.n_env_steps, model_path))
        self.saver.save(self.session, model_path)
