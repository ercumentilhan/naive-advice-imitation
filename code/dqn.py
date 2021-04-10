import os
import random
import tensorflow as tf
from universal_replay_buffer import ReplayBuffer, PrioritizedReplayBuffer


class DQN(object):
    def __init__(self, id, config, session, stats):

        # Extract relevant configuration:
        self.config = {}
        self.config['env_name'] = config['env_name']
        self.config['env_n_actions'] = config['env_n_actions']
        self.config['env_obs_dims'] = config['env_obs_dims']
        self.config['env_obs_form'] = config['env_obs_form']

        self.config['n_training_frames'] = config['n_training_frames']
        self.config['rm_extra_content'] = config['rm_extra_content']

        dqn_config_params = [
            'dqn_gamma',
            'dqn_rm_type',
            'dqn_rm_init',
            'dqn_rm_max',
            'dqn_per_ims',
            'dqn_per_alpha',
            'dqn_per_beta',
            'dqn_target_update',
            'dqn_batch_size',
            'dqn_learning_rate',
            'dqn_train_period',
            'dqn_adam_eps',
            'dqn_huber_loss_delta',
            'dqn_hidden_size'
        ]
        for param in dqn_config_params:
            self.config[param] = config[param]

        self.id = id
        self.session = session
        self.stats = stats

        # Scoped names
        self.name_online = self.id + '/' + 'ONLINE'
        self.name_target = self.id + '/' + 'TARGET'

        self.tf_vars = {}

        self.tf_vars['obs'] = self.build_input_obs(self.name_online)
        self.tf_vars['obs_tar'] = self.build_input_obs(self.name_target)

        self.replay_memory = None
        self.minibatch_keys = None

        self.post_init_steps = 0
        self.training_steps = 0
        self.training_steps_since_target_update = 0
        self.n_episode = 0

        print('# of training frames:', self.config['n_training_frames'])

        self.total_optimiser_steps = max(0, (self.config['n_training_frames'] - self.config['dqn_rm_init']) \
                                     / self.config['dqn_train_period'])

        print('# of optimiser steps:', self.total_optimiser_steps)

        self.replay_memory = None
        self.per_beta = None
        self.per_beta_inc = None

    # ------------------------------------------------------------------------------------------------------------------

    def create_replay_memory(self):
        if self.config['dqn_rm_type'] == 'uniform':
            self.replay_memory = ReplayBuffer(self.config['dqn_rm_max'],
                                              extra_content=self.config['rm_extra_content'])
        elif self.config['dqn_rm_type'] == 'per':
            self.replay_memory = PrioritizedReplayBuffer(self.config['dqn_rm_max'],
                                                         extra_content=self.config['rm_extra_content'],
                                                         alpha=self.config['dqn_per_alpha'])

            self.per_beta = self.config['dqn_per_beta']
            self.per_beta_inc = (1.0 - self.per_beta) / float(self.total_optimiser_steps)

    # ------------------------------------------------------------------------------------------------------------------

    def build_input_obs(self, name):
        return tf.compat.v1.placeholder(tf.compat.v1.float32, [None, self.config['env_obs_dims'][0],
                                                               self.config['env_obs_dims'][1],
                                                               self.config['env_obs_dims'][2]], name=name + '_OBS')

    # ------------------------------------------------------------------------------------------------------------------

    def conv_layers(self, scope, inputs):
        with tf.compat.v1.variable_scope(scope, reuse=tf.compat.v1.AUTO_REUSE):
            layer_1 = tf.compat.v1.layers.conv2d(inputs=inputs,
                                                 filters=32,
                                                 kernel_size=(8, 8),
                                                 strides=(4, 4),
                                                 padding='VALID',
                                                 kernel_initializer=tf.keras.initializers.VarianceScaling(),
                                                 activation=tf.nn.relu,
                                                 name='CONV_LAYER_1')

            layer_2 = tf.compat.v1.layers.conv2d(inputs=layer_1,
                                                 filters=64,
                                                 kernel_size=(4, 4),
                                                 strides=(2, 2),
                                                 padding='VALID',
                                                 kernel_initializer=tf.keras.initializers.VarianceScaling(),
                                                 activation=tf.nn.relu,
                                                 name='CONV_LAYER_2')

            layer_3 = tf.compat.v1.layers.conv2d(inputs=layer_2,
                                                 filters=64,
                                                 kernel_size=(3, 3),
                                                 strides=(1, 1),
                                                 padding='VALID',
                                                 kernel_initializer=tf.keras.initializers.VarianceScaling(),
                                                 activation=tf.nn.relu,
                                                 name='CONV_LAYER_3')

            output = tf.compat.v1.layers.flatten(layer_3)
            return output

    # ------------------------------------------------------------------------------------------------------------------

    def dense_layers(self, scope, inputs, is_dueling, hidden_size, output_size, head_id):
        with tf.compat.v1.variable_scope(scope, reuse=tf.compat.v1.AUTO_REUSE):
            layer_1 = tf.compat.v1.layers.dense(inputs, hidden_size, use_bias=True,
                                                kernel_initializer=tf.keras.initializers.VarianceScaling(),
                                                activation=tf.nn.relu, name='DENSE_LAYER_' + str(head_id) + '_1')

            if is_dueling:
                layer_2_adv = tf.compat.v1.layers.dense(layer_1, output_size, use_bias=True,
                                                        kernel_initializer=tf.keras.initializers.VarianceScaling(),
                                                        activation=None, name='DENSE_LAYER_' + str(head_id) + '_2_ADV')

                layer_2_val = tf.compat.v1.layers.dense(layer_1, 1, use_bias=True,
                                                        kernel_initializer=tf.keras.initializers.VarianceScaling(),
                                                        activation=None, name='DENSE_LAYER_' + str(head_id) + '_2_VAL')

                advantage = (layer_2_adv - tf.compat.v1.reduce_mean(layer_2_adv, axis=-1, keepdims=True))
                value = tf.compat.v1.tile(layer_2_val, [1, output_size])
                return advantage + value, layer_1

            else:
                layer_2 = tf.compat.v1.layers.dense(layer_1, output_size, use_bias=True,
                                                    kernel_initializer=tf.keras.initializers.VarianceScaling(),
                                                    activation=None, name='DENSE_LAYER_' + str(head_id) + '_2')
                return layer_2, layer_1

    # ------------------------------------------------------------------------------------------------------------------

    def build_copy_ops(self):
        trainable_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=self.name_online)
        trainable_vars_by_name = {var.name[len(self.name_online):]: var for var in trainable_vars}

        trainable_vars_t = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
                                                       scope=self.name_target)
        trainable_vars_by_name_t = {var.name[len(self.name_target):]: var for var in trainable_vars_t}

        copy_ops = [target_var.assign(trainable_vars_by_name[var_name])
                    for var_name, target_var in trainable_vars_by_name_t.items()]

        return tf.compat.v1.group(*copy_ops)

    # ------------------------------------------------------------------------------------------------------------------

    def fix_batch_form(self, var, is_batch):
        return var if is_batch else [var]

    # ------------------------------------------------------------------------------------------------------------------

    def random_action(self):
        return random.randrange(self.config['env_n_actions'])

    # ------------------------------------------------------------------------------------------------------------------

    def save_model(self, saver, models_dir, session_name, checkpoint):
        model_path = os.path.join(os.path.join(models_dir, session_name), 'model-{}.ckpt').format(checkpoint)
        print('[{}] Saving model... {}'.format(checkpoint, model_path))
        saver.save(self.session, model_path)

    # ------------------------------------------------------------------------------------------------------------------

    def restore(self, models_dir, session_name, checkpoint):
        print('Restoring...')
        print('Scope: {}'.format(self.id))
        print('# of variables: {}'.format(
            len(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope=self.id))))
        loader = tf.compat.v1.train.Saver(
            var_list=tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope=self.id))
        loader.restore(self.session,
                       os.path.join(os.path.join(models_dir, session_name), 'model-' + str(int(checkpoint)) + '.ckpt'))
