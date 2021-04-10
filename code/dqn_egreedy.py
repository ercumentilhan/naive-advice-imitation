from dqn import *
import numpy as np

class EpsilonGreedyDQN(DQN):
    def __init__(self, id, config, session, eps_start, eps_final, eps_steps, stats):
        super(EpsilonGreedyDQN, self).__init__(id, config, session, stats)

        self.create_replay_memory()

        self.minibatch_keys = ('obs', 'action', 'reward', 'obs_next', 'done')

        self.eps_start = eps_start
        self.eps_final = eps_final
        self.eps_steps = eps_steps

        self.eps = self.eps_start
        self.eps_step = (self.eps_start - self.eps_final) / self.eps_steps

        self.tf_vars['pre_fc_features'], self.tf_vars['mid_fc_features'], self.tf_vars['q_values'] = \
            self.build_network(self.name_online, self.tf_vars['obs'], True, self.config['dqn_hidden_size'],
                               self.config['env_n_actions'])

        self.tf_vars['pre_fc_features_tar'], self.tf_vars['mid_fc_features_tar'], self.tf_vars['q_values_tar'] = \
            self.build_network(self.name_target, self.tf_vars['obs_tar'], True, self.config['dqn_hidden_size'],
                               self.config['env_n_actions'])

        self.update_target_weights = super().build_copy_ops()
        self.build_training_ops()

    # ------------------------------------------------------------------------------------------------------------------

    def build_network(self, name, input, is_dueling, dense_hidden_size, output_size):
        pre_fc_features = super().conv_layers(name, input)
        q_values, mid_fc_features = super().dense_layers(name,
                                                         inputs=pre_fc_features,
                                                         is_dueling=is_dueling,
                                                         hidden_size=dense_hidden_size,
                                                         output_size=output_size,
                                                         head_id=1)

        return pre_fc_features, mid_fc_features, q_values

    # ------------------------------------------------------------------------------------------------------------------

    def build_training_ops(self):
        self.tf_vars['action'] = tf.compat.v1.placeholder(tf.compat.v1.int32, [None], name='ACTIONS_' + str(self.id))
        self.tf_vars['td_target'] = tf.compat.v1.placeholder(tf.compat.v1.float32, [None],
                                                             name='LABELS_' + str(self.id))

        self.tf_vars['source'] = tf.compat.v1.placeholder(tf.compat.v1.float32, [None], name='SOURCES_' + str(self.id))
        self.tf_vars['ims_weights'] = tf.compat.v1.placeholder(tf.compat.v1.float32, [None],
                                                               name='IMS_WEIGHTS_' + str(self.id))

        action_one_hot = tf.compat.v1.one_hot(self.tf_vars['action'], self.config['env_n_actions'], 1.0, 0.0)
        q_values_reduced = tf.compat.v1.reduce_sum(tf.compat.v1.math.multiply(self.tf_vars['q_values'], action_one_hot),
                                                   reduction_indices=1)

        self.tf_vars['td_error'] = tf.compat.v1.abs(self.tf_vars['td_target'] - q_values_reduced)

        # Q-Learning Loss (1-step)
        if self.config['dqn_rm_type'] == 'per' and self.config['dqn_per_ims']:
            loss_ql = tf.compat.v1.losses.huber_loss(labels=self.tf_vars['td_target'],
                                           predictions=q_values_reduced,
                                           delta=self.config['dqn_huber_loss_delta'],
                                           weights=self.tf_vars['ims_weights'])
        else:
            loss_ql = tf.compat.v1.losses.huber_loss(labels=self.tf_vars['td_target'],
                                           predictions=q_values_reduced,
                                           delta=self.config['dqn_huber_loss_delta'])

        self.tf_vars['loss_ql'] = loss_ql
        self.tf_vars['loss_ql_weighted'] = tf.compat.v1.math.multiply(self.tf_vars['loss_ql'], 1.0)
        self.tf_vars['loss'] = self.tf_vars['loss_ql_weighted']

        optimizer = tf.compat.v1.train.AdamOptimizer(self.config['dqn_learning_rate'], epsilon=self.config['dqn_adam_eps'])
        self.tf_vars['grads_update'] = optimizer.minimize(self.tf_vars['loss'])

    # ------------------------------------------------------------------------------------------------------------------
    # obs, action, reward, obs_next, done, source, expert_action
    def feedback_observe(self, transition):
        if transition['done']:
            self.n_episode += 1

        old_transition = self.replay_memory.add(transition)
        return old_transition

    # ------------------------------------------------------------------------------------------------------------------

    def feedback_learn(self):
        loss = 0.0
        td_error_batch = [0.0]

        perform_learning = self.replay_memory.__len__() >= self.config['dqn_rm_init']

        if perform_learning:
            if self.eps > self.eps_final:
                self.eps -= self.eps_step

            self.post_init_steps += 1
            if self.post_init_steps % self.config['dqn_train_period'] == 0:

                td_error_batch, loss = self.train_model()

                if self.training_steps_since_target_update >= self.config['dqn_target_update']:
                    self.training_steps_since_target_update = 0
                    self.session.run(self.update_target_weights)

                if self.config['dqn_rm_type'] == 'per':
                    self.per_beta += self.per_beta_inc

        return td_error_batch, loss

    # ------------------------------------------------------------------------------------------------------------------

    def train_model(self):
        self.training_steps += 1
        self.training_steps_since_target_update += 1

        if self.config['dqn_rm_type'] == 'uniform':
            minibatch_ = self.replay_memory.sample(self.config['dqn_batch_size'],
                                                   in_numpy_form=True)
        elif self.config['dqn_rm_type'] == 'per':
            minibatch_ = self.replay_memory.sample(self.config['dqn_batch_size'],
                                                   beta=self.per_beta,
                                                   in_numpy_form=True)

        minibatch = {}
        for i, key in enumerate(self.minibatch_keys):
            if key == 'obs' or key == 'obs_next':  # LazyFrames Correction
                minibatch[key] = np.moveaxis(np.asarray(minibatch_[i], dtype=np.float32) / 255.0, 1, -1)
            else:
                minibatch[key] = minibatch_[i]
            if i == 4:
                break

        if self.config['dqn_rm_type'] == 'uniform':
            minibatch['source'] = minibatch_[-1]['source']
        elif self.config['dqn_rm_type'] == 'per':
            minibatch['source'] = minibatch_[-3]['source']
            minibatch['weights'] = minibatch_[-2]
            minibatch['idxes'] = minibatch_[-1]

        td_error, loss = self.get_grads_update(minibatch)
        prios = np.abs(td_error[:self.config['dqn_batch_size']]) + float(1e-6)

        if self.config['dqn_rm_type'] == 'per':
            self.replay_memory.update_priorities(minibatch['idxes'], prios)

        return td_error, loss

    # ------------------------------------------------------------------------------------------------------------------

    def get_q_values(self, obs):
        obs = np.moveaxis(np.asarray(obs, dtype=np.float32) / 255.0, 0, -1)
        feed_dict = {self.tf_vars['obs']: [obs.astype(dtype=np.float32)]}
        return self.session.run(self.tf_vars['q_values'], feed_dict=feed_dict)

    def get_latent_features(self, obs):
        obs = np.moveaxis(np.asarray(obs, dtype=np.float32) / 255.0, 0, -1)
        feed_dict = {self.tf_vars['obs']: [obs.astype(dtype=np.float32)]}
        return self.session.run(self.tf_vars['latent_features'], feed_dict=feed_dict)

    def get_td_error(self, minibatch):
        feed_dict, is_batch = self.arrange_feed_dict(minibatch)
        td_error_batch = self.session.run(self.tf_vars['td_error'], feed_dict=feed_dict)
        return td_error_batch if is_batch else td_error_batch[0]

    def get_loss(self, minibatch):
        feed_dict, is_batch = self.arrange_feed_dict(minibatch)
        loss_batch = self.session.run(self.tf_vars['loss'], feed_dict=feed_dict)
        return loss_batch if is_batch else loss_batch[0]

    def get_grads_update(self, minibatch):
        feed_dict, is_batch = self.arrange_feed_dict(minibatch)
        td_error_batch, loss_batch, _, q_vals = \
            self.session.run([self.tf_vars['td_error'], self.tf_vars['loss'], self.tf_vars['grads_update'],
                              self.tf_vars['q_values']],
                             feed_dict=feed_dict)

        return td_error_batch if is_batch else td_error_batch[0], loss_batch if is_batch else loss_batch[0]

    # ------------------------------------------------------------------------------------------------------------------

    def get_td_target(self, reward_batch_in, obs_next_batch_in, done_batch_in):

        is_batch = isinstance(reward_batch_in, list) or isinstance(reward_batch_in, np.ndarray)

        obs_next_batch = obs_next_batch_in if isinstance(obs_next_batch_in, list) \
            else obs_next_batch_in.astype(dtype=np.float32)

        reward_batch = super().fix_batch_form(reward_batch_in, is_batch)
        obs_next_batch = super().fix_batch_form(obs_next_batch, is_batch)
        done_batch = super().fix_batch_form(done_batch_in, is_batch)

        feed_dict = {self.tf_vars['obs']: obs_next_batch,
                     self.tf_vars['obs_tar']: obs_next_batch}

        q_values_next_batch, q_values_next_target_batch = \
            self.session.run([self.tf_vars['q_values'], self.tf_vars['q_values_tar']], feed_dict=feed_dict)

        action_next_batch = np.argmax(q_values_next_batch, axis=1)

        td_target_batch = []
        for j in range(len(reward_batch)):
            td_target = reward_batch[j] + (1.0 - done_batch[j]) * self.config['dqn_gamma'] * \
                        q_values_next_target_batch[j][action_next_batch[j]]
            td_target_batch.append(td_target)

        return td_target_batch if is_batch else td_target_batch[0]

    # ------------------------------------------------------------------------------------------------------------------

    def arrange_feed_dict(self, minibatch):
        is_batch = isinstance(minibatch['reward'], list) or isinstance(minibatch['reward'], np.ndarray)

        obs_batch = minibatch['obs'] if isinstance(minibatch['obs'], list) \
            else minibatch['obs'].astype(dtype=np.float32)

        obs_next_batch = minibatch['obs_next'] if isinstance(minibatch['obs_next'], list) \
            else minibatch['obs_next'].astype(dtype=np.float32)

        obs_batch = super().fix_batch_form(obs_batch, is_batch)
        action_batch = super().fix_batch_form(minibatch['action'], is_batch)
        reward_batch = super().fix_batch_form(minibatch['reward'], is_batch)
        obs_next_batch = super().fix_batch_form(obs_next_batch, is_batch)
        done_batch = super().fix_batch_form(minibatch['done'], is_batch)
        td_target_batch = self.get_td_target(reward_batch, obs_next_batch, done_batch)

        feed_dict = {self.tf_vars['obs']: obs_batch,
                     self.tf_vars['action']: action_batch,
                     self.tf_vars['td_target']: td_target_batch}

        source_batch = super().fix_batch_form(minibatch['source'], is_batch)
        feed_dict[self.tf_vars['source']] = source_batch

        if self.config['dqn_rm_type'] == 'per' and self.config['dqn_per_ims']:
            ims_weights_batch = super().fix_batch_form(minibatch['weights'], is_batch)
            feed_dict[self.tf_vars['ims_weights']] = ims_weights_batch

        return feed_dict, is_batch

    # ------------------------------------------------------------------------------------------------------------------
    # Returns: action, is_action_explorative
    def get_action(self, obs):
        if random.random() < self.eps:
            return super().random_action(), True
        else:
            return self.get_greedy_action(obs), False

    # ------------------------------------------------------------------------------------------------------------------

    def get_greedy_action(self, obs):
        q_values = self.get_q_values(obs)
        return np.argmax(q_values)

    def get_random_action(self):
        return super().random_action(), True
