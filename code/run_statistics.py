import tensorflow as tf


class Statistics(object):
    def __init__(self, summary_writer, session):

        self.n_steps_per_update = 100

        self.summary_writer = summary_writer
        self.session = session

        self.n_evaluations = 0

        # Number of environment interactions
        self.n_env_steps = 0
        self.n_env_steps_var = tf.compat.v1.Variable(0)

        # Number of episodes
        self.n_episodes = 0
        self.n_episodes_var = tf.compat.v1.Variable(0)

        self.loss = 0.0
        self.loss_var = tf.compat.v1.Variable(0.0)
        self.loss_ph = tf.compat.v1.placeholder(tf.compat.v1.float32)

        self.epsilon = 0.0
        self.epsilon_var = tf.compat.v1.Variable(0.0)
        self.epsilon_ph = tf.compat.v1.placeholder(tf.compat.v1.float32)

        self.exploration_steps_taken = 0.0
        self.exploration_steps_taken_var = tf.compat.v1.Variable(0.)
        self.exploration_steps_taken_ph = tf.compat.v1.placeholder(tf.compat.v1.float32)

        self.exploration_steps_taken_cum = 0.0
        self.exploration_steps_taken_cum_var = tf.compat.v1.Variable(0.)
        self.exploration_steps_taken_cum_ph = tf.compat.v1.placeholder(tf.compat.v1.float32)

        # --------------------------------------------------------------------------------------------------------------
        # Advice Exchange

        self.advices_taken = 0
        self.advices_taken_var = tf.compat.v1.Variable(0.0)
        self.advices_taken_ph = tf.compat.v1.placeholder(tf.compat.v1.float32)

        self.advices_taken_cum = 0
        self.advices_taken_cum_var = tf.compat.v1.Variable(0.0)
        self.advices_taken_cum_ph = tf.compat.v1.placeholder(tf.compat.v1.float32)

        # --------------------------------------------------------------------------------------------------------------
        # Advice use (Collect || Reuse)

        self.advices_used = 0
        self.advices_used_var = tf.compat.v1.Variable(0.0)
        self.advices_used_ph = tf.compat.v1.placeholder(tf.compat.v1.float32)

        self.advices_used_cum = 0
        self.advices_used_cum_var = tf.compat.v1.Variable(0.0)
        self.advices_used_cum_ph = tf.compat.v1.placeholder(tf.compat.v1.float32)

        # --------------------------------------------------------------------------------------------------------------
        # Advice Reuse

        self.advices_reused = 0
        self.advices_reused_var = tf.compat.v1.Variable(0.0)
        self.advices_reused_ph = tf.compat.v1.placeholder(tf.compat.v1.float32)

        self.advices_reused_cum = 0
        self.advices_reused_cum_var = tf.compat.v1.Variable(0.0)
        self.advices_reused_cum_ph = tf.compat.v1.placeholder(tf.compat.v1.float32)

        self.advices_reused_correct = 0
        self.advices_reused_correct_var = tf.compat.v1.Variable(0.0)
        self.advices_reused_correct_ph = tf.compat.v1.placeholder(tf.compat.v1.float32)

        self.advices_reused_correct_cum = 0
        self.advices_reused_correct_cum_var = tf.compat.v1.Variable(0.0)
        self.advices_reused_correct_cum_ph = tf.compat.v1.placeholder(tf.compat.v1.float32)

        # --------------------------------------------------------------------------------------------------------------

        # Steps
        self.steps_reward_var = tf.compat.v1.Variable(0.)
        self.steps_reward_ph = tf.compat.v1.placeholder(tf.compat.v1.float32)

        self.steps_error_in_var = tf.compat.v1.Variable(0.)
        self.steps_error_in_ph = tf.compat.v1.placeholder(tf.compat.v1.float32)

        self.steps_error_out_var = tf.compat.v1.Variable(0.)
        self.steps_error_out_ph = tf.compat.v1.placeholder(tf.compat.v1.float32)

        self.steps_reward_last = 0.0
        self.steps_reward_auc = 0.0
        self.steps_reward_auc_var = tf.compat.v1.Variable(0.)
        self.steps_reward_auc_ph = tf.compat.v1.placeholder(tf.compat.v1.float32)

        # Episodes
        self.episode_reward_var = tf.compat.v1.Variable(0.)
        self.episode_reward_ph = tf.compat.v1.placeholder(tf.compat.v1.float32)

        self.episode_error_in_var = tf.compat.v1.Variable(0.)
        self.episode_error_in_ph = tf.compat.v1.placeholder(tf.compat.v1.float32)

        self.episode_error_out_var = tf.compat.v1.Variable(0.)
        self.episode_error_out_ph = tf.compat.v1.placeholder(tf.compat.v1.float32)

        self.episode_duration = 0
        self.episode_duration_var = tf.compat.v1.Variable(0.)
        self.episode_duration_ph = tf.compat.v1.placeholder(tf.compat.v1.float32)

        self.episode_reward_last = 0.0
        self.episode_reward_auc = 0.0
        self.episode_reward_auc_var = tf.compat.v1.Variable(0.)
        self.episode_reward_auc_ph = tf.compat.v1.placeholder(tf.compat.v1.float32)

        # Evaluation
        self.evaluation_reward_var = tf.compat.v1.Variable(0.)
        self.evaluation_reward_ph = tf.compat.v1.placeholder(tf.compat.v1.float32)

        self.evaluation_duration = 0
        self.evaluation_duration_var = tf.compat.v1.Variable(0.)
        self.evaluation_duration_ph = tf.compat.v1.placeholder(tf.compat.v1.float32)

        self.evaluation_reward_last = 0.0

        self.evaluation_reward_auc = 0.0
        self.evaluation_reward_auc_var = tf.compat.v1.Variable(0.)
        self.evaluation_reward_auc_ph = tf.compat.v1.placeholder(tf.compat.v1.float32)

        # RM Evaluation
        self.rm_uncertainty_self = 0.0
        self.rm_uncertainty_self_var = tf.compat.v1.Variable(0.)
        self.rm_uncertainty_self_ph = tf.compat.v1.Variable(0.)

        self.rm_uncertainty_12 = 0.0
        self.rm_uncertainty_12_var = tf.compat.v1.Variable(0.)
        self.rm_uncertainty_12_ph = tf.compat.v1.Variable(0.)

        self.rm_uncertainty_22 = 0.0
        self.rm_uncertainty_22_var = tf.compat.v1.Variable(0.)
        self.rm_uncertainty_22_ph = tf.compat.v1.Variable(0.)

        # --------------------------------------------------------------------------------------------------------------

        self.summary_op_steps = self.setup_summary_steps()
        self.summary_op_episode = self.setup_summary_episode()
        self.summary_op_evaluation = self.setup_summary_evaluation()
        self.summary_op_rm_evaluation = self.setup_summary_rm_evaluation()

        # --------------------------------------------------------------------------------------------------------------

        self.assignments_steps = [
            self.loss_var.assign(self.loss_ph),
            self.epsilon_var.assign(self.epsilon_ph),
            self.exploration_steps_taken_var.assign(self.exploration_steps_taken_ph),
            self.exploration_steps_taken_cum_var.assign(self.exploration_steps_taken_cum_ph),
            self.advices_taken_var.assign(self.advices_taken_ph),
            self.advices_taken_cum_var.assign(self.advices_taken_cum_ph),
            self.advices_used_var.assign(self.advices_used_ph),
            self.advices_used_cum_var.assign(self.advices_used_cum_ph),
            self.advices_reused_var.assign(self.advices_reused_ph),
            self.advices_reused_cum_var.assign(self.advices_reused_cum_ph),
            self.advices_reused_correct_var.assign(self.advices_reused_correct_ph),
            self.advices_reused_correct_cum_var.assign(self.advices_reused_correct_cum_ph),

            self.steps_reward_var.assign(self.steps_reward_ph),
            self.steps_reward_auc_var.assign(self.steps_reward_auc_ph),
            self.steps_error_in_var.assign(self.steps_error_in_ph),
            self.steps_error_out_var.assign(self.steps_error_out_ph),
        ]

        self.assignments_episode = [
            self.episode_reward_var.assign(self.episode_reward_ph),
            self.episode_reward_auc_var.assign(self.episode_reward_auc_ph),
            self.episode_duration_var.assign(self.episode_duration_ph),
            self.episode_error_in_var.assign(self.episode_error_in_ph),
            self.episode_error_out_var.assign(self.episode_error_out_ph),
        ]

        self.assignments_evaluation = [
            self.evaluation_reward_var.assign(self.evaluation_reward_ph),
            self.evaluation_duration_var.assign(self.evaluation_duration_ph),
            self.evaluation_reward_auc_var.assign(self.evaluation_reward_auc_ph)
        ]

        self.assignments_rm_evaluation = [
            self.rm_uncertainty_self_var.assign(self.rm_uncertainty_self_ph),
            self.rm_uncertainty_12_var.assign(self.rm_uncertainty_12_ph),
            self.rm_uncertainty_22_var.assign(self.rm_uncertainty_22_ph),
        ]

    # ------------------------------------------------------------------------------------------------------------------

    def setup_summary_steps(self):
        loss_sc = tf.compat.v1.summary.scalar('Loss', self.loss_var)
        epsilon_sc = tf.compat.v1.summary.scalar('Epsilon', self.epsilon_var)

        exploration_steps_taken_sc = tf.compat.v1.summary.scalar('Exploration Steps Taken', self.exploration_steps_taken_var)
        exploration_steps_taken_cum_sc = tf.compat.v1.summary.scalar('Exploration Steps Taken Cumulative', self.exploration_steps_taken_cum_var)

        advices_taken_sc = tf.compat.v1.summary.scalar('Advices Taken', self.advices_taken_var)
        advices_taken_cumulative_sc = tf.compat.v1.summary.scalar('Advices Taken Cumulative', self.advices_taken_cum_var)

        advices_used_sc = tf.compat.v1.summary.scalar('Advices Used', self.advices_used_var)
        advices_used_cum_sc = tf.compat.v1.summary.scalar('Advices Used Cumulative', self.advices_used_cum_var)

        advices_reused_sc = tf.compat.v1.summary.scalar('Advices Reused', self.advices_reused_var)
        advices_reused_cum_sc = tf.compat.v1.summary.scalar('Advices Reused Cumulative', self.advices_reused_cum_var)

        advices_reused_correct_sc = tf.compat.v1.summary.scalar('Advices Reused Correct', self.advices_reused_correct_var)
        advices_reused_correct_cum_sc = tf.compat.v1.summary.scalar('Advices Reused Correct Cumulative', self.advices_reused_correct_cum_var)

        steps_reward_sc = tf.compat.v1.summary.scalar('Reward/Steps', self.steps_reward_var)
        steps_reward_auc_sc = tf.compat.v1.summary.scalar('Reward AUC/Steps', self.steps_reward_auc_var)
        steps_error_in_sc = tf.compat.v1.summary.scalar('Error In/Steps', self.steps_error_in_var)
        steps_error_out_sc = tf.compat.v1.summary.scalar('Error Out/Steps', self.steps_error_out_var)

        to_be_merged = [loss_sc, epsilon_sc, advices_taken_sc, advices_taken_cumulative_sc, steps_reward_sc,
                        steps_reward_auc_sc, exploration_steps_taken_sc, exploration_steps_taken_cum_sc,
                        advices_used_sc, advices_used_cum_sc,
                        advices_reused_sc, advices_reused_cum_sc,
                        advices_reused_correct_sc, advices_reused_correct_cum_sc,
                        steps_error_in_sc, steps_error_out_sc]

        return tf.compat.v1.summary.merge(to_be_merged)

    # ------------------------------------------------------------------------------------------------------------------

    def setup_summary_episode(self):
        episode_reward_sc = tf.compat.v1.summary.scalar('Reward/Episode', self.episode_reward_var)
        episode_reward_auc_sc = tf.compat.v1.summary.scalar('Reward AUC/Episode', self.episode_reward_auc_var)
        episode_duration_sc = tf.compat.v1.summary.scalar('Episode Duration', self.episode_duration_var)

        episode_error_in_sc = tf.compat.v1.summary.scalar('Error In/Episode', self.episode_error_in_var)
        episode_error_out_sc = tf.compat.v1.summary.scalar('Error Out/Episode', self.episode_error_out_var)

        to_be_merged = [episode_reward_sc, episode_reward_auc_sc, episode_duration_sc,
                        episode_error_in_sc, episode_error_out_sc]

        return tf.compat.v1.summary.merge(to_be_merged)

    # ------------------------------------------------------------------------------------------------------------------

    def setup_summary_evaluation(self):
        evaluation_reward_sc = tf.compat.v1.summary.scalar('Evaluation Reward', self.evaluation_reward_var)
        evaluation_duration_sc = tf.compat.v1.summary.scalar('Evaluation Duration', self.evaluation_duration_var)
        evaluation_reward_auc_sc = tf.compat.v1.summary.scalar('Evaluation Reward AUC', self.evaluation_reward_auc_var)
        return tf.compat.v1.summary.merge([evaluation_reward_sc, evaluation_duration_sc, evaluation_reward_auc_sc])

    # ------------------------------------------------------------------------------------------------------------------

    def setup_summary_rm_evaluation(self):
        rm_uncertainty_self_sc = tf.compat.v1.summary.scalar('RM Uncertainty Self', self.rm_uncertainty_self_var)
        rm_uncertainty_12_sc = tf.compat.v1.summary.scalar('RM Uncertainty 12', self.rm_uncertainty_12_var)
        rm_uncertainty_22_sc = tf.compat.v1.summary.scalar('RM Uncertainty 22', self.rm_uncertainty_22_var)
        return tf.compat.v1.summary.merge([rm_uncertainty_self_sc, rm_uncertainty_12_sc, rm_uncertainty_22_sc])

    # ------------------------------------------------------------------------------------------------------------------

    def update_summary_steps(self, steps_reward, steps_reward_auc, steps_error_in, steps_error_out):
        self.loss /= self.n_steps_per_update

        requested_ops = [assignment for assignment in self.assignments_steps]

        feed_dict = {
            self.loss_ph: self.loss,
            self.epsilon_ph: self.epsilon,
            self.exploration_steps_taken_ph: self.exploration_steps_taken,
            self.exploration_steps_taken_cum_ph: self.exploration_steps_taken_cum,
            self.advices_taken_ph: self.advices_taken,
            self.advices_taken_cum_ph: self.advices_taken_cum,
            self.advices_used_ph: self.advices_used,
            self.advices_used_cum_ph: self.advices_used_cum,
            self.advices_reused_ph: self.advices_reused,
            self.advices_reused_cum_ph: self.advices_reused_cum,
            self.advices_reused_correct_ph: self.advices_reused_correct,
            self.advices_reused_correct_cum_ph: self.advices_reused_correct_cum,
            self.steps_reward_ph: steps_reward,
            self.steps_reward_auc_ph: steps_reward_auc,
            self.steps_error_in_ph: steps_error_in,
            self.steps_error_out_ph: steps_error_out}

        self.session.run(requested_ops, feed_dict=feed_dict)
        summary = self.session.run(self.summary_op_steps)
        self.summary_writer.add_summary(summary, self.n_env_steps)

        self.loss = 0.0

    # ------------------------------------------------------------------------------------------------------------------

    def update_summary_episode(self, episode_reward, episode_reward_auc, episode_duration,
                               episode_error_in, episode_error_out):
        requested_ops = [assignment for assignment in self.assignments_episode]

        feed_dict = {
            self.episode_reward_ph: episode_reward,
            self.episode_reward_auc_ph: episode_reward_auc,
            self.episode_duration_ph: episode_duration,
            self.episode_error_in_ph: episode_error_in,
            self.episode_error_out_ph: episode_error_out}

        self.session.run(requested_ops, feed_dict=feed_dict)
        summary = self.session.run(self.summary_op_episode)
        self.summary_writer.add_summary(summary, self.n_episodes)

    # ------------------------------------------------------------------------------------------------------------------

    def update_summary_evaluation(self, evaluation_reward, evaluation_duration, evaluation_reward_auc):
        requested_ops = [assignment for assignment in self.assignments_evaluation]

        feed_dict = {
            self.evaluation_reward_ph: evaluation_reward,
            self.evaluation_duration_ph: evaluation_duration,
            self.evaluation_reward_auc_ph: evaluation_reward_auc}

        self.session.run(requested_ops, feed_dict=feed_dict)
        summary = self.session.run(self.summary_op_evaluation)
        self.summary_writer.add_summary(summary, self.n_env_steps)

    # ------------------------------------------------------------------------------------------------------------------

    def update_summary_rm_evaluation(self, rm_uncertainty_self, rm_uncertainty_12, rm_uncertainty_22):
        requested_ops = [assignment for assignment in self.assignments_rm_evaluation]

        feed_dict = {
            self.rm_uncertainty_self_ph: rm_uncertainty_self,
            self.rm_uncertainty_12_ph: rm_uncertainty_12,
            self.rm_uncertainty_22_ph: rm_uncertainty_22}

        self.session.run(requested_ops, feed_dict=feed_dict)
        summary = self.session.run(self.summary_op_rm_evaluation)
        self.summary_writer.add_summary(summary, self.n_env_steps)