# Built on OpenAI baselines:
# https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
# https://github.com/openai/baselines/commit/7859f603cd4df3b647318f2f6e3e68555ea8d4d8

import random
import numpy as np
from segment_tree import SumSegmentTree, MinSegmentTree


class ReplayBuffer(object):
    def __init__(self, size, extra_content):
        """Create Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

        # determines the # of first N samples to be preserved forever - set after adding samples
        self.n_preserved = 0

        self.extra_content = extra_content

    def __len__(self):
        return len(self._storage)

    def add(self, transition):  # obs_t, action, reward, obs_tp1, done, *args, **kwargs)
        old_data = None

        data = [
            transition['obs'],
            transition['action'],
            transition['reward'],
            transition['obs_next'],
            transition['done']
        ]

        for content in self.extra_content:
            data.append(transition[content])

        data = tuple(data)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            old_data = self._storage[self._next_idx]
            self._storage[self._next_idx] = data

        if self.n_preserved > 0:
            self._next_idx = self.n_preserved + \
                             int((self._next_idx + 1 - self.n_preserved) % (self._maxsize - self.n_preserved))
        else:
            self._next_idx = int((self._next_idx + 1) % self._maxsize)

        return old_data

    def _encode_sample(self, idxes, in_numpy_form):
        sample = {
            'obs_t': [],
            'action': [],
            'reward': [],
            'obs_tp1': [],
            'done': []
        }

        for content in self.extra_content:
            sample[content] = []

        for i in idxes:
            data = self._storage[i]

            sample['obs_t'].append(np.array(data[0], copy=False))
            sample['action'].append(np.array(data[1], copy=False))
            sample['reward'].append(data[2])
            sample['obs_tp1'].append(np.array(data[3], copy=False))
            sample['done'].append(data[4])

            for j, content in enumerate(self.extra_content):
                sample[content].append(data[5 + j])

        extra_content = {}
        for content in self.extra_content:
            if in_numpy_form:
                extra_content[content] = np.array(sample[content])
            else:
                extra_content[content] = sample[content]

        if in_numpy_form:
            return np.array(sample['obs_t']),  np.array(sample['action']), np.array(sample['reward']), \
                   np.array(sample['obs_tp1']), np.array(sample['done']), extra_content
        else:
            return sample['obs_t'], sample['action'], sample['reward'], \
                   sample['obs_tp1'], sample['done'], extra_content


    def sample(self, batch_size, in_numpy_form=True):
        """Sample a batch of experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes, in_numpy_form)

class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, size, extra_content, alpha):
        """Create Prioritized Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        alpha: float
            how much prioritization is used
            (0 - no prioritization, 1 - full prioritization)
        See Also
        --------
        ReplayBuffer.__init__
        """
        super(PrioritizedReplayBuffer, self).__init__(size, extra_content)
        assert alpha >= 0
        self._alpha = alpha

        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def add(self, *args, **kwargs):
        """See ReplayBuffer.store_effect"""

        idx = self._next_idx
        old_data = super().add(*args, **kwargs)

        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

        return old_data



    def _sample_proportional(self, batch_size):
        res = []
        p_total = self._it_sum.sum(0, len(self._storage) - 1)
        #if i == 0:
        #print(len(self._storage) - 1, p_total)
        every_range_len = p_total / batch_size
        #print('sampling...', len(self._storage))
        for i in range(batch_size):
            r = random.random()
            mass = np.float32(r * every_range_len + i * every_range_len)
            #if i == 0:
            #    print(r, every_range_len, p_total, len(self._storage)) # p_total is different
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def sample(self, batch_size, beta, in_numpy_form=True):
        """Sample a batch of experiences.
        compared to ReplayBuffer.sample
        it also returns importance weights and idxes
        of sampled experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        beta: float
            To what degree to use importance weights
            (0 - no corrections, 1 - full correction)
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        weights: np.array
            Array of shape (batch_size,) and dtype np.float32
            denoting importance weight of each sampled transition
        idxes: np.array
            Array of shape (batch_size,) and dtype np.int32
            idexes in buffer of sampled experiences
        """
        assert beta > 0

        idxes = self._sample_proportional(batch_size)

        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self._storage)) ** (-beta)

        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self._storage)) ** (-beta)
            weights.append(weight / max_weight)
        weights = np.array(weights)
        encoded_sample = self._encode_sample(idxes, in_numpy_form)
        return tuple(list(encoded_sample) + [weights, idxes])

    def update_priorities(self, idxes, priorities):
        """Update priorities of sampled transitions.
        sets priority of transition at index idxes[i] in buffer
        to priorities[i].
        Parameters
        ----------
        idxes: [int]
            List of idxes of sampled transitions
        priorities: [float]
            List of updated priorities corresponding to
            transitions at the sampled idxes denoted by
            variable `idxes`.
        """
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            assert priority > 0
            assert 0 <= idx < len(self._storage)
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha

            self._max_priority = max(self._max_priority, priority)

            #print(self._max_priority)
            #print(self._max_priority)