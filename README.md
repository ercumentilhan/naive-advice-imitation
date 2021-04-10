# Naive Advice Imitation

This repository contains the source code for the advice imitation algorithm proposed in the paper titled ["Student-Initiated Action Advising via Advice Novelty"](http://diego-perez.net/papers/ActionAdvising_AAMAS21.pdf).

## Requirements

**Packages (and their tested versions with Python 3.7):**
- numpy=1.19.4
- gym=0.18.0
- tensorflow=2.2.0
- pathlib2=2.3.5

**Environment:**
- [OpenAI Gym - Arcade Learning Environment](https://github.com/openai/gym/blob/master/docs/environments.md#atari)

## Execution

The code can be executed simply by running
```
python code/main.py
```

The input arguments with their default values can be found in `code/main.py`.

---

The teacher agent(s) to be used can be set in `code/constants.py` with their saved model checkpoints as follows:
```python
TEACHER = {
    'ALE-Enduro': ('ALE24V0_EG_000_20201105-130625', '0', 6000e3),
    'ALE-Freeway': ('ALE26V0_EG_000_20201105-172634', '0', 3000e3),
    'ALE-Pong': ('ALE43V0_EG_000_20201106-011948', '0', 5800e3),
}
```
where each dictionary entry uses the following format:
```
<Game Name>: (<Model directory>, <Model subdirectory (seed)>, <Checkpoint (timesteps)>)
```
We provide these example teacher models in the `checkpoints` directory.

---

Here are some example commands to run 3 types of experiments (No Advising, Early Advising, Advice Imitation) in Freeway game:

```
python main.py --run-id ALE26V0_NA --process-index 0 --machine-name HOME --n-training-frames 3000000 --n-evaluation-trials 10 --evaluation-period 25000 --evaluation-visualization-period 10 --dqn-gamma 0.99 --dqn-rm-type uniform --dqn-per-alpha 0.4 --dqn-per-beta 0.6 --dqn-rm-init 50000 --dqn-rm-max 500000 --dqn-target-update 7500 --dqn-batch-size 32 --dqn-learning-rate 6.25e-05 --dqn-train-per-step 1 --dqn-train-period 4 --dqn-adam-eps 0.00015 --dqn-eps-start 1.0 --dqn-eps-final 0.01 --dqn-eps-steps 500000 --dqn-huber-loss-delta 1.0 --dqn-hidden-size 512 --model-save-period 100000 --visualization-period 200 --env-name ALE-Freeway --env-training-seed 0 --env-evaluation-seed 1 --seed 103 --action-advising-method none
```

```
python main.py --run-id ALE26V0_EA --process-index 0 --machine-name HOME --n-training-frames 3000000 --n-evaluation-trials 10 --evaluation-period 25000 --evaluation-visualization-period 10 --dqn-gamma 0.99 --dqn-rm-type uniform --dqn-per-alpha 0.4 --dqn-per-beta 0.6 --dqn-rm-init 50000 --dqn-rm-max 500000 --dqn-target-update 7500 --dqn-batch-size 32 --dqn-learning-rate 6.25e-05 --dqn-train-per-step 1 --dqn-train-period 4 --dqn-adam-eps 0.00015 --dqn-eps-start 1.0 --dqn-eps-final 0.01 --dqn-eps-steps 500000 --dqn-huber-loss-delta 1.0 --dqn-hidden-size 512 --model-save-period 100000 --visualization-period 200 --env-name ALE-Freeway --env-training-seed 0 --env-evaluation-seed 1 --seed 103 --action-advising-method ea --action-advising-budget 10000
```

```
python main.py --run-id ALE26V0_AR --process-index 0 --machine-name HOME --n-training-frames 3000000 --n-evaluation-trials 10 --evaluation-period 25000 --evaluation-visualization-period 10 --dqn-gamma 0.99 --dqn-rm-type uniform --dqn-per-alpha 0.4 --dqn-per-beta 0.6 --dqn-rm-init 50000 --dqn-rm-max 500000 --dqn-target-update 7500 --dqn-batch-size 32 --dqn-learning-rate 6.25e-05 --dqn-train-per-step 1 --dqn-train-period 4 --dqn-adam-eps 0.00015 --dqn-eps-start 1.0 --dqn-eps-final 0.01 --dqn-eps-steps 500000 --dqn-huber-loss-delta 1.0 --dqn-hidden-size 512 --model-save-period 100000 --visualization-period 200 --env-name ALE-Freeway --env-training-seed 0 --env-evaluation-seed 1 --seed 103 --action-advising-method ai --action-advising-budget 10000 --bc-uc-threshold 0.01 --bc-batch-size 32 --bc-learning-rate 0.0001 --bc-adam-eps 0.00015 --bc-dropout-rate 0.2 --bc-training-iters 50000 --bc-hidden-size 512 --bc-uc-ensembles 100
```
