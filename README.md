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
```python
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
```python
<Game Name>: (<Model directory>, <Model subdirectory (seed)>, <Checkpoint (timesteps)>)
```
We provide these example teacher models in the `checkpoints` directory.