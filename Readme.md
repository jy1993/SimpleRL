# SimpleRL
My implementations of RL algorithms like GRPO/GSPO with minimal code.

# RL Infrastructure
![rl_infra](/assets/images/rl-infra.png)

# Features
+ Supported models
	+ Qwen2/Qwen2.5/Qwen3 language models
	+ Qwen2.5 vision language models
+ Supported algorithms
	+ GRPO
	+ Dr-GRPO
	+ GSPO
	+ KL-Conv
	+ StableReinforce
+ Supported tricks
	+ clip higher from DAPO
	+ token level policy loss
	+ dual clip
	+ kl term removal

# Requirements
	pip install -r requirements

# Train on logic
	bash scripts/run_logic.sh

## Dataset
https://github.com/Unakar/Logic-RL/tree/main/data/kk/instruct

## Results on test set
| Size |  Algorithm  | Bits |  LR  | KL  | Group Size | Steps | Test Score |
| ---- | ----------- | ---- | ---- | --- | ---------- | ----- | ---------- |
|  3B  |    GRPO     | AMP  | 1e-6 |  0  |     8      |  1600 | 0.12->0.54 |
|  7B  |    GRPO     | AMP  | 1e-6 |  0  |     8      |  1350 | 0.23->0.89 |

## Details
|        Model              | 2ppl | 3ppl | 4ppl | 5ppl | 6ppl | 7ppl | 8ppl |
| ------------------------- | ---  | ---- | ---- | ---- | ---- | ---- | ---- |
| Qwen2.5-3B-Instruct       | 0.37 | 0.13 | 0.17 | 0.12 | 0.04 | 0.02 | 0.02 |
| Qwen2.5-3B-Instruct-GRPO  | 0.76 | 0.70 | 0.68 | 0.50 | 0.47 | 0.33 | 0.33 |
| Qwen2.5-7B-Instruct       | 0.56 | 0.35 | 0.23 | 0.25 | 0.14 | 0.09 | 0.02 |
| Qwen2.5-7B-Instruct-GRPO  | 0.97 | 0.96 | 0.96 | 0.94 | 0.88 | 0.79 | 0.72 |

## Training curves:
### train_curves_of_qwen2.5_3B_instruct_grpo
![train_curves_of_qwen2.5_3b_instruct_on_logic](/assets/images/train_curves_of_qwen2.5_3b_instruct_on_logic.png)

### train_curves_of_qwen2.5_7B_instruct_grpo
![train_curves_of_qwen2.5_7b_instruct_on_logic](/assets/images/train_curves_of_qwen2.5_7b_instruct_on_logic.png)

# Train on geometry3k
	bash scripts/run_geometry3k.sh

## dataset
https://huggingface.co/datasets/hiyouga/geometry3k

## Results on test set
| Size |    Algorithm    | Bits |  LR  | KL  | Group Size | Steps | Test Score |
| ---- | --------------- | ---- | ---- | --- | ---------- | ----- | ---------- |
|  3B  |      GRPO       | AMP  | 1e-6 |  0  |     8      |  700  | 0.24->0.43 |
|  3B  |      GSPO       | AMP  | 1e-6 |  0  |     8      |  750  | 0.24->0.43 |
|  3B  | stable reinforce| AMP  | 1e-6 |  0  |     12     |  1200 | 0.25->0.44 |
|  3B  |     kl-conv     | AMP  | 1e-6 |  0  |     12     |  900  | 0.23->0.45 |
|  7B  |      GRPO       | AMP  | 1e-6 |  0  |     8      |  800  | 0.38->0.50 |

## Training curves:
### train_curves_of_qwen2.5_3B_vl_instruct_grpo
![train_curves_of_qwen2.5_3b_vl_instruct_on_geometry3k](/assets/images/train_curves_of_qwen2.5_3b_vl_instruct_on_geometry3k.png)

### train_cuvres_of_qwen2.5_7B_vl_instruct_grpo
![train_curves_of_qwen2.5_7b_vl_instruct_on_geometry3k](/assets/images/train_curves_of_qwen2.5_7b_vl_instruct_on_geometry3k.png)

# TODO
+ train on math
+ <del>train with Dr-GRPO/GSPO/KL-Conv/StableReinforce algos</del>
+ support dynamic sampling from dapo
+ support ppo/reinforce++/RLOO
+ <del>support vision language models</del>
+ support Retrieval-Augmented Reasoning
+ support agent training
+ support code eval