# SimpleRL
My implementations of RL algorithms like GRPO/GSPO with minimal code(about 1400 lines).

# Features
+ Supported models
	+ Qwen2/Qwen2.5/Qwen3 language models
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
|  7B  |    GRPO     | AMP  | 1e-6 |  0  |     8      |  750  | 0.20->0.69 |

## Details
|        Model              | 2ppl | 3ppl | 4ppl | 5ppl | 6ppl | 7ppl | 8ppl |
| ------------------------- | ---  | ---- | ---- | ---- | ---- | ---- | ---- |
| Qwen2.5-3B-Instruct       | 0.37 | 0.13 | 0.17 | 0.12 | 0.04 | 0.02 | 0.02 |
| Qwen2.5-3B-Instruct-GRPO  | 0.76 | 0.70 | 0.68 | 0.50 | 0.47 | 0.33 | 0.33 |
| Qwen2.5-7B-Instruct       | 0.51 | 0.34 | 0.24 | 0.15 | 0.11 | 0.03 | 0.05 |
| Qwen2.5-7B-Instruct-GRPO  | 0.86 | 0.77 | 0.81 | 0.75 | 0.63 | 0.49 | 0.54 |

## Training curves:
### train_curves_of_qwen2.5_3B_instruct
![train_curves_of_qwen2.5_3b_instruct_on_logic](/assets/images/train_curves_of_qwen2.5_3b_instruct_on_logic.png)

### train_curves_of_qwen2.5_7B_instruct
![train_curves_of_qwen2.5_7b_instruct_on_logic](/assets/images/train_curves_of_qwen2.5_7b_instruct_on_logic.png)

# TODO
+ train on math
+ train with Dr-GRPO/GSPO/KL-Conv/StableReinforce algos
+ support dynamic sampling from dapo
+ support ppo/reinforce++/RLOO
+ support vision language models
+ support agent training
+ support code eval
