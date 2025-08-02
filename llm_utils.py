import torch
import json
import torch.nn.functional as F
from datasets import load_dataset
import os

def apply_chat_template(messages):
	text = ''
	for message in messages:
		text += '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'
	return text

def collate_for_lm_idcm(batch):
	return torch.LongTensor([x['input_ids'] for x in batch]), torch.LongTensor([x['attention_mask_in_length'] for x in batch]), torch.LongTensor([x['labels'] for x in batch])

def collate_for_lm(batch):
	return torch.LongTensor([x['input_ids'] for x in batch]), torch.LongTensor([x['labels'] for x in batch])

def collate_for_lm_simpo(batch):
	return torch.LongTensor([x['chosen_input_ids'] for x in batch]), torch.LongTensor([x['chosen_labels'] for x in batch]), torch.LongTensor([x['rejected_input_ids'] for x in batch]), torch.LongTensor([x['rejected_labels'] for x in batch])

def prepare_model_inputs(batch, intra_doc_causal_mask):
	if intra_doc_causal_mask:
		inputs = {'input_ids': batch[0], 'attention_mask_in_length': batch[1], 'labels': batch[2]}
	else:
		inputs = {'input_ids': batch[0], 'labels': batch[1]}
	return inputs

def prepare_model_inputs_for_simpo(batch):
	inputs = {'input_ids': torch.cat([batch[0], batch[2]], dim=0)}
	return inputs

def get_simpo_loss(args, batch, logits):
	chosen_logits, rejected_logits = torch.split(logits, logits.shape[0] // 2, dim=0)
	shift_chosen_logits = chosen_logits[:, :-1]
	shift_rejected_logits = rejected_logits[:, :-1]
	shift_chosen_labels = batch[1][:, 1:]
	shift_rejected_labels = batch[3][:, 1:]
	chosen_mask = shift_chosen_labels.ne(-100).to(logits.dtype)
	rejected_mask = shift_rejected_labels.ne(-100).to(logits.dtype)
	loss_fct = nn.CrossEntropyLoss(reduction='none')
	# bsz, seq_len
	chosen_loss = - loss_fct(shift_chosen_logits.transpose(1, 2), shift_chosen_labels)
	rejected_loss = - loss_fct(shift_rejected_logits.transpose(1, 2), shift_rejected_labels)
	chosen_loss_avg = torch.sum(chosen_loss * chosen_mask, dim=-1) / chosen_mask.sum(dim=-1)
	rejected_loss_avg = torch.sum(rejected_loss * rejected_mask, dim=-1) / rejected_mask.sum(dim=-1)
	loss = - nn.LogSigmoid()(args.beta * (chosen_loss_avg - rejected_loss_avg) - args.margin)
	return loss.mean()

def get_train_ds_config(offload,
						stage,
						global_batch_size,
						micro_batch_size,
						grad_acc,
						bf16=False,
						job_name=None,
						enable_hybrid_engine=False,
						inference_tp_size=1,
						release_inference_cache=False,
						pin_parameters=True,
						tp_gather_partition_size=8,
						max_out_tokens=512):
	device = "cpu" if offload else "none"
	zero_opt_dict = {
		"stage": stage,
		"offload_param": {
			"device": device,
			"pin_memory": True
		},
		"offload_optimizer": {
			"device": device,
			"pin_memory": True
		},
		"stage3_param_persistence_threshold": 1e4,
		"stage3_max_live_parameters": 3e7,
		"stage3_prefetch_bucket_size": 3e7,
		"memory_efficient_linear": False,
		"contiguous_gradients": False,
		"overlap_comm": True,
		"reduce_scatter": False
	}
	return {
		"train_batch_size": global_batch_size,
		"train_micro_batch_size_per_gpu": micro_batch_size,
		"steps_per_print": 500,
		"zero_optimization": zero_opt_dict,
		"fp16": {
			"enabled": True if not bf16 else False,
			"auto_cast": False,
			"loss_scale": 0,
			"initial_scale_power": 16,
			"loss_scale_window": 1000,
			"hysteresis": 2,
			"consecutive_hysteresis": False,
			"min_loss_scale": 1
		},
		"bf16":{
			"enabled": True if bf16 else False
		},
		"gradient_clipping": 1.0,
		"prescale_gradients": False,
		"wall_clock_breakdown": False,
		"gradient_acculation_steps": grad_acc,
		# "tensorboard": {
		# 	"enabled": True,
		# 	"output_path": "logs",
		# 	"job_name": job_name
		# },
	}

def NEFTune(model, noise_alpha=5):
    def noised_embed(orig_embed, noise_alpha):
        def new_func(x):
            # during training, we add noise to the embedding
            # during generation, we don't add noise to the embedding
            if model.training:
                embed_init = orig_embed(x)
                dims = torch.tensor(embed_init.size(1) * embed_init.size(2))
                mag_norm = noise_alpha/torch.sqrt(dims)
                return embed_init + torch.zeros_like(embed_init).uniform_(-mag_norm, mag_norm)
            else:
                return orig_embed(x)
        return new_func
    model.model.embed_tokens.forward = noised_embed(model.model.embed_tokens, noise_alpha)
    return model
