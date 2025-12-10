import random
import torch
from torch.utils.tensorboard import SummaryWriter
import json
from tqdm import tqdm, trange
from grpo_utils import GRPOBuffer, caculate_grpo_loss, caculate_gspo_loss, caculate_grpo_loss_kl_cov, gather_logps, get_entropy, get_weight_ipc_handles
from data_utils import read_jsonl, is_equal, process_multi_modal_data, extract_answer_from_model_response
from llm_utils import get_train_ds_config, get_all_reduce_mean, get_all_reduce_max, get_all_reduce_min, broadcast_object_list
from agent_utils import repeat_dict, call_python_server
import os
import copy
from pprint import pprint
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM, Qwen2_5_VLForConditionalGeneration, AutoProcessor, get_linear_schedule_with_warmup, get_constant_schedule_with_warmup, StoppingCriteriaList
# from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, get_linear_schedule_with_warmup, get_constant_schedule_with_warmup, StoppingCriteriaList
from args import get_grpo_args
import torch
from torch.nn import DataParallel
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SequentialSampler
import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from vllm import LLM, SamplingParams
from unittest.mock import patch
import numpy as np
from safetensors.torch import save_file
from prompt import get_prompt_prefix
from datetime import timedelta
# from mathruler.grader import extract_boxed_content, grade_answer

class FixedAutoTokenizer(AutoTokenizer):
	def convert_tokens_to_string(self, tokens):
		tokens = [t for t in tokens if t is not None]
		return super().convert_tokens_to_string(tokens)

class Trainer(object):
	"""docstring for Trainer"""
	def __init__(self, args):
		super(Trainer, self).__init__()
		self.args = args
		if self.args.task_type in ['vl_math']:
			policy_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model_path, attn_implementation='flash_attention_2', torch_dtype=torch.bfloat16)
			if self.args.kl_coeff > 0:
				ref_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model_path, attn_implementation='flash_attention_2', torch_dtype=torch.bfloat16)
			else:
				ref_model = None
			self.processor = AutoProcessor.from_pretrained(args.model_path, min_pixels=args.min_vl_tokens*28*28, max_pixels=args.max_vl_tokens*28*28)
			self.processor.tokenizer.padding_side = 'left'
			self.tokenizer = self.processor.tokenizer
		elif self.args.task_type in ['logic', 'math', 'agent_math']:
			os.environ['VLLM_USE_V1'] = '0'
			policy_model = AutoModelForCausalLM.from_pretrained(args.model_path, attn_implementation='flash_attention_2', torch_dtype=torch.bfloat16)
			if self.args.kl_coeff > 0:
				ref_model = AutoModelForCausalLM.from_pretrained(args.model_path, attn_implementation='flash_attention_2', torch_dtype=torch.bfloat16)
			else:
				ref_model = None
			# self.tokenizer = Qwen2Tokenizer.from_pretrained(args.model_path)
			# self.tokenizer = FixedQwen2Tokenizer.from_pretrained(args.model_path)
			self.tokenizer = AutoTokenizer.from_pretrained(args.model_path)
			self.tokenizer.padding_side = 'left'
		self.system_prompt, self.prompt_prefix = get_prompt_prefix(self.args.task_type)
		self.data_buffer = GRPOBuffer(tokenizer=self.tokenizer, args=self.args)
		self.init_ds(policy_model, ref_model)
		self.onload_or_offload('cpu', None)
		self.train_data = read_jsonl(self.args.train_path)
		self.sampling_params = SamplingParams(
			n=self.args.group_size,
			temperature=self.args.temperature,
			top_p=self.args.topp,
			max_tokens=self.args.max_new_tokens,
			stop_token_ids=[151643, 151645],
			detokenize=False
		)
		self.eval_sampling_params = SamplingParams(
			n=1,
			temperature=0.5,
			top_p=0.99,
			repetition_penalty=1,
			max_tokens=self.args.max_new_tokens,
			stop_token_ids=[151643, 151645]
		)
		if self.args.task_type == 'agent_math':
			# self.start_str = '<code>\n```python'
			# self.stop_str = '```\n</code>'
			self.start_str = '```python\n'
			self.stop_str = '\n```\n'
			self.sampling_params_agent_stop = SamplingParams(
				n=1,
				temperature=self.args.temperature,
				top_p=self.args.topp,
				max_tokens=self.args.max_new_tokens,
				stop=[self.stop_str],
				include_stop_str_in_output=True,
				stop_token_ids=[151643, 151645]
			)
		if self.args.global_rank == 0:
			# this is for saturday training platform
			os.environ['USER'] = 'USER'
			self.eval_data = read_jsonl(self.args.test_path)
			self.writer = SummaryWriter(args.tensorboard_path)
			with patch("torch.distributed.get_world_size", return_value=self.args.ngpus_for_vllm):
				self.llm = LLM(
					model=self.args.model_path,
					tensor_parallel_size=self.args.ngpus_for_vllm,
					max_model_len=8192,
					gpu_memory_utilization=0.6,
					max_num_seqs=64,
					enforce_eager=True,
					enable_sleep_mode=True,
					disable_mm_preprocessor_cache=True,
					worker_extension_cls="grpo_utils.ColocateWorkerExtension",
				)
			self.llm.sleep(level=1)
			torch.cuda.empty_cache()
		torch.distributed.barrier()

	def onload_or_offload(self, ref_device=None, device=None):
		if ref_device is not None and self.ref_model_engine is not None:
			self.ref_model_engine.to(ref_device)
		if device is not None:
			self.model_engine.to(device)

	def init_ds(self, policy_model, ref_model):
		torch.cuda.set_device(self.args.local_rank)
		self.args.device = torch.device('cuda', self.args.local_rank)
		deepspeed.init_distributed(timeout=timedelta(seconds=3600))
		self.args.global_rank = torch.distributed.get_rank()
		self.args.n_gpus = torch.distributed.get_world_size()
		torch.distributed.barrier()
		no_decay = ["bias", "norm.weight"]
		optimizer_grouped_parameters = [
			{
				"params": [p for n, p in policy_model.named_parameters() if not any(nd in n for nd in no_decay)],
				"weight_decay": self.args.weight_decay, 
				"lr": args.lr
			},     
			{   
				"params": [p for n, p in policy_model.named_parameters() if any(nd in n for nd in no_decay)], 
				"weight_decay": 0.0, 
				"lr": self.args.lr
			}           
		]
		AdamOptimizer = DeepSpeedCPUAdam if self.args.offload else FusedAdam
		optimizer = AdamOptimizer(optimizer_grouped_parameters, lr=self.args.lr, betas=(0.9, 0.95))

		# lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=self.args.total_steps)
		lr_scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup_steps)
		ds_config = get_train_ds_config(offload=args.offload, 
			stage=self.args.zero_stage, 
			global_batch_size=self.args.per_device_train_batch_size*self.args.gradient_accumulation_steps*self.args.n_gpus,
			micro_batch_size=self.args.per_device_train_batch_size,
			grad_acc=self.args.gradient_accumulation_steps,
			bf16=self.args.bf16,
			job_name=self.args.exp_name)
		model_engine, _, _, _ = deepspeed.initialize(
			model=policy_model,
			optimizer=optimizer,
			args=self.args,
			config=ds_config,
			lr_scheduler=lr_scheduler,
			dist_init_required=True)
		if self.args.gradient_checkpointing:
			model_engine.gradient_checkpointing_enable()
		self.model_engine = model_engine
		if ref_model is not None:
			ref_model_engine = deepspeed.init_inference(
				model=ref_model, config={"dtype": 'bfloat16', "replace_with_kernel_inject": True})
			self.ref_model_engine = ref_model_engine
		else:
			self.ref_model_engine = None

	def generate_sequences(self, batch_prompts):
		outputs = self.llm.generate(batch_prompts, self.sampling_params)
		all_token_ids = []
		for output in outputs:
			for one in output.outputs:
				all_token_ids.append(list(one.token_ids))
		max_len = max(len(token_ids) for token_ids in all_token_ids)
		seqs = []
		for token_ids in all_token_ids:
			seqs.append(token_ids + [self.tokenizer.pad_token_id] * (max_len - len(token_ids)))
		return seqs

	def compute_logps(self, model, input_ids, attention_mask, images, logits_to_keep, offload=False):
		if images is not None:
			pixel_values, image_grid_thw = self.prepare_mm_inputs(images)
			logits = model.forward(input_ids, 
				attention_mask=attention_mask, 
				pixel_values=pixel_values,
				image_grid_thw=image_grid_thw,
				use_cache=False).logits
		else:
			logits = model.forward(input_ids, attention_mask=attention_mask, use_cache=False).logits
		if offload:
			logits = logits.cpu()
			input_ids = input_ids.cpu()
		logits = logits[:, :-1]
		logits = logits[:, -logits_to_keep:]
		logps = gather_logps(logits, input_ids[:, -logits_to_keep:])
		# entropy = get_entropy(logits, attention_mask[:, -logits_to_keep:])
		return logps

	def batch_compute_logps(self, model, input_ids, attention_mask, images, logits_to_keep):
		batch_logps = []
		mini_batch_size = self.args.mini_batch_size
		assert input_ids.shape[0] % mini_batch_size == 0
		if images is not None:
			assert input_ids.shape[0] == attention_mask.shape[0] == len(images), print(input_ids.shape, attention_mask.shape, len(images))
		for i in range(max(1, len(input_ids) // mini_batch_size)):
			logps = self.compute_logps(model, 
				input_ids[i*mini_batch_size:(i+1)*mini_batch_size], 
				attention_mask[i*mini_batch_size:(i+1)*mini_batch_size], 
				images[i*mini_batch_size:(i+1)*mini_batch_size] if images is not None else None,
				logits_to_keep)
			batch_logps.append(logps)
		batch_logps = torch.cat(batch_logps, dim=0)
		return batch_logps

	def generate_experiences(self, seqs, attention_mask, images, logits_to_keep):
		with torch.no_grad():
			self.onload_or_offload('cpu', None)
			logps = self.batch_compute_logps(self.model_engine, seqs, attention_mask, images, logits_to_keep)
			# logps = self.compute_logps(self.model_engine, seqs, attention_mask, position_ids, pixel_values, image_grid_thw, logits_to_keep)
			if self.ref_model_engine is not None:
				self.onload_or_offload('cuda', 'cpu')
				ref_logps = self.batch_compute_logps(self.ref_model_engine, seqs, attention_mask, images, logits_to_keep)
				self.onload_or_offload('cpu', 'cuda')
			else:
				ref_logps = logps
		self.model_engine.train()
		return logps, ref_logps

	def get_attention_mask(self, seqs, prompt_len):
		attention_mask, position_ids, finished_list = [], [], []
		for seq in seqs.tolist():
			start, end, finished = self.get_start_and_end_index(seq, prompt_len)
			mask = [0] * start + [1] * (end - start + 1) + [0] * (len(seq) - end - 1)
			pos_ids = [0] * start + list(range(len(seq) - start))
			attention_mask.append(mask)
			position_ids.append(pos_ids)
			finished_list.append(finished)
			# if self.args.global_rank == 0:
			# 	print('*' * 10)
			# 	print(start, end, len(seq), len(mask))
			# 	print(seq[start:end+1])
			# 	print(mask[start:end+1])
			# 	print(pos_ids[start:end+1])
			# 	print(finished)
		attention_mask = torch.LongTensor(attention_mask).to(seqs.device)
		# position_ids = torch.LongTensor(position_ids).to(seqs.device)
		assert attention_mask.shape == seqs.shape
		return attention_mask, finished_list

	def get_start_and_end_index(self, token_ids, prompt_len):
		start, end = 0, len(token_ids) - 1
		for idx, token_id in enumerate(token_ids):
			if token_id != self.tokenizer.pad_token_id:
				start = idx
				break
		for idx, token_id in enumerate(token_ids):
			if token_id == self.tokenizer.eos_token_id and idx > prompt_len:
				end = idx
				break
		finished = False
		if token_ids[-1] == self.tokenizer.pad_token_id:
			finished = True
		return start, end, finished

	def get_samples(self):
		start = 0
		all_samples = []
		n = self.args.per_device_rollout_batch_size * self.args.n_gpus
		while start + n < len(self.train_data):
			all_samples.append(self.train_data[start:start+n])
			start += n
		return all_samples

	def get_prompts_and_labels(self, samples, extra_keys=None):
		all_prompts, all_labels = [], []
		if extra_keys:
			extra_info = {key: [] for key in extra_keys}
		else:
			extra_info = None
		for sample in samples:
			messages = [
				{"role": "system", "content": self.system_prompt},
				{"role": "user", "content": self.prompt_prefix + sample['query']}
			]
			text = self.tokenizer.apply_chat_template(
				messages,
				tokenize=False,
				add_generation_prompt=True
			)
			all_prompts.append(text)
			all_labels.append(sample['answer'])
			if extra_keys:
				for key in extra_keys:
					extra_info[key].append(sample[key])
		if extra_keys:
			return all_prompts, all_labels, extra_info
		else:
			return all_prompts, all_labels

	def eval(self):
		if self.args.global_rank == 0:
			eval_results = self.eval_model()
			print(eval_results)

	def train(self):
		global_step, step = 0, 0
		# if self.args.global_rank == 0:
		# 	self.reload_vllm()
		# 	eval_results = self.eval_model()
		# 	self.writer.add_scalars('Eval/acc', eval_results, global_step)
		# torch.distributed.barrier()
		for _ in range(self.args.epochs):
			random.shuffle(self.train_data)
			all_samples = self.get_samples()
			for samples in all_samples:
				logits_to_keep = self.playout(samples)
				data_stats = self.data_buffer.get_stat()

				for k, v in data_stats.items():
					if 'max_len' in k:
						data_stats[k] = get_all_reduce_max(v.to(self.args.device)).item()
					elif 'min_len' in k:
						data_stats[k] = get_all_reduce_min(v.to(self.args.device)).item()
					else:
						data_stats[k] = get_all_reduce_mean(v.to(self.args.device)).item()
				if self.args.global_rank == 0:
					self.writer.add_scalars('Buffer/reward', {k: v for k, v in data_stats.items() if '_reward' in k}, global_step)
					self.writer.add_scalars('Buffer/prompt_length', 
						{k: v for k, v in data_stats.items() if 'prompt' in k}, 
						global_step)
					self.writer.add_scalars('Buffer/response_length', 
						{k: v for k, v in data_stats.items() if 'response' in k}, 
						global_step)
					left_keys = []
					for k in data_stats:
						if '_reward' in k or 'prompt' in k or 'response' in k:
							continue
						left_keys.append(k)
					self.writer.add_scalars('Buffer/group_stat', 
						{k: v for k, v in data_stats.items() if k in left_keys}, 
						global_step)
				torch.distributed.barrier()
				if True:
					for batch in self.data_buffer.get_all_batches_v2(self.args.per_device_train_batch_size):
						# if self.args.global_rank == 0:
						# 	for t in batch:
						# 		print(t.shape)
						tensor_batch = tuple(t.to(self.args.device) for t in batch['tensor_batch'])
						logps = self.compute_logps(self.model_engine, tensor_batch[0], tensor_batch[1], batch['non_tensor_batch']['images'], logits_to_keep)
						if self.args.task_type == 'agent_math':
							loss_mask = tensor_batch[5][:, -logits_to_keep:]
							# advs = tensor_batch[4][:, -logits_to_keep:]
							advs = tensor_batch[4]
						else:
							loss_mask = tensor_batch[1][:, -logits_to_keep:]
							advs = tensor_batch[4]
						if self.args.algo == 'gspo':
							losses = caculate_gspo_loss(self.args, logps, 
								tensor_batch[2], tensor_batch[3], advs, loss_mask)
						elif self.args.algo == 'kl_conv':
							losses = caculate_grpo_loss_kl_cov(self.args, logps, 
								tensor_batch[2], tensor_batch[3], advs, loss_mask)
						else:
							losses = caculate_grpo_loss(self.args, logps, 
								tensor_batch[2], tensor_batch[3], advs, loss_mask)
						# optimizer.zero_grad()
						self.model_engine.backward(losses['loss'])
						# torch.nn.utils.clip_grad_norm_(self.model_engine.parameters(), self.args.clip_grad_norm)
						self.model_engine.step()
						step += 1
						for k, v in losses.items():
							losses[k] = get_all_reduce_mean(v).item()
						if step % self.args.gradient_accumulation_steps == 0:
							global_step += 1
							if self.args.global_rank == 0:
								self.writer.add_scalars('Train/loss', 
									{k: v for k, v in losses.items() if 'loss' in k}, global_step)
								self.writer.add_scalars('Train/ratio', 
									{k: v for k, v in losses.items() if 'ratio' in k}, global_step)
								self.writer.add_scalar('Train/lr', self.model_engine.optimizer.optimizer.param_groups[0]['lr'], global_step)
								print('global_step: %s, loss: %.3f' % (global_step, losses['loss']))
						
						if global_step % args.save_steps == 0 and global_step > 0 and step % self.args.gradient_accumulation_steps == 0:
							print('saving model: %s' % global_step)
							self.save_model(global_step)

						if global_step % args.eval_steps == 0 and global_step > 0 and step % self.args.gradient_accumulation_steps == 0:
							if self.args.global_rank == 0:
								self.reload_vllm()
								print('eval model in step %s' % global_step)
								eval_results = self.eval_model()
								self.writer.add_scalars('Eval/acc', eval_results, global_step)
								self.onload_or_offload('cpu', None)
							torch.distributed.barrier()
							
				self.data_buffer.reset()
				torch.cuda.empty_cache()
				torch.distributed.barrier()
		self.save_model(global_step)
		print('saving final model: %s' % global_step)
		# print('eval final model: %s' % global_step)
		# eval_acc = self.eval_model()
		# self.writer.add_scalar('Eval/acc', eval_acc, global_step)

	def save_model(self, global_step):
		save_path = os.path.join(self.args.output_dir, 'checkpoint_%s' % global_step)
		if args.zero_stage == 3:
			save_zero_three_model(self.model_engine, self.args, save_path)
			if args.global_rank == 0:
				self.tokenizer.save_pretrained(save_path)
				os.system('cp %s/config.json %s' % (self.args.model_path, save_path))
				os.system('cp %s/tokenizer_config.json %s' % (self.args.model_path, save_path))
		else:
			if self.args.global_rank == 0:
				os.makedirs(save_path, exist_ok=True)
				# to vllm format
				if self.args.task_type in ['vl_math']:
					sd = self.model_engine.module.state_dict() if hasattr(self.model_engine, 'module') else self.model_engine.state_dict()
					sd = {k.replace('model.visual', 'visual').replace('model.language_model', 'language_model.model'): v for k, v in sd.items()}
					# torch.save(sd, os.path.join(self.args.output_dir, 'checkpoint_%s' % global_step, 'pytorch_model.bin'))
					save_file(sd, os.path.join(save_path, 'model.safetensors'))
				else:
					self.model_engine.save_pretrained(
						save_path,
						safe_serialization=True,
						max_shard_size="5GB",
					)
				self.tokenizer.save_pretrained(save_path)
				os.system('cp %s/config.json %s' % (self.args.model_path, save_path))
				os.system('cp %s/tokenizer_config.json %s' % (self.args.model_path, save_path))
				if self.args.task_type in ['vl_math']:
					os.system('cp %s/preprocessor_config.json %s' % (self.args.model_path, save_path))

class TextTrainer(Trainer):
	def reload_vllm(self):
		self.onload_or_offload('cpu', None)
		if self.args.global_rank == 0:
			torch.cuda.empty_cache()
			sd = self.model_engine.module.state_dict() if hasattr(self.model_engine, 'module') else self.model_engine.state_dict()
			self.llm.wake_up()
			llm_model = self.llm.llm_engine.model_executor.driver_worker.worker.model_runner.model
			# print('****start-of-sd****')
			# for k, v in sd.items():
			# 	print(k, v[0])
			# print('****end-of-sd****')
			llm_model.load_weights((name, param) for name, param in sd.items())
			# for vllm v1 engine
			# self.llm.collective_rpc("report_device_id", args=tuple())
			# ipc_handles = {}
			# ipc_handles.update(get_weight_ipc_handles(sd))
			# self.llm.collective_rpc("update_weights_from_ipc_handles", args=(ipc_handles, ))
			# self.llm.collective_rpc("check_weights_changed", args=(ipc_handles, ))
			del sd

	def eval_model(self):
		all_prompts, all_labels = self.get_prompts_and_labels(self.eval_data)
		print('-----prompt-----', self.args.global_rank, all_prompts[0], all_labels[0])
		outputs = self.llm.generate(all_prompts, self.eval_sampling_params)
		self.llm.sleep(level=1)
		torch.cuda.empty_cache()
		responses = [output.outputs[0].text for output in outputs]
		assert len(responses) == len(all_labels)
		all_correct = []
		if self.args.task_type == 'logic':
			all_np = []
			empty_answer_count = 0
			for response, label in zip(responses, all_labels):
				all_np.append(len(json.loads(label)))
				answer = extract_answer_from_model_response(response, 'logic')
				if answer == '':
					empty_answer_count += 1
					print('---empty_answer---')
					print(response)
				if is_equal(answer, label, 'logic'):
					all_correct.append(1)
				else:
					all_correct.append(0)
			assert len(all_np) == len(all_correct) == len(all_labels)
			counter = defaultdict(list)
			for np, cor in zip(all_np, all_correct):
				counter[np].append(cor)
			acc = {'%sp_acc' % k: sum(v) / len(v) for k, v in counter.items()}
			acc['final_acc'] = sum(all_correct) / len(all_correct)
			acc['empty_answer_ratio'] = empty_answer_count / len(all_correct)
			return acc
		else:
			for response, label in zip(responses, all_labels):
				if is_equal(response, label):
					all_correct.append(1)
				else:
					all_correct.append(0)
			assert len(all_correct) == len(all_labels)
			return {'final_acc': sum(all_correct) / len(all_correct)}

	def playout(self, samples):
		self.reload_vllm()
		per_device_n = len(samples) // self.args.n_gpus
		assert per_device_n == self.args.per_device_rollout_batch_size
		if self.args.global_rank == 0:
			all_prompts, all_labels = self.get_prompts_and_labels(samples)
			print('****prompt*****', self.args.global_rank, all_prompts[0])
			seqs = self.generate_sequences(all_prompts)
			self.llm.sleep(level=1)
			torch.cuda.empty_cache()
		else:
			all_prompts = [None] * len(samples)
			all_labels = [None] * len(samples)
			seqs = [None] * len(samples) * self.args.group_size
		all_prompts = broadcast_object_list(all_prompts)
		seqs = broadcast_object_list(seqs)
		all_labels = broadcast_object_list(all_labels)
		model_inputs = self.tokenizer(all_prompts, padding=True, truncation=True, return_tensors="pt")
		local_input_ids = model_inputs['input_ids'][self.args.local_rank*per_device_n:(self.args.local_rank+1)*per_device_n].repeat_interleave(self.args.group_size, dim=0)
		local_labels = all_labels[self.args.local_rank*per_device_n:(self.args.local_rank+1)*per_device_n]
		local_seqs = seqs[self.args.local_rank*per_device_n*self.args.group_size:(self.args.local_rank+1)*per_device_n*self.args.group_size]
		prompt_len = model_inputs['input_ids'].shape[1]
		logits_to_keep = len(seqs[0])
		local_whole = torch.cat([local_input_ids, torch.LongTensor(local_seqs)], dim=-1).to(self.args.device)
		assert prompt_len + logits_to_keep == local_whole.shape[1], print(prompt_len, logits_to_keep, local_whole.shape[1])
		if self.args.global_rank == 0:
			print('playout...', local_whole.shape, prompt_len, logits_to_keep)
		attention_mask, finished_list = self.get_attention_mask(local_whole, prompt_len)
		logps, ref_logps = self.generate_experiences(local_whole, attention_mask, None, logits_to_keep)
		try:
			response = self.tokenizer.batch_decode(local_whole[:, -logits_to_keep:], skip_special_tokens=True)
		except:
			print('-----batch decode error-----')
			print(local_whole[:, -logits_to_keep:].tolist())
			response = ['fake answer'] * local_whole.shape[0]
		for jdx in range(len(local_whole) // self.args.group_size):
			start = jdx * self.args.group_size
			end = start + self.args.group_size
			self.data_buffer.add_for_text(local_whole[start:end].tolist(), attention_mask[start:end].tolist(), logps[start:end].tolist(), ref_logps[start:end].tolist(), 
				response[start:end], local_labels[jdx], logits_to_keep, finished_list)
		return logits_to_keep

class VLTrainer(Trainer):
	"""docstring for VLTrainer"""
	def apply_chat_template(self, system, user):
		return "<|im_start|>system\n%s<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>%s<|im_end|>\n<|im_start|>assistant\n" % (system, user)

	def get_prompts_and_labels_for_vl(self, samples):
		all_prompts, all_labels = [], []
		for sample in samples:
			all_prompts.append({
				'prompt': self.apply_chat_template(self.system_prompt, self.prompt_prefix + sample['query']), 
				'multi_modal_data': process_multi_modal_data({"images": [sample['image']]}, 
					min_pixels=28*28*self.args.min_vl_tokens, 
					max_pixels=28*28*self.args.max_vl_tokens, 
					video_fps=1)
				})
		all_labels.append(sample['answer'])
		return all_prompts, all_labels

	def prepare_mm_inputs(self, images):
		if not isinstance(images, list):
			images = images.tolist()
		multi_modal_inputs = dict(self.processor.image_processor(images=images, return_tensors="pt"))
		multi_modal_inputs = {k: v.to(self.args.device) for k, v in multi_modal_inputs.items()}
		return multi_modal_inputs['pixel_values'], multi_modal_inputs['image_grid_thw']

	def reload_vllm(self):
		self.onload_or_offload('cpu', None)
		if self.args.global_rank == 0:
			torch.cuda.empty_cache()
			sd = self.model_engine.module.state_dict() if hasattr(self.model_engine, 'module') else self.model_engine.state_dict()
			self.llm.wake_up()
			# llm_model = self.llm.llm_engine.model_executor.driver_worker.worker.model_runner.model
			# print('****start-of-sd****')
			# for k, v in sd.items():
			# 	print(k, v[0])
			# print('****end-of-sd****')
			# llm_model.load_weights((name.replace('model.', ''), param) for name, param in sd.items())
			# for vllm v1 engine
			self.llm.collective_rpc("report_device_id", args=tuple())
			ipc_handles = {}
			ipc_handles.update(get_weight_ipc_handles(sd))
			self.llm.collective_rpc("update_weights_from_ipc_handles", args=(ipc_handles, ))
			# self.llm.collective_rpc("check_weights_changed", args=(ipc_handles, ))
			del sd

	def eval_model(self):
		all_prompts, all_labels = self.get_prompts_and_labels_for_vl(self.eval_data)
		print('-----prompt-----', self.args.global_rank, all_prompts[0], all_labels[0])
		outputs = self.llm.generate(all_prompts, self.eval_sampling_params)
		self.llm.sleep(level=1)
		torch.cuda.empty_cache()
		responses = [output.outputs[0].text for output in outputs]
		assert len(responses) == len(all_labels)
		all_correct, empty_answer_count = [], 0
		for response, label in zip(responses, all_labels):
			answer = extract_answer_from_model_response(response)
			if answer == '':
				empty_answer_count += 1
				print('---empty_answer---')
				print(response)
			if is_equal(answer, label):
				all_correct.append(1)
			else:
				all_correct.append(0)
		assert len(all_correct) == len(all_labels)
		return {'final_acc': sum(all_correct) / len(all_correct), 'empty_answer_ratio': empty_answer_count / len(all_correct)}

	def playout(self, samples):
		self.reload_vllm()
		per_device_n = len(samples) // self.args.n_gpus
		assert per_device_n == self.args.per_device_rollout_batch_size
		if self.args.global_rank == 0:
			all_prompts, all_labels = get_prompts_and_labels_for_vl(samples)
			print('****prompt*****', self.args.global_rank, all_prompts[0])
			seqs = self.generate_sequences(all_prompts)
			self.llm.sleep(level=1)
			torch.cuda.empty_cache()
		else:
			all_prompts = [None] * len(samples)
			all_labels = [None] * len(samples)
			seqs = [None] * len(samples) * self.args.group_size
		all_prompts = broadcast_object_list(all_prompts)
		seqs = broadcast_object_list(seqs)
		all_labels = broadcast_object_list(all_labels)
		all_images = [p['multi_modal_data']['image'][0] for p in all_prompts]
		local_images = np.array(all_images[self.args.local_rank*per_device_n:(self.args.local_rank+1)*per_device_n], dtype=object)
		local_images = np.repeat(local_images, self.args.group_size, axis=0)
		model_inputs = self.processor(all_images, [p['prompt'] for p in all_prompts], padding=True, truncation=True, return_tensors="pt")
		local_input_ids = model_inputs['input_ids'][self.args.local_rank*per_device_n:(self.args.local_rank+1)*per_device_n].repeat_interleave(self.args.group_size, dim=0)
		local_labels = all_labels[self.args.local_rank*per_device_n:(self.args.local_rank+1)*per_device_n]
		local_seqs = seqs[self.args.local_rank*per_device_n*self.args.group_size:(self.args.local_rank+1)*per_device_n*self.args.group_size]
		prompt_len = model_inputs['input_ids'].shape[1]
		logits_to_keep = len(seqs[0])
		local_whole = torch.cat([local_input_ids, torch.LongTensor(local_seqs)], dim=-1).to(self.args.device)
		assert prompt_len + logits_to_keep == local_whole.shape[1], print(prompt_len, logits_to_keep, local_whole.shape[1])
		attention_mask, finished_list = self.get_attention_mask(local_whole, prompt_len)
		logps, ref_logps = self.generate_experiences(local_whole, attention_mask, local_images, logits_to_keep)
		response = self.processor.tokenizer.batch_decode(local_whole[:, -logits_to_keep:], skip_special_tokens=True)
		for jdx in range(len(local_whole) // self.args.group_size):
			start = jdx * self.args.group_size
			end = start + self.args.group_size
			self.data_buffer.add_for_mm(local_whole[start:end].tolist(), attention_mask[start:end].tolist(),
				local_images[start:end].tolist(), logps[start:end].tolist(), ref_logps[start:end].tolist(), 
				response[start:end], local_labels[jdx], logits_to_keep, finished_list)
		return logits_to_keep

class AgentTrainer(TextTrainer):
	"""docstring for AgentTrainer"""
	def get_sampling_params_list(self, max_tokens_list):
		sp_list = [copy.copy(self.sampling_params_agent_stop) for _ in range(len(max_tokens_list))]
		for sp, max_tokens in zip(sp_list, max_tokens_list):
			assert max_tokens > 0
			sp.max_tokens = max_tokens
		return sp_list

	def get_adv_mask(self, response_ids):
		code_start_ids = self.tokenizer('```python')['input_ids']
		start_index = None
		for idx in range(len(response_ids)-len(code_start_ids)+1):
			if response_ids[idx:idx+len(code_start_ids)] == code_start_ids:
				start_index = idx
		if start_index:
			adv_mask = [1] * start_index + [0] * (len(response_ids) - start_index)
		else:
			adv_mask = [1] * len(response_ids)
		return adv_mask

	def generate_sequences_for_agent(self, batch_prompts, repeat_time):
		if repeat_time > 1:
			batch_prompts = repeat_dict(batch_prompts, repeat_time)
		# to id
		inputs = self.tokenizer(batch_prompts, padding=False)
		prompt_lens = [len(x) for x in inputs['input_ids']]
		batch_prompt_ids = [{'prompt_token_ids': x} for x in inputs['input_ids']]
		batch_attention_mask = [{'attention_mask': copy.copy(x)} for x in inputs['attention_mask']]
		batch_loss_mask = [{'loss_mask': copy.copy(x)} for x in inputs['attention_mask']]
		tool_call_counter = [0] * len(batch_prompt_ids)
		token_budget = [self.args.max_new_tokens] * len(batch_prompt_ids)
		tool_call_history = [''] * len(batch_prompt_ids)
		for ridx in range(self.args.max_tool_calls):
			# stop reason for each round: stop_str/stop_token_ids/max_len
			stop_for_tool_call, index_mapping, max_tokens_list = [], {}, []
			for idx in range(len(batch_prompt_ids)):
				if tool_call_counter[idx] < self.args.max_tool_calls and token_budget[idx] > 0:
					stop_for_tool_call.append(batch_prompt_ids[idx])
					max_tokens_list.append(token_budget[idx])
					index_mapping[len(index_mapping)] = idx
			if len(stop_for_tool_call) > 0:
				sp_list = self.get_sampling_params_list(max_tokens_list)
				# print([sp.max_tokens for sp in sp_list])
				stop_outputs = self.llm.generate(stop_for_tool_call, sp_list)
				for idx, output in enumerate(stop_outputs):
					response = output.outputs[0].text
					finish_reason = output.outputs[0].finish_reason
					response_ids = self.tokenizer(response)['input_ids']
					if token_budget[index_mapping[idx]] < len(response_ids):
						print('Warning: Token Budget Exceed!, Budget: %s, Generated: %s' % (token_budget[index_mapping[idx]], len(response_ids)))
						response_ids = response_ids[:token_budget[index_mapping[idx]]]
					batch_prompt_ids[index_mapping[idx]]['prompt_token_ids'] += response_ids
					batch_attention_mask[index_mapping[idx]]['attention_mask'] += [1] * len(response_ids)
					batch_loss_mask[index_mapping[idx]]['loss_mask'] += [1] * len(response_ids)
					token_budget[index_mapping[idx]] -= len(response_ids)
					if finish_reason == 'stop':
						# stop for stop_str
						if response.endswith(self.stop_str):
							tool_call_counter[index_mapping[idx]] += 1
							tool_call, tool_call_result = call_python_server(tool_call_history[index_mapping[idx]], response, self.start_str, self.stop_str)
							tool_call_result_ids = self.tokenizer(tool_call_result)['input_ids'][:self.args.max_tool_response_len]
							if token_budget[index_mapping[idx]] < len(tool_call_result_ids):
								tool_call_result_ids = tool_call_result_ids[:token_budget[index_mapping[idx]]]
							tool_call_history[index_mapping[idx]] += tool_call + '\n'
							batch_prompt_ids[index_mapping[idx]]['prompt_token_ids'] += tool_call_result_ids
							batch_attention_mask[index_mapping[idx]]['attention_mask'] += [1] * len(tool_call_result_ids)
							batch_loss_mask[index_mapping[idx]]['loss_mask'] += [0] * len(tool_call_result_ids)
							assert token_budget[index_mapping[idx]] >= len(tool_call_result_ids)
							token_budget[index_mapping[idx]] -= len(tool_call_result_ids)
						# stop for eos
						else:
							tool_call_counter[index_mapping[idx]] = 1000
					else:
						assert finish_reason == 'length'
			# all stop
			if all([x > self.args.max_tool_calls for x in tool_call_counter]):
				break
			if all([x == 0 for x in token_budget]):
				break
			print('-----------------end-of-stop--------------')
			# take care of tool call
		# lef pad to max_prompt_len
		max_prompt_len = max(prompt_lens)
		for idx in range(len(batch_prompt_ids)):
			if prompt_lens[idx] < max_prompt_len:
				batch_prompt_ids[idx]['prompt_token_ids'] = [self.tokenizer.pad_token_id] * (max_prompt_len - prompt_lens[idx]) + batch_prompt_ids[idx]['prompt_token_ids']
				batch_attention_mask[idx]['attention_mask'] = [0] * (max_prompt_len - prompt_lens[idx]) + batch_attention_mask[idx]['attention_mask']
				batch_loss_mask[idx]['loss_mask'] = [0] * (max_prompt_len - prompt_lens[idx]) + batch_loss_mask[idx]['loss_mask']
		# pad to max_len
		max_len = max(len(prompt['prompt_token_ids']) for prompt in batch_prompt_ids)
		seqs, attention_masks, loss_masks = [], [], []
		for prompt in batch_prompt_ids:
			seqs.append(prompt['prompt_token_ids'] + [self.tokenizer.pad_token_id] * (max_len - len(prompt['prompt_token_ids'])))
		for mask in batch_attention_mask:
			attention_masks.append(mask['attention_mask'] + [0] * (max_len - len(mask['attention_mask'])))
		for mask in batch_loss_mask:
			loss_masks.append(mask['loss_mask'] + [0] * (max_len - len(mask['loss_mask'])))
		return seqs, attention_masks, loss_masks, [max_prompt_len]

	def estimate_pass_at_k(self, n, c, k):
		"""
		Calculates 1 - comb(n - c, k) / comb(n, k).
		"""
		if n - c < k:
			return 1.0
		return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

	def eval_model(self):
		all_prompts, all_labels, extra_info = self.get_prompts_and_labels(self.eval_data, extra_keys=['data_source'])
		all_data_sources = extra_info['data_source']
		print('-----prompt-----', self.args.global_rank, all_prompts[0], all_labels[0])
		seqs, _, _, _, _ = self.generate_sequences_for_agent(all_prompts, self.args.unbiased_pass_k_n)
		self.llm.sleep(level=1)
		torch.cuda.empty_cache()
		responses = self.tokenizer.batch_decode(seqs, skip_special_tokens=True)
		assert len(responses) == len(all_labels) * self.args.unbiased_pass_k_n
		pass_at_k = defaultdict(list)
		for k in [int(i) for i in self.args.pass_k.split(',') if i != '']:
			for i in range(len(all_prompts)):
				c = 0
				data_source = all_data_sources[i]
				label = all_labels[i]
				for response in responses[i*self.args.unbiased_pass_k_n:(i+1)*self.args.unbiased_pass_k_n]:
					answer = extract_answer_from_model_response(response, self.args.task_type)
					if is_equal(answer, label, self.args.task_type):
						c += 1
				pass_at_k[data_source + '_pass@' + str(k)].append(self.estimate_pass_at_k(self.args.unbiased_pass_k_n, c, k))
		return {key: sum(value) / len(value) for key, value in pass_at_k.items()}

	def playout(self, samples):
		self.reload_vllm()
		per_device_n = len(samples) // self.args.n_gpus
		assert per_device_n == self.args.per_device_rollout_batch_size
		if self.args.global_rank == 0:
			all_prompts, all_labels = self.get_prompts_and_labels(samples)
			print('****prompt*****', self.args.global_rank, all_prompts[0])
			seqs, all_attention_mask, all_loss_mask, prompt_len = self.generate_sequences_for_agent(all_prompts, self.args.group_size)
			self.llm.sleep(level=1)
			torch.cuda.empty_cache()
		else:
			all_prompts = [None] * len(samples)
			all_labels = [None] * len(samples)
			seqs = [None] * len(samples) * self.args.group_size
			all_attention_mask = [None] * len(samples) * self.args.group_size
			all_loss_mask = [None] * len(samples) * self.args.group_size
			prompt_len = [None]
		all_prompts = broadcast_object_list(all_prompts)
		all_labels = broadcast_object_list(all_labels)
		seqs = broadcast_object_list(seqs)
		all_attention_mask = broadcast_object_list(all_attention_mask)
		all_loss_mask = broadcast_object_list(all_loss_mask)
		prompt_len = broadcast_object_list(prompt_len)
		local_labels = all_labels[self.args.local_rank*per_device_n:(self.args.local_rank+1)*per_device_n]
		local_seqs = seqs[self.args.local_rank*per_device_n*self.args.group_size:(self.args.local_rank+1)*per_device_n*self.args.group_size]
		local_seqs = torch.LongTensor(local_seqs).to(self.args.device)
		local_attention_mask = all_attention_mask[self.args.local_rank*per_device_n*self.args.group_size:(self.args.local_rank+1)*per_device_n*self.args.group_size]
		local_attention_mask = torch.LongTensor(local_attention_mask).to(self.args.device)
		local_loss_mask = all_loss_mask[self.args.local_rank*per_device_n*self.args.group_size:(self.args.local_rank+1)*per_device_n*self.args.group_size]
		logits_to_keep = len(seqs[0]) - prompt_len[0]
		if self.args.global_rank == 0:
			print('playout...', local_seqs.shape, prompt_len, logits_to_keep)
		logps, ref_logps = self.generate_experiences(local_seqs, local_attention_mask, None, logits_to_keep)
		try:
			response = self.tokenizer.batch_decode(local_seqs[:, -logits_to_keep:], skip_special_tokens=True)
		except:
			print('-----batch decode error-----')
			print(local_seqs[:, -logits_to_keep:].tolist())
			response = ['fake answer'] * local_seqs.shape[0]
		for jdx in range(len(local_seqs) // self.args.group_size):
			start = jdx * self.args.group_size
			end = start + self.args.group_size
			self.data_buffer.add_for_text(local_seqs[start:end].tolist(), local_attention_mask[start:end].tolist(), 
				logps[start:end].tolist(), ref_logps[start:end].tolist(), 
				response[start:end], local_labels[jdx], logits_to_keep, [1] * self.args.group_size, 
				local_loss_mask[start:end])
		return logits_to_keep

if __name__ == '__main__':
	args = get_grpo_args()
	if args.task_type in ['vl_math']:
		trainer = VLTrainer(args)
	elif args.task_type in ['logic', 'math']:
		trainer = TextTrainer(args)
	elif args.task_type in ['agent_math']:
		trainer = AgentTrainer(args)
	trainer.train()