import json
import torch
import random
import numpy as np
from torch.distributions import Categorical
import copy
import re
from data_utils import extract_answer_from_model_response, is_equal
import pandas as pd
import os
from transformers import StoppingCriteria
# from mathruler.grader import extract_boxed_content

class GRPOBuffer(object):
	"""docstring for GRPOBuffer"""
	def __init__(self, tokenizer, args):
		super(GRPOBuffer, self).__init__()
		self.tokenizer = tokenizer
		self.args = args
		self.reset()

	def add_for_mm(self, group_token_ids, group_attention_mask, group_images, 
			group_old_logps, group_ref_logps, group_response, gold_label, logits_to_keep, finished_list):
		advs, rewards, answers = self.cal_advs(group_response, gold_label)
		# if self.args.completition_keep_ratio > 0:
		# 	sorted_abs_advs = sorted([abs(adv) for adv in advs])
		# 	keep_point = sorted_abs_advs[-int(len(advs) * self.args.completition_keep_ratio)]
		# else:
		# 	keep_point = 0
		# cnt = 0
		# for token_ids, attention_mask, old_logps, ref_logps, adv in zip(group_token_ids, group_attention_mask, group_old_logps, group_ref_logps, advs):
		# 	if abs(adv) >= keep_point and cnt < int(len(advs) * self.args.completition_keep_ratio):
		# 		self.all_token_ids.append(token_ids)
		# 		self.all_attention_mask.append(attention_mask)
		# 		self.all_old_logps.append(old_logps)
		# 		self.all_ref_logps.append(ref_logps)
		# 		self.all_advs.append(adv)
		# 		cnt += 1
		# assert cnt == int(len(advs) * self.args.completition_keep_ratio)
		self.all_token_ids += group_token_ids
		self.all_attention_mask += group_attention_mask
		self.all_images += group_images
		self.all_old_logps += group_old_logps
		self.all_ref_logps += group_ref_logps
		self.all_advs += advs
		for attention_mask in group_attention_mask:
			if sum(attention_mask) > 0:
				self.prompt_len_dist.append(sum(attention_mask[:-logits_to_keep]))
				self.response_len_dist.append(sum(attention_mask[-logits_to_keep:]))
		self.finished_list += finished_list
		if all([r >= 1 for r in rewards]):
			self.all_correct_group_cnt += 1
		if all([r == 0 for r in rewards]):
			self.all_wrong_group_cnt += 1
		self.print_rewards(group_response, gold_label, rewards, answers)
		for answer in answers:
			if answer == '':
				self.empty_answer_count += 1

	def add_for_text(self, group_token_ids, group_attention_mask, group_old_logps, 
			group_ref_logps, group_response, gold_label, logits_to_keep, finished_list):
		advs, rewards, answers = self.cal_advs(group_response, gold_label)
		self.all_token_ids += group_token_ids
		self.all_attention_mask += group_attention_mask
		self.all_old_logps += group_old_logps
		self.all_ref_logps += group_ref_logps
		self.all_advs += advs
		for attention_mask in group_attention_mask:
			if sum(attention_mask) > 0:
				self.prompt_len_dist.append(sum(attention_mask[:-logits_to_keep]))
				self.response_len_dist.append(sum(attention_mask[-logits_to_keep:]))
		self.finished_list += finished_list
		if all([r >= 1 for r in rewards]):
			self.all_correct_group_cnt += 1
		if all([r == 0 for r in rewards]):
			self.all_wrong_group_cnt += 1
		self.print_rewards(group_response, gold_label, rewards, answers)
		for answer in answers:
			if answer == '':
				self.empty_answer_count += 1

	def print_rewards(self, group_response, gold_label, rewards, answers):
		if self.args.debug and self.args.local_rank < 1:
			for r, rw, answer in zip(group_response, rewards, answers):
				print('*' * 20)
				print(r)
				print('gold_label: ', gold_label)
				print('answer: ', answer)
				print('rw:', rw)
				# print(self.all_token_ids)
				# print(self.all_attention_mask)
				# print(self.all_old_logps)

	def reset(self):
		self.all_token_ids = []
		self.all_attention_mask = []
		self.all_position_ids = []
		self.all_images = []
		self.all_old_logps = []
		self.all_ref_logps = []
		self.all_advs = []
		self.correct = 0
		self.total = 0
		self.correct_reward = 0
		self.format_reward = 0
		self.prompt_len_dist = []
		self.response_len_dist = []
		self.finished_list = []
		self.all_correct_group_cnt = 0
		self.all_wrong_group_cnt = 0
		self.empty_answer_count = 0
		self.reward_std = 0

	def get_stat(self):
		pass_at_k = self.correct / self.total if self.total > 0 else 0
		prompt_avg_len = sum(self.prompt_len_dist) / len(self.prompt_len_dist) if len(self.prompt_len_dist) > 0 else 0
		prompt_max_len = max(self.prompt_len_dist) if len(self.prompt_len_dist) > 0 else 0
		prompt_min_len = min(self.prompt_len_dist) if len(self.prompt_len_dist) > 0 else 0
		response_avg_len = sum(self.response_len_dist) / len(self.response_len_dist) if len(self.response_len_dist) > 0 else 0
		response_max_len = max(self.response_len_dist) if len(self.response_len_dist) > 0 else 0
		response_min_len = min(self.response_len_dist) if len(self.response_len_dist) > 0 else 0
		finish_rate = sum(self.finished_list) / len(self.finished_list)
		avg_correct_reward = self.correct_reward / self.total / self.args.group_size if self.total > 0 else 0
		avg_format_reward = self.format_reward / self.total / self.args.group_size if self.total > 0 else 0
		avg_reward_std = self.reward_std / self.total
		all_correct_ratio = self.all_correct_group_cnt / self.total
		all_wrong_ratio = self.all_wrong_group_cnt / self.total
		avg_empty_answer_ratio = self.empty_answer_count / self.total / self.args.group_size
		data_stats = {
			'prompt_avg_len': torch.tensor(prompt_avg_len), 
			'prompt_max_len': torch.tensor(prompt_max_len), 
			'prompt_min_len': torch.tensor(prompt_min_len),
			'response_avg_len': torch.tensor(response_avg_len), 
			'response_max_len': torch.tensor(response_max_len), 
			'response_min_len': torch.tensor(response_min_len),
			'pass_at_k': torch.tensor(pass_at_k),
			'finish_rate': torch.tensor(finish_rate),
			# 'all_correct_ratio': torch.tensor(all_correct_ratio),
			# 'all_wrong_ratio': torch.tensor(all_wrong_ratio),
			'avg_empty_answer_ratio': torch.tensor(avg_empty_answer_ratio),
			'avg_correct_reward': torch.tensor(avg_correct_reward),
			'avg_format_reward': torch.tensor(avg_format_reward),
			'avg_reward_std': torch.tensor(avg_reward_std)
			}
		return data_stats

	# def inter_shuffle(self, data, ids):
	# 	da = []
	# 	for i in ids:
	# 		da += data[i*self.args.group_size:(i+1)*self.args.group_size]
	# 	return da

	# def shuffle(self):
	# 	# inter-group shuffle
	# 	ids = list(range(len(self.all_token_ids) // self.args.group_size))
	# 	random.shuffle(ids)
	# 	self.all_token_ids = self.inter_shuffle(self.all_token_ids, ids)
	# 	self.all_attention_mask = self.inter_shuffle(self.all_attention_mask, ids)
	# 	self.all_old_logps = self.inter_shuffle(self.all_old_logps, ids)
	# 	self.all_ref_logps = self.inter_shuffle(self.all_ref_logps, ids)
	# 	self.all_advs = self.inter_shuffle(self.all_advs, ids)

	def random_shuffle(self, data, ids):
		return [data[i] for i in ids]

	def shuffle(self):
		ids = list(range(len(self.all_token_ids)))
		random.shuffle(ids)
		self.all_token_ids = self.random_shuffle(self.all_token_ids, ids)
		self.all_attention_mask = self.random_shuffle(self.all_attention_mask, ids)
		self.all_old_logps = self.random_shuffle(self.all_old_logps, ids)
		self.all_ref_logps = self.random_shuffle(self.all_ref_logps, ids)
		self.all_advs = self.random_shuffle(self.all_advs, ids)
		if len(self.all_images) > 0:
			self.all_images = self.random_shuffle(self.all_images, ids)

	def get_all_batches(self, batch_size):
		self.shuffle()
		all_batches = []
		# if self.args.local_rank < 1:
		# 	print(len(self.all_token_ids), batch_size)
		# 	print(len(self.all_pixel_values), len(self.all_image_grid_thw))
		# TODO, fix this
		# assert batch_size == self.args.group_size
		assert len(self.all_token_ids) == len(self.all_attention_mask) == len(self.all_old_logps) == len(self.all_ref_logps) == len(self.all_advs)
		if self.args.task_type == 'vl_math':
			for i in range(len(self.all_token_ids) // batch_size):
				all_batches.append({
					'tensor_batch': (
						torch.LongTensor(self.all_token_ids[i*batch_size:i*batch_size+batch_size]), 
						torch.LongTensor(self.all_attention_mask[i*batch_size:i*batch_size+batch_size]), 
						torch.FloatTensor(self.all_old_logps[i*batch_size:i*batch_size+batch_size]),
						torch.FloatTensor(self.all_ref_logps[i*batch_size:i*batch_size+batch_size]),
						torch.FloatTensor(self.all_advs[i*batch_size:i*batch_size+batch_size])
						),
					'non_tensor_batch': {'images': self.all_images[i*batch_size:i*batch_size+batch_size]}
					})
		else:
			for i in range(len(self.all_token_ids) // batch_size):
				all_batches.append((
					torch.LongTensor(self.all_token_ids[i*batch_size:i*batch_size+batch_size]), 
					torch.LongTensor(self.all_attention_mask[i*batch_size:i*batch_size+batch_size]), 
					torch.FloatTensor(self.all_old_logps[i*batch_size:i*batch_size+batch_size]),
					torch.FloatTensor(self.all_ref_logps[i*batch_size:i*batch_size+batch_size]),
					torch.FloatTensor(self.all_advs[i*batch_size:i*batch_size+batch_size])
					))
		return all_batches

	def get_all_batches_v2(self, batch_size):
		self.shuffle()
		all_batches = []
		# if self.args.local_rank < 1:
		# 	print(len(self.all_token_ids), batch_size)
		# 	print(len(self.all_pixel_values), len(self.all_image_grid_thw))
		# TODO, fix this
		# assert batch_size == self.args.group_size
		assert len(self.all_token_ids) == len(self.all_attention_mask) == len(self.all_old_logps) == len(self.all_ref_logps) == len(self.all_advs)
		if self.args.task_type == 'vl_math':
			for i in range(len(self.all_token_ids) // batch_size):
				all_batches.append({
					'tensor_batch': (
						torch.LongTensor(self.all_token_ids[i*batch_size:i*batch_size+batch_size]), 
						torch.LongTensor(self.all_attention_mask[i*batch_size:i*batch_size+batch_size]), 
						torch.FloatTensor(self.all_old_logps[i*batch_size:i*batch_size+batch_size]),
						torch.FloatTensor(self.all_ref_logps[i*batch_size:i*batch_size+batch_size]),
						torch.FloatTensor(self.all_advs[i*batch_size:i*batch_size+batch_size])
						),
					'non_tensor_batch': {'images': self.all_images[i*batch_size:i*batch_size+batch_size]}
					})
		else:
			for i in range(len(self.all_token_ids) // batch_size):
				all_batches.append({
					'tensor_batch':(
						torch.LongTensor(self.all_token_ids[i*batch_size:i*batch_size+batch_size]), 
						torch.LongTensor(self.all_attention_mask[i*batch_size:i*batch_size+batch_size]), 
						torch.FloatTensor(self.all_old_logps[i*batch_size:i*batch_size+batch_size]),
						torch.FloatTensor(self.all_ref_logps[i*batch_size:i*batch_size+batch_size]),
						torch.FloatTensor(self.all_advs[i*batch_size:i*batch_size+batch_size])
						),
					'non_tensor_batch': {'images': None}
					})
		return all_batches

	def cal_rewards_for_logic(self, response, gold_label):
		if is_equal(response, gold_label, 'logic'):
			correct, reward = True, 1
		else:
			correct, reward = False, 0
		format_reward = self.get_format_reward(response)
		self.correct_reward += reward
		self.format_reward += format_reward
		return reward + format_reward, correct

	def cal_rewards_for_math(self, response, gold_label):
		if is_equal(response, gold_label):
			correct = True
			reward = 1
		else:
			correct, reward = False, 0
		format_reward = self.get_format_reward(response)
		self.correct_reward += reward
		self.format_reward += format_reward
		return reward + format_reward, correct

	def get_format_reward(self, response):
		format_reward = 0.0
		if '<think>' in response and '</think>' in response and '<answer>' in response and '</answer>' in response:
			if response.count('<think>') == 1 and response.count('</think>') == 1 and response.count('<answer>') == 1 and response.count('</answer>') == 1:
				think_start = response.index('<think>')
				think_end = response.index('</think>')
				answer_start = response.index('<answer>')
				answer_end = response.index('</answer>')
				if think_start < think_end < answer_start < answer_end:
					format_reward = 0.1
		return format_reward

	def cal_advs(self, group_response, gold_label):
		rewards, all_correct, all_answers = [], [], []
		for r in group_response:
			if self.args.task_type == 'logic':
				reward, correct = self.cal_rewards_for_logic(r, gold_label)
			elif self.args.task_type == 'vl_math':
				reward, correct = self.cal_rewards_for_math(r, gold_label)
			elif self.args.task_type == 'math':
				reward, correct = self.cal_rewards_for_math(r, gold_label)
			rewards.append(reward)
			all_correct.append(correct)
			if self.args.task_type == 'logic':
				all_answers.append(extract_answer_from_model_response(r, 'logic'))
			else:
				all_answers.append(extract_answer_from_model_response(r))
		rewards = torch.tensor(rewards)
		std = torch.std(rewards)
		# len_rewards = torch.tensor(self.get_len_reward(group_response, all_correct))
		if self.args.algo == 'rloo':
			advs = []
			for r in rewards:
				baseline = (sum(rewards) - r) / (len(rewards) - 1)
				advs.append(r - baseline)	
		else:
			mean = torch.mean(rewards)
			if self.args.algo == 'dr_grpo':
				advs = rewards - mean
			elif self.args.algo == 'stable_reinforce':
				advs = (rewards - mean) / (std + 1e-6)
				advs[advs.abs() > 3] = 0
			else:
				advs = (rewards - mean) / (std + 1e-6)
		# print(rewards)
		self.total += 1
		if any(all_correct):
			self.correct += 1
		self.reward_std += std.item()
		return advs, rewards, all_answers

def print_rank0(args, x):
	if args.local_rank < 1:
		print(x)

def get_ratio_stat(args, ratio, loss_mask):
	lower = ratio < (1 - args.clip_lower)
	higher = ratio > (1 + args.clip_higher)
	clipped = (lower * loss_mask).sum() + (higher * loss_mask).sum()
	return clipped / (loss_mask.sum() + 1e-6)

def caculate_grpo_loss(args, logps, old_logps, ref_logps, advs, loss_mask):
	advs = advs.unsqueeze(-1)
	if args.algo == 'stable_reinforce':
		logp_diff = torch.clamp(logps - old_logps, np.log(1e-3), np.log(1e3))
	else:
		logp_diff = torch.clamp(logps - old_logps, -20, 20)
	ratio = torch.exp(logp_diff)
	clipped_ratio = torch.exp(torch.clamp(logp_diff, np.log(1 - args.clip_lower), np.log(1 + args.clip_higher)))
	entropy_loss = - (torch.exp(logps) * logps * loss_mask).sum() / (loss_mask.sum() + 1e-6)
	if args.use_dual_clip:
		pg_loss = advs * ratio
		pg_loss2 = advs * clipped_ratio
		pg_loss3 = advs * args.clip_ratio_dual

		clipped_pg_loss_higer = - torch.min(pg_loss, pg_loss2)
		clipped_pg_loss_lower = - torch.max(clipped_pg_loss_higer, pg_loss3)
		policy_loss = torch.where(advs < 0, clipped_pg_loss_lower, clipped_pg_loss_higer)
	else:
		policy_loss = - torch.min(advs * ratio, advs * clipped_ratio)
	if args.algo == 'dr_grpo':
		policy_loss = (policy_loss * loss_mask).sum() / args.max_seq_len
	else:
		policy_loss = (policy_loss * loss_mask).sum() / (loss_mask.sum() + 1e-6)
	kl = torch.clamp(ref_logps - logps, -20, 20)
	kl_loss = torch.exp(kl) - kl - 1
	kl_loss = torch.clamp(kl_loss, -10, 10)
	if args.algo == 'dr_grpo':
		kl_loss = (kl_loss * loss_mask).sum() / args.max_seq_len
	else:
		kl_loss = (kl_loss * loss_mask).sum() / (loss_mask.sum() + 1e-6)
	clipped_ratio = get_ratio_stat(args, ratio, loss_mask)
	metrics = {
		'policy_loss': policy_loss, 
		'kl_loss': kl_loss, 
		'entropy_loss': entropy_loss, 
		'loss': policy_loss + args.kl_coeff * kl_loss,
		'clipped_ratio': clipped_ratio
		}
	return metrics

def caculate_gspo_loss(args, logps, old_logps, ref_logps, advs, loss_mask):
	advs = advs.unsqueeze(-1)
	logp_diff = torch.clamp(logps - old_logps, -20, 20)
	avg_logp_diff = (logp_diff * loss_mask) / (loss_mask.sum(dim=-1, keepdim=True) + 1e-6)
	avg_ratio = torch.exp(avg_logp_diff)
	clipped_avg_ratio = torch.exp(torch.clamp(avg_logp_diff, np.log(1 - args.clip_lower), np.log(1 + args.clip_higher)))
	entropy_loss = - (torch.exp(logps) * logps * loss_mask).sum() / (loss_mask.sum() + 1e-6)
	policy_loss = - torch.min(advs * avg_ratio, advs * clipped_avg_ratio)
	policy_loss = torch.mean((policy_loss * loss_mask).sum(dim=-1) / (loss_mask.sum(dim=-1) + 1e-6))
	kl = torch.clamp(ref_logps - logps, -20, 20)
	kl_loss = torch.exp(kl) - kl - 1
	kl_loss = torch.clamp(kl_loss, -10, 10)
	kl_loss = (kl_loss * loss_mask).sum() / (loss_mask.sum() + 1e-6)
	clipped_ratio = get_ratio_stat(args, avg_ratio, loss_mask)
	metrics = {
		'policy_loss': policy_loss, 
		'kl_loss': kl_loss, 
		'entropy_loss': entropy_loss, 
		'loss': policy_loss + args.kl_coeff * kl_loss,
		'clipped_ratio': clipped_ratio
		}
	return metrics

def check_update(args, logps, logps_after_update, advs, loss_mask):
	advs = advs.unsqueeze(-1)
	diff = torch.where(advs > 0, logps_after_update - logps, logps - logps_after_update)
	if ((diff * loss_mask) > 0).sum().item() != loss_mask.sum().item():
		if args.global_rank == 1:
			for i in range(logps.shape[0]):
				print(advs)
				print(logps[i, :30])
				print(logps_after_update[i, :30])
				print(diff[i, :30])
				print(loss_mask[i, :30])

# def caculate_grpo_loss(args, logps, old_logps, ref_logps, advs, loss_mask):
# 	advs = advs.unsqueeze(-1)
# 	# ratio = torch.exp(logps - logps.detach())
# 	ratio = torch.exp(torch.clamp(logps - old_logps, -20, 20))
# 	# clip = torch.max(- advs * ratio, - advs * torch.clamp(ratio, 1 - clip_range, 1 + clip_range))
# 	if args.clip_higher is not None and args.clip_lower is not None:
# 		clip = - torch.min(advs * ratio, advs * torch.clamp(ratio, 1 - args.clip_lower, 1 + args.clip_higher))
# 	else:
# 		clip = - torch.min(advs * ratio, advs * torch.clamp(ratio, 1 - args.clip_range, 1 + args.clip_range))
# 	# policy_loss = (- advs * ratio * loss_mask).sum() / loss_mask.sum()
# 	if args.use_dr_grpo:
# 		policy_loss = (clip * loss_mask).sum() / args.max_seq_len
# 	else:
# 		policy_loss = (clip * loss_mask).sum() / (loss_mask.sum() + 1e-6)
# 	kl = torch.clamp(ref_logps - logps, -20, 20)
# 	kl_loss = torch.exp(kl) - kl - 1
# 	kl_loss = torch.clamp(kl_loss, -10, 10)
# 	if args.use_dr_grpo:
# 		kl_loss = (kl_loss * loss_mask).sum() / args.max_seq_len
# 	else:
# 		kl_loss = (kl_loss * loss_mask).sum() / (loss_mask.sum() + 1e-6)
# 	kl_loss = torch.clamp(kl_loss, max=200)
# 	# print_rank0(args, [i for i, m in zip(logps[0].tolist(), loss_mask[0].tolist()) if m == 1])
# 	# print_rank0(args, [i for i, m in zip(old_logps[0].tolist(), loss_mask[0].tolist()) if m == 1])
# 	# print_rank0(args, [i for i, m in zip(ref_logps[0].tolist(), loss_mask[0].tolist()) if m == 1])
# 	# print_rank0(args, advs[0].tolist())
# 	return policy_loss + args.kl_coeff * kl_loss, policy_loss, kl_loss

def caculate_grpo_loss_kl_cov(args, logps, old_logps, ref_logps, advs, loss_mask):
	advs = advs.unsqueeze(-1).repeat(1, logps.shape[1])
	# ratio = torch.exp(logps - logps.detach())
	ratio = torch.exp(torch.clamp(logps - old_logps, -20, 20))
	# clip = torch.max(- advs * ratio, - advs * torch.clamp(ratio, 1 - clip_range, 1 + clip_range))
	pg_losses1 = - advs * ratio
	pg_losses_kl = - advs * ratio + (logps - old_logps).abs()
	pg_losses = pg_losses1
	all_valid = (loss_mask > 0)
	all_valid_idx = torch.nonzero(all_valid.reshape(-1), as_tuple=True)[0] 
	all_valid_adv = advs[all_valid].detach().reshape(-1).cpu()
	all_valid_logp = logps[all_valid].detach().reshape(-1).cpu()
	k_percent = args.k_percent
	k = min(k_percent, len(all_valid_adv))
	if k != 0:
		cov_lst_all = (all_valid_adv - all_valid_adv.mean()) * (all_valid_logp - all_valid_logp.mean())
		k_percent_nums = max(1, int(len(cov_lst_all) * k / 100))
		large_cov_idxs = torch.topk(cov_lst_all, k_percent_nums, largest=True).indices
		
		if len(large_cov_idxs) != 0:
			large_cov_idxs = all_valid_idx[large_cov_idxs]
			pg_losses[large_cov_idxs // advs.shape[1], large_cov_idxs % advs.shape[1]] = pg_losses_kl[large_cov_idxs // advs.shape[1], large_cov_idxs % advs.shape[1]]
	policy_loss = (pg_losses * loss_mask).sum() / (loss_mask.sum() + 1e-6)
	clipped_ratio = get_ratio_stat(args, ratio, loss_mask)
	metrics = {
		'loss': policy_loss,
		'clipped_ratio': clipped_ratio
		}
	return metrics

def gather_logps(logits, ids, shift=True):
	# logps: bsz, seq_len, vocab_size
	# ids: bsz, seq_len
	# if shift:
	# 	logits = logits[:, :-1]
	# 	ids = ids[:, 1:]
	return torch.gather(torch.log_softmax(logits, dim=-1), dim=-1, index=ids.unsqueeze(-1)).squeeze(-1)
	# selected_logits = torch.gather(logits, dim=-1, index=ids.unsqueeze(-1)).squeeze(-1)
	# # loop to reduce peak mem consumption
	# logsumexp_values = torch.stack([torch.logsumexp(lg, dim=-1) for lg in logits])
	# per_token_logps = selected_logits - logsumexp_values  # log_softmax(x_i) = x_i - logsumexp(x)
	# return torch.clamp(per_token_logps, min=-10)

def get_entropy(logits, mask):
	probs = torch.nn.functional.softmax(logits, dim=-1)
	entropy = torch.logsumexp(logits, dim=-1) - torch.sum(probs * logits, dim=-1)
	# print(logits.shape, entropy.shape, mask.shape)
	return (entropy * mask).sum() / (mask.sum() + 1e-6)

class RepeatTokenStoppingCriteria(StoppingCriteria):
	def __init__(self, max_repeat: int = 5):
		self.max_repeat = max_repeat

	def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
		if input_ids.shape[1] < self.max_repeat:
			return torch.full((input_ids.shape[0],), False, device=input_ids.device, dtype=torch.bool)
		last_tokens = input_ids[:, -self.max_repeat:]
		done_list = []
		for l in last_tokens:
			done_list.append(torch.all(l==l[0]).item())
		return torch.BoolTensor(done_list).to(input_ids.device)

class EarlyStoppingCrieria(StoppingCriteria):
	def __init__(self, tokenizer):
		self.tokenizer = tokenizer

	def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
		done_list = []
		for one in input_ids:
			done_list.append(self.check_response(self.tokenizer.decode(one.tolist())))
		return torch.BoolTensor(done_list).to(input_ids.device)

	def check_response(self, answer):
		answer = answer[answer.index('<|im_start|>assistant'):].replace('<|im_end|>', '')
		if '</think>' in answer:
			think_end_index = answer.index('</think>')
			if '<answer>' in answer:
				answer_start_index = answer.index('<answer>')
			else:
				answer_start_index = len(answer)
			if answer_start_index - think_end_index > 20:
				return True
		if '</answer>' in answer:
			answer_end_index = answer.index('</answer>')
			if len(answer) - answer_end_index > 10:
				return True
		words = answer.split()
		if len(re.findall('[.,:;!?]', ''.join(words[-100:]))) == 0 and len(words) > 100:
			return True
		for word in words:
			# extremely long word
			if len(word) > 100 and re.sub('[a-zA-Z]', '', word) == '':
				return True
		return False

def get_weight_ipc_handles(sd):
	from torch.multiprocessing.reductions import reduce_tensor
	from vllm.platforms import current_platform
	data = {}
	for name, p in sd.items():
		data[name] = reduce_tensor(p.detach())
	return {current_platform.get_device_uuid(0): data}

class ColocateWorkerExtension:
	"""
	The class for vLLM's worker to inherit from, in the colocate setting.
	By defining an extension class, the code can work no matter what is
	the underlying worker class. This way, the code can be compatible
	with both vLLM V0 and V1.
	NOTE: we define this class in a separate module, and the main module
	should pass the full qualified name as `worker_extension_cls` argument.
	"""
	def report_device_id(self) -> str:
		from vllm.platforms import current_platform
		self.device_uuid = current_platform.get_device_uuid(self.device.index)
		return self.device_uuid

	def update_weights_from_ipc_handles(self, ipc_handles):
		handles = ipc_handles[self.device_uuid]
		device_id = self.device.index
		weights = []
		for name, handle in handles.items():
			func, args = handle
			list_args = list(args)
			# the key is to change device id to the current device id
			# in case two processes have different CUDA_VISIBLE_DEVICES
			tensor = func(*list_args).to(device_id)
			# drop model. in name
			weights.append((name.replace('model.visual', 'visual').replace('model.language_model', 'language_model.model'), tensor))
			# print(name, tensor.shape)
		# print(self.model_runner.model)
		# weight_names = [w[0] for w in weights]
		# print('*****start-of-update-weights*****')
		# for name, p in self.model_runner.model.named_parameters():
		# 	if name not in weight_names:
		# 		print(name)
		self.model_runner.model.load_weights(weights=weights)
		torch.cuda.synchronize()
		# for name, p in self.model_runner.model.named_parameters():
		# 	print(name, p[0])
		# print('*****end-of-update-weights*****')

	def check_weights_changed(self, ipc_handles):
		"""
		Check if the weights are updated.
		"""
		handles = ipc_handles[self.device_uuid]
		device_id = self.device.index
		weights = {}
		for name, handle in handles.items():
			func, args = handle
			list_args = list(args)
			# the key is to change device id to the current device id
			# in case two processes have different CUDA_VISIBLE_DEVICES
			tensor = func(*list_args).to(device_id)
			# drop model. in name
			weights[name.replace('model.visual', 'visual').replace('model.language_model', 'language_model.model')] =  tensor
		
		print('******start-of-check-weights*****')
		print(list(weights.keys()))
		weights_updated = True
		for name, p in self.model_runner.model.named_parameters():
			print(name, p[0])
			weights_updated = weights_updated and torch.allclose(p, weights[name])
		print(weights_updated)
		return weights_updated