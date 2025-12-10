import copy
from agent_utils import repeat_dict, call_python_server
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from data_utils import read_jsonl, to_json, extract_answer_from_model_response, is_equal
import argparse
from prompt import get_prompt_prefix
import torch
from collections import defaultdict
import numpy as np
import os

os.environ['USER'] = 'USER'
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default=None)
parser.add_argument('--max_new_tokens', type=int, default=None)
parser.add_argument('--max_tool_calls', type=int, default=None)
parser.add_argument('--test_path', type=str, default=None)
parser.add_argument('--max_test_samples', type=int, default=100)
parser.add_argument('--metric', type=str, default=None, choices=['pass@k', 'avg@k', 'acc'])
parser.add_argument('--k', type=int, default=8)
parser.add_argument('--save_details_path', type=str, default=None)
parser.add_argument('--save_evaluation_path', type=str, default=None)
parser.add_argument('--n_gpus', type=int, default=1)
args = parser.parse_args()

class AgentEval():
	"""docstring for AgentEval"""
	def __init__(self, tokenizer, llm):
		self.tokenizer = tokenizer
		self.tokenizer.padding_side = 'left'
		self.llm = llm
		self.eval_data = read_jsonl(args.test_path)[:args.max_test_samples]
		self.max_new_tokens = args.max_new_tokens
		self.max_tool_calls = args.max_tool_calls
		# self.start_str = '<code>\n```python'
		# self.stop_str = '```\n</code>'
		self.start_str = '```python\n'
		self.stop_str = '\n```\n'
		self.sampling_params_agent_stop = SamplingParams(
			n=1,
			temperature=1.0,
			top_p=0.7,
			max_tokens=self.max_new_tokens,
			stop=[self.stop_str],
			include_stop_str_in_output=True,
			stop_token_ids=[151643, 151645]
		)
		self.sampling_params_final = SamplingParams(
			n=1,
			temperature=1.0,
			top_p=0.7,
			max_tokens=self.max_new_tokens,
			stop_token_ids=[151643, 151645]
		)
		self.metric = args.metric
		self.group_size = args.k
		self.unbiased_pass_k_n = args.k
		self.pass_k = '1'
		self.task_type = 'agent_math'
		self.system_prompt = 'You are a helpful assistant.'
		self.prompt_prefix = get_prompt_prefix(self.task_type)
		self.details = []
		self.evaluation_results = {'test_set': args.test_path, 'model_path': args.model_path}
		
	def get_prompts_and_labels(self, samples, extra_keys=None):
		# system = "Solve the following problem step by step. You now have the ability to selectively write executable Python code to enhance your reasoning process. The Python code will be executed by an external sandbox, and the output (wrapped in `<interpreter>output_str</interpreter>`) can be returned to aid your reasoning and help you arrive at the final answer. The Python code should be complete scripts, including necessary imports. \nEach code snippet is wrapped with `<code>\n```python\ncode snippet\n```\n</code>`.\nThe last part of your response should be in the following format:\n<answer>\n\\boxed{{'The final answer goes here.'}}\n</answer>"
		system = "Solve the following problem step by step. You now have the ability to selectively write executable Python code to enhance your reasoning process. The Python code will be executed by an external sandbox, and the output (wrapped in '<tool_response> output_str </tool_response>') can be returned to aid your reasoning and help you arrive at the final answer. The Python code should be complete scripts, including necessary imports. \nEach code snippet is wrapped with '```python\ncode snippet\n```\n'.\nThe last part of your response should be in the following format:\n<answer>\n\\boxed{{'The final answer goes here.'}}\n</answer>"
		all_prompts, all_labels = [], []
		if extra_keys:
			extra_info = {key: [] for key in extra_keys}
		else:
			extra_info = None
		for sample in samples:
			messages = [
				{"role": "system", "content": system},
				{"role": "user", "content": sample['query']}
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
					extra_info[key].append(sample.get(key, 'default'))
		if extra_keys:
			return all_prompts, all_labels, extra_info
		else:
			return all_prompts, all_labels

	def within_max_tokens(self, prompt_len, total_len, max_tokens):
		response_len = total_len - prompt_len
		if response_len < max_tokens:
			return True
		return False

	def get_sampling_params_list(self, total_len_list, len_list):
		sp_list = [copy.copy(self.sampling_params_agent_stop) for _ in range(len(len_list))]
		for sp, l, tl in zip(sp_list, len_list, total_len_list):
			sp.max_tokens = tl - l
		return sp_list

	def generate_sequences_for_agent(self, batch_prompts, repeat_time):
		if repeat_time > 1:
			batch_prompts = repeat_dict(batch_prompts, repeat_time)
		# to id
		inputs = self.tokenizer(batch_prompts, padding=False)
		prompt_lens = [len(x) for x in inputs['input_ids']]
		batch_prompt_ids = [{'prompt_token_ids': x} for x in inputs['input_ids']]
		batch_attention_mask = [{'attention_mask': x} for x in inputs['attention_mask']]
		k = [0] * len(batch_prompt_ids)
		tool_call_history = [''] * len(batch_prompt_ids)
		for ridx in range(self.max_tool_calls):
			# stop reason for each round: stop_str/stop_token_ids/max_len
			stop_for_tool_call, index_mapping = [], {}
			sftc_pl_list = []
			for idx in range(len(batch_prompt_ids)):
				if k[idx] < self.max_tool_calls and self.within_max_tokens(prompt_lens[idx], len(batch_prompt_ids[idx]['prompt_token_ids']), self.max_new_tokens):
					stop_for_tool_call.append(batch_prompt_ids[idx])
					sftc_pl_list.append(prompt_lens[idx])
					index_mapping[len(index_mapping)] = idx
			if len(stop_for_tool_call) > 0:
				sp_list = self.get_sampling_params_list([pl + self.max_new_tokens for pl in sftc_pl_list], [len(i['prompt_token_ids']) for i in stop_for_tool_call])
				# print([sp.max_tokens for sp in sp_list])
				stop_outputs = self.llm.generate(stop_for_tool_call, sp_list)
				for idx, output in enumerate(stop_outputs):
					response = output.outputs[0].text
					finish_reason = output.outputs[0].finish_reason
					response_ids = self.tokenizer(response)['input_ids']
					batch_prompt_ids[index_mapping[idx]]['prompt_token_ids'] += response_ids
					batch_attention_mask[index_mapping[idx]]['attention_mask'] += [1] * len(response_ids)
					if finish_reason == 'stop':
						# stop for stop_str
						if response.endswith(self.stop_str):
							k[index_mapping[idx]] += 1
							tool_call, tool_call_result = call_python_server(tool_call_history[index_mapping[idx]], response, self.start_str, self.stop_str)
							tool_call_result_ids = self.tokenizer(tool_call_result)['input_ids']
							tool_call_history[index_mapping[idx]] += tool_call + '\n'
							batch_prompt_ids[index_mapping[idx]]['prompt_token_ids'] += tool_call_result_ids
							batch_attention_mask[index_mapping[idx]]['attention_mask'] += [0] * len(tool_call_result_ids)
						# stop for eos
						else:
							k[index_mapping[idx]] = 1000
					else:
						assert finish_reason == 'length'
						k[index_mapping[idx]] = 1000
			# all stop
			if all([x > self.max_tool_calls for x in k]):
				break
			print('-----------------end-of-stop--------------')
			# take care of tool call
		# for i in range(len(batch_prompt_ids)):
		# 	print('*' * 10)
		# 	print(self.tokenizer.decode(batch_prompt_ids[i]['prompt_token_ids']))
		# pad to max_len
		# print(tool_call_history)
		# fix ill stop
		# for _ in range(10):
		# 	with_no_answer, index_mapping = [], {}
		# 	for idx, prompt_ids in enumerate(batch_prompt_ids):
		# 		if self.ill_stop(prompt_ids):
		# 			with_no_answer.append(prompt_ids)
		# 			index_mapping[len(index_mapping)] = idx
		# 		if len(with_no_answer) > 0:
		# 			outputs = self.llm.generate(with_no_answer, self.sampling_params_final)
		# 			for idx, output in enumerate(outputs):
		# 				response = output.outputs[0].text
		# 				response_ids = self.tokenizer(response)['input_ids']
		# 				batch_prompt_ids[index_mapping[idx]]['prompt_token_ids'] += response_ids

		seqs, masks = [], []
		for prompt, prompt_len in zip(batch_prompt_ids, prompt_lens):
			seqs.append(prompt['prompt_token_ids'][prompt_len:])
		return seqs

	def ill_stop(self, prompt_ids):
		seq = self.tokenizer.decode(prompt_ids['prompt_token_ids'])
		if seq.endswith('</interpreter>') and '<answer>' not in seq:
			return True
		return False

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
		print('-----prompt-----', all_prompts[0], all_labels[0])
		if self.metric == 'pass@k':
			seqs = self.generate_sequences_for_agent(all_prompts, self.unbiased_pass_k_n)
		elif self.metric == 'avg@k':
			seqs = self.generate_sequences_for_agent(all_prompts, self.group_size)
		elif self.metric == 'acc':
			seqs = self.generate_sequences_for_agent(all_prompts, 1)
		self.llm.sleep(level=1)
		torch.cuda.empty_cache()
		responses = self.tokenizer.batch_decode(seqs)
		if self.metric == 'pass@k':
			assert len(responses) == len(all_labels) * self.unbiased_pass_k_n
		elif self.metric == 'avg@k':
			assert len(responses) == len(all_labels) * self.group_size
		elif self.metric == 'acc':
			assert len(responses) == len(all_labels)
		pass_at_k = defaultdict(list)
		empty_count = 0
		# showcase = 2
		if self.metric == 'pass@k':
			for k in [int(i) for i in self.pass_k.split(',') if i != '']:
				for i in range(len(all_prompts)):
					c = 0
					data_source = all_data_sources[i]
					label = all_labels[i]
					for response in responses[i*self.unbiased_pass_k_n:(i+1)*self.unbiased_pass_k_n]:
						answer = extract_answer_from_model_response(response, self.task_type)
						if is_equal(answer, label, self.task_type):
							c += 1
						if answer == '':
							empty_count += 1
						# if showcase > 0:
						# 	print('*' * 10)
						# 	print('query: ' + self.eval_data[i]['query'])
						# 	print(f'response: ' + response)
						# 	print(f'answer: {answer}')
						# 	print(f'label: {label}')
						# 	showcase -= 1
						self.details.append({
							'data_source': data_source,
							'query': self.eval_data[i]['query'], 
							'response': response, 
							'gt': label, 
							'pred': answer,
							'equal': is_equal(answer, label, self.task_type)})
					pass_at_k[data_source + '_pass@' + str(k)].append(self.estimate_pass_at_k(self.unbiased_pass_k_n, c, k))
			pass_at_k['empty_ratio'] = [empty_count / len(responses)]
			self.evaluation_results.update({key: sum(value) / len(value) for key, value in pass_at_k.items()})
		elif self.metric == 'avg@k':
			avg_at_k = {}
			for i in range(len(all_prompts)):
				data_source = all_data_sources[i]
				label = all_labels[i]
				if data_source not in avg_at_k:
					avg_at_k[data_source] = {'correct': 0, 'total': 0, 'empty': 0, 'group_correct': 0, 'group_total': 0}
				for response in responses[i*self.group_size:(i+1)*self.group_size]:
					answer = extract_answer_from_model_response(response, self.task_type)
					avg_at_k[data_source]['total'] += 1
					if is_equal(answer, label, self.task_type):
						avg_at_k[data_source]['correct'] += 1
					if answer == '':
						avg_at_k[data_source]['empty'] += 1
					self.details.append({
						'data_source': data_source,
						'query': self.eval_data[i]['query'], 
						'response': response, 
						'gt': label, 
						'pred': answer,
						'equal': is_equal(answer, label, self.task_type)})
				avg_at_k[data_source]['group_total'] += 1
				if any([x['equal'] for x in self.details[-self.group_size:]]):
					avg_at_k[data_source]['group_correct'] += 1
			for k, v in avg_at_k.items():
				self.evaluation_results[k + '_avg@' + str(self.group_size)] = v['correct'] / v['total']
				self.evaluation_results[k + '_empty_ratio'] = v['empty'] / v['total']
				self.evaluation_results[k + '_group_acc'] = v['group_correct'] / v['group_total']
		elif self.metric == 'acc':
			empty_count, correct = 0, 0
			for i in range(len(all_prompts)):
				data_source = all_data_sources[i]
				label = all_labels[i]
				response = responses[i]
				answer = extract_answer_from_model_response(response, self.task_type)
				if is_equal(answer, label, self.task_type):
					correct += 1
				if answer == '':
					empty_count += 1
				self.details.append({
					'data_source': data_source,
					'query': self.eval_data[i]['query'], 
					'response': response, 
					'gt': label, 
					'pred': answer,
					'equal': is_equal(answer, label, self.task_type)})
			total = len(all_prompts)
			self.evaluation_results.update({'acc': correct / total, 'empty_ratio': empty_count / total})

	def save_everything(self):
		to_json(self.evaluation_results, args.save_evaluation_path)
		to_json(self.details, args.save_details_path)

llm = LLM(
	model=args.model_path,
	tensor_parallel_size=args.n_gpus,
	max_model_len=30000,
	gpu_memory_utilization=0.8,
	enforce_eager=True,
	# enforce_eager=False,
	enable_sleep_mode=True,
	disable_mm_preprocessor_cache=True,
)
tokenizer = AutoTokenizer.from_pretrained(args.model_path)
agent_eval = AgentEval(tokenizer, llm)
agent_eval.eval_model()
agent_eval.save_everything()
