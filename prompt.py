def get_prompt_prefix(task_type):
	system_prompt = 'You are a helpful assistant.'
	if task_type == 'vl_math':
		prompt_prefix = "Please reason step by step and put your final answer in \\boxed{}."
	elif task_type == 'logic':
		prompt_prefix = (
			'First thinks about the reasoning process in the mind and then provides the user with the answer.' 
			'The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags,'
			'respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. ' 
			'Now the user asks you to solve a logical reasoning problem. After thinking, when you finally reach a conclusion, clearly state the identity of each character within <answer> </answer> tags.'
			'List the identity of each person one by one, for example, <answer> (1) xx is a knight\n(2) xx is a knave\n(3)... </answer>.\n'
			)
	elif task_type == 'math':
		prompt_prefix = (
			'First thinks about the reasoning process in the mind and then provides the user with the answer.' 
			'The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags,'
			'respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. \n' 
			)
	elif task_type == 'agent_math':
		# prompt_prefix = (
		# 	'To answer questions, you must conduct reasoning first. ' 
		# 	'After reasoning, write python code to do the math and '
		# 	'user will return the results to you. '
		# 	'Once you have sufficient information, put you final answer inside \\boxed{} tag. '
		# 	'\nNote:\n'
		# 	'1.Allowed modules: math, cmath, numpy, scipy, sympy, itertools, fractions, collections\n'
		# 	'2.Try to write efficient code, since code running has time limit\n'
		# 	'3.Do not generate output of python code, the user will provide it to you\n'
		# 	'4.Remember to use print function to show your result\n'
		# 	'Question:\n'
		# 	)
		# retool prompt
		# system_prompt = "Solve the following problem step by step. You now have the ability to selectively write executable Python code to enhance your reasoning process. The Python code will be executed by an external sandbox, and the output (wrapped in `<interpreter>output_str</interpreter>`) can be returned to aid your reasoning and help you arrive at the final answer. The Python code should be complete scripts, including necessary imports. \nEach code snippet is wrapped with `<code>\n```python\ncode snippet\n```\n</code>`.\nThe last part of your response should be in the following format:\n<answer>\n\\boxed{{'The final answer goes here.'}}\n</answer>\n\n*user question:*\n"
		system_prompt = "Solve the following problem step by step. You now have the ability to selectively write executable Python code to enhance your reasoning process. The Python code will be executed by an external sandbox, and the output (wrapped in '<tool_response> output_str </tool_response>') can be returned to aid your reasoning and help you arrive at the final answer. The Python code should be complete scripts, including necessary imports. \nEach code snippet is wrapped with '```python\ncode snippet\n```\n'.\nThe last part of your response should be in the following format:\n<answer>\n\\boxed{{'The final answer goes here.'}}\n</answer>"
		prompt_prefix = ""
	return system_prompt, prompt_prefix