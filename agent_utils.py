from copy import copy
import requests

def repeat_dict(x, n):
	new = []
	for i in x:
		for _ in range(n):
			new.append(copy(i))
	return new

def call_python_server(history, response, start_str, stop_str, max_result_len=300):
	try:
		json_data = requests.post('http://127.0.0.1:8000/run_code', 
			json={'history': history, 'response': response, 'start_str':start_str, 'stop_str': stop_str}, timeout=1).json()
		code = json_data['code']
		code_result = json_data['result']
	except:
		code = ''
		code_result = 'Exec code time out'
	# user = f'<interpreter>\n{code_result}\n</interpreter>'
	user = f'<tool_response>\n{code_result}\n</tool_response>'
	return code, user

