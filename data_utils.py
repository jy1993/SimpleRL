import json
import sys
import re
from PIL import Image
# from mathruler.grader import extract_boxed_content, grade_answer

def to_jsonl(data, fout):
	with open(fout, 'w', encoding='utf8') as f:
		for row in data:
			f.write(json.dumps(row, ensure_ascii=False) + '\n')

def read_jsonl(filename):
	data = []
	with open(filename, 'r', encoding='utf8') as f:
		for line in f.readlines():
			data.append(json.loads(line))
	return data

def read_json(filename):
	with open(filename, 'r', encoding='utf8') as f:
		data = json.load(f)
	return data

def to_json(data, fout):
	with open(fout, 'w', encoding='utf8') as f:
		json.dump(data, f, ensure_ascii=False, indent=4)	

def extract_last_box(response):
	box = re.findall('boxed{[0-9a-zA-Z\.\+\\\{\},°/\s^-]+}', response)
	if box:
		return box[0][6:-1]
	return ''

def extract_answer_from_model_response(response, task_type='math'):
	if task_type == 'math':
		return extract_last_box(response)
	if '<answer>' in response and '</answer>' in response:
		answer = response[response.index('<answer>')+8:response.index('</answer>')]
		return answer
	else:
		return ''

def parse_model_answer_for_logic(answer_text, expected_names):
	pred_dict = {}
	knight_count = answer_text.lower().count('knight')
	knave_count = answer_text.lower().count('knave')
	if knight_count + knave_count != len(expected_names):
		return None
	for name in expected_names:
		pattern = re.compile(
			rf'\b{re.escape(name)}\b\s+is\s+a\s+\b(knight|knave)\b', 
			re.IGNORECASE
		)
		match = pattern.search(answer_text)
		if match:
			role = match.group(1).lower()
			pred_dict[name] = role
		else:
			return None
	return pred_dict

def number_equal(a, b):
	if a == b:
		return True
	try:
		if abs(eval(a) - eval(b)) < 0.1:
			return True
	except:
		return False

def latex_norm(answer):
	if answer.endswith(';') or answer.endswith('.'):
		answer = answer[:-1]
	answer = answer.replace('\\left', '').replace('\\right', '')
	answer = answer.replace('－', '-').replace('∶', ':').replace('，', ',')
	answer = answer.replace('$', '').strip().replace(' ', '')
	# number{x} ==> x
	number = re.findall('number{[0-9\.]+}', answer)
	if number:
		if '\\' + number[0] in answer:
			answer = answer.replace('\\' + number[0], number[0][number[0].index('{')+1:-1])
		else:
			answer = answer.replace(number[0], number[0][number[0].index('{')+1:-1])
	# text{a} ==> a
	text = re.findall('text{[0-9\.]+}', answer)
	if text:
		if '\\' + text[0] in answer:
			answer = answer.replace('\\' + text[0], text[0][text[0].index('{')+1:-1])
		else:
			answer = answer.replace(text[0], text[0][text[0].index('{')+1:-1])
	# ^circ ==> °
	answer = answer.replace('^{\\circ}', '°').replace('{}^\\circ', '°').replace('^\\circ', '°').replace('degrees', '°').replace('^{°}', '°')
	degree = re.findall('{{[0-9\.]+}°}', answer)
	if degree:
		answer = answer.replace(degree[0], degree[0][2:-3] + '°')
	# meters ==> m
	answer = answer.replace('meters', 'm')
	# pi ==> π
	answer = answer.replace('\\pi', 'π')
	# a:b ==> a/b
	answer = answer.replace(':', '/')
	# pm ==> ±
	answer = answer.replace('\\pm', '±')
	# le ==> ≤
	answer = answer.replace('\\leqslant', '≤').replace('\\le', '≤')
	# ge ==> ≥
	answer = answer.replace('\\ge', '≥')
	# %
	answer = answer.replace('\\%', '%')
	# angel ==> ∠
	answer = answer.replace('\\angle', '∠')
	# parallel ==> ||
	answer = answer.replace('\\parallel', '||')
	# perp ==> ⊥
	answer = answer.replace('\\perp', '⊥')
	# sqrt{x} ==> √
	# sqrt{x} ==> (x ** 0.5) 
	sqrt = re.findall('sqrt{[0-9π\.]+}', answer)
	if sqrt:
		for idx in range(len(sqrt)):
			if '\\' + sqrt[idx] in answer:
				answer = answer.replace('\\' + sqrt[idx], '(' + sqrt[idx][5:-1] + '**0.5)')
			else:
				answer = answer.replace(sqrt[idx], '(' + sqrt[idx][5:-1] + '**0.5)')
	# dfrac/tfrac ==> frac
	answer = answer.replace('\\dfrac', '\\frac').replace('\\tfrac', '\\frac')
	# frac{a}{b} ==> a/b
	frac = re.findall('frac{[0-9π√\(\)\*\.]+}{[0-9π√\(\)\*\.]+}', answer)
	if frac:
		for idx in range(len(frac)):
			if '\\' + frac[idx] in answer:
				answer = answer.replace('\\' + frac[idx], frac[idx][frac[idx].index('{')+1: frac[idx].index('}')] + '/' + frac[idx][frac[idx].index('}')+2:-1])
			else:
				answer = answer.replace(frac[idx], frac[idx][frac[idx].index('{')+1: frac[idx].index('}')] + '/' + frac[idx][frac[idx].index('}')+2:-1])
	# unit{cm} ==> cm
	unit = re.findall('unit{[a-z°]+}', answer)
	if unit:
		if '\\' + unit[0] in answer:
			answer = answer.replace('\\' + unit[0], unit[0][unit[0].index('{')+1:-1])
		else:
			answer = answer.replace(unit[0], unit[0][unit[0].index('{')+1:-1])
	answer = answer.replace('\\text{km}^2', 'km²')
	answer = answer.replace('\\text{km}', 'km')
	answer = answer.replace('\\text{m}^2', 'm²')
	answer = answer.replace('\\text{m}', 'm')
	answer = answer.replace('\\text{cm}^2', 'cm²')
	answer = answer.replace('\\text{cm}', 'cm')
	answer = answer.replace('\\text{mm}^2', 'mm²')
	answer = answer.replace('\\text{mm}', 'mm')
	answer = answer.replace('\\text{ft}^2', 'ft²')
	answer = answer.replace('\\text{squarefeet}', 'ft²')
	answer = answer.replace('\\text{yd}^2', 'yd²')
	answer = answer.replace('\\text{inches}', 'inches')
	answer = answer.replace('\\text{squareinches}', 'inches²')
	answer = answer.replace('{{m}^{2}}', 'm²')
	answer = answer.replace('\\text{units}', '')

	answer = answer.replace('\,', '')

	match = re.match('[0-9\.]+%', answer)
	if match:
		number = match[0]
		answer = answer.replace(number, '%.3f' % (float(number[:-1])*0.01))
	sqrt_post = re.findall('[0-9]+\(', answer)
	if sqrt_post:
		for idx in range(len(sqrt_post)):
			answer = answer.replace(sqrt_post[idx], sqrt_post[idx][:-1] + '*(')
	# TODO: expression: -2a+b-ab
	for unit in ['km²', 'cm²', 'mm²', 'm²', '°', 'ft²', 'yd²', 'inches²', 'inches', 'km', 'cm', 'mm', 'm']:
		if answer.endswith(unit):
			answer = answer[:-len(unit)]
	return answer

def safe_load_image(img_path):
	with open(img_path, "rb") as f:
		img = Image.open(f)
		img.load()
	return img

def is_equal(response, label, task_type='math'):
	if task_type == 'math':
		pred = extract_answer_from_model_response(response)
		pred = pred.replace(' ', '')
		label = label.replace(' ', '')
		if pred == label:
			return True
		elif number_equal(latex_norm(pred), latex_norm(label)):
			return True
		return False
	elif task_type == 'logic':
		pred = extract_answer_from_model_response(response, task_type='logic')
		gold_dict = json.loads(label)
		pred_dict = parse_model_answer_for_logic(pred, list(gold_dict.keys()))
		if pred_dict and gold_dict == pred_dict:
			return True
		return False
# def accuracy_reward(response: str, ground_truth: str) -> float:
# 	answer = extract_boxed_content(response)
# 	return 1.0 if grade_answer(answer, ground_truth) else 0.0

def process_image(image, min_pixels: int, max_pixels: int):
	if isinstance(image, str):
		image = Image.open(image)
	elif isinstance(image, dict):
		image = Image.open(BytesIO(image["bytes"]))
	elif isinstance(image, bytes):
		image = Image.open(BytesIO(image))
	image.load()  # avoid "Too many open files" errors
	if max_pixels is not None and (image.width * image.height) > max_pixels:
		resize_factor = math.sqrt(max_pixels / (image.width * image.height))
		width, height = int(image.width * resize_factor), int(image.height * resize_factor)
		image = image.resize((width, height))
	if min_pixels is not None and (image.width * image.height) < min_pixels:
		resize_factor = math.sqrt(min_pixels / (image.width * image.height))
		width, height = int(image.width * resize_factor), int(image.height * resize_factor)
		image = image.resize((width, height))
	if image.mode != "RGB":
		image = image.convert("RGB")
	return image

def process_multi_modal_data(multi_modal_data, min_pixels: int, max_pixels: int, video_fps: float):
	# may convert image path to image object
	images, videos = [], []
	if "images" in multi_modal_data:
		for image in multi_modal_data["images"]:
			images.append(process_image(image, min_pixels, max_pixels))
	if "videos" in multi_modal_data:
		for video in multi_modal_data["videos"]:
			videos.append(process_video(video, min_pixels, max_pixels, video_fps))
	if len(images) != 0:
		return {"image": images}
	if len(videos) != 0:
		return {"video": videos}
	return None
