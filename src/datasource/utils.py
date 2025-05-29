import json
import os
import re
from typing import Optional
import numpy as np

import six
from tqdm import tqdm

def load_jsonl(jsonl_path: str, indexed: bool = False, id_key_name: Optional[str] = "example_id", desc:str=None):
	"""
	Loads the jsonl dataset into an indexed dictionary with the following structure {example_id: example}
	:param jsonl_path: path to jsonl dataset
	:param indexed:
	:param id_key_name:
	:param desc: description for tqdm
	:return:
	"""
	result = {} if indexed else []
	with open(jsonl_path, "r", encoding="utf-8") as f:
		for line in tqdm(f, desc=desc):
			line = six.ensure_text(line, "utf-8")
			example = json.loads(line)
			if indexed:
				result[example[id_key_name]] = example
			else:
				result.append(example)
	return result

def load_json(json_path: str, indexed: bool = False, id_key_name: str | int = "example_id"):
	"""
	Loads the json dataset into a list of indexed dictionary
	:param json_path: path to jsonl dataset
	:param indexed:
	:param id_key_name:
	:return:
	"""
	with open(json_path) as f:
		json_file = json.load(f)
		if not indexed and isinstance(json_file, list):
			return json_file
		elif not indexed and isinstance(json_file, dict):
			unindexed_json = []
			for key, val in json_file.items():
				if isinstance(val, list):
					unindexed_json.append([key]+val)
				else:
					val["id"] = key
					unindexed_json.append(val)
			return unindexed_json
		elif indexed and isinstance(json_file, dict):
			return json_file
		else:
			if not isinstance(json_file, list):
				raise Exception("Only json lists can be indexed")
			d = {}
			for element in json_file:
				d[element[id_key_name]] = element
			return d


def save_jsonl(out_path: str, dataset: list):
	# Create the directories if they do not exist
	if not os.path.exists(os.path.dirname(out_path)):
		os.makedirs(os.path.dirname(out_path))
	with open(os.path.join(out_path), 'w', encoding='utf8') as outfile:
		for sample in dataset:
			json.dump(sample, outfile, ensure_ascii=False, cls=NpEncoder)
			outfile.write('\n')

def reindex_by(dataset:list, new_index_key: str=None, key_process_fn=None, key_factory_fn=None, append_duplicates:bool=True) -> dict:
	"""
	:param dataset:
	:param new_index_key: Requires unless you prodice a key_factory_fn
	:param key_process_fn: function to mutate the key (e.g. set low lower a property of example[new_index_key]
	:param key_factory_fn: lambda function that receives the example and produces the key
	:param append_duplicates:
	:return:
	"""
	if key_process_fn is None:
		key_process_fn = lambda a : a
	if new_index_key is None and key_factory_fn is None:
		raise ValueError("If new_index_key is None you must provide a key_factory_fn")
	if key_factory_fn is None:
		key_factory_fn = lambda a: a[new_index_key]
	d = {}
	for example in dataset:
		key = key_process_fn(key_factory_fn(example))
		if append_duplicates:
			if key in d:
				d[key].append(example)
			else:
				d[key] = [example]
		else:
			d[key] = example
	return d

def split_into_tokens(text, tokenizer):
	"""
	This is a more sophisticated way of splitting a string into tokens than using split by ' '. It mimics the token split
	that the subword tokenizer would do. We merge part '##' tokens into the same token. This is commonly used for
	spliting the topìc into tokens
	:param text:
	:return:
	"""
	all_sub_token = tokenizer.tokenize(text)

	no_subword_tokens = []

	for sub_token in all_sub_token:
		if len(sub_token) > 2 and sub_token[0:2] == '##':
			no_subword_tokens[-1] += sub_token[2:]
		else:
			no_subword_tokens.append(sub_token)
	return no_subword_tokens


class NpEncoder(json.JSONEncoder):
	def default(self, obj):
		if isinstance(obj, np.integer):
			return int(obj)
		if isinstance(obj, np.floating):
			return float(obj)
		if isinstance(obj, np.ndarray):
			return obj.tolist()
		return super(NpEncoder, self).default(obj)

def chunks(lst, n):
	"""Yield successive n-sized chunks from lst."""
	for i in range(0, len(lst), n):
		yield lst[i:i + n]

def divide_into_chunks(lst, n):
	"""Divides a list into n chunks. Last chunk might be different size"""
	avg = len(lst) // n
	remainder = len(lst) % n
	sublists = []
	start = 0
	for i in range(n):
		extra = 1 if i < remainder else 0  # Distribute remainder items
		sublists.append(lst[start:start + avg + extra])
		start += avg + extra
	return sublists


def safe_file_name(input_string):
	"""
	Converts a string into a safe file name for Linux systems.

	Parameters:
		input_string (str): The original string to be converted.

	Returns:
		str: A safe file name.
	"""
	# Replace spaces with underscores
	safe_name = input_string.replace(" ", "_")

	# Remove invalid characters
	safe_name = re.sub(r'[<>:"/\\|?*]', '', safe_name)

	# Remove any control characters or invisible characters
	safe_name = re.sub(r'[\x00-\x1F\x7F]', '', safe_name)

	# Replace multiple underscores or dashes with a single one
	safe_name = re.sub(r'[_-]+', '_', safe_name)

	# Trim leading and trailing underscores or dashes
	safe_name = safe_name.strip("_-")

	# Return the cleaned and safe name
	return safe_name