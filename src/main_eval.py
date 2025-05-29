import copy
import hashlib
import json
import os
import glob
import pickle
import random
import re
import statistics
from collections import defaultdict
from collections import Counter
import math
import matplotlib.ticker as mtick
from scipy.stats import chisquare
from scipy.stats import ttest_ind

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from datasource.meta_correlation.utils import get_key_attributes_for_object, get_gold_object, get_linking_attributes, \
	remove_nonlinking_attributes, get_unique_identifiers, group_by_overlap
from datasource.utils import load_jsonl, reindex_by, save_jsonl

ANSWER_JSON_REGEX = r'\{ *"answer": .*?\}'

def clean_prediction(text: str):
	text = text.strip()
	#text = f"{text}}}" if text[-1] != "}" else text
	text = text.replace("\\", "")
	text = text.replace("\n", "")
	matches = re.findall(ANSWER_JSON_REGEX, text)
	try:
		str_json = matches[-1]
		return json.loads(str_json), True
	except:
		return text, False


def calc_acc(dataset, accept_mention:bool = False):
	total = []
	for ex in dataset:
		result = 0
		pred_json, is_json = clean_prediction(ex["prediction"])
		if is_json:
			result = 1 if str(json.loads(ex["gold_reference"])["answer"]) == str(pred_json["answer"]).strip() else 0
		elif accept_mention:
			result = 1 if str(json.loads(ex["gold_reference"])["answer"]) in pred_json else 0
		total.append(result)
	return round(np.mean(total)*100, 1)

def extract_points(molmo_output, image_w, image_h):
	"""
	Extracts pointing points from a molmo output, e.g. '<point x="59.6" y="28.3" alt="red cube">red cube</point>'
	@param molmo_output:
	@param image_w:
	@param image_h:
	@return:
	"""
	all_points = []
	for match in re.finditer(r'x\d*="\s*([0-9]+(?:\.[0-9]+)?)"\s+y\d*="\s*([0-9]+(?:\.[0-9]+)?)"', molmo_output):
		try:
			point = [float(match.group(i)) for i in range(1, 3)]
		except ValueError:
			pass
		else:
			point = np.array(point)
			if np.max(point) > 100:
				# Treat as an invalid output
				continue
			point /= 100.0
			point = point * np.array([image_w, image_h])
			all_points.append(point)
	return all_points

def build_results(pred_dir:str, accept_mention_correct:bool = False, setting:str = "org"):
	if ".jsonl" in pred_dir.split("/")[-1]:
		file_names = [pred_dir]
	else:
		file_names = glob.glob(os.path.join(pred_dir, "*.jsonl"))
	results = defaultdict(lambda:defaultdict(lambda:defaultdict(dict)))
	for file_name in file_names:
		if (setting == "org" and "_cot_" in file_name) or (setting == "cot" and "_cot_" not in file_name):
			continue
		dataset = load_jsonl(file_name)
		model_name = dataset[0]["model"]
		scene_format = dataset[0]["scene_format"]
		#nshots = len(dataset[0]["few_shot_colors"]) if "few_shot_colors" in dataset[0] else len(dataset[0]["few_shot_names"])
		nshots = len(dataset[0]["few_shot_attributes"])
		modality = "mm" if "mm_" in os.path.basename(file_name) else "um"
		# All
		results[scene_format][nshots][model_name][f"overall_{modality}"] = calc_acc(dataset, accept_mention=accept_mention_correct)
		# Task
		task_dataset = reindex_by(dataset, "task", append_duplicates=True)
		for task, subset in task_dataset.items():
			results[scene_format][nshots][model_name][task] = calc_acc(subset)
		# Object count
		obj_count_dataset = reindex_by(dataset, f"object_count", append_duplicates=True)
		results[scene_format][nshots][model_name][f"object_count_{modality}"] = {}
		for obj_count, subset in obj_count_dataset.items():
			results[scene_format][nshots][model_name][f"object_count_{modality}"][str(obj_count)] = calc_acc(subset)
		# Target attribute
		for attr in ["target_attribute", "pointer_attribute"]:
			tgt_dataset = reindex_by(dataset, key_factory_fn=lambda a:list(a[attr].keys())[0], append_duplicates=True)
			results[scene_format][nshots][model_name][f"{attr}_{modality}"] = {}
			for task in task_dataset.keys():
				results[scene_format][nshots][model_name][f"{attr}_{modality}"][task] = {}
				for attr_name, subset in tgt_dataset.items():
					task_subset = [ex for ex in subset if ex["task"] == task]
					if len(task_subset) > 0:
						results[scene_format][nshots][model_name][f"{attr}_{modality}"][task][attr_name] = calc_acc(task_subset)
	return results

def add_task_chance(pred_dir:str, task_table:dict, nshots:int, scene_format:str):
	file_names = glob.glob(os.path.join(pred_dir, f"*_{scene_format}_{nshots}shot_.jsonl"))

	pass

def evaluate(pred_dir:str, accept_mention_correct:bool = False):
	format_results = build_results(pred_dir, accept_mention_correct)
	hue_order = [
		"llava-hf/llava-1.5-7b-hf",
		"llava-hf/llava-1.5-13b-hf",
		"llava-hf/llava-v1.6-mistral-7b-hf",
		"llava-hf/llava-v1.6-vicuna-7b-hf",
		"llava-hf/llava-v1.6-vicuna-13b-hf",
		"llava-hf/llava-v1.6-34b-hf",
		"llava-hf/llama3-llava-next-8b-hf",
		"allenai/MolmoE-1B-0924",
		"allenai/Molmo-7B-O-0924",
		"allenai/Molmo-7B-D-0924",
		"unsloth/Llama-3.2-11B-Vision-Instruct",
		"Qwen/Qwen2-VL-2B-Instruct",
		"Qwen/Qwen2-VL-7B-Instruct",
		"Qwen/Qwen2.5-VL-7B-Instruct"
	]
	for scene_format, nshots_results in format_results.items():
		for nshots, results in nshots_results.items():
			tasks = [key for key in list(results.values())[0].keys() if '_' not in key]
			task_table = {'model': []} | {task: [] for task in tasks}
			object_count_tables = defaultdict(lambda :{'model': [], 'object_count': [], 'score':[]})
			for model_name in hue_order:
				model_results = results[model_name]
			#for model_name, model_results in results.items():
				task_table['model'].append(model_name)
				#object_count_table['model'].append(model_name)
				for obj_count_key in [k for k in list(model_results.keys()) if 'object_count' in k]:
					object_count = model_results.pop(obj_count_key)
					object_count_table = object_count_tables[obj_count_key]
					for key, value in object_count.items():
						object_count_table['model'].append(model_name)
						object_count_table['object_count'].append(int(key))
						object_count_table['score'].append(value)
				cols = set(task_table.keys())
				cols.remove('model')
				for col in cols:
					task_table[col].append(model_results[col])
			add_task_chance(pred_dir, task_table, nshots, scene_format)
			# Print task_table
			df_table = pd.DataFrame(data=task_table)
			print(df_table.to_markdown())

			# Plot object count
			for obj_count_type, object_count_table in object_count_tables.items():
				sns.barplot(object_count_table, x="object_count", y="score", hue="model", hue_order=hue_order)
				# Place the legend outside the figure
				modality = "mm" if "_mm" in obj_count_type else "um"
				plt.legend(loc='upper left', bbox_to_anchor=(1, 1), title='Model')
				plt.title(f"{modality}: {str(scene_format).upper()} - {nshots}-shot")
				plt.gcf().set_size_inches(8, 5)  # Wider to accommodate the legend
				plt.tight_layout()  # Prevent layout issues

				plt.show()

def print_main_table(pred_dir:str, accept_mention_correct:bool = False, format:str="json", shots:int=2):
	hue_order = {
		"llava-hf/llava-1.5-7b-hf": "LLaVA 1.5 7B",
		"llava-hf/llava-1.5-13b-hf": "LLaVA 1.5 13B",
		#"llava-hf/llava-v1.6-mistral-7b-hf": "LLaVA 1.6 mistral 7B",
		"llava-hf/llava-v1.6-vicuna-7b-hf": "LLaVA 1.6 7B",
		"llava-hf/llava-v1.6-vicuna-13b-hf": "LLaVA 1.6 13B",
		"llava-hf/llava-v1.6-34b-hf": "LLaVA 1.6 34B",
		"llava-hf/llama3-llava-next-8b-hf": "LLaVA 1.6 llama3 8B",
		"allenai/MolmoE-1B-0924": "Molmo 1B",
		#"allenai/Molmo-7B-O-0924": "Molmo 7B O",
		"allenai/Molmo-7B-D-0924": "Molmo 7B",
		"unsloth/Llama-3.2-11B-Vision-Instruct": "Llama 3.2 11B",
		"Qwen/Qwen2-VL-2B-Instruct": "Qwen2-VL-2B",
		"Qwen/Qwen2-VL-7B-Instruct": "Qwen2-VL-7B",
		"Qwen/Qwen2-VL-72B-Instruct": "Qwen2-VL-72B",
		"Qwen/Qwen2.5-VL-3B-Instruct": "Qwen2.5-VL-3B",
		"Qwen/Qwen2.5-VL-7B-Instruct": "Qwen2.5-VL-7B",
		"TIGER-Lab/VL-Rethinker-7B": "Rethinker-7B",
		"gpt-4o-2024-11-20": "GPT-4o",
		"claude-3-5-sonnet-20241022": "Claude 3.5",
		"gemini-1.5-flash": "Gemini 1.5",
		"random": "Random"
	}
	results = build_results(pred_dir, accept_mention_correct)
	task_table = {'model': [], 'img2img': [], 'data2data':[], 'img2data':[], 'data2img':[]}
	for model_name in list(hue_order.keys()):
		task_table['model'].append(hue_order[model_name])
		cols = set(task_table.keys())
		cols.remove('model')
		for col in cols:
			task_table[col].append(results[format][shots][model_name][col] if col in results[format][shots][model_name] else '-')
	df_table = pd.DataFrame(data=task_table)
	df_table[['img2img', 'data2data', 'img2data', 'data2img']] = df_table[['img2img', 'data2data', 'img2data', 'data2img']].replace('-', 0).apply(pd.to_numeric)
	df_table['avg_mm'] = df_table[['img2data', 'data2img']].mean(axis=1)
	df_table['avg_um'] = df_table[['img2img', 'data2data']].mean(axis=1)
	print(df_table.to_markdown())
	print(df_table.to_latex(float_format="%.1f", index=False))

def print_all_models_table(pred_dir:str, accept_mention_correct:bool = False, format:str="json", shots:list=None, include_avgs:bool = False, tasks:list=None):
	shots = [2] if shots is None else shots
	hue_order = {
		"llava-hf/llava-1.5-7b-hf": "llava-1.5-7b",
		"llava-hf/llava-1.5-13b-hf": "llava-1.5-13b",
		"llava-hf/llava-v1.6-mistral-7b-hf": "llava-v1.6-mistral-7b",
		"llava-hf/llava-v1.6-vicuna-7b-hf": "llava-v1.6-vicun",
		"llava-hf/llava-v1.6-vicuna-13b-hf": "llava-v1.6-vicuna-13b",
		"llava-hf/llava-v1.6-34b-hf": "llava-v1.6-yi-34b",
		"llava-hf/llama3-llava-next-8b-hf": "llama3-llava-next-8b",
		"allenai/MolmoE-1B-0924": "MolmoE-1B-0924",
		"allenai/Molmo-7B-O-0924": "Molmo-7B-O-0924",
		"allenai/Molmo-7B-D-0924": "Molmo-7B-D-0924",
		"unsloth/Llama-3.2-11B-Vision-Instruct": "Llama-3.2-11B-Vision-Instruct",
		"Qwen/Qwen2-VL-2B-Instruct": "Qwen2-VL-2B-Instruct",
		"Qwen/Qwen2-VL-7B-Instruct": "Qwen2-VL-7B-Instruct",
		"Qwen/Qwen2.5-VL-7B-Instruct": "Qwen2.5-VL-7B-Instruct",
		"gpt-4o-2024-11-20": "gpt-4o-2024-11-20",
		"claude-3-5-sonnet-20241022": "claude-3-5-sonnet-20241022",
		"gemini-1.5-flash": "gemini-1.5-flash"
	}
	results = build_results(pred_dir, accept_mention_correct)
	task_table = {'model': []}#, 'img2img': [], 'data2data':[], 'img2data':[], 'data2img':[]}
	col_factory = {'i2i': "img2img",
				   'd2d': "data2data",
				   'i2d': "img2data",
				   'd2i': "data2img"}
	if tasks is not None:
		col_factory = {k:col_factory[k] for k in tasks}
	for col_n in col_factory.keys():
		for shot in shots:
			task_table[f"{shot}_{col_n}"] = []
	cols = set(task_table.keys())
	cols.remove('model')
	for model_name in list(hue_order.keys()):
		task_table['model'].append(hue_order[model_name])
		for col in cols:
			shot, task = col.split('_')
			task = col_factory[task]
			task_table[col].append(results[format][int(shot)][model_name][task] if task in results[format][int(shot)][model_name] else '-')
	df_table = pd.DataFrame(data=task_table)
	if include_avgs:
		df_table[cols] = df_table[cols].replace('-', 0).apply(pd.to_numeric)
		df_table['avg_mm'] = df_table[[c for c in cols if len(set(c.split("_")[-1].split("2"))) == 1]].mean(axis=1)
		df_table['avg_um'] = df_table[[c for c in cols if len(set(c.split("_")[-1].split("2"))) > 1]].mean(axis=1)
	print(df_table.to_markdown())
	print(df_table.to_latex(float_format="%.1f", index=False))

def print_cot_table(pred_dir:str, accept_mention_correct:bool = False, format:str="json", shots:int=2, colums:list = None, add_ose = False, add_delta_to_cols:bool = False):
	hue_order = {
		#"llava-hf/llava-1.5-7b-hf": "LLaVA 1.5 7B",
		"llava-hf/llava-1.5-13b-hf": "LLaVA 1.5",
		#"llava-hf/llava-v1.6-mistral-7b-hf": "LLaVA 1.6 mistral 7B",
		#"llava-hf/llava-v1.6-vicuna-7b-hf": "LLaVA 1.6 7B",
		#"llava-hf/llava-v1.6-vicuna-13b-hf": "LLaVA 1.6 13B",
		"llava-hf/llava-v1.6-34b-hf": "LLaVA 1.6",
		#"llava-hf/llama3-llava-next-8b-hf": "LLaVA 1.6 llama3 8B",
		#"allenai/MolmoE-1B-0924": "Molmo 1B",
		#"allenai/Molmo-7B-O-0924": "Molmo 7B O",
		"allenai/Molmo-7B-D-0924": "Molmo",
		"unsloth/Llama-3.2-11B-Vision-Instruct": "Llama 3.2",
		#"Qwen/Qwen2-VL-2B-Instruct": "Qwen2-VL-2B",
		"Qwen/Qwen2-VL-7B-Instruct": "Qwen2-VL",
		"Qwen/Qwen2.5-VL-7B-Instruct": "Qwen2.5-VL",
		"TIGER-Lab/VL-Rethinker-7B": "Rethinker-7B",
		#"Qwen/Qwen2-VL-72B-Instruct": "Qwen2-VL-72B",
		"gpt-4o-2024-11-20": "GPT-4o",
		"claude-3-5-sonnet-20241022": "Claude 3.5",
		"gemini-1.5-flash": "Gemini 1.5",
		"human": "Human",
		"random": "Random"
	}
	results = build_results(pred_dir, accept_mention_correct)
	results_cot = build_results(pred_dir, accept_mention_correct, setting="cot")
	if add_ose:
		error_df = errors_in_data(pred_dir)
	task_table = defaultdict(list)
	for model_name in list(hue_order.keys()):
		task_table['model'].append(hue_order[model_name])
		cols = set(task_table.keys())
		cols.remove('model')
		avg_org = np.mean([a for a in results[format][shots][model_name]['object_count_mm'].values()])

		if model_name not in results_cot[format][shots]:
			task_table["all"].append(f"{avg_org:.1f}")
			for col in colums:
				task_table[col].append(results[format][shots][model_name]['object_count_mm'][col])
			if add_ose:
				task_table["OSE"].append("0")
			continue

		avg_cot = np.mean([a for a in results_cot[format][shots][model_name]['object_count_mm'].values()])
		delta = avg_cot - avg_org
		cmd = "deltaneg" if delta < 0 else "deltapos"
		all_avg = f"{avg_cot:.1f} \\{cmd}{{{np.abs(delta):.1f}}}"
		task_table["all"].append(all_avg)
		for col in colums:
			res_cot = results_cot[format][shots][model_name]['object_count_mm'][col]
			if not add_delta_to_cols:
				task_table[col].append(res_cot)
			else:
				res_org = results[format][shots][model_name]['object_count_mm'][col]
				res_delta = res_cot - res_org
				res_cmd = "deltaneg" if res_delta < 0 else "deltapos"
				task_table[col].append(f"{res_cot:.1f} \\{res_cmd}{{{np.abs(res_delta):.1f}}}")
		if add_ose:
			soe_org = 100 - error_df[(error_df['model'] == model_name) & (error_df['setting'] == 'org') & (error_df['modality'] == 'mm')].iloc[0]['pred_in_data']
			soe_cot = 100 - error_df[(error_df['model'] == model_name) & (error_df['setting'] == 'cot') & (error_df['modality'] == 'mm')].iloc[0]['pred_in_data']
			soe_delta = soe_cot - soe_org
			soe_cmd = "deltaneggreen" if soe_delta < 0 else "deltaposred"
			task_table["OSE"].append(f"{soe_cot:.1f} \\{soe_cmd}{{{np.abs(soe_delta):.1f}}}")
	df_table = pd.DataFrame(data=dict(task_table))
	print(df_table.to_markdown())
	print(df_table.to_latex(float_format="%.1f", index=False))

def print_attribute_table(pred_dir:str, accept_mention_correct:bool = False, format:str="json", shots:int=2, modalities:list=["mm"], attribute_fncs:list=["target_attribute"]):
	hue_order = {
		"llava-hf/llava-1.5-7b-hf": "LLaVA 1.5 7B",
		"llava-hf/llava-1.5-13b-hf": "LLaVA 1.5 13B",
		#"llava-hf/llava-v1.6-mistral-7b-hf": "LLaVA 1.6 mistral 7B",
		"llava-hf/llava-v1.6-vicuna-7b-hf": "LLaVA 1.6 7B",
		"llava-hf/llava-v1.6-vicuna-13b-hf": "LLaVA 1.6 13B",
		"llava-hf/llava-v1.6-34b-hf": "LLaVA 1.6 34B",
		"llava-hf/llama3-llava-next-8b-hf": "LLaVA 1.6 llama3 8B",
		"allenai/MolmoE-1B-0924": "Molmo 1B",
		#"allenai/Molmo-7B-O-0924": "Molmo 7B O",
		"allenai/Molmo-7B-D-0924": "Molmo 7B",
		"unsloth/Llama-3.2-11B-Vision-Instruct": "Llama 3.2 11B",
		"Qwen/Qwen2-VL-2B-Instruct": "Qwen2-VL-2B",
		"Qwen/Qwen2-VL-7B-Instruct": "Qwen2-VL-7B",
		"Qwen/Qwen2.5-VL-7B-Instruct": "Qwen2.5-VL",
		#"Qwen/Qwen2-VL-72B-Instruct": "Qwen2-VL-72B",
		"gpt-4o-2024-11-20": "GPT-4o",
		"claude-3-5-sonnet-20241022": "Claude 3.5",
		"gemini-1.5-flash": "Gemini 1.5",
		"human": "Human",
		"random": "Random"
	}
	results = build_results(pred_dir, accept_mention_correct)
	task_table = {'model': [], 'shape': [], 'color':[], 'name':[], '3d_coords':[], 'rotation':[], 'size':[], 'attribute_fnc': [], 'modality':[]}
	for model_name in list(hue_order.keys()):
		#task_table['model'].append(hue_order[model_name])
		cols = set(task_table.keys())
		cols.remove('model')
		cols.remove('attribute_fnc')
		cols.remove('modality')
		for attr_type_name in ["target_attribute_um", "pointer_attribute_um", "target_attribute_mm", "pointer_attribute_mm"]:
			tgt_subset = results[format][shots][model_name][attr_type_name]
			for task, attr_subset in tgt_subset.items():
				for col in cols:
					task_table[col].append(results[format][shots][model_name][attr_type_name][task][col] if col in results[format][shots][model_name][attr_type_name][task] else None)
				task_table['attribute_fnc'].append(attr_type_name.replace("_mm","").replace("_um", ""))
				task_table['model'].append(hue_order[model_name])
				task_table['modality'].append(attr_type_name.split("_")[-1])
	df_table = pd.DataFrame(data=task_table)
	#df_table = df_table.sort_values(['attribute_fnc', 'model'])
	df_table = df_table.groupby(['model', 'attribute_fnc', 'modality'], dropna=True).sum()
	df_table = df_table.sort_values(['modality','attribute_fnc', 'model'])
	df_table = df_table.reset_index()
	df_table = df_table[(df_table["modality"].isin(modalities)) & (df_table["attribute_fnc"].isin(attribute_fncs))]
	df_table.drop(['attribute_fnc', 'modality'], axis=1, inplace=True)
	print(df_table.to_markdown())
	print(df_table.to_latex(float_format="%.1f", index=False))

def build_confusion_matrix_df(file_name, accept_mention_correct:bool = False):
	dataset = load_jsonl(file_name)
	attributes = ['shape', 'color', 'name', '3d_coords', 'rotation', 'size']
	example_matrix = [[[] for _ in range(len(attributes))] for _ in range(len(attributes))]
	confusion_matrix = [[0 for _ in range(len(attributes))] for _ in range(len(attributes))]
	for example in dataset:
		tgt_idx = attributes.index(list(example['target_attribute'].keys())[0])
		pnt_idx = attributes.index(list(example['pointer_attribute'].keys())[0])
		example_matrix[tgt_idx][pnt_idx].append(example)

	for tgt_idx, tgt_row in enumerate(example_matrix):
		for pnt_idx, ex_group in enumerate(tgt_row):
			confusion_matrix[tgt_idx][pnt_idx] = calc_acc(ex_group, accept_mention_correct)

	df_cm = pd.DataFrame(confusion_matrix, index=attributes.copy(),
						 columns=attributes.copy())
	return df_cm

def attribute_heatmap(file_name:str, accept_mention_correct:bool = False, format:str="json", shots:int=2, combine_modalities:bool = True):
	df_cm = build_confusion_matrix_df(file_name, accept_mention_correct)
	if combine_modalities:
		df_cm = df_cm.combine(build_confusion_matrix_df(file_name.replace("mm_", "um_"), accept_mention_correct), lambda s1, s2: s1.fillna(0) + s2.fillna(0))
		#df_cm = df_cm + build_confusion_matrix_df(file_name.replace("mm_", "um_"), accept_mention_correct)

	df_cm = df_cm.replace(0, np.nan)
	plt.figure(figsize=(10, 7))
	#sns.heatmap(df_cm, annot=True)
	sns.heatmap(df_cm, annot=True, annot_kws={"size": 16}, fmt=".0f", cmap=sns.color_palette("ch:start=.2,rot=-.3", as_cmap=True).reversed())#, vmin=20, vmax=100)  # Adjust annotation font size
	#plt.xlabel("Pointer Attributes", fontsize=14)
	#plt.ylabel("Target Attributes", fontsize=14)
	plt.xticks(fontsize=12)  # Increase font size of x-axis tick labels
	plt.yticks(fontsize=12)  # Increase font size of y-axis tick labels
	#plt.title(f"Confusion Matrix Qwen2-VL-7B", fontsize=16)
	plt.show()


# Shades of gray
GREY10 = "#1a1a1a"
GREY30 = "#4d4d4d"
GREY40 = "#666666"
GREY50 = "#7f7f7f"
GREY60 = "#999999"
GREY75 = "#bfbfbf"
GREY91 = "#e8e8e8"
GREY98 = "#fafafa"
WHITE = "#ffffff"


def hex_to_rgba(hex_color, alpha=None):
	"""Convert HEX color to RGBA, handling both #RRGGBB and #RRGGBBAA formats.

	Args:
		hex_color (str): The HEX color code (with or without alpha).
		alpha (float, optional): Override the alpha value (0.0 to 1.0).

	Returns:
		tuple: (R, G, B, A) with values between 0 and 1.
	"""
	hex_color = hex_color.lstrip('#')

	if len(hex_color) == 8:  # #RRGGBBAA format
		r, g, b, a = (int(hex_color[i:i + 2], 16) / 255.0 for i in (0, 2, 4, 6))
		return (r, g, b, a if alpha is None else alpha)

	elif len(hex_color) == 6:  # #RRGGBB format
		r, g, b = (int(hex_color[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
		return (r, g, b, 1.0 if alpha is None else alpha)

	else:
		raise ValueError("Invalid HEX color format. Use #RRGGBB or #RRGGBBAA.")

def object_figure(pred_dir:str, accept_mention_correct:bool = False, format:str="json", shots:int=2,
				  barplot_models:list=None, line_models:list=None, barplot_modality="mm", line_modality="um", human_line_modality="mm",
				  highlighted_models: list = None):
	barplot_models = [] if barplot_models is None else barplot_models
	highlighted_models = [] if highlighted_models is None else highlighted_models
	line_models = [] if line_models is None else line_models
	if not os.path.isfile("out/results.pk"):
		results = build_results(pred_dir, accept_mention_correct)
		process_human_eval_files(pred_dir=os.path.join(pred_dir, "human"), out_pred_dir=pred_dir)
		with open("out/results.pk", 'wb') as handle:
			def defaultdict_to_dict(d):
				if isinstance(d, defaultdict):
					return {key: defaultdict_to_dict(value) for key, value in d.items()}
				return d
			pickle.dump(defaultdict_to_dict(results), handle, protocol=pickle.HIGHEST_PROTOCOL)
	else:
		with open("out/results.pk", 'rb') as handle:
			results = pickle.load(handle)
	model_plot_info = {
		"llava-hf/llava-1.5-7b-hf": {"name": "LLaVA 1.5", "color": ""},
		"llava-hf/llava-1.5-13b-hf": {"name": "LLaVA 1.5", "color": "#924B9D"},#"#106C61"},
		"llava-hf/llava-v1.6-mistral-7b-hf": {"name": "LLaVA 1.6 mistral 7B", "color": "#106C61"},
		"llava-hf/llava-v1.6-vicuna-7b-hf": {"name": "LLaVA 1.6 7B", "color": ""},
		"llava-hf/llava-v1.6-vicuna-13b-hf": {"name": "LLaVA 1.6 13B", "color": "#3284B6"},
		"llava-hf/llava-v1.6-34b-hf": {"name": "LLaVA 1.6", "color": "#D47F68"},
		"llava-hf/llama3-llava-next-8b-hf": {"name": "LLaVA 1.6 llama3 8B", "color": "#E73F74"},
		"allenai/MolmoE-1B-0924": {"name": "Molmo 1B", "color": ""},
		"allenai/Molmo-7B-O-0924": {"name": "Molmo", "color": "#BDBDBD"},
		"allenai/Molmo-7B-D-0924": {"name": "Molmo 7B D", "color": ""},
		"unsloth/Llama-3.2-11B-Vision-Instruct": {"name": "Llama 3.2", "color": "#8CA252"},#"#E68310"},
		"Qwen/Qwen2-VL-2B-Instruct": {"name": "Qwen2-VL-2B", "color": ""},
		"Qwen/Qwen2-VL-7B-Instruct": {"name": "Qwen2-VL", "color": "#FD8D3C"}, #5975A4"},
		"Qwen/Qwen2.5-VL-7B-Instruct": {"name": "Qwen2.5-VL", "color": "#FD8D3C"}, #5975A4"},
		"gpt-4o-2024-11-20": {"name": "GPT-4o", "color": "#6BAED6"}, #5975A4"},
		"claude-3-5-sonnet-20241022": {"name": "Claude 3.5", "color": "#5254A3"}, #5975A4"},
		"gemini-1.5-flash": {"name": "Gemini 1.5", "color": "#A089C2"}, #5975A4"},
		"human": {"name": "Human", "color": "#E32636"}, #E3263630
		"random": {"name": "Random", "color": "#999999"}
	}
	object_count_table = {'model': [], 'model_name': [], 'object_count': [], 'score': [], 'modality': []}
	for model_name in model_plot_info.keys():
		model_results = results[format][shots][model_name]
		for obj_count_key in [k for k in list(model_results.keys()) if 'object_count' in k]:
			modality = "mm" if "_mm" in obj_count_key else "um"
			object_count = model_results.pop(obj_count_key)
			for key, value in object_count.items():
				object_count_table['model'].append(model_name)
				object_count_table['model_name'].append(model_plot_info[model_name]["name"])
				object_count_table['object_count'].append(int(key))
				object_count_table['score'].append(value)
				object_count_table['modality'].append(modality)
	df = pd.DataFrame(data=object_count_table)

	x_offset = df['object_count'].min()
	x_max_value = df['object_count'].max()

	# Create the plot
	fig, ax = plt.subplots(figsize=(8, 6))

	# Background color
	fig.patch.set_facecolor(WHITE)
	ax.set_facecolor(WHITE)

	# Vertical lines every 1 object count
	for h in np.arange(0, 8, 1):
		ax.axvline(h, color=GREY91, lw=0.6, zorder=0)

	# Horizontal lines
	ax.hlines(y=np.arange(0, 105, 5), xmin=-0.5, xmax=x_max_value-x_offset+0.5, color=GREY91, lw=0.6, zorder=0)

	# Get barplot models
	df_barplot = df[(df['modality'] == barplot_modality) & (df['model'].isin(barplot_models))]
	hue_pallete = {key: hex_to_rgba(value["color"]) for key, value in model_plot_info.items() if key in barplot_models}
	sns.barplot(
		data=df_barplot,
		x="object_count",
		y="score",
		hue="model",
		palette=hue_pallete,
		hue_order=list(hue_pallete.keys()),
		ax=ax,
		width = 0.9
	)
	# Manually set alpha for bars with RGBA colors
	for patch, hue_category in zip(ax.patches, df_barplot['model']):
		patch.set_alpha(hue_pallete[hue_category][3])  # Apply stored alpha

	# Labels
	# First, adjust axes limits so annotations fit in the plot
	ax.set_xlim(-0.5, 8.5)
	ax.set_ylim(0, 100.5)

	# Customize axes labels and ticks
	ax.set_yticks([y for y in np.arange(0, 110, 10)])
	ax.set_yticklabels(
		[f"{y}%" for y in np.arange(0, 110, 10)],
		fontname="Montserrat",
		fontsize=14,
		weight=500,
		color=GREY40
	)

	ax.set_xticks([x for x in np.arange(0, x_max_value-x_offset+1, 1)])
	ax.set_xticklabels(
		[f"{x+x_offset}" for x in np.arange(0, x_max_value-x_offset+1, 1)],
		fontname="Montserrat",
		fontsize=16,
		weight=500,
		color=GREY40
	)

	# Increase size and change color of axes ticks
	ax.tick_params(axis="x", length=12, color=GREY91)
	ax.tick_params(axis="y", length=x_max_value-x_offset, color=GREY91)

	# Customize spines
	ax.spines["left"].set_color(GREY91)
	ax.spines["bottom"].set_color(GREY91)
	ax.spines["right"].set_color("none")
	ax.spines["top"].set_color("none")

	# Update legend with model names
	handles, labels = ax.get_legend_handles_labels()
	updated_labels = [model_plot_info[label]["name"] for label in labels]
	ax.legend_.remove()  # Remove default legend
	# Shrink current axis's height by 10% on the bottom
	box = ax.get_position()
	ax.set_position([box.x0, box.y0 - box.height * 0.26,
					 box.width, box.height * 1.2])

	# Put a legend below current axis
	fig.legend(
		handles, updated_labels,
		title=None,
		loc='upper center',
		bbox_to_anchor=(0.51, 0.21),
		ncol=3,
		prop={'size': 15}
	)

	plt.tight_layout()

	df = df.sort_values(by=['model', 'object_count'])
	df['object_count'] = df['object_count'] - x_offset

	# Lines
	if len(line_models) > 0:
		# Human is a special case, we remove it form lines and add it from its modality if it's in highlighted
		df_lines = df[(df['modality'] == line_modality) & (df['model'].isin(line_models)) & (df['model'] != 'human')]

		if "human" in line_models:
			df_line_human = df[(df['modality'] == human_line_modality) & (df["model"].isin(["human"]))]
			df_lines = pd.concat([df_lines, df_line_human], ignore_index=True)

		df_highlight = df_lines[df_lines["model"].isin(highlighted_models)]
		df_others = df_lines[~df_lines["model"].isin(highlighted_models)]

		for group in df_others["model"].unique():
			data = df_others[df_others["model"] == group]
			ax.plot("object_count", "score", c=GREY75, lw=1.2, alpha=0.5, data=data)

		for idx, model in enumerate(df_highlight["model"].unique()):
			data = df_highlight[df_highlight["model"] == model]
			color = model_plot_info[model]["color"]
			line_style, lw = ('dashed', 1) if model == "Human" else ('solid', 1.8)
			ax.plot("object_count", "score", color=color, lw=lw, data=data, ls=line_style)

		# Labels
		x_start = x_max_value-x_offset
		x_end = x_max_value-x_offset+0.5
		PAD = 0.01

		# Add labels for highlighted countries honly
		for idx, model in enumerate(df_highlight["model"].unique()):
			data = df_highlight[(df_highlight["model"] == model) & (df_highlight["object_count"] == x_max_value-x_offset)]
			color = model_plot_info[model]["color"]

			# Highlight label
			text = data['model_name'].values[0]
			if data['modality'].values[0] != barplot_modality:
				text += f"\nSingle M." if data['modality'].values[0] == "um" else f"\nCross-M."

			# Vertical start of line
			y_start = data["score"].values[0]
			# Vertical end of line
			y_offset = 0 if data['model_name'].values[0] == "Human" else 0
			y_end = data["score"].values[0] - y_offset

			# Add line based on three points
			ax.plot(
				[x_start, (x_start + x_end - PAD) / 2, x_end - PAD],
				[y_start, y_end, y_end],
				color=color,
				alpha=0.5,
				ls="dashed"
			)

			# Add text
			ax.text(
				x_end,
				y_end,
				text,
				color=color,
				fontsize=11,
				weight="bold",
				fontfamily="Montserrat",
				va="center"
			)

	# ---------
	# Customize the plot
	"""
	plt.title("Barplot of Score by Object Count and Model", fontsize=14)
	plt.xlabel("Object Count", fontsize=12)
	plt.ylabel("Score", fontsize=12)
	plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc='upper left')
	plt.tight_layout()
	"""
	plt.xlabel("Object Count", fontsize=16, fontfamily="Montserrat", color=GREY50, x=0.45)
	plt.ylabel("Accuracy", fontsize=16, fontfamily="Montserrat", color=GREY50)

	fig.savefig(f"out/figures/{barplot_modality}_obj_2_updated.pdf", format="pdf", bbox_inches='tight')

	# Show the plot
	plt.show()

def add_dict_to_image(image, obj:dict):
	# Text to add
	lines_of_text = [f"{str(key)}: {str(value)}" for key, value in obj.items()]

	# Settings for the new canvas
	padding = 20
	font_size = 20

	# Load a default font
	font = ImageFont.load_default(size=20)

	# Calculate the new image size using getbbox()
	text_width = max(font.getbbox(line)[2] for line in lines_of_text) + padding * 2
	text_height = sum(font.getbbox(line)[3] for line in lines_of_text) + padding * (len(lines_of_text) + 1)
	new_width = image.width + text_width
	new_height = max(image.height, text_height)

	# Create a new image with a white background
	new_image = Image.new("RGB", (new_width, new_height), color="white")

	# Paste the original image
	new_image.paste(image, (0, 0))

	# Add the text on the right side
	draw = ImageDraw.Draw(new_image)
	current_height = padding
	for line in lines_of_text:
		bbox = font.getbbox(line)
		draw.text((image.width + padding, current_height), line, fill="black", font=font)
		current_height += bbox[3] + padding

	return new_image

def evaluate_point(pred_dir: str, img_dir: str = "data/meta_correlation/v2/img", circle_radius: int = 3, example_count=50):
	file_names = glob.glob(os.path.join(pred_dir, "*.jsonl"))
	for file_name in file_names:
		predictions = load_jsonl(file_name)[:example_count]
		out_dir = os.path.join(pred_dir, os.path.splitext(os.path.basename(file_name))[0])
		os.makedirs(out_dir, exist_ok=True)
		for pred in predictions:
			img = Image.open(os.path.join(img_dir, pred["image"]))
			width, height = img.size
			points = extract_points(pred["prediction"], width, height)
			draw = ImageDraw.Draw(img)
			for point in points:
				bounding_box = [
					(point[0] - circle_radius, point[1] - circle_radius),
					(point[0] + circle_radius, point[1] + circle_radius)
				]
				draw.ellipse(bounding_box, fill="red", outline="white")
				obj = [o for o in pred["scene"]["objects"] if o["name"] == pred["target_name"]][0]
				img = add_dict_to_image(img, obj)
			img.save(os.path.join(out_dir, pred["image"]), format="PNG")

	# Manually evaluated results
	results = {
		"allenaiMolmoE_1B_0924": {
			"shape": [0,0,0,1,0,1,0,1,1,0,1,0,1,1,0,1,1,1,0,0,1,0,0,1,1],
			"material": [0,1,1,1,1,1,0,1,1,1,1,1,0,1,0,1,0,1,0,1,1,0,1,0,1],
			"color": [0,0,0,0,0,1,0,1,1,0,1,0,1,1,0,1,0,0,0,0,1,0,0,0,0],
			"3d_coords": [0,0,0,0,0,1,0,1,1,0,1,0,0,1,0,1,0,0,0,0,0,0,0,0,0],
			"object_count": [5,5,7,4,3,9,7,4,3,6,7,8,9,6,10,5,6,6,8,5,5,5,7,3,4],
		},
		"allenaiMolmo_7B_O_0924": {
			"shape": [1,1,0,1,1,0,1,1,1,0,0,0,0,1,0,1,1,1,0,1,1,0,1,1,1],
			"material": [1,1,1,1,1,0,1,1,1,0,0,1,1,1,0,1,0,1,1,1,1,0,1,1,1],
			"color": [1,1,0,1,1,0,1,1,1,0,0,0,0,1,0,1,0,1,0,0,1,0,1,1,1],
			"3d_coords": [5,5,7,4,3,9,7,4,3,6,7,8,9,6,10,5,6,6,8,5,5,5,7,3,4],
		},
		"allenaiMolmo_7B_D_0924": {
			"shape": 	[1,0,0,1,0,1,1,1,0,0,1,1,0,0,1,0,1,1,1,1,1,0,1,1,1],
			"material": [1,1,1,1,0,1,1,1,0,0,1,0,0,0,1,0,0,1,1,1,1,0,1,1,1],
			"color": [1,0,0,1,0,1,1,1,0,0,1,0,0,0,0,0,0,1,1,1,1,0,1,1,1],
			"3d_coords": [1,0,0,1,0,1,1,1,0,0,1,0,0,0,0,0,0,1,1,1,0,0,1,1,1],
		}
	}

def process_human_eval_files(pred_dir:str, out_pred_dir:str="data/mate/dev/predictions"):
	datasets = {
		"data2img": load_jsonl("data/mate/dev/mm_2shot.jsonl", indexed=True, id_key_name="example_id"),
		"img2data": load_jsonl("data/mate/dev/mm_2shot.jsonl", indexed=True, id_key_name="example_id"),
		"data2data": load_jsonl("data/mate/dev/um_2shot.jsonl", indexed=True, id_key_name="example_id"),
		"img2img": load_jsonl("data/mate/dev/um_2shot.jsonl", indexed=True, id_key_name="example_id"),
	}
	file_names = glob.glob(os.path.join(pred_dir, "*.txt"))
	results = {}
	all_examples = {"um":[], "mm":[]}
	for file_name in file_names:
		evaluator = os.path.basename(file_name).split("_")[0]
		if evaluator not in results:
			results[evaluator] = {key: [] for key in datasets.keys()}
		with open(file_name, 'r') as file:
			lines = file.readlines()

		id_lines, answers = [], []
		for i, line in enumerate(lines):
			if line == "-------------------------------------------------------\n":
				if lines[i - 1].startswith("##"):
					id_lines.append(lines[i - 1].split(" ##-- ")[-1].strip())
				else:
					answers.append(lines[i - 1].split("?")[-1].strip())
		for answer, data_example_id in zip(answers, id_lines):
			task, example_id = data_example_id.split(":")
			pnt_modality, tgt_modality = task.split("2")
			example = copy.deepcopy(datasets[task][example_id])
			#if evaluator == "inigo":
			#	print(json.loads(example["gold_reference"])["answer"])
			example["evaluator"] = hashlib.shake_256(evaluator.encode()).hexdigest(10)
			example["model"] = "human"
			answer = answer if answer[0] in ['"', '[', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] else f'"{answer}"'
			example["prediction"] = f'{{"answer": {answer}}}'
			results[evaluator][task].append(example)
			all_examples["um" if pnt_modality==tgt_modality else "mm"].append(example)

	for modality, examples in all_examples.items():
		save_jsonl(os.path.join(out_pred_dir, f"{modality}_2shot_human.jsonl"), examples)
	return results

def evaluate_human(pred_dir:str, accept_mention_correct:bool = False):
	results = process_human_eval_files(pred_dir)
	# Print
	total_datasets = defaultdict(list)
	for evaluator, dataset in results.items():
		print(f"{evaluator}")
		for dataset_name, examples in dataset.items():
			if len(examples) == 0:
				continue
			#scores = [1 if example["gold_reference"] == example["prediction"] else 0 for example in examples]
			scores = []
			for example in examples:
				scores.append(1 if example["gold_reference"] == example["prediction"] else 0)
			print(f"- {dataset_name}: {sum(scores)/len(scores)*100:.1f}%")
			total_datasets[dataset_name].extend(scores)
	total = []
	print("Total")
	for dataset_name, scores in total_datasets.items():
		print(f"- {dataset_name}: {sum(scores) / len(scores) * 100:.1f}%")
		total.extend(scores)
	print(f"TOTAL: {sum(total) / len(total) * 100:.1f}% ({len(total)})")


def get_least_key_attributes(example):
	linking_attributes = get_linking_attributes(example)
	return min([len(ids) for ids in linking_attributes], default=0)

def link_attr_groups(dataset, unique:bool=True):
	plot_2_results = defaultdict(dict)
	for lnk_attr_count, lnk_attr_count_dataset in dataset.items():
		single_lnk_attr_acc = defaultdict(list)
		if lnk_attr_count == 0:
			# We will add 3d_coords to 1 later
			continue
		for example in dataset[lnk_attr_count]:
			lnk_attr_all = get_linking_attributes(example)
			lnk_attrs = [attr for attr in lnk_attr_all if len(attr) == lnk_attr_count]
			# Discard if it has multiple 1 lnk_attrs
			if unique and len(lnk_attrs) > 1:
				continue
			for lnk_attr in lnk_attrs:
				lnk_attr_key = list(lnk_attr.keys())
				single_lnk_attr_acc[" | ".join(lnk_attr_key)].append(example)
		if lnk_attr_count == 1:
			single_lnk_attr_acc["3d_coords"] = dataset[0]
		plot_2_results[lnk_attr_count] = {key: calc_acc(data) for key, data in single_lnk_attr_acc.items()}
	return plot_2_results

def key_attr_plot(pred_path:str, add_stdev:bool = False):
	dataset = reindex_by(load_jsonl(pred_path), key_factory_fn=get_least_key_attributes)
	# Plot 1
	plot_1_results = {key: calc_acc(data) for key, data in dataset.items()}
	# Plot 2
	plot_2_results = link_attr_groups(dataset, unique=True)
	# Object count and linking attributes
	lnk_obj = {key: {k:v for k,v in reindex_by(val, "object_count").items()} for key, val in dataset.items()}
	lnk_obj_count = {key: {k:len(v) for k,v in val.items()} for key, val in lnk_obj.items()}

	plot_1_results = {key: calc_acc(data[7]) for key, data in lnk_obj.items()}

	# Plot
	# Sort data for the left plot in descending order
	plot_1_results = {str(key): value for key, value in plot_1_results.items()}
	plot_1_results["3dc"] = plot_1_results.pop("0")
	#sorted_left = dict(sorted(plot_1_results.items(), key=lambda item: item[1], reverse=True))
	left_x_order = ["1", "2", "3", "3dc"]
	sorted_left = {x: plot_1_results[x] for x in left_x_order}
	#sorted_left = dict(sorted(plot_1_results.items(), key=lambda item: item[1], reverse=True))

	fig, axes = plt.subplots(1, 2, figsize=(7, 5), sharey=True)

	#std_devs = {"1": 5.3, "2": 8.4, "3": 9.9}#, 'coords': 5.1}
	std_devs = {key: statistics.stdev([v for k, v in val.items() if k != '3d_coords']) for key, val in plot_2_results.items()}
	# Standard deviations

	# Left bar plot
	#axes[0].bar(["1", "2", "3", "coords"], sorted_left.values(), yerr=errors, color='royalblue')
	axes[0].bar(left_x_order, sorted_left.values(), color='#5975A4')
	if add_stdev:
		axes[0].errorbar(["1", "2", "3"], [v for k, v in sorted_left.items() if k != 'coords'], std_devs.values(), fmt='.', color='#344664', elinewidth=2, capthick=1, errorevery=1, alpha=1.0, ms=4, capsize=4)

	axes[0].set_xlabel("(a) Required lnk. attr. count", fontsize=16, fontfamily="Montserrat", color=GREY30, x=0.40, labelpad=10)
	axes[0].set_ylabel("Accuracy", fontsize=16, fontfamily="Montserrat", color=GREY50)
	apply_barplot_style(axes[0], left_x_order)

	# Right bar plot
	plot_2_results_filtered = dict(sorted(dict(plot_2_results[1]).items(), key=lambda item: item[1], reverse=True))
	plot_2_results_filtered = {key[0:3].replace("_", key[3]):val for key, val in plot_2_results_filtered.items()}
	axes[1].bar(plot_2_results_filtered.keys(), plot_2_results_filtered.values(), color='#D4A820')
	axes[1].set_xlabel("(b) Linking attribute type", fontsize=16, fontfamily="Montserrat", color=GREY30, x=0.50, labelpad=10)
	#axes[1].set_title("Linking attribute type", fontsize=16, fontfamily="Montserrat", color=GREY30, x=0.45)
	apply_barplot_style(axes[1], list(plot_2_results_filtered.keys()))

	plt.tight_layout()
	fig.savefig("out/figures/lnk_attr_updated.pdf", format="pdf", bbox_inches='tight')
	plt.show()


	object_count_table = {'lnk_attr': [], 'object_count': [], 'share': []}
	for lnk_attr, obj_count_set in lnk_obj_count.items():
		total = sum(obj_count_set.values())
		object_count_table['lnk_attr'].extend([lnk_attr]*len(obj_count_set))
		object_count_table['object_count'].extend(list(obj_count_set.keys()))
		object_count_table['share'].extend([v/total*100 for v in obj_count_set.values()])
	df = pd.DataFrame(data=object_count_table)

	x_offset = df['object_count'].min()
	x_max_value = df['object_count'].max()

	# Create the plot
	fig, ax = plt.subplots(figsize=(8, 6))

	# Background color
	fig.patch.set_facecolor(WHITE)
	ax.set_facecolor(WHITE)

	# Vertical lines every 1 object count
	for h in np.arange(0, 8, 1):
		ax.axvline(h, color=GREY91, lw=0.6, zorder=0)

	# Horizontal lines
	ax.hlines(y=np.arange(0, 105, 5), xmin=-0.5, xmax=x_max_value - x_offset + 0.5, color=GREY91, lw=0.6, zorder=0)

	# Get barplot models
	hue_pallete = {0: "#003f5c", 1: "#58508d", 2:"#ffa600", 3:"#ff6361"}
	sns.barplot(
		data=df,
		x="object_count",
		y="share",
		hue="lnk_attr",
		palette=hue_pallete,
		hue_order=list(hue_pallete.keys()),
		ax=ax
	)
	plt.show()

	print(":D")

def apply_barplot_style(ax,  x_labels:list):
	# Vertical lines every 1 object count
	for h in x_labels:
		ax.axvline(h, color=GREY91, lw=0.6, zorder=0)

	# Horizontal lines
	ax.hlines(y=np.arange(0, 105, 5), xmin=-0.7, xmax=len(x_labels)-0.5, color=GREY91, lw=0.6, zorder=0)

	# Labels
	# First, adjust axes limits so annotations fit in the plot
	ax.set_xlim(-0.7, len(x_labels))
	ax.set_ylim(0, 100.5)

	# Customize axes labels and ticks
	ax.set_yticks([y for y in np.arange(0, 110, 10)])
	ax.set_yticklabels(
		[f"{y}%" for y in np.arange(0, 110, 10)],
		fontname="Montserrat",
		fontsize=14,
		weight=500,
		color=GREY40
	)

	ax.set_xticks(x_labels)
	ax.set_xticklabels(
		[f"{x}" for x in x_labels],
		fontname="Montserrat",
		fontsize=16,
		weight=500,
		color=GREY40
	)

	# Increase size and change color of axes ticks
	ax.tick_params(axis="x", length=len(x_labels), color=GREY91)
	#ax.tick_params(axis="y", length=x_max_value - x_offset, color=GREY91)

	# Customize spines
	ax.spines["left"].set_color(GREY91)
	ax.spines["bottom"].set_color(GREY91)
	ax.spines["right"].set_color("none")
	ax.spines["top"].set_color("none")

def errors_in_data(pred_dir:str):
	if ".jsonl" in pred_dir.split("/")[-1]:
		file_names = [pred_dir]
	else:
		file_names = glob.glob(os.path.join(pred_dir, "*.jsonl"))
	results = {"model":[], "setting":[], "modality":[], "total_errors":[], "pred_in_data":[], 'color': [], 'shape': [], 'name': [], 'size': [], 'rotation': [], '3d_coords': []}
	for file_name in file_names:
		if "2shot" not in file_name:
			continue
		dataset = load_jsonl(file_name)
		model_name = dataset[0]["model"]
		setting = "cot" if "_cot_" in file_name else "org"
		modality = "mm" if "mm_" in os.path.basename(file_name) else "um"
		total = []
		fail_attr = []
		for ex in dataset:
			tgt_attr = list(ex["target_attribute"].keys())[0]
			pred_json, is_json = clean_prediction(ex["prediction"])
			if is_json:
				str_pred = str(pred_json["answer"]).strip()
				result = 1 if str(json.loads(ex["gold_reference"])["answer"]) == str_pred else 0
				if result == 0:
					exists = 1 if sum([1 for obj in ex["scene"]["objects"] if str(obj[tgt_attr]) == str_pred]) > 0 else 0
					if exists == 0:
						fail_attr.append(tgt_attr)
					total.append(exists)
			else:
				total.append(0)
		results["model"].append(model_name)
		results["modality"].append(modality)
		results["setting"].append(setting)
		results["total_errors"].append(len(total))
		results["pred_in_data"].append(round(np.mean(total) * 100, 1))
		cnt = Counter(fail_attr)
		for k in ['color', 'shape', 'name', 'size', 'rotation', '3d_coords']:
			results[k].append(cnt.get(k, 0))
		#print(f"{model_name}: {len(total)} -> {round(np.mean(total) * 100, 1)} | {Counter(fail_attr)}")
	df_table = pd.DataFrame(data=results)
	df_table = df_table.sort_values(by=["setting", "modality", "model"])
	print(df_table.to_markdown())
	return df_table

def euclidean_distance(coord1, coord2, axis=None):
	if axis is not None:
		return abs(coord1[axis] - coord2[axis])
	return math.sqrt(sum((a - b) ** 2 for a, b in zip(coord1, coord2)))


def order_by_distance(objects, reference_name, axis=None):
	gold_attr_name = next(iter(reference_name))
	gold_attr_value = reference_name[gold_attr_name]
	# Find the reference object
	ref_obj = next((obj for obj in objects if obj[gold_attr_name] == gold_attr_value), None)
	if ref_obj is None:
		raise ValueError(f"Object with name '{reference_name}' not found.")

	ref_coords = ref_obj["3d_coords"]

	# Compute distances and sort
	ordered = sorted(
		(obj for obj in objects if obj[gold_attr_name] != gold_attr_value),
		key=lambda obj: euclidean_distance(ref_coords, obj["3d_coords"], axis)
	)

	return ordered

def object_distance(obj1, obj2, axis=None):
	coords1 = obj1["3d_coords"]
	coords2 = obj2["3d_coords"]
	return euclidean_distance(coords1, coords2, axis)

def trans_axis(axis):
	match axis:
		case None:
			return 'all'
		case 0:
			return 'depth'
		case 1:
			return 'horizontal'
		case 2:
			return 'vertical'
		case _:
			return None

def analyse_3dc_lnk_attr(pred_path:str, ablate_coord_axis:list=None):
	if ablate_coord_axis is None:
		ablate_coord_axis = [None, 0, 1]
	dataset = reindex_by(load_jsonl(pred_path), key_factory_fn=get_least_key_attributes)
	coords_only_examples_fail = [ex for ex in dataset[0] if calc_acc([ex]) == 0]
	#coords_only_examples_fail = dataset[0]
	#coords_only_examples_fail = reindex_by(coords_only_examples_fail, "object_count")[10]
	results = {trans_axis(ax):[] for ax in ablate_coord_axis}
	all_dist = {trans_axis(ax):[] for ax in ablate_coord_axis}
	overlapping_dist = defaultdict(int)
	distance_overlap_group = {trans_axis(ax):defaultdict(list) for ax in ablate_coord_axis}
	for example in coords_only_examples_fail:
		pnt_attr = example["pointer_attribute"]
		pnt_att_name = next(iter(pnt_attr))
		pnt_att_value = pnt_attr[pnt_att_name]
		tgt_attr_name = next(iter(example["target_attribute"]))
		# Get predicted object
		pred_json, is_json = clean_prediction(example["prediction"])
		if not is_json or "answer" not in pred_json or next((obj for obj in example["scene"]["objects"] if obj[tgt_attr_name] == pred_json["answer"]), None) is None:
			# If prediction is not JSON or the predicted JSON doesn't point at an existing object
			continue

		# Order objects by euclidean distance from gold object
		ordered_objects = order_by_distance(example["scene"]["objects"], reference_name=pnt_attr)
		#pred_obj_idx = next((i+1 for i, obj in enumerate(ordered_objects) if obj[tgt_attr_name] == pred_json["answer"]), None)
		#results[trans_axis(axis)].append(pred_obj_idx / len(ordered_objects))
		pred_obj = next((obj for i, obj in enumerate(ordered_objects) if obj[tgt_attr_name] == pred_json["answer"]), None)
		gold_obj = next((obj for obj in example["scene"]["objects"] if obj[pnt_att_name] == pnt_att_value), None)

		# We check whether the predicted object matches all linking attributes (was the other object matching the linking attributes)

		ordered_objs_lnk_attrs = copy.deepcopy(ordered_objects)
		ordered_objs_lnk_attrs = [remove_nonlinking_attributes(obj, example, skip_coords=False) for obj in ordered_objs_lnk_attrs]
		gold_obj_lnk_attrs = remove_nonlinking_attributes(copy.deepcopy(gold_obj), example, skip_coords=False)
		overlap_obj_dict = group_by_overlap(ordered_objs_lnk_attrs, gold_obj_lnk_attrs)
		pred_obj_lnk_attrs = remove_nonlinking_attributes(copy.deepcopy(pred_obj), example, skip_coords=False)

		# Record distribution of overlapping attrs for predicted objects
		for overlap_count, objs in overlap_obj_dict.items():
			for obj in objs:
				if obj == pred_obj_lnk_attrs:
					overlapping_dist[overlap_count] += 1
					for axis in ablate_coord_axis:
						dist_pred = euclidean_distance(obj["3d_coords"], gold_obj["3d_coords"], axis)
						avg_dist = np.average([object_distance(gold_obj, obj, axis) for obj in objs])
						distance_overlap_group[trans_axis(axis)][overlap_count].append(dist_pred<=avg_dist)
		print("")
		"""
		# We assume that the object that overlaps the most completely overlaps with gold as this is a 3dc only lnk_attr
		fully_overlap_objs = overlap_obj_dict[max(overlap_obj_dict.keys().list())]
		if len(fully_overlap_objs) == 1:
			# There is only one fully-overlapping object
			if pred_obj_lnk_attrs == fully_overlap_objs[0]:
				nuanced_results["fully_overlapping_single"] += 1
			if fully_overlap_objs[0] == remove_nonlinking_attributes(ordered_objects[0], example, skip_coords=False):
				nuanced_results["fully_overlapping_single_closest"] += 1
			pass
		else:
			# There's more than one fully overlapping object
			pass
		# TODO let's see what unique identifiers are the check if pred_obj's unique identifier's value == gold
		"""
		"""
		results[trans_axis(axis)].append(object_distance(gold_obj, pred_obj, axis) if calc_acc([example]) == 0 else 0.0)
		#all_dist[trans_axis(axis)].extend([object_distance(gold_obj, obj, axis) for obj in ordered_objects])
		#random_objects = [random.choice(ordered_objects) for _ in range(1)]
		#all_dist[trans_axis(axis)].append(np.average([object_distance(gold_obj, obj) for obj in random_objects]))
		#results[trans_axis(axis)].append(object_distance(gold_obj, random.choice(ordered_objects), axis))
		all_dist[trans_axis(axis)].append(np.average([object_distance(gold_obj, obj, axis=axis) for obj in ordered_objects]))
		"""

	final_overlapping_dist = copy.deepcopy(overlapping_dist)
	final_overlapping_dist["-"] = len(coords_only_examples_fail) - sum(overlapping_dist.values())
	final_overlapping_dist[max(list(overlapping_dist.keys()))+1] = len(dataset[0])-len(coords_only_examples_fail)
	plt_labels, plt_count, plt_perc = [], [], []
	for i in range(max(list(overlapping_dist.keys()))+1, -1, -1):
		plt_labels.append(str(i))
		plt_count.append(final_overlapping_dist[i])
		plt_perc.append(final_overlapping_dist[i]/sum(final_overlapping_dist.values())*100)
		print(f"{plt_labels[-1]}: {plt_count[-1]} ({plt_perc[-1]:.1f}%)")
	plt_labels.append("-")
	plt_count.append(final_overlapping_dist["-"])
	plt_perc.append(final_overlapping_dist["-"] / sum(final_overlapping_dist.values()) * 100)
	print(f"{plt_labels[-1]}: {plt_count[-1]} ({plt_perc[-1]:.1f}%)")

	plot_score_distribution(plt_labels, plt_count, plt_perc)

	for axis, result in distance_overlap_group.items():
		print(f"{axis.upper()}")
		for i in range(max(list(result.keys())), -1, -1):
			print(f"{i}: {sum(result[i])/len(result[i])*100:.1f}%")
	"""
	for axis, result in results.items():
		#plot_histogram(result, bins=9, desc=f"Histogram for axis '{axis}'")
		#plot_histogram(all_dist[axis], bins=9, desc=f"General prob for axis '{axis}'")
		print(f"{axis.upper()}")
		#chi2_stat, p_value = chi_squared_from_samples(result, all_dist[axis], num_bins=9)
		#print(f"Chi-squared statistic: {chi2_stat:.4f}")
		#print(f"P-value: {p_value:.4f}")
		#plot_observed_vs_expected(result, all_dist[axis], num_bins=9, title=f"{axis}")

		t_stat, p = t_test_from_samples(result, all_dist[axis])
		print(f"t-statistic: {t_stat:.4f}, p-value: {p:.4f}")
		plot_observed_vs_expected(result, all_dist[axis], num_bins=9, title=f"{axis}")

		wins = []
		for exp, obs in zip(all_dist[axis], results[axis]):
			wins.append(obs < exp)
		print(f"Win ratio: {(sum(wins)/len(wins))*100:.1f}")
	"""

def plot_score_distribution(labels:list, counts:list, percentages:list):
	# Data
	#labels = ['4', '3', '2', '1', '0', '-']
	#counts = [585, 346, 105, 49, 12, 27]
	#percentages = [52.0, 30.8, 9.3, 4.4, 1.1, 2.4]

	# Plot setup
	fig, ax = plt.subplots(figsize=(6, 4))

	bars = ax.bar(labels, counts, color='gray', edgecolor='black')

	# Add percentages as labels on top of bars
	for bar, pct in zip(bars, percentages):
		height = bar.get_height()
		ax.text(bar.get_x() + bar.get_width()/2, height + 10,
				f'{pct:.1f}%', ha='center', va='bottom', fontsize=10)

	# Styling
	ax.set_xlabel('Common attributes with gold object', fontsize=12)
	ax.set_ylabel('Predicted objects', fontsize=12)
	#ax.set_title('Score Distribution', fontsize=14)
	ax.spines[['top', 'right']].set_visible(False)
	ax.yaxis.grid(True, linestyle='--', alpha=0.7)
	ax.set_axisbelow(True)
	plt.tight_layout()
	fig.savefig("out/figures/3dc_attr_overlap.pdf", format="pdf", bbox_inches='tight')
	plt.show()

def plot_histogram(values, bins, desc):
	if not values:
		print("Empty list provided.")
		return
	if bins <= 0:
		print("Number of steps must be positive.")
		return

	# Filter values to [0, 100]
	#filtered_values = [x * 100 for x in values]
	filtered_values = values

	# Plot histogram with density, then scale to percent
	plt.figure(figsize=(8, 5))
	plt.hist(
		filtered_values,
		bins=bins,
		range=(min(filtered_values), max(filtered_values)),
		edgecolor='black',
		density=True,
	)

	# Convert to percentages
	#plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))

	plt.title(desc)
	plt.xlabel('Value Range')
	plt.ylabel('Probability')
	plt.grid(True, linestyle='--', alpha=0.6)
	plt.tight_layout()
	plt.show()

def chi_squared_from_samples(observed_samples, expected_samples, num_bins=10, range=None):
	"""
	Bins both sample sets and computes the chi-squared statistic and p-value.

	Parameters:
		observed_samples (list or np.array): Sampled data from the observed distribution
		expected_samples (list or np.array): Sampled data from the expected (standard) distribution
		num_bins (int): Number of histogram bins
		range (tuple): Value range (min, max) for consistent binning (optional)

	Returns:
		chi2_stat (float): Chi-squared statistic
		p_value (float): Corresponding p-value
	"""
	# Convert inputs to numpy arrays
	observed_samples = np.asarray(observed_samples)
	expected_samples = np.asarray(expected_samples)

	# Use combined min/max for binning unless specified
	if range is None:
		min_val = min(observed_samples.min(), expected_samples.min())
		max_val = max(observed_samples.max(), expected_samples.max())
		range = (min_val, max_val)

	# Create histogram counts for both distributions
	observed_counts, bin_edges = np.histogram(observed_samples, bins=num_bins, range=range)
	expected_counts, _ = np.histogram(expected_samples, bins=bin_edges)
	expected_counts = expected_counts * (observed_counts.sum() / expected_counts.sum())

	# Ensure expected counts are not zero to avoid division errors
	# Add a small epsilon or merge bins if necessary
	epsilon = 1e-6
	expected_counts = np.where(expected_counts == 0, epsilon, expected_counts)

	# Perform chi-squared test
	chi2_stat, p_value = chisquare(f_obs=observed_counts, f_exp=expected_counts)

	return chi2_stat, p_value

def t_test_from_samples(observed_samples, expected_samples, equal_var=False):
	"""
	Performs a two-sample t-test between observed and expected samples.

	Parameters:
		observed_samples (list or np.array): Sampled data from the observed distribution
		expected_samples (list or np.array): Sampled data from the expected (standard) distribution
		equal_var (bool): Assume equal population variances? Set to False to use Welch’s t-test

	Returns:
		t_stat (float): t-statistic value
		p_value (float): two-tailed p-value
	"""
	observed_samples = np.asarray(observed_samples)
	expected_samples = np.asarray(expected_samples)

	t_stat, p_value = ttest_ind(observed_samples, expected_samples, equal_var=equal_var)

	return t_stat, p_value

def plot_observed_vs_expected(observed_samples, expected_samples, num_bins=20, range=None, title=''):
	"""
	Plots side-by-side histograms of observed and expected sample distributions.

	Parameters:
		observed_samples (list or np.array): The observed data samples
		expected_samples (list or np.array): The expected (standard) data samples
		num_bins (int): Number of bins to use in histograms
		range (tuple): Optional (min, max) range for bins
		title (str): Title for the plot
	"""
	observed_samples = np.asarray(observed_samples)
	expected_samples = np.asarray(expected_samples)

	if range is None:
		min_val = min(observed_samples.min(), expected_samples.min())
		max_val = max(observed_samples.max(), expected_samples.max())
		range = (min_val, max_val)

	# Calculate bin edges
	bin_edges = np.linspace(range[0], range[1], num_bins + 1)

	# Histogram counts (not density)
	obs_counts, _ = np.histogram(observed_samples, bins=bin_edges, density=True)
	exp_counts, _ = np.histogram(expected_samples, bins=bin_edges, density=True)

	bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
	width = (bin_edges[1] - bin_edges[0]) * 0.4  # narrower bars to separate them

	plt.figure(figsize=(8, 4))
	plt.bar(bin_centers - width/2, obs_counts, width=width, label='Observed', alpha=0.7)
	plt.bar(bin_centers + width/2, exp_counts, width=width, label='Expected', alpha=0.7)
	plt.xlabel('Value')
	plt.ylabel('Count')
	plt.title(title)
	plt.legend()
	plt.grid(True, linestyle='--', alpha=0.5)
	plt.tight_layout()
	plt.show()

if __name__ == '__main__':
	#evaluate("out/meta_correlation/predictions/mm", accept_mention_correct=False)
	#evaluate("out/meta_correlation/predictions/reverse", accept_mention_correct=False)
	#evaluate_point("out/meta_correlation/predictions/point", example_count=50)
	#evaluate_human("data/mate/dev/predictions/human")
	#print_main_table("data/mate/dev/predictions", accept_mention_correct=False)
	"""
	object_figure("data/mate/dev/predictions", barplot_models=[
		"llava-hf/llava-v1.6-mistral-7b-hf", "llava-hf/llava-v1.6-vicuna-13b-hf", "llava-hf/llava-v1.6-34b-hf",
		"llava-hf/llama3-llava-next-8b-hf", "allenai/Molmo-7B-O-0924", "unsloth/Llama-3.2-11B-Vision-Instruct",
		"Qwen/Qwen2-VL-7B-Instruct", "random"], line_models=[
		"llava-hf/llava-v1.6-mistral-7b-hf", "llava-hf/llava-v1.6-vicuna-13b-hf", "llava-hf/llava-v1.6-34b-hf",
		"llava-hf/llama3-llava-next-8b-hf", "allenai/Molmo-7B-O-0924", "unsloth/Llama-3.2-11B-Vision-Instruct",
		"Qwen/Qwen2-VL-7B-Instruct", "human"], highlighted_models=["Qwen/Qwen2-VL-7B-Instruct", "human"], human_line_modality="mm")
	""
	object_figure("data/mate/dev/predictions", barplot_models=[
		"llava-hf/llava-1.5-13b-hf", "llava-hf/llava-v1.6-34b-hf", "allenai/Molmo-7B-O-0924",
		"unsloth/Llama-3.2-11B-Vision-Instruct", "Qwen/Qwen2-VL-7B-Instruct", "random", "human"], barplot_modality="mm")
	""
	object_figure("data/mate/dev/predictions", barplot_models=[
		"llava-hf/llava-1.5-13b-hf", "llava-hf/llava-v1.6-34b-hf", "allenai/Molmo-7B-O-0924",
		"unsloth/Llama-3.2-11B-Vision-Instruct", "Qwen/Qwen2-VL-7B-Instruct", "random", "human"], barplot_modality="um")
	""
	object_figure("data/mate/dev/predictions", barplot_models=[
		"llava-hf/llava-v1.6-34b-hf", "allenai/Molmo-7B-O-0924", "unsloth/Llama-3.2-11B-Vision-Instruct",
		"Qwen/Qwen2.5-VL-7B-Instruct", "gpt-4o-2024-11-20", "claude-3-5-sonnet-20241022", "gemini-1.5-flash"], line_models=["human"], highlighted_models=["human"],
				  human_line_modality="mm")
	""
	object_figure("data/mate/dev/predictions", barplot_models=[
		"llava-hf/llava-v1.6-34b-hf", "allenai/Molmo-7B-O-0924", "unsloth/Llama-3.2-11B-Vision-Instruct",
		"Qwen/Qwen2.5-VL-7B-Instruct", "gpt-4o-2024-11-20", "claude-3-5-sonnet-20241022", "gemini-1.5-flash"], line_models=["human"], highlighted_models=["human"], barplot_modality="um",
				  human_line_modality="um")
	"""
	#print_attribute_table("data/mate/dev/predictions/", accept_mention_correct=False, modalities=["mm"], attribute_fncs=["target_attribute"])
	#key_attr_plot("data/mate/dev/predictions/mm_2shot_QwenQwen2.5_VL_7B_Instruct.jsonl")
	#errors_in_data("data/mate/dev/predictions/")
	#print_cot_table("data/mate/dev/predictions", colums=["10"], add_ose= True, add_delta_to_cols=True)
	#print_all_models_table("data/mate/dev/predictions", accept_mention_correct=False, shots=[0, 1, 2], tasks=['i2d', 'd2i'])
	#analyse_3dc_lnk_attr("data/mate/dev/predictions/mm_2shot_QwenQwen2.5_VL_7B_Instruct.jsonl")