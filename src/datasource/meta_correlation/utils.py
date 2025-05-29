"""
Helper functions for the dataset to evaluate whether a VLM can correlate visual information (objects in an image) with metadata information (scene
serialization of the image)
"""
import os.path
import random
import json
from argparse import Namespace
import copy
import math
import shutil
import xml.dom.minidom
import re

import yaml
from dicttoxml import dicttoxml
import hashlib
from collections import defaultdict
import argparse

from tqdm import tqdm
from itertools import chain, combinations

import datasource.clevr.utils as clevr_utils
from blender.utils import scene_to_blend, blend_to_scene, blend_to_img, scene_to_img

from datasource.utils import NpEncoder, load_jsonl, save_jsonl, reindex_by, divide_into_chunks

# Understanding our coordinates.

PROMPT_POINT = """This image shows a minimalist arrangement of 3D geometric shapes made of different materials and colors. The {scene_format} provided contains information about all objects in the scene, including their name, 3D coordinates (represented as [X, Y, Z]), material (metal = shinny, rubber = matte), and other attributes. Your job is to analyze this information, identify which object in the image that corresponds to the object named '{target_name}' in the metadata, and point at it.

Understanding the Coordinate System [X, Y, Z]:
• X (Depth): Represents the depth relative to the camera. Smaller values indicate objects that are farther away.
• Y (Horizontal Position): Represents the left-to-right position. A value of zero means the object is centered in the scene, negative values place the object to the left, and positive values to the right.
• Z (Vertical Position): Represents the height of the object's center point. Larger values correspond to higher vertical positions.

Here is the {scene_format} containing details about all objects in the scene in the image:

{scene}

{question} 
"""

UNIMODAL_IMG = """This image shows a minimalist arrangement of 3D geometric shapes made of different materials and colors.
{str_task}
{options_str}Do not add any additional text to your answer, provide your answer always in the following format: {{"answer": {info_to_retrieve_json}}}.
{few_shot_examples}
{target_question} {pointer_question} """

UNIMODAL_DATA = """The following {scene_format_name} contains information about all objects in a scene, including their name, 3D coordinates (represented as {coords_example}), material, and other attributes.
{str_task}
Do not add any additional text to your answer, provide your answer always in the following format: {{"answer": {info_to_retrieve_json}}}.

Here is the {scene_format_name} containing details about all objects in the scene in the image:

{scene}
{few_shot_examples}
{target_question} {pointer_question} """

PROMPT = """This image shows a minimalist arrangement of 3D geometric shapes made of different materials and colors. The {scene_format_name} provided contains information about all objects in the scene, including their name, 3D coordinates (represented as {coords_example}), material (metal = shinny, rubber = matte), and other attributes. 
{str_task}
{options_str}Do not add any additional text to your answer, provide your answer always in the following format: {{"answer": {info_to_retrieve_json}}}. 

Understanding the Coordinate System X, Y, Z:
• X (Depth): Represents the depth relative to the camera. Smaller values indicate objects that are farther away.
• Y (Horizontal Position): Represents the left-to-right position. A value of zero means the object is centered in the scene, negative values place the object to the left, and positive values to the right.
• Z (Vertical Position): Represents the height of the object's center point. Larger values correspond to higher vertical positions.

Here is the {scene_format_name} containing details about all objects in the scene in the image:

{scene}
{few_shot_examples}
{target_question} {pointer_question} """

MODALITIES = {
	"img": "image",
	"data": "text",
}

TASKS = {
	"img2data": {
		"target_attributes": ["name", "3d_coords", "rotation", "size"],
		"pointer_attributes": ["color", "shape"],
		"str_task": "Your job is to identify {pointer_str} in the image, match its attributes to identify which of the objects mentioned in the {scene_format_name} corresponds to that object, and determine its {info_to_retrieve} in the following format: {{\"answer\": {info_to_retrieve_json}}}",
		"str_task_cot": "Your job is to first identify {pointer_str} in the image. Then, using only the attributes present in the objects in the {scene_format_name}, list the attributes of this object and list the ones that distinguish this object from others in the image. Next, find the object in the {scene_format_name} that contains these same attributes. Finally, return the value of the attribute '{info_to_retrieve}' of this object in the following format: {{\"answer\": {info_to_retrieve_json}}}",
		#"str_shot_cot": "{pointer_str} is the only object in the image that is {lnk_attrs_img}. In the {scene_format_name}, the {info_to_retrieve} of the only object that matches the attributes {lnk_attrs_data} is {tgt_attr_value}, therefore: {answer_str}"
		"str_shot_cot": "{pointer_str} in the image is {all_visual_atts_img} and is the only object in the image that is {lnk_attrs_img}. In the {scene_format_name}, the {info_to_retrieve} of the only object that matches the attributes {lnk_attrs_data} and also contains the attributes {other_visual_atts_data} is {tgt_attr_value}, therefore: {answer_str}"
	},
	"data2img": {
		"target_attributes": ["color", "shape"],
		"pointer_attributes": ["name", "3d_coords", "rotation", "size"],
		"str_task": "Your job is to analyze this information, find {pointer_str}, use its attributes to identify which object in the image corresponds to that object, and retrieve its {info_to_retrieve} in the following format: {{\"answer\": {info_to_retrieve_json}}}",
		"str_task_cot": "Your job is to first identify {pointer_str} in the {scene_format_name}. Then, list the attributes of this object and list the ones that distinguish this object from others in the {scene_format_name}. Next, find the object in the image that matches these same attributes. Finally, return the {info_to_retrieve} of this object in the following format: {{\"answer\": {info_to_retrieve_json}}}",
		#"str_shot_cot": "{pointer_str} is the only object in the {scene_format_name} with the attributes {lnk_attrs_data}. In the image, the {info_to_retrieve} of the only object that is {lnk_attrs_img} is {tgt_attr_value}, therefore: {answer_str}"
		"str_shot_cot": "{pointer_str} in the {scene_format_name} also contains the attributes {all_visual_atts_data} and is the only object in the {scene_format_name} with the attributes {lnk_attrs_data}. In the image, the {info_to_retrieve} of the only object that is {lnk_attrs_img} and is also {other_visual_atts_img} is {tgt_attr_value}, therefore: {answer_str}"
	},
	"img2img": {
		"target_attributes": ["color", "shape"],
		"pointer_attributes": ["color", "shape"],
		"str_task": "Your job is to identify {pointer_str} in the image and determine its {info_to_retrieve} in the following format: {{\"answer\": {info_to_retrieve_json}}}"
	},
	"data2data": {
		"target_attributes": ["name", "3d_coords", "rotation", "size"],
		"pointer_attributes": ["name", "3d_coords", "rotation", "size"],
		"str_task": "Your job is to analyze this information, find {pointer_str}, and retrieve its {info_to_retrieve} from its attributes in the following format: {{\"answer\": {info_to_retrieve_json}}}"
	},
}

ATTRIBUTES = {
	"name": {
		"info_to_retrieve": "name",
		"info_to_retrieve_json": "\"NAME\"",
		"target_question": "What is the name",
		"pointer_question": "of the object named '{pnt_attr_value}'?",
		"pointer_str": "the object named '{pnt_attr_value}'",
	},
	"material": {
		"info_to_retrieve": "material",
		"target_question": "What is the material",
		"pointer_question": "of the object whose material is {pnt_attr_value}?",
		"pointer_str": "the object made of {pnt_attr_value}",
		"info_to_retrieve_json": "\"MATERIAL\"",
		"options": "metal or rubber",
		"options_lst": ["metal", "rubber"],
		"lnk_attr_img": "made of {attr_value}",
		"desc_order": 2
	},
	"3d_coords": {
		"info_to_retrieve": "3D coordinates",
		"info_to_retrieve_json": "[X, Y, Z]",
		"target_question": "Which are the 3D coordinates",
		"pointer_question": "of the object at the coordinates {pnt_attr_value}?",
		"pointer_str": "the object at coordinates {pnt_attr_value}",
		"lnk_attr_img": "located around the coordinates {attr_value}",
		"desc_order": 4
	},
	"shape": {
		"info_to_retrieve": "shape",
		"info_to_retrieve_json": "\"SHAPE\"",
		"target_question": "What is the shape",
		"pointer_question": "of the {pnt_attr_value}?",
		"pointer_str": "the {pnt_attr_value}",
		"options": "cylinder, sphere, cone, or cube",
		"options_lst": ["cylinder", "sphere", "cube", "cone"],
		"lnk_attr_img": "a {attr_value}",
		"desc_order": 1
	},
	"color": {
		"info_to_retrieve": "color",
		"info_to_retrieve_json": "\"COLOR\"",
		"target_question": "What is the color",
		"pointer_question": "of the {pnt_attr_value} colored object?",
		"pointer_str": "the {pnt_attr_value} colored object",
		"options": "gray, red, blue, green, brown, purple, cyan, or yellow",
		"options_lst": ["gray", "red", "blue", "green", "brown", "purple", "cyan", "yellow"],
		"lnk_attr_img": "{attr_value}",
		"desc_order": 0
	},
	"rotation": {
		"info_to_retrieve": "rotation",
		"info_to_retrieve_json": "\"ROTATION\"",
		"target_question": "What is the rotation value",
		"pointer_question": "of the object with a rotation of {pnt_attr_value}?",
		"pointer_str": "the object with a rotation of {pnt_attr_value}",
	},
	"size": {
		"info_to_retrieve": "size",
		"info_to_retrieve_json": "\"SIZE\"",
		"target_question": "What is the size value",
		"pointer_question": "of the object with a size of {pnt_attr_value}?",
		"pointer_str": "the object with a size of {pnt_attr_value}",
		"lnk_attr_img": "with an approximate size of {attr_value}",
		"desc_order": 3
	},
}

SCENE_FORMAT = {
	"json": {
		"name":"JSON",
		"3d_coords_example": "[X, Y, Z]"
	},
	"yaml":{
		"name": "YAML",
		"3d_coords_example": "- X\n  - Y\n  - Z\n"
	},
	"xml": {
		"name": "XML",
		"3d_coords_example": "<item>X</item><item>Y</item><item>Z</item>"
	},
	"text": {
		"name": "text",
		"3d_coords_example": "[X, Y, Z]"
	}
}

def rotate_z_axis(point, angle_rad):
	x, y, z = point

	# Rotation matrix components
	cos_theta = math.cos(angle_rad)
	sin_theta = math.sin(angle_rad)

	# Apply rotation
	x_prime = cos_theta * x - sin_theta * y
	y_prime = sin_theta * x + cos_theta * y
	z_prime = z  # Z remains unchanged for rotation around Z-axis

	return [x_prime, y_prime, z_prime]

def simplify_geometry(scene):
	"""
	The plane in CLEVR scenes is rotated 'sideways' making it difficult to locate objects with the given coordinates
	To simplify this we rotate the camera to align with horizontal axis (Y=0)
	We also rotate every element of the scene to keep it the same as in clever
	I couldn't figure ot how to rotate one of the lamps tho (future bug fix)
	@param scene:
	@return:
	"""
	# Rotate the camera
	org_x = scene["camera_location"][0]
	cam_h = math.hypot(scene["camera_location"][0], scene["camera_location"][1])
	scene["camera_location"] = [cam_h, 0, scene["camera_location"][2]]
	arcsine_radians = math.asin(org_x / cam_h)
	cam_rot_angle = math.radians(90) - arcsine_radians

	# Rotate objects
	for obj in scene["objects"]:
		obj["3d_coords"] = rotate_z_axis(obj["3d_coords"], cam_rot_angle)

	# Rotate lamps
	for lamp in {'Lamp_Key', 'Lamp_Back', 'Lamp_Fill'}:
		scene["metadata"][lamp] = rotate_z_axis(scene["metadata"][lamp], cam_rot_angle)

	return scene

def group_by_overlap(dict_list, reference):
	"""
	Given this list and a reference dictionary, returns a dictionary where the key is the number of overlapping values
	and the value is a list of such overlapping dictionaries
	:param dict_list:
	:param reference:
	:return:
	"""
	overlap_groups = {}

	for d in dict_list:
		# Count how many key-value pairs overlap with the reference
		overlap_count = sum(
			1 for key, value in d.items()
			if key in reference and reference[key] == value
		)

		# Group dictionaries by the overlap count
		overlap_groups.setdefault(overlap_count, []).append(d)

	return overlap_groups

def find_overlap_key(grouped_dicts, target_dict):
	for overlap_count, dicts in grouped_dicts.items():
		for d in dicts:
			if d == target_dict:
				return overlap_count
	return None  # Return None if not found

def get_unique_identifiers(selected, group):
	"""
	function that given a selected dictionary and list of other dictionaries, all with the same keys, gets all the
	combinations of key: value of the selected object that could uniquely identify it among the group.
	For example: selected object is {a: 1, b: 2, c: 3} and the group is [{a: 1, b: 3, c: 3}, {a: 2, b: 2, c: 4}] and
	the answer must be [[a: 1, b: 2], [b: 2, c: 3], [a:1, b: 2, c: 3]]
	@param selected:
	@param group:
	@return:
	"""
	keys = list(selected.keys())
	all_combinations = chain.from_iterable(combinations(keys, r) for r in range(1, len(keys) + 1))
	unique_identifiers = []

	for key_comb in all_combinations:
		subset = {k: selected[k] for k in key_comb}
		if all(any(obj[k] != selected[k] for k in key_comb) for obj in group):
			unique_identifiers.append(subset)

	return unique_identifiers

def remove_nonlinking_attributes(obj, example, skip_coords:bool = True):
	"""
	WARNING: This function MUTATES THE OBJ
	Remove the keys of the object in a scene that are not used to link the object, leaving only the linking attrs.
	:param obj:
	:param pnt_modality:
	:param tgt_modality:
	:param pnt_attr:
	:param tgt_attr:
	:param skip_coords:
	:return:
	"""
	pnt_attr, tgt_attr = list(example["pointer_attribute"].keys())[0], list(example["target_attribute"].keys())[0]
	pnt_modality, tgt_modality = example["task"].split("2")

	obj["size"] = "small" if obj["size"] < 0.5 else "large"
	obj.pop("name", None)  # Invisible
	obj.pop("rotation", None)  # Invisible
	if skip_coords:
		obj.pop("3d_coords", None)  # We analyse attribute combinations on top of coordinates, because coordinates are usually left for last
	if pnt_modality != tgt_modality:
		obj.pop(pnt_attr if pnt_modality == "img" else tgt_attr, None)  # pointer won't be in the data
	return obj

def get_linking_attributes(example, skip_coords:bool = True):
	gold_obj = copy.deepcopy(get_gold_object(example))
	all_objs = copy.deepcopy(example["scene"]["objects"])
	pnt_attr, tgt_attr = list(example["pointer_attribute"].keys())[0], list(example["target_attribute"].keys())[0]
	# Get other objects without the gold
	other_objs = [obj for obj in all_objs if obj["name"] != gold_obj["name"]]
	all_objs.append(gold_obj)
	pnt_modality, tgt_modality = example["task"].split("2")
	for obj in all_objs:
		remove_nonlinking_attributes(obj, example, skip_coords)
	unique_identifiers = get_unique_identifiers(gold_obj, other_objs)
	# key_attr = {key: val for key, val in key_attr.items() if key not in {"name", "rotation"}}
	return unique_identifiers

def get_gold_object(example:dict):
	pnt_attr_name, pnt_attr_val = list(example["pointer_attribute"].keys())[0], list(example["pointer_attribute"].values())[0]
	tgt_attr_name, tgt_attr_val = list(example["target_attribute"].keys())[0], list(example["target_attribute"].values())[0]
	for obj in example["scene"]["objects"]:
		if obj[pnt_attr_name] == pnt_attr_val:
			assert obj[tgt_attr_name] == tgt_attr_val
			return obj

def get_key_attributes_for_object(scene_object:dict, scene_objects:list):
	"""
	Given an object, get all attributes that can uniquely identify it (later remove rotation, name)
	@return:
	"""
	attr_count = {key: 0 for key in scene_object.keys()}
	for obj in scene_objects:
		for attr_name, attr_val in scene_object.items():
			candidate_attr_val = obj[attr_name]
			if attr_name == "size":
				# Discrete to the two only visually perceivable sizes
				attr_val = "small" if attr_val < 0.5 else "large"
				candidate_attr_val = "small" if candidate_attr_val < 0.5 else "large"
			attr_count[attr_name] += 1 if attr_val == candidate_attr_val else 0
	# Keep only the 1s
	return {key: val for key, val in attr_count.items() if val == 1}

def get_key_data_attributes(scene_objects:list):
	"""
	Find the attributes that can uniquely identify an object in the data
	@param scene_objects:
	@return:
	"""
	unique_combinations = {}
	count_dict = defaultdict(lambda:{"f_count":0, "pair":None})
	for obj in scene_objects:
		for key, value in obj.items():
			count_dict[f"{key}#{str(value)}"]["f_count"] += 1
			count_dict[f"{key}#{str(value)}"]["pair"] = (key, value)
	# Get only the 1s
	key_data_attributes = defaultdict(list)
	for _, count_obj in count_dict.items():
		if count_obj["f_count"] == 1 and count_obj["pair"][0] in TASKS["data2img"]["pointer_attributes"]:
			key_data_attributes[count_obj["pair"][0]].append(count_obj["pair"][1])
	return dict(key_data_attributes)

def update():
	src_dataset = load_jsonl("data/meta_correlation/v4/src.jsonl")
	for example in src_dataset:
		example["key_attributes"]["data"] = get_key_data_attributes(example["scene"]["objects"])
	save_jsonl("data/meta_correlation/v4/src2.jsonl", src_dataset)

def build_example_src(clevr_scene, out_path: str, max_key_colors: int = 3, max_key_shapes: int = 1):
	# Let's make it our own
	scene = blend_to_scene(scene_to_blend(clevr_scene, output_blend_path="/tmp/tmp.blend", is_clevr_scene=True))

	# We don't want unique shapes and colors to overlap in the same object. There are scenes with only 3 objects so we prioritise that one unique shape
	key_attributes = {
		"color": random.sample(ATTRIBUTES["color"]["options_lst"], min(max_key_colors, len(scene["objects"])-max_key_shapes)),
		"shape": random.sample(ATTRIBUTES["shape"]["options_lst"], min(max_key_shapes, len(scene["objects"]))),
	}
	other_colors = [c for c in ATTRIBUTES["color"]["options_lst"] if c not in key_attributes["color"]]
	other_shapes = [c for c in ATTRIBUTES["shape"]["options_lst"] if c not in key_attributes["shape"]]

	# Align to horizontal axis so coordinates can be interpreted easier
	scene = simplify_geometry(scene)

	# Replace names with our own (non-descriptive)
	for i, scene_object in enumerate(scene["objects"]):
		scene_object["name"] = f"Object_{i}"

	# Re-assign other colors and shapes to objects with unique attributes
	for scene_object in scene["objects"]:
		for attr_name, attr_list in key_attributes.items():
			if scene_object[attr_name] in attr_list:
				scene_object[attr_name] = random.choice(other_colors if attr_name == "color" else other_shapes)
	target_objects = random.sample(scene["objects"], min(max_key_colors+max_key_shapes, len(scene["objects"])))

	key_attr_tuple_lst = [(key, value) for key, values in key_attributes.items() for value in values]
	for t_obj, (attr_name, attr_val) in zip(target_objects, key_attr_tuple_lst):
		t_obj[attr_name] = attr_val
	example_id = hashlib.md5(str(json.dumps(scene)).encode()).hexdigest()
	img_filename = f"{example_id}.png"
	img_relative_path = os.path.join(out_path, "img", img_filename)
	os.makedirs(os.path.dirname(img_relative_path), exist_ok=True)
	scene_to_img(scene, output_img_path=img_relative_path)

	key_data_attributes = get_key_data_attributes(scene["objects"])

	return {
		"example_id": example_id,
		"image": img_filename,
		"scene": scene,
		"key_attributes": {"img": key_attributes, "data": key_data_attributes},
		"object_count": len(scene["objects"])
	}

def rearrange_coordinates(data):
	# We assume all lists of size 3 in the scene are coordinates
	if isinstance(data, dict):
		return {key: rearrange_coordinates(value) for key, value in data.items()}
	elif isinstance(data, list):
		if len(data) == 3:
			return [data[1], data[2], data[0]]
		return [rearrange_coordinates(item) for item in data]
	else:
		return data

# Custom func to indent jsons but not the coordinates
def custom_json_dumps(data, indent=4):
	def custom_serializer(obj, level=0):
		if isinstance(obj, dict):
			items = []
			for key, value in obj.items():
				if key == "3d_coords" and isinstance(value, list) and len(value) == 3:
					items.append(f'{json.dumps(key)}: {json.dumps(value)}')
				else:
					items.append(f'{json.dumps(key)}: {custom_serializer(value, level + 1)}')
			return "{\n" + ",\n".join(" " * ((level + 1) * indent) + item for item in items) + "\n" + " " * (
						level * indent) + "}"
		elif isinstance(obj, list):
			return "[\n" + ",\n".join(
				" " * ((level + 1) * indent) + custom_serializer(item, level + 1) for item in obj) + "\n" + " " * (
						level * indent) + "]"
		else:
			return json.dumps(obj)

	return custom_serializer(data)

def convert_scene(scene, scene_format, indent:bool=False):
	match scene_format:
		case "json":
			#return json.dumps(scene, indent=4) if indent else json.dumps(scene)
			return custom_json_dumps(scene, indent=4) if indent else json.dumps(scene)
		case "yaml":
			return yaml.dump(scene, default_flow_style=False)
		case "xml":
			xml_str = dicttoxml(scene, custom_root='root', attr_type=False).decode('utf-8')
			if indent:
				dom = xml.dom.minidom.parseString(xml_str)
				pretty_xml = dom.toprettyxml(indent="    ")
				return "\n".join([line for line in pretty_xml.splitlines() if line.strip()])
			else:
				return xml_str
		case "text":
			text = f"In this scene, the camera is located at the coordinates: {scene['camera_location']}. "
			#text += f"There are three sources of light in the scene: one lamp located at the coordinates {scene['Lamp_Back']}, another one at {scene['Lamp_Fill']}, and a third one at {scene['Lamp_Key']}. "
			text += f"The scene contains the following objects:"
			for scene_obj in scene["objects"]:
				color_str = f"{scene_obj['color']} " if "color" in scene else ""
				text += f" A {color_str}{scene_obj['shape']} named {scene_obj['name']} located at the coordinates {scene_obj['3d_coords']}. This object has a size of {scene_obj['size']}, is rotated {scene_obj['rotation']} degrees, and is made of {scene_obj['material']}."
			return text
		case _:
			raise Exception(f"Unknown scene format {scene_format}")

def build_example(example, task:str, pnt_attr: str = "random", pnt_attr_value: str = "random", tgt_attr: str = "random", n_shots: int = 2, scene_format: str = "json"):
	"""

	@param example:
	@param pnt_attr: the attribute we will use to point at the object we want to query e.g. (random, color, shape)
	@param pnt_attr_value: the value of the attribute we use to point at the object we want to query e.g. (random, red, green, blue, a specific shape)
	@param tgt_attr: kind of question in this inference
	@param n_shots: an int between [0 and 2] for 0 shot, 1 shot, 2 shot
	@param scene_format: json | yaml | xml | text
	@return:
	"""
	pnt_modality, tgt_modality = task.split("2")
	if tgt_attr == "random":
		tgt_attr = random.choice(TASKS[task]["target_attributes"])
	if pnt_attr == "random":
		pnt_attr = random.choice(TASKS[task]["pointer_attributes"])
	if pnt_attr_value == "random":
		pnt_attr_value = random.choice(example["key_attributes"][pnt_modality][pnt_attr])

	scene = copy.deepcopy(example["scene"])
	del scene["image_filename"]
	del scene["metadata"]

	# Get target object
	target_object = [o for o in example["scene"]["objects"] if o[pnt_attr] == pnt_attr_value][0]

	# N-shot
	few_shot_pnt_attrs = [(attr_type, attr_val) for attr_type, attr_vals in example["key_attributes"][pnt_modality].items() for attr_val in attr_vals if not (attr_type == pnt_attr and attr_val == pnt_attr_value)]
	few_shot_pnt_attrs = random.sample(few_shot_pnt_attrs, n_shots)
	few_shot_examples = "\nFor example:\n" if n_shots > 0 else ""
	for shot_pnt_attr, shot_pnt_attr_val in few_shot_pnt_attrs:
		shot_object = [o for o in example["scene"]["objects"] if o[shot_pnt_attr] == shot_pnt_attr_val][0]
		# In case of single modality we can end up asking the same pointer
		shot_tgt_attr = tgt_attr if tgt_attr != shot_pnt_attr else random.choice([t for t in TASKS[task]["target_attributes"] if t != shot_pnt_attr])
		few_shot_examples += f"{ATTRIBUTES[shot_tgt_attr]['target_question']} {ATTRIBUTES[shot_pnt_attr]['pointer_question'].format(pnt_attr_value=shot_pnt_attr_val)} {json.dumps({'answer': shot_object[shot_tgt_attr]})}\n"
	few_shot_examples += "Now answer the following question:" if n_shots > 0 else ""

	# Remove the pointer property from the scene in the prompt if this is not an unimodal example
	if pnt_modality != tgt_modality:
		for scene_object in scene["objects"]:
			del scene_object[pnt_attr if task == "img2data" else tgt_attr]

	str_task = TASKS[task]["str_task"].format(
		pointer_str=ATTRIBUTES[pnt_attr]["pointer_str"].format(pnt_attr_value=pnt_attr_value),
		scene_format_name=SCENE_FORMAT[scene_format]["name"],
		info_to_retrieve=ATTRIBUTES[tgt_attr]["info_to_retrieve"],
		info_to_retrieve_json=ATTRIBUTES[tgt_attr]["info_to_retrieve_json"]
	)
	options_str = f"Select your answer from the following options: {ATTRIBUTES[tgt_attr]['options']}. " if tgt_modality == "img" else ""

	match task:
		case "img2img":
			input_str = UNIMODAL_IMG.format(
				scene_format_name=SCENE_FORMAT[scene_format]["name"],
				coords_example=SCENE_FORMAT[scene_format]["3d_coords_example"],
				str_task=str_task,
				options_str=options_str,
				info_to_retrieve_json=ATTRIBUTES[tgt_attr]["info_to_retrieve_json"],
				scene=convert_scene(scene, scene_format),
				few_shot_examples=few_shot_examples,
				target_question=ATTRIBUTES[tgt_attr]['target_question'],
				pointer_question=ATTRIBUTES[pnt_attr]['pointer_question'].format(pnt_attr_value=pnt_attr_value))
		case "data2data":
			input_str = UNIMODAL_DATA.format(
				scene_format_name=SCENE_FORMAT[scene_format]["name"],
				coords_example=SCENE_FORMAT[scene_format]["3d_coords_example"],
				str_task=str_task,
				options_str=options_str,
				info_to_retrieve_json=ATTRIBUTES[tgt_attr]["info_to_retrieve_json"],
				scene=convert_scene(scene, scene_format),
				few_shot_examples=few_shot_examples,
				target_question=ATTRIBUTES[tgt_attr]['target_question'],
				pointer_question=ATTRIBUTES[pnt_attr]['pointer_question'].format(pnt_attr_value=pnt_attr_value))
		case _:
			input_str = PROMPT.format(
				scene_format_name=SCENE_FORMAT[scene_format]["name"],
				coords_example=SCENE_FORMAT[scene_format]["3d_coords_example"],
				str_task=str_task,
				options_str=options_str,
				info_to_retrieve_json=ATTRIBUTES[tgt_attr]["info_to_retrieve_json"],
				scene=convert_scene(scene, scene_format),
				few_shot_examples=few_shot_examples,
				target_question=ATTRIBUTES[tgt_attr]['target_question'],
				pointer_question=ATTRIBUTES[pnt_attr]['pointer_question'].format(pnt_attr_value=pnt_attr_value))

	example["target_attribute"] = {tgt_attr: target_object[tgt_attr]}
	example["pointer_attribute"] = {pnt_attr: pnt_attr_value}
	example["few_shot_attributes"] = few_shot_pnt_attrs
	example["scene_format"] = scene_format
	example["image"] = "00000000000000000000000000000000.png" if task == "data2data" else example["image"]
	example["input_str"] = input_str
	example["task"] = task
	example["gold_reference"] = json.dumps({"answer": target_object[tgt_attr]})
	return example

def build_point_example(example, scene_format: str = "json"):
	"""
	I bet this can easly merged with the function above (no time)
	@param example:
	@param task: kind of question in this inference
	@param target_color: the color we will use to target the object of the question e.g. (random, red, green, blue)
	@param n_shots: an int between [0 and 2] for 0 shot, 1 shot, 2 shot
	@param scene_format: json | yaml | xml | text
	@return:
	Model output should be something in the line of ' <point x="59.6" y="28.3" alt="red cube">red cube</point>'
	"""
	target_object = random.choice(example["scene"]["objects"])
	target_name = target_object["name"]

	scene = copy.deepcopy(example["scene"])
	del scene["image_filename"]
	del scene["metadata"]

	# Let's re-arrange the coordinates in a more familiar format from [Z, X, Y] to [X, Y, Z]. No: We'll keep it as it is
	#scene = rearrange_coordinates(scene)

	input_str = PROMPT_POINT.format(target_name=target_name,
							  scene_format=SCENE_FORMAT[scene_format],
							  scene=convert_scene(scene, scene_format),
							  question=f"Point at the object named '{target_name}'")
	example["task"] = "color"
	example["scene_format"] = scene_format
	example["target_name"] = target_name
	example["input_str"] = input_str
	return example


def build_conversation(example):
	return [("user", example["input_str"], example["image"])]


def generate_src_dataset(args):
	"""
	Generates source examples with not task or target that will later be the foundation form which we will generate
	examples for other types of evaluation (original, reverse, pointing...)
	@param args:
	@return:
	"""
	random.seed(args.random_seed)
	clevr_scenes = clevr_utils.load_dataset(args.clever_split, "scenes")
	random.shuffle(clevr_scenes)
	clevr_scenes = clevr_scenes[:args.dataset_size]
	with open(os.path.join(args.out_path, "src.jsonl"), 'w', encoding='utf8') as outfile:
		for clevr_scene in tqdm(clevr_scenes):
			example = build_example_src(clevr_scene, args.out_path)
			json.dump(example, outfile, ensure_ascii=False, cls=NpEncoder)
			outfile.write('\n')

def min_idx_matrix(grid: list, rows_to_check:list = None, columns_to_check:list = None, exclude_diagonal: bool = False):
	rows_to_check = len(grid) if rows_to_check is None else rows_to_check
	columns_to_check = len(grid[0]) if columns_to_check is None else columns_to_check

	# Find the i, j index of the minimum value in the specified rows and columns
	min_value = float('inf')  # Set initial minimum to infinity
	min_index = (-1, -1)  # Placeholder for the minimum index

	for i in rows_to_check:
		for j in columns_to_check:
			if exclude_diagonal and i == j:
				continue  # Skip if i equals j and exclude_diagonal is True
			if grid[i][j] < min_value:
				min_value = grid[i][j]
				min_index = (i, j)
	return min_index

def generate_dataset(args):
	src_dataset = load_jsonl(args.src_dataset_path)
	if args.tasks == "point":
		args.out_path = os.path.join(args.out_path, "point")
	tasks = args.tasks.split(";")
	with open(os.path.join(args.out_path, f"mm_{args.n_shots}shot.jsonl"), 'w', encoding='utf8') as outfile:
		# To divide tasks uniformly across object_count
		task_dist = defaultdict(lambda: {key: 0 for key in tasks}.copy())
		attr_dist = defaultdict(lambda: copy.deepcopy({task: [[0 for _ in range(len(TASKS[task]["pointer_attributes"]))] for _ in range(len(TASKS[task]["target_attributes"]))] for task in tasks}))
		for src_example in src_dataset:
			# Find the least represented task for this object_count
			count_task_dist = task_dist[src_example["object_count"]]
			task = min(count_task_dist, key=count_task_dist.get)
			count_task_dist[task] += 1
			pnt_modality, tgt_modality = task.split("2")
			match task:
				case "point":
					example = build_point_example(src_example, scene_format=args.scene_format)
				case _:
					available_tgts = [TASKS[task]["target_attributes"].index(x) for x in src_example["key_attributes"][tgt_modality].keys()]
					available_pnts = [TASKS[task]["pointer_attributes"].index(x) for x in src_example["key_attributes"][pnt_modality].keys()]
					tgt_idx, pnt_idx = min_idx_matrix(attr_dist[src_example["object_count"]][task], available_tgts, available_pnts, exclude_diagonal=pnt_modality==tgt_modality)
					tgt_attr = TASKS[task]["target_attributes"][tgt_idx]
					pnt_attr = TASKS[task]["pointer_attributes"][pnt_idx]
					attr_dist[src_example["object_count"]][task][tgt_idx][pnt_idx] += 1

					example = build_example(src_example, task, pnt_attr=pnt_attr, tgt_attr=tgt_attr, n_shots=args.n_shots, scene_format=args.scene_format)

			#json.dump(example, outfile, ensure_ascii=False, cls=NpEncoder)
			#outfile.write('\n')
		mean_val, std_val = calculate_stats(attr_dist)
		print(":D")


def calculate_stats(data):
	"""
	Recursively traverse the nested dictionary to extract all integers,
	and then compute the mean and standard deviation.

	Parameters:
	data (dict): The input dictionary with nested lists/dictionaries containing integers.

	Returns:
	tuple: mean and standard deviation of all integers.
	"""
	import numpy as np
	values = []

	def extract_numbers(item):
		if isinstance(item, int):
			values.append(item)
		elif isinstance(item, list):
			for elem in item:
				extract_numbers(elem)
		elif isinstance(item, dict):
			for sub_item in item.values():
				extract_numbers(sub_item)

	# Start the recursive extraction
	extract_numbers(data)

	# Calculate mean and standard deviation
	mean_val = np.mean(values)
	std_val = np.std(values)  # Note: This computes the population standard deviation.

	return mean_val, std_val

def generate_human_eval_set(evaluators: list, task_order:list = None, ex_per_comb:int = 1):
	if task_order == None:
		task_order = ["data2data", "img2img", "img2data", "data2img"]
	datasets = {
		"mm_2shot": {"dataset": load_jsonl("data/meta_correlation/v4/mm_2shot.jsonl"), "pointer": 0},
		#"um_2shot": {"dataset": load_jsonl("data/meta_correlation/v4/um_2shot.jsonl"), "pointer": 0},
	}
	for key, dataset_info in datasets.items():
		prev_pred_path = os.path.join("data/meta_correlation/v4/predictions/", f"{key}_human.jsonl")
		if os.path.isfile(prev_pred_path):
			prev_dataset = load_jsonl(prev_pred_path, indexed=True, id_key_name="example_id")
			dataset_info["dataset"] = [ex for ex in dataset_info["dataset"] if ex["example_id"] not in prev_dataset]

	# Shuffle datasets
	random.seed(117)
	random.shuffle(evaluators)

	# Let's build a confusion matrix for each object count
	attributes = ['shape', 'color', 'name', '3d_coords', 'rotation', 'size']
	obj_example_matrix = defaultdict(lambda: [[[] for _ in range(len(attributes))] for _ in range(len(attributes))])
	for key, dataset_info in datasets.items():
		dataset = dataset_info["dataset"]
		for example in dataset:
			tgt_idx = attributes.index(list(example['target_attribute'].keys())[0])
			pnt_idx = attributes.index(list(example['pointer_attribute'].keys())[0])
			obj_example_matrix[example['object_count']][tgt_idx][pnt_idx].append(example)

	# Now we've got one example per pnt-tgt attr combination and object count. Let's merge all into one list, shuffle and distribute them
	examples_to_eval = []
	for obj_count, example_matrix in obj_example_matrix.items():
		for tgt_row in example_matrix:
			for pnt_examples in tgt_row:
				if len(pnt_examples) > 0: # Skip vertical
					examples_to_eval.extend(random.sample(pnt_examples, ex_per_comb))
	#random.shuffle(examples_to_eval)
	task_examples_to_eval = reindex_by(examples_to_eval, "task")
	task_examples_to_eval = {task: divide_into_chunks(task_examples_to_eval[task], len(evaluators)) for task in task_order}

	sheets = {evaluator: "" for evaluator in evaluators}
	for eval_idx, evaluator in enumerate(evaluators):
		evaluator_dir = os.path.join("out/meta_correlation/human_eval", evaluator)
		os.makedirs(evaluator_dir)
		total_examples = sum([len(evaluator_examples_to_eval[eval_idx]) for task, evaluator_examples_to_eval in task_examples_to_eval.items()])
		idx = 0
		for i, task in enumerate(task_order):
			evaluator_examples_to_eval = task_examples_to_eval[task]
			for j, example in enumerate(evaluator_examples_to_eval[eval_idx]):
				sheets[evaluator] += f"## {idx}/{total_examples} ##-- {task}:{example['example_id']}\n"
				sheets[evaluator] += f"-------------------------------------------------------\n"
				input_str:str = example["input_str"]
				pnt_attr = list(example['pointer_attribute'].keys())[0]
				tgt_attr = list(example['target_attribute'].keys())[0]
				pnt_modality, tgt_modality = task.split("2")
				# replace
				if task != "img2img":
					scene = copy.deepcopy(example["scene"])
					del scene["image_filename"]
					del scene["metadata"]
					# Remove the pointer property from the scene in the prompt if this is not an unimodal example
					if pnt_modality != tgt_modality:
						for scene_object in scene["objects"]:
							del scene_object[pnt_attr if task == "img2data" else tgt_attr]
					indented_scene = convert_scene(scene, "json", indent=True)
					input_str = replace_between(input_str, "about all objects in the scene in the image:\n", "\nFor example:", indented_scene)
				if task != "data2data":
					shutil.copy(os.path.join("data/meta_correlation/v4/img", example["image"]), os.path.join(evaluator_dir, f"{idx}.png"))
				sheets[evaluator] += input_str
				sheets[evaluator] += f"\n-------------------------------------------------------\n"
				idx += 1
		with open(os.path.join(evaluator_dir, "eval_file.txt"), "w") as file:
			file.write(sheets[evaluator])

def replace_between(text, start, end, replacement):
	pattern = re.escape(start) + r"(.*?)" + re.escape(end)
	return re.sub(pattern, start + replacement + end, text, flags=re.DOTALL)

def random_prediction(example):
	tgt_attr_key = list(example["target_attribute"].keys())[0]  #{'name': 'Object_7'}
	tgt_modality = example["task"].split("2")[-1]
	possible_answers = []
	if tgt_modality == "data":
		for s_object in example["scene"]["objects"]:
			possible_answers.append(s_object[tgt_attr_key])
	else:
		possible_answers = ATTRIBUTES[tgt_attr_key]["options_lst"]
	return json.dumps({"answer": random.choice(possible_answers)})

def execute_random(dataset_path: str, runs: int, output_path: str = "out/meta_correlation/predictions"):
	dataset = load_jsonl(dataset_path)
	model_name = "random"
	result = []
	for run in range(runs):
		for example in dataset:
			new_example = copy.deepcopy(example)
			new_example["input_str"] = ""
			new_example["model"] = model_name
			new_example["run"] = run
			new_example["prediction"] = random_prediction(example)
			result.append(new_example)

	pred_file_name = f"{os.path.splitext(os.path.basename(dataset_path))[0]}_{model_name}.jsonl"
	save_jsonl(os.path.join(output_path, pred_file_name), result)

def update_with_lnk_attrs(dataset_dir:str):
	"""
	Updates an already genearted v3 dataset with the linking attributes for the gold entity
	@return:
	"""
	dataset = load_jsonl(dataset_dir)
	for example in tqdm(dataset):
		example["linking_attributes"] = get_linking_attributes(example, skip_coords=False)
	save_jsonl(dataset_dir, dataset)

def update_with_cot(dataset_dir:str):
	"""
	Updates an already generated v3 dataset and replaces examples and prompt with chain of thought
	@return:

	"""
	dataset = load_jsonl(dataset_dir)
	if "linking_attributes" not in dataset[0]:
		update_with_lnk_attrs(dataset_dir)
	for example in tqdm(dataset):
		# Sketchy but fast to implement replace
		# shots "? {"
		task = example["task"]
		pnt_modality, tgt_modality = task.split("2")
		pnt_attr = list(example["pointer_attribute"].keys())[0]
		pnt_attr_value = example["pointer_attribute"][pnt_attr]
		tgt_attr = list(example["target_attribute"].keys())[0]
		scene_format = example["scene_format"]
		input_str = example["input_str"]
		str_task = TASKS[task]["str_task_cot"].format(
			pointer_str=ATTRIBUTES[pnt_attr]["pointer_str"].format(pnt_attr_value=pnt_attr_value),
			scene_format_name=SCENE_FORMAT[scene_format]["name"],
			info_to_retrieve=ATTRIBUTES[tgt_attr]["info_to_retrieve"],
			info_to_retrieve_json=ATTRIBUTES[tgt_attr]["info_to_retrieve_json"]
		)
		input_str = re.sub(r'^Your job is.*\n?', f"{str_task}\n", input_str, flags=re.MULTILINE)
		input_str = re.sub(r'Do not add any additional text to your answer.*\n?', '', input_str, flags=re.MULTILINE)

		if "For example:" in input_str:
			matches = re.findall(r'^(.*?)(?=\? \{)', input_str, re.MULTILINE)
			tgt_attrs = [next((k for k, v in ATTRIBUTES.items() if v["target_question"] in m), None) for m in matches]
			cot_examples = []
			for question, shot_pnt_attr, shot_tgt_attr in zip(matches, example['few_shot_attributes'], tgt_attrs):
				#lnk_attrs = sorted(example["linking_attributes"], key=len)
				shot_pnt_attr, shot_pnt_attr_value = shot_pnt_attr
				shot_object = next((obj for obj in example["scene"]["objects"] if obj[shot_pnt_attr] == shot_pnt_attr_value), None)
				# We do this to extract the lnk_attrs for the gold in the shot example, not this example's gold
				shot_poof_example = copy.deepcopy(example)
				shot_poof_example["pointer_attribute"] = {shot_pnt_attr: shot_pnt_attr_value}
				shot_poof_example["target_attribute"] = {shot_tgt_attr: shot_object[shot_tgt_attr]}
				lnk_attrs = sorted(get_linking_attributes(shot_poof_example, skip_coords=True), key=len)
				if len(lnk_attrs) == 0:
					# No other choice but to use coordinates
					lnk_attrs = [{"3d_coords": shot_object["3d_coords"]}]
				# We sort values in descriptive order
				sortest_lnk_attrs = dict(sorted(lnk_attrs[0].items(), key=lambda a:ATTRIBUTES[a[0]]["desc_order"]))
				visual_attr_black_list = ["name", "rotation", shot_pnt_attr]
				if tgt_modality == "img":
					visual_attr_black_list.append(shot_tgt_attr)
				all_shot_visual_attrs = {k:v for k,v in shot_object.items() if k not in visual_attr_black_list}
				all_shot_visual_attrs = dict(sorted(all_shot_visual_attrs.items(), key=lambda a: ATTRIBUTES[a[0]]["desc_order"]))
				other_shot_visual_attrs = {k:v for k,v in all_shot_visual_attrs.items() if k not in sortest_lnk_attrs}

				all_shot_visual_attrs_img = ", ".join([ATTRIBUTES[attr]["lnk_attr_img"].format(attr_value=shot_object[attr]) for attr, val in all_shot_visual_attrs.items()])
				other_shot_visual_attrs_img = ", ".join([ATTRIBUTES[attr]["lnk_attr_img"].format(attr_value=shot_object[attr]) for attr, val in other_shot_visual_attrs.items()])

				lnk_attrs_img = ", ".join([ATTRIBUTES[attr]["lnk_attr_img"].format(attr_value=shot_object[attr]) for attr, val in sortest_lnk_attrs.items()])
				shot_str = TASKS[task]["str_shot_cot"].format(
					pointer_str='T' + ATTRIBUTES[shot_pnt_attr]["pointer_str"].format(pnt_attr_value=shot_pnt_attr_value)[1:],
					lnk_attrs_img=lnk_attrs_img,
					scene_format_name=SCENE_FORMAT[scene_format]["name"],
					info_to_retrieve=ATTRIBUTES[shot_tgt_attr]["info_to_retrieve"],
					lnk_attrs_data=json.dumps({k:shot_object[k] for k,v in sortest_lnk_attrs.items()}),
					tgt_attr_value=shot_object[shot_tgt_attr],
					answer_str=json.dumps({'answer': shot_object[shot_tgt_attr]}),
					all_visual_atts_data=json.dumps(all_shot_visual_attrs),
					other_visual_atts_data=json.dumps(other_shot_visual_attrs),
					all_visual_atts_img=all_shot_visual_attrs_img,
					other_visual_atts_img=other_shot_visual_attrs_img
				)
				input_str = re.sub(rf'^{re.escape(question)}.*\n?', f"{question}? {shot_str}\n", input_str, flags=re.MULTILINE)
		example["input_str"] = input_str
	save_jsonl(dataset_dir.replace(".jsonl", "_cot.jsonl"), dataset)

def parse_config() -> Namespace:
	parser = argparse.ArgumentParser(description='arg parser')
	#parser.add_argument('--key_colors', type=str, default="red,green,blue", help="Colors used to target objects uniquely. A maximum of 3 colors. E.g. red,green")
	parser.add_argument('--clever_split', type=str, default="dev")
	parser.add_argument('--out_path', type=str, default="out/meta_correlation")
	parser.add_argument('--dataset_size', type=int, default=5500)
	parser.add_argument('--random_seed', type=int, default=117)
	#parser.add_argument('--variant', type=str, default="orig")  # orig | reverse | point
	parser.add_argument('--tasks', type=str, default="img2data;data2img")  # multimodal = img2data;data2img | unimodal = img2img;data2data
	parser.add_argument('--src_dataset_path', type=str, default="data/meta_correlation/v4/src.jsonl")
	parser.add_argument('--n_shots', type=int, default=2)  # 0 | 1 | 2
	parser.add_argument('--scene_format', type=str, default="json")  # json | yaml | xml | text
	args = parser.parse_args()
	return args

if __name__ == '__main__':
	args = parse_config()
	generate_src_dataset(args)
	"""
	for nshot in [0, 1, 2]:#range(3):
		args.n_shots = nshot
		for sformat in {'json'}:#, 'yaml', 'xml', 'text'}:
			args.scene_format = sformat
			generate_dataset(args)
	"""
	generate_human_eval_set(['1', '2', '3', '4', '5'], task_order=["img2data", "data2img"], ex_per_comb=1)
	#execute_random("data/meta_correlation/v4/um_2shot.jsonl", runs=10)
	#update_with_cot("data/meta_correlation/v4/mm_2shot.jsonl")