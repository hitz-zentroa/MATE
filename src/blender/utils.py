import json
import os
import sys

import numpy as np

import math
from mathutils import Vector

import bpy

BASE_SCENE_BLENDFILE = "resources/clevr/base_scene.blend"
MATERIAL_DIR = "resources/clevr/materials"
PROPERTIES_JSON = "resources/clevr/properties.json"
SHAPE_DIR = "resources/clevr/shapes"


def extract_args(input_argv=None):
	"""
	Pull out command-line arguments after "--". Blender ignores command-line flags
	after --, so this lets us forward command line arguments from the blender
	invocation to our own script.
	"""
	if input_argv is None:
		input_argv = sys.argv
	output_argv = []
	if '--' in input_argv:
		idx = input_argv.index('--')
		output_argv = input_argv[(idx + 1):]
	return output_argv


def parse_args(parser, argv=None):
	return parser.parse_args(extract_args(argv))


def load_materials(material_dir):
	"""
	Load materials from a directory. We assume that the directory contains .blend
	files with one material each. The file X.blend has a single NodeTree item named
	X; this NodeTree item must have a "Color" input that accepts an RGBA value.
	"""
	for fn in os.listdir(material_dir):
		if not fn.endswith('.blend'):
			continue
		name = os.path.splitext(fn)[0]
		filepath = os.path.join(material_dir, fn, 'NodeTree', name)
		bpy.ops.wm.append(filename=filepath)


def delete_object(obj):
	""" Delete a specified blender object """
	for o in bpy.data.objects:
		o.select = False
	obj.select = True
	bpy.ops.object.delete()


def add_object(object_dir, name, scale, loc, theta=0):
	"""
	Load an object from a file. We assume that in the directory object_dir, there
	is a file named "$name.blend" which contains a single object named "$name"
	that has unit size and is centered at the origin.

	- scale: scalar giving the size that the object should be in the scene
	- loc: tuple (x, y) giving the coordinates on the ground plane where the
	object should be placed.
	"""
	# First figure out how many of this object are already in the scene so we can
	# give the new object a unique name
	count = 0
	for obj in bpy.data.objects:
		if obj.name.startswith(name):
			count += 1

	filename = os.path.join(object_dir, '%s.blend' % name, 'Object', name)
	bpy.ops.wm.append(filename=filename)

	# Give it a new name to avoid conflicts
	new_name = '%s_%d' % (name, count)
	bpy.data.objects[name].name = new_name

	# Set the new object as active, then rotate, scale, and translate it
	x, y = loc
	#bpy.context.scene.objects.active = bpy.data.objects[new_name]
	bpy.context.view_layer.objects.active = bpy.data.objects[new_name]
	bpy.context.object.rotation_euler[2] = theta
	bpy.ops.transform.resize(value=(scale, scale, scale))
	bpy.ops.transform.translate(value=(x, y, scale))


def add_material(name, **properties):
	"""
  Create a new material and assign it to the active object. "name" should be the
  name of a material that has been previously loaded using load_materials.
  """
	# Figure out how many materials are already in the scene
	mat_count = len(bpy.data.materials)

	# Create a new material; it is not attached to anything and
	# it will be called "Material"
	bpy.ops.material.new()

	# Get a reference to the material we just created and rename it;
	# then the next time we make a new material it will still be called
	# "Material" and we will still be able to look it up by name
	mat = bpy.data.materials['Material']
	#mat.name = 'Material_%d' % mat_count
	mat.name = f"{name}_{mat_count}" #'Material_%d' % mat_count

	# Attach the new material to the active object
	# Make sure it doesn't already have materials
	obj = bpy.context.active_object
	assert len(obj.data.materials) == 0
	obj.data.materials.append(mat)

	# Find the output node of the new material
	output_node = None
	for n in mat.node_tree.nodes:
		if n.name == 'Material Output':
			output_node = n
			break

	# Add a new GroupNode to the node tree of the active material,
	# and copy the node tree from the preloaded node group to the
	# new group node. This copying seems to happen by-value, so
	# we can create multiple materials of the same type without them
	# clobbering each other
	group_node = mat.node_tree.nodes.new('ShaderNodeGroup')
	group_node.node_tree = bpy.data.node_groups[name]

	# Find and set the "Color" input of the new group node
	for inp in group_node.inputs:
		if inp.name in properties:
			inp.default_value = properties[inp.name]

	# Wire the output of the new group node to the input of
	# the MaterialOutput node
	mat.node_tree.links.new(
		group_node.outputs['Shader'],
		output_node.inputs['Surface'],
	)


def find_closest_vector(color_name_to_rgba, target_tgba_color):

	vectors = np.array(list(color_name_to_rgba.values()))
	target_vector = np.array(target_tgba_color)

	# Compute the Euclidean distances
	distances = np.linalg.norm(vectors - target_vector, axis=1)

	# Find the index of the smallest distance
	closest_index = np.argmin(distances)
	return list(color_name_to_rgba.keys())[closest_index]

def load_properties(properties_path: str = PROPERTIES_JSON):
	with open(properties_path, 'r') as f:
		properties = json.load(f)
		color_name_to_rgba = {}
		for name, rgb in properties['colors'].items():
			rgba = [float(c) / 255.0 for c in rgb] + [1.0]
			color_name_to_rgba[name] = rgba
		#material_mapping = [(v, k) for k, v in properties['materials'].items()]
		material_mapping = properties['materials']
		#object_mapping = [(v, k) for k, v in properties['shapes'].items()]
		object_mapping = properties['shapes']
		size_mapping = properties['sizes']
	return material_mapping, object_mapping, size_mapping, color_name_to_rgba

fd = None
old = None

def switch_logs(enable: bool):
	"""
	Supress blender logs. Not the fanciest way, I know
	@param enable:
	@return:
	"""
	global fd
	global old
	if not enable:
		# redirect output to log file
		logfile = '/tmp/blender_render.log'
		open(logfile, 'a').close()
		old = os.dup(sys.stdout.fileno())
		sys.stdout.flush()
		os.close(sys.stdout.fileno())
		fd = os.open(logfile, os.O_WRONLY)
	elif fd is not None and old is not None:
		# disable output redirection
		os.close(fd)
		os.dup(old)
		os.close(old)

def blend_to_scene(blend_path: str, show_blender_logs:bool = False):
	switch_logs(show_blender_logs)
	bpy.ops.wm.open_mainfile(filepath=blend_path)
	render_args = bpy.context.scene.render
	camera = bpy.data.objects['Camera']
	scene = {
		"camera_location": list(camera.location),
		"image_filename": render_args.filepath,
		"objects": [],
		"metadata": {
			"width": render_args.resolution_x,
			"height": render_args.resolution_y,
			"resolution_percentage": render_args.resolution_percentage,
			#"sample_as_light": bpy.data.worlds['World'].cycles.sample_as_light,
			"blur_glossy": bpy.context.scene.cycles.blur_glossy,
			"render_num_samples": bpy.context.scene.cycles.samples,
			#"transparent_min_bounces": bpy.context.scene.cycles.transparent_min_bounces,
			#"transparent_max_bounces": bpy.context.scene.cycles.transparent_max_bounces,
			"render_tile_size": bpy.context.scene.cycles.tile_size,
			"Lamp_Key": list(bpy.data.objects['Lamp_Key'].location),
			"Lamp_Back": list(bpy.data.objects['Lamp_Back'].location),
			"Lamp_Fill": list(bpy.data.objects['Lamp_Fill'].location),
		}
	}
	material_mapping, object_mapping, _, color_name_to_rgba = load_properties()
	for blend_object in bpy.context.scene.objects:
		if len([k for k, v in object_mapping.items() if v in blend_object.name]) == 0:
			continue
		scene_object = {
			"name": blend_object.name,
			"shape": [k for k, v in object_mapping.items() if v in blend_object.name][0],
			"size": math.ceil(blend_object.scale[0]*1000)/1000,  # We do ceil to match the predefined sizes in CLEVR properties.json
			"material": [k for k, v in material_mapping.items() if v in blend_object.active_material.name][0],
			"3d_coords": list(blend_object.location),
			"rotation": blend_object.rotation_euler[2],
			"color": find_closest_vector(color_name_to_rgba, list(bpy.data.materials[blend_object.active_material.name].node_tree.nodes["Group"].inputs[0].default_value)),
		}
		scene["objects"].append(scene_object)
	switch_logs(enable=True)
	return scene


def scene_to_blend(scene, output_blend_path: str = None, is_clevr_scene: bool = False, show_blender_logs:bool = False):
	switch_logs(show_blender_logs)
	# Load the main blend file
	bpy.ops.wm.open_mainfile(filepath=BASE_SCENE_BLENDFILE)

	# Load materials
	load_materials(MATERIAL_DIR)

	# Set render arguments so we can get pixel coordinates later.
	# We use functionality specific to the CYCLES renderer so BLENDER_RENDER
	# cannot be used.
	metadata = scene["metadata"]
	render_args = bpy.context.scene.render
	output_blend_path = output_blend_path if output_blend_path is not None else scene["image_filename"]
	render_args.filepath = output_blend_path

	render_args.resolution_x = metadata["width"]
	render_args.resolution_y = metadata["height"]
	render_args.resolution_percentage = metadata["resolution_percentage"]

	# Some CYCLES-specific stuff
	bpy.data.worlds['World'].cycles.sample_as_light = True #metadata["sample_as_light"] # I don't think this changes anything
	bpy.context.scene.cycles.blur_glossy = metadata["blur_glossy"]
	bpy.context.scene.cycles.samples = metadata["render_num_samples"]
	bpy.context.scene.cycles.transparent_min_bounces = 8 #metadata["transparent_min_bounces"] # I don't think this changes anything
	bpy.context.scene.cycles.transparent_max_bounces = 8 #metadata["transparent_max_bounces"] # I don't think this changes anything
	bpy.context.scene.cycles.tile_size = metadata["render_tile_size"]

	# Recalculate the quaternion rotation based on saved direction vectors
	camera = bpy.data.objects['Camera']

	# Assuming you saved the camera location separately in scene_struct
	if 'camera_location' in scene:
		camera.location = Vector(scene['camera_location'])
	# TODO find camera "direction"
	else:
		# Retrieve saved directions from the scene_struct (just for CLEVR)
		cam_behind = Vector(scene['directions']['behind'])
		cam_left = Vector(scene['directions']['left'])
		cam_up = Vector(scene['directions']['above'])

		# The camera's 'right' direction is the negative of 'left', and 'below' is the negative of 'above'
		cam_right = -cam_left
		cam_down = -cam_up
		cam_front = -cam_behind

		# To restore camera orientation, compute the quaternion that aligns the camera
		# Restore the camera's orientation using the direction vectors
		# Using the 'front', 'up', and 'left' vectors to compute the correct rotation
		quat_front = cam_front.to_track_quat('Z', 'Y')  # 'Z' as front and 'Y' as up

		# Set the camera's rotation
		camera.rotation_quaternion = quat_front

	# Add random jitter to lamp positions
	# TODO get the default locations
	bpy.data.objects['Lamp_Key'].location = metadata["Lamp_Key"]
	bpy.data.objects['Lamp_Back'].location = metadata["Lamp_Back"]
	bpy.data.objects['Lamp_Fill'].location = metadata["Lamp_Fill"]

	material_mapping, object_mapping, size_mapping, color_name_to_rgba = load_properties()

	for scene_object in scene["objects"]:
		# Retrieve object properties from the saved data
		obj_name_out = scene_object['shape']
		size_name = scene_object['size']
		mat_name_out = scene_object['material']
		location = Vector(scene_object['3d_coords'])
		rotation = scene_object['rotation']
		color_name = scene_object['color']

		# Map the object name back to its actual file name in Blender
		obj_name = object_mapping[scene_object['shape']]
		mat_name = material_mapping[scene_object['material']]

		# Get color in RGBA format
		rgba = color_name_to_rgba[color_name]

		# Retrieve object size
		r = size_mapping[size_name] if is_clevr_scene else scene_object['size']

		# If the object is a cube, adjust its size as done in the original function
		if obj_name == 'Cube':
			r /= math.sqrt(2)

		# Add the object back to the scene
		add_object(SHAPE_DIR, obj_name, r, (location.x, location.y), theta=rotation)
		obj = bpy.context.object

		# Attach the material and color
		add_material(mat_name, Color=rgba)

		# Set the object's location and rotation
		obj.location = location
		obj.rotation_euler[2] = rotation  # Z-axis rotation

	bpy.ops.wm.save_as_mainfile(filepath=output_blend_path)

	switch_logs(enable=True)
	return output_blend_path


def blend_to_img(blend_path: str, output_img_path: str = None, width: int = None, height: int = None,
				 resolution_percentage: float = None, use_gpu: bool = True, max_render_attempts: int = 10, show_blender_logs:bool = False):
	"""
	@param blend_path: path to blend file to render
	@param output_img_path:
	@param width:
	@param height:
	@param resolution_percentage:
	@param use_gpu:
	@param max_render_attempts:
	@param show_blender_logs:
	@return:

	"""
	switch_logs(show_blender_logs)
	# Load the main blendfile
	bpy.ops.wm.open_mainfile(filepath=blend_path)

	render_args = bpy.context.scene.render
	render_args.engine = "CYCLES"
	# allow user to specify where to save images, if desired
	output_img_path = output_img_path if output_img_path is not None else blend_path.replace(".blend", ".png")
	render_args.filepath = output_img_path
	render_args.resolution_x = width if width is not None else render_args.resolution_x
	render_args.resolution_y = height if height is not None else render_args.resolution_y
	render_args.resolution_percentage = resolution_percentage if resolution_percentage is not None else render_args.resolution_percentage
	bpy.context.scene.render.engine = 'CYCLES'
	if use_gpu:
		# Blender changed the API for enabling CUDA at some point
		if bpy.app.version < (2, 78, 0):
			bpy.context.user_preferences.system.compute_device_type = 'CUDA'
			bpy.context.user_preferences.system.compute_device = 'CUDA_0'
		else:
			if sys.platform == "darwin":
				bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'METAL'
				bpy.context.preferences.addons['cycles'].preferences.refresh_devices()
				bpy.context.preferences.addons['cycles'].preferences.devices['Apple M1 Pro (GPU - 16 cores)'].use = True
			else:
				bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
		bpy.context.scene.cycles.device = 'GPU'

	attempts = 0
	while attempts < max_render_attempts:
		try:
			bpy.ops.render.render(write_still=True)
			break
		except Exception as e:
			print(e)
			attempts += 1
	switch_logs(enable=True)
	return output_img_path


def scene_to_img(scene, is_clevr_scene: bool = False, output_img_path: str = None, width: int = None,
				 height: int = None, resolution_percentage: float = None, use_gpu: bool = True,
				 max_render_attempts: int = 10, show_blender_logs:bool = False):
	blend_path = scene_to_blend(scene, is_clevr_scene=is_clevr_scene, show_blender_logs=show_blender_logs)
	return blend_to_img(blend_path, output_img_path=output_img_path, width=width, height=height,
						resolution_percentage=resolution_percentage, use_gpu=use_gpu,
						max_render_attempts=max_render_attempts, show_blender_logs=show_blender_logs)
