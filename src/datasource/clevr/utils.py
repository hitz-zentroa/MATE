import os.path

from datasource.utils import load_json

DATASET_NAME = "CLEVR"
ROOT_PATH = "data/clevr/v1"

FILE_NAMES = {
	"train": {
		"scenes": "CLEVR_train_scenes.json",
		"questions": "CLEVR_train_questions.json"
	},
	"dev": {
		"scenes": "CLEVR_val_scenes.json",
		"questions": "CLEVR_val_questions.json"
	},
	"test": {
		#"scenes": "",
		"questions": "CLEVR_test_questions.json"
	}
}

DEFAULT_METADATA = {
		"width": 480,
		"height": 320,
		"render_tile_size": 256,
		"resolution_percentage": 100,
		"blur_glossy": 2.0,
		"sample_as_light": True,
		"render_num_samples": 512,
		"transparent_min_bounces": 8,
		"transparent_max_bounces": 8,
		#"camera_location": None, # We should add this in the future
		"Lamp_Key": [6.446710109710693, -2.905174732208252, 4.258398056030273],
		"Lamp_Back": [-1.168501377105713, 2.646024465560913, 5.815736770629883],
		"Lamp_Fill": [-4.671124458312988, -4.01359748840332, 3.0112245082855225],
	}

def load_dataset(split: str, variant: str):
	path = os.path.join(ROOT_PATH, variant, FILE_NAMES[split][variant])
	dataset = load_json(path, indexed=True)
	for scene in dataset["scenes"]:
		scene["metadata"] = DEFAULT_METADATA
	return dataset[variant]