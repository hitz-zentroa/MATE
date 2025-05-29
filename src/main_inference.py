import json
import os
import time

import argparse

from tqdm import tqdm

from datasource.datasource_factory import get_dataset_utils
from datasource.utils import load_jsonl, chunks, save_jsonl, safe_file_name, NpEncoder, reindex_by
from model.model_factory import get_vlm


def run_inference(args):
	base_model: str = args.base_model
	batch_size: int = args.batch_size
	batch_delay: float = args.batch_delay
	dataset_path: str = args.dataset_path
	output_path: str = args.output_path
	img_dir: str = args.img_dir
	max_new_tokens: int = args.max_new_tokens
	include_prompt: bool = args.include_prompt
	test_n: int = args.test_n

	model = get_vlm(base_model, max_new_tokens, img_dir)
	dataset = load_jsonl(dataset_path)
	if test_n > 0: # For testing purposes
		dataset = dataset[:test_n]
	dataset_utils = get_dataset_utils(dataset_path)
	pred_file_name = f"{os.path.splitext(os.path.basename(dataset_path))[0]}_{safe_file_name(model.model_name)}.jsonl"
	output_path = os.path.join(os.path.dirname(dataset_path), pred_file_name) if output_path is None else os.path.join(output_path, pred_file_name)

	file_mode = "w"
	if args.continue_from_existing and os.path.isfile(output_path):
		file_mode = 'a'
		prev_dataset = load_jsonl(output_path, indexed=True, id_key_name="example_id")
		dataset = [ex for ex in dataset if ex["example_id"] not in prev_dataset]
	chunk_dataset = chunks(dataset, batch_size)
	for batch in tqdm(chunk_dataset):
		if batch_delay > 0:
			time.sleep(batch_delay)
		conversations = [dataset_utils.build_conversation(example) for example in batch]
		inputs, predictions = model.batch_generate(conversations=conversations)
		with open(output_path, file_mode, encoding='utf8') as outfile:
			for example, input_str, prediction in zip(batch, inputs, predictions):
				example["model"] = model.model_name
				example["prediction"] = prediction
				if include_prompt:
					example["prompt"] = input_str
				json.dump(example, outfile, ensure_ascii=False, cls=NpEncoder)
				outfile.write('\n')
			file_mode = "a"


def parse_config():
	parser = argparse.ArgumentParser(description='arg parser')
	parser.add_argument('--base_model', type=str, default="llava-hf/llava-1.5-7b-hf")
	# parser.add_argument('--context_size', type=int, default=-1, help='context size during fine-tuning')
	# parser.add_argument('--flash_attn', type=bool, default=False, help='')
	# parser.add_argument('--temperature', type=float, default=0.6, help='')
	# parser.add_argument('--top_p', type=float, default=0.9, help='')
	parser.add_argument('--max_new_tokens', type=int, default=80, help='')
	parser.add_argument('--dataset_path', type=str, default="data/mate/dev/eval.jsonl", help='')
	parser.add_argument('--output_path', type=str, default="out/predictions", help='')
	parser.add_argument('--img_dir', type=str, default="data/mate/dev/img", help='')
	parser.add_argument('--batch_size', type=int, default=5, help='')
	parser.add_argument('--batch_delay', type=float, default=0.0, help='Delay in seconds between each batch.')
	parser.add_argument('--include_prompt', type=bool, default=True, help='')
	parser.add_argument('--continue_from_existing', type=bool, default=True, help='')
	parser.add_argument('--test_n', type=int, default=0, help='If set to a positive integer, it will only process that many examples')
	args = parser.parse_args()
	return args


if __name__ == '__main__':
	args = parse_config()
	run_inference(args)
