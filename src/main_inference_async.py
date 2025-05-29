import argparse
import os
import time

from datasource.utils import load_jsonl, safe_file_name, chunks
from model.model_factory import get_async_vlm


def run_inference(args):
	base_model: str = args.base_model
	dataset_path: str = args.dataset_path
	output_batch_path: str = args.output_batch_path
	output_path: str = args.output_path
	img_dir: str = args.img_dir
	batch_size: int = args.batch_size
	max_new_tokens: int = args.max_new_tokens
	test_n : int = args.test_n

	dataset = load_jsonl(dataset_path)
	if test_n > 0: # For testing purposes
		dataset = dataset[:test_n]
	# dataset_utils = get_dataset_utils(dataset_path)
	
	# Create endpoint
	endpoint = get_async_vlm(base_model=base_model, img_dir=img_dir, max_gen_tokens=max_new_tokens)
	chunk_dataset = chunks(dataset, batch_size)
	
	processing_batches = []
	processing_batch_metadata = []
	processing_output_files = []

	for idx, mini_batch in enumerate(chunk_dataset):
		# Define files to be loaded and output
		batch_filename = f"batch_file_{os.path.splitext(os.path.basename(dataset_path))[0]}_{safe_file_name(base_model)}_part{idx}.jsonl"
		output_filename = f"results_file_{os.path.splitext(os.path.basename(dataset_path))[0]}_{safe_file_name(base_model)}_part{idx}.jsonl"
		batch_file = os.path.join(os.path.dirname(dataset_path), batch_filename) if output_batch_path is None else os.path.join(output_batch_path, batch_filename)
		output_file = os.path.join(os.path.dirname(dataset_path), output_filename) if output_path is None else os.path.join(output_path, output_filename)

		mini_batch_metadata = endpoint.create_and_send_batch(mini_batch, batch_file)
		
		processing_batches.append(mini_batch)
		processing_batch_metadata.append(mini_batch_metadata)
		processing_output_files.append(output_file)
	
	while len(processing_batches) > 0:
		time.sleep(30)
		for i in range(len(processing_batches) - 1, -1, -1):
			# Check if it is finished and retrieve the results
			batch_job = endpoint.check_batch_status_and_retrieve(processing_batch_metadata[i].id)
			if batch_job is not None:
				# Save the results
				endpoint.save_processed_batch(processing_batches[i], batch_job, processing_output_files[i])
				processing_batches.pop(i)
				processing_batch_metadata.pop(i)
				processing_output_files.pop(i)
	
	



def parse_config():
	parser = argparse.ArgumentParser(description='arg parser')
	parser.add_argument('--base_model', type=str, default="gpt-4o-2024-11-20")
	parser.add_argument('--batch_size', type=int, default=768, help='')
	parser.add_argument('--max_new_tokens', type=int, default=80, help='')
	parser.add_argument('--dataset_path', type=str, default="data/mate/dev/um_2shot.jsonl", help='')
	parser.add_argument('--output_batch_path', type=str, default="out/async_batches", help='Just for OpenAI')
	parser.add_argument('--output_path', type=str, default="out/async_predictions", help='')
	parser.add_argument('--img_dir', type=str, default="data/mate/dev/img", help='')
	parser.add_argument('--test_n', type=int, default=0, help='If set to a positive integer, it will only process that many examples')
	args = parser.parse_args()
	return args


if __name__ == '__main__':
	args = parse_config()
	run_inference(args)
