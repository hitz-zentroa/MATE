import base64
import json
import os
import PIL
import pathlib
from typing import Literal
import time

from openai import OpenAI
import google.generativeai as genai
from anthropic import Anthropic

from openai.types.batch import Batch
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request
from anthropic.types.messages.message_batch import MessageBatch

from datasource.utils import load_jsonl, safe_file_name, NpEncoder
# from datasource.datasource_factory import get_dataset_utils
from model.vlm_models import LuminousVLM


# Set this variables in the os environment with the correct keys
OPENAI_API_KEY = ""
GOOGLE_API_KEY = ""
ANTHROPIC_API_KEY = ""


### SYNCHRONOUS ENDPOINTS

class OpenAIEndpoint(LuminousVLM):
	DEFAULT_MODEL = "gpt-4o-2024-11-20"
	SUPPORTED_MODELS = {
		"gpt-4o-2024-11-20", # use "gpt-4o-latest" for the latest version
		"gpt-4o-mini",
		"o1-latest",
		"gpt-3.5-turbo",
	}

	def __init__(self, base_model: str = DEFAULT_MODEL, img_dir: str = "",  img_ext: str ="png", show_warnings: Literal['never', 'once', 'allways'] = 'once'):
		"""
		@param base_model: name of the GPT model to use (e.g. "gpt-4o-mini")
		@param img_dir: this should be the absolute path to the image
		@param img_ext: image extension used for input images. Supported: "png", "gif", "jpeg" and "webp"
		"""
		self.client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
		self.img_ext = img_ext
		super().__init__(base_model=base_model, img_dir=img_dir, show_warnings=show_warnings, processor="")

	def encode_image(self, image_path: str) -> str:
		with open(image_path, "rb") as image_file:
			return base64.b64encode(image_file.read()).decode("utf-8")
	
	def format_conversation_step(self, text: str, role: Literal['user', 'assistant', 'system', 'tool'] = "user", img: str = None) -> dict:
		if img is not None:
			base64_image = self.encode_image(os.path.join(self.img_dir, img))
			return {"role": role, "content": [{"type": "text", "text": text}, {"type": "image_url", "image_url": {"url": f"data:image/{self.img_ext};base64,{base64_image}"}}]}
		else:
			return {"role": role, "content": text}

	def call_openai_api(self, messages: list[dict]) -> str:
		response = self.client.chat.completions.create(
			model=self.model_name,
			messages=messages
		)
		return response.choices[0].message.content

	def simple_generate(self, user_text: str, image_path: str) -> str:
		formatted_data = [self.format_conversation_step(role="user", text=user_text, img=image_path)]
		response = self.call_openai_api(messages=formatted_data)
		return response
	
	def generate(self, conversation: list[(str, str, str)]) -> str:
		formatted_conversation = [self.format_conversation_step(role=role, text=text, img=img) for role, text, img in conversation]
		response = self.call_openai_api(messages=formatted_conversation)
		return response

	def batch_generate(self, conversations: list[list[(str, str, str)]]) -> tuple[list[str], list[str]]:
		"""
		We don't support batch generate with OpenAI models yet. For now, it will be just a sequential inference
		@param conversations: ('role', 'text', 'img_path')
		@return:
		"""
		self.show_warning("WARNING!! We don't support batch generate with OpenAI models yet. For now, it will be just a sequential inference", "batch_gen_openai")
		return ["" for _ in conversations], [self.generate(conversation) for conversation in conversations] # TODO: ADAPT CODE TO RETURN INPUT_PROMPT



class GoogleEndpoint(LuminousVLM):
	
	DEFAULT_MODEL = "gemini-1.5-flash"
	SUPPORTED_MODELS = {
		"gemini-2.0-flash-exp", # THIS IS FREE
		"gemini-1.5-flash",
		"gemini-1.5-flash-8b",
		"gemini-1.5-pro"
	}

	def __init__(self, base_model: str = DEFAULT_MODEL, img_dir: str = "", upload_media: bool = False, show_warnings: Literal['never', 'once', 'allways'] = 'once'):
		"""
		@param base_model: name of the Gemini model to use (e.g. "gemini-2.0-flash-exp")
		@param img_dir: this should be the absolute path to the image
		@param upload_media: set to True for an efficient way to send large image payloads
		"""
		genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
		self.client = genai.GenerativeModel(base_model)
		self.upload_media = upload_media
		super().__init__(base_model=base_model, img_dir=img_dir, show_warnings=show_warnings, processor="")

	def format_conversation_step(self, text: str, role: Literal['user', 'assistant', 'system', 'tool'] = "user", img: str = None) -> dict:
		return {"role": role, "parts": [text, img] if img is not None else [text]}

	def upload_files(self, imgs:list[str]) -> list[str]:
		return [genai.upload_file(os.path.join(self.img_dir, img)) if img is not None else None for img in imgs]

	def simple_generate(self, user_text: str, image_path: str) -> str:
		if self.upload_media:
			image_content = self.upload_files([image_path])[0]
		else:
			image_content = PIL.Image.open(os.path.join(self.img_dir, image_path))
		response = self.client.generate_content([user_text, image_content])
		return response.text.strip()

	def generate(self, conversation: list[(str, str, str)]) -> str:
		if self.upload_media:
			current_imgs = self.upload_files([img for _, _, img in conversation])
		else:
			current_imgs = [PIL.Image.open(os.path.join(self.img_dir, img)) for _, _, img in conversation]
		formatted_conversation = [self.format_conversation_step(role=role, text=text, img=current_img) for (role, text, _), current_img in zip(conversation[:-1], current_imgs[:-1])]
		chat = self.client.start_chat(history=formatted_conversation)
		response = chat.send_message(self.format_conversation_step(role=conversation[-1][0], text=conversation[-1][1], img=current_imgs[-1]))
		return response.text.strip()

	def batch_generate(self, conversations: list[list[(str, str, str)]]) -> tuple[list[str], list[str]]:
		"""
		We don't support batch generate with Google models yet. For now, it will be just a sequential inference
		@param conversations: ('role', 'text', 'img_path')
		@return:
		"""
		self.show_warning("WARNING!! We don't support batch generate with Google models yet. For now, it will be just a sequential inference", "batch_gen_google")
		return ["" for _ in conversations], [self.generate(conversation) for conversation in conversations] # TODO: ADAPT CODE TO RETURN INPUT_PROMPT


# Anthropic only supports base64 encoded images
class AnthropicEndpoint(LuminousVLM):
	
	DEFAULT_MODEL = "claude-3-5-sonnet-20241022"
	SUPPORTED_MODELS = {
		"claude-3-5-sonnet-20241022", # use "....sonnet-latest" for the latest version
		"claude-3-5-haiku-20241022",
		"claude-3-opus-20240229"
	}
	
	def __init__(self, base_model: str = DEFAULT_MODEL, img_dir: str = "", img_ext: str ="png", show_warnings: Literal['never', 'once', 'allways'] = 'once'):
		"""
		@param base_model: name of the Anthropic model to use
		@param img_dir: this should be the absolute path to the image
		@param img_ext: image extension used for input images. Supported: "png", "gif", "jpeg" and "webp"
		"""
		self.client = Anthropic(api_key=os.environ['ANTHROPIC_API_KEY'])
		self.img_ext = img_ext
		super().__init__(base_model=base_model, img_dir=img_dir, show_warnings=show_warnings, processor="")
	
	def encode_image(self, image_path: str) -> str:
		with open(image_path, "rb") as f:
			encoded_string = base64.standard_b64encode(f.read())
		return encoded_string.decode("utf-8")

	def format_conversation_step(self, text: str, role: Literal['user', 'assistant', 'system', 'tool'] = "user", image_path: str = None) -> dict:
		if image_path is not None:
			base64_image = self.encode_image(os.path.join(self.img_dir, image_path))
			return {
				"role": role, 
				"content": [
					{"type": "text", "text": text}, 
					{"type": "image", "source": {"type": "base64", "media_type": f"image/{self.img_ext}", "data": base64_image}},
				]
			}  
		else:
			return {"role": role, "content": text}

	def call_anthropic_api(self, messages: list[dict]) -> str:
		# print(messages)
		response = self.client.messages.create(
			model=self.model_name,
			max_tokens=2048,
			messages=messages
		)
		return response.content[0].text
	
	def simple_generate(self, user_text: str, image_path: str) -> str:
		messages = [self.format_conversation_step(user_text, "user", image_path)]
		response = self.call_anthropic_api(messages=messages)
		return response

	def generate(self, conversation: list[(str, str, str)]) -> str:
		formatted_conversation = [self.format_conversation_step(role=role, text=text, image_path=img) for role, text, img in conversation]
		response = self.call_anthropic_api(messages=formatted_conversation)
		return response

	def batch_generate(self, conversations: list[list[(str, str, str)]]) -> tuple[list[str], list[str]]:
		"""
		We don't support batch generate with Anthropic models yet. For now, it will be just a sequential inference
		@param conversations: ('role', 'text', 'img_path')
		@return:
		"""
		self.show_warning("WARNING!! We don't support batch generate with Anthropic models yet. For now, it will be just a sequential inference", "batch_gen_anthropic")
		return ["" for _ in conversations], [self.generate(conversation) for conversation in conversations] # TODO: ADAPT CODE TO RETURN INPUT_PROMPT


### ASYNCHRONOUS ENDPOINTS

class OpenAIAsyncEndpoint(LuminousVLM):
	DEFAULT_MODEL = "gpt-4o-2024-11-20"
	SUPPORTED_MODELS = {
		"gpt-4o-2024-11-20", # use "gpt-4o-latest" for the latest version
		"gpt-4o-mini",
		"o1-latest",
		"gpt-3.5-turbo",
	}

	def __init__(self, base_model: str = DEFAULT_MODEL, max_new_tokens: int = 80, img_dir: str = "",  img_ext: str ="png", show_warnings: Literal['never', 'once', 'allways'] = 'once'):
		"""
		@param base_model: name of the GPT model to use (e.g. "gpt-4o-mini")
		@param img_dir: this should be the absolute path to the image
		@param img_ext: image extension used for input images. Supported: "png", "gif", "jpeg" and "webp"
		"""
		self.client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
		self.img_ext = img_ext
		super().__init__(base_model=base_model, img_dir=img_dir, show_warnings=show_warnings, processor="", max_new_tokens=max_new_tokens)

	def encode_image(self, image_path: str) -> str:
		with open(image_path, "rb") as image_file:
			return base64.b64encode(image_file.read()).decode("utf-8")
	
	def format_conversation_step(self, text: str, role: Literal['user', 'assistant', 'system', 'tool'] = "user", img: str = None) -> dict:
		if img is not None:
			base64_image = self.encode_image(os.path.join(self.img_dir, img))
			return {"role": role, "content": [{"type": "text", "text": text}, {"type": "image_url", "image_url": {"url": f"data:image/{self.img_ext};base64,{base64_image}"}}]}
		else:
			return {"role": role, "content": text}
	
	def create_and_send_batch(self, dataset: list[dict], batch_file: str) -> Batch:

		batch = []
		for elem in dataset:
			instance = {
				"custom_id": elem['example_id'],
				"method": "POST",
				"url": "/v1/chat/completions",
				"body": {
					# This is what you would have in your Chat Completions API call
					"model": self.model_name,
					"temperature": 0.0,
					"response_format": { 
						"type": "json_object"
					},
					"max_tokens": self.max_new_tokens,
					"messages": [self.format_conversation_step(text=elem['input_str'], role="user", img=elem['image'] if elem['image'] is not None else None)]
				}
			}
			batch.append(instance)
		
		with open(batch_file, 'w') as f:
			for obj in batch:
				f.write(json.dumps(obj) + '\n')
		
		print("* File created!")

		file_metadata = self.client.files.create(
			file=open(batch_file, "rb"),
			purpose="batch"
		)

		print("* File uploaded!\n", file_metadata)

		batch_job = self.client.batches.create(
			input_file_id=file_metadata.id,
			endpoint="/v1/chat/completions",
			completion_window="24h",
			metadata={
				"description": f"OpenAI MATE Experiment: {batch_file}"
			}
		)

		print("* Batch job sent!\n", batch_job)
		
		return batch_job

	def check_batch_status_and_retrieve(self, batch_id: str) -> Batch:
		batch_job = self.client.batches.retrieve(batch_id)
		if batch_job.status == "completed":
			print(f"Batch {batch_id} has finished! Retrieving results...", flush=True)
			return batch_job
		else:
			completed = batch_job.request_counts.completed
			total = batch_job.request_counts.total
			if total == 0:
				print(f"Batch {batch_id} status: {batch_job.status}.\n * Waiting for completion...", flush=True)
			else:
				print(f"Batch {batch_id} status: {batch_job.status}.\n * Waiting for completion... {completed}/{total} ({100*completed/total:.2f}%) requests completed.", flush=True)
			return None
	
	def save_processed_batch(self, dataset: list[dict], batch_job: Batch, output_file: str):
		
		tmp_file = f"tmp_{safe_file_name(batch_job.id)}.jsonl"

		# Get the results from the batch
		result = self.client.files.content(batch_job.output_file_id).content
		with open(tmp_file, 'wb') as file:
			file.write(result)

		predictions = load_jsonl(tmp_file)
		prediction_ids = [result['custom_id'] for result in predictions]
		
		# Write the results and merge them with the original dataset
		with open(output_file, 'w') as f:
			for elem in dataset:
				cur_idx = prediction_ids.index(elem['example_id'])
				assert predictions[cur_idx]['custom_id'] == elem['example_id']
				elem["model"] = self.model_name
				elem["prediction"] = predictions[cur_idx]['response']['body']['choices'][0]['message']['content']
				json.dump(elem, f, ensure_ascii=False, cls=NpEncoder)
				f.write('\n')
		
		pathlib.Path.unlink(tmp_file)
		"""
		result = self.client.files.content(batch_job.output_file_id).content
		with open(output_file, 'wb') as file:
			file.write(result)
		"""

	def cancel_batch(self, batch_id: str):
		self.client.batches.cancel(batch_id)
		print(f"Batch {batch_id} has been cancelled.")
	
	def batch_generate(self, conversations):
		raise NotImplementedError


# Anthropic only supports base64 encoded images
class AnthropicAsyncEndpoint(LuminousVLM):
	
	DEFAULT_MODEL = "claude-3-5-sonnet-20241022"
	SUPPORTED_MODELS = {
		"claude-3-5-sonnet-20241022", # use "....sonnet-latest" for the latest version
		"claude-3-5-haiku-20241022",
		"claude-3-opus-20240229"
	}
	
	def __init__(self, base_model: str = DEFAULT_MODEL, max_new_tokens: int = 80, img_dir: str = "", img_ext: str ="png", show_warnings: Literal['never', 'once', 'allways'] = 'once'):
		"""
		@param base_model: name of the Anthropic model to use
		@param img_dir: this should be the absolute path to the image
		@param img_ext: image extension used for input images. Supported: "png", "gif", "jpeg" and "webp"
		"""
		self.client = Anthropic(api_key=os.environ['ANTHROPIC_API_KEY'])
		self.img_ext = img_ext
		super().__init__(base_model=base_model, max_new_tokens=max_new_tokens, img_dir=img_dir, show_warnings=show_warnings, processor="")
	
	def encode_image(self, image_path: str) -> str:
		with open(image_path, "rb") as f:
			encoded_string = base64.standard_b64encode(f.read())
		return encoded_string.decode("utf-8")

	def format_conversation_step(self, text: str, role: Literal['user', 'assistant', 'system', 'tool'] = "user", image_path: str = None) -> dict:
		if image_path is not None:
			base64_image = self.encode_image(os.path.join(self.img_dir, image_path))
			return {
				"role": role, 
				"content": [
					{"type": "text", "text": text}, 
					{"type": "image", "source": {"type": "base64", "media_type": f"image/{self.img_ext}", "data": base64_image}},
				]
			}  
		else:
			return {"role": role, "content": text}

	def create_and_send_batch(self, dataset: list[dict], batch_file: str = None) -> MessageBatch:

		batch = []
		for elem in dataset:
			instance = Request(
				custom_id=elem['example_id'],
				params=MessageCreateParamsNonStreaming(
					model=self.model_name,
					temperature=0.0,
					max_tokens=self.max_new_tokens,
					messages=[self.format_conversation_step(text=elem['input_str'], role="user", image_path=elem['image'] if elem['image'] is not None else None)]
				)
			)
			batch.append(instance)
		
		batch_job = self.client.messages.batches.create(
			requests=batch
		)

		print("* Batch job sent!\n", batch_job)
		
		return batch_job

	def check_batch_status_and_retrieve(self, batch_id: str) -> Batch:
		batch_job = self.client.messages.batches.retrieve(batch_id)
		if batch_job.processing_status == "ended":
			print(f"Batch {batch_id} has finished! Retrieving results...")
			return batch_job
		else:
			completed = batch_job.request_counts.succeeded
			total = batch_job.request_counts.processing
			if total == 0:
				print(f"Batch {batch_id} status: {batch_job.processing_status}.\n * Waiting for completion...")
			else:
				print(f"Batch {batch_id} status: {batch_job.processing_status}.\n * Waiting for completion... {completed}/{total} ({100*completed/total:.2f}%) requests completed.")
			return None
	
	def save_processed_batch(self, dataset: list[dict], batch_job: MessageBatch, output_file: str):
				
		# Write the results and merge them with the original dataset
		with open(output_file, 'w') as f:
			all_results = [result for result in self.client.messages.batches.results(batch_job.id)]
			all_result_ids = [result.custom_id for result in all_results]
			for elem in dataset:
				cur_idx = all_result_ids.index(elem['example_id'])
				assert all_results[cur_idx].custom_id == elem['example_id']
				elem["model"] = self.model_name
				elem["prediction"] = all_results[cur_idx].result.message.content[0].text
				json.dump(elem, f, ensure_ascii=False, cls=NpEncoder)
				f.write('\n')
		"""
		with open(output_file, 'w') as f:
			for result in self.client.messages.batches.results(batch_job.id):
				f.write(json.dumps(result) + '\n')
		"""

	def cancel_batch(self, batch_id: str):
		self.client.messages.batches.cancel(batch_id)
		print(f"Batch {batch_id} has been cancelled.")
	
	def batch_generate(self, conversations):
		raise NotImplementedError


if __name__ == '__main__':
	
	prompt = "What do you see in this image?"
	
	# The file was originally jpg, but it is not supported by anthropic (unless it works with jpeg extension)
	img_dir = "../blendr_ambiguity/storyboards/images/"
	img_file = "test_0002-1.png" 
	img_ext = "png"
	
	### SYNCHRONOUS ENDPOINTS
	"""
	endpoint = OpenAIEndpoint(img_dir=img_dir)
	response = endpoint.simple_generate(user_text=prompt, image_path=img_file)
	print(f"OpenAI Response: {response}")
	response = endpoint.generate([('user', prompt, img_file)])
	print(f"OpenAI Response: {response}")
	"""

	endpoint = GoogleEndpoint(img_dir=img_dir)
	response = endpoint.simple_generate(user_text=prompt, image_path=img_file)
	print(f"Google Response: {response}")
	response = endpoint.generate([('user', prompt, img_file)])
	print(f"Google Response: {response}")

	"""
	endpoint = AnthropicEndpoint(img_dir=img_dir, img_ext=img_ext)
	response = endpoint.simple_generate(user_text=prompt, image_path=img_file)
	print(f"Anthropic Response: {response}")
	response = endpoint.generate([('user', prompt, img_file)])
	print(f"Anthropic Response: {response}")
	"""

	### ASYNCHRONOUS ENDPOINTS
	
	dataset_path = "../../data/mate/dev/um_2shot.jsonl"  # um_2shot.jsonl, mm_2shot.jsonl
	img_dir = "../../data/mate/dev/img"
	
	dataset = [elem for elem in load_jsonl(dataset_path)[:10] if 'img' not in elem['task']][:2] # Only data2data tasks

	output_batch_path = "../out/meta_correlation/async_batches"
	output_pred_path = "../out/meta_correlation/async_predictions"

	### OpenAI Async
	model_name = "gpt-4o-2024-11-20"

	# Define files to be loaded and output
	batch_filename = f"batch_file_{os.path.splitext(os.path.basename(dataset_path))[0]}_{safe_file_name(model_name)}.jsonl"
	output_filename = f"pred_file_{os.path.splitext(os.path.basename(dataset_path))[0]}_{safe_file_name(model_name)}.jsonl"
	batch_file = os.path.join(os.path.dirname(dataset_path), batch_filename) if output_batch_path is None else os.path.join(output_batch_path, batch_filename)
	output_file = os.path.join(os.path.dirname(dataset_path), output_filename) if output_pred_path is None else os.path.join(output_pred_path, output_filename)

	# Create endpoint and send batch
	endpoint = OpenAIAsyncEndpoint(base_model=model_name, img_dir=img_dir)
	batch_metadata = endpoint.create_and_send_batch(dataset, batch_file)
	
	# Check if it is finished and retrieve the results
	batch_job = endpoint.check_batch_status_and_retrieve(batch_metadata.id)
	while batch_job is None:
		time.sleep(15)
		batch_job = endpoint.check_batch_status_and_retrieve(batch_metadata.id)
	
	# Save the results
	endpoint.save_processed_batch(dataset, batch_job, output_file)


	### Anthropic Async
	model_name = "claude-3-5-sonnet-20241022"

	# Define output file
	output_filename = f"pred_file_{os.path.splitext(os.path.basename(dataset_path))[0]}_{safe_file_name(model_name)}.jsonl"
	output_file = os.path.join(os.path.dirname(dataset_path), output_filename) if output_pred_path is None else os.path.join(output_pred_path, output_filename)

	# Create endpoint and send batch
	endpoint = AnthropicAsyncEndpoint(base_model=model_name, img_dir=img_dir)
	batch_metadata = endpoint.create_and_send_batch(dataset=dataset)

	# Check if it is finished and retrieve the results
	batch_job = endpoint.check_batch_status_and_retrieve(batch_metadata.id)
	while batch_job is None:
		time.sleep(15)
		batch_job = endpoint.check_batch_status_and_retrieve(batch_metadata.id)
	
	# Save the results
	endpoint.save_processed_batch(dataset, batch_job, output_file)
	