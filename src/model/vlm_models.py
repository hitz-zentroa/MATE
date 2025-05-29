import os
from abc import ABC, abstractmethod
from typing import Literal

from PIL import Image
import torch
import warnings
from transformers import AutoProcessor, LlavaForConditionalGeneration, LlavaNextForConditionalGeneration, \
	PaliGemmaForConditionalGeneration, AutoModelForCausalLM, GenerationConfig, MllamaForConditionalGeneration, \
	Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration


class LuminousVLM(ABC):
	def __init__(self, base_model: str, model=None, processor=None, max_new_tokens=50, img_dir: str = "",
				 show_warnings: Literal['never', 'once', 'allways'] = 'once'):
		self.model_name = base_model
		self.model = model
		self.img_dir = img_dir
		self.processor = AutoProcessor.from_pretrained(base_model) if processor is None else processor
		self.show_warnings = show_warnings
		self.max_new_tokens = max_new_tokens
		self.warnings_shown = {}

	def simple_generate(self, text: str, image_path: str):
		return self.generate([("user", text, image_path)])

	def generate(self, conversation: list[(str, str, str)]):
		"""
		@param conversation: [(role, text, img_path), (role, text, img_path), ...] img_path can be None
		@return:
		"""
		input_texts, generated_texts = self.batch_generate([conversation])
		return input_texts[0], generated_texts[0]

	@abstractmethod
	def batch_generate(self, conversations: list[list[(str, str, str)]]):
		"""
		@param conversations:  [[(role, text, img_path), (role, text, img_path), ...], [...]] img_path can be None
		@return:
		"""
		pass

	@staticmethod
	def image_paths(conversations: list[list[(str, str, str)]], apply_func=None):
		"""
		@param conversations:
		@param apply_func: lambda function to apply to every path (for example append an absolute path)
		@return:
		"""
		if apply_func is None:
			apply_func = lambda a: a
		return [[apply_func(image_path) for _, _, image_path in conversation if image_path is not None] for conversation
				in conversations]

	def show_warning(self, text: str, warn_type: str):
		if warn_type not in self.warnings_shown:
			self.warnings_shown[warn_type] = 0
		if self.show_warnings == 'allways' or (self.show_warnings == 'once' and self.warnings_shown[warn_type] == 0):
			self.warnings_shown[warn_type] += 1
			warnings.warn(text)

class LlavaModel(LuminousVLM):
	DEFAULT_MODEL = "llava-hf/llava-1.5-7b-hf"
	SUPPORTED_MODELS = {"llava-hf/llava-1.5-7b-hf", "llava-hf/llava-1.5-13b-hf"}

	def __init__(self, base_model: str = DEFAULT_MODEL, max_new_tokens: int = 50, model=None, processor=None,
				 img_dir: str = "", precision = torch.float16, show_warnings: Literal['never', 'once', 'allways'] = 'once'):
		self.precision = precision
		model = LlavaForConditionalGeneration.from_pretrained(base_model, torch_dtype=self.precision, device_map="auto") if model is None else model
		super().__init__(base_model=base_model, model=model, processor=processor, max_new_tokens=max_new_tokens, img_dir=img_dir, show_warnings=show_warnings)

	@staticmethod
	def format_conversation_step(text: str, role: str = "user", img: str = None):
		# https://huggingface.co/docs/transformers/en/model_doc/llava#single-image-inference
		conversation_step = {
			"role": role,
			"content": []
		}
		if img is not None:
			conversation_step["content"].append({"type": "image"})
		conversation_step["content"].append({"type": "text", "text": text})
		return conversation_step

	def batch_generate(self, conversations: list[list[(str, str, str)]]):
		"""
		WARNING!! HF batch inference on LlaVa models doesn't work properly (I don't know why tho) Use batches of 1 until fixed
		@param conversations:  [[(role, text, img_path), (role, text, img_path), ...], [...]] img_path can be None
		@return:
		"""
		if len(conversations) > 1:
			self.show_warning("WARNING!! HuggingFace batch inference on LlaVa models doesn't work properly. Generations might (will) be wrong", "llava_batch_not_supported")

		formatted_conversations = [
			[self.format_conversation_step(role=role, text=text, img=img) for role, text, img in conversation] for
			conversation in conversations]
		images = self.image_paths(conversations, lambda a: Image.open(os.path.join(self.img_dir, a)).convert("RGB"))
		# Get the first appearing image per conversation as LLava only supports 1 image per conversation
		images = [conver_imgs[0] for conver_imgs in images]

		prompts = [self.processor.apply_chat_template(conversation, add_generation_prompt=True) for conversation in
				   formatted_conversations]
		inputs = self.processor(images=images, text=prompts, padding=True, return_tensors="pt").to(self.model.device,
																								   self.precision)
		output = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)

		input_tokens = output[:, :inputs['input_ids'].size(1)]
		generated_tokens = output[:, inputs['input_ids'].size(1):]
		input_texts = self.processor.batch_decode(input_tokens, skip_special_tokens=True)
		generated_texts = self.processor.batch_decode(generated_tokens, skip_special_tokens=True)
		return input_texts, generated_texts

class LlavaNextModel(LlavaModel):
	DEFAULT_MODEL = "llava-hf/llava-v1.6-mistral-7b-hf"
	SUPPORTED_MODELS = {
		"llava-hf/llava-v1.6-mistral-7b-hf",
		"llava-hf/llava-v1.6-vicuna-7b-hf",
		"llava-hf/llava-v1.6-vicuna-13b-hf",
		"llava-hf/llava-v1.6-34b-hf",
		"llava-hf/llama3-llava-next-8b-hf",
		"llava-hf/llava-next-72b-hf",
		"llava-hf/llava-next-110b-hf"}

	def __init__(self, base_model: str = DEFAULT_MODEL, max_new_tokens: int = 50, model=None, processor=None,
				 img_dir: str = "", precision = torch.float16, show_warnings: Literal['never', 'once', 'allways'] = 'once'):
		model = LlavaNextForConditionalGeneration.from_pretrained(base_model, torch_dtype=precision,
																  device_map="auto",
																  low_cpu_mem_usage=True) if model is None else model
		super().__init__(base_model=base_model, max_new_tokens=max_new_tokens, model=model, processor=processor,
						 img_dir=img_dir, show_warnings=show_warnings, precision=precision)

class PaliGemmaModel(LuminousVLM):
	DEFAULT_MODEL = "google/paligemma-3b-mix-224"
	SUPPORTED_MODELS = {
		"google/paligemma-3b-mix-224": {"precision": torch.float32},
		"google/paligemma-3b-mix-448": {"precision": torch.float32},
		"google/paligemma-3b-mix-896": {"precision": torch.float32},
		"google/paligemma2-3b-pt-224": {"precision": torch.bfloat16},
		"google/paligemma2-3b-pt-448": {"precision": torch.bfloat16},
		"google/paligemma2-3b-pt-896": {"precision": torch.bfloat16},
		"google/paligemma2-10b-pt-224": {"precision": torch.bfloat16},
		"google/paligemma2-10b-pt-448": {"precision": torch.bfloat16},
		"google/paligemma2-10b-pt-896": {"precision": torch.bfloat16},
		"google/paligemma2-28b-pt-224": {"precision": torch.bfloat16},
		"google/paligemma2-28b-pt-448": {"precision": torch.bfloat16},
		"google/paligemma2-28b-pt-896": {"precision": torch.bfloat16},
		"google/paligemma2-3b-ft-docci-448": {"precision": torch.bfloat16},
		"google/paligemma2-10b-ft-docci-448": {"precision": torch.bfloat16},
	}

	def __init__(self, base_model: str = DEFAULT_MODEL, max_new_tokens: int = 50, processor=None,
				 img_dir: str = "", show_warnings: Literal['never', 'once', 'allways'] = 'once'):
		self.precision = self.SUPPORTED_MODELS[base_model]["precision"]
		model = PaliGemmaForConditionalGeneration.from_pretrained(base_model, torch_dtype=self.precision, device_map="auto").eval()
		super().__init__(base_model=base_model, model=model, processor=processor, max_new_tokens=max_new_tokens,
						 img_dir=img_dir, show_warnings=show_warnings)

	@staticmethod
	def build_task_prompt(text: str | list = None, lang: str = "en", task: Literal['cap', 'caption', 'describe', 'ocr', 'answer', 'question', 'detect', 'segment'] = 'answer'):
		# https://ai.google.dev/gemma/docs/paligemma/prompt-system-instructions
		match task:
			case 'cap':
				return f"cap {lang}\n"
			case 'caption':
				return f"caption {lang}\n"
			case 'describe':
				return f"describe {lang}\n"
			case 'ocr':
				return f"ocr"
			case 'answer':
				return f"answer {lang} {text}\n"
			case 'question':
				return f"question {lang} {text}\n"
			case 'detect':
				return f"detect {text.join(' ; ')}\n"
			case 'segment':
				return f"segment {text}\n"

	def simple_generate(self, text: str, image_path: str, lang: str = "en", task: Literal['cap', 'caption', 'describe', 'ocr', 'answer', 'question', 'detect', 'segment'] = 'answer'):
		image = Image.open(os.path.join(self.img_dir, image_path)).convert("RGB")

		# Instruct the model to create a caption in English
		prompt = self.build_task_prompt(text, lang, task)
		model_inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(torch.bfloat16).to(self.model.device)
		input_len = model_inputs["input_ids"].shape[-1]

		with torch.inference_mode():
			output = self.model.generate(**model_inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
			input_tokens = output[0][:input_len]
			generated_tokens = output[0][input_len:]
			input_text = self.processor.decode(input_tokens, skip_special_tokens=True)
			generated_text = self.processor.decode(generated_tokens, skip_special_tokens=True)
			return input_text, generated_text

	def generate(self, conversation: list[(str, str, str)], lang: str = "en", task: Literal['cap', 'caption', 'describe', 'ocr', 'answer', 'question', 'detect', 'segment'] = 'answer'):
		if len(conversation) > 1:
			self.show_warning("WARNING! PaliGemma doesn't support conversational input, we will use only the first step of the conversation for this inference.", "paligemma_conversation")
			conversation = [conversation[0]]
		input_texts, generated_texts = self.batch_generate(conversation, lang=lang, task=task)
		return input_texts[0], generated_texts[0]

	@staticmethod
	def format_conversation_step(text: str, role: str = "user", img: str = None):
		# I got this form a warning while processing a prompt without <image> and <bos>
		return f"{'<image>' if img is not None else ''}{text}<bos>"

	def batch_generate(self, conversations: list[list[(str, str, str)]], lang: str = "en", task: Literal['cap', 'caption', 'describe', 'ocr', 'answer', 'question', 'detect', 'segment'] = 'answer'):
		if len(conversations) > 1:
			self.show_warning("WARNING!! HuggingFace batch inference on PaliGemma models doesn't work properly. Generations might (will) be wrong", "hf_batch_not_supported")

		#formatted_conversations = [[self.format_conversation_step(role=role, text=text, img=img) for role, text, img in conversation] for conversation in conversations]
		formatted_conversations = [[self.build_task_prompt(text=text, lang=lang, task=task) for _, text, _ in conversation] for conversation in conversations]
		images = self.image_paths(conversations, lambda a: Image.open(os.path.join(self.img_dir, a)).convert("RGB"))
		# Get the first appearing image per conversation as PaliGemma only supports 1 image per conversation
		images = [conver_imgs[0] for conver_imgs in images]

		prompts = []
		for conversation in formatted_conversations:
			if len(conversation) > 1:
				self.show_warning(
					"WARNING! PaliGemma doesn't support conversational input, we will use only the first step of the conversation for this inference.",
					"paligemma_conversation")
			prompts.append(conversation[0])

		inputs = self.processor(images=images, text=prompts,  padding=True, return_tensors="pt").to(self.precision).to(self.model.device)
		with torch.inference_mode():
			output = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
			input_tokens = output[:, :inputs['input_ids'].size(1)]
			generated_tokens = output[:, inputs['input_ids'].size(1):]
			input_texts = self.processor.batch_decode(input_tokens, skip_special_tokens=True)
			generated_texts = self.processor.batch_decode(generated_tokens, skip_special_tokens=True)
			return input_texts, generated_texts

class MolmoModel(LuminousVLM):
	DEFAULT_MODEL = "allenai/Molmo-7B-O-0924"
	SUPPORTED_MODELS = {
		"allenai/MolmoE-1B-0924",
		"allenai/Molmo-7B-O-0924",
		"allenai/Molmo-7B-D-0924",
		"allenai/Molmo-72B-0924"
	}

	def __init__(self, base_model: str = DEFAULT_MODEL, max_new_tokens: int = 50, model=None, processor=None,
				 img_dir: str = "", show_warnings: Literal['never', 'once', 'allways'] = 'once'):
		model = AutoModelForCausalLM.from_pretrained(base_model, trust_remote_code=True, torch_dtype='auto', device_map='auto') if model is None else model
		processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True, torch_dtype='auto', device_map='auto') if processor is None else processor
		super().__init__(base_model=base_model, model=model, processor=processor, max_new_tokens=max_new_tokens,
						 img_dir=img_dir, show_warnings=show_warnings)

	def simple_generate(self, text: str, image_path: str):
		images = [Image.open(os.path.join(self.img_dir, image_path)).convert("RGB")]

		inputs = self.processor.process(images=images, text=text)

		# move inputs to the correct device and make a batch of size 1
		inputs = {k: v.to(self.model.device).unsqueeze(0) for k, v in inputs.items()}

		with torch.autocast(device_type=self.model.device.type, enabled=True, dtype=torch.bfloat16):
			output = self.model.generate_from_batch(inputs, GenerationConfig(max_new_tokens=self.max_new_tokens, stop_strings="<|endoftext|>"), tokenizer=self.processor.tokenizer)

		# only get generated tokens; decode them to text
		input_tokens = output[0, :inputs['input_ids'].size(1)]
		generated_tokens = output[0, inputs['input_ids'].size(1):]
		#all_tokens = output[0, :]
		input_text = self.processor.tokenizer.decode(input_tokens, skip_special_tokens=True)
		generated_text = self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

		# print the generated text
		return input_text, generated_text

	def generate(self, conversation: list[(str, str, str)]):
		if len(conversation) > 1:
			self.show_warning("WARNING! Molmo doesn't support conversational input, we will use only the first step of the conversation for this inference.", "paligemma_conversation")
		_, text, img = conversation[0]
		return self.simple_generate(text=text, image_path=img)

	def batch_generate(self, conversations: list[list[(str, str, str)]]):
		"""
		WARNING!! HF batch inference on Molmo models doesn't work properly (I don't know why tho) Use batches of 1 until fixed
		@param conversations:  [[(role, text, img_path), (role, text, img_path), ...], [...]] img_path can be None
		@return:
		"""
		if len(conversations) > 1:
			self.show_warning(
				"WARNING!! HuggingFace batch inference on Molmo models doesn't work properly. We will perform single batch inference",
				"llava_batch_not_supported")
		"""
		import torch.nn.functional as F
		formatted_conversations = [[text for _, text, _ in conversation] for conversation in conversations]
		images = self.image_paths(conversations, lambda a: Image.open(os.path.join(self.img_dir, a)).convert("RGB"))
		# Get the first appearing image per conversation as PaliGemma only supports 1 image per conversation
		images = [conver_imgs[0] for conver_imgs in images]

		prompts = []
		for conversation in formatted_conversations:
			if len(conversation) > 1:
				self.show_warning(
					"WARNING! Molmo doesn't support conversational input, we will use only the first step of the conversation for this inference.",
					"molmo_conversation")
			prompts.append(conversation[0])

		batch_inputs = {}
		for prompt, image in zip(prompts, images):
			inputs = self.processor.process(images=image, text=prompt, return_tensors="pt")
			for k, v in inputs.items():
				unsqueezed = v.to(self.model.device).unsqueeze(0)
				if k not in batch_inputs:
					batch_inputs[k] = unsqueezed
				else:
					if k == "input_ids":
						# Add padding
						max_len = max(batch_inputs[k].shape[1], unsqueezed.shape[1])
						padded_tensor1 = F.pad(batch_inputs[k], (0, max_len - batch_inputs[k].shape[1]))
						padded_tensor2 = F.pad(unsqueezed, (0, max_len - unsqueezed.shape[1]))
						batch_inputs[k] = torch.cat([padded_tensor1, padded_tensor2], dim=0)
					else:
						batch_inputs[k] = torch.cat([batch_inputs[k], unsqueezed], dim=0)

		outputs = self.model.generate_from_batch(batch_inputs,
												GenerationConfig(max_new_tokens=self.max_new_tokens, stop_strings="<|endoftext|>"),
												tokenizer=self.processor.tokenizer)

		return self.processor.tokenizer.batch_decode(outputs, skip_special_tokens=True)
		"""
		result = [self.generate(conversation) for conversation in conversations]
		return zip(*result)

class LlamaVLMModel(LlavaModel):
	DEFAULT_MODEL = "unsloth/Llama-3.2-11B-Vision-Instruct"
	SUPPORTED_MODELS = {
		"unsloth/Llama-3.2-11B-Vision-Instruct",
		"meta-llama/Llama-3.2-11B-Vision",
	}

	def __init__(self, base_model: str = DEFAULT_MODEL, max_new_tokens: int = 50, model=None, processor=None,
				 img_dir: str = "", precision=torch.bfloat16, show_warnings: Literal['never', 'once', 'allways'] = 'once'):
		model = MllamaForConditionalGeneration.from_pretrained(base_model, torch_dtype=precision, device_map="auto") if model is None else model
		processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True, torch_dtype='auto', device_map='auto') if processor is None else processor
		super().__init__(base_model=base_model, model=model, processor=processor, max_new_tokens=max_new_tokens,
						 img_dir=img_dir, show_warnings=show_warnings, precision=precision)

class QwenVLModel(LlavaModel):
	DEFAULT_MODEL = "Qwen/Qwen2-VL-7B-Instruct"
	SUPPORTED_MODELS = {
		"Qwen/Qwen2-VL-2B-Instruct",
		"Qwen/Qwen2-VL-7B-Instruct",
		"Qwen/Qwen2-VL-72B-Instruct",
		"Qwen/Qwen2.5-VL-3B-Instruct",
		"Qwen/Qwen2.5-VL-7B-Instruct",
		"Qwen/Qwen2.5-VL-32B-Instruct",
		"Qwen/Qwen2.5-VL-72B-Instruct",
	}

	def __init__(self, base_model: str = DEFAULT_MODEL, max_new_tokens: int = 50, model=None, processor=None,
				 img_dir: str = "", precision=torch.bfloat16, show_warnings: Literal['never', 'once', 'allways'] = 'once'):
		if "Qwen2.5" in base_model:
			model = Qwen2_5_VLForConditionalGeneration.from_pretrained(base_model, torch_dtype="auto")
		else:
			model = Qwen2VLForConditionalGeneration.from_pretrained(base_model, torch_dtype="auto", device_map="auto") if model is None else model
		processor = AutoProcessor.from_pretrained(base_model, use_fast=True) if processor is None else processor
		super().__init__(base_model=base_model, model=model, processor=processor, max_new_tokens=max_new_tokens,
						 img_dir=img_dir, show_warnings=show_warnings, precision=precision)