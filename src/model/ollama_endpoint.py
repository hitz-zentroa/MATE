import os
from typing import Literal

from ollama import Client, Message

from model.vlm_models import LuminousVLM

MODELFILES_PATH = "resources/modelfiles"

def _load_modelfile(name: str):
	path = os.path.join(MODELFILES_PATH, name)
	with open(path, 'r') as file:
		file_content = file.read()
	return file_content


class OllamaEndpoint(LuminousVLM):

	def __init__(self, base_model: str, img_dir: str = "", endpoint: str = "http://localhost:11434", show_warnings: Literal['never', 'once', 'allways'] = 'once'):
		"""
		@param base_model: name of the ollama model to use
		@param img_dir: this should be the absolute path to the image
		@param endpoint: where ollama is serving the model (usually http://localhost:11434 )
		"""
		self.endpoint = endpoint
		self.client = Client(host=self.endpoint)
		self.client.create(model=base_model, modelfile=_load_modelfile(base_model))
		self.context = []
		super().__init__(base_model=base_model, img_dir=img_dir, show_warnings=show_warnings)

	def format_conversation_step(self, text: str, role: Literal['user', 'assistant', 'system', 'tool'] = "user", img: str = None)-> Message:
		# https://ollama.com/blog/vision-models
		return Message(role=role, content=text, images=[os.path.join(self.img_dir, img)] if img is not None else [])

	def simple_generate(self, user_text: str, image_path: str):
		response = self.client.generate(model=self.model_name, prompt=user_text, images=[image_path])
		return response['response']

	def generate(self, conversation: list[(str, str, str)]):
		formatted_conversation = [self.format_conversation_step(role=role, text=text, img=img) for role, text, img in conversation]
		response = self.client.chat(model=self.model_name, messages=formatted_conversation)
		return response['message']['content'].strip()

	def batch_generate(self, conversations: list[list[(str, str, str)]]):
		"""
		We don't support batch generate in Ollama yet. For now, it will be just a sequential inference
		@param conversations: ('role', 'text', 'img_path')
		@return:
		"""
		self.show_warning("WARNING!! We don't support batch generate in Ollama yet. For now, it will be just a sequential inference", "batch_gen_ollama")
		return [self.generate(conversation) for conversation in conversations]


if __name__ == '__main__':
	endpoint = OllamaEndpoint("ollama_llava")
	endpoint.generate([('user', "What do you see in this image?", '/Users/alonsoapp/Library/Mobile Documents/com~apple~CloudDocs/Pictures/1547947334Tan_12.jpg')])