from typing import Literal

from model.ollama_endpoint import OllamaEndpoint
from model.vlm_models import LlavaModel, LlavaNextModel, PaliGemmaModel, MolmoModel, LlamaVLMModel, QwenVLModel
from model.private_vlm_endpoints import OpenAIEndpoint, GoogleEndpoint, AnthropicEndpoint, OpenAIAsyncEndpoint, AnthropicAsyncEndpoint

def get_vlm(base_model: str, max_gen_tokens: int = 50, img_dir: str = "", show_warnings: Literal['never', 'once', 'allways'] = 'once'):
	if base_model in LlavaModel.SUPPORTED_MODELS:
		return LlavaModel(base_model=base_model, max_new_tokens=max_gen_tokens, img_dir=img_dir, show_warnings=show_warnings)
	elif base_model in LlavaNextModel.SUPPORTED_MODELS:
		return LlavaNextModel(base_model=base_model, max_new_tokens=max_gen_tokens, img_dir=img_dir, show_warnings=show_warnings)
	elif "ollama" in base_model:
		return OllamaEndpoint(base_model=base_model, img_dir=img_dir, show_warnings=show_warnings)
	elif base_model in PaliGemmaModel.SUPPORTED_MODELS:
		return PaliGemmaModel(base_model=base_model, max_new_tokens=max_gen_tokens, img_dir=img_dir, show_warnings=show_warnings)
	elif base_model in MolmoModel.SUPPORTED_MODELS:
		return MolmoModel(base_model=base_model, max_new_tokens=max_gen_tokens, img_dir=img_dir, show_warnings=show_warnings)
	elif base_model in LlamaVLMModel.SUPPORTED_MODELS:
		return LlamaVLMModel(base_model=base_model, max_new_tokens=max_gen_tokens, img_dir=img_dir, show_warnings=show_warnings)
	elif base_model in QwenVLModel.SUPPORTED_MODELS:
		return QwenVLModel(base_model=base_model, max_new_tokens=max_gen_tokens, img_dir=img_dir, show_warnings=show_warnings)
	elif base_model in GoogleEndpoint.SUPPORTED_MODELS:
		return GoogleEndpoint(base_model=base_model, img_dir=img_dir, show_warnings=show_warnings)
	elif base_model in OpenAIEndpoint.SUPPORTED_MODELS:
		return OpenAIEndpoint(base_model=base_model, img_dir=img_dir, show_warnings=show_warnings)
	elif base_model in AnthropicEndpoint.SUPPORTED_MODELS:
		return AnthropicEndpoint(base_model=base_model, img_dir=img_dir, show_warnings=show_warnings)

def get_async_vlm(base_model: str, max_gen_tokens: int = 50, img_dir: str = "", show_warnings: Literal['never', 'once', 'allways'] = 'once'):
	if base_model in OpenAIAsyncEndpoint.SUPPORTED_MODELS:
		return OpenAIAsyncEndpoint(base_model=base_model, max_new_tokens=max_gen_tokens, img_dir=img_dir, show_warnings=show_warnings)
	elif base_model in AnthropicAsyncEndpoint.SUPPORTED_MODELS:
		return AnthropicAsyncEndpoint(base_model=base_model, max_new_tokens=max_gen_tokens, img_dir=img_dir, show_warnings=show_warnings)