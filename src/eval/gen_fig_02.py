from main_eval import object_figure

if __name__ == '__main__':
	object_figure("data/mate/dev/predictions", barplot_models=[
		"llava-hf/llava-v1.6-34b-hf", "allenai/Molmo-7B-O-0924", "unsloth/Llama-3.2-11B-Vision-Instruct",
		"Qwen/Qwen2.5-VL-7B-Instruct", "gpt-4o-2024-11-20", "claude-3-5-sonnet-20241022", "gemini-1.5-flash"],
				  line_models=["human"], highlighted_models=["human"],
				  human_line_modality="mm")