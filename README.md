# MATE: Vision-Language Models Struggle to Align Entities across Modalities

This repository provides the code and data for our ACL 2025 paper:

> **[Vision-Language Models Struggle to Align Entities across Modalities](https://arxiv.org/abs/2503.03854)**  
> *Iñigo Alonso, Gorka Azkune, Ander Salaberria, Jeremy Barnes, Oier Lopez de Lacalle*

---

## Getting Started
1. **Clone the repository and install dependencies**
   ```bash
   git clone https://github.com/hitz-zentroa/MATE.git
   cd MATE
   pip install -r requirements.txt
   export PYTHONPATH="$PWD/src"
   ```
2.	**Download the Development Version of MATE Benchmark**
Please download the development version of MATE [here](https://drive.google.com/file/d/1_joeNBesJSUsMEdgVpnKEySTW2wmodeN/view?usp=share_link) and extract it into the MATE/data/ directory.
While the public version is hosted on 🤗 [HuggingFace](https://huggingface.co/datasets/HiTZ/MATE), we use the development version for experiments in the paper. This version includes the same examples, with additional metadata useful for evaluation.

---

## Inference

To run inference with any of the supported models:
```
python3 ./src/main_inference.py --base_model MODEL_NAME --batch_size 1 --max_new_tokens 80 --dataset_path data/mate/dev/MATE_VARIANT
```
**MATE variants** (available at MATE/data/dev/*):
* `mm_0shot.jsonl`: cross-modal (img2data, data2img) 0-shot
* `mm_1shot.jsonl`: cross-modal (img2data, data2img) 1-shot
* `mm_2shot.jsonl`: cross-modal (img2data, data2img) 2-shot
* `mm_2shot_cot.jsonl`: cross-modal (img2data, data2img) 2-shot with CoT (recommended --max_new_tokens=500)
* `um_0shot.jsonl`: unimodal (img2img, data2data)  0-shot
* `um_1shot.jsonl`: unimodal (img2img, data2data)  1-shot
* `um_2shot.jsonl`: unimodal (img2img, data2data)  2-shot

**Supported models**:
* llava-hf/llava-1.5-7b-hf 
* llava-hf/llava-1.5-13b-hf
* llava-hf/llava-v1.6-mistral-7b-hf
* llava-hf/llava-v1.6-vicuna-7b-hf
* llava-hf/llava-v1.6-vicuna-13b-hf
* llava-hf/llava-v1.6-34b-hf
* llava-hf/llama3-llava-next-8b-hf
* allenai/MolmoE-1B-0924
* allenai/Molmo-7B-O-0924
* allenai/Molmo-7B-D-0924
* meta-llama/Llama-3.2-11B-Vision
* Qwen/Qwen2-VL-2B-Instruct
* Qwen/Qwen2-VL-7B-Instruct
* Qwen/Qwen2.5-VL-7B-Instruct

While these are the models used in the paper, this code support a wider range of models. See `src/model/vlm_models.py` 
for a complete list of supported models.

---

## Evaluation 
### Tables
Cross-modal and unimodal overall results. To reproduce **Table 1** and **Table 3**, run:
```bash
python3 src/eval/gen_table_01.py
```
Performance per attribute results. To reproduce **Table 2**, run:
```bash
python3 src/eval/gen_table_02.py
```
Chain-of-thought results. To reproduce **Table 4**, run:
```bash
python3 src/eval/gen_table_04.py
```
Complete performance per attribute results. To reproduce  **Table 8**, run:
```bash
python3 src/eval/gen_table_08.py
```
Complete performance for 0, 1, and 2 shot prompts. To reproduce  **Table 9**, run:
```bash
python3 src/eval/gen_table_09.py
```
### Figures
Cross-modal performance per object count. To reproduce  **Figure 2**, run:
```bash
python3 src/eval/gen_fig_02.py
```
Uni-modal performance per object count. To reproduce  **Figure 3**, run:
```bash
python3 src/eval/gen_fig_03.py
```
Linking attribute analysis. To reproduce  **Figure 4**, run:
```bash
python3 src/eval/gen_fig_04.py
```
Predicted Object Attribute Overlapping in 3D Coordinate-Only Linking Attribute Cases. To reproduce  **Figure 5**, run:
```bash
python3 src/eval/gen_fig_05.py
```

---

## Citations
If you find MATE useful in your research, please consider giving us a star 🌟 and citing it by the following BibTeX entry.
```
@article{alonso2025vision,
  title={Vision-Language Models Struggle to Align Entities across Modalities},
  author={Alonso, I{\~n}igo and Salaberria, Ander and Azkune, Gorka and Barnes, Jeremy and de Lacalle, Oier Lopez},
  journal={arXiv preprint arXiv:2503.03854},
  year={2025}
}
```
