<div align="center">
  <h1> Transformer-Based Vision Models for Brazilian Portuguese Image Captioning </h1>
  <p>By Computational Intelligence and Information Systems Laboratory (LAICSI-IFES)</p>
</div>

<div align="center">
 <img src='/images/illustration-general.jpg' width='800'>
</div>

<br>

## 📌 Project Overview
This repository hosts a comprehensive collection of research resources for **Brazilian Portuguese Image Captioning**. It encompasses various approaches, including standard Vision Encoder-Decoder (VED) models, Vision-Language Models (VLMs) fine-tuning, Zero-Shot inference with Large Multimodal Models (LMMs), and advanced evaluation metrics.

Our goal is to provide a robust benchmark and set of tools for the Portuguese Multi-modal community.

## 🔬 Research Branches & Modules
This repository is organized into several independent research modules. Each module contains its own documentation, source code, and specific requirements.

| Module | Description | Link |
| :--- | :--- | :---: |
| **VED Models** | The core Vision Encoder-Decoder transformer experiments (e.g., ViT+BERT, Swin+GPT2). Includes seminal work on Portuguese captioning. | [Go to VED](./ved) |
| **VLM Fine-Tuning** | Fine-tuning pipelines for modern Vision-Language Models (e.g., PaliGemma, Llama-Vision) on Portuguese datasets. | [Go to VLM](./vlm) |
| **VLM Zero/Few-Shot** | Inference pipelines using large pre-trained models (e.g., GPT-4o, Gemini, Claude) in zero-shot or few-shot settings. | [Go to Zero-Shot](./vlm_zero_shot) |
| **Metrics Analysis** | Tools for analyzing captioning metrics, including reference-free metrics and correlation studies. | [Go to Metrics](./metrics_analysis) |
| **Model as Evaluator** | Experiments using Large Language Models (LLMs) as judges to evaluate caption quality. | [Go to Evaulator](./model_as_evaluator) |

## 📂 Available Resources
The project utilizes and provides access to key resources for Portuguese Image Captioning:

- **Flickr-Translated**: A Portuguese translation of the Flickr30k dataset.
- **Flickr-Native**: A dataset of human generated captions for the Flickr30k dataset.
- **PraCegoVer**: A dataset focused on accessibility with richer descriptions (currently unavailable publicly).
- **VLMs Adapters**: A collection of adapters for Vision-Language Models (e.g., PaliGemma, Llama-Vision) on Portuguese datasets.
- **VED Models**: A collection of Vision Encoder-Decoder models (e.g., ViT+BERT, Swin+GPT2) on Portuguese datasets.

| Resource Version | HuggingFace ID | Description |
| :--- | :--- | :--- |
| **Datasets for Portuguese Image Captioning** | [`laicsiifes/datasets-for-portuguese-image-captioning`](https://hf.co/collections/laicsiifes/datasets-for-portuguese-image-captioning) | A collection of datasets for Portuguese Image Captioning. |
| **VEDs for Brazilian Portuguese IC** | [`laicsiifes/veds-for-brazilian-portuguese-ic`](https://hf.co/collections/laicsiifes/veds-for-brazilian-portuguese-ic) | A collection of Vision Encoder-Decoder models (e.g., ViT+BERT, Swin+GPT2) on Portuguese datasets. |
| **VLMs for Brazilian Portuguese IC** | [`laicsiifes/vlms-for-brazilian-portuguese-ic`](https://hf.co/collections/laicsiifes/vlms-for-brazilian-portuguese-ic) | A collection of Vision-Language Models (e.g., PaliGemma, Llama-Vision) on Portuguese datasets. |

## 🚀 Getting Started
Since each module operates independently, we recommend navigating to the specific folder of interest (table above) and following the `README.md` instructions there.

 However, for general environment setup that might apply to shared utilities, you can use:

```bash
$ chmod +x setup.sh
$ ./setup.sh
```

## 🏆 Collections & Leaderboard
- **HuggingFace Collection**: [Vision Encoder-Decoder Brazilian Portuguese Image Captioning](https://hf.co/collections/laicsiifes/veds-for-brazilian-portuguese-ic)
- **Top Models**:
  - 🥇 [Swin-DistilBERTimbau](https://hf.co/laicsiifes/swin-distilbertimbau)
  - 🥈 [Swin-GPorTuguese-2](https://hf.co/laicsiifes/swin-gportuguese-2)

## 📋 Citation
If you use our work, code, or datasets, please cite:

```bibtex
@inproceedings{bromonschenkel2024comparative,
  title={A Comparative Evaluation of Transformer-Based Vision Encoder-Decoder Models for Brazilian Portuguese Image Captioning},
  author={Bromonschenkel, Gabriel and Oliveira, Hil{\'a}rio and Paix{\~a}o, Thiago M},
  booktitle={2024 37th SIBGRAPI Conference on Graphics, Patterns and Images (SIBGRAPI)},
  pages={1--6},
  year={2024},
  organization={IEEE}
}
```
