<div align="center">
  <h1> Vision-Language Models (VLMs) for Brazilian Portuguese Image Captioning </h1>
  <p>By Computational Intelligence and Information Systems Laboratory (LAICSI-IFES)</p>
</div>

<div align="center">
 <img src='/images/illustration-transformers.jpg' width='800'>
</div>

### :wrench: To set up the environment, use:
```bash
$ chmod +x setup.sh
$ ./setup.sh
```

### :wrench: To set the environment variables, create a .env file in root:
```bash
HF_API_KEY="..."
WANDB_API_KEY="..."
WANDB_PROJECT_NAME="..."
```

### :gear: To run the complete train and evaluate, use:
```bash
$ python train.py
```

### :gear: To run only the evaluation, use:
```bash
$ python eval.py
```

### :tophat: Don't forget of setting up the training/model attributes in ```config.yml```. An example:
```yaml
config:
  model_name: "paligemma"
  dataset: "flickr30k_pt"
  push_to_hub: True
  dataset_from_hub: True
  interval_strategy: "epoch"
  evaluate_from_model: True
  turn_off_computer: False

generate_args:
  max_new_tokens: "auto"

callbacks:
  early_stopping:
    patience: 3
    threshold: 0.0

mllm:
  llama3-vision:
    id: "meta-llama/Llama-3.2-11B-Vision-Instruct"
    batch_size: 1
    use_flash_attention: False
    question: "Descreva a imagem em português:"
    qlora_args:
      use_qlora: True
      lora_rank: 8
      freeze_vision: True
      lora_all_linear: False
    training_args:
      num_train_epochs: 1
      do_eval: True
      gradient_accumulation_steps: 6
      gradient_checkpointing: True
      gradient_checkpointing_kwargs:
        use_reentrant: False
      save_total_limit: 1
      warmup_steps: 2
      max_grad_norm: 1
      warmup_ratio: 0.1
      weight_decay: 1.0e-6
      bf16: True
      optim: "adamw_torch"
      learning_rate: 1.0e-4
      load_best_model_at_end: True
      push_to_hub: False
      remove_unused_columns: False
      dataloader_pin_memory: False
  phi3-vision:
    id: "microsoft/Phi-3-vision-128k-instruct"
    batch_size: 1
    use_flash_attention: False
    question: "Descreva a imagem em português:"
    qlora_args:
      use_qlora: True
      lora_rank: 8
      freeze_vision: True
      lora_all_linear: False
    training_args:
      num_train_epochs: 1
      do_eval: True
      gradient_accumulation_steps: 6
      gradient_checkpointing: True
      gradient_checkpointing_kwargs:
        use_reentrant: False
      save_total_limit: 1
      warmup_steps: 2
      max_grad_norm: 1
      warmup_ratio: 0.1
      weight_decay: 1.0e-6
      bf16: True
      optim: "adamw_torch"
      learning_rate: 1.0e-4
      load_best_model_at_end: True
      push_to_hub: False
      remove_unused_columns: False
      dataloader_pin_memory: False
  paligemma:
    id: "google/paligemma-3b-pt-224"
    batch_size: 1
    use_flash_attention: False
    question: "caption pt\n"
    qlora_args:
      use_qlora: True
      lora_rank: 8 
      freeze_vision: True
      lora_all_linear: False
    training_args:
      num_train_epochs: 1
      do_eval: True
      gradient_accumulation_steps: 6
      gradient_checkpointing: True
      gradient_checkpointing_kwargs:
        use_reentrant: False
      save_total_limit: 1
      warmup_steps: 2
      max_grad_norm: 1
      warmup_ratio: 0.1
      weight_decay: 1.0e-6
      bf16: True
      adam_beta2: 0.999
      optim: 'paged_adamw_8bit'
      learning_rate: 2.0e-5
      load_best_model_at_end: True
      push_to_hub: False
      remove_unused_columns: False
      dataloader_pin_memory: False

dataset:
  flickr30k_pt:
    id: "laicsiifes/flickr30k-pt-br-5k"
    max_length: 25
    image_column: "image"
    text_column: "caption"
  pracegover63k:
    id: "laicsiifes/pracegover63k-5k"
    max_length: 70
    image_column: "image"
    text_column: "text"
```

### :barber: Directory structure:
```
├── README.md          <- The top-level README for developers using this project.
├── data
│   └── pracegover_63k     <- Dataset #PraCegoVer 63k
│       ├── test.hf        <- Data for testing split.
│       ├── train.hf       <- Data for training split.
│       └── validation.hf  <- Data for validation split.
│
├── docs               <- A default HTML for docs.
│
├── models             <- The models and its artifacts will be saved here.
│
├── requirements.txt   <- The requirements file for reproducing the training and evaluation pipelines.
│
└── src                <- Source code for use in this project.
    ├── __init__.py    <- Makes src a Python module
    │
    ├── utils          <- Modularization for configuration, splits processing and evaluation metrics.
    │   ├── config.py
    │   ├── data_processing.py
    │   └── metrics.py
    │
    ├── eval.py
    └── train.py
```
