"""
Training Module for Vision-Language Models
==========================================

This module provides the `train_model` function to configure, train, and evaluate 
a multimodal vision-language model for image captioning tasks in Brazilian Portuguese. 
The function allows optional use of QLoRA for low-rank adaptation and integrates with 
Weights and Biases (W&B) for experiment tracking and logging.

Authors
-------
BSc, Gabriel Mota Bromonschenkel Lima
Email: gabriel.mota.b.lima@gmail.com

PhD, Hilário Tomaz Alves de Oliveira
Email: hilariotomaz@gmail.com

PhD, Thiago Meireles Paixão
Email: thiago.paixao@ifes.edu.br

Functions
---------
train_model(config, training_args, generate_args, qlora_args, callbacks)
    Configure, train, and evaluate a vision-language model, with options for 
    QLoRA adaptation and W&B logging. This function also saves evaluation 
    results to CSV files.
"""

import os
import time
import yaml
import wandb

import pandas as pd

from peft import get_peft_model
from transformers import EarlyStoppingCallback, Seq2SeqTrainer, Seq2SeqTrainingArguments

from config.config import config_vars, configure_model_and_processor, create_lora_config

from data_prep.data_processing import load_datasets, preprocess, transform_datasets
from data_prep.data_collator import DataCollatorForTraining, DataCollatorForGeneration

from evaluation.eval_prediction import evaluate_predictions
from evaluation.eval_finetuning import compute_metrics

from generation.generation import batch_generation

from dotenv import load_dotenv
from huggingface_hub import login
from pprint import pprint


def train_model(config, training_args, generate_args, qlora_args, callbacks):
    """
    Train a vision-language model using specified configuration and dataset, with optional QLoRA
    low-rank adaptation and logging using Weights and Biases (W&B).

    Parameters
    ----------
    config : dict
        Configuration dictionary containing model ID, dataset information, device, and other settings.
    training_args : dict
        Training arguments passed to the Trainer, including batch size, learning rate, and optimizer settings.
    generate_args : dict
        Generation arguments used for evaluation.
    qlora_args : dict
        QLoRA-specific arguments, including rank, alpha-to-rank ratio, dropout, and linear layer settings.
    callbacks : dict
        Callback configurations, including early stopping parameters.

    Returns
    -------
    None
        This function trains the model, evaluates it on the test dataset, and saves results to CSV files.
    """
    # Configure model, processor and Quantization
    model, processor = configure_model_and_processor(
        model_id=config["model_id"],
        use_bnb=qlora_args["use_bnb"], # True
        use_flash_attention=config["use_flash_attention"] # False
    )

    # Config LoRA
    if qlora_args["use_lora"]:
        lora_config = create_lora_config(
            model_id=config["model_id"],
            rank=qlora_args["lora_rank"], # 16 8
            linear_modules=qlora_args["linear_modules"],
            alpha_to_rank_ratio=qlora_args["alpha_to_rank_ratio"],
            dropout=qlora_args["dropout"],
            is_all_linear=qlora_args["lora_all_linear"] # False
        )

        model = get_peft_model(model, lora_config)
        
        model.print_trainable_parameters()

    # Load datasets from HuggingFace Hub
    train_ds, valid_ds, test_ds = load_datasets(
        data_dir=config["data_dir"],
        hf_dataset=config["hf_dataset"],
        dataset_from_hub=config["dataset_from_hub"]
    )

    print('\nDataset')
    print(f'\tTrain: {len(train_ds)}')
    print(f'\tVal: {len(valid_ds)}')
    print(f'\tTest: {len(test_ds)}\n')

    # Prepare datasets to be used
    train_dataset, valid_dataset, test_dataset = transform_datasets(
        train_ds=train_ds,
        valid_ds=valid_ds,
        test_ds=test_ds,
        preprocess_fn=preprocess(
            question=config["question"].format(max_length=config["max_length"]),
            text_per_image=config["text_per_image"],
            image_column=config["image_column"],
            text_column=config["text_column"],
        ),
    )

    model.config.use_cache = False
    model.generation_config.max_new_tokens = config["max_length"]

    kwargs_collator = {
        "model_id": config["model_id"],
        "device": config["device"],
        "processor": processor,
        "max_length": config["max_length"],
        "question": config["question"].format(max_length=config["max_length"]),
        # "image_processor": model.vision_tower._image_processor if 'vitucano' in config["model_id"].lower() else None
    }

    # Config trainer for supervised fine-tuning
    trainer = Seq2SeqTrainer(
        model=model,
        args=Seq2SeqTrainingArguments(**training_args),
        compute_metrics=compute_metrics(processor=processor, model_id=config["model_id"]),
        data_collator=DataCollatorForTraining(**kwargs_collator),
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=processor if 'vitucano' in config["model_id"].lower() else processor.tokenizer,
        # peft_config=lora_config if qlora_args["use_lora"] else None,
        # callbacks=[
        #     EarlyStoppingCallback(
        #         early_stopping_patience=callbacks["early_stopping"]["patience"],
        #         early_stopping_threshold=callbacks["early_stopping"]["threshold"]
        #     )
        # ],
    )

    # wandb.login(key=os.getenv("WANDB_API_KEY"))

    # Model training monitoring
    with wandb.init(project=os.getenv("WANDB_PROJECT_NAME")) as run:
        run.name = f'{config["model_name"]}-ft-{config["dataset"]}'

        output_dir = training_args["output_dir"]

        if os.path.exists(output_dir) and len(os.listdir(output_dir)) > 0:
            try:
                trainer.train(resume_from_checkpoint=True)
            except:
                trainer.train()
        else:
            trainer.train()

    # Save model
    processor.chat_template = processor.tokenizer.chat_template

    model.save_pretrained(config["model_dir"])
    processor.save_pretrained(config["model_dir"])

    if config["push_to_hub"]:
        model.push_to_hub(f'{config["model_name"]}-{config["dataset"]}', private=True)
        processor.push_to_hub(f'{config["model_name"]}-{config["dataset"]}', private=True)

    pd.DataFrame(trainer.state.log_history).to_csv(
        path_or_buf=os.path.join(config['results_dir'], f'training_log_history.csv'),
        index=False
    )

    # Evaluation in Test set
    evaluate_predictions(
        raw_dataset=test_ds,
        predictions=batch_generation(
            raw_dataset=test_ds,
            model=trainer.model,
            config=config,
            collate_fn=DataCollatorForGeneration(**kwargs_collator),
            processor=processor,
            generate_args=generate_args
        ),
        text_column=config["text_column"],
        results_dir=config["results_dir"]
    )


if __name__ == "__main__":
    """
    Main function to run the training process based on configurations specified in a YAML file.
    """
    load_dotenv(dotenv_path="../.env")
    login(os.getenv("HF_API_KEY"))

    with open("config/config_finetuning.yml", "r") as file:
        setups = config_vars(yaml.safe_load(file))

    print("\nConfiguration:", end="\t")
    pprint(setups)

    if not os.path.exists(setups["config"]['results_dir']):
        os.makedirs(setups["config"]['results_dir'])

    train_model(
        config=setups["config"],
        training_args=setups["training_args"],
        generate_args=setups["generate_args"],
        qlora_args=setups["qlora_args"],
        callbacks=setups["callbacks"]
    )

    if setups["config"]["turn_off_computer"]:
        print('\nTurning off computer ...')
        time.sleep(2 * 60)
        os.system('shutdown -h now')
