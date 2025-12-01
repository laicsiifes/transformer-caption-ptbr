"""
Metrics Evaluation Module for Image Captioning
==============================================

This module provides functions to calculate evaluation metrics for image captioning 
models, particularly for multimodal vision-language models. It includes helper 
functions for computing BERTScore, CLIPScore, ROUGE, BLEU, METEOR, and other 
captioning metrics, providing both individual and aggregated scores for 
comprehensive analysis.

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
batch_decode_filter(tokens_ids, processor)
    Decode token IDs into text, replacing ignored tokens with padding tokens.

compute_metrics(eval_pred, processor, model_id)
    Returns a function to compute ROUGE metrics for model evaluation.
"""

import evaluate
import numpy as np


def batch_decode_filter(tokens_ids, processor):
    """
    Decode token IDs into text, replacing ignored tokens (-100) with padding tokens.

    Parameters
    ----------
    tokens_ids : np.array or Tensor
        Array of token IDs, where -100 indicates ignored tokens to be replaced with padding tokens.
    processor : ProcessorMixin
        The processor used to decode the token IDs.

    Returns
    -------
    list of str
        A list of decoded strings with special tokens removed.
    """
    return processor.batch_decode(
        np.where(tokens_ids != -100, tokens_ids, processor.tokenizer.pad_token_id),
        skip_special_tokens=True
    )


def compute_metrics(processor, model_id):
    """
    Returns a function to compute ROUGE metrics for model evaluation.

    Parameters
    ----------
    processor : ProcessorMixin
        The processor used to decode token IDs into text.
    model_id : str
        Identifier of the model being evaluated, used to adjust prediction format.

    Returns
    -------
    function
        A function that computes ROUGE scores between predictions and labels.
    """
    def compute_rouge(eval_pred):
        """
        Computes ROUGE scores between predicted and reference sequences.

        Parameters
        ----------
        eval_pred : EvalPrediction
            An object containing the predictions and label IDs from the evaluation.

        dict
            A dictionary containing ROUGE scores (e.g., ROUGE-1, ROUGE-2, ROUGE-L)
            between the predictions and the references.
        """
        rouge = evaluate.load("rouge")

        pred_ids = eval_pred.predictions
        label_ids = eval_pred.label_ids

        if "paligemma" in model_id.lower():
            pred_ids = pred_ids[0]

        predictions = batch_decode_filter(pred_ids.argmax(-1), processor)
        labels = batch_decode_filter(label_ids, processor)

        return rouge.compute(
            predictions=predictions,
            references=labels,
            use_stemmer=False,
            tokenizer=lambda x: x.split()
        )
    
    return compute_rouge