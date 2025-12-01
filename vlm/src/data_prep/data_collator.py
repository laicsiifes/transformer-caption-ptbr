"""
Data Collators for Training and Generation with Vision-Language Models
======================================================================

This module defines data collator classes to prepare input batches for training and evaluation 
with various multimodal vision-language models, including LLaMa 3.2 Vision, Phi-3 Vision, 
and PaliGemma. These classes support customized batching of images and text, ensuring compatibility 
with model-specific requirements.

Authors
-------
BSc, Gabriel Mota Bromonschenkel Lima
Email: gabriel.mota.b.lima@gmail.com

PhD, Hilário Tomaz Alves de Oliveira
Email: hilariotomaz@gmail.com

PhD, Thiago Meireles Paixão
Email: thiago.paixao@ifes.edu.br

Classes
-------
DataCollatorForTraining(processor, model_id, device, max_length)
    Selects and applies appropriate collator functions to prepare input data into batches 
    for training with the specified model.

DataCollatorForGeneration(processor, model_id, device, max_length)
    Selects and applies appropriate collator functions to prepare input data into batches 
    for evaluation or generation with the specified model.
"""

import torch
from dataclasses import dataclass, field
from enum import auto, Enum
from typing import List
from transformers import ProcessorMixin, ImageProcessingMixin, PretrainedConfig


class SeparatorStyle(Enum):
    TWO = auto()


@dataclass
class Conversation:
    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    sep_style: SeparatorStyle
    sep: str
    sep2: str
    version: str
    skip_next: bool = False

    def get_prompt(self):
        seps = [self.sep, self.sep2]
        ret = self.system + seps[0]
        for i, (role, message) in enumerate(self.messages):
            if message:
                ret += role + ": " + message + seps[i % 2]
            else:
                ret += role + ":"
        return ret

    def append_message(self, role, message):
        self.messages.append([role, message])

    def copy(self):
        return Conversation(
            system=self.system, roles=self.roles, messages=[[x, y] for x, y in self.messages],
            offset=self.offset, sep_style=self.sep_style, sep=self.sep, sep2=self.sep2, version=self.version
        )

@dataclass
class VLMsDataCollator:
    processor: ProcessorMixin
    model_id: str
    device: str
    max_length: int
    question: str
    # TinyLLaVA Factory (TLF) constants for ViTucano
    image_processor: ImageProcessingMixin = None
    IGNORE_INDEX: int = -100
    IMAGE_TOKEN_INDEX: int = -200
    DEFAULT_IMAGE_TOKEN: str = "<image>"
    conv_tucano_v0: Conversation = field(
        default_factory= lambda : Conversation(
            system="Um bate-papo entre um usuário curioso e um assistente de inteligência artificial. "
                   "O assistente dá respostas úteis, detalhadas e educadas às perguntas do usuário.",
            roles=("\n Usuário", "\n Assistente"),
            version="llama",
            messages=(),
            offset=0,
            sep_style=SeparatorStyle.TWO,
            sep=" ",
            sep2="</s>"
        )
    )


    def _check_header(self, targets, seq):
        """
        Check if any target sequence from a list of targets is present in the specified sequence.

        Parameters
        ----------
        targets : list of list of int
            A list of target sub-sequences to search for within `seq`.
        seq : list of int
            The sequence in which to search for target sub-sequences.

        Returns
        -------
        bool
            True if any target sub-sequence is found in `seq`, False otherwise.
        """
        for i in range(len(seq)-3):
            if seq[i:i+3] in targets:
                return True

        return False


    def _replace_target(self, target, seq):
        """
        Replace occurrences of a specific target sequence within a given sequence by 
        setting elements to -100, effectively masking these values from model input.

        Parameters
        ----------
        target : list of int
            The target sub-sequence to replace in `seq`.
        seq : list of int
            The sequence in which to replace occurrences of `target`.

        Returns
        -------
        list of int
            The modified sequence with occurrences of `target` replaced by -100.
        """
        for i in range(len(seq)-3):
            if seq[i:i+3] == target:
                seq[i],seq[i+1],seq[i+2] = -100,-100,-100

        return seq

    def _tokenizer_image_token(self, prompt, question_tokens=None, return_tensors=None):
        prompt_chunks = [self.processor.tokenizer(prompt.split('<image>')[0]).input_ids]

        if question_tokens is None:
            prompt_chunks.append(self.processor.tokenizer(prompt.split('<image>')[1]).input_ids)
        else:
            prompt_chunks.append(
                self.processor.tokenizer(
                    text=prompt.split('<image>')[1],
                    max_length=question_tokens+self.max_length, 
                    padding='max_length',
                    truncation=True
                ).input_ids
            )
            
        def insert_separator(X, sep):
            return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

        input_ids = []
        offset = 0
        if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == self.processor.tokenizer.bos_token_id:
            offset = 1
            input_ids.append(prompt_chunks[0][0])

        for x in insert_separator(prompt_chunks, [self.IMAGE_TOKEN_INDEX] * (offset + 1)):
            input_ids.extend(x[offset:])

        if return_tensors is not None:
            if return_tensors == 'pt':
                return torch.tensor(input_ids, dtype=torch.long)
            raise ValueError(f'Unsupported tensor type: {return_tensors}')
        return input_ids


@dataclass
class DataCollatorForTraining(VLMsDataCollator):
    """
    Data collator for preparing batches for training with multimodal vision-language models.

    This class selects and applies the appropriate collator function based on the model ID to prepare 
    input data into batches compatible with the model's specific requirements. Supports models like 
    LLaMa 3.2 Vision, Phi-3 Vision, and PaliGemma.

    Parameters
    ----------
    processor : ProcessorMixin
        Processor instance to handle tokenization and data formatting.
    model_id : str
        Model identifier to determine which collator function to apply (e.g., "llama-3", "phi-3", "paligemma").
    device : str
        Device on which the tensors should be stored (e.g., "cpu" or "cuda").
    max_length : int
        Maximum length of the sequence for tokenized inputs.

    Methods
    -------
    __call__(examples)
        Selects and applies the appropriate collator function to prepare input data into batches.
    llama3_collator_for_training(example)
        Prepares data batches for training with the LLaMa 3.2 Vision model.
    phi3_collator_for_training(example)
        Prepares data batches for training with the Phi-3 Vision model.
    paligemma_collator_for_training(example)
        Prepares data batches for training with the PaliGemma model.
    """


    def __call__(self, examples):
        """
        Select and apply the appropriate collator function based on the specified model ID to prepare 
        input data into batches for training or evaluation.

        Parameters
        ----------
        examples : list of dict
            A list containing preprocessed image, question, and answer data for batching.

        Returns
        -------
        dict
            A batch dictionary with processed input data for the specified model.
        """
        batch = None

        mini_batches = []

        for example in examples:
            if "llama-3" in self.model_id.lower():
                tmp = self.llama3_collator_for_training(example)
            elif "phi-3" in self.model_id.lower():
                tmp = self.phi3_collator_for_training(example)
            elif "vitucano" in self.model_id.lower():
                tmp = self.vitucano_collator_for_training(example)
            else:
                tmp = self.paligemma_collator_for_training(example)

            mini_batches.append(tmp)

        if len(examples) == 1:
            batch = tmp
        else:
            batch = {}
            for key in tmp:
                batch[key] = torch.stack([mini_batch[key] for mini_batch in mini_batches])

        return batch


    def vitucano_collator_for_training(self, example):
        """
        Process a single ViTucano example to get unpadded input_ids, labels, and pixel_values.
        """
        # Construct the user-only part of the prompt to determine its length
        user_conv = self.conv_tucano_v0.copy()
        user_conv.append_message(user_conv.roles[0], f"{self.DEFAULT_IMAGE_TOKEN}\n{self.question}")
        user_conv.append_message(user_conv.roles[1], None)
        user_prompt = user_conv.get_prompt()

        question_tokens = len(self.processor.tokenizer(user_prompt).input_ids)
        max_tokens = self.max_length

        # Construct the full prompt with the answer
        full_conv = self.conv_tucano_v0.copy()
        full_conv.append_message(full_conv.roles[0], f"{self.DEFAULT_IMAGE_TOKEN}\n{self.question}")
        full_conv.append_message(full_conv.roles[1], example["answer"])
        full_prompt = full_conv.get_prompt()

        # Correctly tokenize using tokenizer_image_token
        # We get the raw list of tokens first to determine lengths
        image_tokens = len(self.processor.tokenizer(user_prompt.split('<image>')[0]).input_ids) + 1
        question_tokens = len(self.processor.tokenizer(user_prompt.split('<image>')[1]).input_ids)

        input_ids = self._tokenizer_image_token(full_prompt, question_tokens=question_tokens, return_tensors='pt')

        labels = input_ids.clone()
        labels[:image_tokens+question_tokens] = self.IGNORE_INDEX

        pixel_values = self.processor.image_processor(example["image"], return_tensors='pt')['pixel_values']

        return {
            "input_ids": input_ids.unsqueeze(0).to(self.device),
            "labels": labels.unsqueeze(0).to(self.device),
            "images": pixel_values
        }


    def llama3_collator_for_training(self, example):
        """
        Prepare and collate input data into batches for training with the LLaMa 3.2 Vision model.

        Parameters
        ----------
        example : dict
            A dictionary containing preprocessed image, question, and answers.

        Returns
        -------
        dict
            A batch dictionary with processed input data for the model.
        """
        text_prompt = self.processor.tokenizer.apply_chat_template(
            [{"role": "user", "content": f"<|image|>\n{example['question']}"}], 
            tokenize=False, 
            add_generation_prompt=True
        )

        batch = self.processor(
            text=text_prompt,
            images=example["image"],
            return_tensors='pt'
        ).to(self.device)

        question_tokens = len(batch["input_ids"][0]) # number of tokens in question
        max_tokens = self.max_length # Apply max length only to answer instead of question+answer 

        text_prompt += f'{example["answer"]}{self.processor.tokenizer.eos_token}' # add answer to prompt

        batch = self.processor(
            text=text_prompt,
            images=example["image"],
            max_length=question_tokens+max_tokens, 
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).to(self.device)

        labels = batch["input_ids"][0].tolist()
        eot_indices = [i for i,n in enumerate(labels) if n == 128009]
        last_idx = 0
        # system prompt header "<|start_header_id|>system<|end_header_id|>" has been tokenized to [128006, 9125, 128007]
        # user prompt header "<|start_header_id|>user<|end_header_id|>" has been tokenized to [128006, 882, 128007]
        # prompt_header_seqs = [[128006, 9125, 128007],[128006, 882, 128007]]
        prompt_header_seqs = [[128006, 882, 128007]]
        for _, idx in enumerate(eot_indices):
            current_seq = labels[last_idx:idx+1]
            if self._check_header(prompt_header_seqs,current_seq):
                # found prompt header, indicating that this seq should be masked
                labels[last_idx:idx+1] = [-100] * (idx-last_idx+1)
            else:
                last_idx = idx+1
            #  Mask all the assistant header prompt <|start_header_id|>assistant<|end_header_id|>, which has been tokenized to [128006, 78191, 128007]
        assistant_header_seq = [128006, 78191, 128007]
        labels = self._replace_target(assistant_header_seq,labels)
        # Mask the padding token and image token 128256 
        for i in range(len(labels)):
            if labels[i] == self.processor.tokenizer.pad_token_id or labels[i] == 128256: #  128256 is image token index
                labels[i] = -100
        batch["labels"] = torch.tensor([labels]).to(self.device)

        return batch


    def phi3_collator_for_training(self, example):
        """
        Prepare and collate input data into batches for training with the Phi-3 Vision model.

        Parameters
        ----------
        example : dict
            A dictionary containing preprocessed image, question, and answer.

        Returns
        -------
        dict
            A batch dictionary with processed input data for the model.
        """
        text_prompt = self.processor.tokenizer.apply_chat_template(
            [{"role": "user", "content": f"<|image_1|>\n{example['question']}"}], 
            tokenize=False, 
            add_generation_prompt=True
        )

        batch = self.processor(
            text=text_prompt,
            images=example["image"],
            return_tensors='pt'
        ).to(self.device)

        prompt_input_ids = batch['input_ids']
        # Do not add bos token to answer
        answer_input_ids = self.processor.tokenizer(
            text=f'{example["answer"]}<|end|>\n<|endoftext|>',
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            add_special_tokens=False,
            return_tensors='pt'
        )['input_ids'].to(self.device)
        input_ids = torch.cat([prompt_input_ids, answer_input_ids], dim=1).to(self.device)
        # mask questions for labels
        ignore_index = -100
        labels = torch.cat(
            [
                torch.tensor([ignore_index] * len(prompt_input_ids[0])).unsqueeze(0).to(self.device),
                answer_input_ids,
            ],
            dim=1,
        ).to(self.device)

        batch['input_ids'] = input_ids
        del batch['attention_mask']
        batch['labels'] = labels

        return batch


    def paligemma_collator_for_training(self, example):
        """
        Prepare and collate input data into batches for training with the PaliGemma model.

        Parameters
        ----------
        example : dict
            A dictionary containing preprocessed image, question, and answer.

        Returns
        -------
        dict
            A batch dictionary with processed input data for the model.
        """
        batch = self.processor(
            text=example["question"],
            images=example["image"],
            suffix=example["answer"],
            max_length=self.max_length,
            padding='max_length',
            truncation="only_second",
            return_tensors='pt'
        ).to(self.device)

        return batch


@dataclass
class DataCollatorForGeneration(VLMsDataCollator):
    """
    Data collator for preparing batches for generation with vision-language models.

    This class selects and applies the appropriate collator function based on the model ID to format 
    input data into batches compatible with the model's generation requirements. Supports models 
    like LLaMa 3.2 Vision, Phi-3 Vision, and PaliGemma.

    Parameters
    ----------
    processor : ProcessorMixin
        Processor instance to handle tokenization and data formatting.
    model_id : str
        Model identifier to determine which collator function to apply (e.g., "llama-3", "phi-3", "paligemma").
    device : str
        Device on which the tensors should be stored (e.g., "cpu" or "cuda").
    max_length : int
        Maximum length of the sequence for tokenized inputs.

    Methods
    -------
    __call__(examples)
        Selects and applies the appropriate collator function to prepare input data into batches.
    llama3_collator_for_generation(example)
        Prepares data batches for generation with the LLaMa 3.2 Vision model.
    phi3_collator_for_generation(example)
        Prepares data batches for generation with the Phi-3 Vision model.
    paligemma_collator_for_generation(example)
        Prepares data batches for generation with the PaliGemma model.
    """


    def __call__(self, examples):
        """
        Select and apply the appropriate collator function based on the specified model ID to prepare 
        input data into batches for generation.

        Parameters
        ----------
        examples : list of dict
            A list containing preprocessed image, question, and answer data for batching.

        Returns
        -------
        dict
            A batch dictionary with processed input data for the specified model.
        """
        batch = None

        mini_batches = []

        for example in examples:
            if "llama-3" in self.model_id.lower():
                tmp = self.llama3_collator_for_generation(example)
            elif "phi-3" in self.model_id.lower():
                tmp = self.phi3_collator_for_generation(example)
            elif "vitucano" in self.model_id.lower():
                tmp = self.vitucano_collator_for_generation(example)
            else:
                tmp = self.paligemma_collator_for_generation(example)

            mini_batches.append(tmp)

        if len(examples) == 1:
            batch = tmp
        else:
            batch = {}
            for key in tmp:
                batch[key] = torch.stack([mini_batch[key] for mini_batch in mini_batches])

        return batch


    def vitucano_collator_for_generation(self, example):
        """
        Prepares a single data sample for generation with the ViTucano model.
        """
        conv = self.conv_tucano_v0.copy()
        conv.append_message(conv.roles[0], f"{self.DEFAULT_IMAGE_TOKEN}\n{self.question}")
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # Process the text to get the correct separation for system tokens, image token, assistant tokens and user tokens
        input_ids = self._tokenizer_image_token(prompt, return_tensors='pt')

        # Process the image to get pixel values
        pixel_values = self.processor.image_processor(example["image"], return_tensors='pt')['pixel_values']

        return {
            "inputs": input_ids.unsqueeze(0).to(self.device),
            "images": pixel_values.to(self.device) # Return the processed image tensor
        }


    def llama3_collator_for_generation(self, example):
        """
        Prepare and collate input data into batches for generation with the LLaMa 3.2 Vision model.

        Parameters
        ----------
        example : dict
            A dictionary containing preprocessed image, question, and answers.

        Returns
        -------
        dict
            A batch dictionary with processed input data for the model.
        """
        text_prompt = self.processor.tokenizer.apply_chat_template(
            [{"role": "user", "content": f"<|image|>\n{self.question}"}],
            tokenize=False,
            add_generation_prompt=True
        )
        return self.processor(
            text=text_prompt,
            images=example["image"],
            return_tensors='pt'
        ).to(self.device)


    def phi3_collator_for_generation(self, example):
        """
        Prepare and collate input data into batches for generation with the Phi-3 Vision model.

        Parameters
        ----------
        example : dict
            A dictionary containing preprocessed image, question, and answer.

        Returns
        -------
        dict
            A batch dictionary with processed input data for the model.
        """
        text_prompt = self.processor.tokenizer.apply_chat_template(
            [{"role": "user", "content": f"<|image_1|>\n{self.question}"}],
            tokenize=False,
            add_generation_prompt=True
        )
        return self.processor(
            text=f'<|user|>\n<|image_1|>\n{self.question}<|end|>\n<|assistant|>\n',
            images=[example["image"]],
            return_tensors='pt'
        ).to(self.device)


    def paligemma_collator_for_generation(self, example):
        """
        Prepare and collate input data into batches for generation with the PaliGemma model.

        Parameters
        ----------
        example : dict
            A dictionary containing preprocessed image, question, and answer.

        Returns
        -------
        dict
            A batch dictionary with processed input data for the model.
        """
        return self.processor(
            text=self.question,
            images=example["image"],
            return_tensors='pt'
        ).to(self.device)