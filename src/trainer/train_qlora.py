import os
import sys
from typing import List, Optional

import bitsandbytes as bnb
import torch
import transformers
import wandb
from datasets import load_dataset, load_from_disk
from loguru import logger
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
)

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

from bitsandbytes.cuda_setup.main import CUDASetup

setup = CUDASetup.get_instance()

HUMAN_PROMPT = "Expand abbreviations for products (receipt items), additionally highlight the product, brand and category: "
ASSISTANT_PROMPT = "Normlized product: "


class QLoraTrainer(Trainer):
    def save_model(self, output_dir=None, _internal_call=False):
        from transformers.trainer import TRAINING_ARGS_NAME

        os.makedirs(output_dir, exist_ok=True)
        logger.info("model save path: {}".format(output_dir))
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
        self.model.save_pretrained(output_dir)


class CherryPickCallback(TrainerCallback):
    def __init__(self, tokenizer, num_examples=1):
        self.tokenizer = tokenizer
        self.num_examples = num_examples

    def on_evaluate(
        self,
        args: transformers.TrainingArguments,
        state: transformers.TrainerState,
        control: transformers.TrainerControl,
        **kwargs,
    ):
        model = kwargs["model"]
        eval_dataloader = kwargs["eval_dataloader"]

        if state.global_step % (args.logging_steps * 5) == 0:
            # Obtain a few example batches from the evaluation dataloader
            example_batches = self._get_example_batches(
                eval_dataloader, num_batches=self.num_examples
            )

            # Generate predictions for these examples
            predictions_texts = self._generate_predictions(model, example_batches)
            # Log the predictions to wandb
            self._log_to_wandb(example_batches, predictions_texts)

    def _get_example_batches(self, dataloader, num_batches):
        example_batches = []

        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
            example_batches.append(batch)

        return example_batches

    def _generate_predictions(self, model, example_batches):
        model.eval()  # Make sure the model is in evaluation mode
        predictions_texts = []

        with torch.no_grad():
            for batch in example_batches:
                inputs = batch["input_ids"].to(model.device)
                attention_mask = (
                    batch.get("attention_mask").to(model.device)
                    if "attention_mask" in batch
                    else None
                )

                # Generate predictions
                outputs = model.generate(inputs, attention_mask=attention_mask)

                # Decode the generated text
                decoded_preds = [
                    self.tokenizer.decode(output, skip_special_tokens=True)
                    for output in outputs
                ]
                predictions_texts.extend(decoded_preds)

        return predictions_texts

    def _log_to_wandb(self, example_batches, predictions):
        for batch, prediction in zip(example_batches, predictions):
            input_text = self.tokenizer.decode(
                batch["input_ids"][0], skip_special_tokens=True
            )
            wandb.log({"example_text": input_text, "prediction": prediction})

            print(f"Input text: {input_text}\nPrediction: {prediction}\n\n")


def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


def train(
    base_model: str,
    data_path: str,
    output_dir: str,
    batch_size: int,
    micro_batch_size: int,
    num_epochs: int,
    learning_rate: float,
    cutoff_len: int,
    val_set_size: float,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    lora_target_modules: List[str],
    group_by_length: bool,
    resume_from_checkpoint: str,
    use_wandb: bool = True,
    wandb_run_name: Optional[str] = None,
):
    logger.info(
        f"Training model with params:\n"
        f"base_model: {base_model}\n"
        f"data_path: {data_path}\n"
        f"output_dir: {output_dir}\n"
        f"batch_size: {batch_size}\n"
        f"micro_batch_size: {micro_batch_size}\n"
        f"num_epochs: {num_epochs}\n"
        f"learning_rate: {learning_rate}\n"
        f"cutoff_len: {cutoff_len}\n"
        f"val_set_size: {val_set_size}\n"
        f"lora_r: {lora_r}\n"
        f"lora_alpha: {lora_alpha}\n"
        f"lora_dropout: {lora_dropout}\n"
        f"lora_target_modules: {lora_target_modules}\n"
        f"group_by_length: {group_by_length}\n"
        f"resume_from_checkpoint: {resume_from_checkpoint}\n"
    )
    assert base_model, "Please specify a --base_model"
    gradient_accumulation_steps = batch_size // micro_batch_size

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # load model in 4bit
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
    )
    for _ in range(5):
        try:
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                device_map=device_map,
                trust_remote_code=True,
                load_in_4bit=True,
                torch_dtype=torch.float16,
                quantization_config=quantization_config,
            )
        except:
            continue

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    # # baichuan model without pad token
    # tokenizer.add_special_tokens({"pad_token": "<pad>"})
    # model.resize_token_embeddings(len(tokenizer))

    def tokenize_conversation(example):
        conversations = example["dialogue"]

        input_ids = []
        labels = []
        for cid, conversation in enumerate(conversations):
            human_text = conversation["human"]
            human_text = HUMAN_PROMPT + human_text
            assistant_text = conversation["assistant"]
            assistant_text = ASSISTANT_PROMPT + assistant_text

            if cid == 0:
                human_text = tokenizer.bos_token + human_text + tokenizer.eos_token
            else:
                human_text = human_text + tokenizer.eos_token
            assistant_text += tokenizer.eos_token

            human_ids = tokenizer.encode(human_text)
            assistant_ids = tokenizer.encode(assistant_text)

            input_ids += human_ids
            labels += len(human_ids) * [-100]

            input_ids += assistant_ids
            labels += assistant_ids

        result = {"input_ids": input_ids, "labels": labels}
        return result

    def data_collator(features: list) -> dict:
        # cut off the input and label
        input_ids_list = [feature["input_ids"][:cutoff_len] for feature in features]
        labels_list = [feature["labels"][:cutoff_len] for feature in features]

        # pad token from left
        input_ids = pad_sequence(
            [torch.tensor(input_ids[::-1]) for input_ids in input_ids_list],
            batch_first=True,
            padding_value=tokenizer.pad_token_id,
        ).flip(dims=[1])
        labels = pad_sequence(
            [torch.tensor(labels[::-1]) for labels in labels_list],
            batch_first=True,
            padding_value=-100,
        ).flip(dims=[1])

        input_ids = input_ids.long()
        labels = labels.long()

        return {
            "input_ids": input_ids,
            "attention_mask": input_ids.ne(tokenizer.pad_token_id),
            "labels": labels,
        }

    model = prepare_model_for_int8_training(model, use_gradient_checkpointing=True)

    # add adapter modules for all linear layer
    lora_target_modules = find_all_linear_names(model)
    logger.info("lora target modules: {}".format(lora_target_modules))

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)

    if data_path.endswith(".jsonl"):
        data = load_dataset("json", data_files=data_path)
    else:
        data = load_from_disk(data_path)

    if resume_from_checkpoint:
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
            resume_from_checkpoint = False  # So the trainer won't try loading its state
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            logger.info(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            model = set_peft_model_state_dict(model, adapters_weights)
        else:
            logger.info(f"Checkpoint {checkpoint_name} not found")

    model.print_trainable_parameters()
    model.config.torch_dtype = torch.float32

    logger.info("data info: {}".format(data))
    if val_set_size > 0:
        train_val = data["train"].train_test_split(
            test_size=val_set_size, shuffle=True, seed=42
        )
        train_data = (
            train_val["train"]
            .shuffle()
            .map(tokenize_conversation, remove_columns=train_val["train"].column_names)
        )
        val_data = (
            train_val["test"]
            .shuffle()
            .map(tokenize_conversation, remove_columns=train_val["test"].column_names)
        )
    else:
        train_data = (
            data["train"]
            .shuffle()
            .map(tokenize_conversation, remove_columns=data["train"].column_names)
        )
        val_data = (
            data["test"]
            .shuffle()
            .map(tokenize_conversation, remove_columns=data["test"].column_names)
        )

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    train_args = transformers.TrainingArguments(
        per_device_train_batch_size=micro_batch_size,
        per_device_eval_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        # warmup_steps=16,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        fp16=True,
        logging_steps=8,
        optim="adamw_torch",
        evaluation_strategy="steps" if val_set_size > 0 else "no",
        save_strategy="steps",
        eval_steps=16,
        save_steps=16,
        output_dir=output_dir,
        save_total_limit=3,
        load_best_model_at_end=True if val_set_size > 0 else False,
        ddp_find_unused_parameters=False if ddp else None,
        group_by_length=group_by_length,
        report_to="wandb" if use_wandb else None,
        run_name=wandb_run_name if use_wandb else None,
    )

    trainer = QLoraTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=train_args,
        data_collator=data_collator,
        callbacks=[
            CherryPickCallback(
                tokenizer=tokenizer,
            ),
            EarlyStoppingCallback(
                early_stopping_patience=5,
            ),
        ],
    )
    model.config.use_cache = False

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    model.save_pretrained(output_dir)
