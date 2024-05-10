import torch
import transformers
import wandb
from transformers import TrainerCallback


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
