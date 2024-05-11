from typing import List, Optional

import typer

from src.generation.generate import generate
from src.trainer.train_qlora import train

application = typer.Typer(name="llama-receipts")


@application.command("train")
def train_model(
    base_model: str = typer.Option(
        ..., "--base-model", exists=True, help="Path to the base model"
    ),
    data_path: str = typer.Option(
        ..., "--data-path", exists=True, help="Path to the data"
    ),
    output_dir: str = typer.Option(
        "../models/", "--output-dir", exists=True, help="Path to the output directory"
    ),
    # training hyperparams
    batch_size: int = typer.Option(128, "--batch-size", help="Batch size"),
    micro_batch_size: int = typer.Option(
        2, "--micro-batch-size", help="Micro batch size"
    ),
    num_epochs: int = typer.Option(1, "--num-epochs", help="Number of epochs"),
    learning_rate: float = typer.Option(2e-5, "--learning-rate", help="Learning rate"),
    cutoff_len: int = typer.Option(1024, "--cutoff-len", help="Cutoff length"),
    val_set_size: float = typer.Option(
        0.0, "--val-set-size", help="Validation set size"
    ),
    # lora hyperparams
    lora_r: int = typer.Option(16, "--lora-r", help="Lora r"),
    lora_alpha: int = typer.Option(16, "--lora-alpha", help="Lora alpha"),
    lora_dropout: float = typer.Option(0.05, "--lora-dropout", help="Lora dropout"),
    lora_target_modules: List[str] = typer.Option(
        ["q_proj", "v_proj"],
        "--lora-target-modules",
        help="Lora target modules",
    ),
    # llm hyperparams
    group_by_length: bool = typer.Option(
        False, "--group-by-length", help="Group by length"
    ),  # faster, but produces an odd training loss curve
    resume_from_checkpoint: str = typer.Option(
        None, "--resume-from-checkpoint", help="Resume from checkpoint"
    ),  # either training checkpoint or final adapter
    use_wandb: bool = typer.Option(True, "--use-wandb", help="Use wandb"),
    wandb_run_name: Optional[str] = typer.Option(
        None, "--wandb-run-name", help="Wandb run name"
    ),
):
    train(
        base_model=base_model,
        data_path=data_path,
        output_dir=output_dir,
        batch_size=batch_size,
        micro_batch_size=micro_batch_size,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        cutoff_len=cutoff_len,
        val_set_size=val_set_size,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_target_modules=lora_target_modules,
        group_by_length=group_by_length,
        resume_from_checkpoint=resume_from_checkpoint,
        use_wandb=use_wandb,
        wandb_run_name=wandb_run_name,
    )

@application.command("generate")
def generate_model(
    load_8bit: bool = typer.Option(
        False, "--load-8bit", help="Load 8bit model"
    ),
    base_model: str = typer.Option(
        ..., "--base-model", help="Path to the base model"
    ),
    lora_weights: str = typer.Option(
        ..., "--lora-weights", exists=True, help="Path to the lora weights"
    ),
) -> None:
    generate(
        load_8bit=load_8bit,
        base_model=base_model,
        lora_weights=lora_weights,
    )



@application.callback()
def dummy_to_force_subcommand() -> None:
    """
    This function exists because Typer won't let you force a single subcommand.
    Since we know we will add other subcommands in the future and don't want to
    break the interface, we have to use this workaround.

    Delete this when a second subcommand is added.
    """
    pass


if __name__ == "__main__":
    application()
