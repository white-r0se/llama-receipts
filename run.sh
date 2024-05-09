poetry run python -m src.cli train \
    --base-model "unsloth/llama-3-8b-Instruct" \
    --data-path "dialogues/receipts_dataset.jsonl" \
    --output-dir "models/" \
    --batch-size 1 \
    --micro-batch-size 1 \
    --num-epochs 1 \
    --wandb-run-name receipts-llama
