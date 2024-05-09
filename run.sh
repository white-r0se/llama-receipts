poetry run src/cli.py train \
    --base-model "unsloth/llama-3-8b-Instruct" \
    --data_path "dialogues/receipts_dataset.jsonl" \
    --output-dir "models/" \
    --batch-size 1 \
    --micro-batch-size 1 \
    --num-epochs 1
