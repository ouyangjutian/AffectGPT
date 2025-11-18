#!/bin/bash
# AU Agent训练脚本

echo "========================================="
echo "AU Agent Training with LLaMA-Factory"
echo "========================================="

# 配置
BASE_MODEL="/home/project/Dataset/Emotion/tools/transformer/LLM/Qwen2.5-7B-Instruct"
TRAIN_DATASET_PATH="/home/project/AffectGPT/AffectGPT/au_agent_finetune/au_instruction_dataset_train.jsonl"
VAL_DATASET_PATH="/home/project/AffectGPT/AffectGPT/au_agent_finetune/au_instruction_dataset_val.jsonl"
OUTPUT_DIR="/home/project/AffectGPT/AffectGPT/output/au_agent_qwen2.5_7b_lora"
LLAMA_FACTORY_PATH="/home/project/LLaMA-Factory"  # 根据实际路径修改

# 检查数据集
if [ ! -f "$TRAIN_DATASET_PATH" ]; then
    echo "❌ Train dataset not found: $TRAIN_DATASET_PATH"
    echo "Please run: python prepare_au_instruction_dataset.py"
    exit 1
fi

if [ ! -f "$VAL_DATASET_PATH" ]; then
    echo "❌ Val dataset not found: $VAL_DATASET_PATH"
    echo "Please run: python prepare_au_instruction_dataset.py"
    exit 1
fi

# 切换到LLaMA-Factory目录
cd "$LLAMA_FACTORY_PATH" || exit 1

# 注册数据集到LLaMA-Factory
echo "Registering dataset..."
cat > data/dataset_info.json.tmp << EOF
{
  "au_instruction_train": {
    "file_name": "$TRAIN_DATASET_PATH",
    "formatting": "alpaca",
    "columns": {
      "prompt": "instruction",
      "query": "input",
      "response": "output"
    }
  },
  "au_instruction_val": {
    "file_name": "$VAL_DATASET_PATH",
    "formatting": "alpaca",
    "columns": {
      "prompt": "instruction",
      "query": "input",
      "response": "output"
    }
  }
}
EOF

# 合并到现有配置
if [ -f data/dataset_info.json ]; then
    python -c "
import json
with open('data/dataset_info.json', 'r') as f:
    existing = json.load(f)
with open('data/dataset_info.json.tmp', 'r') as f:
    new_data = json.load(f)
existing.update(new_data)
with open('data/dataset_info.json', 'w') as f:
    json.dump(existing, f, indent=2, ensure_ascii=False)
"
else
    mv data/dataset_info.json.tmp data/dataset_info.json
fi

rm -f data/dataset_info.json.tmp
echo "✅ Dataset registered"

# 启动训练
echo ""
echo "Starting training..."
echo ""

CUDA_VISIBLE_DEVICES=0 python src/train.py \
    --model_name_or_path "$BASE_MODEL" \
    --do_train \
    --do_eval \
    --dataset au_instruction_train \
    --eval_dataset au_instruction_val \
    --template qwen \
    --finetuning_type lora \
    --lora_target all \
    --lora_rank 64 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --output_dir "$OUTPUT_DIR" \
    --overwrite_output_dir \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-5 \
    --num_train_epochs 3 \
    --lr_scheduler_type cosine \
    --warmup_steps 100 \
    --logging_steps 10 \
    --save_steps 500 \
    --eval_steps 500 \
    --eval_strategy steps \
    --save_strategy steps \
    --save_total_limit 3 \
    --load_best_model_at_end \
    --metric_for_best_model eval_loss \
    --bf16 \
    --gradient_checkpointing \
    --report_to tensorboard \
    --cutoff_len 1024 \
    --preprocessing_num_workers 16

echo ""
echo "========================================="
echo "Training complete!"
echo "========================================="
echo "Output directory: $OUTPUT_DIR"
echo ""
echo "Next steps:"
echo "1. Test AU Agent: python test_au_agent.py"
echo "2. Merge LoRA weights: python merge_au_agent_weights.py"
echo "3. Integrate into AffectGPT"
