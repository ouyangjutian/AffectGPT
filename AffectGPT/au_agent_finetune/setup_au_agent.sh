#!/bin/bash
# AU Agent一键设置脚本

echo "========================================="
echo "AU Agent Setup - Complete Pipeline"
echo "========================================="
echo ""

# 配置
MER_FACTORY_OUTPUT="/home/project/MER-Factory/output"
LLAMA_FACTORY_PATH="/home/project/LLaMA-Factory"
BASE_MODEL="/home/project/Dataset/Emotion/tools/transformer/LLM/Qwen2.5-7B-Instruct"

# 检查依赖
echo "Step 0: Checking dependencies..."
echo ""

if [ ! -d "$MER_FACTORY_OUTPUT" ]; then
    echo "❌ MER-Factory output not found: $MER_FACTORY_OUTPUT"
    echo "Please run MER-Factory AU analysis first"
    exit 1
fi

if [ ! -d "$LLAMA_FACTORY_PATH" ]; then
    echo "⚠️ LLaMA-Factory not found, installing..."
    git clone https://github.com/hiyouga/LLaMA-Factory.git "$LLAMA_FACTORY_PATH"
    cd "$LLAMA_FACTORY_PATH"
    pip install -e .
    cd -
fi

if [ ! -d "$BASE_MODEL" ]; then
    echo "❌ Base model not found: $BASE_MODEL"
    echo "Please download Qwen2.5-7B-Instruct first"
    exit 1
fi

echo "✅ All dependencies ready"
echo ""

# 步骤1: 准备数据集
echo "========================================="
echo "Step 1: Preparing AU instruction dataset"
echo "========================================="
echo ""

python prepare_au_instruction_dataset.py

if [ $? -ne 0 ]; then
    echo "❌ Dataset preparation failed"
    exit 1
fi

echo ""
echo "✅ Dataset ready"
echo ""

# 步骤2: 训练AU Agent
echo "========================================="
echo "Step 2: Training AU Agent (this will take 8-12 hours)"
echo "========================================="
echo ""

echo "Starting training automatically..."
bash train_au_agent.sh

if [ $? -ne 0 ]; then
    echo "❌ Training failed"
    exit 1
fi

echo ""
echo "✅ Training complete"

echo ""

# 步骤3: 测试AU Agent
echo "========================================="
echo "Step 3: Testing AU Agent"
echo "========================================="
echo ""

if [ -d "/home/project/AffectGPT/AffectGPT/output/au_agent_qwen2.5_7b_lora/checkpoint-best" ]; then
    echo "Testing AU Agent automatically..."
    python test_au_agent.py
else
    echo "⚠️ AU Agent checkpoint not found, skipping test"
fi

echo ""

# 完成
echo "========================================="
echo "Setup Complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo "1. Review AU Agent output quality"
echo "2. Integrate into AffectGPT training config"
echo "3. Run training: python train.py --cfg-path train_configs/config_with_au_agent.yaml"
echo ""
echo "See AU_AGENT_INTEGRATION_GUIDE.md for detailed instructions"
