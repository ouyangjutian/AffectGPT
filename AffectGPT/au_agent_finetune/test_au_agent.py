#!/usr/bin/env python3
"""
测试AU Agent
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def load_au_agent(
    base_model_path: str,
    lora_weights_path: str,
    device: str = "cuda"
):
    """加载AU Agent（基础模型 + LoRA权重）"""
    print("Loading AU Agent...")
    
    # 加载基础模型
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # 加载LoRA权重
    model = PeftModel.from_pretrained(base_model, lora_weights_path)
    model = model.merge_and_unload()  # 合并权重
    model.eval()
    
    print("✅ AU Agent loaded successfully")
    return model, tokenizer


def generate_au_description(
    model,
    tokenizer,
    au_values: dict,
    instruction: str = "Based on the following Action Unit detections, describe the facial expression:",
    max_length: int = 512
) -> str:
    """
    使用AU Agent生成AU描述
    
    Args:
        model: AU Agent模型
        tokenizer: Tokenizer
        au_values: AU检测结果，如 {"AU01": 0.98, "AU05": 0.98, ...}
        instruction: 指令文本
        max_length: 最大生成长度
    
    Returns:
        AU描述文本
    """
    # 构建输入
    au_input = ", ".join([f"AU{au_id}: {value:.2f}" for au_id, value in au_values.items()])
    
    prompt = f"{instruction}\n\nAction Units: {au_input}\n\nDescription:"
    
    # Qwen2.5模板
    messages = [
        {"role": "system", "content": "You are an expert in facial expression analysis and Action Unit interpretation."},
        {"role": "user", "content": prompt}
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    # 生成
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
    
    # 解码
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 提取描述部分
    if "Description:" in response:
        description = response.split("Description:")[-1].strip()
    else:
        description = response.split("assistant")[-1].strip() if "assistant" in response else response
    
    return description


def test_au_agent():
    """测试AU Agent"""
    print("="*60)
    print("AU Agent Testing")
    print("="*60)
    
    # 配置
    BASE_MODEL = "/home/project/Dataset/Emotion/tools/transformer/LLM/Qwen2.5-7B-Instruct"
    LORA_WEIGHTS = "../output/au_agent_qwen2.5_7b_lora/checkpoint-best"  # 在上级output目录
    
    # 加载模型
    model, tokenizer = load_au_agent(BASE_MODEL, LORA_WEIGHTS)
    
    # 测试样例
    test_cases = [
        {
            "name": "Neutral Expression",
            "au_values": {
                "AU01": 0.98,
                "AU05": 0.98,
                "AU07": 2.35,
                "AU25": 1.76
            }
        },
        {
            "name": "Genuine Smile",
            "au_values": {
                "AU06": 3.50,
                "AU12": 3.20,
                "AU25": 1.50
            }
        },
        {
            "name": "Anger/Frustration",
            "au_values": {
                "AU04": 3.00,
                "AU07": 2.50,
                "AU23": 2.00
            }
        }
    ]
    
    # 逐个测试
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"Test Case {i}: {test_case['name']}")
        print(f"{'='*60}")
        print(f"AU Values: {test_case['au_values']}")
        
        description = generate_au_description(model, tokenizer, test_case['au_values'])
        
        print(f"\nGenerated Description:")
        print(f"  {description}")
    
    print("\n" + "="*60)
    print("Testing complete!")
    print("="*60)


if __name__ == '__main__':
    test_au_agent()
