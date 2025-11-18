"""
快速测试AU Agent是否可以正常加载和推理
"""
import sys
sys.path.append('.')

from my_affectgpt.models.au_agent import AUAgent

def test_au_agent():
    print("=" * 60)
    print("测试AU Agent模型加载和推理")
    print("=" * 60)
    
    # 配置
    base_model = "/home/project/Dataset/Emotion/tools/transformer/LLM/Qwen2.5-7B-Instruct"
    lora_weights = "/home/project/AffectGPT/AffectGPT/output/au_agent_qwen2.5_7b_lora"
    
    # 加载模型
    print("\n[1/3] 加载AU Agent模型...")
    try:
        au_agent = AUAgent(
            base_model_path=base_model,
            lora_weights_path=lora_weights,
            device='cuda',
            use_lora=True
        )
        print("✅ 模型加载成功！")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return
    
    # 测试样例（来自MER-Factory）
    print("\n[2/3] 准备测试数据...")
    test_au_values = {
        'AU04_r': 0.88,
        'AU10_r': 2.37,
        'AU12_r': 1.73,
        'AU14_r': 2.45
    }
    test_au_description = "Brow lowerer (intensity: 0.88), Upper lip raiser (intensity: 2.37), Lip corner puller (smile) (intensity: 1.73), Dimpler (intensity: 2.45)"
    
    print(f"AU values: {test_au_values}")
    print(f"AU description: {test_au_description}")
    
    # 生成描述
    print("\n[3/3] 生成Facial Content描述...")
    try:
        description = au_agent.generate_description(
            au_values=test_au_values,
            au_description=test_au_description,
            temperature=0.7,
            max_length=256
        )
        print("✅ 生成成功！")
        print("\n" + "=" * 60)
        print("生成的描述：")
        print("=" * 60)
        print(description)
        print("=" * 60)
    except Exception as e:
        print(f"❌ 生成失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "=" * 60)
    print("✅ AU Agent测试通过！")
    print("=" * 60)
    print("\n下一步：")
    print("1. 模型已可用，可以开始AffectGPT训练")
    print("2. 配置文件: train_configs/emercoarse_highlevelfilter4_outputhybird_bestsetup_bestfusion_lz_face_frame.yaml")
    print("3. 确保已生成所有数据集的AU result（使用MER-Factory）")

if __name__ == '__main__':
    test_au_agent()
