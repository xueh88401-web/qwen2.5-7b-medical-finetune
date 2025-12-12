#!/usr/bin/env python3
"""
Qwen2.5-7B 医学模型推理演示脚本。
用途：1. 快速测试模型效果；2. 作为项目使用示例。
使用方法: python inference_demo.py --query "你的问题"
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import argparse
import sys
import os

def load_model_and_tokenizer(adapter_path="./adapter", use_hf_adapter=False):
    """
    加载基础模型和LoRA适配器。
    参数:
        adapter_path: 本地LoRA适配器目录路径，默认为'./adapter'
        use_hf_adapter: 是否使用HuggingFace上的适配器
    返回:
        model, tokenizer
    """
    print("=== 正在加载模型，首次加载可能需要几分钟... ===")
    
    # 基础模型ID
    base_model_id = "Qwen/Qwen2.5-7B-Instruct"
    
    try:
        # 1. 加载分词器
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_id,
            trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("✅ 分词器加载完成")
        
        # 2. 决定使用本地适配器还是HF适配器
        if use_hf_adapter:
            adapter_id = "xueh88401/qwen2.5-7b-medical"
            print(f"使用HuggingFace适配器: {adapter_id}")
        else:
            adapter_id = adapter_path
            print(f"使用本地适配器: {adapter_path}")
        
        # 3. 加载基础模型
        # 注意：如果你的显存小于16GB，可以尝试添加 load_in_4bit=True 或 load_in_8bit=True
        model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=torch.float16,
            device_map="auto",  # 自动分配GPU/CPU
            trust_remote_code=True
        )
        print("✅ 基础模型加载完成")
        
        # 4. 加载LoRA适配器
        model = PeftModel.from_pretrained(model, adapter_id)
        model.eval()  # 切换到评估模式
        print("✅ LoRA适配器加载完成")
        
        return model, tokenizer
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        print("\n💡 可能的原因及解决方案:")
        print("1. 显存不足 -> 添加参数: load_in_4bit=True")
        print("2. 网络问题 -> 确保可以访问HuggingFace")
        print("3. 路径错误 -> 检查adapter_path参数")
        sys.exit(1)

def generate_response(query, model, tokenizer, max_new_tokens=300):
    """
    生成回复的核心函数。
    参数:
        query: 用户输入的问题
        model: 加载好的模型
        tokenizer: 加载好的分词器
        max_new_tokens: 最大生成长度
    返回:
        模型生成的回复文本
    """
    # 构建符合Qwen2.5对话格式的输入
    prompt = f"用户：{query}\n助手："
    
    # 编码输入
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # 生成配置（你可以调整这些参数）
    generate_kwargs = {
        "max_new_tokens": max_new_tokens,
        "temperature": 0.7,      # 控制随机性：越低越确定，越高越多样
        "top_p": 0.9,           # 核采样参数
        "do_sample": True,      # 启用采样
        "repetition_penalty": 1.1,  # 重复惩罚
    }
    
    # 开始生成
    print("正在生成回复...")
    with torch.no_grad():
        outputs = model.generate(**inputs, **generate_kwargs)
    
    # 解码并提取助手的新回复部分
    response = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:], 
        skip_special_tokens=True
    )
    
    return response.strip()

def main():
    """主函数：解析命令行参数并运行推理"""
    parser = argparse.ArgumentParser(description="Qwen2.5-7B医学模型推理演示")
    parser.add_argument("--query", type=str, default="感冒了怎么办？",
                       help="要咨询的医疗问题（默认：'感冒了怎么办？'）")
    parser.add_argument("--adapter_path", type=str, default="./adapter",
                       help="LoRA适配器本地路径（默认：'./adapter'）")
    parser.add_argument("--hf", action="store_true",
                       help="使用HuggingFace上的适配器（而不是本地文件）")
    parser.add_argument("--max_tokens", type=int, default=300,
                       help="最大生成长度（默认：300）")
    
    args = parser.parse_args()
    
    # 检查本地适配器路径是否存在（如果不使用HF适配器）
    if not args.hf and not os.path.exists(args.adapter_path):
        print(f"⚠️  本地适配器路径 '{args.adapter_path}' 不存在。")
        print("将自动切换到HuggingFace适配器...")
        args.hf = True
    
    # 加载模型
    model, tokenizer = load_model_and_tokenizer(args.adapter_path, args.hf)
    
    print(f"\n{'='*50}")
    print(f"问题：{args.query}")
    print(f"{'='*50}")
    
    # 生成回复
    response = generate_response(args.query, model, tokenizer, args.max_tokens)
    
    print(f"\n🤖 助手回复：\n{response}")
    print(f"\n{'='*50}")
    print(f"回复长度：{len(response)} 字符")

if __name__ == "__main__":
    main()