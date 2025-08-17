#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
修复版Qwen3-0.6B微调训练脚本
解决tokenization问题
"""

import json
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import (
    LoraConfig, 
    get_peft_model, 
    TaskType,
)
from datasets import Dataset
import os
import psutil

def load_and_prepare_data(data_path):
    """加载并准备数据"""
    print(f"正在加载训练数据: {data_path}")
    
    conversations = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            sample = json.loads(line.strip())
            prompt = sample['prompt']
            intents = sample['output']
            
            # 格式化为对话
            intent_text = ", ".join(intents)
            conversation = f"用户: {prompt}\\n助手: {intent_text}"
            conversations.append(conversation)
    
    print(f"数据集准备完成，共 {len(conversations)} 个样本")
    return conversations

def tokenize_data(texts, tokenizer):
    """简单的tokenization"""
    print("正在tokenize数据...")
    
    tokenized = []
    for text in texts:
        tokens = tokenizer(
            text,
            truncation=True,
            padding=False,
            max_length=256,
            return_tensors=None,
        )
        # 添加labels
        tokens["labels"] = tokens["input_ids"].copy()
        tokenized.append(tokens)
    
    return tokenized

def main():
    """主函数"""
    print("🎯 修复版Qwen3-0.6B意图分类器训练")
    print("=" * 50)
    
    # 检查设备
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✅ 使用Apple Metal Performance Shaders (MPS)加速")
    else:
        device = torch.device("cpu")
        print("⚠️ 使用CPU")
    
    # 检查内存
    memory = psutil.virtual_memory()
    print(f"系统内存: {memory.total // (1024**3)}GB")
    print(f"可用内存: {memory.available // (1024**3)}GB")
    
    # 加载模型和tokenizer
    model_name = "Qwen/Qwen3-0.6B"
    print(f"正在加载模型: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="right"
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    
    model = model.to(device)
    print("模型和分词器加载完成")
    
    # 设置LoRA
    print("正在配置LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        bias="none",
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # 准备数据
    data_path = "data/enhanced_multi_intent_training_data.jsonl"
    conversations = load_and_prepare_data(data_path)
    
    # Tokenize数据
    tokenized_data = tokenize_data(conversations, tokenizer)
    
    # 创建Dataset
    dataset = Dataset.from_list(tokenized_data)
    print(f"创建数据集完成，共 {len(dataset)} 个样本")
    
    # 数据收集器
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # 训练参数
    output_dir = "models/qwen3_fixed_classifier"
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=2,
        learning_rate=2e-4,
        fp16=False,
        logging_steps=50,
        save_steps=500,
        save_total_limit=2,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
        report_to=None,
        warmup_steps=20,
        max_grad_norm=1.0,
        save_safetensors=True,
    )
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )
    
    # 开始训练
    print("🚀 开始训练...")
    try:
        trainer.train()
        print("✅ 训练完成")
        
        # 清理缓存
        if device.type == "mps":
            torch.mps.empty_cache()
            print("🧹 MPS缓存已清理")
        
        # 保存模型
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        print(f"✅ 模型已保存到: {output_dir}")
        
        # 快速测试
        print("\\n🧪 快速测试...")
        test_prompts = [
            "我想查看订单状态",
            "登录后查看订单并处理支付", 
            "支付成功后发送通知",
            "今天天气怎么样"
        ]
        
        for prompt in test_prompts:
            input_text = f"用户: {prompt}\\n助手: "
            inputs = tokenizer(input_text, return_tensors="pt", max_length=256, truncation=True)
            
            if device.type != "cpu":
                inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=20,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            predicted = response.split("助手: ")[-1].strip()
            
            print(f"输入: {prompt}")
            print(f"预测: {predicted}")
            print("-" * 40)
        
        print("🎉 训练和测试完成！")
        
    except Exception as e:
        print(f"❌ 训练出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()