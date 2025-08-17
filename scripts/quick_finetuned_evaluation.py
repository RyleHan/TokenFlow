#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
快速评估微调后Qwen3模型
简化版本，专注核心指标
"""

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import time
from typing import List, Dict

def load_test_samples(n=20):
    """加载少量测试样本"""
    samples = []
    with open("data/enhanced_multi_intent_training_data.jsonl", 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            sample = json.loads(line.strip())
            samples.append({
                "prompt": sample["prompt"],
                "expected": sample["output"]
            })
    return samples

def quick_predict(model, tokenizer, prompt, device):
    """快速预测"""
    input_text = f"用户: {prompt}\\n助手: "
    
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=256)
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
    
    # 简单解析意图
    intent_categories = ["order_management", "user_auth", "payment", "inventory", "notification", "none"]
    predicted_intents = []
    
    for intent in intent_categories:
        if intent in predicted.lower():
            predicted_intents.append(intent)
    
    if not predicted_intents:
        predicted_intents = ["none"]
    
    return predicted_intents

def calculate_accuracy(predictions, targets):
    """计算准确率"""
    exact_match = 0
    partial_match = 0
    total = len(predictions)
    
    for pred, target in zip(predictions, targets):
        pred_set = set(pred)
        target_set = set(target)
        
        if pred_set == target_set:
            exact_match += 1
            partial_match += 1
        elif pred_set & target_set:
            partial_match += 1
    
    return {
        "exact_match": exact_match / total,
        "partial_match": partial_match / total,
        "total": total
    }

def main():
    print("🎯 Qwen3-0.6B微调效果快速验证")
    print("=" * 50)
    
    # 设备检查
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✅ 使用MPS加速")
    else:
        device = torch.device("cpu")
        print("⚠️ 使用CPU")
    
    # 加载测试数据
    test_samples = load_test_samples(20)
    print(f"加载测试样本: {len(test_samples)}个")
    
    results = {}
    
    try:
        # 1. 测试微调后模型
        print("\\n🔍 测试微调后模型...")
        finetuned_tokenizer = AutoTokenizer.from_pretrained(
            "models/qwen3_fixed_classifier", trust_remote_code=True
        )
        if finetuned_tokenizer.pad_token is None:
            finetuned_tokenizer.pad_token = finetuned_tokenizer.eos_token
        
        base_model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen3-0.6B", torch_dtype=torch.float32, trust_remote_code=True
        )
        finetuned_model = PeftModel.from_pretrained(base_model, "models/qwen3_fixed_classifier")
        finetuned_model = finetuned_model.to(device)
        
        finetuned_predictions = []
        start_time = time.time()
        
        for sample in test_samples:
            pred = quick_predict(finetuned_model, finetuned_tokenizer, sample["prompt"], device)
            finetuned_predictions.append(pred)
        
        finetuned_time = time.time() - start_time
        
        # 计算微调后准确率
        targets = [sample["expected"] for sample in test_samples]
        finetuned_accuracy = calculate_accuracy(finetuned_predictions, targets)
        
        results["finetuned"] = {
            "exact_match_accuracy": finetuned_accuracy["exact_match"],
            "partial_match_accuracy": finetuned_accuracy["partial_match"],
            "evaluation_time": finetuned_time,
            "samples": finetuned_accuracy["total"]
        }
        
        print(f"微调后模型:")
        print(f"  精确匹配: {finetuned_accuracy['exact_match']:.3f}")
        print(f"  部分匹配: {finetuned_accuracy['partial_match']:.3f}")
        print(f"  评估耗时: {finetuned_time:.2f}秒")
        
        # 显示部分预测结果
        print("\\n📋 预测示例:")
        for i in range(min(5, len(test_samples))):
            sample = test_samples[i]
            pred = finetuned_predictions[i]
            target = sample["expected"]
            match = "✅" if set(pred) == set(target) else ("🟡" if set(pred) & set(target) else "❌")
            
            print(f"  {match} 输入: {sample['prompt'][:30]}...")
            print(f"     期望: {target}")
            print(f"     预测: {pred}")
        
        del finetuned_model, base_model
        if device.type == "mps":
            torch.mps.empty_cache()
        
    except Exception as e:
        print(f"❌ 微调模型测试失败: {e}")
        results["finetuned"] = {"error": str(e)}
    
    # 2. 对比之前的基线结果
    print("\\n📊 与基线对比:")
    baseline_path = "reports/pretrained_qwen3_test.json"
    try:
        with open(baseline_path, 'r', encoding='utf-8') as f:
            baseline_data = json.load(f)
        
        pretrained_accuracy = baseline_data["results"]["pretrained_qwen3"]["overall_accuracy"]
        mock_accuracy = baseline_data["results"]["mock"]["overall_accuracy"]
        
        print(f"预训练模型准确率: {pretrained_accuracy:.3f}")
        print(f"Mock分类器准确率: {mock_accuracy:.3f}")
        
        if "finetuned" in results and "error" not in results["finetuned"]:
            finetuned_acc = results["finetuned"]["exact_match_accuracy"]
            improvement = finetuned_acc - pretrained_accuracy
            relative_improvement = (improvement / pretrained_accuracy * 100) if pretrained_accuracy > 0 else 0
            
            print(f"微调后模型准确率: {finetuned_acc:.3f}")
            print(f"\\n🎯 微调效果:")
            print(f"  绝对提升: +{improvement:.3f}")
            print(f"  相对提升: +{relative_improvement:.1f}%")
            
            gap_before = mock_accuracy - pretrained_accuracy
            gap_after = mock_accuracy - finetuned_acc
            print(f"  与Mock差距: {gap_before:.3f} → {gap_after:.3f}")
            
            if finetuned_acc > pretrained_accuracy:
                print("\\n🎉 微调成功！模型性能显著提升")
            else:
                print("\\n⚠️ 微调效果不明显，需要进一步分析")
        
        results["baseline_comparison"] = {
            "pretrained_accuracy": pretrained_accuracy,
            "mock_accuracy": mock_accuracy,
            "improvement": improvement if "finetuned" in results else None
        }
        
    except Exception as e:
        print(f"⚠️ 无法加载基线结果: {e}")
    
    # 保存结果
    results["test_info"] = {
        "test_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "test_samples": len(test_samples),
        "device": str(device)
    }
    
    with open("reports/quick_finetuned_evaluation.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\\n✅ 快速评估完成，结果保存到: reports/quick_finetuned_evaluation.json")
    return results

if __name__ == "__main__":
    main()