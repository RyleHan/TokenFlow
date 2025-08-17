#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
评估微调后的Qwen3-0.6B模型性能
对比微调前后的效果
"""

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os
import time
from typing import List, Dict, Tuple
import random
from collections import defaultdict

class Qwen3Evaluator:
    """Qwen3模型评估器"""
    
    def __init__(self):
        self.base_model_name = "Qwen/Qwen3-0.6B"
        self.finetuned_model_path = "models/qwen3_fixed_classifier"
        self.intent_categories = [
            "order_management", "user_auth", "payment", 
            "inventory", "notification", "none"
        ]
        
        # 检查设备
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("✅ 使用Apple Metal Performance Shaders (MPS)")
        else:
            self.device = torch.device("cpu")
            print("⚠️ 使用CPU")
    
    def load_test_data(self, test_size: int = 50) -> List[Dict]:
        """加载测试数据"""
        print(f"正在加载测试数据 (前{test_size}个样本)...")
        
        test_data = []
        data_path = "data/enhanced_multi_intent_training_data.jsonl"
        
        with open(data_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= test_size:
                    break
                    
                sample = json.loads(line.strip())
                test_data.append({
                    "prompt": sample["prompt"],
                    "expected": sample["output"]
                })
        
        print(f"测试数据加载完成: {len(test_data)} 个样本")
        return test_data
    
    def load_pretrained_model(self):
        """加载预训练模型"""
        print("正在加载预训练模型...")
        
        tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name,
            trust_remote_code=True,
            padding_side="right"
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.float32,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        
        model = model.to(self.device)
        print("预训练模型加载完成")
        
        return tokenizer, model
    
    def load_finetuned_model(self):
        """加载微调后的模型"""
        print("正在加载微调后的模型...")
        
        # 加载分词器
        tokenizer = AutoTokenizer.from_pretrained(
            self.finetuned_model_path,
            trust_remote_code=True,
            padding_side="right"
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # 加载基础模型
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.float32,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        
        # 加载微调适配器
        model = PeftModel.from_pretrained(base_model, self.finetuned_model_path)
        model = model.to(self.device)
        
        print("微调模型加载完成")
        return tokenizer, model
    
    def predict_intent(self, model, tokenizer, prompt: str) -> List[str]:
        """预测意图"""
        input_text = f"用户: {prompt}\\n助手: "
        
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=256
        )
        
        if self.device.type != "cpu":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=25,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        predicted = response.split("助手: ")[-1].strip()
        
        # 解析预测的意图
        predicted_intents = []
        for intent in self.intent_categories:
            if intent in predicted.lower():
                predicted_intents.append(intent)
        
        # 如果没有匹配到任何意图，返回none
        if not predicted_intents:
            predicted_intents = ["none"]
        
        return predicted_intents
    
    def calculate_accuracy(self, predictions: List[List[str]], targets: List[List[str]]) -> Dict:
        """计算准确率"""
        total = len(predictions)
        exact_match = 0
        partial_match = 0
        
        for pred, target in zip(predictions, targets):
            pred_set = set(pred)
            target_set = set(target)
            
            if pred_set == target_set:
                exact_match += 1
                partial_match += 1
            elif pred_set & target_set:  # 有交集
                partial_match += 1
        
        return {
            "exact_match_accuracy": exact_match / total,
            "partial_match_accuracy": partial_match / total,
            "total_samples": total
        }
    
    def evaluate_model(self, model, tokenizer, test_data: List[Dict], model_name: str) -> Dict:
        """评估模型"""
        print(f"\\n正在评估 {model_name}...")
        
        predictions = []
        targets = []
        
        start_time = time.time()
        
        for i, sample in enumerate(test_data):
            if i % 20 == 0:
                print(f"  进度: {i}/{len(test_data)}")
            
            pred = self.predict_intent(model, tokenizer, sample["prompt"])
            predictions.append(pred)
            targets.append(sample["expected"])
        
        eval_time = time.time() - start_time
        
        # 计算准确率
        metrics = self.calculate_accuracy(predictions, targets)
        metrics["evaluation_time"] = eval_time
        metrics["model_name"] = model_name
        
        print(f"{model_name} 评估完成:")
        print(f"  精确匹配准确率: {metrics['exact_match_accuracy']:.3f}")
        print(f"  部分匹配准确率: {metrics['partial_match_accuracy']:.3f}")
        print(f"  评估时间: {eval_time:.2f}秒")
        
        return metrics, predictions, targets
    
    def analyze_predictions(self, predictions: List[List[str]], targets: List[List[str]], 
                          test_data: List[Dict], model_name: str):
        """分析预测结果"""
        print(f"\\n📊 {model_name} 详细分析:")
        
        # 按意图类别分析
        intent_stats = defaultdict(lambda: {"correct": 0, "total": 0})
        
        for pred, target, sample in zip(predictions, targets, test_data):
            for intent in target:
                intent_stats[intent]["total"] += 1
                if intent in pred:
                    intent_stats[intent]["correct"] += 1
        
        print("\\n各意图类别准确率:")
        for intent, stats in intent_stats.items():
            if stats["total"] > 0:
                accuracy = stats["correct"] / stats["total"]
                print(f"  {intent}: {accuracy:.3f} ({stats['correct']}/{stats['total']})")
        
        # 显示错误案例
        print(f"\\n❌ {model_name} 错误预测示例:")
        error_count = 0
        for pred, target, sample in zip(predictions, targets, test_data):
            if set(pred) != set(target) and error_count < 5:
                print(f"  输入: {sample['prompt']}")
                print(f"  期望: {target}")
                print(f"  预测: {pred}")
                print(f"  ---")
                error_count += 1
    
    def run_comparison(self):
        """运行对比评估"""
        print("🎯 Qwen3-0.6B 微调前后性能对比评估")
        print("=" * 60)
        
        # 加载测试数据
        test_data = self.load_test_data(50)
        
        results = {}
        
        try:
            # 评估预训练模型
            print("\\n🔍 评估预训练模型...")
            pretrained_tokenizer, pretrained_model = self.load_pretrained_model()
            pretrained_metrics, pretrained_preds, targets = self.evaluate_model(
                pretrained_model, pretrained_tokenizer, test_data, "预训练Qwen3-0.6B"
            )
            results["pretrained"] = pretrained_metrics
            
            # 分析预训练模型结果
            self.analyze_predictions(pretrained_preds, targets, test_data, "预训练模型")
            
            # 清理内存
            del pretrained_model, pretrained_tokenizer
            if self.device.type == "mps":
                torch.mps.empty_cache()
            
        except Exception as e:
            print(f"❌ 预训练模型评估失败: {e}")
            results["pretrained"] = {"error": str(e)}
        
        try:
            # 评估微调模型
            print("\\n🔍 评估微调后模型...")
            finetuned_tokenizer, finetuned_model = self.load_finetuned_model()
            finetuned_metrics, finetuned_preds, targets = self.evaluate_model(
                finetuned_model, finetuned_tokenizer, test_data, "微调后Qwen3-0.6B"
            )
            results["finetuned"] = finetuned_metrics
            
            # 分析微调模型结果
            self.analyze_predictions(finetuned_preds, targets, test_data, "微调模型")
            
        except Exception as e:
            print(f"❌ 微调模型评估失败: {e}")
            results["finetuned"] = {"error": str(e)}
        
        # 对比分析
        self.generate_comparison_report(results)
        
        # 保存结果
        with open("reports/finetuning_comparison_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print("\\n✅ 评估完成，结果已保存到: reports/finetuning_comparison_results.json")
        return results
    
    def generate_comparison_report(self, results: Dict):
        """生成对比报告"""
        print("\\n📈 微调效果对比报告")
        print("=" * 60)
        
        if "pretrained" in results and "finetuned" in results:
            if "error" not in results["pretrained"] and "error" not in results["finetuned"]:
                pretrained = results["pretrained"]
                finetuned = results["finetuned"]
                
                print("\\n📊 核心指标对比:")
                print(f"{'指标':<20} {'预训练':<15} {'微调后':<15} {'提升':<15}")
                print("-" * 70)
                
                # 精确匹配准确率
                pre_exact = pretrained["exact_match_accuracy"]
                fine_exact = finetuned["exact_match_accuracy"]
                exact_improvement = fine_exact - pre_exact
                exact_relative = (exact_improvement / pre_exact * 100) if pre_exact > 0 else 0
                
                print(f"{'精确匹配准确率':<20} {pre_exact:<15.3f} {fine_exact:<15.3f} {exact_improvement:+.3f} ({exact_relative:+.1f}%)")
                
                # 部分匹配准确率
                pre_partial = pretrained["partial_match_accuracy"]
                fine_partial = finetuned["partial_match_accuracy"]
                partial_improvement = fine_partial - pre_partial
                partial_relative = (partial_improvement / pre_partial * 100) if pre_partial > 0 else 0
                
                print(f"{'部分匹配准确率':<20} {pre_partial:<15.3f} {fine_partial:<15.3f} {partial_improvement:+.3f} ({partial_relative:+.1f}%)")
                
                # 评估时间
                pre_time = pretrained["evaluation_time"]
                fine_time = finetuned["evaluation_time"]
                print(f"{'评估时间(秒)':<20} {pre_time:<15.2f} {fine_time:<15.2f} {fine_time-pre_time:+.2f}")
                
                print("\\n🎯 微调效果总结:")
                if exact_improvement > 0:
                    print(f"  ✅ 精确匹配准确率提升了 {exact_improvement:.3f} ({exact_relative:+.1f}%)")
                else:
                    print(f"  ❌ 精确匹配准确率下降了 {abs(exact_improvement):.3f} ({abs(exact_relative):.1f}%)")
                
                if partial_improvement > 0:
                    print(f"  ✅ 部分匹配准确率提升了 {partial_improvement:.3f} ({partial_relative:+.1f}%)")
                else:
                    print(f"  ❌ 部分匹配准确率下降了 {abs(partial_improvement):.3f} ({abs(partial_relative):.1f}%)")
                
                # 整体评价
                if exact_improvement > 0 and partial_improvement > 0:
                    print("\\n🎉 微调成功！模型在两个核心指标上都有提升")
                elif exact_improvement > 0 or partial_improvement > 0:
                    print("\\n🎈 微调部分成功，在某些指标上有提升")
                else:
                    print("\\n⚠️ 微调效果不明显，可能需要进一步优化")

def main():
    """主函数"""
    evaluator = Qwen3Evaluator()
    results = evaluator.run_comparison()
    return results

if __name__ == "__main__":
    main()