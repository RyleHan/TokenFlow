#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
纯模型对比：微调前后Qwen3-0.6B模型性能对比
不使用混合策略，只测试神经网络模型本身
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import json
import time
from typing import Dict, List, Any

class PureModelComparator:
    """纯神经网络模型对比器"""
    
    def __init__(self):
        print("🔧 初始化纯模型对比器...")
        
        # 设置设备
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("✅ 使用MPS加速")
        else:
            self.device = torch.device("cpu")
            print("⚠️ 使用CPU")
        
        self.intent_categories = [
            "order_management", "user_auth", "payment", 
            "inventory", "notification", "none"
        ]
        
        # 测试样本
        self.test_cases = [
            {
                "prompt": "我想查看我的订单状态",
                "expected_intents": ["order_management"],
                "description": "单意图-订单查询"
            },
            {
                "prompt": "登录后查看订单并处理支付",
                "expected_intents": ["user_auth", "order_management", "payment"],
                "description": "多意图-用户认证+订单+支付"
            },
            {
                "prompt": "支付成功后发送通知",
                "expected_intents": ["payment", "notification"],
                "description": "多意图-支付+通知"
            },
            {
                "prompt": "库存不足时需要及时提醒",
                "expected_intents": ["inventory", "notification"],
                "description": "多意图-库存+通知"
            },
            {
                "prompt": "用户注册、订单处理和支付确认",
                "expected_intents": ["user_auth", "order_management", "payment"],
                "description": "复杂多意图"
            },
            {
                "prompt": "今天天气怎么样",
                "expected_intents": ["none"],
                "description": "无关查询"
            },
            {
                "prompt": "修改身份要是有问题就付款充值",
                "expected_intents": ["user_auth", "payment"],
                "description": "复杂表达-身份+支付"
            },
            {
                "prompt": "因为账单问题需要支付因为标记短信",
                "expected_intents": ["payment", "notification"],
                "description": "复杂表达-支付+通知"
            },
            {
                "prompt": "怎么绑定用户名",
                "expected_intents": ["user_auth"],
                "description": "单意图-用户认证"
            },
            {
                "prompt": "我要结算订单",
                "expected_intents": ["order_management"],
                "description": "单意图-订单结算"
            }
        ]
    
    def load_pretrained_model(self):
        """加载预训练模型"""
        print("📥 加载预训练Qwen3-0.6B模型...")
        
        tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen3-0.6B",
            trust_remote_code=True,
            padding_side="right"
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen3-0.6B",
            torch_dtype=torch.float32,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        
        model = model.to(self.device)
        model.eval()
        
        print("✅ 预训练模型加载完成")
        return tokenizer, model
    
    def load_finetuned_model(self):
        """加载微调后模型"""
        print("📥 加载微调后Qwen3-0.6B模型...")
        
        # 加载分词器
        tokenizer = AutoTokenizer.from_pretrained(
            "models/qwen3_fixed_classifier",
            trust_remote_code=True,
            padding_side="right"
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # 加载基础模型
        base_model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen3-0.6B",
            torch_dtype=torch.float32,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        
        # 加载微调适配器
        model = PeftModel.from_pretrained(base_model, "models/qwen3_fixed_classifier")
        model = model.to(self.device)
        model.eval()
        
        print("✅ 微调模型加载完成")
        return tokenizer, model
    
    def predict_with_model(self, model, tokenizer, prompt: str, is_finetuned: bool = False) -> List[str]:
        """使用模型进行预测"""
        
        if is_finetuned:
            # 微调模型使用训练时的格式
            input_text = f"用户: {prompt}\\n助手: "
        else:
            # 预训练模型使用通用格式
            input_text = f"请识别以下文本的意图类别（order_management, user_auth, payment, inventory, notification, none）:\\n{prompt}\\n意图："
        
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
                max_new_tokens=30,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if is_finetuned:
            # 微调模型解析
            if "助手: " in response:
                predicted = response.split("助手: ")[-1].strip()
            else:
                predicted = response.strip()
        else:
            # 预训练模型解析
            if "意图：" in response:
                predicted = response.split("意图：")[-1].strip()
            elif "意图:" in response:
                predicted = response.split("意图:")[-1].strip()
            else:
                predicted = response.strip()
        
        # 解析意图
        predicted = predicted.lower().replace("\\n", " ").replace("\\t", " ")
        found_intents = []
        
        for intent in self.intent_categories:
            if intent != "none":
                intent_clean = intent.replace("_", "")
                if intent in predicted or intent_clean in predicted.replace("_", ""):
                    found_intents.append(intent)
        
        # 后处理：如果找到业务意图，就不返回none
        if found_intents:
            return found_intents
        
        # 检查是否明确为none
        if "none" in predicted or not found_intents:
            return ["none"]
        
        return found_intents if found_intents else ["none"]
    
    def calculate_metrics(self, predicted: List[str], expected: List[str]) -> Dict[str, float]:
        """计算详细指标"""
        predicted_set = set(predicted)
        expected_set = set(expected)
        
        # 精确匹配
        exact_match = 1.0 if predicted_set == expected_set else 0.0
        
        # Jaccard相似度
        intersection = len(predicted_set & expected_set)
        union = len(predicted_set | expected_set)
        jaccard = intersection / union if union > 0 else 0.0
        
        # Precision, Recall, F1
        if len(predicted_set) == 0:
            precision = 0.0
        else:
            precision = intersection / len(predicted_set)
        
        if len(expected_set) == 0:
            recall = 0.0
        else:
            recall = intersection / len(expected_set)
        
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        
        return {
            "exact_match": exact_match,
            "jaccard": jaccard,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
    
    def run_comparison(self) -> Dict[str, Any]:
        """运行对比测试"""
        print("🚀 开始纯模型对比测试...")
        
        results = {
            "test_info": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_test_cases": len(self.test_cases),
                "model_comparison": "预训练 vs 微调 Qwen3-0.6B"
            },
            "individual_results": [],
            "summary_stats": {}
        }
        
        # 加载模型
        pretrained_tokenizer, pretrained_model = self.load_pretrained_model()
        finetuned_tokenizer, finetuned_model = self.load_finetuned_model()
        
        # 累计统计
        pretrained_stats = {"exact_match": [], "jaccard": [], "f1": [], "precision": [], "recall": []}
        finetuned_stats = {"exact_match": [], "jaccard": [], "f1": [], "precision": [], "recall": []}
        
        for i, test_case in enumerate(self.test_cases):
            print(f"\\n📋 测试案例 {i+1}/{len(self.test_cases)}: {test_case['description']}")
            print(f"   输入: {test_case['prompt']}")
            print(f"   期望: {test_case['expected_intents']}")
            
            test_result = {
                "test_case": test_case,
                "results": {}
            }
            
            # 1. 预训练模型测试
            try:
                pretrained_prediction = self.predict_with_model(
                    pretrained_model, pretrained_tokenizer, test_case['prompt'], is_finetuned=False
                )
                pretrained_metrics = self.calculate_metrics(pretrained_prediction, test_case['expected_intents'])
                
                test_result["results"]["pretrained"] = {
                    "predicted_intents": pretrained_prediction,
                    "metrics": pretrained_metrics
                }
                
                # 累计统计
                for key in pretrained_stats:
                    pretrained_stats[key].append(pretrained_metrics[key])
                
                print(f"   预训练:   {pretrained_prediction} (F1: {pretrained_metrics['f1']:.3f})")
                
            except Exception as e:
                print(f"   ❌ 预训练模型测试失败: {e}")
                test_result["results"]["pretrained"] = {"error": str(e)}
            
            # 2. 微调模型测试  
            try:
                finetuned_prediction = self.predict_with_model(
                    finetuned_model, finetuned_tokenizer, test_case['prompt'], is_finetuned=True
                )
                finetuned_metrics = self.calculate_metrics(finetuned_prediction, test_case['expected_intents'])
                
                test_result["results"]["finetuned"] = {
                    "predicted_intents": finetuned_prediction,
                    "metrics": finetuned_metrics
                }
                
                # 累计统计
                for key in finetuned_stats:
                    finetuned_stats[key].append(finetuned_metrics[key])
                
                print(f"   微调后:   {finetuned_prediction} (F1: {finetuned_metrics['f1']:.3f})")
                
            except Exception as e:
                print(f"   ❌ 微调模型测试失败: {e}")
                test_result["results"]["finetuned"] = {"error": str(e)}
            
            results["individual_results"].append(test_result)
        
        # 计算总体统计
        def calc_avg(values):
            return sum(values) / len(values) if values else 0.0
        
        results["summary_stats"] = {
            "pretrained": {
                "avg_exact_match": calc_avg(pretrained_stats["exact_match"]),
                "avg_jaccard": calc_avg(pretrained_stats["jaccard"]),
                "avg_f1": calc_avg(pretrained_stats["f1"]),
                "avg_precision": calc_avg(pretrained_stats["precision"]),
                "avg_recall": calc_avg(pretrained_stats["recall"]),
            },
            "finetuned": {
                "avg_exact_match": calc_avg(finetuned_stats["exact_match"]),
                "avg_jaccard": calc_avg(finetuned_stats["jaccard"]),
                "avg_f1": calc_avg(finetuned_stats["f1"]),
                "avg_precision": calc_avg(finetuned_stats["precision"]),
                "avg_recall": calc_avg(finetuned_stats["recall"]),
            }
        }
        
        # 计算改进
        if finetuned_stats["f1"] and pretrained_stats["f1"]:
            pretrained_f1 = results["summary_stats"]["pretrained"]["avg_f1"]
            finetuned_f1 = results["summary_stats"]["finetuned"]["avg_f1"]
            
            results["summary_stats"]["improvement"] = {
                "f1_absolute": finetuned_f1 - pretrained_f1,
                "f1_relative": ((finetuned_f1 - pretrained_f1) / pretrained_f1 * 100) if pretrained_f1 > 0 else 0,
                "exact_match_improvement": results["summary_stats"]["finetuned"]["avg_exact_match"] - results["summary_stats"]["pretrained"]["avg_exact_match"]
            }
        
        return results
    
    def generate_report(self, results: Dict[str, Any]):
        """生成对比报告"""
        print("\\n" + "="*80)
        print("🎯 纯神经网络模型对比报告 (微调前 vs 微调后)")
        print("="*80)
        
        stats = results["summary_stats"]
        
        if "pretrained" in stats and "finetuned" in stats:
            print("\\n📊 详细性能对比:")
            print("-"*80)
            print(f"{'指标':<15} {'预训练模型':<15} {'微调后模型':<15} {'改进':<15}")
            print("-"*80)
            
            metrics = ["avg_exact_match", "avg_jaccard", "avg_f1", "avg_precision", "avg_recall"]
            metric_names = ["精确匹配", "Jaccard", "F1分数", "精确率", "召回率"]
            
            for metric, name in zip(metrics, metric_names):
                pretrained_val = stats["pretrained"][metric]
                finetuned_val = stats["finetuned"][metric]
                improvement = finetuned_val - pretrained_val
                
                print(f"{name:<15} {pretrained_val:<15.3f} {finetuned_val:<15.3f} {improvement:+.3f}")
            
            if "improvement" in stats:
                imp = stats["improvement"]
                print("\\n💡 关键改进分析:")
                print(f"  🎯 F1分数提升: {imp['f1_absolute']:+.3f} ({imp['f1_relative']:+.1f}%)")
                print(f"  🎯 精确匹配提升: {imp['exact_match_improvement']:+.3f}")
                
                print("\\n🚀 微调效果评估:")
                if imp['f1_relative'] > 100:
                    print("  🌟 微调效果显著！F1分数提升超过100%")
                elif imp['f1_relative'] > 50:
                    print("  ✨ 微调效果良好，F1分数有大幅提升")
                elif imp['f1_relative'] > 20:
                    print("  📈 微调有明显效果，性能有所改善")
                elif imp['f1_relative'] > 0:
                    print("  📊 微调有一定效果，略有改进")
                else:
                    print("  ⚠️ 微调效果不明显，需要进一步优化")
        
        # 分析具体案例
        print("\\n🔍 典型案例分析:")
        for result in results["individual_results"]:
            if "pretrained" in result["results"] and "finetuned" in result["results"]:
                if "error" not in result["results"]["pretrained"] and "error" not in result["results"]["finetuned"]:
                    case = result["test_case"]
                    pre_f1 = result["results"]["pretrained"]["metrics"]["f1"]
                    fine_f1 = result["results"]["finetuned"]["metrics"]["f1"]
                    
                    if fine_f1 - pre_f1 > 0.5:  # 显著改进的案例
                        print(f"  ✨ {case['description']}: F1从{pre_f1:.3f}提升到{fine_f1:.3f}")
                        print(f"     输入: {case['prompt']}")
                        print(f"     预训练: {result['results']['pretrained']['predicted_intents']}")
                        print(f"     微调后: {result['results']['finetuned']['predicted_intents']}")
                        print(f"     期望: {case['expected_intents']}")

def main():
    """主函数"""
    comparator = PureModelComparator()
    
    # 运行对比
    results = comparator.run_comparison()
    
    # 生成报告
    comparator.generate_report(results)
    
    # 保存结果
    with open("reports/pure_model_comparison.json", "w", encoding="utf-8") as f:
        # 处理可能的numpy类型
        def convert_types(obj):
            if hasattr(obj, 'item'):
                return obj.item()
            return obj
        
        import numpy as np
        results_clean = json.loads(json.dumps(results, default=str))
        json.dump(results_clean, f, ensure_ascii=False, indent=2)
    
    print(f"\\n✅ 详细对比结果已保存到: reports/pure_model_comparison.json")

if __name__ == "__main__":
    main()