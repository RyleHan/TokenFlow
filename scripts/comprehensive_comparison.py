#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
全面对比报告：微调前 vs 微调后(纯模型) vs 微调后混合策略
最全面的性能评估和分析
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from src.classifiers.hybrid_intent_classifier import HybridIntentClassifier
from src.classifiers.intent_classifier import MockIntentClassifier
import json
import time
from typing import Dict, List, Any
import numpy as np

class ComprehensiveComparator:
    """全面对比分析器"""
    
    def __init__(self):
        print("🔧 初始化全面对比分析器...")
        
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
        
        # 扩展测试样本集，包含各种复杂度
        self.test_cases = [
            # 简单单意图
            {
                "prompt": "我想查看我的订单状态",
                "expected_intents": ["order_management"],
                "description": "简单单意图-订单查询",
                "complexity": "simple"
            },
            {
                "prompt": "怎么绑定用户名",
                "expected_intents": ["user_auth"],
                "description": "简单单意图-用户认证",
                "complexity": "simple"
            },
            {
                "prompt": "我要结算订单",
                "expected_intents": ["order_management"],
                "description": "简单单意图-订单结算",
                "complexity": "simple"
            },
            
            # 明确多意图
            {
                "prompt": "登录后查看订单并处理支付",
                "expected_intents": ["user_auth", "order_management", "payment"],
                "description": "明确多意图-登录+订单+支付",
                "complexity": "medium"
            },
            {
                "prompt": "支付成功后发送通知",
                "expected_intents": ["payment", "notification"],
                "description": "明确多意图-支付+通知",
                "complexity": "medium"
            },
            {
                "prompt": "库存不足时需要及时提醒",
                "expected_intents": ["inventory", "notification"],
                "description": "明确多意图-库存+通知",
                "complexity": "medium"
            },
            
            # 复杂隐含多意图
            {
                "prompt": "修改身份要是有问题就付款充值",
                "expected_intents": ["user_auth", "payment"],
                "description": "复杂多意图-身份修改+付款",
                "complexity": "complex"
            },
            {
                "prompt": "因为账单问题需要支付因为标记短信",
                "expected_intents": ["payment", "notification"],
                "description": "复杂多意图-账单支付+短信通知",
                "complexity": "complex"
            },
            {
                "prompt": "用户注册、订单处理和支付确认",
                "expected_intents": ["user_auth", "order_management", "payment"],
                "description": "复杂多意图-注册+订单+支付",
                "complexity": "complex"
            },
            
            # 边界情况
            {
                "prompt": "今天天气怎么样",
                "expected_intents": ["none"],
                "description": "无关查询-天气",
                "complexity": "simple"
            },
            {
                "prompt": "推荐一部好看的电影",
                "expected_intents": ["none"],
                "description": "无关查询-电影推荐",
                "complexity": "simple"
            },
            
            # 歧义和困难案例
            {
                "prompt": "账户登录订单支付库存通知全部都要处理",
                "expected_intents": ["user_auth", "order_management", "payment", "inventory", "notification"],
                "description": "极复杂多意图-全业务覆盖",
                "complexity": "very_complex"
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
        
        tokenizer = AutoTokenizer.from_pretrained(
            "models/qwen3_fixed_classifier",
            trust_remote_code=True,
            padding_side="right"
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        base_model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen3-0.6B",
            torch_dtype=torch.float32,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        
        model = PeftModel.from_pretrained(base_model, "models/qwen3_fixed_classifier")
        model = model.to(self.device)
        model.eval()
        
        print("✅ 微调模型加载完成")
        return tokenizer, model
    
    def load_hybrid_classifier(self):
        """加载混合分类器"""
        print("📥 加载混合分类器...")
        
        try:
            hybrid_classifier = HybridIntentClassifier(use_finetuned=True)
            print("✅ 混合分类器加载完成")
            return hybrid_classifier
        except Exception as e:
            print(f"❌ 混合分类器加载失败: {e}")
            return None
    
    def predict_with_pretrained(self, model, tokenizer, prompt: str) -> List[str]:
        """预训练模型预测"""
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
        
        if "意图：" in response:
            predicted = response.split("意图：")[-1].strip()
        elif "意图:" in response:
            predicted = response.split("意图:")[-1].strip()
        else:
            predicted = response.strip()
        
        return self._parse_intents(predicted)
    
    def predict_with_finetuned(self, model, tokenizer, prompt: str) -> List[str]:
        """微调模型预测"""
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
        
        if "助手: " in response:
            predicted = response.split("助手: ")[-1].strip()
        else:
            predicted = response.strip()
        
        return self._parse_intents(predicted)
    
    def predict_with_hybrid(self, hybrid_classifier, prompt: str) -> List[str]:
        """混合分类器预测"""
        if hybrid_classifier is None:
            return ["none"]
        
        try:
            result = hybrid_classifier.classify(prompt)
            return result.get("intents", [result.get("intent", "none")])
        except Exception as e:
            print(f"⚠️ 混合分类器预测失败: {e}")
            return ["none"]
    
    def _parse_intents(self, predicted: str) -> List[str]:
        """解析意图文本"""
        predicted = predicted.lower().replace("\\n", " ").replace("\\t", " ")
        found_intents = []
        
        for intent in self.intent_categories:
            if intent != "none":
                intent_clean = intent.replace("_", "")
                if intent in predicted or intent_clean in predicted.replace("_", ""):
                    found_intents.append(intent)
        
        if found_intents:
            return found_intents
        
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
    
    def estimate_token_usage(self, intents: List[str], prompt: str) -> Dict[str, int]:
        """估算Token使用量"""
        total_docs = 20
        avg_doc_tokens = 400
        
        if "none" in intents:
            retrieved_docs = 2
        else:
            base_docs = 3
            intent_bonus = len([i for i in intents if i != "none"])
            retrieved_docs = min(base_docs + intent_bonus, 8)
        
        without_filtering = total_docs * avg_doc_tokens
        with_filtering = retrieved_docs * avg_doc_tokens
        
        return {
            "without_filtering": without_filtering,
            "with_filtering": with_filtering,
            "tokens_saved": without_filtering - with_filtering,
            "savings_percentage": ((without_filtering - with_filtering) / without_filtering * 100) if without_filtering > 0 else 0
        }
    
    def run_comprehensive_comparison(self) -> Dict[str, Any]:
        """运行全面对比"""
        print("🚀 开始全面对比分析...")
        
        results = {
            "test_info": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_test_cases": len(self.test_cases),
                "comparison_types": ["预训练Qwen3", "微调纯模型", "微调混合策略"],
                "complexity_levels": ["simple", "medium", "complex", "very_complex"]
            },
            "individual_results": [],
            "summary_stats": {},
            "complexity_analysis": {},
            "token_usage_analysis": {}
        }
        
        # 加载所有模型
        pretrained_tokenizer, pretrained_model = self.load_pretrained_model()
        finetuned_tokenizer, finetuned_model = self.load_finetuned_model()
        hybrid_classifier = self.load_hybrid_classifier()
        
        # 累计统计
        stats = {
            "pretrained": {"exact_match": [], "jaccard": [], "f1": [], "precision": [], "recall": []},
            "finetuned": {"exact_match": [], "jaccard": [], "f1": [], "precision": [], "recall": []},
            "hybrid": {"exact_match": [], "jaccard": [], "f1": [], "precision": [], "recall": []}
        }
        
        # 复杂度统计
        complexity_stats = {
            "simple": {"pretrained": [], "finetuned": [], "hybrid": []},
            "medium": {"pretrained": [], "finetuned": [], "hybrid": []},
            "complex": {"pretrained": [], "finetuned": [], "hybrid": []},
            "very_complex": {"pretrained": [], "finetuned": [], "hybrid": []}
        }
        
        # Token使用统计
        token_stats = {"pretrained": [], "finetuned": [], "hybrid": []}
        
        for i, test_case in enumerate(self.test_cases):
            print(f"\\n📋 测试案例 {i+1}/{len(self.test_cases)}: {test_case['description']}")
            print(f"   输入: {test_case['prompt']}")
            print(f"   期望: {test_case['expected_intents']}")
            print(f"   复杂度: {test_case['complexity']}")
            
            test_result = {
                "test_case": test_case,
                "results": {}
            }
            
            # 1. 预训练模型
            try:
                pretrained_prediction = self.predict_with_pretrained(
                    pretrained_model, pretrained_tokenizer, test_case['prompt']
                )
                pretrained_metrics = self.calculate_metrics(pretrained_prediction, test_case['expected_intents'])
                pretrained_tokens = self.estimate_token_usage(pretrained_prediction, test_case['prompt'])
                
                test_result["results"]["pretrained"] = {
                    "predicted_intents": pretrained_prediction,
                    "metrics": pretrained_metrics,
                    "token_usage": pretrained_tokens
                }
                
                for key in stats["pretrained"]:
                    stats["pretrained"][key].append(pretrained_metrics[key])
                
                complexity_stats[test_case['complexity']]["pretrained"].append(pretrained_metrics["f1"])
                token_stats["pretrained"].append(pretrained_tokens["tokens_saved"])
                
                print(f"   预训练:   {pretrained_prediction} (F1: {pretrained_metrics['f1']:.3f})")
                
            except Exception as e:
                print(f"   ❌ 预训练模型测试失败: {e}")
                test_result["results"]["pretrained"] = {"error": str(e)}
            
            # 2. 微调纯模型
            try:
                finetuned_prediction = self.predict_with_finetuned(
                    finetuned_model, finetuned_tokenizer, test_case['prompt']
                )
                finetuned_metrics = self.calculate_metrics(finetuned_prediction, test_case['expected_intents'])
                finetuned_tokens = self.estimate_token_usage(finetuned_prediction, test_case['prompt'])
                
                test_result["results"]["finetuned"] = {
                    "predicted_intents": finetuned_prediction,
                    "metrics": finetuned_metrics,
                    "token_usage": finetuned_tokens
                }
                
                for key in stats["finetuned"]:
                    stats["finetuned"][key].append(finetuned_metrics[key])
                
                complexity_stats[test_case['complexity']]["finetuned"].append(finetuned_metrics["f1"])
                token_stats["finetuned"].append(finetuned_tokens["tokens_saved"])
                
                print(f"   微调纯模型: {finetuned_prediction} (F1: {finetuned_metrics['f1']:.3f})")
                
            except Exception as e:
                print(f"   ❌ 微调模型测试失败: {e}")
                test_result["results"]["finetuned"] = {"error": str(e)}
            
            # 3. 混合策略
            try:
                hybrid_prediction = self.predict_with_hybrid(hybrid_classifier, test_case['prompt'])
                hybrid_metrics = self.calculate_metrics(hybrid_prediction, test_case['expected_intents'])
                hybrid_tokens = self.estimate_token_usage(hybrid_prediction, test_case['prompt'])
                
                test_result["results"]["hybrid"] = {
                    "predicted_intents": hybrid_prediction,
                    "metrics": hybrid_metrics,
                    "token_usage": hybrid_tokens
                }
                
                for key in stats["hybrid"]:
                    stats["hybrid"][key].append(hybrid_metrics[key])
                
                complexity_stats[test_case['complexity']]["hybrid"].append(hybrid_metrics["f1"])
                token_stats["hybrid"].append(hybrid_tokens["tokens_saved"])
                
                print(f"   混合策略:   {hybrid_prediction} (F1: {hybrid_metrics['f1']:.3f})")
                
            except Exception as e:
                print(f"   ❌ 混合策略测试失败: {e}")
                test_result["results"]["hybrid"] = {"error": str(e)}
            
            results["individual_results"].append(test_result)
        
        # 计算总体统计
        def calc_avg(values):
            return sum(values) / len(values) if values else 0.0
        
        results["summary_stats"] = {}
        for model_type in ["pretrained", "finetuned", "hybrid"]:
            if stats[model_type]["f1"]:
                results["summary_stats"][model_type] = {
                    "avg_exact_match": calc_avg(stats[model_type]["exact_match"]),
                    "avg_jaccard": calc_avg(stats[model_type]["jaccard"]),
                    "avg_f1": calc_avg(stats[model_type]["f1"]),
                    "avg_precision": calc_avg(stats[model_type]["precision"]),
                    "avg_recall": calc_avg(stats[model_type]["recall"]),
                    "avg_token_savings": calc_avg(token_stats[model_type])
                }
        
        # 复杂度分析
        results["complexity_analysis"] = {}
        for complexity in ["simple", "medium", "complex", "very_complex"]:
            results["complexity_analysis"][complexity] = {}
            for model_type in ["pretrained", "finetuned", "hybrid"]:
                if complexity_stats[complexity][model_type]:
                    results["complexity_analysis"][complexity][model_type] = calc_avg(complexity_stats[complexity][model_type])
        
        # Token使用分析
        results["token_usage_analysis"] = {
            "pretrained": calc_avg(token_stats["pretrained"]),
            "finetuned": calc_avg(token_stats["finetuned"]),
            "hybrid": calc_avg(token_stats["hybrid"])
        }
        
        return results
    
    def generate_comprehensive_report(self, results: Dict[str, Any]):
        """生成全面报告"""
        print("\\n" + "="*100)
        print("🎯 TokenFlow 模型性能全面对比报告")
        print("="*100)
        print(f"测试时间: {results['test_info']['timestamp']}")
        print(f"测试案例: {results['test_info']['total_test_cases']}个")
        print(f"对比模型: {', '.join(results['test_info']['comparison_types'])}")
        
        stats = results["summary_stats"]
        
        # 总体性能对比
        print("\\n📊 总体性能对比")
        print("-"*100)
        print(f"{'模型类型':<20} {'精确匹配':<10} {'Jaccard':<10} {'F1分数':<10} {'精确率':<10} {'召回率':<10} {'Token节省':<12}")
        print("-"*100)
        
        model_names = {
            "pretrained": "预训练Qwen3-0.6B",
            "finetuned": "微调纯模型",
            "hybrid": "微调混合策略"
        }
        
        for model_type in ["pretrained", "finetuned", "hybrid"]:
            if model_type in stats:
                s = stats[model_type]
                print(f"{model_names[model_type]:<20} {s['avg_exact_match']:<10.3f} {s['avg_jaccard']:<10.3f} {s['avg_f1']:<10.3f} {s['avg_precision']:<10.3f} {s['avg_recall']:<10.3f} {s['avg_token_savings']:<12.0f}")
        
        # 改进分析
        if "pretrained" in stats and "finetuned" in stats and "hybrid" in stats:
            print("\\n💡 改进分析")
            print("-"*100)
            
            # 微调纯模型 vs 预训练
            pre_f1 = stats["pretrained"]["avg_f1"]
            fine_f1 = stats["finetuned"]["avg_f1"]
            hybrid_f1 = stats["hybrid"]["avg_f1"]
            
            print(f"🚀 微调纯模型改进:")
            print(f"   F1分数: {pre_f1:.3f} → {fine_f1:.3f} (+{((fine_f1-pre_f1)/pre_f1*100):+.1f}%)")
            print(f"   精确匹配: {stats['pretrained']['avg_exact_match']:.3f} → {stats['finetuned']['avg_exact_match']:.3f} (+{((stats['finetuned']['avg_exact_match']-stats['pretrained']['avg_exact_match'])/stats['pretrained']['avg_exact_match']*100):+.1f}%)")
            
            print(f"\\n🎭 混合策略效果:")
            print(f"   vs 预训练: {pre_f1:.3f} → {hybrid_f1:.3f} (+{((hybrid_f1-pre_f1)/pre_f1*100):+.1f}%)")
            print(f"   vs 微调纯模型: {fine_f1:.3f} → {hybrid_f1:.3f} ({((hybrid_f1-fine_f1)/fine_f1*100):+.1f}%)")
        
        # 复杂度分析
        print("\\n🎪 不同复杂度场景表现")
        print("-"*100)
        print(f"{'复杂度':<15} {'预训练':<12} {'微调纯模型':<12} {'混合策略':<12} {'最佳':<10}")
        print("-"*100)
        
        complexity_names = {
            "simple": "简单",
            "medium": "中等", 
            "complex": "复杂",
            "very_complex": "极复杂"
        }
        
        for complexity in ["simple", "medium", "complex", "very_complex"]:
            if complexity in results["complexity_analysis"]:
                comp_data = results["complexity_analysis"][complexity]
                
                pre_score = comp_data.get("pretrained", 0)
                fine_score = comp_data.get("finetuned", 0)
                hybrid_score = comp_data.get("hybrid", 0)
                
                best_score = max(pre_score, fine_score, hybrid_score)
                best_model = "预训练" if best_score == pre_score else ("微调纯模型" if best_score == fine_score else "混合策略")
                
                print(f"{complexity_names[complexity]:<15} {pre_score:<12.3f} {fine_score:<12.3f} {hybrid_score:<12.3f} {best_model:<10}")
        
        # Token节省分析
        print("\\n💰 Token节省效果")
        print("-"*60)
        token_usage = results["token_usage_analysis"]
        
        for model_type in ["pretrained", "finetuned", "hybrid"]:
            if model_type in token_usage:
                savings = token_usage[model_type]
                savings_pct = (savings / 8000) * 100  # 相对于无过滤的8000 tokens
                print(f"{model_names[model_type]:<20}: 平均节省 {savings:.0f} tokens ({savings_pct:.1f}%)")
        
        # 关键发现总结
        print("\\n🎯 关键发现")
        print("-"*100)
        
        if "pretrained" in stats and "finetuned" in stats:
            improvement = ((stats["finetuned"]["avg_f1"] - stats["pretrained"]["avg_f1"]) / stats["pretrained"]["avg_f1"]) * 100
            
            if improvement > 100:
                print("🌟 微调取得巨大成功！F1分数提升超过100%")
            elif improvement > 50:
                print("✨ 微调效果显著，性能大幅提升")
            elif improvement > 20:
                print("📈 微调有明显效果，性能改善显著")
            
            # 分析混合策略的价值
            if "hybrid" in stats:
                hybrid_vs_pure = ((stats["hybrid"]["avg_f1"] - stats["finetuned"]["avg_f1"]) / stats["finetuned"]["avg_f1"]) * 100
                
                if hybrid_vs_pure > 5:
                    print("🎭 混合策略进一步提升了性能")
                elif hybrid_vs_pure > -5:
                    print("🎭 混合策略保持了稳定的性能")
                else:
                    print("🎭 混合策略略微降低了纯模型性能")
        
        # 最佳配置推荐
        print("\\n🏆 最佳配置推荐")
        print("-"*60)
        
        if "hybrid" in stats and "finetuned" in stats:
            if stats["hybrid"]["avg_f1"] >= stats["finetuned"]["avg_f1"]:
                print("🎯 推荐使用：微调混合策略")
                print("   理由：在保持高性能的同时提供更稳定的预测")
            else:
                print("🎯 推荐使用：微调纯模型") 
                print("   理由：纯神经网络模型性能最优")
        
        print("\\n🎊 评估完成！所有模型配置已全面对比分析。")

def main():
    """主函数"""
    comparator = ComprehensiveComparator()
    
    # 运行全面对比
    results = comparator.run_comprehensive_comparison()
    
    # 生成报告
    comparator.generate_comprehensive_report(results)
    
    # 保存结果
    try:
        with open("reports/comprehensive_comparison.json", "w", encoding="utf-8") as f:
            # 处理numpy类型
            results_clean = json.loads(json.dumps(results, default=str))
            json.dump(results_clean, f, ensure_ascii=False, indent=2)
        
        print(f"\\n✅ 详细对比结果已保存到: reports/comprehensive_comparison.json")
    except Exception as e:
        print(f"⚠️ 保存结果时出错: {e}")

if __name__ == "__main__":
    main()