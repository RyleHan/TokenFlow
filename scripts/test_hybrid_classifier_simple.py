#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
简化测试：仅测试混合分类器性能
不依赖文档检索组件
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.classifiers.hybrid_intent_classifier import HybridIntentClassifier
from src.classifiers.intent_classifier import MockIntentClassifier
import json
import time
from typing import Dict, List, Any

class SimpleClassifierTester:
    """简化分类器测试器"""
    
    def __init__(self):
        print("🔧 初始化分类器测试器...")
        
        # 初始化不同的分类器
        print("📥 初始化Mock分类器...")
        self.mock_classifier = MockIntentClassifier()
        
        print("📥 初始化微调混合分类器...")
        try:
            self.finetuned_classifier = HybridIntentClassifier(use_finetuned=True)
            self.finetuned_available = True
        except Exception as e:
            print(f"⚠️ 微调分类器初始化失败: {e}")
            self.finetuned_available = False
        
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
            }
        ]
    
    def calculate_accuracy(self, predicted: List[str], expected: List[str]) -> Dict[str, float]:
        """计算准确率指标"""
        predicted_set = set(predicted)
        expected_set = set(expected)
        
        # 精确匹配
        exact_match = 1.0 if predicted_set == expected_set else 0.0
        
        # 部分匹配（Jaccard相似度）
        intersection = len(predicted_set & expected_set)
        union = len(predicted_set | expected_set)
        jaccard = intersection / union if union > 0 else 0.0
        
        # F1分数
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
    
    def estimate_token_savings(self, intents: List[str], prompt: str) -> Dict[str, Any]:
        """估算Token节省效果"""
        
        # 基础假设
        total_docs = 20  # 总文档数
        avg_doc_tokens = 400  # 每个文档平均token数
        
        # 根据意图过滤文档
        if "none" in intents:
            # 无关查询，检索较少文档
            retrieved_docs = 2
        else:
            # 有业务意图，根据意图数量调整检索文档数
            base_docs = 3
            intent_bonus = len([i for i in intents if i != "none"]) * 1
            retrieved_docs = min(base_docs + intent_bonus, 8)
        
        # 计算token使用量
        without_filtering = total_docs * avg_doc_tokens
        with_filtering = retrieved_docs * avg_doc_tokens
        
        savings = without_filtering - with_filtering
        savings_percentage = (savings / without_filtering) * 100 if without_filtering > 0 else 0
        
        return {
            "total_docs": total_docs,
            "retrieved_docs": retrieved_docs,
            "without_filtering_tokens": without_filtering,
            "with_filtering_tokens": with_filtering,
            "tokens_saved": savings,
            "savings_percentage": savings_percentage,
            "efficiency": (total_docs - retrieved_docs) / total_docs * 100
        }
    
    def run_test(self) -> Dict[str, Any]:
        """运行测试"""
        print("🚀 开始分类器性能测试...")
        
        results = {
            "test_info": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_test_cases": len(self.test_cases),
                "finetuned_available": self.finetuned_available
            },
            "individual_results": [],
            "summary_stats": {}
        }
        
        # 累计统计
        mock_stats = {"accuracy": [], "token_savings": []}
        finetuned_stats = {"accuracy": [], "token_savings": []}
        
        for i, test_case in enumerate(self.test_cases):
            print(f"\\n📋 测试案例 {i+1}/{len(self.test_cases)}: {test_case['description']}")
            print(f"   输入: {test_case['prompt']}")
            print(f"   期望: {test_case['expected_intents']}")
            
            test_result = {
                "test_case": test_case,
                "results": {}
            }
            
            # 1. Mock分类器测试
            mock_intent = self.mock_classifier.predict_top_intent(test_case['prompt'])
            mock_intents = [mock_intent]
            mock_accuracy = self.calculate_accuracy(mock_intents, test_case['expected_intents'])
            mock_token_savings = self.estimate_token_savings(mock_intents, test_case['prompt'])
            
            test_result["results"]["mock"] = {
                "predicted_intents": mock_intents,
                "accuracy_metrics": mock_accuracy,
                "token_savings": mock_token_savings
            }
            
            mock_stats["accuracy"].append(mock_accuracy["f1"])
            mock_stats["token_savings"].append(mock_token_savings["tokens_saved"])
            
            print(f"   Mock:     {mock_intents} (F1: {mock_accuracy['f1']:.3f}, 节省: {mock_token_savings['tokens_saved']} tokens)")
            
            # 2. 微调混合分类器测试
            if self.finetuned_available:
                try:
                    finetuned_result = self.finetuned_classifier.classify(test_case['prompt'])
                    finetuned_intents = finetuned_result.get("intents", [finetuned_result.get("intent", "none")])
                    finetuned_accuracy = self.calculate_accuracy(finetuned_intents, test_case['expected_intents'])
                    finetuned_token_savings = self.estimate_token_savings(finetuned_intents, test_case['prompt'])
                    
                    test_result["results"]["finetuned"] = {
                        "predicted_intents": finetuned_intents,
                        "accuracy_metrics": finetuned_accuracy,
                        "token_savings": finetuned_token_savings,
                        "full_result": finetuned_result
                    }
                    
                    finetuned_stats["accuracy"].append(finetuned_accuracy["f1"])
                    finetuned_stats["token_savings"].append(finetuned_token_savings["tokens_saved"])
                    
                    print(f"   Finetuned: {finetuned_intents} (F1: {finetuned_accuracy['f1']:.3f}, 节省: {finetuned_token_savings['tokens_saved']} tokens)")
                    
                except Exception as e:
                    print(f"   ❌ 微调分类器测试失败: {e}")
                    test_result["results"]["finetuned"] = {"error": str(e)}
            
            results["individual_results"].append(test_result)
        
        # 计算总体统计
        def calc_avg(values):
            return sum(values) / len(values) if values else 0.0
        
        results["summary_stats"] = {
            "mock": {
                "avg_f1": calc_avg(mock_stats["accuracy"]),
                "avg_token_savings": calc_avg(mock_stats["token_savings"]),
                "total_token_savings": sum(mock_stats["token_savings"])
            }
        }
        
        if self.finetuned_available and finetuned_stats["accuracy"]:
            results["summary_stats"]["finetuned"] = {
                "avg_f1": calc_avg(finetuned_stats["accuracy"]),
                "avg_token_savings": calc_avg(finetuned_stats["token_savings"]),
                "total_token_savings": sum(finetuned_stats["token_savings"])
            }
            
            # 计算改进
            f1_improvement = results["summary_stats"]["finetuned"]["avg_f1"] - results["summary_stats"]["mock"]["avg_f1"]
            token_improvement = results["summary_stats"]["finetuned"]["avg_token_savings"] - results["summary_stats"]["mock"]["avg_token_savings"]
            
            results["summary_stats"]["improvement"] = {
                "f1_improvement": f1_improvement,
                "token_savings_improvement": token_improvement,
                "f1_relative_improvement": (f1_improvement / results["summary_stats"]["mock"]["avg_f1"] * 100) if results["summary_stats"]["mock"]["avg_f1"] > 0 else 0,
                "token_relative_improvement": (token_improvement / results["summary_stats"]["mock"]["avg_token_savings"] * 100) if results["summary_stats"]["mock"]["avg_token_savings"] > 0 else 0
            }
        
        return results
    
    def generate_report(self, results: Dict[str, Any]):
        """生成测试报告"""
        print("\\n" + "="*80)
        print("🎯 微调模型分类器性能测试报告")
        print("="*80)
        
        stats = results["summary_stats"]
        
        print("\\n📊 分类器性能对比:")
        print("-"*70)
        print(f"{'分类器':<15} {'平均F1分数':<12} {'平均Token节省':<15} {'总Token节省':<12}")
        print("-"*70)
        
        # Mock分类器
        mock_stats = stats["mock"]
        print(f"{'Mock分类器':<15} {mock_stats['avg_f1']:<12.3f} {mock_stats['avg_token_savings']:<15.1f} {mock_stats['total_token_savings']:<12.0f}")
        
        # 微调分类器
        if "finetuned" in stats:
            finetuned_stats = stats["finetuned"]
            print(f"{'微调混合':<15} {finetuned_stats['avg_f1']:<12.3f} {finetuned_stats['avg_token_savings']:<15.1f} {finetuned_stats['total_token_savings']:<12.0f}")
            
            # 改进统计
            if "improvement" in stats:
                imp = stats["improvement"]
                print("\\n💡 微调效果分析:")
                print(f"  📈 F1分数改进: {imp['f1_improvement']:+.3f} ({imp['f1_relative_improvement']:+.1f}%)")
                print(f"  📉 Token节省提升: {imp['token_savings_improvement']:+.1f} tokens ({imp['token_relative_improvement']:+.1f}%)")
                
                print("\\n🎯 微调模型评估:")
                if imp['f1_improvement'] > 0 and imp['token_savings_improvement'] > 0:
                    print("  🌟 微调取得全面成功！准确率和Token节省都有显著提升")
                elif imp['f1_improvement'] > 0.1:
                    print("  ✨ 微调在准确率方面取得显著成功")
                elif imp['token_savings_improvement'] > 100:
                    print("  ⚡ 微调在Token节省方面有明显改进")
                elif imp['f1_improvement'] > 0:
                    print("  📈 微调有一定效果，准确率有所提升")
                else:
                    print("  🔧 微调效果有限，建议进一步优化策略")
        else:
            print("⚠️ 微调分类器测试失败")
        
        # 多意图识别能力分析
        print("\\n🎪 多意图识别能力分析:")
        multi_intent_cases = [case for case in results["individual_results"] if len(case["test_case"]["expected_intents"]) > 1 and case["test_case"]["expected_intents"] != ["none"]]
        
        if multi_intent_cases:
            print(f"  📊 多意图测试案例: {len(multi_intent_cases)}个")
            
            if "finetuned" in stats:
                multi_intent_performance = []
                for case in multi_intent_cases:
                    if "finetuned" in case["results"] and "error" not in case["results"]["finetuned"]:
                        f1 = case["results"]["finetuned"]["accuracy_metrics"]["f1"]
                        multi_intent_performance.append(f1)
                
                if multi_intent_performance:
                    avg_multi_f1 = sum(multi_intent_performance) / len(multi_intent_performance)
                    print(f"  🎯 微调模型多意图平均F1: {avg_multi_f1:.3f}")
                    
                    if avg_multi_f1 > 0.7:
                        print("  🌟 多意图识别能力优秀")
                    elif avg_multi_f1 > 0.5:
                        print("  ✅ 多意图识别能力良好")
                    elif avg_multi_f1 > 0.3:
                        print("  📈 多意图识别能力一般，有改进空间")
                    else:
                        print("  ⚠️ 多意图识别能力需要改进")

def main():
    """主函数"""
    tester = SimpleClassifierTester()
    
    # 运行测试
    results = tester.run_test()
    
    # 生成报告
    tester.generate_report(results)
    
    # 保存结果
    with open("reports/simple_classifier_test.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\\n✅ 详细测试结果已保存到: reports/simple_classifier_test.json")

if __name__ == "__main__":
    main()