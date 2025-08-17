#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试微调模型的Token节省效果
对比微调前后的检索精度和Token使用量
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.retriever import DocumentRetriever
from src.classifiers.hybrid_intent_classifier import HybridIntentClassifier
from src.classifiers.mock_intent_classifier import MockIntentClassifier
from src.core.large_llm_simulator import LargeLLMSimulator
import json
import time
from typing import Dict, List, Any

class TokenSavingsAnalyzer:
    """Token节省效果分析器"""
    
    def __init__(self):
        print("🔧 初始化Token节省分析器...")
        
        # 初始化文档检索器
        self.retriever = DocumentRetriever()
        self.retriever.initialize()
        
        # 初始化LLM模拟器
        self.llm_simulator = LargeLLMSimulator()
        
        # 初始化不同的分类器
        print("📥 初始化基准分类器...")
        self.mock_classifier = MockIntentClassifier()
        
        print("📥 初始化微调混合分类器...")
        self.finetuned_classifier = HybridIntentClassifier(use_finetuned=True)
        
        print("📥 初始化规则分类器...")
        self.rule_classifier = HybridIntentClassifier(use_finetuned=False)
        
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
                "prompt": "帮我查询账户余额和订单信息",
                "expected_intents": ["user_auth", "order_management"],
                "description": "多意图-账户+订单"
            },
            {
                "prompt": "订单支付失败，需要重新付款",
                "expected_intents": ["order_management", "payment"],
                "description": "多意图-订单+支付问题"
            }
        ]
    
    def run_classification_test(self, prompt: str) -> Dict[str, Any]:
        """运行单个分类测试"""
        
        # 1. Mock分类器
        mock_intent = self.mock_classifier.predict_intent(prompt)
        mock_intents = [mock_intent] if mock_intent != "none" else ["none"]
        
        # 2. 微调混合分类器
        finetuned_result = self.finetuned_classifier.classify(prompt)
        finetuned_intents = finetuned_result.get("intents", [finetuned_result.get("intent", "none")])
        
        # 3. 规则分类器
        rule_result = self.rule_classifier.classify(prompt)
        rule_intents = rule_result.get("intents", [rule_result.get("intent", "none")])
        
        return {
            "mock": {
                "intents": mock_intents,
                "confidence": 1.0,  # Mock分类器总是很确定
                "details": "基于规则的Mock分类器"
            },
            "finetuned": {
                "intents": finetuned_intents,
                "confidence": finetuned_result.get("confidence", 0.0),
                "details": finetuned_result
            },
            "rule_only": {
                "intents": rule_intents,
                "confidence": rule_result.get("confidence", 0.0),
                "details": rule_result
            }
        }
    
    def calculate_document_relevance(self, prompt: str, intents: List[str], max_docs: int = 5) -> Dict[str, Any]:
        """计算文档检索的相关性和token使用"""
        
        # 无过滤：直接检索
        no_filter_docs = self.retriever.retrieve_docs(prompt, top_k=max_docs)
        
        # 意图过滤：基于意图增强查询
        if "none" not in intents:
            intent_query = f"{prompt} {' '.join(intents)}"
            filtered_docs = self.retriever.retrieve_docs(intent_query, top_k=max_docs)
        else:
            filtered_docs = self.retriever.retrieve_docs(prompt, top_k=2)  # 减少无关查询的文档数
        
        # 估算token使用量
        def estimate_tokens(docs):
            total_tokens = 0
            for doc in docs:
                # 粗略估算：中文字符数 * 1.5
                content_length = len(doc.get('chunk_content', ''))
                tokens = int(content_length * 1.5)
                total_tokens += tokens
            return total_tokens
        
        no_filter_tokens = estimate_tokens(no_filter_docs)
        filtered_tokens = estimate_tokens(filtered_docs)
        
        return {
            "no_filter": {
                "docs_count": len(no_filter_docs),
                "estimated_tokens": no_filter_tokens,
                "docs": no_filter_docs
            },
            "with_filter": {
                "docs_count": len(filtered_docs),
                "estimated_tokens": filtered_tokens,
                "docs": filtered_docs
            },
            "token_savings": {
                "absolute": no_filter_tokens - filtered_tokens,
                "percentage": ((no_filter_tokens - filtered_tokens) / no_filter_tokens * 100) if no_filter_tokens > 0 else 0
            }
        }
    
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """运行全面测试"""
        print("🚀 开始全面Token节省效果测试...")
        
        results = {
            "test_info": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_test_cases": len(self.test_cases),
                "classifiers": ["Mock", "微调混合", "纯规则"]
            },
            "individual_results": [],
            "summary_stats": {}
        }
        
        total_token_savings = {"mock": 0, "finetuned": 0, "rule_only": 0}
        total_accuracy = {"mock": 0, "finetuned": 0, "rule_only": 0}
        
        for i, test_case in enumerate(self.test_cases):
            print(f"\\n📋 测试案例 {i+1}/{len(self.test_cases)}: {test_case['description']}")
            print(f"   输入: {test_case['prompt']}")
            print(f"   期望意图: {test_case['expected_intents']}")
            
            # 分类测试
            classification_results = self.run_classification_test(test_case['prompt'])
            
            # 文档检索和token计算
            doc_results = {}
            for classifier_name, class_result in classification_results.items():
                intents = class_result['intents']
                doc_result = self.calculate_document_relevance(
                    test_case['prompt'], 
                    intents, 
                    max_docs=5
                )
                doc_results[classifier_name] = doc_result
                
                # 累计token节省
                total_token_savings[classifier_name] += doc_result['token_savings']['absolute']
                
                # 计算准确率
                expected_set = set(test_case['expected_intents'])
                predicted_set = set(intents)
                
                if expected_set == predicted_set:
                    accuracy = 1.0
                elif expected_set & predicted_set:  # 有交集
                    accuracy = len(expected_set & predicted_set) / len(expected_set | predicted_set)
                else:
                    accuracy = 0.0
                
                total_accuracy[classifier_name] += accuracy
                
                print(f"   {classifier_name:12}: {intents} (准确率: {accuracy:.2f}, Token节省: {doc_result['token_savings']['absolute']})")
            
            # 保存单个测试结果
            test_result = {
                "test_case": test_case,
                "classification_results": classification_results,
                "document_retrieval": doc_results
            }
            results["individual_results"].append(test_result)
        
        # 计算总体统计
        num_cases = len(self.test_cases)
        results["summary_stats"] = {
            "average_accuracy": {
                "mock": total_accuracy["mock"] / num_cases,
                "finetuned": total_accuracy["finetuned"] / num_cases,
                "rule_only": total_accuracy["rule_only"] / num_cases
            },
            "total_token_savings": total_token_savings,
            "average_token_savings": {
                "mock": total_token_savings["mock"] / num_cases,
                "finetuned": total_token_savings["finetuned"] / num_cases,
                "rule_only": total_token_savings["rule_only"] / num_cases
            }
        }
        
        return results
    
    def generate_report(self, results: Dict[str, Any]):
        """生成测试报告"""
        print("\\n" + "="*80)
        print("🎯 Token节省效果测试报告")
        print("="*80)
        
        stats = results["summary_stats"]
        
        print("\\n📊 分类器性能对比:")
        print("-"*60)
        print(f"{'分类器':<15} {'平均准确率':<12} {'平均Token节省':<15}")
        print("-"*60)
        
        for classifier in ["mock", "finetuned", "rule_only"]:
            name_map = {"mock": "Mock分类器", "finetuned": "微调混合", "rule_only": "纯规则"}
            accuracy = stats["average_accuracy"][classifier]
            avg_savings = stats["average_token_savings"][classifier]
            
            print(f"{name_map[classifier]:<15} {accuracy:<12.3f} {avg_savings:<15.1f}")
        
        print("\\n💡 关键发现:")
        
        # 找出最佳分类器
        best_accuracy = max(stats["average_accuracy"].values())
        best_savings = max(stats["average_token_savings"].values())
        
        best_acc_classifier = max(stats["average_accuracy"], key=stats["average_accuracy"].get)
        best_savings_classifier = max(stats["average_token_savings"], key=stats["average_token_savings"].get)
        
        name_map = {"mock": "Mock分类器", "finetuned": "微调混合分类器", "rule_only": "纯规则分类器"}
        
        print(f"  ✅ 最高准确率: {name_map[best_acc_classifier]} ({best_accuracy:.3f})")
        print(f"  ✅ 最大Token节省: {name_map[best_savings_classifier]} ({best_savings:.1f} tokens)")
        
        # 微调效果分析
        finetuned_acc = stats["average_accuracy"]["finetuned"]
        mock_acc = stats["average_accuracy"]["mock"]
        improvement = finetuned_acc - mock_acc
        
        print(f"\\n🚀 微调效果评估:")
        print(f"  📈 准确率改进: {improvement:+.3f} ({improvement/mock_acc*100:+.1f}%)")
        
        finetuned_savings = stats["average_token_savings"]["finetuned"]
        mock_savings = stats["average_token_savings"]["mock"]
        savings_improvement = finetuned_savings - mock_savings
        
        print(f"  📉 Token节省提升: {savings_improvement:+.1f} tokens ({savings_improvement/mock_savings*100:+.1f}%)")
        
        print("\\n🎊 总结:")
        if improvement > 0 and savings_improvement > 0:
            print("  🌟 微调模型在准确率和Token节省方面都有提升")
        elif improvement > 0:
            print("  ✨ 微调模型在准确率方面有显著提升")
        elif savings_improvement > 0:
            print("  ⚡ 微调模型在Token节省方面有改进")
        else:
            print("  🔧 微调模型有改进空间，建议进一步优化")

def main():
    """主函数"""
    analyzer = TokenSavingsAnalyzer()
    
    # 运行测试
    results = analyzer.run_comprehensive_test()
    
    # 生成报告
    analyzer.generate_report(results)
    
    # 保存详细结果
    with open("reports/token_savings_analysis.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\\n✅ 详细测试结果已保存到: reports/token_savings_analysis.json")

if __name__ == "__main__":
    main()