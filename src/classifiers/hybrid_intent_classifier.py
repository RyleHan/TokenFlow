#!/usr/bin/env python3
"""
混合意图分类器：结合规则匹配和神经网络
目标：利用Mock分类器的高准确率 + 神经网络的泛化能力
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import numpy as np
from typing import List, Dict, Tuple, Optional
import re
from sentence_transformers import SentenceTransformer
import json

class HybridIntentClassifier:
    def __init__(self, model_path: Optional[str] = None, use_finetuned: bool = True):
        self.intent_categories = [
            "order_management", "user_auth", "payment", 
            "inventory", "notification", "none"
        ]
        
        # 权重配置 - 微调模型权重更高
        if use_finetuned:
            self.rule_weight = 0.3  # 规则匹配权重
            self.neural_weight = 0.7  # 神经网络权重
        else:
            self.rule_weight = 0.7  # 规则匹配权重
            self.neural_weight = 0.3  # 神经网络权重
        
        self.use_finetuned = use_finetuned
        
        # 初始化各个组件
        self._init_rule_classifier()
        self._init_neural_classifier(model_path)
        self._init_semantic_matcher()
        
        print(f"🔧 混合分类器初始化完成")
        print(f"📊 权重配置: 规则={self.rule_weight}, 神经网络={self.neural_weight}")
        print(f"🎯 使用微调模型: {self.use_finetuned}")

    def _init_rule_classifier(self):
        """初始化规则分类器（基于Mock分类器）"""
        self.intent_keywords = {
            "order_management": [
                "订单", "下单", "购买", "买", "商品", "产品", "购物车", "结算", 
                "checkout", "order", "购物", "下订单", "商品信息", "产品详情"
            ],
            "user_auth": [
                "登录", "注册", "账户", "账号", "密码", "用户名", "认证", "验证",
                "login", "register", "account", "用户", "身份", "权限"
            ],
            "payment": [
                "支付", "付款", "费用", "价格", "金额", "账单", "扣费", "充值",
                "payment", "pay", "钱", "余额", "收费", "退款"
            ],
            "inventory": [
                "库存", "现货", "有货", "缺货", "补货", "进货", "出货", "盘点",
                "inventory", "stock", "存货", "仓库", "库房"
            ],
            "notification": [
                "通知", "提醒", "消息", "推送", "邮件", "短信", "提示", "警告",
                "notification", "message", "alert", "消息提醒"
            ]
        }
        
        # 语义相似度基准句子
        self.intent_examples = {
            "order_management": ["我想查看我的订单", "如何下单购买商品", "购物车结算"],
            "user_auth": ["怎么登录账户", "忘记密码了", "注册新用户"],
            "payment": ["支付失败了", "怎么付款", "查看账单费用"],
            "inventory": ["库存不足", "查看现货情况", "补货通知"],
            "notification": ["发送提醒消息", "设置通知", "查看推送"],
            "none": ["今天天气怎么样", "推荐一部电影", "什么是人工智能"]
        }

    def _init_neural_classifier(self, model_path: Optional[str]):
        """初始化神经网络分类器"""
        # 如果没有指定路径且使用微调模型，使用默认的微调模型路径
        if model_path is None and self.use_finetuned:
            model_path = "models/qwen3_fixed_classifier"
        
        if model_path and self._model_exists(model_path):
            try:
                print(f"📥 加载神经网络模型: {model_path}")
                
                # 设置设备
                if torch.backends.mps.is_available():
                    self.device = torch.device("mps")
                    print("✅ 使用MPS加速")
                else:
                    self.device = torch.device("cpu")
                    print("⚠️ 使用CPU")
                
                # 加载tokenizer
                if self.use_finetuned:
                    # 微调模型使用自己的tokenizer
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        model_path, trust_remote_code=True
                    )
                    base_model_name = "Qwen/Qwen3-0.6B"
                else:
                    # 原有模型逻辑
                    self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                    base_model_name = "Qwen/Qwen2.5-0.5B-Instruct"
                
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                # 加载基础模型
                base_model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    torch_dtype=torch.float32,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                )
                
                # 加载PEFT模型
                self.neural_model = PeftModel.from_pretrained(base_model, model_path)
                self.neural_model.eval()
                self.neural_model = self.neural_model.to(self.device)
                
                # 禁用梯度计算
                for param in self.neural_model.parameters():
                    param.requires_grad = False
                
                self.neural_available = True
                print("✅ 神经网络模型加载成功")
                
            except Exception as e:
                print(f"⚠️ 神经网络模型加载失败: {e}")
                self.neural_available = False
        else:
            print("⚠️ 神经网络模型不可用，使用纯规则模式")
            self.neural_available = False

    def _init_semantic_matcher(self):
        """初始化语义匹配器"""
        try:
            print("📥 加载语义匹配模型...")
            self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # 预计算意图示例的嵌入
            self.intent_embeddings = {}
            for intent, examples in self.intent_examples.items():
                embeddings = self.semantic_model.encode(examples)
                self.intent_embeddings[intent] = np.mean(embeddings, axis=0)
            
            self.semantic_available = True
            print("✅ 语义匹配器初始化成功")
        except Exception as e:
            print(f"⚠️ 语义匹配器初始化失败: {e}")
            self.semantic_available = False

    def _model_exists(self, path: str) -> bool:
        """检查模型是否存在"""
        import os
        return os.path.exists(path) and os.path.exists(os.path.join(path, "adapter_config.json"))

    def classify_by_rules(self, text: str) -> Dict[str, float]:
        """基于规则的分类"""
        text_lower = text.lower()
        scores = {intent: 0.0 for intent in self.intent_categories}
        
        # 关键词匹配
        for intent, keywords in self.intent_keywords.items():
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    scores[intent] += 1.0
        
        # 语义相似度匹配（如果可用）
        if self.semantic_available:
            query_embedding = self.semantic_model.encode([text])
            for intent, intent_embedding in self.intent_embeddings.items():
                similarity = np.dot(query_embedding[0], intent_embedding) / (
                    np.linalg.norm(query_embedding[0]) * np.linalg.norm(intent_embedding)
                )
                scores[intent] += similarity * 0.5  # 语义相似度权重
        
        # 归一化
        total_score = sum(scores.values())
        if total_score > 0:
            scores = {k: v / total_score for k, v in scores.items()}
        else:
            scores["none"] = 1.0
        
        return scores

    def classify_by_neural(self, text: str) -> Dict[str, float]:
        """基于神经网络的分类"""
        if not self.neural_available:
            return {intent: 1.0/len(self.intent_categories) for intent in self.intent_categories}
        
        try:
            if self.use_finetuned:
                # 使用微调模型的格式
                prompt = f"用户: {text}\\n助手: "
            else:
                # 原有模型格式
                prompt = f"分类意图: {text}\\n意图:"
            
            # Tokenize
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                max_length=256, 
                truncation=True
            )
            
            # 移动到正确设备
            if self.device.type != "cpu":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 生成
            with torch.no_grad():
                outputs = self.neural_model.generate(
                    **inputs,
                    max_new_tokens=25,  # 增加生成长度以适应多意图
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            # 解码
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            predicted_intents = self._parse_neural_output(generated_text)
            
            # 转换为概率分布
            scores = {intent: 0.05 for intent in self.intent_categories}  # 基础概率
            
            if predicted_intents:
                # 分配概率给预测的意图
                prob_per_intent = 0.9 / len(predicted_intents)
                for intent in predicted_intents:
                    if intent in self.intent_categories:
                        scores[intent] = prob_per_intent
            else:
                scores["none"] = 0.9
            
            return scores
            
        except Exception as e:
            print(f"⚠️ 神经网络推理失败: {e}")
            return {intent: 1.0/len(self.intent_categories) for intent in self.intent_categories}

    def _parse_neural_output(self, output: str) -> List[str]:
        """解析神经网络输出，支持多意图"""
        if self.use_finetuned:
            # 微调模型格式解析
            if "助手: " in output:
                intent_part = output.split("助手: ")[-1].strip()
            else:
                intent_part = output.strip()
        else:
            # 原有格式解析
            if "意图:" in output:
                intent_part = output.split("意图:")[-1].strip()
            else:
                intent_part = output.strip()
        
        # 清理和标准化
        intent_part = intent_part.lower().replace("\\n", " ")
        
        # 提取意图
        found_intents = []
        for intent in self.intent_categories:
            if intent != "none":  # 先检查业务意图
                intent_clean = intent.replace("_", "")
                if intent in intent_part or intent_clean in intent_part.replace("_", ""):
                    found_intents.append(intent)
        
        # 后处理：移除多余的none预测（根据我们的分析）
        if found_intents:
            return found_intents
        
        # 如果没有找到业务意图，检查是否为none
        if "none" in intent_part or not found_intents:
            return ["none"]
        
        return found_intents if found_intents else ["none"]

    def classify(self, text: str) -> Dict[str, any]:
        """混合分类，支持多意图"""
        # 规则分类
        rule_scores = self.classify_by_rules(text)
        
        # 神经网络分类
        neural_scores = self.classify_by_neural(text)
        
        # 加权融合
        final_scores = {}
        for intent in self.intent_categories:
            final_scores[intent] = (
                self.rule_weight * rule_scores.get(intent, 0) +
                self.neural_weight * neural_scores.get(intent, 0)
            )
        
        # 多意图支持：选择所有超过阈值的意图
        threshold = 0.3 if self.use_finetuned else 0.5
        predicted_intents = []
        
        for intent, score in final_scores.items():
            if score >= threshold and intent != "none":
                predicted_intents.append(intent)
        
        # 如果没有业务意图，返回none
        if not predicted_intents:
            predicted_intents = ["none"]
        
        # 主要意图（最高分）
        primary_intent = max(final_scores, key=final_scores.get)
        primary_confidence = final_scores[primary_intent]
        
        return {
            "intent": primary_intent,  # 兼容性：主要意图
            "intents": predicted_intents,  # 多意图结果
            "confidence": primary_confidence,
            "rule_scores": rule_scores,
            "neural_scores": neural_scores,
            "final_scores": final_scores,
            "threshold": threshold
        }

    def predict_intent(self, text: str) -> str:
        """简化的预测接口，与现有代码兼容"""
        result = self.classify(text)
        return result["intent"]

    def batch_classify(self, texts: List[str]) -> List[Dict]:
        """批量分类"""
        return [self.classify(text) for text in texts]

def test_hybrid_classifier():
    """测试混合分类器"""
    print("🧪 测试混合分类器...")
    
    # 创建分类器 - 使用微调模型
    classifier = HybridIntentClassifier(use_finetuned=True)
    
    # 测试用例
    test_cases = [
        "我想查看我的订单状态",
        "怎么登录我的账户",
        "支付失败了怎么办",
        "库存不足怎么处理",
        "发送通知消息",
        "今天天气怎么样",
        "用户登录后查看订单",  # 多意图样本
        "支付成功后发送通知",   # 多意图样本
        "修改身份要是有问题就付款充值",  # 复杂多意图
        "因为账单问题需要支付因为标记短信",  # 复杂多意图
    ]
    
    print("\\n📊 分类结果:")
    print("-" * 80)
    
    for text in test_cases:
        result = classifier.classify(text)
        print(f"输入: {text}")
        print(f"主意图: {result['intent']} (置信度: {result['confidence']:.3f})")
        print(f"所有意图: {result['intents']}")
        print(f"阈值: {result['threshold']}")
        print(f"最终分数: {result['final_scores']}")
        print("-" * 80)

if __name__ == "__main__":
    test_hybrid_classifier()