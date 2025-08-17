import re
from typing import List, Dict, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json

class MockIntentClassifier:
    """
    模拟的意图识别器
    在实际应用中，这里应该是微调后的小型LLM
    """
    
    def __init__(self):
        # 定义意图类别和关键词
        self.intent_mapping = {
            "order_management": {
                "keywords": [
                    "订单", "order", "购买", "下单", "状态", "取消", "查询", 
                    "物流", "配送", "发货", "收货", "退货", "换货", "订单号",
                    "购物", "买", "商品", "快递", "包裹", "追踪", "跟踪"
                ],
                "patterns": [
                    r"订单.*状态", r"查看.*订单", r"取消.*订单", r"订单.*查询",
                    r"物流.*信息", r"快递.*状态", r"配送.*情况"
                ]
            },
            "user_auth": {
                "keywords": [
                    "登录", "注册", "密码", "账户", "认证", "令牌", "退出", 
                    "重置", "验证", "登出", "用户", "账号", "身份", "权限",
                    "激活", "验证码", "找回", "修改密码", "个人信息"
                ],
                "patterns": [
                    r"登录.*账户", r"注册.*用户", r"重置.*密码", r"忘记.*密码",
                    r"验证.*身份", r"退出.*登录", r"账户.*认证"
                ]
            },
            "payment": {
                "keywords": [
                    "支付", "付款", "退款", "金额", "交易", "银行卡", "微信", 
                    "支付宝", "订单支付", "费用", "价格", "结算", "账单",
                    "扣款", "充值", "余额", "优惠券", "折扣", "支付方式"
                ],
                "patterns": [
                    r"支付.*失败", r"付款.*问题", r"退款.*申请", r"支付.*方式",
                    r"交易.*状态", r"费用.*查询", r"账单.*信息"
                ]
            },
            "inventory": {
                "keywords": [
                    "库存", "商品", "数量", "预留", "缺货", "补货", "仓库", 
                    "调拨", "盘点", "现货", "断货", "进货", "出货", "存货",
                    "备货", "货源", "供货", "库房", "存量", "余量"
                ],
                "patterns": [
                    r"库存.*不足", r"商品.*缺货", r"库存.*查询", r"补货.*申请",
                    r"仓库.*管理", r"库存.*预警", r"货品.*数量"
                ]
            },
            "notification": {
                "keywords": [
                    "通知", "消息", "邮件", "短信", "推送", "提醒", "发送", 
                    "接收", "通告", "公告", "提示", "警报", "告知", "通报",
                    "信息", "提醒", "催告", "知会", "传达"
                ],
                "patterns": [
                    r"发送.*通知", r"推送.*消息", r"邮件.*提醒", r"短信.*通知",
                    r"消息.*推送", r"通知.*设置", r"提醒.*功能"
                ]
            }
        }
        
        # 初始化TF-IDF向量化器（用于语义相似度计算）
        self.vectorizer = None
        self.intent_vectors = {}
        self._build_intent_vectors()
    
    def _build_intent_vectors(self):
        """构建意图向量化表示"""
        all_texts = []
        intent_labels = []
        
        # 为每个意图构建文本表示
        for intent, info in self.intent_mapping.items():
            # 将关键词组合成文本
            intent_text = " ".join(info["keywords"])
            all_texts.append(intent_text)
            intent_labels.append(intent)
        
        # 训练TF-IDF向量化器
        self.vectorizer = TfidfVectorizer(
            analyzer='word',
            ngram_range=(1, 2),
            max_features=1000,
            stop_words=None
        )
        
        vectors = self.vectorizer.fit_transform(all_texts)
        
        # 存储每个意图的向量
        for i, intent in enumerate(intent_labels):
            self.intent_vectors[intent] = vectors[i]
    
    def _extract_keywords(self, text: str) -> List[str]:
        """从文本中提取关键词"""
        # 简单的关键词提取，实际应用中可以使用更复杂的方法
        words = re.findall(r'\b\w+\b', text.lower())
        return words
    
    def _keyword_matching_score(self, text: str, intent: str) -> float:
        """基于关键词匹配计算意图得分"""
        text_lower = text.lower()
        intent_info = self.intent_mapping[intent]
        
        # 关键词匹配
        keyword_matches = 0
        for keyword in intent_info["keywords"]:
            if keyword.lower() in text_lower:
                keyword_matches += 1
        
        # 正则模式匹配
        pattern_matches = 0
        for pattern in intent_info.get("patterns", []):
            if re.search(pattern, text_lower):
                pattern_matches += 2  # 模式匹配权重更高
        
        # 计算得分
        total_keywords = len(intent_info["keywords"])
        keyword_score = keyword_matches / total_keywords if total_keywords > 0 else 0
        pattern_score = min(pattern_matches * 0.1, 0.5)  # 模式匹配最多贡献0.5分
        
        return keyword_score + pattern_score
    
    def _semantic_similarity_score(self, text: str, intent: str) -> float:
        """基于语义相似度计算意图得分"""
        if self.vectorizer is None:
            return 0.0
        
        try:
            # 向量化输入文本
            text_vector = self.vectorizer.transform([text])
            
            # 计算与意图向量的相似度
            intent_vector = self.intent_vectors[intent]
            similarity = cosine_similarity(text_vector, intent_vector)[0][0]
            
            return float(similarity)
        except:
            return 0.0
    
    def predict_intent(self, text: str, threshold: float = 0.05) -> List[Tuple[str, float]]:
        """
        预测文本的意图
        
        Args:
            text: 输入文本
            threshold: 意图识别阈值
            
        Returns:
            List of (intent, score) tuples, sorted by score
        """
        scores = {}
        
        for intent in self.intent_mapping.keys():
            # 结合关键词匹配和语义相似度
            keyword_score = self._keyword_matching_score(text, intent)
            semantic_score = self._semantic_similarity_score(text, intent)
            
            # 加权组合得分
            final_score = 0.8 * keyword_score + 0.2 * semantic_score
            scores[intent] = final_score
        
        # 排序并过滤低于阈值的意图
        sorted_intents = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        filtered_intents = [(intent, score) for intent, score in sorted_intents if score >= threshold]
        
        # 如果没有意图超过阈值，返回"none"
        if not filtered_intents:
            return [("none", 1.0)]
        
        return filtered_intents
    
    def predict_top_intent(self, text: str) -> str:
        """返回最可能的单个意图"""
        predictions = self.predict_intent(text)
        if predictions:
            return predictions[0][0]
        return "none"
    
    def predict_intents(self, text: str, max_intents: int = 3) -> List[str]:
        """返回多个可能的意图（用于多意图场景）"""
        predictions = self.predict_intent(text)
        return [intent for intent, score in predictions[:max_intents]]

def test_intent_classifier():
    """测试意图识别器"""
    classifier = MockIntentClassifier()
    
    test_cases = [
        "我想查看我的订单状态",
        "怎么登录我的账户",
        "支付失败了怎么办",
        "库存不够怎么处理",
        "怎么发送通知给用户",
        "今天天气怎么样",
        "我要买东西并且需要登录",
        "订单支付遇到问题了",
        "查看商品库存和价格信息"
    ]
    
    print("=== 意图识别测试 ===\n")
    
    for text in test_cases:
        predictions = classifier.predict_intent(text)
        top_intent = classifier.predict_top_intent(text)
        
        print(f"输入: {text}")
        print(f"最可能意图: {top_intent}")
        print("所有可能意图:")
        for intent, score in predictions:
            print(f"  {intent}: {score:.3f}")
        print("-" * 50)

if __name__ == "__main__":
    test_intent_classifier()