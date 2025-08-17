import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import time
import json
from typing import List, Tuple
import re

class RealIntentClassifier:
    """
    使用微调后的Qwen3-0.6B模型的真实意图识别器
    替代MockIntentClassifier，提供相同的接口
    """
    
    def __init__(self, model_dir: str = "./intent_classifier_model_qwen3"):
        self.model_dir = model_dir
        # 为了避免MPS的内存限制，推理时使用CPU
        self.device = torch.device("cpu")
        print(f"RealIntentClassifier使用设备: {self.device}")
        
        # 意图映射
        self.intent_categories = [
            "order_management", "user_auth", "payment", 
            "inventory", "notification", "none"
        ]
        
        self._load_model()
    
    def _load_model(self):
        """加载微调后的模型"""
        try:
            print("正在加载微调后的意图识别模型...")
            
            # 加载分词器
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
            
            # 加载基础模型
            base_model = AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen3-0.6B-Instruct",
                torch_dtype=torch.float32,  # CPU使用float32
                device_map="cpu",
                trust_remote_code=True
            )
            
            # 加载LoRA适配器
            self.model = PeftModel.from_pretrained(base_model, self.model_dir)
            
            # 合并适配器权重以提高推理速度
            self.model = self.model.merge_and_unload()
            
            # 移动到CPU
            self.model = self.model.to(self.device)
            self.model.eval()  # 设置为评估模式
            
            print("✅ 真实意图识别模型加载完成!")
            
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            print("💡 回退到Mock分类器...")
            self._fallback_to_mock()
    
    def _fallback_to_mock(self):
        """回退到Mock分类器"""
        from intent_classifier import MockIntentClassifier
        self.mock_classifier = MockIntentClassifier()
        self.use_mock = True
        self.model = None
        self.tokenizer = None
        print("⚠️ 使用Mock意图分类器作为备用方案")
    
    def _predict_with_model(self, user_prompt: str) -> str:
        """使用真实模型进行预测"""
        if self.model is None or self.tokenizer is None:
            return "none"
        
        try:
            # 构建输入格式（与训练时一致）
            input_text = f"用户: {user_prompt}\n助手: "
            
            # tokenize
            inputs = self.tokenizer(
                input_text,
                return_tensors="pt",
                truncation=True,
                max_length=256
            )
            
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 生成响应
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=15,  # 减少生成长度
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            # 解码输出
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            predicted_intent = full_response.split("助手: ")[-1].strip()
            
            # 清理和解析预测结果
            return self._parse_intent(predicted_intent)
            
        except Exception as e:
            print(f"⚠️ 模型预测失败: {e}")
            return "none"
    
    def _parse_intent(self, raw_output: str) -> str:
        """解析模型输出，提取标准化的意图"""
        if not raw_output:
            return "none"
        
        raw_output = raw_output.lower().strip()
        
        # 检查是否包含标准意图类别
        for intent in self.intent_categories:
            if intent in raw_output:
                return intent
        
        # 基于关键词的回退解析
        intent_keywords = {
            "order_management": ["订单", "order", "购买", "下单"],
            "user_auth": ["登录", "注册", "密码", "账户", "认证"],
            "payment": ["支付", "付款", "退款", "交易"],
            "inventory": ["库存", "商品", "数量", "缺货"],
            "notification": ["通知", "消息", "邮件", "推送"]
        }
        
        for intent, keywords in intent_keywords.items():
            if any(keyword in raw_output for keyword in keywords):
                return intent
        
        return "none"
    
    def predict_intent(self, text: str, threshold: float = 0.05) -> List[Tuple[str, float]]:
        """
        预测文本的意图
        保持与MockIntentClassifier相同的接口
        """
        # 如果使用Mock分类器
        if hasattr(self, 'use_mock') and self.use_mock:
            return self.mock_classifier.predict_intent(text, threshold)
        
        # 使用真实模型预测
        predicted_intent = self._predict_with_model(text)
        
        # 为了保持接口一致性，返回列表格式
        if predicted_intent and predicted_intent != "none":
            return [(predicted_intent, 0.9)]  # 给一个高置信度
        else:
            return [("none", 1.0)]
    
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

def test_real_intent_classifier():
    """测试真实意图识别器"""
    classifier = RealIntentClassifier()
    
    test_cases = [
        "我想查看我的订单状态",
        "怎么登录我的账户", 
        "支付失败了怎么办",
        "库存不够怎么处理",
        "怎么发送通知给用户",
        "今天天气怎么样"
    ]
    
    print("=== 真实意图识别器测试 ===\n")
    
    for text in test_cases:
        start_time = time.time()
        top_intent = classifier.predict_top_intent(text)
        predictions = classifier.predict_intent(text)
        inference_time = time.time() - start_time
        
        print(f"输入: {text}")
        print(f"预测意图: {top_intent}")
        print(f"推理时间: {inference_time:.3f}s")
        print(f"详细预测: {predictions}")
        print("-" * 50)

if __name__ == "__main__":
    test_real_intent_classifier()