import random
import time
from typing import List, Dict, Any

class LargeLLMSimulator:
    """
    大型LLM调用模拟器
    在实际应用中，这里应该是调用真正的大型LLM API（如GPT-4、Claude等）
    """
    
    def __init__(self):
        # 模拟不同意图的响应模板
        self.response_templates = {
            "order_management": {
                "templates": [
                    "根据您的订单查询需求，我可以帮您：\n1. 查询订单状态和物流信息\n2. 修改订单信息（如收货地址）\n3. 取消订单并处理退款\n\n请提供您的订单号，我将为您查询详细信息。",
                    "关于订单管理，系统支持以下操作：\n- 创建新订单\n- 查询订单状态\n- 更新订单信息\n- 取消订单\n\n请告诉我您具体需要什么帮助，我会引导您完成相应操作。",
                    "我理解您想要查看订单相关信息。基于系统功能，我可以协助您：\n• 实时查询订单状态\n• 获取物流追踪信息\n• 处理订单变更请求\n\n如需查询特定订单，请提供订单ID。"
                ],
                "action_suggestions": [
                    "调用 get_order_status API 查询订单",
                    "使用 create_order 创建新订单",
                    "通过 cancel_order 取消订单"
                ]
            },
            "user_auth": {
                "templates": [
                    "关于用户认证，我可以协助您：\n1. 账户登录问题排查\n2. 密码重置流程指导\n3. 账户安全设置建议\n\n请描述您遇到的具体问题，我将提供相应解决方案。",
                    "用户认证系统支持：\n- 用户登录/注册\n- 密码管理\n- 令牌验证\n- 会话管理\n\n请告诉我您需要哪方面的帮助。",
                    "我了解您的认证需求。系统提供安全的身份验证机制：\n• JWT令牌认证\n• 多重身份验证\n• 密码加密存储\n\n请说明您的具体需求。"
                ],
                "action_suggestions": [
                    "调用 login API 进行用户登录",
                    "使用 register 创建新用户账户", 
                    "通过 reset_password 重置密码"
                ]
            },
            "payment": {
                "templates": [
                    "关于支付问题，我可以帮助您：\n1. 诊断支付失败原因\n2. 重新发起支付流程\n3. 申请退款处理\n\n请提供支付ID或订单号，我将查询支付状态并提供解决方案。",
                    "支付系统功能包括：\n- 创建支付订单\n- 查询支付状态\n- 处理退款请求\n- 管理支付方式\n\n请描述您遇到的具体支付问题。",
                    "我理解您的支付困扰。系统支持多种支付方式并提供安全保障：\n• 多种支付渠道\n• 实时状态查询\n• 自动退款处理\n\n请告诉我具体的支付问题。"
                ],
                "action_suggestions": [
                    "调用 create_payment 创建支付订单",
                    "使用 get_payment_status 查询支付状态",
                    "通过 process_refund 处理退款"
                ]
            },
            "inventory": {
                "templates": [
                    "关于库存管理，我可以协助您：\n1. 查询商品库存信息\n2. 处理缺货补货流程\n3. 库存预留和释放操作\n\n请提供商品ID，我将查询当前库存状况。",
                    "库存管理系统提供：\n- 实时库存查询\n- 库存预留机制\n- 仓库间调拨\n- 库存预警提醒\n\n请说明您的具体需求。",
                    "我了解您的库存需求。系统支持：\n• 多仓库库存管理\n• 智能补货建议\n• 库存状态实时监控\n\n请提供更多详细信息。"
                ],
                "action_suggestions": [
                    "调用 check_inventory 查询库存",
                    "使用 reserve_inventory 预留库存",
                    "通过 add_inventory 增加库存"
                ]
            },
            "notification": {
                "templates": [
                    "关于通知服务，我可以帮您：\n1. 设置消息推送偏好\n2. 发送各类通知消息\n3. 查询消息发送状态\n\n请说明您需要发送什么类型的通知。",
                    "通知系统支持多渠道消息推送：\n- 邮件通知\n- 短信提醒\n- 应用推送\n- Webhook回调\n\n请描述您的通知需求。",
                    "我理解您的通知需求。系统提供：\n• 多渠道消息发送\n• 消息模板管理\n• 发送状态跟踪\n\n请告诉我具体的通知场景。"
                ],
                "action_suggestions": [
                    "调用 send_email 发送邮件通知",
                    "使用 send_sms 发送短信",
                    "通过 send_push_notification 推送消息"
                ]
            },
            "none": {
                "templates": [
                    "抱歉，我无法识别您的具体业务需求。系统主要支持以下服务：\n• 订单管理\n• 用户认证\n• 支付处理\n• 库存管理\n• 通知服务\n\n请重新描述您的问题，我会尽力帮助您。",
                    "很抱歉，您的问题超出了我当前的服务范围。我主要协助处理：\n- 电商业务相关问题\n- 用户账户和认证\n- 支付和订单管理\n\n如果您有相关问题，请重新提问。",
                    "我暂时无法理解您的具体需求。为了更好地帮助您，请描述与以下服务相关的问题：\n1. 订单和物流\n2. 用户登录认证\n3. 支付和退款\n4. 商品库存\n5. 消息通知"
                ],
                "action_suggestions": [
                    "建议用户重新描述问题",
                    "引导用户使用系统支持的功能",
                    "提供帮助文档链接"
                ]
            }
        }
    
    def _calculate_tokens(self, text: str) -> int:
        """简单的token计算（实际应用中应使用对应模型的tokenizer）"""
        # 粗略估算：中文1字符≈1.5token，英文1单词≈1.3token
        chinese_chars = len([c for c in text if '\u4e00' <= c <= '\u9fff'])
        english_words = len([w for w in text.split() if any(c.isalpha() for c in w)])
        other_chars = len(text) - chinese_chars
        
        estimated_tokens = int(chinese_chars * 1.5 + english_words * 1.3 + other_chars * 0.5)
        return max(estimated_tokens, 1)
    
    def call_large_llm(self, 
                      context_docs: List[Dict], 
                      user_prompt: str, 
                      predicted_intents: List[str],
                      simulate_processing: bool = True) -> Dict[str, Any]:
        """
        模拟调用大型LLM
        
        Args:
            context_docs: 检索到的相关文档
            user_prompt: 用户原始问题
            predicted_intents: 预测的意图列表
            simulate_processing: 是否模拟处理延迟
            
        Returns:
            包含响应和元数据的字典
        """
        start_time = time.time()
        
        # 模拟处理延迟
        if simulate_processing:
            processing_delay = random.uniform(0.5, 2.0)  # 0.5-2秒延迟
            time.sleep(processing_delay)
        
        # 构建上下文
        context_text = self._build_context(context_docs)
        
        # 选择主要意图
        main_intent = predicted_intents[0] if predicted_intents else "none"
        
        # 生成响应
        response_text = self._generate_response(main_intent, user_prompt, context_text)
        
        # 生成建议的API调用
        suggested_actions = self._get_suggested_actions(main_intent)
        
        processing_time = time.time() - start_time
        
        # 计算token使用情况
        system_prompt = self._build_system_prompt()
        full_input = f"{system_prompt}\n\n用户问题: {user_prompt}\n\n相关文档:\n{context_text}"
        
        input_tokens = self._calculate_tokens(full_input)
        output_tokens = self._calculate_tokens(response_text)
        total_tokens = input_tokens + output_tokens
        
        return {
            "response": response_text,
            "suggested_actions": suggested_actions,
            "processing_time": processing_time,
            "token_usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens
            },
            "context_used": len(context_docs),
            "main_intent": main_intent,
            "metadata": {
                "model_type": "large_llm_simulation",
                "context_length": len(context_text),
                "response_confidence": random.uniform(0.7, 0.95)
            }
        }
    
    def _build_system_prompt(self) -> str:
        """构建系统提示"""
        return """你是一个专业的API网关助手，专门处理电商平台的各种业务请求。你可以访问以下服务：

1. 订单管理服务 - 处理订单创建、查询、更新、取消等操作
2. 用户认证服务 - 处理用户登录、注册、密码管理等
3. 支付处理服务 - 处理支付创建、查询、退款等操作
4. 库存管理服务 - 处理库存查询、预留、补货等操作  
5. 通知服务 - 处理邮件、短信、推送通知等

请根据用户的问题和提供的相关文档，给出准确、有用的回答，并建议合适的API调用。"""
    
    def _build_context(self, context_docs: List[Dict]) -> str:
        """构建文档上下文"""
        if not context_docs:
            return "没有找到相关文档"
        
        context_parts = []
        for i, doc in enumerate(context_docs, 1):
            doc_name = doc.get('doc_name', 'Unknown')
            content = doc.get('content', '')
            context_parts.append(f"文档{i}: {doc_name}\n{content}\n")
        
        return "\n".join(context_parts)
    
    def _generate_response(self, intent: str, user_prompt: str, context: str) -> str:
        """生成响应文本"""
        templates = self.response_templates.get(intent, self.response_templates["none"])["templates"]
        base_response = random.choice(templates)
        
        # 根据上下文长度和用户问题调整响应
        if len(context) > 1000:  # 上下文较长
            context_note = "\n\n基于系统文档，我为您提供了详细的功能说明。"
        else:
            context_note = "\n\n如需更详细的信息，请告诉我您的具体需求。"
        
        return base_response + context_note
    
    def _get_suggested_actions(self, intent: str) -> List[str]:
        """获取建议的API调用"""
        return self.response_templates.get(intent, self.response_templates["none"])["action_suggestions"]

def test_large_llm_simulator():
    """测试大型LLM模拟器"""
    simulator = LargeLLMSimulator()
    
    # 模拟测试用例
    test_cases = [
        {
            "prompt": "我想查看订单状态",
            "intents": ["order_management"],
            "docs": [
                {"doc_name": "Server_OrderManagement.md", "content": "订单管理服务提供查询功能..."}
            ]
        },
        {
            "prompt": "支付失败了怎么办",
            "intents": ["payment"],  
            "docs": [
                {"doc_name": "Server_Payment.md", "content": "支付服务提供退款功能..."}
            ]
        },
        {
            "prompt": "今天天气怎么样",
            "intents": ["none"],
            "docs": []
        }
    ]
    
    print("=== 大型LLM模拟器测试 ===\n")
    
    for i, case in enumerate(test_cases, 1):
        print(f"测试案例 {i}: {case['prompt']}")
        print(f"预测意图: {case['intents']}")
        
        result = simulator.call_large_llm(
            context_docs=case['docs'],
            user_prompt=case['prompt'],
            predicted_intents=case['intents'],
            simulate_processing=False  # 测试时不延迟
        )
        
        print(f"响应: {result['response']}")
        print(f"建议操作: {result['suggested_actions']}")
        print(f"Token使用: {result['token_usage']}")
        print(f"处理时间: {result['processing_time']:.3f}秒")
        print("-" * 60)

if __name__ == "__main__":
    test_large_llm_simulator()