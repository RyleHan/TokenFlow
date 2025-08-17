#!/usr/bin/env python3
"""
测试集成了真实意图识别模型的TokenFlow系统
"""
import asyncio
import sys
sys.path.append('.')

from main import *
import time

async def test_tokenflow_with_real_model():
    """测试使用真实模型的TokenFlow系统"""
    print("🧪 测试集成真实模型的TokenFlow系统")
    print("=" * 60)
    
    # 初始化组件
    print("正在初始化系统组件...")
    await startup_event()
    
    # 测试用例
    test_cases = [
        "我想查看我的订单状态",
        "怎么登录账户", 
        "支付遇到问题了",
        "库存管理相关问题",
        "发送消息通知",
        "今天天气怎么样"
    ]
    
    print("\n开始系统测试:")
    print("-" * 60)
    
    total_time = 0
    for i, prompt in enumerate(test_cases, 1):
        print(f"\n测试 {i}: {prompt}")
        
        # 创建请求
        request = RouteRequest(prompt=prompt)
        
        # 处理请求
        start_time = time.time()
        try:
            result = await route_request(request)
            processing_time = time.time() - start_time
            total_time += processing_time
            
            print(f"✅ 处理成功")
            print(f"   识别意图: {result.predicted_intent}")
            print(f"   检索文档: {len(result.relevant_docs)}个")
            print(f"   处理时间: {processing_time:.3f}s")
            print(f"   Token节省: {result.token_savings['savings']['percentage_saved']}%")
            
            # 打印部分响应
            response_preview = result.final_response[:80] + "..." if len(result.final_response) > 80 else result.final_response
            print(f"   系统响应: {response_preview}")
            
        except Exception as e:
            print(f"❌ 处理失败: {e}")
        
        print("-" * 40)
    
    # 统计总结
    avg_time = total_time / len(test_cases)
    print(f"\n📊 测试完成统计:")
    print(f"   测试用例数: {len(test_cases)}")
    print(f"   总处理时间: {total_time:.3f}s")
    print(f"   平均处理时间: {avg_time:.3f}s")
    print(f"   真实模型状态: {'✅ 运行中' if hasattr(intent_classifier, 'model') and intent_classifier.model else '⚠️ 备用Mock'}")
    
    print(f"\n🎉 TokenFlow系统集成测试完成!")

if __name__ == "__main__":
    asyncio.run(test_tokenflow_with_real_model())