#!/usr/bin/env python3
"""
TokenFlow 服务启动脚本
"""

import os
import sys

# 设置环境变量避免MPS相关问题
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

try:
    # 导入torch并设置CPU模式
    import torch
    torch.set_num_threads(1)
    
    # 禁用MPS后端
    if hasattr(torch.backends, 'mps'):
        torch.backends.mps.is_available = lambda: False
    
    print("✅ PyTorch环境配置完成")
    
    # 测试组件初始化
    from src.classifiers.hybrid_intent_classifier import HybridIntentClassifier
    
    print("🧪 测试混合意图分类器初始化...")
    classifier = HybridIntentClassifier("./models/intent_classifier_model_m3")
    
    print("🧪 测试简单推理...")
    result = classifier.predict_intent("我想查看我的订单")
    print(f"测试结果: {result}")
    
    print("✅ 组件测试成功，启动主应用...")
    
    # 如果测试成功，启动主应用
    from main import app
    import uvicorn
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
    
except Exception as e:
    print(f"❌ 启动失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)