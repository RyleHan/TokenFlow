from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from typing import Dict, Any, List
from contextlib import asynccontextmanager
from src.core.retriever import DocumentRetriever
from src.classifiers.hybrid_intent_classifier import HybridIntentClassifier
from src.core.large_llm_simulator import LargeLLMSimulator
import time

# 全局变量，用于存储各个组件实例
retriever = None
intent_classifier = None
large_llm = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时初始化所有组件
    global retriever, intent_classifier, large_llm
    print("正在初始化文档检索器...")
    retriever = DocumentRetriever()
    retriever.initialize()
    print("文档检索器初始化完成!")
    
    print("正在初始化混合意图分类器...")
    # 使用新的微调Qwen3-0.6B模型
    intent_classifier = HybridIntentClassifier(use_finetuned=True)
    print(f"混合意图分类器初始化完成! 权重配置: 规则={intent_classifier.rule_weight}, 神经网络={intent_classifier.neural_weight}")
    
    print("正在初始化大型LLM模拟器...")
    large_llm = LargeLLMSimulator()
    print("大型LLM模拟器初始化完成!")
    
    yield
    
    # 关闭时清理资源（如果需要）
    print("正在关闭应用...")

app = FastAPI(
    title="TokenFlow - Intelligent API Gateway",
    description="智能API网关，使用小型LLM进行意图识别和上下文路由",
    version="1.0.0",
    lifespan=lifespan
)

def _build_final_response(prompt: str, intents: List[str], docs: List) -> str:
    """构建最终响应"""
    if "none" in intents:
        return f"根据您的问题「{prompt}」，系统未能识别到相关的业务意图。已为您检索到 {len(docs)} 个可能相关的文档片段。"
    
    intent_desc = {
        "order_management": "订单管理",
        "user_auth": "用户认证",
        "payment": "支付处理", 
        "inventory": "库存管理",
        "notification": "通知服务"
    }
    
    intent_names = [intent_desc.get(intent, intent) for intent in intents]
    intent_text = "、".join(intent_names)
    
    return f"根据您的问题「{prompt}」，系统识别到的业务意图为：{intent_text}。已为您检索到 {len(docs)} 个相关的文档片段以提供上下文信息。"

def _calculate_token_savings(original_prompt: str, all_docs: int, filtered_docs: int, llm_tokens: int) -> Dict[str, Any]:
    """计算Token节省效果"""
    
    # 估算如果没有意图识别和文档过滤，需要使用的token数量
    # 假设完整文档库有20个文档，每个文档平均500个token
    full_doc_tokens = 20 * 500  # 10000 tokens
    
    # 当前实际使用的文档token（根据检索到的文档估算）
    current_doc_tokens = filtered_docs * 350  # 平均每个检索片段350 tokens
    
    # 系统提示和用户问题的token
    system_tokens = len(original_prompt) * 1.5  # 粗略估算
    
    # 对比计算
    without_routing_tokens = full_doc_tokens + system_tokens + llm_tokens
    with_routing_tokens = current_doc_tokens + system_tokens + llm_tokens
    
    savings_tokens = without_routing_tokens - with_routing_tokens
    savings_percentage = (savings_tokens / without_routing_tokens) * 100 if without_routing_tokens > 0 else 0
    
    return {
        "status": "Token计算完成",
        "without_routing": {
            "total_tokens": int(without_routing_tokens),
            "context_tokens": full_doc_tokens,
            "description": "未使用意图路由时的预估token使用量"
        },
        "with_routing": {
            "total_tokens": int(with_routing_tokens),
            "context_tokens": current_doc_tokens,
            "description": "使用意图路由后的实际token使用量"
        },
        "savings": {
            "tokens_saved": int(savings_tokens),
            "percentage_saved": round(savings_percentage, 2),
            "description": f"通过智能路由节省了{int(savings_tokens)}个token，节省率{savings_percentage:.1f}%"
        },
        "documents": {
            "total_available": 20,
            "retrieved_and_used": filtered_docs,
            "filtering_efficiency": round((filtered_docs / 20) * 100, 1)
        }
    }

class RouteRequest(BaseModel):
    prompt: str

class RetrievedDoc(BaseModel):
    doc_name: str
    score: float
    content: str

class LLMResponse(BaseModel):
    response_text: str
    suggested_actions: List[str] = []
    token_usage: Dict[str, int] = {}
    processing_time: float = 0.0
    metadata: Dict[str, Any] = {}

class RouteResponse(BaseModel):
    original_prompt: str
    relevant_docs: List[RetrievedDoc] = []
    predicted_intent: List[str] = []
    llm_response: LLMResponse = None
    final_response: str = ""
    token_savings: Dict[str, Any] = {}
    processing_time: float = 0.0

@app.get("/")
async def root():
    return {
        "message": "TokenFlow API Gateway",
        "description": "智能上下文路由器，减少大模型Token使用",
        "endpoints": ["/route", "/docs"]
    }

@app.post("/route", response_model=RouteResponse)
async def route_request(request: RouteRequest):
    """
    主要路由端点 - 接收用户请求，进行意图识别和文档检索
    """
    start_time = time.time()
    
    try:
        if retriever is None or intent_classifier is None or large_llm is None:
            raise HTTPException(status_code=500, detail="系统组件未初始化")
        
        # 1. 意图识别 - 支持多意图
        classification_result = intent_classifier.classify(request.prompt)
        predicted_intents = classification_result.get("intents", [classification_result.get("intent", "none")])
        
        # 2. 基于意图过滤和检索相关文档
        if "none" not in predicted_intents:
            # 构建意图相关的查询
            intent_enhanced_query = f"{request.prompt} {' '.join(predicted_intents)}"
            retrieved_results = retriever.retrieve_docs(intent_enhanced_query, top_k=3)
        else:
            # 如果无相关意图，使用原始查询
            retrieved_results = retriever.retrieve_docs(request.prompt, top_k=2)
        
        # 3. 转换为响应格式
        relevant_docs = []
        doc_list_for_llm = []
        for result in retrieved_results:
            relevant_docs.append(RetrievedDoc(
                doc_name=result['doc_name'],
                score=result['score'],
                content=result['chunk_content']
            ))
            doc_list_for_llm.append({
                'doc_name': result['doc_name'],
                'content': result['chunk_content']
            })
        
        # 4. 调用大型LLM生成最终响应
        llm_result = large_llm.call_large_llm(
            context_docs=doc_list_for_llm,
            user_prompt=request.prompt,
            predicted_intents=predicted_intents,
            simulate_processing=False  # 在生产环境中可以启用
        )
        
        # 5. 构建LLM响应对象
        llm_response = LLMResponse(
            response_text=llm_result['response'],
            suggested_actions=llm_result['suggested_actions'],
            token_usage=llm_result['token_usage'],
            processing_time=llm_result['processing_time'],
            metadata=llm_result['metadata']
        )
        
        processing_time = time.time() - start_time
        
        # 6. 计算Token节省效果
        token_savings = _calculate_token_savings(
            original_prompt=request.prompt,
            all_docs=len(retrieved_results),
            filtered_docs=len(relevant_docs),
            llm_tokens=llm_result['token_usage']['total_tokens']
        )
        
        response = RouteResponse(
            original_prompt=request.prompt,
            relevant_docs=relevant_docs,
            predicted_intent=predicted_intents,
            llm_response=llm_response,
            final_response=llm_result['response'],
            token_savings=token_savings,
            processing_time=processing_time
        )
        return response
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理请求时出错: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "TokenFlow API Gateway运行正常"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)