import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Tuple, Dict
import pickle
import glob
from pathlib import Path

class DocumentRetriever:
    def __init__(self, docs_dir: str = "docs", model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.docs_dir = docs_dir
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.documents = []
        self.doc_chunks = []
        self.chunk_to_doc_mapping = []
        
    def load_documents(self):
        """加载所有Markdown文档"""
        print(f"正在加载文档目录: {self.docs_dir}")
        
        doc_files = glob.glob(os.path.join(self.docs_dir, "*.md"))
        if not doc_files:
            print(f"警告: 在目录 {self.docs_dir} 中没有找到.md文件")
            return
            
        for doc_file in doc_files:
            with open(doc_file, 'r', encoding='utf-8') as f:
                content = f.read()
                doc_name = os.path.basename(doc_file)
                self.documents.append({
                    'name': doc_name,
                    'path': doc_file,
                    'content': content
                })
        
        print(f"已加载 {len(self.documents)} 个文档")
    
    def chunk_documents(self, chunk_size: int = 500, overlap: int = 50):
        """将文档切分成小块"""
        self.doc_chunks = []
        self.chunk_to_doc_mapping = []
        
        for doc_idx, doc in enumerate(self.documents):
            content = doc['content']
            # 按段落分割
            paragraphs = content.split('\n\n')
            
            current_chunk = ""
            for paragraph in paragraphs:
                # 如果添加当前段落后会超过chunk_size，先保存当前chunk
                if len(current_chunk) + len(paragraph) > chunk_size and current_chunk:
                    self.doc_chunks.append(current_chunk.strip())
                    self.chunk_to_doc_mapping.append(doc_idx)
                    
                    # 保留overlap部分
                    if len(current_chunk) > overlap:
                        current_chunk = current_chunk[-overlap:] + "\n\n" + paragraph
                    else:
                        current_chunk = paragraph
                else:
                    if current_chunk:
                        current_chunk += "\n\n" + paragraph
                    else:
                        current_chunk = paragraph
            
            # 保存最后一个chunk
            if current_chunk:
                self.doc_chunks.append(current_chunk.strip())
                self.chunk_to_doc_mapping.append(doc_idx)
        
        print(f"文档切分完成，共生成 {len(self.doc_chunks)} 个文档块")
    
    def build_index(self):
        """构建向量索引"""
        if not self.doc_chunks:
            print("警告: 没有文档块可以索引")
            return
            
        print("正在计算文档向量...")
        # 计算所有文档块的向量
        embeddings = self.model.encode(self.doc_chunks, show_progress_bar=True)
        
        # 构建Faiss索引
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # 内积相似度
        
        # 标准化向量（用于余弦相似度）
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        
        print(f"向量索引构建完成，维度: {dimension}")
    
    def retrieve_docs(self, query_text: str, top_k: int = 3) -> List[Dict]:
        """检索最相关的文档块"""
        if self.index is None:
            print("错误: 向量索引未构建，请先调用build_index()")
            return []
        
        # 计算查询向量
        query_embedding = self.model.encode([query_text])
        faiss.normalize_L2(query_embedding)
        
        # 搜索最相似的文档块
        scores, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.doc_chunks):  # 确保索引有效
                doc_idx = self.chunk_to_doc_mapping[idx]
                doc_info = self.documents[doc_idx]
                
                results.append({
                    'rank': i + 1,
                    'score': float(score),
                    'doc_name': doc_info['name'],
                    'doc_path': doc_info['path'],
                    'chunk_content': self.doc_chunks[idx],
                    'full_doc_content': doc_info['content']
                })
        
        return results
    
    def save_index(self, index_path: str = "faiss_index"):
        """保存索引到文件"""
        os.makedirs(index_path, exist_ok=True)
        
        # 保存Faiss索引
        faiss.write_index(self.index, os.path.join(index_path, "index.faiss"))
        
        # 保存文档信息
        with open(os.path.join(index_path, "documents.pkl"), 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'doc_chunks': self.doc_chunks,
                'chunk_to_doc_mapping': self.chunk_to_doc_mapping
            }, f)
        
        print(f"索引已保存到: {index_path}")
    
    def load_index(self, index_path: str = "faiss_index"):
        """从文件加载索引"""
        try:
            # 加载Faiss索引
            self.index = faiss.read_index(os.path.join(index_path, "index.faiss"))
            
            # 加载文档信息
            with open(os.path.join(index_path, "documents.pkl"), 'rb') as f:
                data = pickle.load(f)
                self.documents = data['documents']
                self.doc_chunks = data['doc_chunks']
                self.chunk_to_doc_mapping = data['chunk_to_doc_mapping']
            
            print(f"索引已从 {index_path} 加载完成")
            return True
        except Exception as e:
            print(f"加载索引失败: {e}")
            return False
    
    def initialize(self):
        """初始化检索器"""
        # 尝试加载已有的索引
        if os.path.exists("faiss_index") and self.load_index():
            print("使用已存在的索引")
        else:
            print("构建新的索引")
            self.load_documents()
            if self.documents:
                self.chunk_documents()
                self.build_index()
                self.save_index()
            else:
                print("警告: 没有找到文档，无法构建索引")

def test_retriever():
    """测试检索器功能"""
    retriever = DocumentRetriever()
    retriever.initialize()
    
    # 测试查询
    test_queries = [
        "我想查看我的订单状态",
        "如何登录账户",
        "支付失败了怎么办",
        "库存不够怎么处理",
        "怎么发送通知给用户"
    ]
    
    for query in test_queries:
        print(f"\n查询: {query}")
        results = retriever.retrieve_docs(query, top_k=2)
        for result in results:
            print(f"  排名 {result['rank']}: {result['doc_name']} (相似度: {result['score']:.3f})")
            print(f"  内容预览: {result['chunk_content'][:100]}...")
            print()

if __name__ == "__main__":
    test_retriever()