from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import dashscope
from dashscope import Generation
import matplotlib.pyplot as plt
import networkx as nx
from datetime import datetime
import io

# 设置阿里云API密钥
dashscope.api_key = "sk-ceff4d200590488f98d96d3b95e915f0"  # 替换为你自己的 DashScope API Key
def call_qwen(prompt):
    response = Generation.call(
        model="qwen-max",  # 可以换成 qwen-plus 或 qwen-turbo
        prompt=prompt
    )
    return response.output.text

# 创建文献分析Agent
class LiteratureAnalysisAgent:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="shibing624/text2vec-base-chinese"
        )
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.vectorstore = None  # 初始化为空，在process_papers中填充

    def process_papers(self, papers):
        """处理论文数据并创建向量数据库"""
        documents = []
        
        for paper in papers:
            content = f"Title: {paper['title']}\nAuthors: {paper['authors']}\nPublished: {paper['published']}\nSummary: {paper['summary']}"
            chunks = self.text_splitter.split_text(content)
            documents.extend(chunks)
        
        self.vectorstore = FAISS.from_texts(documents, self.embeddings)  # 存储vectorstore实例
        
    def generate_summary(self, papers, query):
        """生成文献综述"""
        if not self.vectorstore:
            raise ValueError("请先调用process_papers方法初始化向量数据库")
        
        # 检索与查询最相关的摘要
        query_embedding = self.embeddings.embed_query(query)
        relevant_docs = self.vectorstore.similarity_search_by_vector(query_embedding, k=5)  # 可根据需要调整k值
        relevant_context = "\n".join([doc.page_content for doc in relevant_docs])
        
        prompt_template = """
        你是一个专业的科研助手，需要根据以下论文信息和提供的上下文撰写文献综述：
        
        上下文：
        {relevant_context}
        
        论文信息：
        {papers}
        
        请按照以下结构撰写综述：
        1. 主要研究领域概述
        2. 关键创新点分析
        3. 使用的数据集和实验结果
        4. 不同论文之间的联系和发展
        
        综述内容应准确、专业，避免产生幻觉或错误信息。
        """
        
        papers_str = ""
        for i, paper in enumerate(papers, 1):
            papers_str += f"{i}. {paper['title']} ({paper['published']})\n"
            papers_str += f"   摘要: {paper['summary']}\n\n"
        
        prompt = PromptTemplate(template=prompt_template, input_variables=["relevant_context", "papers"])
        final_prompt = prompt.format(relevant_context=relevant_context, papers=papers_str)
        
        summary = call_qwen(final_prompt)
        return summary
    
    def create_evolution_graph(self, papers):
        """创建算法发展演进图"""
        G = nx.DiGraph()
        
        # 提取论文年份和标题
        paper_years = []
        for paper in papers:
            year = datetime.strptime(paper['published'], "%Y-%m-%d").year
            paper_years.append((year, paper['title']))
        
        # 按时间排序
        paper_years.sort()
        
        # 创建节点和边
        for year, title in paper_years:
            G.add_node(title, year=year)
        
        nodes = list(G.nodes)
        for i in range(len(nodes) - 1):
            G.add_edge(nodes[i], nodes[i + 1])
        
        # 绘制图表
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G, seed=42)
        nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=3000, font_size=10, font_weight='bold')
        nx.draw_networkx_edges(G, pos, arrows=True)
        
        edge_labels = {(u, v): d['weight'] if 'weight' in d else '' for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
        
        plt.title("算法发展演进图")
        plt.axis('off')
        
        # 保存为字节流
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=300, bbox_inches="tight")
        plt.close()
        buf.seek(0)
        
        return buf.getvalue()