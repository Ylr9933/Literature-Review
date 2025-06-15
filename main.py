# main.py

import os
os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"
import arxiv
import requests
import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import dashscope
from dashscope import Generation
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from datetime import datetime
import re
import PyPDF2
import io
import base64
from utils.logger import Logger
# 设置阿里云API密钥
dashscope.api_key = "sk-ceff4d200590488f98d96d3b95e915f0"  # 替换为你自己的 DashScope API Key
def call_qwen(prompt):
    response = Generation.call(
        model="qwen-max",  # 可以换成 qwen-plus 或 qwen-turbo
        prompt=prompt
    )
    return response.output.text

# 创建文献检索Agent
class LiteratureSearchAgent:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
    
    def search_arxiv(self, topic, max_results=10):
        """搜索arXiv论文"""
        client = arxiv.Client()
        search = arxiv.Search(
            query=topic,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate
        )
        
        papers = []
        for result in client.results(search):
            paper_info = {
                "title": result.title,
                "authors": ", ".join(author.name for author in result.authors),
                "summary": result.summary,
                "url": result.pdf_url,
                "published": result.published.strftime("%Y-%m-%d"),
                "code_link": None
            }
            papers.append(paper_info)
        
        return papers
    
    def find_github_code(self, topic, num_repos=5):
        """在GitHub上搜索相关代码"""
        url = f"https://api.github.com/search/repositories?q={topic}&sort=stars&order=desc"
        response = requests.get(url, headers=self.headers)
        
        if response.status_code == 200:
            repos = response.json().get('items', [])
            code_links = []
            
            for repo in repos[:num_repos]:
                code_info = {
                    "name": repo['full_name'],
                    "description": repo['description'] or "",
                    "url": repo['html_url'],
                    "stars": repo['stargazers_count']
                }
                code_links.append(code_info)
            
            return code_links
        else:
            st.error(f"GitHub搜索失败: {response.status_code}")
            return []

# 创建文献分析Agent
class LiteratureAnalysisAgent:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="shibing624/text2vec-base-chinese"
        )
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        
    def process_papers(self, papers):
        """处理论文数据并创建向量数据库"""
        documents = []
        
        for paper in papers:
            content = f"Title: {paper['title']}\nAuthors: {paper['authors']}\nPublished: {paper['published']}\nSummary: {paper['summary']}"
            chunks = self.text_splitter.split_text(content)
            documents.extend(chunks)
        
        vectorstore = FAISS.from_texts(documents, self.embeddings)
        return vectorstore
    
    def generate_summary(self, papers):
        """生成文献综述"""
        prompt_template = """
        你是一个专业的科研助手，需要根据以下论文信息撰写文献综述：
        
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
        
        prompt = PromptTemplate(template=prompt_template, input_variables=["papers"])
        final_prompt = prompt.format(papers=papers_str)
        
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

# 创建海报生成器
class PosterGenerator:
    def generate_poster(self, summary, graph_image, papers):
        """生成包含综述和演进图的海报"""
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
        # 注册中文字体
        font_path = os.path.join(os.path.dirname(__file__), "fonts", "simsun.ttc")
        if os.path.exists(font_path):
            pdfmetrics.registerFont(TTFont('SimSun', font_path))
        else:
            raise FileNotFoundError(f"字体文件未找到: {font_path}")

        # 创建PDF文档
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        
        styles = getSampleStyleSheet()
        
        # 自定义样式（使用中文字体）
        styles.add(ParagraphStyle(name='Chinese', fontName='SimSun', fontSize=12, leading=14))
        styles.add(ParagraphStyle(name='ChineseTitle', fontName='SimSun', fontSize=18, leading=22, alignment=TA_CENTER))
        styles.add(ParagraphStyle(name='ChineseHeading2', fontName='SimSun', fontSize=14, leading=16, bold=True))

        story = []

        # 添加标题
        story.append(Paragraph("文献综述海报", styles['ChineseTitle']))
        story.append(Spacer(1, 24))

        # 添加演进图
        story.append(Paragraph("算法发展演进图", styles['ChineseHeading2']))
        img = io.BytesIO(graph_image)
        image = Image(img)
        image.drawHeight = 400
        image.drawWidth = 600
        story.append(image)
        story.append(Spacer(1, 12))

        # 添加综述内容
        story.append(Paragraph("文献综述", styles['ChineseHeading2']))
        paragraphs = summary.split('\n\n')
        for para in paragraphs:
            # 使用中文样式
            story.append(Paragraph(para, styles['Chinese']))
            story.append(Spacer(1, 6))

        # 添加参考文献
        story.append(Paragraph("参考文献", styles['ChineseHeading2']))
        for i, paper in enumerate(papers, 1):
            ref_text = f"[{i}] {paper['title']} ({paper['published']})"
            story.append(Paragraph(ref_text, styles['Chinese']))

        # 构建文档
        doc.build(story)

        return buffer.getvalue()
# 初始化日志系统
logger = Logger(save_dir="logs", fname="streamlit_app.log")

import streamlit as st
from utils.logger import Logger

logger = Logger(save_dir="logs", fname="streamlit_app.log")
def main():
    logger.log("【启动】Streamlit 应用开始运行")

    st.set_page_config(page_title="文献综述Agent", page_icon="📚", layout="wide")
    st.title("📚 文献综述Agent工具")
    # 初始化Agent
    search_agent = LiteratureSearchAgent()
    analysis_agent = LiteratureAnalysisAgent()
    poster_generator = PosterGenerator()
    
    # 初始化 session_state
    if 'papers' not in st.session_state:
        st.session_state.papers = None
    if 'summary' not in st.session_state:
        st.session_state.summary = None
    if 'evolution_graph' not in st.session_state:
        st.session_state.evolution_graph = None
    if 'poster_pdf' not in st.session_state:
        st.session_state.poster_pdf = None

    # 用户输入
    col1, col2 = st.columns([3, 1])
    with col1:
        topic = st.text_input("请输入要搜索的文献主题:")
    with col2:
        num_papers = st.slider("论文数量", min_value=5, max_value=20, value=10)

    if st.button("开始搜索与分析"):
        logger.log(f"【用户操作】点击 '开始搜索与分析'，主题为 '{topic}'，数量 {num_papers}")

        if not topic:
            st.warning("请输入一个文献主题!")
            return

        try:
            with st.spinner("正在从arXiv获取最新论文..."):
                logger.log("【阶段】开始从 arXiv 获取论文")
                papers = search_agent.search_arxiv(topic, num_papers)
                st.session_state.papers = papers
                logger.log(f"【完成】成功获取 {len(papers)} 篇论文")
                st.subheader("找到的论文:")
                for paper in papers:
                    st.markdown(f"**{paper['title']}** ({paper['published']})")
                    st.markdown(f"*摘要:* {paper['summary']}")
                    st.markdown(f"[PDF链接]({paper['url']})")
                    st.markdown("---")
        except Exception as e:
            logger.error(f"【错误】搜索论文失败: {e}")
            st.error("论文搜索失败，请检查网络或API配置")
            return
                # 显示论文信息

        try:
            with st.spinner("正在生成文献综述..."):
                logger.log("【阶段】开始生成文献综述")
                summary = analysis_agent.generate_summary(papers)
                st.session_state.summary = summary
                logger.log("【完成】文献综述生成完成")
        except Exception as e:
            logger.error(f"【错误】生成综述失败: {e}")
            st.error("文献综述生成失败")
            return

        try:
            with st.spinner("正在创建算法发展演进图..."):
                logger.log("【阶段】开始绘制算法演进图")
                evolution_graph = analysis_agent.create_evolution_graph(papers)
                st.session_state.evolution_graph = evolution_graph
                logger.log("【完成】算法演进图绘制完成")
        except Exception as e:
            logger.error(f"【错误】绘图失败: {e}")
            st.error("算法演进图绘制失败")
            return

        try:
            with st.spinner("正在生成海报..."):
                logger.log("【阶段】开始生成PDF海报")
                poster_pdf = poster_generator.generate_poster(summary, evolution_graph, papers)
                st.session_state.poster_pdf = poster_pdf
                logger.log("【完成】PDF海报生成完成")
        except Exception as e:
            logger.error(f"【错误】生成海报失败: {e}")
            st.error("海报生成失败")
            return

    # 展示结果（无论是否刚执行完搜索还是从 session_state 中恢复）
    if st.session_state.summary:
        st.subheader("生成的文献综述:")
        st.write(st.session_state.summary)

    if st.session_state.evolution_graph:
        st.subheader("算法发展演进图")
        st.image(st.session_state.evolution_graph, use_container_width=True)

    if st.session_state.poster_pdf:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.download_button(
                label="下载海报",
                data=st.session_state.poster_pdf,
                file_name=f"literature_review_{topic.replace(' ', '_')}.pdf",
                mime="application/pdf"
            )
        with col2:
            st.download_button(
                label="下载综述文本",
                data=st.session_state.summary,
                file_name=f"summary_{topic.replace(' ', '_')}.txt",
                mime="text/plain"
            )
        with col3:
            import pandas as pd
            df = pd.DataFrame(st.session_state.papers)
            csv = df.to_csv(index=False)
            st.download_button(
                label="下载论文列表",
                data=csv,
                file_name=f"papers_{topic.replace(' ', '_')}.csv",
                mime="text/csv"
            )

    logger.log("【结束】当前任务已完成")

if __name__ == "__main__":
    main()