# main.py

import os
os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"
import streamlit as st
from utils import Logger,PosterGenerator
from agents import LiteratureSearchAgent,LiteratureAnalysisAgent
import datetime
def main():
    # 初始化日志系统
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_fname = f"streamlit_app_{timestamp}.log"
    logger = Logger(save_dir="logs", fname=log_fname)
    logger.log("【启动】Streamlit 应用开始运行")

    # 初始化Agent
    search_agent = LiteratureSearchAgent()
    analysis_agent = LiteratureAnalysisAgent()

    #初始化海报生成器
    poster_generator = PosterGenerator(font_path="fonts/simsun.ttc")

    st.set_page_config(page_title="文献综述Agent", page_icon="📚", layout="wide")
    st.title("📚 文献综述Agent工具")

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
                
                # 确保已经处理过papers并创建了vectorstore
                if analysis_agent.vectorstore is None:
                    analysis_agent.process_papers(st.session_state.papers)
                
                # 调用generate_summary时传入papers和query(topic)
                summary = analysis_agent.generate_summary(papers=st.session_state.papers, query=topic)
                
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