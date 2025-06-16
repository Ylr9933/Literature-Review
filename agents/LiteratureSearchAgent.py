import arxiv
import requests
import streamlit as st
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