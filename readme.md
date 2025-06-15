# 论文搜索器

这是一个简单的应用，旨在帮助用户通过关键词快速查找学术论文，并且在每次搜索时会自动清空上次的结果，避免新旧内容混淆。

## 功能

- 输入关键词搜索相关论文。
- 自动清空上一次的搜索结果，保证界面清晰。
- 显示每篇论文的基本信息：标题、摘要及链接。

## 技术栈

- **Streamlit**：用于构建Web应用程序的框架。
- **Requests**：用于发送HTTP请求，与arXiv API交互。
- **Feedparser**：解析arXiv返回的RSS格式数据。

## 安装指南

请确保已安装Python 3.8或更高版本。

1. 克隆此仓库到本地：
   ```
   git clone git@github.com:Ylr9933/Literature-Review.git
   ```

2. 创建虚拟环境（conda）：
   ```
    conda create -n python_for_lr python=3.8 -y
    conda activate python_for_lr
   ```

3. 安装所需的Python包：
   ```
   pip install -r requirements.txt
   ```
   
4. 运行应用：
   ```
   streamlit run main.py
   ```

打开浏览器并访问 `http://localhost:8501` 即可开始使用本应用（国内需科学上网）

## 使用方法

1. 在首页输入你感兴趣的关键词，例如“Remote sensing data retrieval”。
2. 点击“搜索论文”按钮。
3. 查看搜索结果，点击论文标题下的链接可以跳转至详细页面。

## 贡献者指南

我们欢迎任何形式的贡献！无论是代码改进、文档更新还是bug报告，都非常感谢您的参与。

如果您希望贡献代码，请先fork此仓库，在本地完成修改后提交pull request。
