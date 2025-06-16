
import io
import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
# 创建海报生成器
class PosterGenerator:
    def __init__(self,font_path):
        if os.path.exists(font_path):
            pdfmetrics.registerFont(TTFont('SimSun', font_path))
        else:
            raise FileNotFoundError(f"字体文件未找到: {font_path}")
    def generate_poster(self,summary, graph_image, papers):
        """生成包含综述和演进图的海报"""
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