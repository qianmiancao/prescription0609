# --- 1. 核心兼容性补丁 (必须处于最顶部，且顺序不能错) ---
import sys

try:
    # 强制使用 pysqlite3 代替标准库中的 sqlite3，这是解决 InternalError 的关键
    import pysqlite3
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import streamlit as st
import os
import json
import warnings
import tempfile
import pandas as pd
import re 
import shutil
from datetime import datetime
from io import BytesIO
from docx import Document 

# --- 2. 基础环境配置 ---
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# 数据库存储路径 (Streamlit Cloud 建议使用相对路径)
DB_PATH = "./drug_db_v2" 

# --- 3. 逻辑类定义 ---

class KnowledgeManager:
    def __init__(self, model_name, db_path):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name, 
            model_kwargs={'device': 'cpu'}
        )
        self.db_path = db_path
        self.init_db()

    def init_db(self):
        """初始化数据库，如果损坏则自动重建"""
        try:
            self.vectorstore = Chroma(
                persist_directory=self.db_path, 
                embedding_function=self.embeddings
            )
        except Exception as e:
            # 如果出现 InternalError，通常是索引损坏，直接删除重建
            if os.path.exists(self.db_path):
                shutil.rmtree(self.db_path)
            self.vectorstore = Chroma(
                persist_directory=self.db_path, 
                embedding_function=self.embeddings
            )

    def upload_docs(self, file_path, original_name):
        try:
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            for doc in docs:
                doc.metadata = {"source": original_name} 
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            splits = splitter.split_documents(docs)
            self.vectorstore.add_documents(splits)
            return len(splits)
        except Exception as e:
            st.error(f"解析文件 {original_name} 失败: {str(e)}")
            return 0

    def retrieve_context_with_sources(self, drug_names):
        results_list = []
        unique_sources = set()
        for name in drug_names:
            if not name or len(name.strip()) < 1: continue
            # 使用相似度搜索
            try:
                docs = self.vectorstore.similarity_search(name, k=3)
                for d in docs:
                    src = os.path.basename(d.metadata.get('source', '未知'))
                    if name[:2] in d.page_content or name[:2] in src: # 关键词二次校验
                        results_list.append(f"【参考《{src}》】: {d.page_content}")
                        unique_sources.add(src)
            except:
                continue
        return "\n\n".join(results_list), list(unique_sources)

# --- 4. 其他辅助函数 ---
def clean_markdown(text):
    return text.replace('*', '').strip()

def generate_docx(text):
    doc = Document()
    doc.add_heading('处方点评报告', 0)
    for line in text.split('\n'):
        if line.strip(): doc.add_paragraph(line.strip())
    bio = BytesIO()
    doc.save(bio)
    bio.seek(0)
    return bio

# --- 5. Streamlit UI ---
def main():
    st.set_page_config(page_title="AI 药师专家系统", layout="wide")
    
    # 初始化状态
    if "report_content" not in st.session_state: st.session_state.report_content = ""
    if "current_sources" not in st.session_state: st.session_state.current_sources = []

    # 缓存管理
    @st.cache_resource
    def load_km():
        return KnowledgeManager("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", DB_PATH)
    
    km = load_km()

    # 侧边栏
    with st.sidebar:
        st.header("⚙️ 知识库管理")
        api_key = st.text_input("DeepSeek API Key:", type="password")
        files = st.file_uploader("上传 PDF 说明书", accept_multiple_files=True)
        if files and st.button("✨ 同步到本地库"):
            with st.spinner("正在写入索引..."):
                for f in files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(f.getvalue())
                        km.upload_docs(tmp.name, f.name)
                st.success("同步完成")
        
        if st.button("🗑️ 清空并重置数据库"):
            if os.path.exists(DB_PATH):
                shutil.rmtree(DB_PATH)
            st.rerun()

    # 主界面
    st.title("🏥 临床药师点评专家平台")
    
    col_in, col_out = st.columns([1, 1.3])

    with col_in:
        st.subheader("📋 处方录入")
        p_type = st.radio("患者类型", ["儿童", "成人"], horizontal=True)
        c1, c2 = st.columns(2)
        age = c1.number_input("年龄", 6)
        weight = c2.number_input("体重 (kg)", 20.0) if p_type == "儿童" else None
        diag = st.text_input("诊断", "急性支气管炎")
        
        med_df = st.data_editor(
            pd.DataFrame([{"药品名称": "阿莫西林", "单次剂量": "0.25g", "频次": "QD"}]), 
            num_rows="dynamic", use_container_width=True
        )

        if st.button("🔍 执行深度点评", type="primary", use_container_width=True):
            if not api_key: st.error("请提供 API Key"); return
            
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(model="deepseek-chat", api_key=api_key, base_url="https://api.deepseek.com", temperature=0.1)
            
            with st.spinner("正在精准匹配说明书..."):
                context, sources = km.retrieve_context_with_sources(med_df["药品名称"].tolist())
                st.session_state.current_sources = sources
                
                prompt = f"你是一位资深药师。根据资料：{context}\n审核处方：{med_df.to_dict('records')}\n诊断：{diag}\n患者：{age}岁, {weight}kg\n要求：不使用星号，纯文本格式。"
                res = llm.invoke(prompt)
                st.session_state.report_content = clean_markdown(res.content)
                st.rerun()

    with col_out:
        st.subheader("📝 点评报告")
        if st.session_state.current_sources:
            st.info(f"参考文件: {', '.join(st.session_state.current_sources)}")
        
        if st.session_state.report_content:
            report = st.text_area("内容:", value=st.session_state.report_content, height=500)
            if st.download_button("📥 导出 Word", data=generate_docx(report), file_name="报告.docx"):
                st.success("导出成功")

if __name__ == "__main__":
    main()
