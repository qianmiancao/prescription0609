# --- 1. 核心兼容性补丁 ---
try:
    import pysqlite3
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import streamlit as st
import os
import json
import warnings
import tempfile
import pandas as pd
from datetime import datetime
from io import BytesIO
from docx import Document 
from docx.shared import Pt

# --- 2. 基础环境配置 ---
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

DB_PATH = "./drug_db"
if not os.path.exists(DB_PATH):
    os.makedirs(DB_PATH, exist_ok=True)

# --- 3. 工具函数 ---

def generate_docx(text):
    """将文本转换为Word"""
    doc = Document()
    doc.add_heading('处方点评最终报告', 0)
    for line in text.split('\n'):
        if line.strip():
            p = doc.add_paragraph(line.strip())
    bio = BytesIO()
    doc.save(bio)
    bio.seek(0)
    return bio

class KnowledgeManager:
    def __init__(self, model_name, db_path):
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs={'device': 'cpu'})
        self.vectorstore = Chroma(persist_directory=db_path, embedding_function=self.embeddings)

    def upload_docs(self, file_path, original_name):
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        # 强制在 metadata 中记录原始文件名
        for doc in docs:
            doc.metadata["source"] = original_name 
        splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
        self.vectorstore.add_documents(splitter.split_documents(docs))
        return len(docs)

    def retrieve_context_with_sources(self, drug_names):
        """检索内容并返回带来源的结构化数据"""
        results_list = []
        unique_sources = set()
        for name in drug_names:
            if not name: continue
            docs = self.vectorstore.similarity_search(name, k=3)
            for d in docs:
                src = d.metadata.get('source', '未知文档')
                results_list.append(f"内容来自《{src}》: {d.page_content}")
                unique_sources.add(src)
        return "\n\n".join(results_list), list(unique_sources)

class PharmacyAgent:
    def __init__(self, api_key, base_url):
        self.llm = ChatOpenAI(model="deepseek-chat", api_key=str(api_key).strip(), base_url=base_url, temperature=0.1)

    def audit(self, prescription_json, context):
        system_prompt = """你是一位资深临床药师。请根据【参考资料库】审核【处方数据】。
        必须在报告末尾明确标注：'本次审核参考了以下文件：[文件名]'。
        若资料库中无相关药物，请基于临床药理学常识说明。"""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "【参考资料库（内含文件名）】:\n{context}\n\n【处方数据】:\n{prescription}")
        ])
        chain = prompt | self.llm
        return chain.invoke({"context": context, "prescription": json.dumps(prescription_json, ensure_ascii=False)}).content

# --- 4. 初始化 ---
@st.cache_resource
def get_km():
    return KnowledgeManager("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", DB_PATH)

# --- 5. UI 界面 ---
def main():
    st.set_page_config(page_title="AI 药师专家系统", layout="wide")
    km = get_km()

    # 初始化 session_state
    if "report_content" not in st.session_state:
        st.session_state.report_content = ""
    if "current_sources" not in st.session_state:
        st.session_state.current_sources = []

    # --- 侧边栏 ---
    with st.sidebar:
        st.header("⚙️ 配置与知识库")
        api_key = st.text_input("DeepSeek API Key:", type="password")
        files = st.file_uploader("上传 PDF 说明书", accept_multiple_files=True)
        if files and st.button("同步知识库"):
            for f in files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(f.getvalue())
                    km.upload_docs(tmp.name, f.name) # 传递原始文件名
            st.success("同步完成")

    # --- 主界面 ---
    st.title("🏥 临床药师点评专家平台")

    col_in, col_out = st.columns([1, 1.3])

    with col_in:
        st.subheader("📋 处方录入")
        p_type = st.radio("患者类型", ["儿童", "成人"], horizontal=True)
        c1, c2 = st.columns(2)
        age = c1.number_input("年龄", 6)
        weight = c2.number_input("体重 (kg)", 20.0) if p_type == "儿童" else None
        diag = st.text_input("临床诊断", "急性支气管炎")
        
        med_df = st.data_editor(pd.DataFrame([{"药品名称": "阿莫西林胶囊", "单次剂量": "0.25g", "频次": "QD"}]), num_rows="dynamic", use_container_width=True)

        if st.button("🔍 执行深度点评", type="primary", use_container_width=True):
            if api_key:
                agent = PharmacyAgent(api_key, "https://api.deepseek.com")
                with st.spinner("正在检索本地库并分析..."):
                    # 1. 检索带来源的上下文
                    context_str, sources = km.retrieve_context_with_sources(med_df["药品名称"].tolist())
                    st.session_state.current_sources = sources
                    
                    # 2. 调用模型
                    prescription = {"patient": {"age": age, "weight": weight, "diagnosis": diag}, "medications": med_df.to_dict('records')}
                    # 获取报告并存入 session_state
                    st.session_state.report_content = agent.audit(prescription, context_str)
            else:
                st.error("请输入 API Key")

    with col_out:
        st.subheader("📝 点评报告与溯源")
        
        # 核心解决：判断参考的文件来源
        if st.session_state.current_sources:
            with st.expander("📂 本次点评参考的本地文件 (溯源)", expanded=True):
                for s in st.session_state.current_sources:
                    st.write(f"✅ 已匹配到说明书：`:blue[{s}]`")
        
        if st.session_state.report_content:
            # 核心解决：实时同步修改内容
            # 使用 key 绑定 session_state.report_content，确保输入即同步
            st.session_state.report_content = st.text_area(
                "手动修正区：",
                value=st.session_state.report_content,
                height=500,
                key="active_report_editor" 
            )
            
            st.divider()
            c1, c2 = st.columns(2)
            with c1:
                docx_data = generate_docx(st.session_state.report_content)
                st.download_button("📥 导出 Word", data=docx_data, file_name="最终点评报告.docx", use_container_width=True)
            with c2:
                if st.button("🗑️ 重置", use_container_width=True):
                    st.session_state.report_content = ""
                    st.session_state.current_sources = []
                    st.rerun()
            
            if st.toggle("预览渲染格式"):
                st.markdown(st.session_state.report_content)

if __name__ == "__main__":
    main()
