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
import re 
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

DB_PATH = "./drug_db"
if not os.path.exists(DB_PATH):
    os.makedirs(DB_PATH, exist_ok=True)

# --- 3. 工具函数 ---

def clean_markdown_symbols(text):
    if not text: return ""
    text = text.replace('*', '')
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def generate_docx(text):
    doc = Document()
    doc.add_heading('处方点评报告', 0)
    for line in text.split('\n'):
        if line.strip():
            doc.add_paragraph(line.strip())
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
        for doc in docs:
            # 关键修复：强制 metadata 只包含原始文件名
            doc.metadata = {"source": original_name} 
        splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
        self.vectorstore.add_documents(splitter.split_documents(docs))
        return len(docs)

    def retrieve_context_with_sources(self, drug_names):
        """增强版检索：增加了药名关键词的硬校验，防止跨药匹配"""
        results_list = []
        unique_sources = set()
        
        for name in drug_names:
            if not name or len(name.strip()) < 1: continue
            
            # 执行相似度检索
            docs = self.vectorstore.similarity_search(name, k=4)
            
            for d in docs:
                raw_src = d.metadata.get('source', '未知文档')
                src_name = os.path.basename(raw_src)
                
                # 过滤条件1：排除包含 tmp 的旧垃圾数据
                if "tmp" in src_name: continue
                
                # 过滤条件2：硬核核对。文件名或内容中必须包含当前药名的核心关键字
                # 比如搜二甲双胍，结果文件名必须包含二甲双胍，否则认为是污染数据
                keyword = name[:2] # 提取前两个字作为核心词
                if keyword in src_name or keyword in d.page_content:
                    results_list.append(f"【参考源自《{src_name}》】: {d.page_content}")
                    unique_sources.add(src_name)
                    
        return "\n\n".join(results_list), list(unique_sources)

class PharmacyAgent:
    def __init__(self, api_key, base_url):
        self.llm = ChatOpenAI(model="deepseek-chat", api_key=str(api_key).strip(), base_url=base_url, temperature=0.1)

    def audit(self, prescription_json, context):
        system_prompt = """你是一位资深临床药师。请根据【参考资料库】审核【处方数据】。
        
        重要：
        1. 必须基于提供的资料进行分析。
        2. 如果参考资料库的内容与处方药名不符，请在报告中明确指出“未找到该药说明书”。
        3. 严禁使用 * 号，使用纯文本序号。"""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "【参考资料库】:\n{context}\n\n【处方数据】:\n{prescription}")
        ])
        chain = prompt | self.llm
        return clean_markdown_symbols(chain.invoke({"context": context, "prescription": json.dumps(prescription_json, ensure_ascii=False)}).content)

# --- 4. 初始化 ---
@st.cache_resource
def get_km():
    return KnowledgeManager("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", DB_PATH)

# --- 5. UI 界面 ---
def main():
    st.set_page_config(page_title="AI 药师点评专家系统", layout="wide")
    km = get_km()

    if "report_content" not in st.session_state: st.session_state.report_content = ""
    if "current_sources" not in st.session_state: st.session_state.current_sources = []
    if "is_confirmed" not in st.session_state: st.session_state.is_confirmed = False

    with st.sidebar:
        st.header("⚙️ 知识库管理")
        api_key = st.text_input("DeepSeek API Key:", type="password")
        files = st.file_uploader("第一步：上传新说明书 (PDF)", accept_multiple_files=True)
        if files and st.button("✨ 同步到本地库"):
            for f in files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(f.getvalue())
                    km.upload_docs(tmp.name, f.name) 
            st.success("同步完成！")
        
        st.divider()
        st.warning("如果切换药品后参考文件不对，请点下方按钮：")
        if st.button("🗑️ 清空所有旧索引(彻底重置)"):
            import shutil
            if os.path.exists(DB_PATH):
                shutil.rmtree(DB_PATH)
            st.rerun()

    st.title("🏥 临床药师点评专家平台")

    col_in, col_out = st.columns([1, 1.3])

    with col_in:
        st.subheader("📋 处方录入")
        p_type = st.radio("患者类型", ["儿童", "成人"], horizontal=True)
        c1, c2 = st.columns(2)
        age = c1.number_input("年龄", value=6)
        weight = c2.number_input("体重 (kg)", value=20.0) if p_type == "儿童" else None
        diag = st.text_input("临床诊断", value="急性支气管炎")
        
        st.markdown("**药品明细表**")
        med_df = st.data_editor(
            pd.DataFrame([{"药品名称": "二甲双胍", "单次剂量": "0.25g", "频次": "QD"}]), 
            num_rows="dynamic", 
            use_container_width=True
        )

        if st.button("🔍 执行深度点评", type="primary", use_container_width=True):
            if api_key:
                st.session_state.report_content = ""
                st.session_state.current_sources = []
                st.session_state.is_confirmed = False
                
                agent = PharmacyAgent(api_key, "https://api.deepseek.com")
                
                with st.spinner("正在精准匹配说明书..."):
                    drug_names = med_df["药品名称"].tolist()
                    context_str, sources = km.retrieve_context_with_sources(drug_names)
                    st.session_state.current_sources = sources
                    
                    prescription = {
                        "patient": {"age": age, "weight": weight, "diagnosis": diag}, 
                        "medications": med_df.to_dict('records')
                    }
                    st.session_state.report_content = agent.audit(prescription, context_str)
                    st.rerun()
            else:
                st.error("请输入 API Key")

    with col_out:
        st.subheader("📝 点评报告与溯源")
        
        if st.session_state.current_sources:
            with st.expander("📂 匹配到的参考文件 (已开启硬核校验)", expanded=True):
                for s in st.session_state.current_sources:
                    st.write(f"✅ 系统已找到：`:blue[{s}]`")
        elif st.session_state.report_content:
            st.error("❌ 未在知识库中找到该药名的匹配文件！")
        
        if st.session_state.report_content:
            st.session_state.report_content = st.text_area(
                "手动修正区：",
                value=st.session_state.report_content,
                height=500,
                key="editor"
            )

            if st.button("✅ 确认修改内容"):
                st.session_state.is_confirmed = True
                st.success("同步成功")

            st.divider()
            if st.session_state.is_confirmed:
                docx_data = generate_docx(st.session_state.report_content)
                st.download_button("📥 导出 Word 报告", data=docx_data, file_name="点评报告.docx")

if __name__ == "__main__":
    main()
