# --- 1. 核心兼容性补丁 ---
import sys
try:
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
from docx.shared import Pt, RGBColor

# --- 2. 基础环境配置 ---
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

DB_PATH = "./drug_db_v5" 

# --- 3. 逻辑类定义 ---

class KnowledgeManager:
    def __init__(self, model_name, db_path):
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs={'device': 'cpu'})
        self.db_path = db_path
        self.vectorstore = self.init_db()

    def init_db(self):
        try:
            return Chroma(persist_directory=self.db_path, embedding_function=self.embeddings)
        except Exception:
            if os.path.exists(self.db_path): shutil.rmtree(self.db_path)
            return Chroma(persist_directory=self.db_path, embedding_function=self.embeddings)

    def upload_docs(self, file_path, original_name):
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        for doc in docs: doc.metadata = {"source": original_name} 
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        self.vectorstore.add_documents(splitter.split_documents(docs))
        return len(docs)

    def retrieve_context(self, drug_names):
        results = []
        sources = set()
        for name in drug_names:
            if not name: continue
            docs = self.vectorstore.similarity_search(name, k=3)
            for d in docs:
                src = os.path.basename(d.metadata.get('source', '资料'))
                if name[:2] in d.page_content or name[:2] in src:
                    results.append(f"【参考《{src}》】: {d.page_content}")
                    sources.add(src)
        return "\n\n".join(results), list(sources)

class PharmacyAgent:
    def __init__(self, api_key):
        self.llm = ChatOpenAI(
            model="deepseek-chat", 
            api_key=api_key, 
            base_url="https://api.deepseek.com", 
            temperature=0.1
        )

    def audit(self, p_data, context):
        # --- 核心改进：明确要求禁止使用表格符号 ---
        system_prompt = """你是一位资深临床药师。请根据参考资料审核处方。
        
        【格式规范（必须遵守）】：
        1. 禁止使用任何 Markdown 表格符号（禁止使用 | 、- 、+ 等符号来构思表格）。
        2. “二、产品明细”部分请使用如下纯文本列表格式：
           1. 药品名称：[名称]；剂量：[剂量]；频次：[频次]；用法：[用法]
           2. 药品名称：[名称]；剂量：[剂量]；频次：[频次]；用法：[用法]
        3. 报告禁止使用星号 * 进行加粗。
        4. 报告必须包含：一、基本信息；二、产品明细；三、适宜性点评；四、结论。
        """
        
        prompt = ChatPromptTemplate.from_template(
            "指令: {sys}\n资料: {ctx}\n处方: {rx}"
        )
        chain = prompt | self.llm
        res = chain.invoke({
            "sys": system_prompt,
            "ctx": context,
            "rx": json.dumps(p_data, ensure_ascii=False)
        })
        # 二次清洗可能存在的表格符号和星号
        clean_text = res.content.replace('*', '').replace('|', '').replace('- -', '')
        return clean_text.strip()

# --- 4. 辅助函数 ---

def generate_docx(text, p_no):
    doc = Document()
    header = doc.add_heading('临床处方点评报告', 0)
    
    # 基础信息展示
    doc.add_paragraph(f"处方编号：{p_no}")
    doc.add_paragraph(f"点评时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}")
    doc.add_paragraph("-" * 40)

    for line in text.split('\n'):
        line = line.strip()
        if not line: continue
        
        para = doc.add_paragraph()
        # 对一级标题加粗
        if any(line.startswith(prefix) for prefix in ["一、", "二、", "三、", "四、"]):
            run = para.add_run(line)
            run.bold = True
            run.font.size = Pt(12)
        else:
            para.add_run(line)
            
    bio = BytesIO()
    doc.save(bio)
    bio.seek(0)
    return bio

# --- 5. UI 界面 ---

def main():
    st.set_page_config(page_title="AI 药师专家系统", layout="wide", page_icon="💊")
    
    @st.cache_resource(show_spinner="药学大脑启动中...")
    def get_km(v="5.0"):
        return KnowledgeManager("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", DB_PATH)
    
    km = get_km()

    if "rpt" not in st.session_state: st.session_state.rpt = ""
    if "srcs" not in st.session_state: st.session_state.srcs = []

    with st.sidebar:
        st.header("⚙️ 知识库管理")
        api_key = st.text_input("DeepSeek API Key:", type="password")
        files = st.file_uploader("上传药品说明书 (PDF)", accept_multiple_files=True)
        if files and st.button("✨ 同步到库"):
            with st.spinner("建立药学索引..."):
                for f in files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(f.getvalue())
                        km.upload_docs(tmp.name, f.name)
            st.success("同步完成")
        
        if st.button("🗑️ 清空数据库"):
            if os.path.exists(DB_PATH): shutil.rmtree(DB_PATH)
            st.rerun()

    st.title("🏥 临床药师点评专家平台")

    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.subheader("📋 处方录入")
        with st.container(border=True):
            p_no = st.text_input("处方号", value="RX"+datetime.now().strftime("%H%M%S"))
            diag = st.text_input("诊断", "社区获得性肺炎")
            
            p_type = st.radio("患者类型", ["成人", "儿童"], horizontal=True)
            c1, c2 = st.columns(2)
            age = c1.number_input("年龄", value=35 if p_type == "成人" else 6)
            weight = c2.number_input("体重 (kg)", value=22.0) if p_type == "儿童" else None
            
            st.markdown("**药品清单录入**")
            med_df = st.data_editor(
                pd.DataFrame([{"产品名称": "阿莫西林胶囊", "剂量": "0.5g", "频次": "TID", "用法": "口服"}]),
                num_rows="dynamic", use_container_width=True
            )

        if st.button("🔍 开始执行点评", type="primary", use_container_width=True):
            if not api_key: st.error("请先输入 API Key")
            else:
                agent = PharmacyAgent(api_key)
                with st.spinner("正在检索匹配并进行推理分析..."):
                    drugs = med_df["产品名称"].tolist()
                    ctx, srcs = km.retrieve_context(drugs)
                    st.session_state.srcs = srcs
                    
                    p_data = {
                        "no": p_no, "diag": diag, "type": p_type,
                        "age": age, "weight": weight, "meds": med_df.to_dict('records')
                    }
                    st.session_state.rpt = agent.audit(p_data, ctx)
                    st.rerun()

    with col2:
        st.subheader("📝 审核报告")
        if st.session_state.srcs:
            st.success(f"参考溯源: {', '.join(st.session_state.srcs)}")
        
        if st.session_state.rpt:
            # 药师手动微调
            final_rpt = st.text_area("内容编辑区域 (纯文本模式):", value=st.session_state.rpt, height=550)
            st.session_state.rpt = final_rpt
            
            st.divider()
            bt1, bt2 = st.columns(2)
            with bt1:
                st.download_button(
                    "📥 导出 Word 报告", 
                    data=generate_docx(st.session_state.rpt, p_no), 
                    file_name=f"点评报告_{p_no}.docx",
                    use_container_width=True,
                    type="primary"
                )
            with bt2:
                if st.button("🗑️ 清空报告", use_container_width=True):
                    st.session_state.rpt = ""
                    st.rerun()
        else:
            st.info("👈 请在左侧录入并执行点评。")

if __name__ == "__main__":
    main()
