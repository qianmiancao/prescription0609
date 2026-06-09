# --- 1. 核心兼容性补丁 (必须处于最顶部) ---
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

# --- 2. 基础环境配置 ---
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# 变更路径以强制刷新数据库索引，解决持久化引起的 InternalError
DB_PATH = "./drug_db_final" 

# --- 3. 逻辑类定义 ---

class KnowledgeManager:
    def __init__(self, model_name, db_path):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name, 
            model_kwargs={'device': 'cpu'}
        )
        self.db_path = db_path
        self.vectorstore = self.init_db()

    def init_db(self):
        """初始化数据库"""
        try:
            return Chroma(persist_directory=self.db_path, embedding_function=self.embeddings)
        except Exception:
            if os.path.exists(self.db_path):
                shutil.rmtree(self.db_path)
            return Chroma(persist_directory=self.db_path, embedding_function=self.embeddings)

    def upload_docs(self, file_path, original_name):
        """上传并索引 PDF"""
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        for doc in docs:
            doc.metadata = {"source": original_name} 
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        splits = splitter.split_documents(docs)
        self.vectorstore.add_documents(splits)
        return len(splits)

    def retrieve_context(self, drug_names):
        """检索知识库并返回结果及来源"""
        results = []
        sources = set()
        for name in drug_names:
            if not name: continue
            # 执行相似度搜索
            docs = self.vectorstore.similarity_search(name, k=3)
            for d in docs:
                src = os.path.basename(d.metadata.get('source', '参考资料'))
                # 简单校验：确保药名在内容中出现，减少误报
                if name[:2] in d.page_content or name[:2] in src:
                    results.append(f"【内容源自《{src}》】: {d.page_content}")
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
        system_prompt = """你是一位资深临床药师。请根据参考资料审核处方。
        格式要求：
        1. 禁止使用星号*。
        2. 必须列出处方号。
        3. 结构包含：一、基本信息；二、药品清单；三、药学点评；四、结论。
        报告末尾注明参考文件名。"""
        
        prompt = ChatPromptTemplate.from_template(
            "系统指令: {sys}\n参考资料: {ctx}\n待审处方: {rx}"
        )
        chain = prompt | self.llm
        res = chain.invoke({
            "sys": system_prompt,
            "ctx": context,
            "rx": json.dumps(p_data, ensure_ascii=False)
        })
        return res.content.replace('*', '').strip()

# --- 4. 辅助函数 ---

def generate_docx(text, p_no):
    doc = Document()
    doc.add_heading('处方点评报告', 0)
    doc.add_paragraph(f"处方号：{p_no}")
    doc.add_paragraph(f"日期：{datetime.now().strftime('%Y-%m-%d %H:%M')}")
    doc.add_paragraph("-" * 20)
    for line in text.split('\n'):
        if line.strip(): doc.add_paragraph(line.strip())
    bio = BytesIO()
    doc.save(bio)
    bio.seek(0)
    return bio

# --- 5. Streamlit UI ---

def main():
    st.set_page_config(page_title="AI 药师点评系统", layout="wide")
    
    # 强制刷新缓存的小技巧：如果代码变动，修改此处的版本号
    @st.cache_resource(show_spinner="正在加载知识引擎...")
    def get_km(v="1.0"):
        return KnowledgeManager("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", DB_PATH)
    
    km = get_km(v="2.0") # 升级版本号强制刷新实例

    if "rpt" not in st.session_state: st.session_state.rpt = ""
    if "srcs" not in st.session_state: st.session_state.srcs = []

    with st.sidebar:
        st.header("⚙️ 知识库管理")
        api_key = st.text_input("DeepSeek API Key:", type="password")
        files = st.file_uploader("上传药品说明书", accept_multiple_files=True)
        if files and st.button("✨ 开始同步"):
            for f in files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(f.getvalue())
                    km.upload_docs(tmp.name, f.name)
            st.success("同步成功")
        
        if st.button("🗑️ 重置数据库"):
            if os.path.exists(DB_PATH): shutil.rmtree(DB_PATH)
            st.rerun()

    st.title("🏥 临床药师点评专家平台")

    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.subheader("📋 处方录入")
        with st.container(border=True):
            p_no = st.text_input("处方号", value="RX"+datetime.now().strftime("%H%M%S"))
            diag = st.text_input("诊断", "急性支气管炎")
            age = st.number_input("年龄", 6)
            weight = st.number_input("体重 (kg)", 22.0)
            
            st.markdown("**具体产品明细**")
            med_df = st.data_editor(
                pd.DataFrame([{"产品名称": "阿莫西林", "用量": "0.25g", "频次": "QD"}]),
                num_rows="dynamic", use_container_width=True
            )

        if st.button("🔍 执行点评", type="primary", use_container_width=True):
            if not api_key: st.error("请输入 API Key")
            else:
                agent = PharmacyAgent(api_key)
                with st.spinner("正在匹配并分析..."):
                    drugs = med_df["产品名称"].tolist()
                    ctx, srcs = km.retrieve_context(drugs) # 此处名称必须与类定义一致
                    st.session_state.srcs = srcs
                    
                    data = {"no": p_no, "diag": diag, "age": age, "weight": weight, "meds": med_df.to_dict('records')}
                    st.session_state.rpt = agent.audit(data, ctx)
                    st.rerun()

    with col2:
        st.subheader("📝 审核报告")
        if st.session_state.srcs:
            st.info(f"参考文件: {', '.join(st.session_state.srcs)}")
        
        if st.session_state.rpt:
            # 实时同步编辑框
            final_rpt = st.text_area("内容修改:", value=st.session_state.rpt, height=500)
            st.session_state.rpt = final_rpt
            
            st.divider()
            c1, c2 = st.columns(2)
            with c1:
                st.download_button(
                    "📥 导出 Word 报告", 
                    data=generate_docx(st.session_state.rpt, p_no), 
                    file_name=f"报告_{p_no}.docx"
                )
            with c2:
                if st.button("🗑️ 清空内容"):
                    st.session_state.rpt = ""; st.rerun()

if __name__ == "__main__":
    main()
