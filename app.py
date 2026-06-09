# --- 1. 核心兼容性补丁 (必须放在最前) ---
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

# --- 3. 工具函数定义 ---

def clean_markdown_symbols(text):
    """移除文本中的所有星号及Markdown加粗符号，保持排版整洁"""
    if not text:
        return ""
    # 移除星号
    text = text.replace('*', '')
    # 移除过多的换行
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def generate_docx(text):
    """将纯文本转换为标准Word文档"""
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
        """上传说明书并记录原始文件名"""
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        for doc in docs:
            doc.metadata["source"] = original_name 
        splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
        self.vectorstore.add_documents(splitter.split_documents(docs))
        return len(docs)

    def retrieve_context_with_sources(self, drug_names):
        """检索知识库并返回去重后的文件名列表"""
        results_list = []
        unique_sources = set()
        for name in drug_names:
            if not name: continue
            docs = self.vectorstore.similarity_search(name, k=3)
            for d in docs:
                raw_src = d.metadata.get('source', '未知文档')
                src_name = os.path.basename(raw_src)
                # 过滤掉系统产生的临时路径记录
                if "tmp" in src_name and len(src_name) > 15:
                    continue
                results_list.append(f"内容来自《{src_name}》: {d.page_content}")
                unique_sources.add(src_name)
        return "\n\n".join(results_list), list(unique_sources)

class PharmacyAgent:
    def __init__(self, api_key, base_url):
        self.llm = ChatOpenAI(model="deepseek-chat", api_key=str(api_key).strip(), base_url=base_url, temperature=0.1)

    def audit(self, prescription_json, context):
        """执行AI药师推理"""
        system_prompt = """你是一位资深临床药师。请根据【参考资料库】对【处方数据】输出正式点评报告。
        
        重要要求：
        1. 严禁使用任何星号（*）进行列表或加粗。
        2. 请使用纯文本数字（1. 2. 3.）进行段落分级。
        3. 若患者为儿童，必须依据体重核算剂量合理性。
        4. 若为成人，重点分析诊断匹配、药物相互作用及医保合规。
        5. 报告最后必须列出：参考文件：[文件名]。"""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "【参考资料库】:\n{context}\n\n【处方数据】:\n{prescription}")
        ])
        chain = prompt | self.llm
        raw_report = chain.invoke({"context": context, "prescription": json.dumps(prescription_json, ensure_ascii=False)}).content
        return clean_markdown_symbols(raw_report)

# --- 4. Streamlit 界面逻辑 ---

def main():
    st.set_page_config(page_title="AI 临床药师点评专家平台", layout="wide", page_icon="💊")
    
    @st.cache_resource
    def get_km():
        return KnowledgeManager("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", DB_PATH)
    
    km = get_km()

    # 初始化状态机
    if "report_content" not in st.session_state:
        st.session_state.report_content = ""
    if "current_sources" not in st.session_state:
        st.session_state.current_sources = []
    if "is_confirmed" not in st.session_state:
        st.session_state.is_confirmed = False

    # --- 侧边栏 ---
    with st.sidebar:
        st.header("⚙️ 系统管理")
        api_key = st.text_input("DeepSeek API Key:", type="password")
        
        st.divider()
        st.subheader("📚 医院说明书库")
        files = st.file_uploader("上传药品说明书 (PDF)", accept_multiple_files=True)
        if files and st.button("✨ 同步新知识"):
            for f in files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(f.getvalue())
                    km.upload_docs(tmp.name, f.name)
            st.success("同步成功！")
        
        if st.sidebar.button("🗑️ 清空本地数据库"):
            import shutil
            if os.path.exists(DB_PATH):
                shutil.rmtree(DB_PATH)
            st.rerun()

    st.title("🏥 临床药师点评专家平台")
    st.caption("基于 RAG 知识检索增强与 DeepSeek-V3 引擎")

    col_in, col_out = st.columns([1, 1.2])

    # --- 左侧：处方录入 ---
    with col_in:
        st.subheader("📋 处方录入")
        with st.container(border=True):
            p_type = st.radio("患者类型", ["儿童", "成人"], horizontal=True)
            c1, c2 = st.columns(2)
            age = c1.number_input("年龄", value=6, min_value=0)
            
            weight = None
            if p_type == "儿童":
                weight = c2.number_input("体重 (kg)", value=20.0, step=0.1)
            else:
                c2.info("💡 成人模式已忽略体重")

            diag = st.text_input("临床诊断", "急性支气管炎")
            
            st.markdown("**药品清单**")
            # 默认展示一行数据，用户可自由输入
            med_df = st.data_editor(
                pd.DataFrame([{"药品名称": "阿莫西林胶囊", "单次剂量": "0.25g", "频次": "QD"}]), 
                num_rows="dynamic", 
                use_container_width=True,
                key="drug_editor"
            )

            # 核心逻辑：执行深度点评
            if st.button("🔍 执行深度点评", type="primary", use_container_width=True):
                if not api_key:
                    st.error("请先在左侧侧边栏输入 API Key")
                else:
                    # 第一步：清空旧状态，防止用户看到旧药的报告
                    st.session_state.report_content = ""
                    st.session_state.current_sources = []
                    st.session_state.is_confirmed = False
                    
                    with st.spinner("AI 药师正在检索资料库并推理中..."):
                        # 第二步：获取当前表格中的药品列表
                        drug_names = med_df["药品名称"].tolist()
                        
                        # 第三步：知识检索
                        context_str, sources = km.retrieve_context_with_sources(drug_names)
                        st.session_state.current_sources = sources
                        
                        # 第四步：构建处方并调用 AI
                        agent = PharmacyAgent(api_key, "https://api.deepseek.com")
                        prescription = {
                            "patient": {"age": age, "weight": weight, "diagnosis": diag, "type": p_type},
                            "medications": med_df.to_dict('records')
                        }
                        st.session_state.report_content = agent.audit(prescription, context_str)
                        st.rerun() # 强制刷新以显示新内容

    # --- 右侧：点评报告 ---
    with col_out:
        st.subheader("📝 点评报告与溯源")
        
        if st.session_state.current_sources:
            with st.expander("📂 本次点评参考的文件溯源", expanded=True):
                for s in st.session_state.current_sources:
                    st.write(f"✅ 已成功匹配说明书：`:blue[{s}]`")
        
        if st.session_state.report_content:
            # 修改区
            edited_text = st.text_area(
                "药师手动修正区 (修改后需点击确认)：",
                value=st.session_state.report_content,
                height=500,
                key="editor_area"
            )

            # 确认按钮，解决导出不同步的问题
            if st.button("✅ 确认并锁定修改内容", use_container_width=True):
                st.session_state.report_content = edited_text
                st.session_state.is_confirmed = True
                st.success("内容已锁定，可以导出！")

            st.divider()
            
            c1, c2 = st.columns(2)
            with c1:
                if st.session_state.is_confirmed:
                    # 导出 Word
                    docx_data = generate_docx(st.session_state.report_content)
                    st.download_button(
                        label="📥 导出 Word 最终报告",
                        data=docx_data,
                        file_name=f"处方点评报告_{datetime.now().strftime('%m%d_%H%M')}.docx",
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        use_container_width=True,
                        type="primary"
                    )
                else:
                    st.warning("⚠️ 请先点击上方确认按钮锁定修改")
            
            with c2:
                if st.button("🗑️ 清空重置", use_container_width=True):
                    st.session_state.report_content = ""
                    st.session_state.is_confirmed = False
                    st.session_state.current_sources = []
                    st.rerun()
        else:
            st.info("💡 请在左侧录入处方并点击“执行深度点评”开始。")

if __name__ == "__main__":
    main()
