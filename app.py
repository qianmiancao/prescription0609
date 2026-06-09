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
    """强制清除文本中的 * 号和 ** 号，保持排版纯净"""
    if not text: return ""
    text = text.replace('*', '')
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def generate_docx(text):
    """将清洗后的纯文本转换为Word文档"""
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
            doc.metadata["source"] = original_name 
        splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
        self.vectorstore.add_documents(splitter.split_documents(docs))
        return len(docs)

    def retrieve_context_with_sources(self, drug_names):
        """核心匹配逻辑：根据当前输入的药品列表，实时匹配数据库中的说明书"""
        results_list = []
        unique_sources = set()
        for name in drug_names:
            if not name or len(name.strip()) < 2: continue
            # 检索最相关的3条片段
            docs = self.vectorstore.similarity_search(name, k=3)
            for d in docs:
                raw_src = d.metadata.get('source', '未知文档')
                src_name = os.path.basename(raw_src)
                # 过滤掉系统临时路径
                if "tmp" in src_name and len(src_name) > 20: continue
                results_list.append(f"【参考源自《{src_name}》】: {d.page_content}")
                unique_sources.add(src_name)
        return "\n\n".join(results_list), list(unique_sources)

class PharmacyAgent:
    def __init__(self, api_key, base_url):
        self.llm = ChatOpenAI(model="deepseek-chat", api_key=str(api_key).strip(), base_url=base_url, temperature=0.1)

    def audit(self, prescription_json, context):
        system_prompt = """你是一位资深临床药师。请根据【参考资料库】审核【处方数据】并输出点评报告。
        
        重要格式要求：
        1. 严禁使用 * 号。
        2. 使用纯文本序号（1. 2. 3.）进行段落划分。
        3. 必须包含：诊断合理性评估、用法用量评估（儿童需核算体重）、相互作用提示、药师建议。
        4. 报告末尾明确列出参考的文件名。"""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "【参考资料库】:\n{context}\n\n【处方数据】:\n{prescription}")
        ])
        chain = prompt | self.llm
        raw_report = chain.invoke({"context": context, "prescription": json.dumps(prescription_json, ensure_ascii=False)}).content
        return clean_markdown_symbols(raw_report)

# --- 4. 初始化 ---
@st.cache_resource
def get_km():
    return KnowledgeManager("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", DB_PATH)

# --- 5. UI 界面 ---
def main():
    st.set_page_config(page_title="AI 药师点评专家系统", layout="wide", page_icon="💊")
    km = get_km()

    # 初始化全局状态
    if "report_content" not in st.session_state: st.session_state.report_content = ""
    if "current_sources" not in st.session_state: st.session_state.current_sources = []
    if "is_confirmed" not in st.session_state: st.session_state.is_confirmed = False

    # --- 侧边栏 ---
    with st.sidebar:
        st.header("⚙️ 知识库管理")
        api_key = st.text_input("DeepSeek API Key:", type="password")
        files = st.file_uploader("上传药品说明书 (PDF)", accept_multiple_files=True)
        if files and st.button("✨ 同步到本地库"):
            for f in files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(f.getvalue())
                    km.upload_docs(tmp.name, f.name) 
            st.success("知识库同步完成！")
        
        st.divider()
        if st.button("🗑️ 清空所有索引记录"):
            import shutil
            if os.path.exists(DB_PATH): shutil.rmtree(DB_PATH)
            st.rerun()

    st.title("🏥 临床药师点评专家平台")

    col_in, col_out = st.columns([1, 1.3])

    # --- 左侧：处方录入 (支持药品自由切换) ---
    with col_in:
        st.subheader("📋 处方录入")
        p_type = st.radio("患者类型", ["儿童", "成人"], horizontal=True)
        c1, c2 = st.columns(2)
        age = c1.number_input("年龄", value=6)
        weight = c2.number_input("体重 (kg)", value=20.0) if p_type == "儿童" else None
        diag = st.text_input("临床诊断", value="急性支气管炎")
        
        st.markdown("**药品明细表 (双击名称可切换药品)**")
        # 用户修改表格内容后，med_df 会立即更新
        med_df = st.data_editor(
            pd.DataFrame([{"药品名称": "阿莫西林胶囊", "单次剂量": "0.25g", "频次": "QD"}]), 
            num_rows="dynamic", 
            use_container_width=True
        )

        if st.button("🔍 执行深度点评", type="primary", use_container_width=True):
            if not api_key:
                st.error("请先在左侧输入 API Key")
            else:
                # 【关键逻辑】：点击按钮时，立即清除所有旧状态
                st.session_state.report_content = ""
                st.session_state.current_sources = []
                st.session_state.is_confirmed = False
                
                agent = PharmacyAgent(api_key, "https://api.deepseek.com")
                
                with st.spinner("🚀 正在重新匹配说明书并分析中..."):
                    # 1. 自动根据当前表格中的药品名称更新溯源信息
                    drug_names = med_df["药品名称"].tolist()
                    context_str, sources = km.retrieve_context_with_sources(drug_names)
                    
                    # 2. 更新状态变量
                    st.session_state.current_sources = sources
                    
                    # 3. 调用 AI 生成新报告
                    prescription = {
                        "patient": {"age": age, "weight": weight, "diagnosis": diag, "type": p_type}, 
                        "medications": med_df.to_dict('records')
                    }
                    st.session_state.report_content = agent.audit(prescription, context_str)
                    st.rerun() # 强制刷新页面显示最新结果

    # --- 右侧：点评报告与动态溯源 ---
    with col_out:
        st.subheader("📝 点评报告与溯源")
        
        # 溯源区：随药品切换自动更新
        if st.session_state.current_sources:
            with st.expander("📂 匹配到的参考文件 (自动更新)", expanded=True):
                for s in st.session_state.current_sources:
                    st.write(f"✅ 系统已找到并参考：`:blue[{s}]`")
        elif st.session_state.report_content:
            st.warning("⚠️ 未在本地库找到该药品的匹配说明书，AI 将基于通用药理学知识回答。")
        
        if st.session_state.report_content:
            # 修改区：绑定 session_state 确保同步
            edited_text = st.text_area(
                "药师手动修正区 (无星号纯净模式)：",
                value=st.session_state.report_content,
                height=500,
                key="active_editor"
            )

            # 同步修改内容的确认按钮
            if st.button("✅ 确认并锁定修改内容", use_container_width=True):
                st.session_state.report_content = edited_text
                st.session_state.is_confirmed = True
                st.success("修改已同步！现在可以导出 Word。")

            st.divider()
            
            c1, c2 = st.columns(2)
            with c1:
                if st.session_state.is_confirmed:
                    docx_data = generate_docx(st.session_state.report_content)
                    st.download_button(
                        label="📥 导出 Word 最终版", 
                        data=docx_data, 
                        file_name=f"点评报告_{datetime.now().strftime('%m%d_%H%M')}.docx", 
                        use_container_width=True,
                        type="primary"
                    )
                else:
                    st.warning("请确认修改后再导出")
            
            with c2:
                if st.button("🗑️ 重置所有结果", use_container_width=True):
                    st.session_state.report_content = ""
                    st.session_state.is_confirmed = False
                    st.session_state.current_sources = []
                    st.rerun()

if __name__ == "__main__":
    main()
