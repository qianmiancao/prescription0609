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

# --- 2. 基础环境配置 ---
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# 数据库存储路径
DB_PATH = "./drug_db"
if not os.path.exists(DB_PATH):
    os.makedirs(DB_PATH, exist_ok=True)

# --- 3. 逻辑类定义 ---

class KnowledgeManager:
    def __init__(self, model_name, db_path):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'}
        )
        self.vectorstore = Chroma(
            persist_directory=db_path,
            embedding_function=self.embeddings
        )

    def upload_docs(self, file_path):
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
        splits = splitter.split_documents(docs)
        self.vectorstore.add_documents(splits)
        return len(splits)

    def retrieve_context(self, drug_names):
        all_context = []
        for name in drug_names:
            if not name: continue
            results = self.vectorstore.similarity_search(name, k=4)
            for res in results:
                source = os.path.basename(res.metadata.get('source', '未知文档'))
                all_context.append(f"【来源:{source}】\n{res.page_content}")
        return "\n\n".join(list(set(all_context)))

class PharmacyAgent:
    def __init__(self, api_key, base_url):
        self.llm = ChatOpenAI(
            model="deepseek-chat",
            api_key=str(api_key).strip(),
            base_url=base_url,
            temperature=0.1
        )

    def audit(self, prescription_json, context):
        system_prompt = """你是一位资深临床药师。请根据【参考资料】审核【处方数据】。
        输出一份结构化的Markdown报告。必须包含：风险等级、各维度评估表、药师意见。
        如果资料不足，请基于药学常识给出合理建议。"""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "【参考资料】:\n{context}\n\n【处方数据】:\n{prescription}")
        ])
        chain = prompt | self.llm
        return chain.invoke({
            "context": context, 
            "prescription": json.dumps(prescription_json, ensure_ascii=False)
        }).content

# --- 4. 缓存与状态初始化 ---

@st.cache_resource
def get_km():
    return KnowledgeManager("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", DB_PATH)

# 初始化 SessionState 用于存储可编辑的报告
if "edit_report" not in st.session_state:
    st.session_state.edit_report = ""

# --- 5. Streamlit UI ---

def main():
    st.set_page_config(page_title="AI 药师审方 (专家编辑版)", layout="wide")
    km = get_km()

    # --- 侧边栏 ---
    with st.sidebar:
        st.title("🔐 系统设置")
        api_key = st.text_input("DeepSeek API Key:", type="password")
        st.divider()
        st.header("📚 知识库上传")
        files = st.file_uploader("上传说明书 (PDF)", accept_multiple_files=True)
        if files and st.button("同步知识库"):
            for f in files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(f.getvalue())
                    km.upload_docs(tmp.name)
            st.success("同步成功！")

    # --- 主界面 ---
    st.title("💊 药剂科 AI 处方审核与点评系统")
    
    if not api_key:
        st.info("💡 请先在侧边栏配置 API Key")
        st.stop()

    agent = PharmacyAgent(api_key, "https://api.deepseek.com")

    col_input, col_report = st.columns([1, 1.2])

    # --- 左侧：处方录入 (可自由输入) ---
    with col_input:
        st.subheader("📋 处方信息录入")
        with st.expander("👤 患者基本信息", expanded=True):
            c1, c2 = st.columns(2)
            age = c1.number_input("年龄", 6)
            weight = c2.number_input("体重 (kg)", 22.0)
            diag = st.text_input("临床诊断", "社区获得性肺炎")

        st.markdown("**💊 药品明细 (可增减行/修改内容)**")
        # 默认示例数据
        default_data = [
            {"药品名称": "阿奇霉素干混悬剂", "剂量": "0.25g", "频次": "QD", "用法": "口服"},
            {"药品名称": "布地奈德混悬液", "剂量": "1mg", "频次": "BID", "用法": "雾化吸入"}
        ]
        # 使用 data_editor 实现药品可输入/可增减
        med_df = st.data_editor(
            pd.DataFrame(default_data), 
            num_rows="dynamic", 
            use_container_width=True,
            key="med_editor"
        )

        if st.button("🚀 开始 AI 辅助审核", type="primary", use_container_width=True):
            with st.spinner("AI 药师分析中..."):
                drug_names = med_df["药品名称"].tolist()
                context = km.retrieve_context(drug_names)
                prescription = {
                    "patient": {"age": age, "weight": weight, "diagnosis": diag},
                    "medications": med_df.to_dict('records')
                }
                # 获取 AI 原始报告并存入 session_state
                st.session_state.edit_report = agent.audit(prescription, context)

    # --- 右侧：点评报告 (可二次修改) ---
    with col_report:
        st.subheader("📝 处方点评报告 (药师可修改)")
        
        if st.session_state.edit_report:
            # 使用 text_area 提供修改权限
            edited_text = st.text_area(
                "您可以直接在下方修改 AI 生成的内容：",
                value=st.session_state.edit_report,
                height=600,
                key="report_area"
            )
            
            # 同步修改到 session_state
            st.session_state.edit_report = edited_text
            
            st.divider()
            c1, c2, c3 = st.columns(3)
            with c1:
                if st.button("💾 保存当前修改"):
                    st.success("修改已保存")
            with c2:
                st.download_button(
                    "📥 导出最终报告 (TXT)",
                    st.session_state.edit_report,
                    file_name=f"处方点评_{datetime.now().strftime('%Y%m%d')}.txt"
                )
            with c3:
                # 预览模式切换
                if st.checkbox("👁️ 预览 Markdown 格式"):
                    st.markdown("---")
                    st.markdown(st.session_state.edit_report)
        else:
            st.info("等待处方提交后生成报告...")
            st.image("https://via.placeholder.com/600x400.png?text=Audit+Report+Area", use_column_width=True)

if __name__ == "__main__":
    main()
