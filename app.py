# --- 1. 核心兼容性补丁 (解决 Streamlit Cloud 的 InternalError) ---
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

DB_PATH = "./drug_db_v3" # 使用新路径避免旧索引污染

# --- 3. 逻辑类定义 ---

class KnowledgeManager:
    def __init__(self, model_name, db_path):
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs={'device': 'cpu'})
        self.db_path = db_path
        self.init_db()

    def init_db(self):
        try:
            self.vectorstore = Chroma(persist_directory=self.db_path, embedding_function=self.embeddings)
        except Exception:
            if os.path.exists(self.db_path): shutil.rmtree(self.db_path)
            self.vectorstore = Chroma(persist_directory=self.db_path, embedding_function=self.embeddings)

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
                if name[:2] in d.page_content or name[:2] in src: # 关键词二次匹配
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

    def audit(self, prescription_data, context):
        system_prompt = """你是一位资深临床药师。请根据【参考资料】对提供的【处方数据】输出正式点评报告。
        
        报告必须严格包含以下结构（纯文本格式，禁止使用星号*）：
        一、处方基本信息
        处方号：[处方号]
        诊断：[临床诊断]
        患者：[年龄, 体重]
        
        二、点评药品清单
        [具体列出所有药品名称及用法用量]
        
        三、药学适宜性点评
        1. 适应症审核：...
        2. 用法用量审核（儿童需核算具体mg/kg）：...
        3. 相互作用与禁忌：...
        
        四、点评结论
        [结论：合理 / 不合理 / 用药不适宜]
        [建议：具体修改建议]
        
        报告末尾列出参考的说明书文件名。"""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "【参考资料】:\n{context}\n\n【处方数据】:\n{prescription}")
        ])
        chain = prompt | self.llm
        res = chain.invoke({
            "context": context, 
            "prescription": json.dumps(prescription_data, ensure_ascii=False)
        })
        # 强制清洗任何 AI 生成的残留星号
        return res.content.replace('*', '').strip()

# --- 4. 辅助函数 ---

def generate_docx(text, prescription_no):
    doc = Document()
    title = doc.add_heading('临床处方点评报告', 0)
    
    # 增加处方号和时间信息
    p_info = doc.add_paragraph()
    p_info.add_run(f"报告编号：{prescription_no}\n").bold = True
    p_info.add_run(f"生成日期：{datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    p_info.add_run(f"审核状态：已完成（智能辅助审核）").italic = True

    doc.add_paragraph("-" * 30)

    for line in text.split('\n'):
        if line.strip():
            para = doc.add_paragraph(line.strip())
            # 对“一、二、三”标题加粗处理
            if any(line.startswith(prefix) for prefix in ["一、", "二、", "三、", "四、"]):
                for run in para.runs: run.bold = True
                
    bio = BytesIO()
    doc.save(bio)
    bio.seek(0)
    return bio

# --- 5. Streamlit UI ---

def main():
    st.set_page_config(page_title="AI 药师点评专家系统", layout="wide", page_icon="💊")
    
    # 状态初始化
    if "report" not in st.session_state: st.session_state.report = ""
    if "sources" not in st.session_state: st.session_state.sources = []

    @st.cache_resource
    def load_km():
        return KnowledgeManager("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", DB_PATH)
    
    km = load_km()

    # --- 侧边栏 ---
    with st.sidebar:
        st.header("⚙️ 知识库管理")
        api_key = st.text_input("DeepSeek API Key:", type="password")
        st.divider()
        files = st.file_uploader("上传 PDF 说明书", accept_multiple_files=True)
        if files and st.button("✨ 同步到本地库"):
            with st.spinner("正在写入索引..."):
                for f in files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(f.getvalue())
                        km.upload_docs(tmp.name, f.name)
                st.success("同步完成")
        
        if st.button("🗑️ 清空数据库"):
            if os.path.exists(DB_PATH): shutil.rmtree(DB_PATH)
            st.rerun()

    # --- 主界面 ---
    st.title("🏥 临床药师处方点评专家平台")

    col_in, col_out = st.columns([1, 1.2])

    with col_in:
        st.subheader("📋 处方详情录入")
        with st.container(border=True):
            p_no = st.text_input("处方号/流水号", value=datetime.now().strftime("RX%Y%m%d%H%M"))
            
            p_type = st.radio("患者类型", ["儿童", "成人"], horizontal=True)
            c1, c2 = st.columns(2)
            age = c1.number_input("年龄", value=6, min_value=0)
            weight = c2.number_input("体重 (kg)", value=22.0) if p_type == "儿童" else None
            
            diag = st.text_input("临床诊断", "急性支气管炎")
            
            st.markdown("**药品清单 (具体产品明细)**")
            df_init = pd.DataFrame([{"产品名称": "阿莫西林胶囊", "规格剂量": "0.25g", "频次": "QD", "用法": "口服"}])
            med_df = st.data_editor(df_init, num_rows="dynamic", use_container_width=True)

        if st.button("🔍 执行深度点评", type="primary", use_container_width=True):
            if not api_key:
                st.error("请先输入 API Key")
            else:
                agent = PharmacyAgent(api_key)
                with st.spinner("正在检索匹配并进行临床推理..."):
                    # 1. 检索
                    drug_names = med_df["产品名称"].tolist()
                    context, sources = km.retrieve_context(drug_names)
                    st.session_state.sources = sources
                    
                    # 2. 调用 AI
                    p_data = {
                        "prescription_no": p_no,
                        "patient": {"age": age, "weight": weight, "diagnosis": diag, "type": p_type},
                        "medications": med_df.to_dict('records')
                    }
                    st.session_state.report = agent.audit(p_data, context)
                    st.rerun()

    with col_out:
        st.subheader("📝 结构化点评报告")
        
        if st.session_state.sources:
            st.success(f"参考文件溯源：{', '.join(st.session_state.sources)}")
        
        if st.session_state.report:
            # 修改区
            final_report = st.text_area(
                "药师修正区（内容将同步至导出的Word文件）：", 
                value=st.session_state.report, 
                height=550
            )
            st.session_state.report = final_report
            
            st.divider()
            c1, c2 = st.columns(2)
            with c1:
                # 导出 Word
                docx_data = generate_docx(st.session_state.report, p_no)
                st.download_button(
                    label="📥 导出 Word 点评报告",
                    data=docx_data,
                    file_name=f"点评报告_{p_no}.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    use_container_width=True,
                    type="primary"
                )
            with c2:
                if st.button("🗑️ 重置", use_container_width=True):
                    st.session_state.report = ""
                    st.session_state.sources = []
                    st.rerun()
        else:
            st.info("请在左侧录入处方信息并点击“执行深度点评”开始。")

if __name__ == "__main__":
    main()
