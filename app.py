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
from docx import Document  # 需要安装 python-docx
from docx.shared import Pt, Inches

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

# --- 3. 核心工具函数 ---

def generate_docx(text):
    """将 Markdown 文本转换为 Word 文档"""
    doc = Document()
    doc.add_heading('处方点评报告', 0)
    
    # 设置全文字体
    style = doc.styles['Normal']
    style.font.name = '微软雅黑'
    style.font.size = Pt(11)

    # 简单处理 Markdown 行
    for line in text.split('\n'):
        line = line.strip()
        if line.startswith('###'):
            doc.add_heading(line.replace('#', '').strip(), level=2)
        elif line.startswith('##'):
            doc.add_heading(line.replace('#', '').strip(), level=1)
        elif line.startswith('- ') or line.startswith('* '):
            doc.add_paragraph(line[2:], style='List Bullet')
        elif line:
            doc.add_paragraph(line)
            
    bio = BytesIO()
    doc.save(bio)
    bio.seek(0)
    return bio

class KnowledgeManager:
    def __init__(self, model_name, db_path):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'}
        )
        self.vectorstore = Chroma(persist_directory=db_path, embedding_function=self.embeddings)

    def upload_docs(self, file_path):
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
        self.vectorstore.add_documents(splitter.split_documents(docs))
        return len(docs)

    def retrieve_context(self, drug_names):
        all_context = []
        for name in drug_names:
            if not name: continue
            results = self.vectorstore.similarity_search(name, k=4)
            for res in results:
                source = os.path.basename(res.metadata.get('source', '资料库'))
                all_context.append(f"【{source}】: {res.page_content}")
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
        system_prompt = """你是一位资深临床药师。请为【处方数据】输出一份正式的点评报告。
        要求：
        1. 格式清晰，使用 Markdown 标题。
        2. 如果患者是儿童，必须核算 mg/kg 剂量。
        3. 如果是成人，重点评估适应症、药物相互作用和医保。
        4. 结论必须明确：[合理/不合理/用药不适宜]。
        5. 对不合理项给出明确的临床建议。"""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "【参考资料库】:\n{context}\n\n【处方数据】:\n{prescription}")
        ])
        chain = prompt | self.llm
        return chain.invoke({
            "context": context, 
            "prescription": json.dumps(prescription_json, ensure_ascii=False)
        }).content

# --- 4. Streamlit UI ---

def main():
    st.set_page_config(page_title="AI 药师专家系统", layout="wide", page_icon="⚖️")
    
    @st.cache_resource
    def get_km():
        return KnowledgeManager("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", DB_PATH)
    
    km = get_km()

    # --- 侧边栏 ---
    with st.sidebar:
        st.header("⚙️ 系统配置")
        api_key = st.text_input("DeepSeek API Key:", type="password")
        st.divider()
        st.header("📚 说明书知识库")
        files = st.file_uploader("上传 PDF 说明书", accept_multiple_files=True)
        if files and st.button("同步知识"):
            for f in files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(f.getvalue())
                    km.upload_docs(tmp.name)
            st.success("知识库同步完成")

    # --- 主界面 ---
    st.title("🏥 临床药师审方及点评专家平台")

    if "report_content" not in st.session_state:
        st.session_state.report_content = ""

    col_in, col_out = st.columns([1, 1.3])

    # --- 左侧：录入 ---
    with col_in:
        st.subheader("📋 处方录入")
        with st.container(border=True):
            # 成人/儿童逻辑切换
            p_type = st.radio("患者类型", ["儿童 (需计算体重剂量)", "成人"], horizontal=True)
            
            c1, c2 = st.columns(2)
            age = c1.number_input("年龄", value=30 if p_type == "成人" else 6, min_value=0)
            
            weight = None
            if p_type == "儿童 (需计算体重剂量)":
                weight = c2.number_input("体重 (kg)", value=20.0, step=0.1)
            else:
                c2.info("💡 成人无需输入体重")

            diag = st.text_input("临床诊断", "急性支气管炎")

            st.markdown("**💊 药品清单**")
            default_data = [{"药品名称": "阿奇霉素干混悬剂", "单次剂量": "0.25g", "频次": "QD", "用法": "口服"}]
            med_df = st.data_editor(pd.DataFrame(default_data), num_rows="dynamic", use_container_width=True)

            if st.button("🔍 执行 AI 深度点评", type="primary", use_container_width=True):
                if not api_key:
                    st.error("请先输入 API Key")
                else:
                    agent = PharmacyAgent(api_key, "https://api.deepseek.com")
                    with st.spinner("临床推理中..."):
                        context = km.retrieve_context(med_df["药品名称"].tolist())
                        prescription = {
                            "patient": {"age": age, "weight": weight, "diagnosis": diag, "type": p_type},
                            "medications": med_df.to_dict('records')
                        }
                        st.session_state.report_content = agent.audit(prescription, context)

    # --- 右侧：点评报告 ---
    with col_out:
        st.subheader("📝 处方点评报告")
        if st.session_state.report_content:
            # 1. 可编辑区域
            edited_report = st.text_area(
                "手动修正区（修改后将实时同步至导出文件）：",
                value=st.session_state.report_content,
                height=550
            )
            st.session_state.report_content = edited_report
            
            # 2. 导出与操作区
            st.divider()
            ctrl_c1, ctrl_c2, ctrl_c3 = st.columns(3)
            
            with ctrl_c1:
                # 预览切换
                if st.toggle("预览渲染后的格式", value=True):
                    with st.expander("报告预览", expanded=True):
                        st.markdown(st.session_state.report_content)
            
            with ctrl_c2:
                # Word 导出
                docx_file = generate_docx(st.session_state.report_content)
                st.download_button(
                    label="📥 导出为 Word 文档",
                    data=docx_file,
                    file_name=f"点评报告_{datetime.now().strftime('%m%d_%H%M')}.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )
            
            with ctrl_c3:
                if st.button("🗑️ 清空报告"):
                    st.session_state.report_content = ""
                    st.rerun()
        else:
            st.info("请在左侧录入处方并点击“执行 AI 深度点评”")
            st.empty()

if __name__ == "__main__":
    main()
