# --- 1. 核心兼容性补丁 (必须放在所有 import 之前) ---
try:
    import pysqlite3
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    # 如果是 Windows 环境或已经自带高版本 sqlite3，忽略此处
    pass

import streamlit as st
import os
import json
import warnings
import tempfile
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
        """初始化向量数据库和嵌入模型"""
        # st.write(f"正在加载嵌入模型...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'}
        )
        self.vectorstore = Chroma(
            persist_directory=db_path,
            embedding_function=self.embeddings
        )

    def upload_docs(self, file_path):
        """处理单个 PDF 并存入数据库"""
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        # 文本切分：800字一段，重叠150字保证上下文连贯
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
        splits = splitter.split_documents(docs)
        self.vectorstore.add_documents(splits)
        return len(splits)

    def retrieve_context(self, query):
        """根据关键词检索最相关的 3 条知识"""
        results = self.vectorstore.similarity_search(query, k=3)
        return "\n".join([res.page_content for res in results])

class PharmacyAgent:
    def __init__(self, api_key, base_url):
        """初始化 DeepSeek LLM"""
        clean_key = str(api_key).strip()
        self.llm = ChatOpenAI(
            model="deepseek-chat",
            api_key=clean_key,
            base_url=base_url,
            temperature=0  # 审方需要严谨，设为 0
        )

    def audit(self, prescription_json, context):
        """执行审核任务"""
        system_prompt = """你是一位资深临床药师。请根据提供的【参考资料】审核【处方数据】。
        审核重点：
        1. 适应症：检查诊断与药品是否匹配。
        2. 儿科剂量：必须根据患者体重(weight)核算剂量是否超标。
        3. 用法用量：检查给药频次。
        4. 医保合规性：检查诊断是否符合医保报销类型。
        
        如果发现问题，请明确指出并给出调整建议；如果没有问题，请回复“审核通过”。"""
        
        prompt = ChatPromptTemplate.from_template(
            system_prompt + "\n\n【参考资料】:\n{context}\n\n【处方数据】:\n{prescription}"
        )
        chain = prompt | self.llm
        return chain.invoke({
            "context": context,
            "prescription": json.dumps(prescription_json, ensure_ascii=False, indent=2)
        }).content

# --- 4. 缓存初始化 ---

@st.cache_resource
def get_knowledge_manager():
    # 使用多语言预训练模型
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    return KnowledgeManager(model_name, DB_PATH)

# --- 5. Streamlit UI 界面 ---

def main():
    st.set_page_config(page_title="AI 药师审方系统", layout="wide", page_icon="💊")
    
    # 初始化知识库管理工具
    km = get_knowledge_manager()

    # --- 侧边栏：登录逻辑 ---
    st.sidebar.title("🔐 系统登录")
    
    if 'api_key' not in st.session_state:
        st.session_state['api_key'] = ""

    input_key = st.sidebar.text_input(
        "请输入 DeepSeek API Key:", 
        type="password", 
        placeholder="sk-...",
        value=st.session_state['api_key']
    )

    if not input_key:
        st.info("👋 欢迎！请在侧边栏输入您的 DeepSeek API Key 以激活 AI 药师。")
        st.stop()
    else:
        st.session_state['api_key'] = input_key
        agent = PharmacyAgent(st.session_state['api_key'], "https://api.deepseek.com")

    # --- 主界面 ---
    st.title("🏥 药剂科 AI 处方审核平台")
    st.markdown("---")

    # --- 侧边栏：知识库管理 (批量上传) ---
    with st.sidebar:
        st.header("📂 知识库管理")
        uploaded_files = st.file_uploader(
            "批量上传药品说明书 (PDF)", 
            type="pdf", 
            accept_multiple_files=True
        )
        
        if uploaded_files:
            if st.button("✨ 立即同步所有知识"):
                total_splits = 0
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, uploaded_file in enumerate(uploaded_files):
                    status_text.text(f"正在处理: {uploaded_file.name} ({i+1}/{len(uploaded_files)})")
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded_file.getvalue())
                        tmp_path = tmp.name
                    
                    try:
                        count = km.upload_docs(tmp_path)
                        total_splits += count
                    except Exception as e:
                        st.error(f"处理 {uploaded_file.name} 出错: {e}")
                    finally:
                        if os.path.exists(tmp_path):
                            os.unlink(tmp_path)
                    
                    progress_bar.progress((i + 1) / len(uploaded_files))
                
                status_text.empty()
                st.success(f"✅ 处理完成！新增 {total_splits} 条知识片段。")
        
        st.markdown("---")
        if st.button("🚪 退出登录"):
            st.session_state['api_key'] = ""
            st.rerun()

    # --- 主界面布局：处方录入与报告 ---
    col_in, col_out = st.columns([1, 1.2])

    with col_in:
        st.subheader("📋 录入处方")
        with st.form("audit_form"):
            r1 = st.columns(2)
            age = r1[0].number_input("年龄", value=5, min_value=0)
            weight = r1[1].number_input("体重 (kg)", value=18.0)
            
            r2 = st.columns(2)
            diagnosis = r2[0].text_input("临床诊断", value="急性支气管炎")
            insurance = r2[1].selectbox("医保类型", ["统筹医保", "自费", "门诊大病"])
            
            st.markdown("---")
            med_name = st.text_input("药品名称", value="阿奇霉素")
            dosage = st.text_input("单次剂量", value="250mg")
            freq = st.text_input("给药频次", value="一日一次")
            
            btn = st.form_submit_button("🧪 提交审核")

    with col_out:
        st.subheader("📝 审核报告")
        if btn:
            prescription = {
                "patient": {
                    "age": age, 
                    "weight": weight, 
                    "diagnosis": diagnosis, 
                    "insurance_type": insurance
                },
                "medications": [
                    {"name": med_name, "dosage": dosage, "frequency": freq}
                ]
            }
            
            with st.spinner("AI 药师正在检索说明书并分析中..."):
                # 1. 检索向量库获取上下文
                context = km.retrieve_context(med_name)
                
                # 2. 调用大模型审核
                try:
                    report = agent.audit(prescription, context)
                    st.markdown("### 诊断结果")
                    st.info(report)
                    st.download_button(
                        "📥 导出报告", 
                        report, 
                        file_name=f"审核报告_{med_name}_{datetime.now().strftime('%Y%m%d')}.txt"
                    )
                except Exception as e:
                    st.error(f"审核失败。原因：{e}")
        else:
            st.info("请在左侧录入处方数据并点击提交。")

if __name__ == "__main__":
    main()
