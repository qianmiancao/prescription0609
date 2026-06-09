# --- 1. 核心兼容性补丁 (必须放在所有 import 之前) ---
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
        """初始化向量数据库"""
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'}
        )
        self.vectorstore = Chroma(
            persist_directory=db_path,
            embedding_function=self.embeddings
        )

    def upload_docs(self, file_path):
        """处理 PDF 并存入数据库"""
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        # 优化：针对医疗文档缩短 chunk_size，提高检索精度
        splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
        splits = splitter.split_documents(docs)
        self.vectorstore.add_documents(splits)
        return len(splits)

    def retrieve_context(self, drug_names):
        """根据所有药品名称检索综合上下文"""
        all_context = []
        for name in drug_names:
            # 增加检索深度 k=5
            results = self.vectorstore.similarity_search(name, k=5)
            for res in results:
                source = res.metadata.get('source', '未知文档')
                all_context.append(f"【参考源: {os.path.basename(source)}】\n{res.page_content}")
        return "\n\n".join(list(set(all_context))) # 去重

class PharmacyAgent:
    def __init__(self, api_key, base_url):
        """初始化 DeepSeek LLM"""
        self.llm = ChatOpenAI(
            model="deepseek-chat",
            api_key=str(api_key).strip(),
            base_url=base_url,
            temperature=0.1 # 保持严谨但允许一定的逻辑推理
        )

    def audit(self, prescription_json, context):
        """执行细致的处方点评"""
        system_prompt = """你是一位资深临床药师。请根据提供的【参考资料】对【处方数据】进行极度细致的点评。
        
        ### 审核核心维度：
        1. **诊断匹配性**：分析药物适应症是否覆盖临床诊断。
        2. **剂量精算**：如果是儿童(年龄<18)，必须基于体重计算单次剂量是否符合说明书范围。
        3. **给药频次与途径**：检查 QD/BID/TID 及静脉/口服的合理性。
        4. **相互作用**：若有多项药品，评估是否存在配伍禁忌或药物相互作用。
        5. **医保合规**：根据医保类型判断报销合规性。

        ### 输出格式（Markdown）：
        ## 📑 处方点评报告
        ---
        ### 1️⃣ 基本信息与风险评估
        - **风险等级**：[🟢正常 / 🟡风险 / 🔴严重不合理]
        - **患者概况**：年龄{age}岁，体重{weight}kg。

        ### 2️⃣ 详细审核维度表
        | 维度 | 结论 | 药师详细分析理由 |
        | :--- | :--- | :--- |
        | 适应症匹配 | ✅/❌ | ... |
        | 剂量准确性 | ✅/❌ | ... |
        | 用法用量合理性 | ✅/❌ | ... |
        | 药物相互作用 | ✅/❌ | ... |
        | 医保报销合规 | ✅/❌ | ... |

        ### 3️⃣ 综合药师意见
        - **存在问题**：(列出具体问题，若无则填“无”)
        - **改进建议**：(给出调整后的具体用法或换药建议)

        ### 4️⃣ 法律及证据来源
        - 依据说明书片段：...
        """
        
        # 填充患者基本信息到 prompt 模板
        patient = prescription_json['patient']
        filled_system_prompt = system_prompt.format(age=patient['age'], weight=patient['weight'])

        prompt = ChatPromptTemplate.from_messages([
            ("system", filled_system_prompt),
            ("user", "【参考资料】:\n{context}\n\n【处方数据】:\n{prescription}")
        ])
        
        chain = prompt | self.llm
        return chain.invoke({
            "context": context,
            "prescription": json.dumps(prescription_json, ensure_ascii=False, indent=2)
        }).content

# --- 4. 缓存与初始化 ---

@st.cache_resource
def get_knowledge_manager():
    # 使用多语言模型处理中文医药术语
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    return KnowledgeManager(model_name, DB_PATH)

# --- 5. Streamlit UI 界面 ---

def main():
    st.set_page_config(page_title="AI 临床药师审方系统", layout="wide", page_icon="💊")
    km = get_knowledge_manager()

    # --- 侧边栏 ---
    with st.sidebar:
        st.title("🔐 系统接入")
        input_key = st.text_input("DeepSeek API Key:", type="password", placeholder="sk-...")
        
        st.markdown("---")
        st.header("📂 说明书知识库")
        uploaded_files = st.file_uploader("上传药品说明书 (PDF)", type="pdf", accept_multiple_files=True)
        
        if uploaded_files and st.button("✨ 索引新知识"):
            with st.status("正在解析医学文档...", expanded=True) as status:
                for f in uploaded_files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(f.getvalue())
                        count = km.upload_docs(tmp.name)
                        st.write(f"✅ {f.name}: 已提取 {count} 个知识点")
                    os.unlink(tmp.name)
                status.update(label="索引同步完成！", state="complete")

        if st.button("🚪 登出系统"):
            st.rerun()

    # --- 主界面 ---
    st.title("🏥 药剂科 AI 临床处方审核平台")
    st.caption("基于 DeepSeek-V3 引擎 & RAG 知识检索库")

    if not input_key:
        st.warning("⚠️ 请在侧边栏输入 API Key 以开始工作。")
        st.stop()

    agent = PharmacyAgent(input_key, "https://api.deepseek.com")

    # --- 界面布局 ---
    col_in, col_out = st.columns([1, 1.2])

    with col_in:
        st.subheader("📋 录入待审核处方")
        with st.container(border=True):
            r1 = st.columns(2)
            age = r1[0].number_input("患者年龄", value=6, min_value=0)
            weight = r1[1].number_input("患者体重 (kg)", value=22.0)
            
            r2 = st.columns(2)
            diagnosis = r2[0].text_input("临床诊断", value="社区获得性肺炎")
            insurance = r2[1].selectbox("医保类型", ["统筹医保", "自费", "门诊大病"])
            
            st.markdown("**药品清单**")
            # 使用 data_editor 实现多药品输入
            df_init = pd.DataFrame([
                {"药品名称": "阿奇霉素干混悬剂", "单次剂量": "0.22g", "频次": "QD", "用法": "口服"},
                {"药品名称": "布地奈德混悬液", "单次剂量": "1mg", "频次": "BID", "用法": "雾化吸入"}
            ])
            med_df = st.data_editor(df_init, num_rows="dynamic", use_container_width=True)
            
            submit_btn = st.button("🧪 开始深度审核", type="primary", use_container_width=True)

    with col_out:
        st.subheader("📝 药师审核报告")
        if submit_btn:
            # 转换数据格式
            med_list = med_df.to_dict('records')
            drug_names = [m['药品名称'] for m in med_list if m['药品名称']]
            
            prescription_data = {
                "patient": {"age": age, "weight": weight, "diagnosis": diagnosis, "insurance_type": insurance},
                "medications": med_list
            }
            
            with st.spinner("🔍 正在检索说明书并执行临床推理..."):
                # 1. 检索所有相关药品的上下文
                context = km.retrieve_context(drug_names)
                
                # 2. 调用 LLM 审核
                try:
                    report = agent.audit(prescription_data, context)
                    
                    # 展示报告
                    st.markdown(report)
                    
                    # 辅助功能
                    st.divider()
                    st.caption(f"审核时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 电子签名: AI-Pharmacist-001")
                    
                    st.download_button(
                        label="📥 导出药学点评报告",
                        data=report,
                        file_name=f"点评报告_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                        mime="text/markdown"
                    )
                except Exception as e:
                    st.error(f"审核过程中发生错误: {str(e)}")
        else:
            st.info("👈 请在左侧完善处方信息并点击提交。")

if __name__ == "__main__":
    main()
