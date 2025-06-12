import streamlit as st
from streamlit_chat import message
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.callbacks.manager import get_openai_callback
import openai, tempfile, os




# ————— Streamlit Page Config —————
st.set_page_config(page_title="AI PDF Chatbot", page_icon="🤖", layout="wide")

# ————— Sidebar Settings —————
st.sidebar.title("🛠️ Chatbot Settings")
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7, 0.05)
top_p = st.sidebar.slider("Top-p", 0.0, 1.0, 0.95, 0.05)
model_name = st.sidebar.selectbox("Model", ["gpt-3.5-turbo", "gpt-4"])

# ————— API Key —————
api_key = st.secrets["API_KEY"]
openai.api_key = api_key
if not api_key:
    st.error("❌ No API key found. Set OPENAI_API_KEY env var or add to Streamlit secrets.")
    st.stop()
openai.api_key = api_key

# ————— Initialize embeddings & LLM —————
embed_model = OpenAIEmbeddings(openai_api_key=api_key)
llm = ChatOpenAI(
    openai_api_key=api_key,
    model_name=model_name,
    temperature=temperature,
    top_p=top_p,
)



# ————— Page Title —————
st.title("📄 AI Chatbot for PDF Documents")
st.caption("Upload a PDF and ask questions. Answers are retrieval-augmented (RAG).")

# ————— File Uploader —————
uploaded_file = st.file_uploader("📤 Upload your PDF", type="pdf")

# ————— Session State —————
if "chain" not in st.session_state:
    st.session_state.chain = None
    st.session_state.history = []
    st.session_state.token_total = 0

# ————— Load PDF and prepare QA chain —————
if uploaded_file and st.session_state.chain is None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        pdf_path = tmp.name
    loader = PyPDFLoader(pdf_path)
    raw_docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.split_documents(raw_docs)
    vectorstore = FAISS.from_documents(docs, embed_model)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    st.session_state.chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        verbose=False,
    )
    st.success("✅ PDF processed. You can now ask questions.")

# ————— Render History —————
for idx, msg in enumerate(st.session_state.history):
    message(msg['content'], is_user=(msg['role']=='user'), key=f"hist_{idx}")

# ————— Chat Input & Streaming —————
if st.session_state.chain:
    user_input = st.chat_input("💬 Your question...")
    if user_input:
        # display user
        st.session_state.history.append({'role':'user', 'content':user_input})
        message(user_input, is_user=True, key=f"hist_{len(st.session_state.history)-1}")

        # retrieve
        docs_ctx = st.session_state.chain.retriever.get_relevant_documents(user_input)
        context = "\n\n".join(d.page_content for d in docs_ctx)
        messages_for_llm = [
            {'role':'system', 'content':'Use the context to answer. you must be friendly and accurate'},
            {'role':'system', 'content':context},
            {'role':'user', 'content':user_input},
        ]

        placeholder = st.empty()
        assistant_text = ""
        with get_openai_callback() as cb:
            for chunk in openai.ChatCompletion.create(
                model=model_name,
                messages=messages_for_llm,
                temperature=temperature,
                top_p=top_p,
                stream=True
            ):
                token = chunk['choices'][0]['delta'].get('content','')
                assistant_text += token
                placeholder.markdown(assistant_text)

        # record assistant
        st.session_state.history.append({'role':'assistant','content':assistant_text})

# ————— Prompt if no PDF yet —————
if st.session_state.chain is None and not st.session_state.history:
    st.info("📝 Please upload a PDF to begin.")
