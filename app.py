import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import streamlit as st

# ==================== STEP 1: PAGE CONFIG ====================
st.set_page_config(page_title="Chat-With-PDF", layout="wide")
st.sidebar.title("SET API CONFIG")
st.title("RAG Based Chat With PDF 📚")

# ==================== STEP 2: API KEY ====================
GOOGLE_API_KEY = st.sidebar.text_input("GOOGLE_API_KEY", type="password")
if GOOGLE_API_KEY:
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    st.sidebar.success("API key Loaded!!")
else:
    st.sidebar.info("Give API key")

# ==================== STEP 3: UPLOAD + SAVE PDF ====================
uploaded_file = st.sidebar.file_uploader("Upload PDF File", type=["pdf"])

file_path = None
if uploaded_file is not None:
    with st.spinner("Reading PDF File"):
        data = uploaded_file.read()
        st.sidebar.pdf(data)

    save_dir = "pdf_files"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    file_path = os.path.join(save_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(data)
    st.write(f"Saved file to: {file_path}")

# Stop early if we don't have what we need yet, instead of crashing further down
if not GOOGLE_API_KEY:
    st.warning("Please enter your Google API key in the sidebar to continue.")
    st.stop()

if uploaded_file is None:
    st.info("Please upload a PDF file in the sidebar to continue.")
    st.stop()

# ==================== STEP 4: CACHED RESOURCES ====================
# NOTE: file_path is now passed IN as an argument, so the cache correctly
# invalidates whenever a different file is uploaded.

@st.cache_data
def load_documents(path):
    loader = PyPDFLoader(path)
    documents = loader.load()
    return documents

@st.cache_resource
def load_embedding():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return embeddings

@st.cache_data
def get_splitted_chunks(_documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = splitter.split_documents(_documents)
    return chunks

# ==================== STEP 5: BUILD VECTOR DB ====================
with st.spinner("Processing document..."):
    documents = load_documents(file_path)
    embeddings = load_embedding()
    chunks = get_splitted_chunks(documents)

@st.cache_resource
def create_vector_db(_chunks, _embeddings, cache_key):
    # cache_key ties the cache to the current file, since _chunks/_embeddings
    # are prefixed with underscore and therefore NOT hashed by Streamlit
    vectorstore = FAISS.from_documents(_chunks, _embeddings)
    vectorstore.save_local("faiss_index")
    return vectorstore

vectorstore = create_vector_db(chunks, embeddings, cache_key=uploaded_file.name)

k_slider = st.sidebar.slider("Select Top K-Value", min_value=1, max_value=10, value=4)
retriever = vectorstore.as_retriever(search_kwargs={"k": k_slider})

# ==================== STEP 6: LCEL RAG CHAIN ====================
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")  # verify this model name is available to your key

prompt = ChatPromptTemplate.from_template("""
Answer the question using ONLY the context below.
If the answer isn't in the context, say "I don't know based on the document."
Context:
{context}
Question: {question}
""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# ==================== STEP 7: USER INPUT ====================
user_question = st.text_area("Ask Question: ")
if user_question:
    if st.button("Get Answer"):
        with st.spinner("Wait.."):
            st.write_stream(rag_chain.stream(user_question))
