import os
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# STEP 1: PAGE CONFIGURATION
st.set_page_config(
    page_title="Chat With PDF",
    page_icon="📚",
    layout="wide"
)
st.title("RAG Based Chat With PDF 📚")

# STEP 2: API KEY
st.sidebar.title("SET API CONFIG")

GOOGLE_API_KEY = st.sidebar.text_input(
    "GOOGLE_API_KEY",
    type="password"
)

if GOOGLE_API_KEY:
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    st.sidebar.success("API Key Loaded!!")
else:
    st.sidebar.info("Enter your Google API Key")
  
# STEP 3: UPLOAD PDF
uploaded_file = st.sidebar.file_uploader(
    "Upload PDF File",
    type=["pdf"]
)


if uploaded_file is not None:
    save_dir = "pdf_files"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_path = os.path.join(
        save_dir,
        uploaded_file.name
    )
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"PDF uploaded: {uploaded_file.name}")

    # STEP 4: LOAD PDF
    with st.spinner("Reading PDF..."):
        loader = PyPDFLoader(file_path)
        documents = loader.load()
    st.success(f"PDF loaded successfully! Pages: {len(documents)}")

    # STEP 5: SPLIT DOCUMENT INTO CHUNKS
    with st.spinner("Splitting document into chunks..."):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = splitter.split_documents(documents)
    st.success(f"Created {len(chunks)} chunks")
    # STEP 6: CREATE EMBEDDINGS
    with st.spinner("Loading embedding model..."):
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    # STEP 7: CREATE VECTOR DATABASE
    with st.spinner("Creating FAISS Vector Database..."):
        vectorstore = FAISS.from_documents(
            chunks,
            embeddings
        )
    st.success("Vector database created successfully!")
    # STEP 8: RETRIEVER
    k_value = st.sidebar.slider(
        "Select Top K-Value",
        min_value=1,
        max_value=10,
        value=4
    )
    retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": k_value
        }
    )

    # STEP 9: GOOGLE GEMINI LLM
    if not GOOGLE_API_KEY:
        st.warning("Please enter your Google API Key in the sidebar.")
        st.stop()

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=GOOGLE_API_KEY,
        temperature=0
    )
    # STEP 10: PROMPT
    prompt = ChatPromptTemplate.from_template(
        """  Answer the question using ONLY the context below.
        If the answer isn't in the context, say:
        "I don't know based on the document."
        Context:
        {context}
        Question:
        {question}
        """
    )

    # STEP 11: FORMAT DOCUMENTS
    def format_docs(docs):
        return "\n\n".join(
            doc.page_content
            for doc in docs
        )

    # STEP 12: CREATE RAG CHAIN

    with st.spinner("Building RAG Chain..."):
        rag_chain = (
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough()
            }
            | prompt
            | llm
            | StrOutputParser()
# STEP 13: ASK QUESTION
user_question = st.text_area("Ask Question:")

if user_question:
    if st.button("Get Answer"):
        with st.spinner("Wait..."):
            st.write_stream(
                rag_chain.stream(user_question)
            )
