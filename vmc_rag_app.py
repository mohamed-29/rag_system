
import os
import shutil
import re
from typing import List

# Third-party imports
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableWithMessageHistory, Runnable
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

# --- Configuration ---
PDF_PATH = "VMC-Upper computer_V3.0_0411.pdf"
DB_PATH = "./chroma_vmc_prod_db"
# API Key should be set in your environment: export GROQ_API_KEY="your-key"
# os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")

# Global store for chat history
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# --- 1. Data Extraction ---
def extract_text_from_pdf(pdf_path):
    print(f"Extracting text from {pdf_path}...")
    try:
        reader = PdfReader(pdf_path)
        full_text = ""
        for page in reader.pages:
            full_text += page.extract_text() + "\n"
        return full_text
    except FileNotFoundError:
        print(f"Error: File {pdf_path} not found.")
        return ""

# --- 2. Custom Parsing Logic ---
class VMCProtocolParser:
    """
    Custom Logic to parse the VMC Protocol text dump.
    Splits by section headers to keep command definitions intact.
    """
    def __init__(self, raw_text: str):
        self.raw_text = raw_text

    def clean_text(self, text: str) -> str:
        return text

    def parse(self) -> List[Document]:
        clean_text = self.clean_text(self.raw_text)
        
        # Regex to split by section headers (e.g., "\n4.1.1 ")
        fragments = re.split(r'(\n\s*\d+\.\d+\.\d+\s+)', clean_text)
        
        docs = []
        for i in range(1, len(fragments), 2):
            header = fragments[i]
            body = fragments[i+1] if i+1 < len(fragments) else ""
            full_section = header + body
            
            # Extract Metadata: Hex Code
            hex_match = re.search(r'\(0x([0-9A-Fa-f]{2})\)', full_section)
            hex_code = f"0x{hex_match.group(1)}" if hex_match else "N/A"
            title_line = header.strip().split('\n')[0]
            
            metadata = {
                "hex_code": hex_code,
                "section_title": title_line[:50], 
                "source": "VMC_Protocol"
            }
            docs.append(Document(page_content=full_section.strip(), metadata=metadata))
            
        return docs

# --- 3. ETL & Vector Store Creation ---
def setup_vector_store(rebuild: bool = False):
    print("Initializing Embeddings (all-MiniLM-L6-v2)...")
    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    if os.path.exists(DB_PATH) and not rebuild:
        print(f"Loading existing Vector Store from {DB_PATH}...")
        return Chroma(persist_directory=DB_PATH, embedding_function=embedding_function)

    print("Rebuilding Vector Store...")
    raw_text = extract_text_from_pdf(PDF_PATH)
    if not raw_text:
        return None

    parser = VMCProtocolParser(raw_text)
    documents = parser.parse()
    print(f"Parsed {len(documents)} logic protocol sections.")

    if os.path.exists(DB_PATH):
        print("Cleaning up old Vector Store...")
        try:
            shutil.rmtree(DB_PATH)
        except Exception as e:
            print(f"Warning: Failed to delete {DB_PATH}: {e}")

    print("Creating Vector Store with Cosine Similarity...")
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embedding_function,
        persist_directory=DB_PATH,
        collection_metadata={"hnsw:space": "cosine"}
    )
    return vector_store

# --- 4. Custom MultiQuery Implementation ---
class SimpleMultiQueryRetriever:
    """
    A simplified MultiQueryRetriever acting as a Runnable.
    Generates variations of the question to improve retrieval recall.
    """
    def __init__(self, base_retriever, llm):
        self.base_retriever = base_retriever
        self.llm = llm
        self.parser = StrOutputParser()
        self.prompt = ChatPromptTemplate.from_template(
            """You are an AI language model assistant. Your task is to generate 3 different versions of the given user question to retrieve relevant documents from a vector database. 
            By generating multiple perspectives on the user question, your goal is to help the user overcome some of the limitations of the distance-based similarity search. 
            Provide these alternative questions separated by newlines. 
            Original question: {question}"""
        )
        self.chain = self.prompt | self.llm | self.parser

    def invoke(self, input_data, config=None):
        # Handle both string input and dict input
        if isinstance(input_data, dict):
            question = input_data.get("question", "")
        else:
            question = str(input_data)

        # Generate variations
        response = self.chain.invoke({"question": question})
        queries = [q.strip() for q in response.split('\n') if q.strip()]
        queries.append(question) # Include original

        # Retrieve and Deduplicate
        unique_docs = {}
        for q in queries:
            docs = self.base_retriever.invoke(q)
            for doc in docs:
                # Use page_content as key for deduplication
                if doc.page_content not in unique_docs:
                    unique_docs[doc.page_content] = doc
        
        return list(unique_docs.values())

# --- 5. RAG Chain Construction ---
def build_rag_chain(vector_store):
    # --- 1. The "Typos & Intent" Fix (MultiQuery) ---
    llm_for_retrieval = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0
    )
    
    base_retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": 0.3, "k": 5}
    )
    
    # Use our Manual Implementation to avoid ImportErrors
    retriever = SimpleMultiQueryRetriever(
        base_retriever=base_retriever,
        llm=llm_for_retrieval
    )

    # --- 2. The "Senior Engineer" Persona ---
    template = """You are a Senior Embedded Systems Engineer helping a colleague. 
The user might use slang, have typos, or ask vague questions. Your job is to understand their INTENT.

CONTEXT FROM SPECS:
{context}

CHAT HISTORY:
{chat_history}

USER QUESTION:
{question}

GUIDELINES:
1. **Be Conversational**: Don't say "Based on the context". Just say "Oh, you're looking for Command 0x22."
2. **Infer Typos**: If they say "dispns", they mean "Dispense (0x03)". If they say "bill reader", they mean "Bill Acceptor (0x21/0x22)".
3. **Show, Don't Just Tell**: If the protocol is complex, briefly explain *why* (e.g., "Watch out for the XOR checksum here").
4. **Code**: If they ask for code, give a clean Python `struct` example.

Answer:
"""
    prompt = ChatPromptTemplate.from_template(template)
    
    llm = ChatGroq(
        model="llama-3.3-70b-versatile", 
        temperature=0.1
    )

    # --- 3. The Conversational Chain ---
    def format_docs(docs):
        return "\n---\n".join([d.page_content for d in docs]) if docs else "No specific docs found, rely on general VMC knowledge if safe."

    rag_chain = (
        RunnablePassthrough.assign(
            context=lambda x: format_docs(retriever.invoke(x["question"]))
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    # Wrap with History
    conversational_rag = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )

    return conversational_rag

# --- 6. Initialization Wrapper ---
def initialize_rag_system():
    print("Initializing VMC RAG System (Conversational Mode)...")
    vector_store = setup_vector_store(rebuild=False)
    if not vector_store:
        raise Exception("Failed to initialize vector store.")
    return build_rag_chain(vector_store)

# --- 7. Main Execution ---
def main():
    print("--- VMC Engineer AI (Conversational Mode) ---")
    
    try:
        rag_chain = initialize_rag_system()
        # Create a session ID for the conversation
        session_id = "engineer_session_1"
    except Exception as e:
        print(f"Startup Error: {e}")
        return

    print("\nSystem Ready. I'm listening. (Type 'exit' to quit)")
    
    while True:
        try:
            user_input = input("\nYOU: ")
            if user_input.lower() in ["exit", "q", "quit"]:
                break
            
            print("-" * 30)
            # Pass the session_id to maintain context
            response = rag_chain.invoke(
                {"question": user_input},
                config={"configurable": {"session_id": session_id}}
            )
            print(f"AI: {response}")
            print("=" * 50)
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
