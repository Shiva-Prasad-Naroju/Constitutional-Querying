import json
import os
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# Load Dataset
with open("indian_constitution.json", "r", encoding="utf-8") as f:
    constitution_data = json.load(f)

# Prepare documents with all relevant fields
documents = []
for article in constitution_data:
    # Include all important fields from your dataset
    text = (
        f"Article Number: {article.get('article_number', '')}\n"
        f"Article Title: {article.get('article_title', '')}\n"  # Fixed field name
        f"Description: {article.get('description', '')}\n"
        f"Constitutional Principle: {article.get('constitutional_principle', '')}\n"
        f"Simple Explanation: {article.get('simple_explanation', '')}\n"  # Added this field
        f"Examples: {', '.join(article.get('examples', []))}\n"  # Added examples
        f"Related Articles: {', '.join(article.get('related_articles', []))}"
    )
    documents.append(text)

# Text Splitting with larger chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # Increased chunk size
    chunk_overlap=100,  # Increased overlap
    separators=["\n\n", "\n", ". ", " ", ""]
)

docs = []
for d in documents:
    chunks = splitter.split_text(d)
    docs.extend(chunks)

print(f"Created {len(docs)} document chunks ✅")

# Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}  # Explicitly set device
)

# Build FAISS
vectorstore = FAISS.from_texts(docs, embeddings)
vectorstore.save_local("faiss_constitution_index")

# Load FAISS (optional reload)
vectorstore = FAISS.load_local(
    "faiss_constitution_index", 
    embeddings, 
    allow_dangerous_deserialization=True
)

# LLM (Groq) 
llm = ChatGroq(
    model="llama-3.1-8b-instant", 
    temperature=0.2, 
    api_key=api_key
)

# Custom prompt template for better responses
prompt_template = """
You are an expert on the Indian Constitution. Use the following context to answer the question about the Indian Constitution.

Context: {context}

Question: {question}

Please provide a comprehensive answer based on the context provided. If the context contains information about the article mentioned in the question, explain it clearly. If you cannot find relevant information in the context, say "I don't have enough information in the provided context to answer this question."

Answer:"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# RAG Pipeline with custom prompt and more retrieval
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}  # Increased from 3 to 5
    ),
    chain_type="stuff",
    chain_type_kwargs={"prompt": PROMPT},
    return_source_documents=True  # This helps with debugging
)

# Test with multiple queries
test_queries = [
    "Tell me Article discuss about Jammu and Kashmir issue?"
    # "What is Article 370 of the Indian Constitution?",
    # "Tell me about the special status of Jammu and Kashmir",
    # "What happened to Article 370 in 2019?"
]

for query in test_queries:
    print(f"\n{'='*50}")
    print(f"Query: {query}")
    print(f"{'='*50}")
    
    try:
        response = qa({"query": query})
        print("Answer:", response['result'])
        
        # Print source documents for debugging
        print("\nSource documents used:")
        for i, doc in enumerate(response['source_documents']):
            print(f"Doc {i+1}: {doc.page_content[:200]}...")
            
    except Exception as e:
        print(f"Error: {e}")

# Test similarity search directly
print(f"\n{'='*50}")
print("Direct Similarity Search Test:")
print(f"{'='*50}")

search_results = vectorstore.similarity_search("Article 370", k=5)
for i, result in enumerate(search_results):
    print(f"\nResult {i+1}:")
    print(result.page_content)