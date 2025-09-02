import json
import os
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

# Load environment variables
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# Load Dataset
with open("indian_constitution.json", "r", encoding="utf-8") as f:
    constitution_data = json.load(f)

print(f"📚 Loaded {len(constitution_data)} articles from constitution dataset")

# Prepare documents with all relevant fields
documents = []
for article in constitution_data:
    # Include all important fields from your dataset
    text = (
        f"Article Number: {article.get('article_number', '')}\n"
        f"Article Title: {article.get('article_title', '')}\n"
        f"Description: {article.get('description', '')}\n"
        f"Constitutional Principle: {article.get('constitutional_principle', '')}\n"
        f"Simple Explanation: {article.get('simple_explanation', '')}\n"
        f"Examples: {', '.join(article.get('examples', []))}\n"
        f"Related Articles: {', '.join(article.get('related_articles', []))}"
    )
    documents.append(text)

print(f"📝 Prepared {len(documents)} formatted documents")

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

print(f"✂️ Created {len(docs)} document chunks after splitting")

# Show sample chunk for verification
print(f"\n📖 Sample chunk preview:")
print(f"{docs[0][:300]}..." if docs else "No chunks created!")

# Embeddings
print(f"\n🧠 Creating embeddings using sentence-transformers...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)

# Build FAISS Vector Store
print(f"🔍 Building FAISS vector store...")
vectorstore = FAISS.from_texts(docs, embeddings)
vectorstore.save_local("faiss_constitution_index")
print(f"💾 FAISS index saved successfully!")

# Build BM25 Retriever for Keyword Search
print(f"🔎 Building BM25 keyword retriever...")
bm25_retriever = BM25Retriever.from_texts(docs)
bm25_retriever.k = 5  # Return top 5 keyword matches

# Build Vector Retriever for Semantic Search  
print(f"🌐 Building semantic vector retriever...")
vector_retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

# 🎯 THE MAGIC: Hybrid Ensemble Retriever
print(f"✨ Creating HYBRID SEARCH ENSEMBLE...")
hybrid_retriever = EnsembleRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    weights=[0.7, 0.3],  # 70% semantic + 30% keyword matching
    c=60  # Number of documents to consider for re-ranking
)

print(f"🔥 Hybrid retriever created! Combining semantic + keyword search")

# LLM (Groq)
llm = ChatGroq(
    model="llama-3.1-8b-instant", 
    temperature=0.2, 
    api_key=api_key
)

# Enhanced prompt template
prompt_template = """You are an expert constitutional lawyer and scholar of the Indian Constitution. 

Use the following context from the Indian Constitution to provide a comprehensive and accurate answer.

Context: {context}

Question: {question}

Instructions:
- Provide detailed explanation based on the context
- Include article numbers, titles, and key principles
- Mention related articles if relevant  
- Use simple language while maintaining accuracy
- If context is insufficient, clearly state what information is missing

Answer:"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# 🚀 RAG Pipeline with HYBRID SEARCH
qa_hybrid = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=hybrid_retriever,  # Using hybrid retriever!
    chain_type="stuff",
    chain_type_kwargs={"prompt": PROMPT},
    return_source_documents=True
)

# Also create traditional semantic-only pipeline for comparison
qa_semantic = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vector_retriever,  # Traditional semantic only
    chain_type="stuff", 
    chain_type_kwargs={"prompt": PROMPT},
    return_source_documents=True
)

print(f"\n🎉 Both pipelines ready! Let's test the magic...")

# Test queries showcasing hybrid search advantages
test_queries = [
    "What is Article 370 of the Indian Constitution?"
    # "Tell me about CAG Article 148",
    # "Right to Equality fundamental rights",
    # "Article 19 freedom of speech",
    # "Jammu Kashmir special status provisions"
]

for query in test_queries:
    print(f"\n{'='*80}")
    print(f"🔍 TESTING QUERY: {query}")
    print(f"{'='*80}")
    
    # Test Hybrid Search
    print(f"\n🎯 HYBRID SEARCH RESULTS:")
    print(f"{'-'*50}")
    try:
        hybrid_response = qa_hybrid({"query": query})
        print(f"Answer: {hybrid_response['result']}")
        
        print(f"\n📚 Hybrid Source Documents:")
        for i, doc in enumerate(hybrid_response['source_documents']):
            print(f"  {i+1}. {doc.page_content[:150]}...")
            
    except Exception as e:
        print(f"❌ Hybrid Error: {e}")
    
    # Test Traditional Semantic Search for Comparison
    print(f"\n🌐 TRADITIONAL SEMANTIC SEARCH RESULTS:")
    print(f"{'-'*50}")
    try:
        semantic_response = qa_semantic({"query": query})
        print(f"Answer: {semantic_response['result']}")
        
        print(f"\n📚 Semantic Source Documents:")
        for i, doc in enumerate(semantic_response['source_documents']):
            print(f"  {i+1}. {doc.page_content[:150]}...")
            
    except Exception as e:
        print(f"❌ Semantic Error: {e}")

    print(f"\n💡 COMPARISON INSIGHT:")
    print(f"   Hybrid combines exact keyword matching + semantic understanding")
    print(f"   Traditional uses only semantic similarity")

# 🔬 DEEP DIVE: Show Individual Retriever Results
print(f"\n{'='*80}")
print(f"🔬 DEEP DIVE: Individual Retriever Analysis")
print(f"{'='*80}")

test_query = "Article 370 Jammu Kashmir"
print(f"Test Query: '{test_query}'")

# BM25 (Keyword) Results
print(f"\n🔤 BM25 KEYWORD SEARCH Results:")
bm25_results = bm25_retriever.get_relevant_documents(test_query)
for i, doc in enumerate(bm25_results[:3]):
    print(f"  BM25-{i+1}: {doc.page_content[:200]}...")

# Vector (Semantic) Results  
print(f"\n🧠 SEMANTIC VECTOR SEARCH Results:")
vector_results = vector_retriever.get_relevant_documents(test_query)
for i, doc in enumerate(vector_results[:3]):
    print(f"  Vector-{i+1}: {doc.page_content[:200]}...")

# Hybrid (Combined) Results
print(f"\n✨ HYBRID ENSEMBLE Results:")
hybrid_results = hybrid_retriever.get_relevant_documents(test_query)
for i, doc in enumerate(hybrid_results[:3]):
    print(f"  Hybrid-{i+1}: {doc.page_content[:200]}...")

print(f"\n🎊 HYBRID SEARCH PIPELINE COMPLETE!")
print(f"🚀 Ready to answer constitutional questions with enhanced accuracy!")

# Final interactive test
print(f"\n{'='*80}")
print(f"🎮 INTERACTIVE TEST MODE")
print(f"{'='*80}")

while True:
    user_query = input(f"\n💬 Ask about the Indian Constitution (or 'quit' to exit): ")
    
    if user_query.lower() in ['quit', 'exit', 'q']:
        print(f"👋 Goodbye! Happy constitutional learning!")
        break
        
    if user_query.strip():
        try:
            result = qa_hybrid({"query": user_query})
            print(f"\n🎯 HYBRID SEARCH ANSWER:")
            print(f"{result['result']}")
            
            print(f"\n📚 Sources used:")
            for i, doc in enumerate(result['source_documents'][:3]):
                print(f"  {i+1}. {doc.page_content[:100]}...")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    else:
        print(f"⚠️ Please enter a valid question!")