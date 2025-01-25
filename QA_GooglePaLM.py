from langchain.vectorstores import FAISS
from langchain.embeddings import GooglePalmEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import GooglePalm
from langchain.prompts import PromptTemplate
from langchain.document_loaders import TextLoader

"""
An LLM is a type of AI model trained on massive amounts of text data to perform natural language processing tasks.

Google PaLM is a LLM developed by Google, designed for advanced natural language understanding and generation tasks and built using Google's Pathways AI architecture, which enables it to generalize across a wide range of tasks more efficiently by using a single model.

LangChain is a framework to simplify development of LLM applications and is widely used for applications such as chatbots, document Q&A systems, and AI-driven automation workflows.
It offers tools for:
- integrating LLMs with data sources (e.g., SQL databases, APIs, or cloud services)
- chain or workflows: where the output of one LLM task becomes the input for another
- Memory: Offers mechanisms for applications to remember past interactions, enabling context-aware conversation
- Retrieval-based Systems: Connects LLMs with vector stores for efficient document search and retrieval.

FAISS (Facebook AI Similarity Search) is an open-source library developed by Meta for efficient similarity search and clustering of dense vectors and is widely used for tasks that involve finding similar items, such as in recommendation systems or document search.
"""

# pip install langchain google-cloud-vectorstore faiss-cpu
# Initialize Google PaLM model via LangChain
palm_api_key = "PALM_API_KEY"
llm = GooglePalm(api_key=palm_api_key)

# Define a custom embedding model for the QA module
embeddings = GooglePalmEmbeddings(api_key=palm_api_key)

# Prepare the QA database and convert it into a vector store
# Example QA database
qa_data = [
    {"question": "What is machine learning?", "answer": "Machine learning is a subset of AI that involves training models to make predictions or decisions without being explicitly programmed."},
    {"question": "How does FAISS work?", "answer": "FAISS is a library for efficient similarity search and clustering of dense vectors."},
    {"question": "What is Google PaLM?", "answer": "Google PaLM is a large language model designed for advanced natural language understanding and generation."},
]

# Convert the QA database into LangChain documents
documents = [f"Question: {item['question']}\nAnswer: {item['answer']}" for item in qa_data]

# Load documents into a FAISS vector store
vector_db = FAISS.from_texts(documents, embeddings)

# Set up the retrieval-based QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vector_db.as_retriever(),
    return_source_documents=True,
)

# Create a QA module function
def ask_question(question):
    result = qa_chain.run(question)
    return result

# Example usage
if __name__ == "__main__":
    user_question = "What is machine learning?"
    response = ask_question(user_question)
    print("Answer:", response)
