import os
import time

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

# Step 1 - Load the document
loader = TextLoader("docs/info.txt")
documents = loader.load()

print(f"Loaded {len(documents)} document(s)")

# Step 2 - Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50,
)
chunks = text_splitter.split_documents(documents)

print(f"Created {len(chunks)} chunk(s)")
print(f"First chunk:\n{chunks[0].page_content}")

# Step 3 - Create embeddings and store in Pinecone
pinecone_api_key = os.getenv("PINECONE_API_KEY", "").strip()
pc = Pinecone(api_key=pinecone_api_key)

index_name = "rag-demo"

existing_indexes = [index["name"] for index in pc.list_indexes()]
if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)

embeddings = OpenAIEmbeddings()
PineconeVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
    index_name=index_name,
)

print("All chunks stored in Pinecone successfully")

# Step 4 - Ask a question and get an answer
vectorstore = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings,
)

llm = ChatOpenAI(model="gpt-3.5-turbo")
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer the question using ONLY the context below. "
            "If the answer is not in the context, say \"I don't know\". "
            "Do not use outside knowledge.\n\nContext:\n{context}",
        ),
        ("human", "{question}"),
    ]
)
chain = prompt | llm

print('\nAsk a question about the document (type "quit" to exit)')
while True:
    question = input("\nQuestion: ").strip()
    if question.lower() == "quit":
        print("Goodbye!")
        break
    if not question:
        continue

    relevant_chunks = vectorstore.similarity_search(question, k=3)

    print("\nRetrieved chunks:")
    for i, chunk in enumerate(relevant_chunks, 1):
        print(f"\n--- Chunk {i} ---\n{chunk.page_content}")

    context = "\n\n".join(chunk.page_content for chunk in relevant_chunks)
    response = chain.invoke({"context": context, "question": question})

    print(f"\nAnswer:\n{response.content}")
