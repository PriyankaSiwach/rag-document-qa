RAG Document Q&A

Ask questions about any document using AI. Built with LangChain, Pinecone, and OpenAI.

What this does

Instead of asking an AI questions from its general knowledge, this app lets you upload any document and ask questions specifically about that document. The AI only answers from your document — not from the internet or its training data.

How it works
Loads your document and splits it into small chunks
Converts each chunk into embeddings (numbers that represent meaning) using OpenAI
Stores those embeddings in Pinecone vector database
When you ask a question, finds the most relevant chunks using semantic search
Sends those chunks plus your question to GPT to generate an accurate answer
Tech stack
Python
LangChain (document loading, splitting, retrieval chain)
Pinecone (vector database for storing and searching embeddings)
OpenAI API (embeddings + answer generation)
python-dotenv (environment variable management)
What I learned
How RAG (Retrieval Augmented Generation) works end to end
The difference between keyword search and semantic search using embeddings
How vector databases store and retrieve information by meaning not exact words
Why RAG is better than fine-tuning for private or frequently updated documents
How to run it
Clone this repo
Run pip install -r requirements.txt
Create a .env file with your OPENAI_API_KEY and PINECONE_API_KEY
Add your document to the docs folder
Run python3 main.py
Type any question about your document and press Enter
Use cases

This same architecture is used in production for customer support bots, internal knowledge bases, and enterprise search systems.
