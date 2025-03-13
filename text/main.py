# Langchain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings

# Database imports
import psycopg2 

# Other imports
import os
from dotenv import load_dotenv

load_dotenv()

# Load environment variables from .env file
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CONNECTION_STRING = os.getenv("PG_CONNECTION_STRING")

# Load and clean the document
documents = TextLoader("Market Wizards.txt").load()

text = documents[0].page_content

page_9_marker = "\n9\n"
page_9_index = text.find(page_9_marker)

text = text[page_9_index + len(page_9_marker):]

cleaned_document = Document(page_content=text)

# Split the document into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
splits = text_splitter.split_documents([cleaned_document])

# Add book metadata to each document chunk
book_title = "Market Wizards"
book_author = "Jack D. Schwager"

for doc in splits:
    doc.metadata = {
        "title": book_title,
        "author": book_author
    }

# Initialize the embeddings model and connect to the database
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

conn = psycopg2.connect(CONNECTION_STRING)
cursor = conn.cursor()

# Make sure pgvector extension is installed - this line is necessary!
cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")

# Create a simple books table with vector column
cursor.execute("""
CREATE TABLE IF NOT EXISTS book_vectors (
    id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    author TEXT NOT NULL,
    content TEXT NOT NULL,
    embedding vector(1536)
);
""")

# Generate embeddings and store in book_vectors table
for doc in splits:
    # Generate embedding for this document chunk
    doc_embedding = embeddings.embed_query(doc.page_content)
    
    cursor.execute(
        "INSERT INTO book_vectors (title, author, content, embedding) VALUES (%s, %s, %s, %s)",
        (
            doc.metadata["title"],
            doc.metadata["author"],
            doc.page_content,
            doc_embedding
        )
    )

conn.commit()
print(f"Successfully stored {len(splits)} document chunks in PostgreSQL book_vectors table.")

# Close the connection
cursor.close()
conn.close()