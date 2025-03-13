# Import langchain
from langchain_core.tools import Tool
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_community.utilities import SerpAPIWrapper

# Import prompts
from prompts import rag_query_prompt

# Import db connection
import psycopg2 

# Import other
import os
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()
OPENAI_API_KEY = None

def set_openai_api_key(api_key: str) -> None:
    """Set the OpenAI API key globally for all tools"""

    global OPENAI_API_KEY
    OPENAI_API_KEY = api_key

def generate_rag_queries(question: str) -> str:
    """Generate RAG queries for a given question"""

    global OPENAI_API_KEY
        
    connection_string = os.getenv("PG_CONNECTION_STRING")
    conn = psycopg2.connect(connection_string)
    cursor = conn.cursor()
    
    # Step 1: Generate multiple query variations using user-provided API key
    prompt_template = PromptTemplate(
        input_variables=["question"],
        template=rag_query_prompt
    )
    generate_queries = (
        prompt_template 
        | ChatOpenAI(model_name="gpt-4o", temperature=0.6, openai_api_key=OPENAI_API_KEY) 
        | StrOutputParser() 
        | (lambda x: x.split("\n"))
    )
    queries = generate_queries.invoke({"question": question})
    
    # Use user-provided API key for embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY)
    
    all_results = []
    
    # Step 2: Search vector database with each query
    for i, query in enumerate(queries):
        if not query.strip():  
            continue
            
        query_embedding = embeddings.embed_query(query)
        
        # Convert the embedding to the format PostgreSQL expects
        embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'
        
        cursor.execute("""
            SELECT content, embedding <-> %s::vector AS distance
            FROM book_vectors
            ORDER BY distance
            LIMIT %s
        """, (embedding_str, 2))
        
        results_count = 0
        for content, distance in cursor.fetchall():
            results_count += 1
            all_results.append({
                "content": content,
                "distance": distance
            })

    cursor.close()
    conn.close()
    
    all_results.sort(key=lambda x: x["distance"])
   
    
    # Deduplicate results
    seen_content = set()
    unique_results = []
    
    for result in all_results:
        content = result["content"]
        if content not in seen_content:
            seen_content.add(content)
            unique_results.append(result)
    
    formatted_results = "\n\n".join([r["content"] for r in unique_results[:3]])
    return formatted_results

def get_stock_analysis(query: str) -> str:
    """Get stock analysis for a given query"""

    serp_api_key = os.getenv("SERP_API_KEY")
    search = SerpAPIWrapper(serpapi_api_key=serp_api_key)
    
    results = search.run(query)
     
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
     
    formatted_results = f"Market Research Results (as of {timestamp}):\n\n{results}"
    return formatted_results