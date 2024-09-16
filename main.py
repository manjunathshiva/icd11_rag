from llama_index.core import SimpleDirectoryReader
docs = SimpleDirectoryReader("Dataset").load_data()

from llama_index.core.node_parser.text import SentenceSplitter
# Initialize the SentenceSplitter with a specific chunk size
text_parser = SentenceSplitter(chunk_size=1024)
text_chunks = [] # This will hold all the chunks of text from all documents
doc_idxs = [] # This will keep track of the document each chunk came from
for doc_idx, doc in enumerate(docs):
    # Split the current document's text into chunks
    cur_text_chunks = text_parser.split_text(doc.text)
 
    # Extend the list of all text chunks with the chunks from the current document
    text_chunks.extend(cur_text_chunks)
 
    # Extend the document index list with the index of the current document, repeated for each chunk
    doc_idxs.extend([doc_idx] * len(cur_text_chunks))

from llama_index.core.schema import TextNode
nodes = [] # This will hold all TextNode objects created from the text chunks
# Iterate over each text chunk and its index
for idx, text_chunk in enumerate(text_chunks):
     # Create a TextNode object with the current text chunk
    node = TextNode(text=text_chunk)
 
     # Retrieve the source document using the current index mapped through doc_idxs
    src_doc = docs[doc_idxs[idx]]
 
     # Assign the source document's metadata to the node's metadata attribute
    node.metadata = src_doc.metadata
 
     # Append the newly created node to the list of nodes
    nodes.append(node)


from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import Settings
from llama_index.core import StorageContext
import qdrant_client

QDRANT_API_KEY = os.environ['QDRANT_API_KEY']
# Create a cloud Qdrant vector store
#client = qdrant_client.QdrantClient(path="financialnews")
client = qdrant_client.QdrantClient(url="https://32ba6540-26df-48e3-a76b-7d831073402f.us-east4-0.gcp.cloud.qdrant.io", api_key=QDRANT_API_KEY)
vector_store = QdrantVectorStore(client=client, collection_name="icd11")

import os
GOOGLE_API_KEY = os.environ['GOOGLE_API_KEY']


from llama_index.embeddings.fastembed import FastEmbedEmbedding

embed_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5")
for node in nodes:
    node_embedding = embed_model.get_text_embedding(
    node.get_content(metadata_mode="all")
 )
    node.embedding = node_embedding

from llama_index.llms.gemini import Gemini
Settings.embed_model = embed_model
Settings.llm = Gemini(model="models/gemini-1.5-flash")
Settings.transformations = [SentenceSplitter(chunk_size=1024)]
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex(
 nodes=nodes,
 storage_context=storage_context,
transformations=Settings.transformations,
)

from llama_index.core import get_response_synthesizer
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
vector_retriever = VectorIndexRetriever(index=index, similarity_top_k=2)
response_synthesizer = get_response_synthesizer()
vector_query_engine = RetrieverQueryEngine(
 retriever=vector_retriever,
 response_synthesizer=response_synthesizer,
)

from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from llama_index.core.query_engine import TransformQueryEngine
hyde = HyDEQueryTransform(include_original=True)
hyde_query_engine = TransformQueryEngine(vector_query_engine, hyde)

def queries(query_str):
    response = hyde_query_engine.query(query_str)
    return str(response)
import gradio as gr
import os
gr.close_all()

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
    """
    # Welcome to Gemini-Powered Stock Predictor RAG Chatbot!
    """)
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.ClearButton([msg, chatbot])
    def respond(message, chat_history):
        bot_message = queries(message)
        chat_history.append((message, bot_message))
        return "", chat_history
    
    msg.submit(respond, [msg, chatbot], [msg, chatbot])
 
demo.launch(share=True)

