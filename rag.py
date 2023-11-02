from langchain.document_loaders import PyPDFLoader, OnlinePDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone
from sentence_transformers import SentenceTransformer
from langchain.chains.question_answering import load_qa_chain

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from huggingface_hub import hf_hub_download
from langchain.chains.question_answering import load_qa_chain

import pinecone
import os

loader = PyPDFLoader('D:\ChromeDownload\Li_Keqiang.pdf')
data = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
docs = text_splitter.split_documents(data)
# print(docs)
len(docs)
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY', '1b84a4d5-1fff-4c1a-a7bf-0e5a03df2797')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV', 'gcp-starter')
pinecone.init(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_API_ENV
)
index_name = 'test'

embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
# embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
# embeddings = SentenceTransformer('all-MiniLM-L6-v2')
docsearch = Pinecone.from_texts([t.page_content for t in docs], embeddings, index_name=index_name)
query = 'When did Li Keqiang die'
docs_sim = docsearch.similarity_search(query)
print(docs_sim)

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

model_name_or_path = "TheBloke/Llama-2-13B-chat-GGML"
model_basename = "llama-2-13b-chat.ggmlv3.q5_1.bin"
# model_path = hf_hub_download(repo_id=model_name_or_path, filename=model_basename)
model_path = r"D:/Study/DL/rag/models/llama-2-13b-chat.ggmlv3.q5_1.bin"

llm = LlamaCpp(
    model_path="D:/Study/DL/rag/models/llama-2-7b-chat.Q2_K.gguf"
)
chain = load_qa_chain(llm, chain_type='stuff')

query = 'When did former Chinese Premier Li Keqiang die'
# docs = vectorstore.similarity_search(query)
# print(docs)
output = chain.run(input_documents=docs_sim, question=query)
print(output)




