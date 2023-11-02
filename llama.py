from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from huggingface_hub import hf_hub_download
from langchain.chains.question_answering import load_qa_chain
import os
import pinecone
from langchain.vectorstores import Pinecone

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY', '1b84a4d5-1fff-4c1a-a7bf-0e5a03df2797')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV', 'gcp-starter')
pinecone.init(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_API_ENV
)
print(pinecone.list_indexes())
index = pinecone.Index('test')

embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')

text_field = "text"
vectorstore = Pinecone(
    index, embeddings.embed_query, text_field
)

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

model_name_or_path = "TheBloke/Llama-2-13B-chat-GGML"
model_basename = "llama-2-13b-chat.ggmlv3.q5_1.bin"
# model_path = hf_hub_download(repo_id=model_name_or_path, filename=model_basename)
model_path = r"D:/Study/DL/rag/models/llama-2-13b-chat.ggmlv3.q5_1.bin"

llm = LlamaCpp(
    model_path="D:/Study/DL/rag/models/llama-2-7b-chat.Q2_K.gguf",
    verbose=True,
    n_ctx=2048
)
chain = load_qa_chain(llm, chain_type='stuff')

query = 'When did former Chinese Premier Li Keqiang die'
docs = vectorstore.similarity_search(query)
print(docs)
output = chain.run(input_documents=docs, question=query)
print(output)







