import os
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.document_loaders import WebBaseLoader
from langchain import ConversationChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.vectorstores import FAISS
from langchain.indexes.vectorstore import VectorStoreIndexWrapper


# @param {type:"string"}
OPENAI_API_KEY = 

# @markdown https://huggingface.co/settings/tokens
# @markdown HuggingFace에서 모델 다운로드나 클라우드 모델 사용하기 위해서 필요 (무료)
# @param {type:"string"}
HUGGINGFACEHUB_API_TOKEN = 

# @markdown https://serpapi.com/manage-api-key
# @markdown 구글 검색하기 위해서 필요 (월 100회 무료)
# @param {type:"string"}
SERPAPI_API_KEY = 

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN
os.environ["SERPAPI_API_KEY"] = SERPAPI_API_KEY
# x = input("질문을 물어보세요:")
chat = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.9)
# sys = SystemMessage(content="당신은 경제전문가 ai입니다.")
# msg = HumanMessage(content=x)

# aimsg = chat([sys, msg])
# # print(aimsg.content)

# tools = load_tools(["wikipedia", "llm-math"], llm=chat)

# agent = initialize_agent(tools, llm=chat, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)


# loader = WebBaseLoader(web_path="https://ko.wikipedia.org/wiki/NewJeans")

# documents = loader.load()

# text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
# docs = text_splitter.split_documents(documents)
# # print(len(docs))

# # print(docs[1].page_content)

# chain = load_summarize_chain(chat, chain_type="map_reduce", verbose=True)
# # chain.run(docs[:3])

# embeddings = HuggingFaceEmbeddings()

# index = VectorstoreIndexCreator(
#     vectorstore_cls=FAISS,
#     embedding=embeddings,
#     # text_splitter=text_splitter,
# ).from_loaders([loader])

# # 파일로 저장
# index.vectorstore.save_local("faiss-nj")
# fdb = FAISS.load_local("faiss-nj", embeddings)
# index2 = VectorStoreIndexWrapper(vectorstore=fdb)

# index2.query("뉴진스의 데뷔 멤버는?", llm=chat, verbose=True)

loader = WebBaseLoader(web_path="https://ko.wikipedia.org/wiki/NewJeans")
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

chain = load_summarize_chain(chat, chain_type="map_reduce", verbose=True)
summaries = chain.run(docs[:3])

embeddings = HuggingFaceEmbeddings()

index_creator = VectorstoreIndexCreator(
    vectorstore_cls=FAISS,
    embedding=embeddings,
)

index_creator.from_loaders([loader])
index_creator.vectorstore.save_local("faiss-nj")

fdb_loaded_from_filesystem = FAISS.load_local("faiss-nj", embeddings)
index2 = VectorStoreIndexWrapper(vectorstore=fdb_loaded_from_filesystem)

user_input_question = "뉴진스의 데뷔 멤버는?"
query_result = index2.query(user_input_question, llm=chat, verbose=True)
print(query_result)  # 결과 출력
