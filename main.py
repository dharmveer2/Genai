import os
from os import environ
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.document_loaders import UnstructuredFileLoader
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.llms import OpenAI
from langchain import OpenAI, PromptTemplate, LLMChain
from langchain.chains.mapreduce import MapReduceChain
from langchain.chains.summarize import load_summarize_chain
from dotenv import load_dotenv , find_dotenv
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import (ConversationBufferMemory, 
                                                  ConversationSummaryMemory, 
                                                  ConversationBufferWindowMemory,
                                                  ConversationKGMemory)
from langchain.callbacks import get_openai_callback

load_dotenv(find_dotenv())
OPEN_API_KEY=os.getenv("seceret_key")
environ["OPENAI_API_KEY"]=OPEN_API_KEY

def search_google(query):
    memory = ConversationBufferMemory()
    qa = ConversationChain(llm=OpenAI(temperature=0,model_name='gpt-3.5-turbo'), memory=memory)
    count_tokens(qa,query)

def count_tokens(chain, query):
    with get_openai_callback() as cb:
        result = chain.run(query)
        print(result)
        print(f'Spent a total of {cb.total_tokens} tokens')

def queryDoc(chunks):
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002") 
    vector_store = FAISS.from_documents(
    chunks,
    embedding=embeddings)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    qa = ConversationalRetrievalChain.from_llm(
    OpenAI(temperature=0),
    vector_store.as_retriever(),
    memory=memory)
    query = " "
    while query != "stop":
        query = input("How can I help you ? \n Are you tired  ? Type stop to exit()   \n")
        if query == "stop":
            break
        result = qa({"question": query})
        print("\n\n")
        print(result["answer"])
        res = str(result["answer"])
        if res.lower() == (" I don't know.").lower():
            flag = input("Sorry this doc has no information regarding this query.. Do you want to search it? Type 1 for yes, or 0 for no.")
            if flag == '1':
                search_google(query)
            else:
                print("okay")
        print("\n\n\n")


def summerize(chunks):
    llm = OpenAI(temperature=0)
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    print("Printing your summery...")
    print(chain.run(chunks))

if __name__ == "__main__":
    while True:
        goal = input("1. for summery  \n 2. for query a \n press 0 to exit..")
        
        if goal=='0':
            break
        type = input("1. For docs file\n 2. for Pdf file\n 3. for txt file \n  press 0 to exit..")
        if input=='0':
            break
        doc=[]
        filePath=""
        if type=='1':
            filePath = input("enter your docx file path. \n")
            if filePath.endswith('.docx'):
                doc_loader=UnstructuredFileLoader(filePath)
                doc=doc_loader.load()
            else:
                print("Sorry you have not entered docx file :) ")
                pass
        elif type=='2':
            filePath = input("enter your pdf file path..  \n")
            if filePath.endswith('.pdf'):
                loader = UnstructuredPDFLoader(filePath, mode="elements")
                doc = loader.load()
            else:
                print("Sorry you have not entered pdf file :) ")
                pass
        elif type=='3':
            filePath = input("enter your .txt file path. \n")
            if filePath.endswith('.txt'):
                loader = UnstructuredFileLoader(filePath)
                doc = loader.load()
            else:
                print("Sorry you have not entered txt file :) ")
                pass
        else:
            print("pls enter valid choice")

        text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=10)
        chunks=text_splitter.split_documents(doc)

        if goal=='1':
            summerize(chunks)

        elif goal=='2':
            
            queryDoc(chunks)
        
        elif goal=='0':
            exit()

        else:
            print("enter valid choice...\n")
