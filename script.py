
from langchain.llms import OpenAI

from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from typing import List
from langchain.chains import RetrievalQA
import os
import sys
import xml.etree.ElementTree as ET
import panel as pn



openai_api_key = "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"
os.environ["OPENAI_API_KEY"] = openai_api_key

embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

# llm = OpenAI(model= "text-davinci-003", temperature=0.75)
llm = OpenAI(temperature=0.75)

chat = ChatOpenAI(temperature=0.75)


def get_questions(prompt):
    response = chat([HumanMessage(content=prompt)])
    questions = [response.content.strip()]
    return questions

def get_answers(question, abstract):
    messages = [HumanMessage(content=question), HumanMessage(content=f"abstract: {abstract}")]
    response = chat(messages)
    answer = response.content.strip()
    return answer

file_input = pn.widgets.FileInput(width=300)


def qa(file, query, chain_type, k):
    # Load document
    loader = PyPDFLoader(file)
    documents = loader.load()
    
    # Split the documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    
    # Select which embeddings we want to use
    embeddings = OpenAIEmbeddings()
    
    # Create the vector store to use as the index
    db = Chroma.from_documents(texts, embeddings)
    
    # Expose this index in a retriever interface
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": k})
    
    # Create a chain to answer questions 
    qa_chain = RetrievalQA.from_chain_type(
        llm=OpenAI(), chain_type=chain_type, retriever=retriever, return_source_documents=True)
    
    conversation = []  # List to store the conversation history
    
    while True:
        user_question = input("Enter a question or type 'exit': ")
        if user_question == 'exit':
            break
        
        # Add the user's question to the conversation history
        conversation.append(f"Inquiry: {user_question}")
        
        # Perform the question-answering
        result = qa_chain({"query": user_question})
        
        # Get the chat bot's answer
        chat_bot_answer = result['result']
        
        # Add the chat bot's answer to the conversation history
        conversation.append(f"Response: {chat_bot_answer}")
        
        # Print and return the result
        print(f"Inquiry: {user_question}")
        print(f"Response: {chat_bot_answer}")
        
    return conversation


def main():

    choice = input("Enter your choice: 1. Upload your own pdf, 2. Analyze ICM-42688P datasheet, 3. Exit: ")

    if choice == '1':
        pdf_file = input("Enter the path to your pdf:")

        while True:
            query = input("Enter a question or type 'exit': ")
            if query == 'exit':
                break
            result = qa(file=pdf_file, query=query, chain_type="map_rerank", k=2)

    elif choice == '2':
        pdf_path = "/Users/corydewitt/Desktop/Research/REU/playgrnd/PX4_devices/pdf_chatbot/ICM.pdf"  # NEEDS TO CHANGE PER USER
        while True:
            query = input("Enter a question or type 'exit': ")
            if query == 'exit':
                break
            result = qa(file=pdf_path, query=query, chain_type="map_rerank", k=2)
    
    elif choice == '3':
        sys.exit(1)



if __name__ == "__main__":
    main()
