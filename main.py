import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv
from htmlTemplates import css, bot_template, user_template
from sec_api import ExtractorApi, QueryApi
from datetime import datetime



def year_to_date_range(year):
    #convert year into integer
    year_integer = int(year) - 1

    # Calculate the start date (June of the previous year)
    start_date = datetime(year_integer - 1, 6, 1)

    # Calculate the end date (June of the input year)
    end_date = datetime(year_integer, 6, 1)

    # Return the date range
    return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")

def get_10k_link(tickername, filedatelower, fileddatehigher,api_key):
    queryapi = QueryApi(api_key)
    query = {
        "query": {"query_string": {
            "query": f"ticker:{tickername} AND filedAt:[{filedatelower} TO {fileddatehigher}] AND formType:\"10-K\"",
            "time_zone": "America/New_York"
        }},
        "from": "0",
        "size": "10",
        "sort": [{"filedAt": {"order": "desc"}}]
    }
    response = queryapi.get_filings(query)
    if response["total"]["value"] == 0:
        return
    else:
        company = response["filings"][0]["companyName"]
        filingUrl = response["filings"][0]["linkToFilingDetails"]

        return company, filingUrl

def get_10k_raw_text(company, filing_10_k_url, api_key):
    extractorApi = ExtractorApi(api_key)

    item_1_text = extractorApi.get_section(filing_10_k_url,"1",'text')  # Risk Factors
    item_1A_text = extractorApi.get_section(filing_10_k_url,"1A",'text')  # Risk Factors
    item_7_text = extractorApi.get_section(filing_10_k_url, "7", 'text')  # Risk Factors
    return (f"The following are the business overview, risk factors and management discussion and analysis within the 10-K "
            f"filing of {company} BUSINESS OVERVIEW:{item_1_text} RISK FACTORS: {item_1A_text} MANAGEMENT "
            f"DISCUSSION AND ANALYSIS: {item_7_text}")



def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    load_dotenv()
    sec_api_key = os.getenv('SEC_API_KEY')
    st.set_page_config(page_title="Chat with multiple PDFs",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        ticker = st.text_input("Insert the ticker symbol")
        filing_year = st.text_input("What is the filing year?")
        if st.button("Find 10K"):
            with st.spinner("Processing"):

                # get year range
                startdate, enddate = year_to_date_range(year=filing_year)
                company_name, filing_link = get_10k_link(ticker,startdate,enddate, sec_api_key)
                raw_text = get_10k_raw_text(company_name, filing_link, sec_api_key)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)


if __name__ == '__main__':
    main()