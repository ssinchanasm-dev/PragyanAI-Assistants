import streamlit as st

st.write("Starting app")

import langchain_community

st.write("langchain community loaded")

import os
import pandas as pd
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain_groq import ChatGroq



# ---------------------------------------------------
# ENVIRONMENT
# ---------------------------------------------------


groq_api_key = os.getenv("GROQ_API_KEY")


# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="PragyanAI Assistant",
    page_icon="🤖",
    layout="wide"
)


# ---------------------------------------------------
# CREATE FAQ EXCEL IF NOT AVAILABLE
# ---------------------------------------------------

if not os.path.exists("pragyan_faq_prices.xlsx"):

    faq_data = {

        "Category": [
            "Program Overview",
            "Program Structure",
            "Program Structure",
            "Pricing & Fees",
            "Pricing & Fees",
            "Curriculum & Skills",
            "Curriculum & Skills",
            "Evaluation & Projects",
            "Career & Placement",
            "Leadership & Contact"
        ],


        "Question": [

            "What is the total duration and structure of the PragyanAI program?",
            "What happens in Phase 1?",
            "What happens in Phase 2?",
            "What is the fee structure?",
            "What is the salary potential?",
            "What modules are covered in Months 1-3?",
            "What modules are covered in Months 4-6?",
            "How are students evaluated?",
            "What career tracks are available?",
            "Who leads PragyanAI?"
        ],


        "Answer": [

            "PragyanAI is an 18-month AI GenAI program with 6 months offline training and 12 months internship and placement drive.",

            "Phase 1 consists of offline classroom training, hands-on labs, projects, hackathons and technical seminars.",

            "Phase 2 includes internship, live projects, mock interviews, resume preparation and startup exposure.",

            "Founding Batch fee: ₹50,000 training fee + ₹50,000 success fee after placement.",

            "AI Engineer ₹6-15 LPA, GenAI Engineer ₹8-18 LPA, Agentic AI Engineer ₹10-25 LPA.",

            "Python Full Stack, Analytics, Data Science, Machine Learning, AutoML and Streamlit deployment.",

            "Deep Learning, Computer Vision, NLP, Generative AI, LLMs, RAG, LangChain, Fine tuning, CrewAI and AutoGen.",

            "Technical seminars and 48-hour hackathons with evaluation and prizes.",

            "Data Analyst, Data Scientist, ML Engineer, AI Engineer, GenAI Engineer, Agentic AI Engineer and Product Engineer.",

            "Led by Sateesh Ambesange. Contact: sateesh.ambesange@pragyanai.com"
        ]
    }


    pd.DataFrame(faq_data).to_excel(
        "pragyan_faq_prices.xlsx",
        index=False
    )



# ---------------------------------------------------
# PROMPTS
# ---------------------------------------------------

SALES_PROMPTS = {


"PragyanAI Student Counselor":

"""
You are Aarav, Academic and Career Advisor for PragyanAI.

Guide students about the AI GenAI program.

Use only the provided context.

Context:
{context}
""",



"PragyanAI Institutional / CoE Advisor":

"""
You are Dr. Kavita, Institutional Relations Lead.

Explain how PragyanAI helps colleges create AI-ready students.

Use only provided context.

Context:
{context}
""",



"PragyanAI Enterprise AI & Placement Lead":

"""
You are Rohan, Enterprise Placement Lead.

Explain hiring opportunities and AI talent solutions.

Use only provided context.

Context:
{context}
"""

}


# ---------------------------------------------------
# EMBEDDINGS
# ---------------------------------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)



# ---------------------------------------------------
# VECTOR STORE
# ---------------------------------------------------

@st.cache_resource
def create_vectorstore():

    docs = []


    df = pd.read_excel(
        "pragyan_faq_prices.xlsx"
    )


    for _, row in df.iterrows():

        text = " | ".join(
            f"{col}: {row[col]}"
            for col in df.columns
        )

        docs.append(
            Document(
                page_content=text
            )
        )


    return FAISS.from_documents(
        docs,
        embeddings
    )


vectorstore = create_vectorstore()



# ---------------------------------------------------
# LOAD EXTRA DOCUMENTS
# ---------------------------------------------------

def load_documents_into_vectorstore(files):

    docs = []


    for file in files:

        temp_path = file.name


        with open(
            temp_path,
            "wb"
        ) as f:

            f.write(
                file.getbuffer()
            )


        if temp_path.endswith(".pdf"):

            loader = PyPDFLoader(
                temp_path
            )

            docs.extend(
                loader.load()
            )


        elif temp_path.endswith(".xlsx"):

            df = pd.read_excel(
                temp_path
            )


            for _, row in df.iterrows():

                docs.append(
                    Document(
                        page_content=" | ".join(
                            f"{c}: {row[c]}"
                            for c in df.columns
                        )
                    )
                )


    if docs:

        vectorstore.add_documents(
            docs
        )


    return f"Added {len(docs)} document chunks"



# ---------------------------------------------------
# GROQ MODEL
# ---------------------------------------------------

llm = ChatGroq(

    groq_api_key=groq_api_key,

    model_name="llama-3.3-70b-versatile",

    temperature=0.3
)



# ---------------------------------------------------
# MEMORY
# ---------------------------------------------------

store = {}


def get_session_history(session_id):

    if session_id not in store:

        store[session_id] = ChatMessageHistory()


    return store[session_id]



# ---------------------------------------------------
# RAG CHAIN
# ---------------------------------------------------

def create_rag_chain(
        persona,
        context
):

    prompt = ChatPromptTemplate.from_messages([

        (
            "system",
            SALES_PROMPTS[persona].format(
                context=context
            )
        ),

        MessagesPlaceholder(
            variable_name="history"
        ),

        (
            "human",
            "{input}"
        )

    ])



    chain = prompt | llm | StrOutputParser()



    return RunnableWithMessageHistory(

        chain,

        get_session_history,

        input_messages_key="input",

        history_messages_key="history"
    )



# ---------------------------------------------------
# RESPONSE FUNCTION
# ---------------------------------------------------

def respond(
    message,
    history,
    persona
):

    retriever = vectorstore.as_retriever(
        search_kwargs={
            "k":4
        }
    )


    docs = retriever.invoke(
        message
    )


    context = "\n".join(

        d.page_content
        for d in docs

    )


    chain = create_rag_chain(
        persona,
        context
    )


    return chain.invoke(

        {
            "input":message
        },

        config={

            "configurable":{

                "session_id":persona

            }

        }
    )



def clear_chat_history(persona):

    if persona in store:

        store[persona].clear()

# ---------------------------------------------------
# STREAMLIT UI
# ---------------------------------------------------

st.title(
    "🤖 PragyanAI Conversational Assistant"
)


st.write(
    "AI counselor powered by LangChain + Groq + FAISS"
)



# SIDEBAR

st.sidebar.header(
    "Settings"
)


persona = st.sidebar.selectbox(

    "Select Persona",

    list(SALES_PROMPTS.keys())

)



uploaded_files = st.sidebar.file_uploader(

    "Upload PDF / Excel",

    type=[
        "pdf",
        "xlsx"
    ],

    accept_multiple_files=True

)



if st.sidebar.button(
    "Update Knowledge Base"
):

    if uploaded_files:

        msg = load_documents_into_vectorstore(
            uploaded_files
        )

        st.sidebar.success(
            msg
        )



if st.sidebar.button(
    "Clear Memory"
):

    clear_chat_history(
        persona
    )

    st.session_state.messages=[]

    st.success(
        "Memory cleared"
    )



# CHAT HISTORY

if "messages" not in st.session_state:

    st.session_state.messages=[]



for msg in st.session_state.messages:

    with st.chat_message(
        msg["role"]
    ):

        st.markdown(
            msg["content"]
        )



# INPUT

question = st.chat_input(
    "Ask about PragyanAI..."
)



if question:


    st.session_state.messages.append(

        {
            "role":"user",
            "content":question
        }

    )


    with st.chat_message(
        "user"
    ):

        st.markdown(
            question
        )



    answer = respond(

        question,

        st.session_state.messages,

        persona

    )


    with st.chat_message(
        "assistant"
    ):

        st.markdown(
            answer
        )



    st.session_state.messages.append(

        {
            "role":"assistant",
            "content":answer
        }

    )
