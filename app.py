import os
import pandas as pd
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain_groq import ChatGroq


# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="PragyanAI Assistant",
    page_icon="🤖",
    layout="wide"
)


# =====================================================
# GROQ KEY
# =====================================================

if "GROQ_API_KEY" not in st.secrets:

    st.error(
        "GROQ_API_KEY missing. Add it in Streamlit Secrets."
    )

    st.stop()


GROQ_API_KEY = st.secrets["GROQ_API_KEY"]



# =====================================================
# PERSONAS
# =====================================================

PERSONAS = {

"Student Counselor": """

You are Aarav, PragyanAI Student Counselor.

Help students understand:
- Program duration
- Fees
- Curriculum
- Career opportunities

Answer only from the provided context.

Context:

{context}

""",


"Institution Advisor": """

You are Dr Kavita, PragyanAI Institutional Advisor.

Explain:
- College partnerships
- AI Center of Excellence
- Student transformation

Use only provided context.

Context:

{context}

""",


"Enterprise Placement Lead": """

You are Rohan, PragyanAI Enterprise Placement Lead.

Explain:
- Hiring opportunities
- AI projects
- GenAI skills
- Agentic AI

Use only provided context.

Context:

{context}

"""

}



# =====================================================
# EMBEDDINGS
# =====================================================

@st.cache_resource
def get_embeddings():

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


embeddings = get_embeddings()



# =====================================================
# LOAD KNOWLEDGE BASE
# =====================================================

def load_documents(files=None):

    docs=[]


    # Load Excel FAQ

    if os.path.exists(
        "pragyan_faq_prices.xlsx"
    ):

        df=pd.read_excel(
            "pragyan_faq_prices.xlsx"
        )


        for _,row in df.iterrows():

            text=" | ".join(

                [
                    f"{col}: {row[col]}"
                    for col in df.columns
                ]

            )


            docs.append(

                Document(
                    page_content=text
                )

            )



    # Load uploaded files

    if files:

        for file in files:


            path=file.name


            with open(path,"wb") as f:

                f.write(
                    file.getbuffer()
                )


            if path.endswith(".pdf"):

                loader=PyPDFLoader(path)

                docs.extend(
                    loader.load()
                )



            elif path.endswith(".xlsx"):


                df=pd.read_excel(path)


                for _,row in df.iterrows():

                    text=" | ".join(

                        [
                            f"{col}: {row[col]}"
                            for col in df.columns
                        ]

                    )


                    docs.append(

                        Document(
                            page_content=text
                        )

                    )



    return docs



@st.cache_resource
def create_vector_database():

    documents=load_documents()

    return FAISS.from_documents(
        documents,
        embeddings
    )



if "vectorstore" not in st.session_state:

    st.session_state.vectorstore = create_vector_database()



# =====================================================
# LLM
# =====================================================

llm = ChatGroq(

    groq_api_key=GROQ_API_KEY,

    model_name="llama-3.3-70b-versatile",

    temperature=0.3

)



# =====================================================
# MEMORY
# =====================================================

if "chat_memory" not in st.session_state:

    st.session_state.chat_memory={}



def get_history(session_id):

    if session_id not in st.session_state.chat_memory:

        st.session_state.chat_memory[session_id] = ChatMessageHistory()


    return st.session_state.chat_memory[session_id]



# =====================================================
# RAG FUNCTION
# =====================================================

def ask_ai(question, persona):


    retriever = (

        st.session_state.vectorstore

        .as_retriever(
            search_kwargs={
                "k":4
            }
        )

    )


    docs=retriever.invoke(
        question
    )


    context="\n\n".join(

        [
            doc.page_content
            for doc in docs
        ]

    )



    prompt=ChatPromptTemplate.from_messages(

        [

            (
                "system",
                PERSONAS[persona].format(
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

        ]

    )


    chain=(

        prompt

        |

        llm

        |

        StrOutputParser()

    )



    conversation = RunnableWithMessageHistory(

        chain,

        get_history,

        input_messages_key="input",

        history_messages_key="history"

    )



    response=conversation.invoke(

        {
            "input":question
        },

        config={

            "configurable":
            {
                "session_id":persona
            }

        }

    )


    return response



# =====================================================
# STREAMLIT UI
# =====================================================


st.title(
    "🤖 PragyanAI Intelligent Assistant"
)


st.caption(
    "LangChain + FAISS + Groq RAG Application"
)



with st.sidebar:


    st.header(
        "Settings"
    )


    persona=st.selectbox(

        "Choose Assistant",

        list(PERSONAS.keys())

    )


    uploaded=st.file_uploader(

        "Upload PDF / Excel",

        type=[
            "pdf",
            "xlsx"
        ],

        accept_multiple_files=True

    )



    if st.button(
        "Reload Documents"
    ):

        docs=load_documents(uploaded)

        st.session_state.vectorstore = FAISS.from_documents(
            docs,
            embeddings
        )

        st.success(
            "Knowledge updated"
        )



    if st.button(
        "Clear Chat"
    ):

        st.session_state.chat_memory={}

        st.session_state.messages=[]



if "messages" not in st.session_state:

    st.session_state.messages=[]



for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.write(
            message["content"]
        )



question=st.chat_input(
    "Ask about PragyanAI..."
)



if question:


    st.session_state.messages.append(

        {
            "role":"user",
            "content":question
        }

    )


    with st.chat_message("user"):

        st.write(question)



    answer=ask_ai(
        question,
        persona
    )


    st.session_state.messages.append(

        {
            "role":"assistant",
            "content":answer
        }

    )


    with st.chat_message("assistant"):

        st.write(answer)
