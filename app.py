import streamlit as st
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts.prompt import PromptTemplate
import json
import os
from streamlit_lottie import st_lottie


st.set_page_config(layout="wide",page_title="Chat PDF App", page_icon="page_icon.jpg", initial_sidebar_state="expanded")

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

css_style = {
    "icon": {"color": "white"},
    "nav-link": {"--hover-color": "grey"},
    "nav-link-selected": {"background-color": "#FF4C1B"},
}

if "messages" not in st.session_state:
       st.session_state.messages = []
for message in st.session_state["messages"]:
       if message["role"] == "user":
              with st.chat_message("user"):
                     st.markdown(message["content"])
       elif message["role"] == "assistant":
              with st.chat_message("assistant"):
                     st.markdown(message["content"])                    

with st.sidebar:
       st.title('Interact with PDF🦜')
       st.markdown('''
       ## About⚙️
       This App helps you to interact with your PDFs.
                   ''')
       #add_vertical_space(1)
       st.write('Made with 💚 by Bibhuti Baibhav Borah')

def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

def main():

       lottie_ani = load_lottiefile("animation.json")
       st_lottie(
        lottie_ani,
        speed =1,
        reverse=False,
        quality= "high",
        width=340)

       st.header("Chat With PDF🔎")

       load_dotenv()



       pdf = st.file_uploader("#### Please Upload the PDF👇🏻", type='pdf')
       if pdf is not None:
              pdf_reader1 = PdfReader(pdf)
             
              text = ""
              for page in pdf_reader1.pages:
                     text += page.extract_text()
              
              text_splitter = RecursiveCharacterTextSplitter(
                     chunk_size = 1000,
                     chunk_overlap = 200,
                     length_function=len
              )
              chunks = text_splitter.split_text(text=text)
              #st.write(chunks)

              store_name = pdf.name[:-4]

              if os.path.exists(f"{store_name}.pkl"):
                      with open(f"{store_name}.pkl","rb") as f:
                             VectorStore = pickle.load(f)

              else:
                     embeddings = OpenAIEmbeddings()
                     VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
                     with open(f"{store_name}.pkl","wb") as f:
                            pickle.dump(VectorStore,f)


              #st.write(text) 
              query = st.text_input("Ask questions about your PDF file")  
              memory = ConversationBufferMemory(memory_key= "chat_history", return_messages=True)

              if query:
                     chat_history = []
                     with st.chat_message("user"):
                            st.markdown(query)
                     st.session_state.messages.append({"role":"user", "content": query})       

                     custom_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question. At the end of standalone question add this 'Answer the question in English language.' If you do not know the answer reply with 'I am sorry'.
                     Chat History:
                     {chat_history}
                     Follow Up Input: {question}
                     Standalone question:
                     Remember to greet the user with hi welcome to pdf chatbot how can i help you? if user asks hi or hello """
 
                     custom_ques_prompt = PromptTemplate.from_template(custom_template)
                     llm = ChatOpenAI(model="gpt-3.5-turbo")

                     conversation_chain =ConversationalRetrievalChain.from_llm(
                            llm,
                            VectorStore.as_retriever(),
                            condense_question_prompt=custom_ques_prompt,
                            memory=memory
                     )
                     response = conversation_chain({"question": query, "chat_history": chat_history})

                     with st.chat_message("assistant"):
                            st.markdown(response["answer"])
                     st.session_state.messages.append({"role": "assistant", "content": response["answer"]})
                     chat_history.append((query, response))




if __name__ == '__main__':
       main()       

