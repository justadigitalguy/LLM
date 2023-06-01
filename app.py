import streamlit as st
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os
import openai

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("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")

def create_chat_message(author, message):
    return {"role": author, "content": message}

def send_message(messages, user_message, docs, llm, chain):
    messages.append(create_chat_message("user", user_message))

    with get_openai_callback() as cb:
        response = chain.run(input_documents=docs, question=user_message, temperature=1.0)  # Adjust the temperature here
        print(cb)

    messages.append(create_chat_message("assistant", response))
    return messages

def main():
    st.title('ChatPDF 💬')

    # Sidebar contents
    with st.sidebar:
        st.title('LionLLM')
        st.markdown('''
        ## About
        This app is an LLM-powered chatbot built using:
        - [Streamlit](https://streamlit.io/)
        - [LangChain](https://python.langchain.com/)
        - [OpenAI](https://platform.openai.com/docs/models) LLM model

        ''')

    # Initial chat message
    messages = [create_chat_message("assistant", "Hello, how can I assist you today?")]

    # upload a PDF file
    pdf = st.file_uploader("Upload your PDF", type='pdf')

    # st.write(pdf)
    if pdf is not None:
        pdf_reader = PdfReader(pdf)

        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
            )
        chunks = text_splitter.split_text(text=text)

        # # embeddings
        store_name = pdf.name[:-4]
        st.write(f'{store_name}')
        # st.write(chunks)

        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f)
        else:
            embeddings = OpenAIEmbeddings()
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(VectorStore, f)

        llm = OpenAI()
        chain = load_qa_chain(llm=llm, chain_type="stuff")

        # Accept user questions/query
        user_message = st.text_input("Ask questions about your PDF file:")

        if st.button("Send"):
            docs = VectorStore.similarity_search(query=user_message, k=3)
            messages = send_message(messages, user_message, docs, llm, chain)

        # Display the chat history
        for msg in messages:
            if msg["role"] == "user":
                st.write(f"You: {msg['content']}")
            else:
                st.write(f"Assistant: {msg['content']}")

        # Display the chat history in the sidebar
        with st.sidebar:
            st.header("Chat History")
            for msg in messages:
                st.write(f"{msg['role'].capitalize()}: {msg['content']}")

if __name__ == '__main__':
    main()









