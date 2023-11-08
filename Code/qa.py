# import streamlit as st
# from langchain.llms import OpenAI
# from langchain.text_splitter import CharacterTextSplitter
# from langchain.embeddings import OpenAIEmbeddings
# from langchain.vectorstores import Chroma
# from langchain.chains import RetrievalQA
# from PyPDF2 import PdfFileReader, PdfFileWriter,PdfReader
# from langchain.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.chains.question_answering import load_qa_chain
# from langchain.llms import Bedrock
# from langchain.chains import RetrievalQA
# from langchain.indexes import VectorstoreIndexCreator
# from langchain.embeddings import BedrockEmbeddings
# from langchain.vectorstores import Chroma
# import boto3
# import os
# from tqdm import tqdm
# import pickle


# # bedrock_runtime = boto3.client(
# #     service_name="bedrock-runtime",
# #     region_name="us-east-1",
# # )
# bedrock_runtime = boto3.client(
#         service_name='bedrock', 
#         region_name="us-east-1"
#     )




# def generate_response(uploaded_file, query_text):
#     # Load document if file is uploaded
#     if uploaded_file is not None:
#         documents = [uploaded_file.read().decode()]
#         # Split documents into chunks
#         text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
#         texts = text_splitter.create_documents(documents)
#         # Select embeddings
#         embeddings =BedrockEmbeddings(
#                                     client=bedrock_runtime,
#                                     model_id="amazon.titan-embed-text-v1",
#                                 )
#         # Create a vectorstore from documents
#         db = Chroma.from_documents(texts, embeddings)
#         # Create retriever interface
#         retriever = db.as_retriever()
#         # Create QA chain
#         llm = Bedrock(
#                     credentials_profile_name="default",
#                     model_id="amazon.titan-text-express-v1"
#                 )
#         qa = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever=retriever)
#         return qa.run(query_text)


# # Page title
# st.set_page_config(page_title='🦜🔗 Ask the Doc App')
# st.title('🦜🔗 Ask the Doc App')

# # File upload
# uploaded_file = st.file_uploader('Upload an article', type='txt')
# # Query text
# query_text = st.text_input('Enter your question:', placeholder = 'Please provide a short summary.', disabled=not uploaded_file)

# # Form input and query
# result = []
# with st.form('myform', clear_on_submit=True):
#     submitted = st.form_submit_button('Submit', disabled=not(uploaded_file and query_text))
  
#     with st.spinner('Calculating...'):
#         response = generate_response(uploaded_file,query_text)
#         result.append(response)
            

# if len(result):
#     st.info(response)