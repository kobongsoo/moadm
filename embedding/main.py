from enum import Enum
from typing import Union
import io
import os
import uuid
import re

from langchain_community.embeddings import (
    HuggingFaceEmbeddings,
    HuggingFaceBgeEmbeddings,
)

##langchain ######################
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.document_loaders.text import TextLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
##################################

# embedding doc
# => langchain을 이용해서 doc을 splitter 하고 embedding 시킴.
def embedding_pdf(huggingfaceembeddings,     # HuggingFaceEmbedding 임베딩 모델 인스턴스     
                  file_path:str,             # 임베딩할 파일 경로
                  chunk_size:int=500,        # chunk 사이즈
                  chunk_overlap:int=0,       # chunk 오버랩
                  upload_file_type:int=0,    # 업로드할 파일 타입(0=pdf 파일, 1=모든파일)
                  no_embedding:int=0):       # 1=임베딩 하지 않고 doc을 split만 해서 리턴.

    assert huggingfaceembeddings, f"[embedding_pdf] huggingfaceembeddings is empty!"
    assert file_path, f"[embedding_pdf] file_path is empty!"
    assert chunk_size > 1, f"[embedding_pdf] chunk_size < 1!"   
    assert chunk_overlap > -1, f"[embedding_pdf] chunk_overlap < 0!"   
    
    if upload_file_type == 0: 
        # => pdf 로딩 후 RecursiveSplitter로 분할함.
        loader = PyPDFLoader(file_path=file_path)
    else:
        loader = TextLoader(file_path=file_path) # text 파일을 로딩함.
      
    #separators = ["\n\n", "\n", " ", ""] # separators(분리기)  지정안할때 기본 설정값
    separators = ["..PAGE:", "\n\n", "\n"] # separators(분리기) 설정
   
    text_splitter = RecursiveCharacterTextSplitter(separators=separators, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    split_docs = loader.load_and_split(text_splitter=text_splitter)

    # split 문서를 docs 리스트에 담음.
    docs:list=[]
    
    # ..PAGE:숫자 패턴과 숫자.숫자 패턴 정의
    # 정규 표현식 패턴 정의
    patterns = [
        r"\.\.PAGE:\d+",           # ..PAGE:숫자 패턴
        r"\d+\.\d+",               # 숫자.숫자 패턴
        r"\d+\.\d+\.\d+",          # 숫자.숫자.숫자 패턴
        r"\d+\.\d+\.\d+\.\d+",     # 숫자.숫자.숫자.숫자 패턴
        r"\d+\.\d+\.\d+\.\d+\.\d+" # 숫자.숫자.숫자.숫자.숫자 패턴
    ]
    
    #print(f'*[embedding_pdf] split_docs: {split_docs}')
    
    for i in range(len(split_docs)):      
        # 모든 패턴을 찾아 제거
        page_content = split_docs[i].page_content
        for pattern in patterns:
            page_content = re.sub(pattern, "", page_content)
        
        if len(page_content) > 1:
            docs.append(page_content)
            #print(f'*[embedding_pdf] page_content: {page_content}')

    #print(f'*[embedding_pdf] docs: {docs}')
            
    # 임베딩하는 경우에만 임베딩하고 벡터 구함        
    docs_vectors:list = []
    if no_embedding == 0:
        # *임베딩 모델 설정 
        # => 여기서 다시 임베딩 모델 호출하면 오래걸림. 따라서 메인에서 1번만 로딩하고, 메인에서 로딩한 모델 인스턴스 재사용함. 
        embedding = huggingfaceembeddings
        #print(f'*[embedding_pdf] 임베딩 모델: {embedding.model_name}')


        # 임베딩 실행
        docs_vectors = embedding.embed_documents(docs)
    return docs_vectors, docs
    