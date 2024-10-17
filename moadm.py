import torch
import time
import os
import numpy as np
import random
import asyncio
import threading
import httpx
import uvicorn
import io
import openai

from enum import Enum
from typing import Union
from pydantic import BaseModel
from fastapi import FastAPI, File, UploadFile, Query, Cookie, Form, Request, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from utils import MyUtils, delete_local_file, generate_random_string, generate_text_GPT2, weighted_reciprocal_rank_fusion
from langchain_community.embeddings import HuggingFaceEmbeddings, HuggingFaceBgeEmbeddings
from docs_func import mpower_save_docs, embedding_file_list_doc, search_docs, embedding_file_doc
from es_8 import My_ElasticSearch

from docs_func import check_mime_type, getfilePath_doc01, extract_save_doc01

from vision import MY_Vision

# settings.yaml 설정값 불러옴.
myutils = MyUtils(yam_file_path='./data/docs_settings.yaml')
settings = myutils.get_options()

#---------------------------------------------------------------
app=FastAPI() # app 인스턴스 생성
templates = Jinja2Templates(directory="template_files") # html 파일이 있는 경로를 지정.
#---------------------------------------------------------------

# 임베딩 모델 로딩
# ./cache/ 경로에 다운로드 받도록 설정
os.environ["HF_HOME"] = "./cache/"
print(f'*임베딩 모델 {settings["EMBEDDING_MODEL"]} 로딩 시작==>')

# model_name = 없으면: 기본 'sentence-transformers/all-mpnet-base-v2' 모델임.
#embeddings = HuggingFaceEmbeddings()  # HuggingFace 임베딩을 생성합니다.
embedding = HuggingFaceEmbeddings(
    model_name=settings["EMBEDDING_MODEL"], 
    show_progress=True
)
print(f'*임베딩 모델: {embedding.model_name}')
#---------------------------------------------------------------
# elastic search 설정
es_url=settings['ES_URL']
es_index_file_path=settings['ES_INDEX_FILE_PATH']
es_api_key=settings['ES_API_KEY']  # *elasticsearch 8.x 버전에서는 ES_API_KEY를 발급받아서 연결해야 함.

# es 8.x 버전이면 es_8 패키지 로딩(*단 로딩을 위해서는 # pip로 elasticsearch 8.14.3으로 업데이트 =>!pip install --upgrade elasticsearch)
myes = My_ElasticSearch(es_url=es_url, index_file_path=es_index_file_path, api_key=es_api_key)
print(f'*엘라스틱서치: {myes}')   
print(f'*ES정보: {myes}')

#---------------------------------------------------------------
# Mpower Synap 추가
from os import sys
sys.path.append('../../MpowerAI')
from pympower.classes.mshaai import MShaAI

shaai = MShaAI()
#---------------------------------------------------------------

# vision 모델 로딩
myvision = MY_Vision(model_folder_path=settings['VISION_MODEL'], device=settings['VISION_DEVICE'])
print(f'*Vision모델: {settings["VISION_MODEL"]}')   
print(f'*myvision: {myvision}')
#-----------------------------------------------

# GPT 모델 - GPT 3.5 Turbo 지정 : => 모델 목록은 : https://platform.openai.com/docs/models/gpt-4 참조                                                
openai.api_key = settings['GPT_TOKEN']# **GPT  key 지정
gpt_model = settings['GPT_MODEL']  #"gpt-4"#"gpt-3.5-turbo" #gpt-4-0314
#---------------------------------------------------------------

# global 인스턴스 dict로 정의
global_instance:dict = {'myutils': myutils, 'settings': settings, 'myvision': myvision, "embedding": embedding, "myes": myes, "shaai": shaai}
#---------------------------------------------------------------

@app.get("/")  # 경로 동작 데코레이터 작성
async def root(): # 경로 동작 함수 작성
	return {"mpower-dm(documet manamger) api server"}
#---------------------------------------------------------------

#---------------------------------------------------------------
# 1.입베딩
# text 추출된 문서내용을 입력 받아서 vector를 구하고 elasticsearch 로 인덱싱 함.
# => post: /es/embedding
# -in: data : rfile_name_list = 문서 rfile_name들, rfile_text_list = 문서 rfile_text(문서내용) 들
# -in: save_folder_path : rfile_text 내용을 파일로 저장할 root 폴더 경로.(*해당 폴더 하위로 rfile_name 으로 파일이 생성됨)  
# -in: indexing : True=elasticsearch로 인덱싱. False=vector값만 리턴
# -in: del_index : True=기존 인덱스 있으면 제거, (*기본=False)   
#---------------------------------------------------------------
class DocsEmbedIn(BaseModel):
    rfile_name_list: list
    rfile_text_list: list
    
@app.post("/es/{esindex}/embedding")
def dm_embedding(Data:DocsEmbedIn,
                 esindex:str,
                 save_folder_path:str="./mpower-dm", 
                 indexing:bool=True,
                 del_index:bool=False):
    
    status:int = 0
    error:str = ""
    response:dict = {}
    
    rfile_name_list = Data.rfile_name_list
    rfile_text_list = Data.rfile_text_list
    
    # 인자 검사
    if len(esindex) < 1:
        status=1
        error="esindex is empty!"
    elif len(rfile_name_list) < 1:
        status=2
        error="rfile_name_list is empty!"
    elif len(rfile_text_list) < 1:
        status=3
        error="rfile_text_list is empty"
    elif len(rfile_name_list) != len(rfile_text_list):
        status=4
        error="the rfile_name listing count and rfile_text listing count do not match."
        
    if status != 0:
        response = {"error": status, "response": error, "time": 0}
        return JSONResponse(content=response) 
        #raise HTTPException(status_code=404, detail=error, headers={"X-Error": error},)
        
    # rfile_text_list 내용들을 임시 파일로 저장.
    response=mpower_save_docs(folder_path=save_folder_path, 
                         rfile_name_list=rfile_name_list, 
                         rfile_text_list=rfile_text_list)
    
    myutils.log_message(f'[/es/{esindex}/embedding] response:{response}')   
    
     # 저장된 파일을 가지고 임베딩 진행.
    file_path_list = response['file_path_list']
    if len(file_path_list) > 0:
        response = embedding_file_list_doc(instance=global_instance, 
                                      index_name=esindex,   # 인덱스명
                                      file_path_list=file_path_list, # 임덱싱할 파일들 경로.
                                      indexing=indexing,    # True=elasticsearch로 인덱싱. False=vector값만 리턴
                                      del_index=del_index)  # True=기존 elasticsearch에 같은 index명이 있으면 제거. False=제거하지 않고 추가.      
        # 파일 삭제 
        for file_path in file_path_list:
            delete_local_file(filepath=file_path)
    else:
        status=10
        error = "embedding_file_list is response error!!"
        response = {"error": status, "response": error, "time": 0}
        return JSONResponse(content=response) 
        #raise HTTPException(status_code=404, detail=error, headers={"X-Error": error},)
        
    return JSONResponse(content=response) 


#---------------------------------------------------------------
# 2. 검색
# elasticsearch에 인덱싱된 문서들을 vector 혹은 bm25 방식으로 검색한다. 
# => post: /es/{인덱스명}/search
# -in: esindex = 임베딩할 인덱스 명칭
# -in: data - rfile_name_list = 검색할 문서 rfile_name들(*없으면 모든 rfile_name에 대해 검색)
# -in: data - file_type = 0,1만 입력=>0=신규문서(query_rfile=문서내용 입력됨) 1=기존문서(query_rfile=rfile_name 명 입력됨)
# -in: data - query_rfile = 신규 문서면 문서내용 혹은 기존 문서면 rfile_name
# -in: search_k : 검색수(몇개까지 검색할지)
# -in: search_method : 검색방식(0=vector 검색, 1=vector+bm25 검색, 2=bm25검색)
# -in: save_folder_path : rfile_text 내용을 파일로 저장할 root 폴더 경로.(*해당 폴더 하위로 rfile_name 으로 파일이 생성됨)  
#--------------------------------------------------------------- 
class DocsSearchIn(BaseModel):
    rfile_name_list:list
    file_type:int  # 0,1만 입력=>0=신규문서(query_rfile=문서내용 입력됨) 1=기존문서(query_rfile=rfile_name 명 입력됨)
    query_rfile:str  # 신규 문서 혹은 기존 문서
    
@app.post("/es/{esindex}/search")
async def dm_search(Data:DocsSearchIn,
                    esindex:str,
                    search_k:int=Query(..., gt=0), # ... 는 필수 입력 이고, gt=0은 0보다 커야 한다. 작으면 422 Unprocessable Entity 응답반환됨
                    search_method:int=0,
                    save_folder_path:str="./mpower-dm" # 임시파일이 저장될 root 폴더 경로.
                   ):
    
    status:int = 0
    error:str = ""
    response:dict = {}
    file_path:str = ""
    
    search_rfile_name_list = Data.rfile_name_list
    rfile_type = Data.file_type # 0=신규문서(query_rfile_text=문서내용 입력됨) 1=기존문서(query_rfile_text=rfile_name 명 입력됨)
    query_rfile = Data.query_rfile
    
    # ==인자 검사 =============================
    if len(esindex) < 1:
        status=1
        error="esindex is empty!"
    elif len(query_rfile) < 1:
        status=2
        error="query_rfite is empty!"
    elif rfile_type not in [0, 1]:
        status=3
        error=f"rfile_type({rfile_type}) is wrong!. rfile_type is 0,1"
    elif search_method not in [0, 1, 2]:
        status=4
        error=f"search_method({search_method}) is wrong!. search_method is 0,1,2"
        
    # index가 존재하는지 체크    
    if myes.check_index_exist(index_name=esindex)==False:
        status=10
        error=f"index({esindex}) is not exsit!"
    
    if status != 0:
        response = {"error": status, "response": error, "time": 0}
        return JSONResponse(content=response) 
        #raise HTTPException(status_code=404, detail=error, headers={"X-Error": error},)
    
    # == 신규문서 or 기존문서 처리==============
    # rfile_type==0, 즉 신규문서내용이 입력된 경우에만 파일명으로 저장.
    if rfile_type == 0:
        # query_rfile_text(문서쿼리내용) 랜덤한 파일명으로 저장
        random_file_name = generate_random_string(10) # 10자리 랜덤한 파일명 만듬.
        res=mpower_save_docs(folder_path=save_folder_path, 
                             rfile_name_list=[random_file_name], 
                             rfile_text_list=[query_rfile])
    
        myutils.log_message(f'[/es/{esindex}/search] res:{res}')   
        file_path = res['file_path_list'][0]
        
    # rfile_type==1, 즉 기존문서 rfile_name이 입력된 경우에는 file_path를 rfile_name으로 지정
    elif rfile_type == 1:
        file_path = query_rfile
        
    # == 임베딩 & 검색==============
    
    response = search_docs(instance=global_instance, 
                           index_name=esindex, 
                           file_type=rfile_type,
                           file_path=file_path, 
                           uids=search_rfile_name_list, 
                           search_k=search_k, 
                           search_method=search_method)
    
    myutils.log_message(f'[/es/{esindex}/search] response:{response}')   
    return JSONResponse(content=response) 
 
#---------------------------------------------------------------
# [옵션] 파일업로드 검색
# :파일업로드->text추출->평균vector생성->검색->검색결과 html로 리턴
# 파일을 선택해서 업로드 하면 file_foloder 경로 + "/org" 폴더파일을 저장하고,
# 이후 저장된 파일 text 추출후 file_foloder 경로 + "/extra" 폴더에 저장후, 임베딩 후 검색함.
#
# => post: /upload/es/{인덱스명}/search
# -in: esindex = 임베딩할 인덱스 명칭
# -in: file: UploadFile = File(...) : 업로드되는 파일
# -in: file_type = 0,1만 입력=>0=신규문서(query_rfile=문서내용 입력됨) 1=기존문서(query_rfile=rfile_name 명 입력됨)
# -in: search_k : 검색수(몇개까지 검색할지)
# -in: search_method : 검색방식(0=vector 검색, 1=vector+bm25 검색, 2=bm25검색)
# -in: local_folder_path : 로컬파일이 있는 경로..
# -in: save_folder_path : rfile_text 내용을 파일로 저장할 root 폴더 경로.(*해당 폴더 하위로 rfile_name 으로 파일이 생성됨)
#---------------------------------------------------------------
@app.post("/upload/es/{esindex}/search")
async def dm_upload_search(request: Request,
                           file: UploadFile = File(...), 
                           esindex:str="mpower10u_vector",
                           file_type:int=0,               # 0,1만 입력=>0=신규문서(query_rfile=문서내용 입력됨) 1=기존문서(query_rfile=rfile_name 명 입력됨)
                           search_k:int=Query(..., gt=0), # ... 는 필수 입력 이고, gt=0은 0보다 커야 한다. 작으면 422 Unprocessable Entity 응답반환됨
                           search_method:int=0,           # 검색방식(0=vector 검색, 1=vector+bm25 검색, 2=bm25검색)
                           save_folder_path:str="./dm_tmp/upload", # 임시파일이 저장될 root 폴더 경로.
                           local_folder_path:str="D:", # 로컬 파일들이 있는 폴더 경로..
                           out_web:int=1,                      # 0=검색결과만 json으로 리턴, 1=웹으로 요약, 검색결과등을 html로 리턴
                          ):
    status:int = 0
    error:str = ""
    response:dict = {}
    rfile_name_list:list=[]
    file_name:str = ""
    
    start_time = time.time()
    
    settings = myutils.get_options()
        
    form = await request.form()
    #user_id = form.get("user_id")
    
    # local_folder_path에 한개 \ 를 역슬레쉬 한개 /로 치환=>이유는 html로 출력할때 '/'로 해야 함.
    local_folder_path = local_folder_path.replace("\\", "/")
    
    myutils.log_message(f'\n[info][/upload/es/{esindex}/search] esindex:{esindex}, file_type:{file_type}, search_k:{search_k}, search_method:{search_method}, local_folder_path:{local_folder_path}')   
    
    #==인자 검사 =============================
    if len(esindex) < 1:
        status=1
        error="esindex is empty!"
    elif file_type not in [0, 1]:
        status=3
        error=f"rfile_type({rfile_type}) is wrong!. rfile_type is 0,1"
    elif search_method not in [0, 1, 2]:
        status=4
        error=f"search_method({search_method}) is wrong!. search_method is 0,1,2"
         
    # index가 존재하는지 체크    
    if myes.check_index_exist(index_name=esindex)==False:
        status=10
        error=f"index({esindex}) is not exsit!"
    
    if status != 0:
        return templates.TemplateResponse("error.html", {"request": request, "error": status, "response":error, "time": 0})
    
    # ==이미지 타입인지 확인================
    mime_type = check_mime_type(file.content_type)    
    
    # ==입력받은 문서내용을 원본파일경로에 파일로 저장================
    # getfilePath_doc01() 함수에서 file_folder + /org 폴더에 입력된 파일 원본이 저장되고,
    # text 추출된 파일은 file_folder + /extra 폴더에 저장된다.
    file_name = file.filename
    #myutils.log_message(f'[/upload/es/{esindex}/search] search_docs\r\nfile_name:{file_name}')  
    
    srcPath, tgtPath = getfilePath_doc01(file_name, save_folder_path, mime_type)
    myutils.log_message(f'\n[info][/upload/es/{esindex}/search] srcPath:{srcPath}, tgtPath:{tgtPath}, mime_type:{mime_type}')   
    with open(srcPath, "wb") as f:
        content = await file.read()
        f.write(content)
    
    
    # ==원본파일에서 text 추출 후 파일로 저장============
    status, response = extract_save_doc01(global_instance, srcPath, tgtPath, mime_type)
    if mime_type == "img":
        ImageToText = response
        
    if status != 0:
        delete_local_file(srcPath) # 검색 후 추출한 파일 삭제
        return templates.TemplateResponse("error.html", {"request": request, "error": status, "response":response, "time": 0})  
    
    # ==임베딩 후 인덱싱 저장.==============
    start_time2 = time.time()
        
    doc_index_name:str = settings['ES_RAG_DOC_INDEX_NAME']  # doc 인덱스명 지정.
    response=embedding_file_doc(instance=global_instance, 
                                index_name=doc_index_name,
                                file_path=tgtPath,
                                del_index=True)
    
    myutils.log_message(f'[/upload/es/{esindex}/search] embedding_file_doc\r\nresponse:{response}')  
    
    # == 임베딩 & 검색==============
    
    response = search_docs(instance=global_instance, 
                           index_name=esindex, 
                           file_type=file_type,
                           file_path=tgtPath, 
                           uids=rfile_name_list, 
                           search_k=search_k, 
                           search_method=search_method)
    
    myutils.log_message(f'[/upload/es/{esindex}/search] search_docs\r\nresponse:{response}')  
    
    # out_web 아니면(0), json으로 검색결과만 리턴
    if out_web == 0:
        response_new = {"num":f"{len(response['response'])}", "response": f"{response['response']}"}
        return JSONResponse(content=response_new)  
    
    # == 내용 요약 =================
    if mime_type != "img":
        prompt_context = settings['PROMPT_CONTEXT']
        query:str = 'Context 내용을 요약해 주세요.'
        # 파일 읽기
        with open(tgtPath, 'r', encoding='utf-8') as file:
            file_content = file.read()
        if len(file_content) > 1024:
            file_content = file_content[:1024]

        prompt = prompt_context.format(query=query, context=file_content) # query에는 요약해달는 prompt, context에는 파일내용.
        #myutils.log_message(f'[/upload/es/{esindex}/search] search_docs\r\nprompt:{prompt}')  

        answer, error = generate_text_GPT2(gpt_model=gpt_model,
                                           prompt=prompt,
                                           stream=settings['GPT_STREAM'],
                                           max_tokens=settings['GPT_MAX_TOKENS'],
                                           temperature=settings['GPT_TEMPERATURE'],
                                           top_p=settings['GPT_TOP_P']
                                          )
        myutils.log_message(f'[/upload/es/{esindex}/search] generate_text_GPT2\r\nanswer:{answer}, error:{error}')  
    else: # 이미지가 입력된 경우에는 ImageToText를 담음.
        answer = ImageToText
        
    total_elapsed_time = "{:.2f}".format(time.time() - start_time)
    embedding_search_time = "{:.2f}".format(time.time() - start_time2)
    
    elapsed_time = f"{total_elapsed_time}({embedding_search_time})" # 총시간(임베딩.검색 시간) 출력
    
    return templates.TemplateResponse("success.html", {'request': request, 'filename':file_name, 'error':response['error'], 'response':response['response'], 'local_folder_path':local_folder_path, 'answer':answer, 'time':elapsed_time})  

#---------------------------------------------------------------    
# RAG Q&A 창 
# => RAG Q&A search.html 페이지를 띄운다.
#---------------------------------------------------------------    
@app.get("/rag")
async def rag(request:Request, filename:str):
    myutils.log_message(f'[/rag] file_name:{filename}')  
    return templates.TemplateResponse("search.html", {"request": request, "user_id": filename})
#---------------------------------------------------------------
# RAG Q&A 창 Q&A 처리
# => 질문에 대한 벡터 생성후 'ES_RAG_DOC_INDEX_NAME' 인덱스에서 유사벡터검색후 질문에 대해 GPT가 답변함
#---------------------------------------------------------------
@app.get("/search/query")
async def search01(request:Request, user_id:str, query:str):
    assert user_id, f'user_id is empty'
    assert query, f'query is empty'
    print(f'*[search] user_id: {user_id}, query: {query}\n')
    
    settings = myutils.get_options()
    rag_prompt_context = settings['RAG_PROMPT_CONTEXT']
    qa_prmpt_context = settings['QA_PROMPT_CONTEXT']
    doc_index_name = settings['ES_RAG_DOC_INDEX_NAME']  # doc 인덱스명 지정.
    
    k = settings['SEARCH_K']
    uid_embed_weigth = settings['RRF_BM25_WEIGTH']
    uid_bm25_weigth = settings['RRF_EMBED_WEIGTH']
    bm25_search_min_score = settings['BM25_SEARCH_MIN_SCORE']
    embedding_search_min_score = settings['EMBEDDING_SEARCH_MIN_SCORE']
        
    gpt_model:str = settings['GPT_MODEL']    
    system_prompt = settings['SYSTEM_PROMPT']
    max_tokens = settings.get('GPT_MAX_TOKENS', 1024)
    temperature = settings.get('GPT_TEMPERATURE', 1.0)
    top_p = settings.get('GPT_TOP_P', 0.1)
    stream = settings.get('GTP_STREAM', False)

    # user_id가 '*.*'이면 모든 검색
    user_id = ""
        
    bm25_docs:list = []
    embed_docs:list = []
    
    # 1.ES로 BM25 검색
    bm25_docs = myes.BM25_search(index_name=doc_index_name, 
                                 query=query, 
                                 k=k, 
                                 min_score=bm25_search_min_score)

    # 2.ES로 임베딩 검색
    embed_docs = myes.Embedding_search(index_name=doc_index_name,
                                       huggingfaceembeddings=embedding, 
                                       query=query, 
                                       k=k, 
                                       min_score=embedding_search_min_score)

    # 3. BM25 + 임베딩검색 RRF 시킴
    RRF_docs:list = []
    if len(embed_docs) > 0 and len(bm25_docs) > 0:
        embed_docs_name = [doc['rfile_name'] for doc in embed_docs]
        bm25_docs_name = [doc['rfile_name'] for doc in bm25_docs]
        
        RRF_scores=weighted_reciprocal_rank_fusion(lists=[embed_docs_name, bm25_docs_name], weights=[uid_embed_weigth, uid_bm25_weigth])

        # bm25_docs 와 embed_docs 두 리스트를 합쳐서 하나의 딕셔너리로 만듬.
        combined_docs = {doc['rfile_name']: doc for doc in embed_docs + bm25_docs}

        # RRF_scores에 있는 name과 일치하는 rfile_text 값을 combined_docs 리스트에서 찾아서, RRF_docs 리스트에 추가함.
        for name, RRF_score in RRF_scores:
            if name in combined_docs:
                RRF_doc = {
                    'rfile_name': combined_docs[name]['rfile_name'],  # combined_docs name
                    'rfile_text': combined_docs[name]['rfile_text'],  # combined_docs rfile_text
                    'score': RRF_score
                }
                RRF_docs.append(RRF_doc)
        
        
    # 4.프롬프릍 생성.
    context:str = ""
    doc_names:str = ""
    context_score:str = ""
    
    # RRF_docs 있으면 context를 문서제목+문서text 식으로 만듬
    count:int = 0
    idx:int = 0
    if len(RRF_docs) > 0:
        for idx, RRF_doc in enumerate(RRF_docs):
            count = idx+1
            doc_names += RRF_doc['rfile_name'] + '<br>'
            context += RRF_doc['rfile_text'] + '\n\n'
            context_score += f"{count}. (score:{RRF_doc['score']:.3f}){RRF_doc['rfile_name']}<br><br>{RRF_doc['rfile_text']}<br><br>"
    else: # RRF_docs 없으면 embed_doc를 context를 문서제목+문서text 식으로 만듬
        for embed_doc in embed_docs:
            count = idx+1
            doc_names += embed_doc['rfile_name'] + '<br>'
            context += embed_doc['rfile_text'] + '\n\n'
            context_score += f"{count}. (sore:{embed_doc['score']:.3f}){embed_doc['rfile_name']}<br><br>{embed_doc['rfile_text']}<br><br>"
            idx = idx+1
            
    myutils.log_message(f'========================================')
    myutils.log_message(f'\n[info][/search] *RRF_docs Len:{len(RRF_docs)}\n*BM24_doc Len:{len(bm25_docs)}\n*embed_docs Len:{len(embed_docs)}\n*doc_name:\n{doc_names}\n*context_score:\n{context_score}\n')
    myutils.log_message(f'========================================')
                        
    prompt:str = ""
    if context: # 검색된 docs가 있는 경우 rag 프롬프트 생성
        prompt = rag_prompt_context.format(query=query, context=context)
    else:       # 검색이 없는 경우 Q&A 프롬프트 생성
        prompt = qa_prmpt_context.format(query=query)
    
    myutils.log_message(f'\n[info][/search] *promptp:{prompt}\n')
        
    # 4.GPT로 쿼리
    response, status = generate_text_GPT2(gpt_model=gpt_model, prompt=prompt, system_prompt=system_prompt, 
                                          assistants=[], stream=stream, timeout=20,
                                          max_tokens=max_tokens, temperature=temperature, top_p=top_p) 
    
    myutils.log_message(f'\n[info][/search] *generate_text_GPT2=>status:{status}\nresponse:\n{response}\n')
    myutils.log_message(f'========================================')
    
    # 5. 결과 출력 (응답결과, context+score, status 에러값, 문서명들)
    return response, context_score, status, doc_names   
#---------------------------------------------------------------    
  