import os
import time
import numpy as np
import sys

sys.path.append('..')
from embedding import embedding_pdf
from utils import weighted_reciprocal_rank_fusion
from es_8 import  make_embedding_query_script

#---------------------------------------------------------------
# == 검색 처리 함수 ==
# -in: instance : 인스턴스
# -in : index_name: 검색할 elasticsearch 인덱스명
# -in : file_type: 0=신규문서(file_path=문서경로가 입력됨) 1=기존문서(file_path에는 rfile_name 명칭이 입력됨)
# -in : file_path : 검색할 신규 문서경로 혹은 기존 rfile_name(*file_type에 따라 입력값 다름)
# -in : uids : 검색대상 rfile_name 목록들. (없으면 모든 rfile_name 검색함)
# -in : search_k : 검색 계수
# -in : search_method : # 검색방식(0=vector 검색, 1=vector+bm25 검색, 2=bm25검색) 
#---------------------------------------------------------------
def search_docs(instance:dict, 
                index_name:str, 
                file_type:int=0, 
                file_path:str="", 
                uids:list=None, 
                search_k:int=5, 
                search_method:int=0):
    
    assert file_path, f'file_path is empty'
    assert file_type < 2, f'file_type is wrong(file_type={file_type})'
    
    myutils = instance['myutils']
    myes = instance['myes']
    embedding = instance['embedding']
        
    start_time = time.time()
    upload_file_type = 1
    
    #myutils.log_message(f'[search_docs] file_path:{file_path}')

    settings = myutils.get_options()
    chunk_size = settings['CHUNK_SIZE']
    chunk_overlap = settings['CHUNK_OVERLAP']
    bm25_search_min_score = settings['BM25_SEARCH_MIN_SCORE']
    embedding_search_min_score = settings['EMBEDDING_SEARCH_MIN_SCORE']
    uid_embed_weigth = settings['RRF_BM25_WEIGTH']
    uid_bm25_weigth = settings['RRF_EMBED_WEIGTH']
    
    ##################################################################################
    # == 기존검색문서 => ES로 vector  얻어옴.==
    ##################################################################################
    if file_type == 1:
        # ==== rfle_name으로 검색===============       
        response = myes.search_rfile_name_docs(index_name=index_name, rfile_name=file_path)  
        #myutils.log_message(f'[info][search_docs] *response: {response}/{type(response)}')
        avg_emb = response[0]['vector0']
        #myutils.log_message(f'[info][search_docs] *file_type==0 => avg_emb: {avg_emb}/{type(avg_emb)}')
    ##################################################################################
    # == 신규검색문서 => 문서내용을 split 및 vector생성 ==
    ##################################################################################
    elif file_type == 0:
        if search_method == 0 or search_method == 1: # 검색방식(0=vector 검색, 1=vector+bm25 검색, 2=bm25검색) 0,1인경우에는 vector 검색함.
            no_embedding = 0
        else: #BM25 검색(2)인 경우에는 임베딩 안함.
            no_embedding = 1
        
        docs_vectors:list = []
        docs:list = []
    
        try:

            # => langchain 이용. splitter 후 임베딩 함
            docs_vectors, docs = embedding_pdf(huggingfaceembeddings=embedding,
                                      file_path=file_path, 
                                      chunk_size=chunk_size, 
                                      chunk_overlap=chunk_overlap,
                                      upload_file_type=upload_file_type,
                                      no_embedding = no_embedding  # 임베딩할지 안할지
                                      )

            docs_vectors_array = np.array(docs_vectors) # docks_vectors는 list이므로 array로 변경해 줌
            avg_emb = docs_vectors_array.mean(axis=0).reshape(1,-1)[0] # 평균을 구함 : (128,) 배열을 (1,128) 형태로 만들기 위해 reshape 해줌          
            #myutils.log_message(f'[info][search_docs] *file_type==1 => avg_emb: {avg_emb}/{type(avg_emb)}')

            # docs_vectors_array 가 없으면 response에는 빈 리스트로 리턴.
            if len(docs_vectors_array) < 1 and no_embedding == 0:
                response = {"error":2000, "response": f"[]", "time": "0"}
                myutils.log_message(f'[info][search_docs][embedding_pdf] {response}')
                return response

        except Exception as e:
            msg = f'*embedding_pdf is Fail!!..(error: {e})'
            myutils.log_message(f'[info][search_docs][embedding_pdf] {msg}')
            response = {"error":1001, "response": f"{msg}", "time": "0"}
            return response
    ##################################################################################
    
    ##################################################################################
    # =벡터검색=
    # =>벡터검색(0) 혹은 벡터검색+BM25(1)인 경우에만 벡터검색 함.
    ##################################################################################
    embed_docs:list=[]
    if search_method == 0 or search_method == 1: 
        # ==ES 쿼리스크립트 만듬 : 평균스크립트 ========
        script_query = make_embedding_query_script(qr_vector=avg_emb, uid_list=uids) # 쿼리를 만듬.   
        #myutils.log_message(f'[info][search_docs][make_embedding_query_script]\nscript_query: {script_query}\n')
            
        # ==== ES로 쿼리 ===============
        # 임베딩 search
        embed_docs = myes.search_docs(index_name=index_name,
                                      script_query=script_query, 
                                      k=search_k, 
                                      min_score=embedding_search_min_score) 
    
    # 벡터 SEARCH 만하는 경우에는 벡터 DOCS 만 리턴 함.
    if search_method == 0: 
        end_time = time.time()
        elapsed_time = "{:.2f}".format(end_time - start_time)
        response = {"error": 0, "response": embed_docs, "time": f"{elapsed_time}"}
        return response 
    ##################################################################################
    
    ##################################################################################
    # ==BM25 검색==
    # =>RRF SEARCH(1) 혹은 BM25 검색(2)인 경우에만 BM25검색 함.
    bm25_docs:list=[]
    if search_method == 1 or search_method == 2: 
        
        # BM25 search 검색 
        query:str = ''
        for doc in docs: # 내용을 query로 전송( query 최대 길이는 1024 넘으면 안됨)
            query += doc
            if len(query) > 1024:
                break

        if len(query) > 1024:
            query = query[:1024]
            #myutils.log_message(f'\n[search_docs] BM25_searchdoc=>query:\n{query}\n')
         
        #myutils.log_message(f'\n[search_docs] BM25_searchdoc=>query:\n{query}\n')
            
        if len(query) > 0:
            bm25_docs = myes.BM25_search_docs(query=query, 
                                              index_name=index_name, 
                                              k=search_k, 
                                              min_score=bm25_search_min_score)
    
    # ==BM25 검색(2)인 경우 bm25결과만 리턴.==
    if search_method == 2:
        end_time = time.time()
        elapsed_time = "{:.2f}".format(end_time - start_time)
        response:dict = {"error": 0, "response": bm25_docs, "time": f"{elapsed_time}"}
        return response
    ##################################################################################
    
    ##################################################################################
    # =RRF=
    # =>BM25 +임베딩검색 RRF 스코어 구함
    ##################################################################################
    RRF_docs:list = []
    if len(embed_docs) > 0 and len(bm25_docs) > 0:
        embed_docs_name = [doc['rfile_name'] for doc in embed_docs]
        bm25_docs_name = [doc['rfile_name'] for doc in bm25_docs]
        
        RRF_scores=weighted_reciprocal_rank_fusion(lists=[embed_docs_name, bm25_docs_name], weights=[uid_embed_weigth, uid_bm25_weigth])

        # bm25_docs 와 embed_docs 두 리스트를 합쳐서 하나의 딕셔너리로 만듬.
        combined_docs = {doc['rfile_name']: doc for doc in embed_docs + bm25_docs}

        # RRF_scores에 있는 name과 일치하는 rfile_text 값을 combined_docs 리스트에서 찾아서, RRF_docs 리스트에 추가함.
        count:int = 0
        for name, RRF_score in RRF_scores:
            if name in combined_docs:
                #RRF_score_2f = {:.2f}.format(RRF_score)
                rounded_RRF_score = round(RRF_score, 3)
                
                RRF_doc = {
                    'rfile_name': combined_docs[name]['rfile_name'],  # combined_docs name
                    #'rfile_text': combined_docs[name]['rfile_text'],  # combined_docs rfile_text
                    'score': rounded_RRF_score
                }
                RRF_docs.append(RRF_doc)
                
                # search_k 계수보다 같거나 크면 stop
                count += 1
                if count >= search_k:
                    break
     
    ##################################################################################
    
    end_time = time.time()
    elapsed_time = "{:.2f}".format(end_time - start_time)
    response:dict = {"error": 0, "response": RRF_docs, "time": f"{elapsed_time}"}
    return response