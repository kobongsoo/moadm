import os
import time
import numpy as np
import sys
import json
from tqdm import tqdm 

sys.path.append('..')
from embedding import embedding_pdf
from utils import weighted_reciprocal_rank_fusion, clustering_embedding  
    
#------------------------------------------------------------------
# file 1개에 대한 벡터를 구하는 함수
#------------------------------------------------------------------
def get_doc_vector(instance:dict, file_path:str):
    
    assert file_path, f"file_path is empty!!"
    esdocs:list=[]
    
    myutils = instance['myutils']
    myes = instance['myes']
    embedding = instance['embedding']
    
    settings = myutils.get_options()
    chunk_size = settings['CHUNK_SIZE']
    chunk_overlap = settings['CHUNK_OVERLAP']
    es_batch_size = settings['ES_BATCH_SIZE']
    upload_file_type = 1
    FLOAT_TYPE = settings['FLOAT_TYPE']    # 클러스터링 할때 벡터 타입(float32, float16)
    
    # ==== 임베딩 ==================
    try:
        # => langchain 이용. splitter 후 임베딩 함
        docs_vectors, docs = embedding_pdf(huggingfaceembeddings=embedding,
                              file_path=file_path, 
                              chunk_size=chunk_size, 
                              chunk_overlap=chunk_overlap,
                              upload_file_type=upload_file_type)
                
        docs_vectors_array = np.array(docs_vectors) # docks_vectors는 list이므로 array로 변경해 줌
                
    except Exception as e:
        msg = f'*embedding_pdf is Fail!!..(error: {e})'
        myutils.log_message(f'[get_doc_vector] {msg}')
        return 1001, msg
    # ==============================
    
    # ==== ES 인덱싱 데이터 변환 =================
    try:
        esdoc = {}
                
        file_name = os.path.basename(file_path) # 파일명만 뽑아냄
        esdoc['rfile_name'] = file_name
                
        with open(file_path, 'r', encoding='utf-8') as f: # 파일내용추가
            data = f.read()
        esdoc['rfile_text'] = data
                
        # vector0에는 평균 임베딩 값을 담음.
        avg_emb = docs_vectors_array.mean(axis=0).reshape(1,-1) #(128,) 배열을 (1,128) 형태로 만들기 위해 reshape 해줌
        esdoc["vector0"] = avg_emb[0]
                
        return 0, esdoc 
        
    except Exception as e:
        msg = f'*create docs vector is Fail!!..(error: {e})'
        myutils.log_message(f'[get_doc_vector] {msg}')
        return 1003, msg
     
#------------------------------------------------------------------------------------
# 파일들에 대해 임베딩하는 함수
# => file_path_list를 입력받고 해당 list에 파일들에 대해 임베딩 함.
#------------------------------------------------------------------------------------
def embedding_file_list_doc(instance:dict, 
                            index_name:str,
                            file_path_list:list, 
                            indexing:bool=True,         # True=elasticsearch로 인덱싱. False=vector값만 리턴
                            del_index:bool=False,       # True=기존 elasticsearch에 같은 index명이 있으면 제거. False=제거하지 않고 추가.
                            ):
           
    assert len(file_path_list) > 0, f"file_path_list is empty"
    start_time = time.time()
    
    myutils = instance['myutils']
    myes = instance['myes']
        
    settings = myutils.get_options()
    es_batch_size = settings['ES_BATCH_SIZE']
  
    # elasticsearch 8.x 버전인 경우 index_doc 값 설정
    # 예)  {"index": {"_index": "my_index"}},
    index_doc = {}
    index_doc['index'] = {"_index": index_name}
            
    # 인덱스 생성(*del_index==True이면 기존 인덱스 삭제하고 생성)
    if indexing==True:
        myes.create_index(index_name=index_name, delete=del_index)
        
    num:int = 0
    esdocs:list=[]
    count:int = 0
    noindexing_docs:list=[]
    
    for idx, file_path in enumerate(tqdm(file_path_list)):
        # ./files/out 폴더에 추출한 text 파일들을 불러와서 임베딩 벡터 생성함.
        # 파일별 임베딩 함
        try:
            error, doc = get_doc_vector(instance, file_path)
            if error == 0:
                count += 1
                if len(doc) > 0:
                    if indexing==True:
                        # elasticsearch 8.x 버전인 경우 index_doc 값 추가
                        esdocs.append(index_doc)
                        esdocs.append(doc)
                    else:
                        doctmp:dict={"rfile_name": doc['rfile_name'], "vector0": list(doc['vector0'])}
                        noindexing_docs.append(doctmp)
                 
            # batch_size 만큼씩 한꺼번에 es로 insert 시킴.
            if indexing==True:
                if count % es_batch_size == 0:
                    num += 1
                    myes.RAG_bulk(docs=esdocs)
                    myutils.log_message(f'[embedding_file_list_doc] *bulk_{num}:{len(esdocs)}')
                    esdocs = []
                
        except Exception as e:
            msg = f'*RAG_bulk is Fail!!..(error: {e})'
            myutils.log_message(f'[embedding_file_list_doc] {msg}')
            response = {"error":1003, "response": f"{msg}", "time": "0"}
            return response
            
    # = 마지막에 한번더 남아있는거 인덱싱 ==
    if indexing==True and esdocs:
        num += 1
        myes.RAG_bulk(docs=esdocs)
        myutils.log_message(f'[embedding_file_list_doc01] *bulk_{num}(last):{len(esdocs)}')
    # ====================================         
                        
    elapsed_time = "{:.2f}".format(time.time() - start_time)
    
    # 인덱싱이면 인덱싱 성공한 count 리턴
    if indexing==True:
        response = {"error":0, "response": f"{count}", "time": f"{elapsed_time}"}
    else: # 인덱싱이 아니면 vector값 esdocs 리스트 리턴
        response = {"error":0, "response": noindexing_docs, "time": f"{elapsed_time}"}
        
    return response
#------------------------------------------------------------------------------------
# 파일 1개에 대해 임베딩 하는 함수
# => 파일 1개를 chunk로 나누고 -> 나눈 chunk에 대해 벡터를 구하고->이를 ES에 documents로 인덱싱함. 
#------------------------------------------------------------------------------------
def embedding_file_doc(instance:dict, 
                       index_name:str,
                       file_path:str, 
                       del_index:bool=False,       # True=기존 elasticsearch에 같은 index명이 있으면 제거. False=제거하지 않고 추가.
                       ):    
    
    assert file_path, f"file_path is empty!!"
    assert index_name, f"index_name is empty!!"
    
    start_time = time.time()
    
    response:dict={}
    myutils = instance['myutils']
    myes = instance['myes']
    embedding = instance['embedding']
    
    settings = myutils.get_options()
    chunk_size = settings['CHUNK_SIZE']
    chunk_overlap = settings['CHUNK_OVERLAP']
    es_batch_size = settings['ES_BATCH_SIZE']
    upload_file_type = 1
    FLOAT_TYPE = settings['FLOAT_TYPE']    # 클러스터링 할때 벡터 타입(float32, float16)
      
    # ==== 벡터값 구함 ===============================
    docs_vectors:list = []
    docs:list=[]
    try:
        # => langchain 이용. splitter 후 임베딩 함
        docs_vectors, docs = embedding_pdf(huggingfaceembeddings=embedding,
                              file_path=file_path, 
                              chunk_size=chunk_size, 
                              chunk_overlap=chunk_overlap,
                              upload_file_type=upload_file_type)
                
        docs_vectors_array = np.array(docs_vectors) # docks_vectors는 list이므로 array로 변경해 줌
                
    except Exception as e:
        msg = f'*embedding_pdf is Fail!!..(error: {e})'
        myutils.log_message(f'[embedding_file_doc] {msg}')
        response = {"error":1001, "response": f"{msg}", "time": 0}
        return response
    # ====================================================
    
    # === ES에 임베딩 =============================
    
    # 인덱스 생성
    myes.create_index(index_name=index_name, delete=del_index)
    
    # elasticsearch 8.x 버전인 경우 index_doc 값 설정
    # 예)  {"index": {"_index": "my_index"}},
    index_doc = {}
    index_doc['index'] = {"_index": index_name}
    
    file_name = os.path.basename(file_path) # 파일명만 뽑아냄
    esdocs:list=[]    
    count:int = 0
    num:int = 0
    
    for doc, vector in zip(docs, docs_vectors):
        esdoc = {}
        esdoc['rfile_name'] = f'{file_name}_{count}'
        esdoc['rfile_text'] = doc
        esdoc["vector0"] = np.array(vector) # * vector 필드는 array로 담아야 함. 리스트를 array로 변환

        esdocs.append(index_doc)
        esdocs.append(esdoc)
        
        count += 1
        if count % es_batch_size == 0:
            num += 1
            myes.RAG_bulk(docs=esdocs)
            myutils.log_message(f'[embedding_file_doc] *bulk_{num}:{len(esdocs)}')
            esdocs = []
            
    # = 마지막에 한번더 남아있는거 인덱싱 ==
    if esdocs:
        num += 1
        myes.RAG_bulk(docs=esdocs)
        myutils.log_message(f'[embedding_file_doc] *bulk_{num}(last):{len(esdocs)}')
    
    elapsed_time = "{:.2f}".format(time.time() - start_time)
    
    response = {"error":0, "response": f"{count}", "time": f"{elapsed_time}"}
    return response
        
    