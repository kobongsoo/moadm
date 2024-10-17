import torch
import time
import os
import numpy as np

############################################################
## rfile_name으로 해당 인덱스 정보 얻는 쿼리 스크립트 
############################################################
def make_rfile_name_query_script(query:str)->str:
    
    # rfile_name으로 검색
    script_query={
        "term": {
            "rfile_name": query
        }
    }

    return script_query

############################################################
## bm25 쿼리 스크립트 만들기
############################################################
def make_bm25_query_script(query:str, user_id:str=None)->str:
    # user_id와 rfile_text 비교를 위해 must 이용함.
    if user_id:
        script_query={
            "bool": {
                "must": [
                    {"match": {"user_id": user_id}},
                    {"match": {"rfile_text": query}}
                ]
            }
        }
    else: # user_id가 없으면 rfile_text만 교교
        script_query={
            "match": {
                "rfile_text": query
            }
        }

    return script_query
 
############################################################
## 임베딩 쿼리 스크립트 만들기
############################################################
def make_embedding_query_script(qr_vector, user_id:str=None, uid_list:list=None)->str:
    
    # 문단별 10개의 벡터와 쿼리벡터를 서로 비교하여 최대값 갖는 문단들중 가장 유사한  문단 출력
    # => script """ 안에 코드는 java 임.
    # => "queryVectorMag": 0.1905 일때 100% 일치하는 값은 9.98임(즉 10점 만점임)

    # uid_list가 있는 경우에는 해당하는 목록만 검색함
    if uid_list:
        query = { "bool" :{ "must": [ { "terms": { "rfile_name": uid_list } } ] } }
    elif user_id:
        query = { "match": {"user_id": user_id } }
    else: # uid_list가 있는 경우에는 해당하는 목록만 검색함
        query = { "match_all" : {} }

    script_query = {
        "script_score":{
            "query":query,
            "script":{
                "source": "cosineSimilarity(params.qrVector, 'vector0') + 1.0",  # 뒤에 1.0 은 코사인유사도 측정된 값 + 1.0을 더해준 출력이 나옴
                #"source": "dotProduct(params.qrVector, 'vector0')",
                "params": {"qrVector": qr_vector}
            }
        }
    }

    return script_query
   
############################################################
## 필드 목록 계수 얻기(*중복 제거)
## => 검색할 필드는 반드시 keyword 타입이어야 함.(*text 타입이면 에러 발생)
############################################################
def make_list_query_script(field_name:str, user_id:str=None)->str:
    assert field_name, f'field_name is empty!!'

    data = {}

    if user_id:
        data = {"term": { "user_id": user_id }}
    else:
        data = {"match_all": {}}

    body = {
            "size": 0,
            "query": data,
                "aggs": {
                    "unique_field": {
                        "terms": {
                        "field": field_name,
                        "size": 1000  
                        }
                    }
                }
            }

    return body

############################################################
## 삭제 스크립트
## => user_id와 rfile_name 리스트 항복에 있는 데이터 삭제
############################################################
def make_del_query_script(fields:list, user_id:str=None)->str:
    assert len(fields) > 0, f'fields is empty!!'

    if user_id:
        body = {
          "query": {
            "bool": {
              "must": [
                {
                  "term": {
                    "user_id": user_id
                  }
                },
                {
                  "terms": {
                    "rfile_name": fields
                  }
                }
              ]
            }
          }
        }
    else:
        body = {
          "query": {
            "terms": {
              "rfile_name": fields
            }
          }
        }

    return body

#---------------------------------------------------------------------------
# ES MAX 쿼리 스크립트 구성
# => 1문서당 10개의 벡터 중에서 가장 유사도가 큰 1개의 벡터 유사도 측정하는 쿼리
# -in: query_vector = 1차원 임베딩 벡터 (예: [10,10,1,1, ....]
# -in: vectornum : ES 인덱스 벡터 수
#---------------------------------------------------------------------------
def make_max_query_script(query_vector, vectornum:int=10, uid_list:list=None)->str:
    # 문단별 10개의 벡터와 쿼리벡터를 서로 비교하여 최대값 갖는 문단들중 가장 유사한  문단 출력
    # => script """ 안에 코드는 java 임.
                        
    # uid_list가 있는 경우에는 해당하는 목록만 검색함
    if uid_list:
        query = { "bool" :{ "must": [ { "terms": { "rfile_name": uid_list } } ] } }
    else: # uid_list가 있는 경우에는 해당하는 목록만 검색함
        query = { "match_all" : {} }
    
    script_query = {
        "script_score":{
             "query":query,
                "script":{
                    "source": """
                    
                      float sqrt(float number) {
                        float guess = number / 2.0f;
                        float epsilon = 0.00001f;  // 원하는 정확도
                        while (Math.abs(guess * guess - number) >= epsilon) {
                            guess = (guess + number / guess) / 2.0f;
                        }
                      return guess;
                      }
                 
                      float queryVector_sum_size = 0.0f;
                      // 쿼리벡터의 크기를 구함.->벡터제곱근 구하고 루트(sqrt) 처리                      
                      for (int j = 0; j < params.queryVector.length; j++) {
                          queryVector_sum_size += params.queryVector[j] * params.queryVector[j];
                      } 
                      
                      float queryVector_size = sqrt(queryVector_sum_size);
                      float max_score = 0.0f; 
                      for(int i = 1; i <= params.VectorNum; i++) 
                      {
                          float[] v = doc['vector'+i].vectorValue; 
                          float vm = doc['vector'+i].magnitude;  
                          if (v[0] != 0)
                          {
                            // dot(내적) 2벡터간 내적을 구함.
                            float dotProduct = 0.0f;
                            
                            for(int j = 0; j < v.length; j++) 
                            {
                                dotProduct += v[j] * params.queryVector[j];
                            }
                            
                            // 스코어를 구함 = dot/(벡터1크기*쿼리벡터크기)
                            float score = dotProduct / (vm * queryVector_size);
                              
                            if(score > max_score) 
                            {
                                max_score = score;
                            }
                          }
                      }
                      return max_score
                    """,
                "params": 
                {
                  "queryVector": query_vector,  # 벡터임베딩값 설정
                  "VectorNum": vectornum        # 벡터 수 설정
                }
            }
        }
    }
    
    return script_query


############################################################
## 쿼리벡터가 여러개이고 es에도 여러개인 경우
# => 문단별 10개의 벡터와 10개의 쿼리 벡터를 서로 비교하여 최대값 갖는 문단들 중 가장 유사한 문단 출력
############################################################
def make_max_query_vectors_script(query_vectors, vectornum:int=10, uid_list:list=None)->str:
    # 문단별 10개의 벡터와 10개의 쿼리 벡터를 서로 비교하여 최대값 갖는 문단들 중 가장 유사한 문단 출력
    # => script """ 안에 코드는 java 임.

    # uid_list가 있는 경우에는 해당하는 목록만 검색함
    if uid_list:
        query = { "bool" :{ "must": [ { "terms": { "rfile_name": uid_list } } ] } }
    else:
        query = { "match_all" : {} }
    
    script_query = {
        "script_score":{
             "query":query,
                "script":{
                    "source": """
                    
                      // 제곱근 구하는 함수
                      float sqrt(float number) {
                        float guess = number / 2.0f;
                        float epsilon = 0.00001f;  // 정확도
                        while (Math.abs(guess * guess - number) >= epsilon) {
                            guess = (guess + number / guess) / 2.0f;
                        }
                      return guess;
                      }
                      
                      float max_score = 0.0f; 
                      float[] dotProducts = new float[params.queryVectors.length];
                      float[] queryVector_sizes = new float[params.queryVectors.length];
                      
                      for(int i = 1; i <= params.VectorNum; i++) 
                      {
                          // 벡터 배열 변수들 초기화
                          for (int d = 0; d < params.queryVectors.length; d++) {
                            dotProducts[d] = 0.0f; 
                            queryVector_sizes[d] = 0.0f;
                          }
                          
                          float[] v = doc['vector'+i].vectorValue; 
                          float vm = doc['vector'+i].magnitude;  
                          
                          if (v[0] != 0)
                          {
                              // dot(내적) 2벡터간 내적을 구함.
                              for (int j = 0; j < v.length; j++) {
                                  for (int q = 0; q < params.queryVectors.length; q++) {
                                      dotProducts[q] += v[j] * params.queryVectors[q][j];
                                  }
                              }
                              
                              // 쿼리벡터의 크기를 구함.
                              for (int q = 0; q < params.queryVectors.length; q++) {
                                  
                                  float queryVector_sum_size = 0.0f;
                                  
                                  // 쿼리벡터의 제곱근을(*) 구
                                  for (int j = 0; j < v.length; j++) {
                                      queryVector_sum_size += params.queryVectors[q][j] * params.queryVectors[q][j];
                                  }
                                  
                                // 제곱근을 구하여 벡터의 크기를 반환
                                  queryVector_sizes[q] = sqrt(queryVector_sum_size);
                              }

                              // 스코어 구함 = 내적/벡터1크기*쿼리벡터크기
                              for (int q = 0; q < params.queryVectors.length; q++) {
                                  float score = dotProducts[q] / (vm * queryVector_sizes[q]);
                                  if (score > max_score) {
                                      max_score = score;
                                  }
                              }
                         }
                      }
                      return max_score
                    """,
                "params": 
                {
                  "queryVectors": query_vectors,  # 여러 벡터임베딩값 설정
                  "VectorNum": vectornum  # 벡터 수 설정
                }
            }
        }
    }
    
    return script_query

############################################################
## 쿼리벡터가 여러개이고 es에도 여러개인 경우 평균 구하기
# => 문단별 10개의 벡터와 10개의 쿼리 벡터를 서로 비교하여 최대값 10개 구하고 평균냄
############################################################
def make_max_query_vectors_and_avg_script(query_vectors, vectornum:int=10, uid_list:list=None)->str:
    # 문단별 10개의 벡터와 10개의 쿼리 벡터를 서로 비교하여 최대값 10개를 평균냄
    # => script """ 안에 코드는 java 임.
    # => "queryVectorMag": 0.1905 일 때 100% 일치하는 값은 9.98임(즉 10점 만점임)

    # uid_list가 있는 경우에는 해당하는 목록만 검색함
    if uid_list:
        query = { "bool" :{ "must": [ { "terms": { "rfile_name": uid_list } } ] } }
    else:
        query = { "match_all" : {} }
    
    script_query = {
        "script_score":{
             "query":query,
                "script":{
                    "source": """
                     
                      // 제곱근 구하는 함수
                      float sqrt(float number) {
                        float guess = number / 2.0f;
                        float epsilon = 0.00001f;  // 정확도
                        while (Math.abs(guess * guess - number) >= epsilon) {
                            guess = (guess + number / guess) / 2.0f;
                        }
                      return guess;
                      }
                      
                      float[] dotProducts = new float[params.queryVectors.length];
                      float[] queryVector_sizes = new float[params.queryVectors.length];
                      float[] max_score = new float[params.VectorNum];
                      
                      for(int i = 1; i <= params.VectorNum; i++) 
                      {
                          // 벡터 배열 변수들 초기화
                          for (int d = 0; d < params.queryVectors.length; d++) {
                            dotProducts[d] = 0.0f; 
                            queryVector_sizes[d] = 0.0f;
                          }
                          
                          float[] v = doc['vector'+i].vectorValue; 
                          float vm = doc['vector'+i].magnitude;  
                          float max_score_tmp = 0.0f;
                          
                          if (v[0] != 0)
                          {
                             // 2개의 벡터간 내적을 구함 
                             for (int q = 0; q < params.queryVectors.length; q++) {
                                for (int j = 0; j < v.length; j++) {
                                      dotProducts[q] += v[j] * params.queryVectors[q][j];
                                }
                              }
                              
                              // 쿼리벡터의 크기를 구함.
                              for (int q = 0; q < params.queryVectors.length; q++) {
                                  
                                  float queryVector_sum_size = 0.0f;
                                  
                                  // 쿼리벡터의 제곱근을(*) 구함
                                  for (int j = 0; j < v.length; j++) {
                                      queryVector_sum_size += params.queryVectors[q][j] * params.queryVectors[q][j];
                                  }
                                  
                                  // 제곱근을 구하여 벡터의 크기를 반환
                                  queryVector_sizes[q] = sqrt(queryVector_sum_size);
                              }

                              // 최대스코어를 구함
                              float score = 0.0f;
                              for (int q = 0; q < params.queryVectors.length; q++) {
                                  score = dotProducts[q] / (vm * queryVector_sizes[q]);
                                  if (score > max_score_tmp) {
                                      max_score_tmp = score;
                                  }
                              }
                              max_score[i-1] = max_score_tmp; // 최대스코어 배열에 저장해둠
                         }
                      }
                      
                      // 최대스코어들의 평균을 구함
                      int count = 0;
                      float total_max_score = 0.0f;
                      for (int f = 0; f <  params.VectorNum; f++) {
                        if (max_score[f] != 0)
                        {
                          count += 1;
                          total_max_score += max_score[f];
                        }
                      }
                      
                      float avg_max_score = total_max_score / count;
                      return avg_max_score
                    """,
                "params": 
                {
                  "queryVectors": query_vectors,  # 여러 벡터임베딩값 설정
                  "VectorNum": vectornum  # 벡터 수 설정
                }
            }
        }
    }
    
    return script_query
        
        