import time
import os
import json
import requests

import openai     # pip install "openai<1.0.0"
import sseclient  # pip install sseclient-py

# 구글 제미나이 SDK 설치
##!pip install google-generativeai
import google.generativeai as genai

from requests.exceptions import Timeout
from functools import lru_cache  # 캐싱

# 캐싱을 위한 256 데코레이터 설정
#@lru_cache(maxsize=256)  
#-----------------------------------------
# GPT를 이용한 text 생성2 
# => 아래 1번 함수와 차이점은 timeout(초)를 설정할수 있음.
#-----------------------------------------
def generate_text_davinci(gpt_model:str, prompt:str,  
                        timeout:int=20, stream:bool=False,
                        max_tokens:int=1024, temperature:float=0.5, top_p:float=0.1):
    
    error = 0
    answer:str = ""
    start_time = time.time()
    
    data = {
        'model': gpt_model,
        'prompt': prompt,
        'max_tokens': max_tokens,
        'temperature': temperature,# temperature 0~2 범위 : 작을수록 정형화된 답변, 클수록 유연한 답변(2는 엉뚱한 답변을 하므로, 1.5정도가 좋은것 같음=기본값은=1)
        'stream': stream,
        'top_p': top_p,      # 기본값은 1 (0.1이라고 하면 10% 토큰들에서 출력 토큰들을 선택한다는 의미)
        #'frequency_penalty':1, # 일반적으로 나오지 않는 단어를 억제하는 정도
        #'presence_penalty': 0.5 # 동일한 단어나 구문이 반복되는 것을 억제하는 정도
    }
    
    #print(f'data:{data}')

    for i in range(2): 
        try:
            response = requests.post(
                'https://api.openai.com/v1/completions',
                headers={'Accept': 'text/event-stream', 'Authorization': f'Bearer {openai.api_key}'},
                json=data,
                stream=stream,
                timeout=timeout  # Set your desired timeout in seconds
            )
            
            # 스트림 아닐때
            if stream == False:
                response.raise_for_status()  # Raise an HTTPError for bad responses
                output = json.loads(response.text)
                answer = output["choices"][0]["text"]
            else:  # 스트림 적용일때 => 한글자(토큰)씩 출력됨.
                client = sseclient.SSEClient(response)
                for event in client.events():
                    if event.data != '[DONE]':
                        chunk = json.loads(event.data)['choices'][0]['text']
                        answer += chunk
                        #print(chunk)
                        
            # 소요된 시간을 계산합니다.
            end_time = time.time()
            elapsed_time = "{:.2f}".format(end_time - start_time)
            #print(f'time:{elapsed_time}, answer:{answer}')
            
            return answer, error
        except Timeout:
            if i==0:
                continue
            answer = f'The request timed out.=>max:{timeout}'
            error = 1001
            return answer, error
        except Exception as e:
            answer = f"Error in API request: {e}"
            error = 1002
            return answer, error
    

# 캐싱을 위한 256 데코레이터 설정
#@lru_cache(maxsize=256)  
#-----------------------------------------
# GPT를 이용한 text 생성2 
# => 아래 1번 함수와 차이점은 timeout(초)를 설정할수 있음.
#-----------------------------------------
def generate_text_GPT2(gpt_model:str, prompt:str, system_prompt:str="", 
                       assistants:list=[], timeout:int=20, stream:bool=False,
                       max_tokens:int=1024, temperature:float=0.5, top_p:float=0.1):
    
    error = 0
    answer:str = ""
    messages:list = []
    start_time = time.time()
    
    if len(system_prompt) > 0:
        messages.append( {"role": "system", "content": system_prompt} )
    
    if len(assistants) > 0:
        for assistant in assistants:
            if assistant:
                messages.append( {"role": "assistant", "content": assistant} )
    
    if len(prompt) > 0:
        messages.append( {"role": "user", "content": prompt} )
        
    #print(f'messages:{messages}')
    
    data = {
        'model': gpt_model,
        'messages': messages,
        'max_tokens': max_tokens,
        'temperature': temperature,# temperature 0~2 범위 : 작을수록 정형화된 답변, 클수록 유연한 답변(2는 엉뚱한 답변을 하므로, 1.5정도가 좋은것 같음=기본값은=1)
        'stream': stream,
        'top_p': top_p,      # 기본값은 1 (0.1이라고 하면 10% 토큰들에서 출력 토큰들을 선택한다는 의미)
        #'frequency_penalty':1, # 일반적으로 나오지 않는 단어를 억제하는 정도
        #'presence_penalty': 0.5 # 동일한 단어나 구문이 반복되는 것을 억제하는 정도
    }
    
    #print(f'data:{data}')

    for i in range(2): 
        try:
            response = requests.post(
                'https://api.openai.com/v1/chat/completions',
                headers={'Content-Type': 'application/json', 'Authorization': f'Bearer {openai.api_key}'},
                json=data,
                stream=stream,
                timeout=timeout  # Set your desired timeout in seconds
            )
            
            #print(response)
            
            # 스트림 아닐때
            if stream == False:
                response.raise_for_status()  # Raise an HTTPError for bad responses
                output = json.loads(response.text)
                answer = output["choices"][0]["message"]["content"]
            else:  # 스트림 적용일때 => 한글자(토큰)씩 출력됨.
                client = sseclient.SSEClient(response)
                for event in client.events():
                    if event.data != '[DONE]':
                        chunk = json.loads(event.data)['choices'][0].get('delta', {}).get('content', '')
                        answer += chunk
                        #print(chunk)
                        
            # 소요된 시간을 계산합니다.
            end_time = time.time()
            elapsed_time = "{:.2f}".format(end_time - start_time)
            #print(f'time:{elapsed_time}')
            
            return answer, error
        except Timeout:
            if i==0:
                continue
            answer = f'The request timed out.=>max:{timeout}'
            error = 1001
            return answer, error
        except Exception as e:
            answer = f"Error in API request: {e}"
            error = 1002
            return answer, error
    
#-----------------------------------------
# GPT를 이용한 text 생성
#-----------------------------------------
# 캐싱을 위한 데코레이터 설정
#@lru_cache(maxsize=256)
def generate_text_GPT(gpt_model:str, prompt:str, system_prompt:str="",
                     assistants:list=[], stream:bool=False,
                     max_tokens:int=1024, temperature:float=0.5, top_p:float=0.1):
    
    error = 0
    answer:str = ""
    messages:list = []
    assert gpt_model, f'gpt_model is empty'
    assert prompt, f'prompt is empty'
    
    start_time = time.time()
    #print(f'len(messages):{len(messages)}') 
    #print()
    
    # 메시지 설정
    if len(system_prompt) > 0:
        messages.append( {"role": "system", "content": system_prompt} )
    
    if len(assistants) > 0:
        for assistant in assistants:
            if assistant:
                messages.append( {"role": "assistant", "content": assistant} )
    
    if len(prompt) > 0:
        messages.append( {"role": "user", "content": prompt} )
         
    try:
        # ChatGPT-API 호출하기
        response = openai.ChatCompletion.create(
            model=gpt_model,
            messages=messages,  
            max_tokens=max_tokens, # 토큰 수 
            temperature=temperature,  # temperature 0~2 범위 : 작을수록 정형화된 답변, 클수록 유연한 답변(2는 엉뚱한 답변을 하므로, 1.5정도가 좋은것 같음=기본값은=1)
            stream=stream,
            top_p=top_p, # 기본값은 1 (0.1이라고 하면 10% 토큰들에서 출력 토큰들을 선택한다는 의미)
            #frequency_penalty=1, # 일반적으로 나오지 않는 단어를 억제하는 정도
            #presence_penalty=0.5 # 동일한 단어나 구문이 반복되는 것을 억제하는 정도
            #stop=["다.","다!"] # . 나오면 중단
        )

        if stream==False:
            answer = response['choices'][0]['message']['content']  # + '다.' # 뒤에 '다' 붙여줌.
        else:
            for line in response:
                chunk = line['choices'][0].get('delta', {}).get('content', '')
                if chunk:
                    answer += chunk
                    #print(chunk)
                    #print(chunk, end='')
            
        # 소요된 시간을 계산합니다.
        end_time = time.time()
        elapsed_time = "{:.2f}".format(end_time - start_time)
        print(f'time:{elapsed_time}')
        
        return answer, error
    except Exception as e:
        answer = f"Error in API request: {e}"
        error = 1002
        return answer, error
#------------------------------------------------------------------

#-----------------------------------------
# 구글 Gemma : Huggingface 이용
# 참고 : https://huggingface.co/docs/api-inference/detailed_parameters
# text generation 처리
# -max_tokens: doc에는 0-250 이라고 되어 있지만 512로 해야 긴 문장 응답이 됨.
#-----------------------------------------
def generate_Gemma(hf_model_name:str, prompt:str, 
                   max_tokens:int=512, temperature:float=0.5, hf_auth_key:str=""):
    
    error = 0
    start_time = time.time()

    print(f'[gemma] : prompt: {prompt}\n\n')

    API_URL = "https://api-inference.huggingface.co/models/" + hf_model_name
    headers = {"Authorization": "Bearer " + hf_auth_key}
    payload = {
                "inputs": prompt,
                "temperature": temperature,          # 1.0=정형화된 결과
                "max_new_tokens": max_tokens        # max=250
              }
     
    try:
        # 파레메터들은  아래 url 첨조
        # https://huggingface.co/docs/api-inference/detailed_parameters
        answer = requests.post(API_URL, headers=headers, json=payload).json()
    except Timeout:
        answer = f'The request timed out.=>max:{timeout}'
        error = 1001
        return answer, error
    except Exception as e:
        answer = f"Error in API request: {e}"
        error = 1002
        return answer, error

    end_time = time.time()
    elapsed_time = "{:.2f}".format(end_time - start_time)

    try:
        text = answer[0]['generated_text']
    except Exception as e:
        error = 1003
        return answer, error
        
    #print(f'[gemma] : {text}\n\n')
    
    #start_index = text.find('\nA:') + 3
    start_index = len(prompt)
    answer = text[start_index:]
    print(f'[gemma] : start_index: {start_index}\n\nanswer:{answer}')
    
    
    #print(f'[gemma] (time:{elapsed_time}) {answer}')
    return answer, error
#-----------------------------------------

#-----------------------------------------
# [bong][2024-04-25]
# 구글 gemini 싱글턴 : Gemini API 이용
# - GOOGLE_API_KEY 발급 받아야 함. => ai.google.dev
# 참고 : https://wikidocs.net/228927
#-----------------------------------------
def generate_Gemini_single(model_name:str, prompt:str, google_api_key:str=""):

    assert model_name, f'mode_name is empty'
    assert google_api_key, f'google_api_key is empty'
    assert prompt, f'prompt is empty'
    
    error = 0
    start_time = time.time()
    #print(f'[Gemini(single)] : prompt: {prompt}\n\n')

    genai.configure(api_key=google_api_key)
    model = genai.GenerativeModel(model_name)
    response = model.generate_content(prompt)
     
    try:
        response = model.generate_content(prompt)
        answer = response.text
    except Timeout:
        answer = f'The request timed out.=>max:{timeout}'
        error = 1001
        return answer, error
    except Exception as e:
        answer = f"Error in API request: {e}"
        error = 1002
        return answer, error

    end_time = time.time()
    elapsed_time = "{:.2f}".format(end_time - start_time)  
    print(f'[Gemini(single)] (time:{elapsed_time}) {answer}')
    return answer, error
#-----------------------------------------

#-----------------------------------------
# [bong][2024-04-25]
# 구글 gemini 멀티턴 : Gemini API 이용
# - GOOGLE_API_KEY 발급 받아야 함. => ai.google.dev
# 참고 : https://wikidocs.net/228927
# history:list  => 이전에 질문/답변했던 내용이 리스트로 있어야 하며..
#-----------------------------------------
#-----------------------------------------
# 제미나이 : 이전질문과 답변 목록들을 받아서 제미나이 content 형태로 만드는 함수
# [Content 구조]
# {'role': 'user' , 'parts' :[쿼리]}, {'role': 'model', 'parts': ['text': 답변]
#----------------------------------------
def Gemini_make_history(prequerys:list=[], preanswers:list=[]):

    history:list=[]
    # 이전 쿼리와 질문을 gemini content 형태로 만들어서 history 리스트에 추가
    for prequery, preanswer in zip(prequerys, preanswers):
       content = {
           'role': 'user',
           'parts': [f'{prequery}']
       }
       history.append(content)
       
       content = {
           'role': 'model',
           'parts': [{
               'text': preanswer
           }]
       }
        
       history.append(content) 

    return history

#----------------------------------------
# gemini 멀티턴 
# => 멀티턴(multi turn) : 이전질문/답변 이어서 하는 방식
#----------------------------------------
def generate_Gemini_multi(model_name:str, prompt:str, google_api_key:str="",
                          prequerys:list=[], preanswers:list=[]):

    assert model_name, f'mode_name is empty'
    assert google_api_key, f'google_api_key is empty'
    assert prompt, f'prompt is empty'
    
    error = 0
    start_time = time.time()

    genai.configure(api_key=google_api_key)
    model = genai.GenerativeModel(model_name)

    # prequery, preanswer를 합쳐서 history 만듬.
    history:list=[]
    history = Gemini_make_history(prequerys=prequerys, preanswers=preanswers)

    #print(f'*[history] {history}\n\n')
    
    # 쿼리 형태를 만들고, history에 추가
    gemini_query = {'role':'user', 'parts':[f'{prompt}']}
    history.append(gemini_query)
      
    try:
        response = model.generate_content(history)
        answer = response.text
    except Timeout:
        answer = f'The request timed out.=>max:{timeout}'
        error = 1001
        return answer, error
    except Exception as e:
        answer = f"Error in API request: {e}"
        error = 1002
        return answer, error
        
    end_time = time.time()
    elapsed_time = "{:.2f}".format(end_time - start_time)  
    print(f'*[answer] (time:{elapsed_time}) {answer}\n\n')
    return answer, error
#-----------------------------------------
            
# main    
if __name__ == '__main__':
    main()