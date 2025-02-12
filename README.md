# 모아디엠(MoA.DM) 
- MocomsysAi.Document Manager 줄임말로 AI 문서관리자.
## 실행환경
- elasticsearch 8.x 구동중이어야 함.
- text 추출을 위한 MpowerAI 설치되어 있어야 함.
- sLLM 모델 연동은 ollama API 이용. 따라서 [ollama가 설치 및 구동](https://github.com/ollama/ollama)되어 있어야 함.
- 기타 주요 패키지
```
$ pip list | grep -E "transformer|elastic|fastapi|langchain|googletrans|httpcore|ollama"

elastic-transport                        8.0.1
elasticsearch                            7.17.0
fastapi                                  0.104.1
googletrans                              4.0.0rc1
httpcore                                 1.0.7
langchain                                0.3.13
langchain-community                      0.3.13
langchain-core                           0.3.28
langchain-experimental                   0.0.57
langchain-google-genai                   1.0.3
langchain-groq                           0.2.1
langchain-openai                         0.1.6
langchain-text-splitters                 0.3.4
langchainhub                             0.1.15
ollama                                   0.4.5
opentelemetry-instrumentation-fastapi    0.45b0
pytorch-transformers                     1.2.0
sentence-transformers                    2.2.2
transformers                             4.46.3

```
## 실행방법
- sh 명령어로 실행. Port는 **9003** 임
```
sh moadm.sh start
```
- ip:9003 접속

![image](https://github.com/user-attachments/assets/76c74a3e-6cd6-4b7d-be93-29605ea4a1b7)

## 테스트 

1. webtest.exe 구동 (test/webtest.zip 압축해제 후 실행)
2. 인덱스명, 탐지폴더, 분류폴더 등 설정
- 탐지폴더: 검색할 문서가 있는 폴더
- 분류폴더: 검색 대상이 될 문서가 있는 폴더 (*이미 벡터 인덱싱되어 있는 문서들이 있는 폴더)
  
![image](https://github.com/user-attachments/assets/99214168-81b4-442d-a96f-9bbf13106a14)

3. 탐지폴더에 있는 PPT 클릭(*탐지는 PPT 만됨)
4. 잠시후 아래처럼 Assistanct 화면이 나옴.
![image](https://github.com/user-attachments/assets/485bbb13-b7a0-4524-915c-f2316debf4a3)

5. [chat] 버튼 클릭하면 열린 문서에 대한 Q&A 할수 있음.

![image](https://github.com/user-attachments/assets/5717dcc9-7264-422f-8987-5b5c497b2808)


