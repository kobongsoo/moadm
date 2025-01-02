# 모아디엠(MoA.DM) 
- MocomsysAi.Document Manager 줄임말로 AI 문서관리자.
### 실행
- 일단 **elasticsearch 8.x** 버전이 구동되어 있어야함.
- text 추출은 사이냅문서필터를 이용함. 따라서 **MpowerAI 모듈**이 설치되어 있어야함.
- data/docs_settings.yaml 에 설정파일 수정해야 함.
- sh 명령어로 다음과 같이 실행. 이때 포트는 **9003** 임

```
sh moadm.sh start
```
![image](https://github.com/user-attachments/assets/0efd1886-0591-4e33-8993-6ed7f004e906)

### 주요 설치 패키지 버전

$ pip list | grep -E "transformer|elastic|fastapi|langchain|googletrans|httpcore"
```
elastic-transport                        8.0.1
elasticsearch                            7.17.0
fastapi                                  0.104.1
googletrans                              4.0.0rc1
httpcore                                 0.9.1
langchain                                0.3.13
langchain-community                      0.3.13
langchain-core                           0.3.28
langchain-experimental                   0.0.57
langchain-google-genai                   1.0.3
langchain-groq                           0.2.1
langchain-openai                         0.1.6
langchain-text-splitters                 0.3.4
langchainhub                             0.1.15
opentelemetry-instrumentation-fastapi    0.45b0
pytorch-transformers                     1.2.0
sentence-transformers                    2.2.2
transformers                             4.46.3
```
### 테스트 방법
1.webtest.exe 실행

![image](https://github.com/user-attachments/assets/03590689-2831-4fa6-8164-af7358d72a02)

2. ES 인덱스명, 검색수 등 설정. 탐지폴더, 분류 폴더 설정.
3. 탐지폴더에 pptx 파일 클릭하여 열면 아래 처럼 assistance 화면 실행됨.(요약, 유사문서등 표기)

![image](https://github.com/user-attachments/assets/f1aedd62-2958-49c8-a13b-cd73f5a30bda)

4. chat 클릭해서 문서에 대한 질문도 할수 있음.
![image](https://github.com/user-attachments/assets/808b0fe2-f93d-421d-8498-1830078020ee)





