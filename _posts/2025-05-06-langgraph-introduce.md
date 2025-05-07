---
title: LangGraph 소개 및 환경 구성
date: 2025-05-06 10:15:43 +/-TTTT
categories: [AI, LangGraph]
tags: [langgraph, langchain, langsmith, python, llm, generative-ai]
math: true
toc: true
pin: true
image:
  path: assets/posts/2025-05-06-langgraph-introduce/langgraph_logo.png
  alt: 
is_series: true
series_title: "LangGraph"
series_order: 1
---
## 1. LangGraph는

LangGraph는 LangChain의 추론(Reasoning) 흐름을 제어할 수 있는 상태 머신 프레임워크로, 상태(State)를 기반으로 각 노드(Node)가 동작합니다. LangGraph는 다음과 같은 문제를 개선하기 위해서 만들어졌습니다.

### 1.  1.  LLM만으로는 제한적

언어 모델(Language Model)은 단독으로 사용하기에는 기능이 다소 제한적입니다. 외부 도구(Tool)와 외부 문서와 같은 콘텍스트(Context)에 접근할 수 없고 다단계 워크플로(Workflow)를 수행할 수가 없습니다.

### 1.  2.  제어 흐름의 신뢰성 문제

LLM 애플리케이션에서 LLM 호출 전후에 `체인(Chain)`이라고 부르는 고정된 단계의 제어 흐름을 통해 매우 신뢰할 수 있는 워크플로를 구성할 수 있습니다. 체인을 사용해서 순서가 바뀌지 않고 동일한 순서로 로직이 실행됩니다. 그러면 문제가 발생했을 때 오류 발생 지점 파악이 쉽습니다. 그러나 우리는 한 단계 더 나아가 LLM 시스템이 자체적으로 제어 흐름을 선택할 수 있기를 원합니다. 이런 기능을 가능하게 하는 것이 `에이전트(Agent)`입니다.

에이전트는 LLM에 의해 정의된 제어 흐름입니다. LLM에서 제공하는 제어 흐름의 수준을 의사 결정의 자유도에 따라서 낮음에서 높음까지 제어 수준 조절이 가능합니다. 예를 들면 낮은 제어 수준은 1단계의 결과에 따라 2단계 혹은 3단계로 이동을 할 수가 있고, 완전 자율 에이전트처럼 높은 제어 수준은 단계 설계에서부터 실행, 오류처리 및 결과까지 스스로 처리할 수가 있습니다. 여기서 가장 큰 문제는 제어 흐름의 수준이 높아지면 신뢰성은 떨어지게 됩니다.

### 1.  3.  LangGraph의 신뢰성 곡선 개선

LangGraph는 상태와 중간 결괏값을 저장할 수 있는 `퍼시스턴스(Persistence)`, 결과를 청크(Chunk) 단위로 전달 할 수 있는 `스트리밍(Streaming)`, 중요 분기나 예외 처리에 사람 개입이 가능한 `휴먼-인-더-루프(Human-in-the-loop)`, 상태 그래프의 엄격한 단계 관리를 할 수 있는 `고급 제어 기능`이라는 네 가지 주요 기반을 제공하며, 이러한 요소들은 에이전트 워크플로 모듈의 핵심이 됩니다.

![신뢰성 곡선](assets/posts/2025-05-06-langgraph-introduce/20250506_120211_module01_01.png)
_신뢰성 곡선 - 제어 수준에 더 많은 자율성을 부여하면 신뢰성은 떨어집니다_

그리고 LangGraph는 전용 IDE를 통해 에이전트를 디버깅하고 시각화할 수 있는 통합된 시각적 환경을 제공합니다. 특히 LangGraph는 LangChain과 원활하게 연동되어, LangChain이 제공하는 다양한 LLM 커넥터(Connector) 및 벡터 스토어(Vector Store) 통합 기능을 그대로 활용할 수 있습니다.

![LangGraph Studio](assets/posts/2025-05-06-langgraph-introduce/20250506_122036_module01_02.png)
_LangGraph Studio로 시각적인 UI 제공_

## 2. LangGraph 설치 및 환경 구성

> 관련 코드는 [LangChain Academy Github](https://github.com/langchain-ai/langchain-academy){: target="_blank"}를 참조합니다.
{: .prompt-info }

### 2.  1.  Python 3.11 이상 버전 확인

```bash
python --version
```

> 미 설치 시 [Python 다운로드](https://www.python.org/downloads/){: target="_blank"}를 합니다.
{: .prompt-info }

### 2.  2.  공식 깃허브 예제 레파지토리 복제

```bash
git clone https://github.com/langchain-ai/langchain-academy.git
$ cd langchain-academy
```

### 2.  3.  가상환경 및 의존성 패키지 설치

```bash
# Mac/Linux/WSL
$ python3 −m venv lc−academy−env
$ source lc-academy-env/bin/activate
$ pip install -r requirements.txt
```

```bash
# Windows Powershell
PS> python3 -m venv lc-academy-env
PS> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
PS> lc-academy-env\scripts\activate
PS> pip install -r requirements.txt

```

### 2.  4.  Visual Studio Code 환경 (선택 1)

Visual Studio 또는 Jupyter 중 원하는 IDE를 선택합니다.

1. [Visual Studio Code 다운로드](https://code.visualstudio.com/){: target="_blank"} 후 실행
2. Visual Studio Code 확장(Extension)에서 Jupyter 설치

![LVisual Studio Code - Jupyter](assets/posts/2025-05-06-langgraph-introduce/20250506_122036_module01_05.png)
_Visual Studio Code 확장에서 Jupyter 설치_

### 2.  5.  Jupyter 환경 (선택 2)

1. [Jupyter 공식 문서](https://jupyter.org/install){: target="_blank"}를 참고하여 Jupyter Notebook을 설치합니다.
2. 설치 완료 후 다음 명령어를 실행합니다.

```bash
$ pip install notebook
$ jupyter notebook
```

### 2.  6.  OpenAI API, LangSmith, Tavily 가입 및 Key 세팅

#### 2. 6.  1.  OpenAI API 설정

OpenAI에서는 [OpenAI API](https://openai.com/index/openai-api/){: target="_blank"}를 ChatGPT 서비스 외에 별도로 제공합니다. 예제 실행을 위해 5달러 정도 구매하시면 여유 있게 사용할 수 있습니다.

> Goolge Gemini 2.0 Flash는 작성 일자 기준 무료로 제공 중입니다. 다음 포스팅에서 설명을 추가했습니다.
{: .prompt-info }

```bash
# Mac/Linux/WSL
$ export OPEN_API_KEY="OpenAPI 키를 입력하세요"
```

```bash
# Windows Powershell
PS> $env:OPEN_API_KEY = "OpenAPI 키를 입력하세요"
```

#### 2. 6.  2.  LangSmith 설정

AI 앱 성능을 디버깅, 테스트 및 모니터링을 할 수 있는 가시성 및 평가 플랫폼인 [LangSmith를 가입](https://www.langchain.com/langsmith){: target="_blank"}합니다.

무료라도 카드 등록을 해야 합니다.

```bash
# Mac/Linux/WSL
$ export LANGSMITH_API_KEY="LangSmith 키를 입력하세요"
```

```bash
# Windows Powershell
PS> $env:LANGSMITH_API_KEY="LangSmith 키를 입력하세요"
```

LangSmith 가입 후 무료 사용 설정을 위해서 `Usage & billing` > `Usage Configuration` > `Workspace usage configurations` 메뉴에서 `Total Trace Limit`을 `5000`으로 설정합니다.

![LangSmith 사용량 설정](assets/posts/2025-05-06-langgraph-introduce/20250506_154931_pic_01_langsmith.png)
_LangSmith 사용량 설정_

#### 2. 6.  3.  Tavily 설정

LLM과 RAG 시스템을 위한 검색 도구인 [Tavily에 가입](https://tavily.com/){: target="_blank"}합니다.

```bash
# Mac/Linux/WSL
$ export TAVILY_API_KEY="Tavily 키를 입력하세요"
```

```bash
# Windows Powershell
PS> $env:TAVILY_API_KEY="Tavily 키를 입력하세요"
```

### 2.  7.  LangGraph Studio Desktop 설치 및 구성

1. [Docker Desktop](https://docs.docker.com/get-started/get-docker/){: target="_blank"}을 OS에 맞게 설치합니다.
2. 다음 .env 환경파일을 구성합니다.
3. [LangGraph Studio Desktop 설치](https://github.com/langchain-ai/langgraph-studio){: target="_blank"}하고 인증 후 메인에서 실행할 `module-숫자/stduio/`폴더를 선택합니다.

```bash
# module-숫자/stduio/ 폴더에서 .env.example파일을 참조하여 .env 파일을 생성합니다.
cp .env.example .env
```

```bash
# 각 서비스 키를 입력합니다.
echo "OPENAI_API_KEY=\"$OPENAI_API_KEY\"" > .env
echo "ANTHROPIC_API_KEY=\"$ANTHROPIC_API_KEY\"" >> .env
echo "TAVILY_API_KEY=\"$TAVILY_API_KEY\"" >> .env
```

![LangGraph Studio Desktop 실행](assets/posts/2025-05-06-langgraph-introduce/20250506_122036_module01_03.png)
_LangGraph Studio Desktop 메인에서 폴더를 선택합니다_

### 2.  8.  LangGraph Studio Web에서 실행

```bash
# module-숫자/studio 폴더에서 LangGraph를 실행합니다.
langgraph dev
```

![LangGraph Studio Web 실행](assets/posts/2025-05-06-langgraph-introduce/20250506_122036_module01_04.png)
_LangGraph Studio Web 실행_


## 정리

LangGraph는 에이전트에 많은 자율성 부여 시 신뢰성이 떨어지는 것을 개선시킵니다.

그리고 Jupyter Notebook 및 Visual Stuio Code 등 다양한 IDE에서 개발할 수 있으며 OpenAI, Anthropic, Google 등 수많은 LLM을 지원합니다.

다음 포스팅은 LangGraph의 핵심 요소와 사용 방법에 대해서 살펴보겠습니다.


## References

* [LangGraph 공식 문서](https://langchain-ai.github.io/langgraph){: target="_blank"}
* [LangChain Academy](https://github.com/langchain-ai/langchain-academy){: target="_blank"}
