---
title: LangGraph 어시스턴트 구축하기 (1)
date: 2025-06-02 11:15:43 +/-TTTT
description : LangGraph 메모리 개념과 휴먼-인-더-루프 기반 멀티 에이전트 워크플로를 살펴보겠습니다.
categories: [AI, LangGraph]
tags: [langgraph, langchain, langsmith, python, llm, generative-ai]
math: true
toc: true
pin: false
image:
    path: assets/posts/2025-05-06-langgraph-introduce/langgraph_logo.png
    alt:
is_series: true
series_title: "LangGraph"
series_order: 6
---
이전 포스팅에서 다뤘던 `메모리` 개념과 `휴먼-인-더-루프` 기반으로 `멀티 에이전트(multi-agent)` 워크플로를 살펴보겠습니다.


> 학습할 리소스는 [LangChain Academy Github](https://github.com/langchain-ai/langchain-academy){: target="_blank"}를 사용합니다.
{: .prompt-info }

## 1.   병렬 노드 실행 (Parallel node execution)

`멀티 에이전트 리서치 어시스턴트`를 구축하기 위해 먼저 LangGraph의 `제어 가능성(controllability)`에 대해서 살펴보겠습니다.

### 1.  1.  팬아웃(Fan out)과 팬인(fan in)

각 단계마다 상태를 덮어쓰는 간단한 `선형 그래프`를 생성합니다.

```python
# 환경 구성
%%capture --no-stderr
%pip install -U  langgraph tavily-python wikipedia langchain_openai langchain_community langgraph_sdk
```

```python
import os, getpass

def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

_set_env("OPENAI_API_KEY")
```

```python
from IPython.display import Image, display

from typing import Any
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    state: str

class ReturnNodeValue:
    def __init__(self, node_secret: str):
        self._value = node_secret

    def __call__(self, state: State) -> Any:
        print(f"Adding {self._value} to {state['state']}")
        return {"state": [self._value]}

# 노드 추가
builder = StateGraph(State)

# 각 노드를 node_secret으로 초기화
builder.add_node("a", ReturnNodeValue("I'm A"))
builder.add_node("b", ReturnNodeValue("I'm B"))
builder.add_node("c", ReturnNodeValue("I'm C"))
builder.add_node("d", ReturnNodeValue("I'm D"))

# 흐름 정의
builder.add_edge(START, "a")
builder.add_edge("a", "b")
builder.add_edge("b", "c")
builder.add_edge("c", "d")
builder.add_edge("d", END)
graph = builder.compile()

그래프 이미지
display(Image(graph.get_graph().draw_mermaid_png(max_retries=5, retry_delay=2.0)))
```

상태를 덮어씁니다.

```python
graph.invoke({"state": []})
```

```
# 출력

Adding I'm A to []
Adding I'm B to ["I'm A"]
Adding I'm C to ["I'm B"]
Adding I'm D to ["I'm C"]

{'state': ["I'm D"]}
```

이제 `b`와 `c`를 병렬로 실행하고 `d`를 실행합니다.

쉽게 `a`에서 `b`와 `c`로 `팬아웃(fan-out)`하고, 다시 `d`에서 `팬인(fan-in)`하는 흐름을 정의할 수 있습니다.

상태 업데이트는 각 단계의 끝에서 적용됩니다.

```python
builder = StateGraph(State)

# 각 노드를 node_secret으로 초기화
builder.add_node("a", ReturnNodeValue("I'm A"))
builder.add_node("b", ReturnNodeValue("I'm B"))
builder.add_node("c", ReturnNodeValue("I'm C"))
builder.add_node("d", ReturnNodeValue("I'm D"))

# 흐름(Flow) 정의
builder.add_edge(START, "a")
builder.add_edge("a", "b")   # a에서 b로 팬아웃
builder.add_edge("a", "c")   # a에서 c로 팬아웃
builder.add_edge("b", "d")   # b와 c가 d에서 팬인
builder.add_edge("c", "d")
builder.add_edge("d", END)
graph = builder.compile()

display(Image(graph.get_graph().draw_mermaid_png()))

```

에러가 발생합니다. 이유는 `b`와 `c`가 같은 단계에서 동일한 `state` 키/채널에 동시에 값을 쓰기 때문입니다.

```python
from langgraph.errors import InvalidUpdateError
try:
    graph.invoke({"state": []})
except InvalidUpdateError as e:
    print(f"An error occurred: {e}")
```

```
# 출력 - b와 c가 state 키에 동시 작성으로 에러 발생

Adding I'm A to []
Adding I'm B to ["I'm A"]
Adding I'm C to ["I'm A"]
An error occurred: At key 'state': Can receive only one value per step. Use an Annotated key to handle multiple values.
For troubleshooting, visit: https://python.langchain.com/docs/troubleshooting/errors/INVALID_CONCURRENT_GRAPH_UPDATE
```

팬아웃을 사용할 때 여러 단계에서 동일한 채널/키에 값을 쓸 경우, 반드시 `리듀서(reducer)`를 사용해야 합니다.

이전 포스팅에서 설명했듯이 `operator.add`는 파이썬 내장 모듈인 operator의 함수입니다.

`operator.add`를 리스트에 적용하면, 리스트를 합치는 동작을 합니다.

```python
import operator
from typing import Annotated

class State(TypedDict):
    # operator.add 리듀서 함수로 인해 추가 전용(append-only)이 됨
    state: Annotated[list, operator.add]

# 노드 추가
builder = StateGraph(State)

# 각 노드를 node_secret으로 초기화
builder.add_node("a", ReturnNodeValue("I'm A"))
builder.add_node("b", ReturnNodeValue("I'm B"))
builder.add_node("c", ReturnNodeValue("I'm C"))
builder.add_node("d", ReturnNodeValue("I'm D"))

# 흐름 정의
builder.add_edge(START, "a")
builder.add_edge("a", "b")   # a에서 b로 팬아웃
builder.add_edge("a", "c")   # a에서 c로 팬아웃
builder.add_edge("b", "d")   # b와 c가 d에서 팬인
builder.add_edge("c", "d")
builder.add_edge("d", END)
graph = builder.compile()

# 그래프 이미지
display(Image(graph.get_graph().draw_mermaid_png()))

```

```python
graph.invoke({"state": []})
```

```
# 출력

Adding I'm A to []
Adding I'm B to ["I'm A"]
Adding I'm C to ["I'm A"]
Adding I'm D to ["I'm A", "I'm B", "I'm C"]

{'state': ["I'm A", "I'm B", "I'm C", "I'm D"]}
```

`b`와 `c`에서 병렬로 업데이트된 내용이 state에 `추가`가 되었습니다.

> 여기서는 `b`와 `c`가 팬인 할 때 순서가 보장되지 않습니다.
{: .prompt-info }

### 1.  2.  노드가 모두 끝날 때까지 대기

이번에는 병렬 경로 중 하나가 다른 경로보다 `더 많은 단계`를 가지는 경우를 살펴보겠습니다.

```python
builder = StateGraph(State)

# 각 노드를 node_secret으로 초기화
builder.add_node("a", ReturnNodeValue("I'm A"))
builder.add_node("b", ReturnNodeValue("I'm B"))
builder.add_node("b2", ReturnNodeValue("I'm B2"))
builder.add_node("c", ReturnNodeValue("I'm C"))
builder.add_node("d", ReturnNodeValue("I'm D"))

# 흐름 정의
builder.add_edge(START, "a")
builder.add_edge("a", "b")    # a에서 b로 이동
builder.add_edge("a", "c")    # a에서 c로 이동 (병렬 분기)
builder.add_edge("b", "b2")   # b에서 b2로 이동 (b 경로가 한 단계 더 있음)
builder.add_edge(["b2", "c"], "d")  # b2와 c가 모두 끝나야 d로 이동 (팬인)
builder.add_edge("d", END)
graph = builder.compile()

display(Image(graph.get_graph().draw_mermaid_png()))

```

![b경로 단계가 많은 경우 흐름](assets/posts/2025-06-02-langgraph-building-assistant-1st/fanout-fanin_01.png)
_b경로 단계가 많은 경우 흐름_

이 경우, `b`, `b2`, 그리고 `c`가 모두 `동일한 단계`에 속하게 됩니다.

그래프는 이 노드들이 `모두 완료될 때까지 기다렸다가` 다음 단계인 `d`로  진행합니다.

```python
graph.invoke({"state": []})
```

```
# 출력 - 순서가 보장되지 않음

Adding I'm A to []
Adding I'm B to ["I'm A"]
Adding I'm C to ["I'm A"]
Adding I'm B2 to ["I'm A", "I'm B", "I'm C"]
Adding I'm D to ["I'm A", "I'm B", "I'm C", "I'm B2"]

{'state': ["I'm A", "I'm B", "I'm C", "I'm B2", "I'm D"]}

```

### 1.  3.  상태 업데이트 순서 지정하기

이전 팬인 예제에서 보듯이 각 단계에서 상태 업데이트의 순서를 제어할 수는 없습니다.

LangGraph가 그래프 토폴로지에 따라 결정하는 고유의 순서가 있으며, 직접 통제할 수는 없습니다.

위 예에서 `c`가 `b2`보다 먼저 추가되는 것을 볼 수 있습니다.

`커스텀 리듀서`를 사용하면, 상태 업데이트를 `정렬하도록 동작`을 커스터마이징할 수 있습니다.

```python
def sorting_reducer(left, right):
    """ 리스트의 값을 결합하고 정렬하는 함수 """
    if not isinstance(left, list):
        left = [left]

    if not isinstance(right, list):
        right = [right]
  
    return sorted(left + right, reverse=False) # 문자열 오름차순 정리

class State(TypedDict):
    # sorting_reducer가 state의 값을 정렬함
    state: Annotated[list, sorting_reducer]

# 노드 추가
builder = StateGraph(State)

# 각 노드를 node_secret으로 초기화
builder.add_node("a", ReturnNodeValue("I'm A"))
builder.add_node("b", ReturnNodeValue("I'm B"))
builder.add_node("b2", ReturnNodeValue("I'm B2"))
builder.add_node("c", ReturnNodeValue("I'm C"))
builder.add_node("d", ReturnNodeValue("I'm D"))

# 흐름 정의
builder.add_edge(START, "a")
builder.add_edge("a", "b")      # a에서 b로 이동
builder.add_edge("a", "c")      # a에서 c로 이동 (병렬 분기)
builder.add_edge("b", "b2")     # b에서 b2로 이동 (b 경로가 한 단계 더 있음)
builder.add_edge(["b2", "c"], "d")  # b2와 c가 모두 끝나야 d로 이동 (팬인)
builder.add_edge("d", END)
graph = builder.compile()

# 그래프 이미지
display(Image(graph.get_graph().draw_mermaid_png()))

```

```python
graph.invoke({"state": []})
```

```
# 출력 - 문자열이 순서대로 정렬됨

Adding I'm A to []
Adding I'm B to ["I'm A"]
Adding I'm C to ["I'm A"]
Adding I'm B2 to ["I'm A", "I'm B", "I'm C"]
Adding I'm D to ["I'm A", "I'm B", "I'm B2", "I'm C"]

{'state': ["I'm A", "I'm B", "I'm B2", "I'm C", "I'm D"]}
```

`리듀서`가 업데이트된 상태 값들을 정렬했습니다.

`sorting_reducer` 예제는 모든 값을 전역적으로 정렬하고 다음 기능도 가능합니다.

1. 병렬 단계에서 각 결과를 합치지 않고 state의 `별도 필드에 저장`할 수 있습니다.
2. 병렬 단계 이후에 `sink 노드`를 사용해 `별도 필드에 저장된 이 결과들을 합치고 정렬`할 수 있습니다.
3. 병합 후 `임시 필드는 정리`, 즉 state에서 지웁니다.

더 자세한 내용은 [공식 문서](https://langchain-ai.github.io/langgraph/how-tos/branching/#stable-sorting){: target="_blank"}를 참고하실 수 있습니다.

### 1.  4.  LLM과 함께 사용하기 (실습)

두 개의 외부 소스(`위키피디아`와 `웹 검색`)에서 컨텍스트를 수집한 다음, LLM이 질문에 답변하도록 구성합니다.

```python
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o", temperature=0) 
```

```python
class State(TypedDict):
    question: str
    answer: str
    context: Annotated[list, operator.add]
```

다양한 웹 검색 도구 중 [Tavily](https://tavily.com/){: target="_blank"}를 사용합니다.

추가로 `TAVILY_API_KEY`가 설정되어 있어야 합니다.

```python
import os, getpass
def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")
_set_env("TAVILY_API_KEY")
```

```python
from langchain_core.messages import HumanMessage, SystemMessage

from langchain_community.document_loaders import WikipediaLoader
from langchain_community.tools import TavilySearchResults

def search_web(state):
  
    """ 웹 검색에서 문서를 가져옵니다 """

    # 검색 수행
    tavily_search = TavilySearchResults(max_results=3)
    search_docs = tavily_search.invoke(state['question'])

     # 포맷팅
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document href="{doc["url"]}">\n{doc["content"]}\n</Document>'
            for doc in search_docs
        ]
    )

    return {"context": [formatted_search_docs]} 

def search_wikipedia(state):
  
    """ 위키피디아에서 문서를 가져옵니다 """

    # 검색 수행
    search_docs = WikipediaLoader(query=state['question'], 
                                  load_max_docs=2).load()

     # 포맷팅
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}">\n{doc.page_content}\n</Document>'
            for doc in search_docs
        ]
    )

    return {"context": [formatted_search_docs]} 

def generate_answer(state):
  
    """ 질문에 답변을 생성하는 노드 """

    # 상태에서 값 가져오기
    context = state["context"]
    question = state["question"]

    # 템플릿 정의
    answer_template = """Answer the question {question} using this context: {context}"""
    answer_instructions = answer_template.format(question=question, 
                                                       context=context)  
  
    # 답변 생성
    answer = llm.invoke([SystemMessage(content=answer_instructions)]+[HumanMessage(content=f"Answer the question.")])
  
    # 상태에 답변 추가
    return {"answer": answer}

# 노드 추가
builder = StateGraph(State)

# 각 노드를 node_secret으로 초기화
builder.add_node("search_web",search_web)
builder.add_node("search_wikipedia", search_wikipedia)
builder.add_node("generate_answer", generate_answer)

# 흐름 정의
builder.add_edge(START, "search_wikipedia")
builder.add_edge(START, "search_web")
builder.add_edge("search_wikipedia", "generate_answer")
builder.add_edge("search_web", "generate_answer")
builder.add_edge("generate_answer", END)
graph = builder.compile()

display(Image(graph.get_graph().draw_mermaid_png()))

```

```python
result = graph.invoke({"question": "How were Nvidia's Q2 2024 earnings"})
result['answer'].content
```

```python
# 출력

"Nvidia's Q2 2024 earnings were exceptionally strong. The company reported fiscal second-quarter revenue of $30.04 billion, more than doubling from the same period a year ago, with net income at $16.6 billion, also more than doubling from the year-ago period. The data center segment was a significant driver of this growth, with revenue reaching a record high of $26.3 billion, up 16% from Q1 and 154% from last year’s Q2. Other segments like Gaming and AI PC, Professional Visualization, and Automotive and Robotics also saw year-over-year growth. Despite the increase in operating expenses, Nvidia's financial performance exceeded analysts' estimates, driven by surging demand for artificial intelligence."
```

위 그래프 흐름은 `병렬로 두 개의 노드가 실행`됩니다.

`search_web 노드`에서 `Tavily API`를 이용해 입력된 질문에 대한 웹 문서를 3개까지 검색 후 포매팅해서 문자열로 만든 후 결과를 state의 `context` 필드에 리스트 타입으로 저장합니다.

`search_wikipedia 노드`에서 위키피디아에서 입력된 질문에 대한 문서를 2개까지 검색 후 포매팅해서 문자열로 만든 후 결과를 마찬가지로 state의 `context` 필드에 리스트 타입으로 저장합니다.

이후 `generate_answer 노드`에서 입력된 질문과 state의 `context`를 읽고 템플릿을 통해 프롬프트를 생성합니다.

마지막으로, 이 프롬프트를 LLM에 전달해 답변을 생성하고 state의 `answer` 필드에 저장합니다.

## 2.   서브 그래프 (Sub-graphs)

`서브 그래프`를 사용하면, 그래프의 서로 다른 부분에서 `각기 다른 상태를 생성하고 관리`할 수 있습니다.

### 2.  1.  상태 (State)

각기 독립적인 상태를 소유한 다수의 에이전트 팀에서는 `멀티 에이전트 시스템`이 특히 유용합니다.

예를 들어 로그 시스템이 있고 이 로그 시스템은 두 개의 하위 작업인 `로그 요약`과 `장애 원인 찾기`를 별도의 서브 그래프에서 수행을 합니다.

중요한 점이 그래프 간의 통신 방식입니다. 중복되는 키(예: docs)를 통해 통신이 이루어집니다.

아래 그림처럼 서브 그래프는 부모 그래프의 `docs`를 받아올 수 있고

부모 그래프는 서브 그래프의 `summary_report`와 `failur_report`를 가져올 수 있습니다.

![부모와 서브 그래프 흐름](assets/posts/2025-06-02-langgraph-building-assistant-1st/subgraph_01.png)
_부모와 서브 그래프 흐름_

### 2.  2.  입력 (Input)

그래프에 입력될 로그를 위한 스키마를 정의합니다.

```python
# 환경 구성
%%capture --no-stderr
%pip install -U  langgraph
```

```python
# 추적을 위한 LangSmith 설정
import os, getpass

def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

_set_env("LANGSMITH_API_KEY")
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "langchain-academy"
```

```python
from operator import add
from typing_extensions import TypedDict
from typing import List, Optional, Annotated

# 로그 구조
class Log(TypedDict):
    id: str
    question: str
    docs: Optional[List]
    answer: str
    grade: Optional[int]
    grader: Optional[str]
    feedback: Optional[str]
```

### 2.  3.  서브 그래프(Sub graphs) 생성

다음은 `FailureAnalysisState`를 사용하는 `장애 분석 서브 그래프`입니다.

```python
from IPython.display import Image, display
from langgraph.graph import StateGraph, START, END

# 장애 분석 서브 그래프
class FailureAnalysisState(TypedDict):
    cleaned_logs: List[Log]
    failures: List[Log]
    fa_summary: str
    processed_logs: List[str]

class FailureAnalysisOutputState(TypedDict):
    fa_summary: str
    processed_logs: List[str]

def get_failures(state):
    """ 실패가 포함된 로그 가져오기 """
    cleaned_logs = state["cleaned_logs"]
    failures = [log for log in cleaned_logs if "grade" in log]
    return {"failures": failures}

def generate_summary(state):
    """ 실패에 대한 요약 생성 """
    failures = state["failures"]
    # Add fxn: fa_summary = summarize(failures)
    fa_summary = "Poor quality retrieval of Chroma documentation."
    return {"fa_summary": fa_summary, "processed_logs": [f"failure-analysis-on-log-{failure['id']}" for failure in failures]}

fa_builder = StateGraph(FailureAnalysisState,output=FailureAnalysisOutputState)
fa_builder.add_node("get_failures", get_failures)
fa_builder.add_node("generate_summary", generate_summary)
fa_builder.add_edge(START, "get_failures")
fa_builder.add_edge("get_failures", "generate_summary")
fa_builder.add_edge("generate_summary", END)

graph = fa_builder.compile()
display(Image(graph.get_graph().draw_mermaid_png()))


```

아래는 `QuestionSummarizationState`를 사용하는 `질문 요약 서브 그래프`입니다.

```python
# 요약 서브 그래프
class QuestionSummarizationState(TypedDict):
    cleaned_logs: List[Log]   
    qs_summary: str   
    report: str   
    processed_logs: List[str]   

class QuestionSummarizationOutputState(TypedDict):
    report: str   
    processed_logs: List[str]   

def generate_summary(state):
    cleaned_logs = state["cleaned_logs"]
    # 함수 추가: summary = summarize(generate_summary)
    fa_summary = "Poor quality retrieval of Chroma documentation."
    return {"qs_summary": summary, "processed_logs": [f"summary-on-log-{log['id']}" for log in cleaned_logs]}

def send_to_slack(state):
    qs_summary = state["qs_summary"]
    # 함수 추가: report = report_generation(qs_summary)
    report = "foo bar baz"
    return {"report": report}

qs_builder = StateGraph(QuestionSummarizationState, output=QuestionSummarizationOutputState)
qs_builder.add_node("generate_summary", generate_summary)
qs_builder.add_node("send_to_slack", send_to_slack)
qs_builder.add_edge(START, "generate_summary")
qs_builder.add_edge("generate_summary", "send_to_slack")
qs_builder.add_edge("send_to_slack", END)

graph = qs_builder.compile()
display(Image(graph.get_graph().draw_mermaid_png()))

```

### 2.  4.  서브 그래프를 부모 그래프에 추가

서브 그래프를 하나로 합치기 위해 `EntryGraphState`로 부모 그래프를 생성합니다.

그리고 서브 그래프들을 노드로 추가합니다.

```
entry_builder.add_node("question_summarization", qs_builder.compile())
entry_builder.add_node("failure_analysis", fa_builder.compile())
```

```python
# 부모 그래프
class EntryGraphState(TypedDict):
    raw_logs: List[Log]
    cleaned_logs: Annotated[List[Log], add] # 두 개의 서브 그래프에서 사용됨
    fa_summary: str # failure_analysis 서브 그래프에서만 생성
    report: str # question_summarization 서브 그래프에서만 생성
    processed_logs:  Annotated[List[int], add] # 두 개의 서브 그래프에서 모두 생성
```

그런데 `cleaned_logs`는 수정되지 않고 각 서브 그래프의 공통 `입력값`으로만 사용되는데 왜 리듀서가 필요할까요?

```python
cleaned_logs: Annotated[List[Log], add] # 이 값은 두 서브그래프에서 모두 사용
```

이유는 병렬로 수행되는 서브 그래프는 `cleand_logs` 키를 포함해서 입력 상태의 `모든 키`를, 출력할 때도 `기본적으로 포함`합니다.

이때 서로 다른 서브 그래프들이 `동일한 키를 반환`하면 충돌이 발생할 수 있기 때문에, 값을 병합하기 위한 `operator.add`와 같은 리듀서가 필요합니다.

다른 방법은 각 서브 그래프마다 `출력 상태 스키마`를 따로 정의하고, 각 서브 그래프가 `서로 다른 키만을 출력`하도록 하면 됩니다. 그래서 모든 서브 그래프가 `cleaned_logs`를 출력할 필요는 없습니다.

> 코드에서는 `cleand_logs`를 출력하진 않지만, 개념 이해를 위해 `리듀서`를 적용
{: .prompt-info }

```python
# 부모 그래프
class EntryGraphState(TypedDict):
    raw_logs: List[Log]
    cleaned_logs: List[Log]
    fa_summary: str # failure_analysis 서브 그래프에서만 생성
    report: str # question_summarization 서브 그래프에서만 생성
    processed_logs:  Annotated[List[int], add] # 두 개의 서브 그래프에서 모두 생성

def clean_logs(state):
    # 로그 가져오기
    raw_logs = state["raw_logs"]
    # 데이터 정제: raw_logs -> docs (실제 정제 기능은 구현하지 않음)
    cleaned_logs = raw_logs
    return {"cleaned_logs": cleaned_logs}

entry_builder = StateGraph(EntryGraphState)
entry_builder.add_node("clean_logs", clean_logs)
entry_builder.add_node("question_summarization", qs_builder.compile())
entry_builder.add_node("failure_analysis", fa_builder.compile())

entry_builder.add_edge(START, "clean_logs")
entry_builder.add_edge("clean_logs", "failure_analysis")
entry_builder.add_edge("clean_logs", "question_summarization")
entry_builder.add_edge("failure_analysis", END)
entry_builder.add_edge("question_summarization", END)

graph = entry_builder.compile()

from IPython.display import Image, display

# xray=1로 설정하면 서브 그래프의 내부 구조가 시각화됨
display(Image(graph.get_graph(xray=1).draw_mermaid_png()))
```

```python
# 더미 로그 정의
question_answer = Log(
    id="1",
    question="How can I import ChatOllama?",
    answer="To import ChatOllama, use: 'from langchain_community.chat_models import ChatOllama.'",
)

question_answer_feedback = Log(
    id="2",
    question="How can I use Chroma vector store?",
    answer="To use Chroma, define: rag_chain = create_retrieval_chain(retriever, question_answer_chain).",
    grade=0,
    grader="Document Relevance Recall",
    feedback="The retrieved documents discuss vector stores in general, but not Chroma specifically",
)

raw_logs = [question_answer,question_answer_feedback]
graph.invoke({"raw_logs": raw_logs})
```

```
# 출력

{'raw_logs': [{'id': '1',
   'question': 'How can I import ChatOllama?',
   'answer': "To import ChatOllama, use: 'from langchain_community.chat_models import ChatOllama.'"},
  {'id': '2',
   'question': 'How can I use Chroma vector store?',
   'answer': 'To use Chroma, define: rag_chain = create_retrieval_chain(retriever, question_answer_chain).',
   'grade': 0,
   'grader': 'Document Relevance Recall',
   'feedback': 'The retrieved documents discuss vector stores in general, but not Chroma specifically'}],
 'cleaned_logs': [{'id': '1',
   'question': 'How can I import ChatOllama?',
   'answer': "To import ChatOllama, use: 'from langchain_community.chat_models import ChatOllama.'"},
  {'id': '2',
   'question': 'How can I use Chroma vector store?',
   'answer': 'To use Chroma, define: rag_chain = create_retrieval_chain(retriever, question_answer_chain).',
   'grade': 0,
   'grader': 'Document Relevance Recall',
   'feedback': 'The retrieved documents discuss vector stores in general, but not Chroma specifically'}],
 'fa_summary': 'Poor quality retrieval of Chroma documentation.',
 'report': 'foo bar baz',
 'processed_logs': ['failure-analysis-on-log-2',
  'summary-on-log-1',
  'summary-on-log-2']}
```

## 3.   맵리듀스 (Map-Reduce) 개념 및 실습

맵리듀스는 작업을 효율적으로 `분해하고 병렬 처리`하는 데 필수적인 개념입니다.

[맵리듀스](https://langchain-ai.github.io/langgraph/how-tos/graph-api/#map-reduce-and-the-send-api){: target="_blank"}에 대해서 예제를 통해 자세히 알아보겠습니다.

> 대규모 데이터 저장과 처리를 하는 Hadoop의 `맵리듀스`와 개념만 비슷하고 LangGraph만의 목적과 방식으로 구성됩니다.
{: .prompt-info }

```python
# 환경 구성
%%capture --no-stderr
%pip install -U langchain_openai langgraph
```

```python
# 추적을 위한 LangSmith 구성
_set_env("LANGSMITH_API_KEY")
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "langchain-academy"
```

### 3.  1.  작업 목표

`맵리듀스` 작업 개념은 다음과 같습니다.

`맵`을 통해 하나의 작업을 더 작은 하위 작업으로 나누고 `병렬로 처리`합니다.

`리듀스`로 병렬로 처리된 하위 작업을 하나로 모아 `집계`합니다.

그리고 `베스트 조크 선택`을 위한 그래프 흐름은 다음과 같습니다.

1. `진입점(entry point)을 정의`합니다.
2. 사용자로부터 `주제를 입력 ("topic": "animals")`받습니다.
3. 해당 주제로부터 조크를 위한 `하위 주제를 생성 (generate_topics)`합니다.
4. `맵` 단계인 각 하위 주제를 `조크 생성 (generate_joke)` 노드에 전달합니다.
5. `리듀스` 단계에서 생성된 조크들 중 가장 재미있는 `베스트 조크 (best_joke)`를 선택합니다.

조크의 생성과 선택 작업은 gpt-4o LLM을 사용합니다.

![맵리듀스 그래프 흐름](assets/posts/2025-06-02-langgraph-building-assistant-1st/map-reduce_01.png)
_맵리듀스 그래프 흐름_

```python
from langchain_openai import ChatOpenAI

# 사용할 프롬프트
subjects_prompt = """Generate a list of 3 sub-topics that are all related to this overall topic: {topic}."""
joke_prompt = """Generate a joke about {subject}"""
best_joke_prompt = """Below are a bunch of jokes about {topic}. Select the best one! Return the ID of the best one, starting 0 as the ID for the first joke. Jokes: \n\n  {jokes}"""

# LLM
model = ChatOpenAI(model="gpt-4o", temperature=0) 
```

### 3.  2.  상태 및 노드 설정하기

`OverallState`상태에는 `jokes`라는 키가 있고 이 키에 병렬로 생성된 조크들이 누적됩니다.

```python
# 상태 정의

import operator
from typing import Annotated
from typing_extensions import TypedDict
from pydantic import BaseModel

class Subjects(BaseModel):
    subjects: list[str]

class BestJoke(BaseModel):
    id: int
  
class OverallState(TypedDict):
    topic: str
    subjects: list
    jokes: Annotated[list, operator.add]
    best_selected_joke: str
```

`generate_topics` 노드에서 조크들을 위한 주제들을 생성합니다.

```python
# 주제(Topic)를 생성할 노드

def generate_topics(state: OverallState):
    prompt = subjects_prompt.format(topic=state["topic"])
    response = model.with_structured_output(Subjects).invoke(prompt)
    return {"subjects": response.subjects}
```

여기에서 핵심인 [`Send`](https://langchain-ai.github.io/langgraph/concepts/low_level/#send){: target="_blank"}를 통해 주제마다 조크를 하나씩 생성하고 개수에 상관없이 `자동으로 병렬 실행`합니다.

Send에서 `generate_joke`는 그래프 내의 노드이고 `{"subject": s}`는 해당 노드에 전달할 `상태`입니다.

여기서 `generate_joke`는 전체 상태가 아닌 `topic 키`만 사용하는 `JokeState` 내부 상태를 입력으로 받습니다.

그래서 `Send`를 통해 전달하는 값은 `JokeState`에 자동으로 매핑되기 때문에,`OverallState` 상태와 동일하지 않아도 됩니다.

```python
# 조건부 엣지 설정을 위한 함수 정의 - 기존 생성된 주제에 맞게 조크들을 생성

from langgraph.constants import Send
def continue_to_jokes(state: OverallState):
    return [Send("generate_joke", {"subject": s}) for s in state["subjects"]]
```

### 3.  3.  조크 생성을 위한 맵 구성하기

맵 구성을 위해 `조크들을 생성하고 병렬 처리`를 하는 `generate_joke` 노드를 생성합니다.

이 노드는 생성된 조크들을 `OverallState`의 `jokes` 키에 기록합니다.

`jokes` 키에는 `리듀서`가 설정되어 있어, 병렬로 생성된 조크 목록을 `자동으로 결합`합니다.

```python
# 각 주제에 대해 하나의 조크를 생성하는 노드

class JokeState(TypedDict):
    subject: str

class Joke(BaseModel):
    joke: str

def generate_joke(state: JokeState):
    prompt = joke_prompt.format(subject=state["subject"])
    response = model.with_structured_output(Joke).invoke(prompt)
    return {"jokes": [response.joke]}
### 베스트 조크 선택을 위한 리듀스 구성하기
```

`리듀스`를 작업을 위한 `베스트 조크 선택` 노드를 구성합니다.

```python
# 모든 조크들 중 하나의 베스트 조크를 선택

def best_joke(state: OverallState):
    jokes = "\n\n".join(state["jokes"])
    prompt = best_joke_prompt.format(topic=state["topic"], jokes=jokes)
    response = model.with_structured_output(BestJoke).invoke(prompt)
    return {"best_selected_joke": state["jokes"][response.id]}
```

### 3.  4.  그래프 실행

마지막으로 그래프 구성하고 실행합니다.

```python
# 그래프 구성 및 실행

from IPython.display import Image
from langgraph.graph import END, StateGraph, START

# 그래프 구성: 모든 구성 요소를 모아 그래프를 만듭니다
graph = StateGraph(OverallState)
graph.add_node("generate_topics", generate_topics)
graph.add_node("generate_joke", generate_joke)
graph.add_node("best_joke", best_joke)
graph.add_edge(START, "generate_topics")
graph.add_conditional_edges("generate_topics", continue_to_jokes, ["generate_joke"])
graph.add_edge("generate_joke", "best_joke")
graph.add_edge("best_joke", END)

# 그래프 컴파일
app = graph.compile()
Image(app.get_graph().draw_mermaid_png())
```

```python
# 그래프 실행: 주어진 주제에 대해 조크 목록을 생성
for s in app.stream({"topic": "animals"}):
    print(s)
```

```
# 출력

{'generate_topics': {'subjects': ['Animal Behavior and Communication', 'Conservation and Wildlife Protection', 'Domestication and Human-Animal Relationships']}}
{'generate_joke': {'jokes': ['Why did the parrot bring a ladder to the comedy club?\n\nBecause it wanted to reach the "punchline" in its jokes!']}}
{'generate_joke': {'jokes': ['Why did the squirrel bring a suitcase to the wildlife conservation meeting?\n\nBecause it heard they were going to "pack" the forest with more trees! 🌳🐿️']}}
{'generate_joke': {'jokes': ["Why did the dog sit in the shade?\n\nBecause it didn't want to be a hot dog, and it knew its human would fetch it a cool drink anyway!"]}}
{'best_joke': {'best_selected_joke': 'Why did the parrot bring a ladder to the comedy club?\n\nBecause it wanted to reach the "punchline" in its jokes!'}}
```

## 정리

이번 포스팅에서는 LangGraph의 제어 가능성(Controllability)을 중심으로 주요 개념을 실습과 함께 살펴보았습니다.

먼저, `병렬 노드 실행(Fan-out)`을 통해 여러 작업을 동시에 수행하고, 완료 후 `팬인(Fan-in)` 구조로 결과를 집계하는 방식을 확인했습니다. 이 과정에서 `sorting_reducer`를 사용하면 병렬 결과의 순서를 보장할 수 있습니다.

실습에서는 `search_web`과 `search_wikipedia` 노드를 병렬로 실행한 뒤, `generate_answer` 노드에서 이들을 통합해 응답을 생성했습니다.

또한, 멀티에이전트 시스템에 적합한 구조로서 `서브 그래프`를 부모 그래프와 별도로 구성하고, 실행 결과를 부모 그래프로 반환하는 방식을 다뤘습니다. 이는 분석이나 요약 등 역할 분담이 필요한 구조에서 유용합니다.

마지막으로, `맵리듀스` 패턴을 활용해 '조크 생성' 예제를 구현했습니다. 이 예제에서는 주제를 여러 개로 분할(map)하고 병렬 처리한 후, 가장 우수한 결과를 선택하는 방식으로 집계(reduce)하였습니다.

다음 포스팅에서는 다뤘던 내용 기반으로 `멀티 에이전트 리서치 어시스턴트`를 구축할 것입니다.

## References

* [LangGraph 공식 문서](https://langchain-ai.github.io/langgraph){: target="_blank"}
* [LangChain Academy](https://github.com/langchain-ai/langchain-academy){: target="_blank"}
