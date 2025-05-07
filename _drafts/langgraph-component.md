---
title: LangGraph 핵심 구성과 그래프 생성
date: 2025-05-07 11:15:43 +/-TTTT
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

> 학습할 리소스는 [LangChain Academy Github](https://github.com/langchain-ai/langchain-academy){: target="_blank"}를 사용합니다.
{: .prompt-info }

## 1.   LangGraph 핵심 구성과 간단한 그래프 생성
그래프를 생성할 때 `노드`, `상태`, `엣지` 핵심 요소들을 먼저 정의하고, 이를 `체인`, `라우터`, `에이전트` 등 상위 워크플로 요소와 결합하여 애플리케이션을 구축할 수 있습니다.
![LangGraph 핵심 구성 요소와 워크플로·에이전트 요소](assets/drafts/2025-05-07-langgraph-component/20250429_121010_pic_M01_04_00.png)
_LangGraph 핵심 구성 요소와 워크플로·에이전트 요소_

설명할 그래프는 `노드`, `엣지`, `상태`를 기반으로 하기의 그림처럼 구성됩니다.
![단순 그래프 생성](assets/drafts/2025-05-07-langgraph-component/20250429_121010_pic_M01_04_simplegraph.png)
_단순 그래프 생성_

먼저, langgraph를 설치합니다.

```python
%%capture --no-stderr
%pip install --quiet -U langgraph
```

### 1.  1.  상태(State)

그래프를 정의할 때 가장 먼저 그래프의 [상태](https://langchain-ai.github.io/langgraph/concepts/low_level/#state){: target="_blank"}를 정의합니다.

상태 스키마는 그래프의 모든 노드와 에지(Edge)에 대한 `입력 스키마` 역할을 합니다. 노드는 상태를 기반으로 작업을 수행하고 상태를 업데이트하고 다음 노드로 제어 흐름을 이동합니다.

그래프의 스키마는 `TypedDict`를 사용합니다. 입출력 스키마는 동일한 스키마를 사용하지만, 명시적으로 입력 및 출력 스키마를 직접 변경할 수 있습니다.

```python
# TypedDict 입력 스키마 예제
from typing_extensions import TypedDict

class State(TypedDict):
    graph_state: str
```

`리듀서(Reducer)`는 새로운 메시지를 기존 리스트에 추가할 수 있게 설정할 수 있습니다. 상태의 각 키는 독립적인 리듀서 함수를 갖고 있습니다.

### 1.  2.  노드(Node)

노드는 단지 Python의 함수로 볼 수 있습니다.

다음 예제에서 첫 번째 인수는 상태이고 두 번째 인수부터는 선택적으로 추가할 수 있는 매개변수를 넣을 수 있습니다.

상태는 `state['상태명']`을 사용하여 키에 접근할 수 있습니다. 노드에서 작업이 수행되면 상태는 업데이트됩니다. 즉 이전 상태 값을 `재정의`합니다.

```python
def node_1(state):
    print("---Node 1---")
    return {"graph_state": state['graph_state'] +" I am"}

def node_2(state):
    print("---Node 2---")
    return {"graph_state": state['graph_state'] +" happy!"}

def node_3(state):
    print("---Node 3---")
    return {"graph_state": state['graph_state'] +" sad!"}
```

### 1.  3.  엣지(Edge)

엣지는 두 가지 유형으로 노드를 연결합니다. `일반 엣지(Normal Edge)`는 노드와 다음 노드로 이동하는 경우에 사용이 되고 `조건부 엣지(Conditional Edge)`는 선택적으로 노드를 이동할 수 있습니다.

일반 엣지에서 노드를 이동하려면 `add_edges`를 조건부 엣지는 `add_conditional_edges` 메소드를 사용합니다.

### 1.  4.  그래프 생성

앞서 설명한 구성 요소를 사용해서 [StateGraph 클래스](https://langchain-ai.github.io/langgraph/concepts/low_level/#stategraph){: target="_blank"}를 사용해서 그래프를 생성합니다.

```python
# 노드 1에서 노드 2와 노드 3에 대한 이동을 조건부 엣지(decide_mood)를 사용해서 이동하는 예제

from IPython.display import Image, display
from langgraph.graph import StateGraph, START, END

# 그래프 생성
builder = StateGraph(State)
builder.add_node("node_1", node_1)
builder.add_node("node_2", node_2)
builder.add_node("node_3", node_3)

# 엣지 설정
builder.add_edge(START, "node_1")
builder.add_conditional_edges("node_1", decide_mood) # 조건부 에지
builder.add_edge("node_2", END)
builder.add_edge("node_3", END)

# 컴파일
graph = builder.compile()

# 그래프 이미지 생성
display(Image(graph.get_graph().draw_mermaid_png()))
```

```python
# 조건부 엣지에서 사용할 decide_mood 함수 정의
import random
from typing import Literal

def decide_mood(state) -> Literal["node_2", "node_3"]:
  
    user_input = state['graph_state'] 
  
    # 노드 2, 3을 50:50 확률로 이동시키는 로직
    if random.random() < 0.5:

        # 50%의 확률로 노드 2로 이동
        return "node_2"
  
    # 50% 확률로 노드 3로 이동
    return "node_3"
```

### 1.  5.    그래프 호출

컴파일된 그래프는 [Runnable](https://python.langchain.com/docs/concepts/runnables/?utm_source=chatgpt.com){: target="_blank"} 프로토콜을 구현합니다. `invoke`가 호출되면 그래프는 `START` 노드에서 실행을 시작하고 정의된 노드 순서대로 진행합니다. 그리고 조건부 에지에서 이동 규칙에 따라 해당하는 노드로 이동합니다. 각 노드는 현재 상태를 입력받고 그래프 상태를 재정의하여 반환합니다. 마지막으로 `END` 노드에서 실행이 종료됩니다.

`invoke`는 그래프를 `동기적`으로 실행합니다. 실행하는 단계가 완료되어야 다음 단계로 넘어가게 됩니다.

```python
graph.invoke({"graph_state" : "Hi, this is Lance."})
```

```python
# 출력
{'graph_state': 'Hi, this is Lance. I am sad!'}j
```

> Runnable은 LLM과 같은 `구성 요소(Component)`가 입력을 받아 무언가를 실행하고 결과를 반환하는 일종의 표준 인터페이스입니다. 실행에는 invoke, batch, stream과 같은 기능을 수행할 수 있습니다.
{: .prompt-info }

## 2.   체인(Chain)

LangChain에서 [체인](https://api.python.langchain.com/en/latest/langchain/chains.html){: target="_blank"}은 하나 이상의 컴포넌트(LLM, Tool, 데이터 전처리 단계 등)를 고정된 순서로 연결한 워크플로입니다.

체인 인터페이스를 사용하면 메모리 사용해 처리 결과를 참조할 수 있는 `상태 저장(Stateful)`, 로깅과 같은 기능을 수행하여 콜백(Callback)을 통한 `관찰 가능(Observable)` 및 다른 체인 및 컴포넌트와 결합으로 `구성 가능(Composable)`한 애플리케이션을 쉽게 만들 수 있습니다.

체인에서 핵심 개념인 `채팅 메시지(Messages)`, `채팅 모델(Chat Model)`, `도구 바인딩(Binding Tool)` 및 LangGraph 내에서 `도구 호출(Tool Calling)`에 대한 개념은 다음과 같습니다.

langgraph와 openai를 패키지를 설치합니다.

```python
%%capture --no-stderr
%pip install --quiet -U langchain_openai langchain_core langgraph
```

### 2.  1.    메시지(Messages)

채팅 모델에서 다양한 유형의 메시지를 사용할 수 있습니다.

LangChain은 `HumanMessage`, `AIMessage`, `SystemMessage`, `ToolMessage`를 포함한 다양한 메시지 유형을 지원합니다.

메시지는 다음과 같은 정보를 제공합니다.

* `content` : 메시지 내용
* `name` : 메시지 작성자 (선택)
* `response_metadata`: 메타데이터 사전 (`AIMessage`는 모델 공급자가 사용하는 경우가 많음)

```python
# AIMessage, HumanMessage 사용 예
from pprint import pprint
from langchain_core.messages import AIMessage, HumanMessage

messages = [AIMessage(content=f"So you said you were researching ocean mammals?", name="Model")]
messages.append(HumanMessage(content=f"Yes, that's right.",name="Lance"))
messages.append(AIMessage(content=f"Great, what would you like to learn about.", name="Model"))
messages.append(HumanMessage(content=f"I want to learn about the best place to see Orcas in the US.", name="Lance"))

for m in messages:
    m.pretty_print()
```

### 2.  2.    채팅 모델(Chat Model)

채팅 모델은 앞서 설명한 다양한 유형의 메시지와 [언어 모델](https://python.langchain.com/v0.2/docs/concepts/#chat-models){: target="_blank"}을 선택할 수 있습니다.

예제는 OpenAI를 사용합니다. `AIMessage`와 `response_metadata`를 확인할 수 있습니다.

```python
# Visual Studio Code 상단에 OPENAI_API_KEY 입력 프롬프트가 나타납니다.
import os, getpass

def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

_set_env("OPENAI_API_KEY")
```

```python
# OpenAI API 사용 및 호출
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o")
result = llm.invoke(messages)
type(result)
```

```python
# langchain_core.messages.ai.AIMessage 출력
result

# AIMessage(content='One of the best places to see orcas in the United States is the Pacific Northwest, particularly around the San Juan Islands in Washington State. Here are some details:\n\n1. **San Juan Islands, Washington**: These islands are a renowned spot for whale watching, with orcas frequently spotted between late spring and early fall. The waters around the San Juan Islands are home to both resident and transient orca pods, making it an excellent location for sightings.\n\n2. **Puget Sound, Washington**: This area, including places like Seattle and the surrounding waters, offers additional opportunities to see orcas, particularly the Southern Resident killer whale population.\n\n3. **Olympic National Park, Washington**: The coastal areas of the park provide a stunning backdrop for spotting orcas, especially during their migration periods.\n\nWhen planning a trip for whale watching, consider peak seasons for orca activity and book tours with reputable operators who adhere to responsible wildlife viewing practices. Additionally, land-based spots like Lime Kiln Point State Park, also known as “Whale Watch Park,” on San Juan Island, offer great opportunities for orca watching from shore.', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 228, 'prompt_tokens': 67, 'total_tokens': 295, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-2024-08-06', 'system_fingerprint': 'fp_50cad350e4', 'finish_reason': 'stop', 'logprobs': None}, id='run-57ed2891-c426-4452-b44b-15d0a5c3f225-0', usage_metadata={'input_tokens': 67, 'output_tokens': 228, 'total_tokens': 295, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}})
```

```python
# 출력
{'token_usage': {'completion_tokens': 228,
  'prompt_tokens': 67,
  'total_tokens': 295,
  'completion_tokens_details': {'accepted_prediction_tokens': 0,
   'audio_tokens': 0,
   'reasoning_tokens': 0,
   'rejected_prediction_tokens': 0},
  'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}},
 'model_name': 'gpt-4o-2024-08-06',
 'system_fingerprint': 'fp_50cad350e4',
 'finish_reason': 'stop',
 'logprobs': None}
```

### 2.  3.    Google Gemini 2.0 Flash 설정 (선택)

작성 일자 기준 무료로 제공하는 [Google Gemini 2.0 Flash](https://gemini.google.com){: target="_blank"} 설정은 다음과 같습니다.

```python
%%capture --no-stderr
%pip install --quiet -U langchain-google-genai langchain_core langgraph
```

```python
import getpass, os

def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

_set_env("GOOGLE_API_KEY")
```

```python
from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.9, transport="rest")
result = llm.invoke(messages)
type(result)
```

### 2.  4.    도구(Tool)

도구를 사용해 외부 시스템(API 등)과 채팅 모델을 연결할 수 있습니다.

외부 시스템은 자연어가 아닌 규정된 입력 스키마나 페이로드가 필요한 경우가 많습니다.

모델은 자연어 입력을 통해 도구를 호출합니다. API를 도구로 바인딩할 때, 모델에 필요한 입력 스키마로 인식하고 도구 스키마를 준수하는 페이로드(Payload) 반환합니다.

![도구 호출](assets/drafts/2025-05-07-langgraph-component/20250429_121010_pic_M01_04_01.png)
_도구(Tool Calling)를 통해 외부 시스템 연결_

다음은 도구 호출(Tool Calling)의 예입니다.

```python
# multiply 함수가 도구로써 사용
def multiply(a: int, b: int) -> int:
    """Multiply a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b

llm_with_tools = llm.bind_tools([multiply])
```

```python
# 입력 '2 곱하기 3은?' 전달하면 도구 호출을 통해 페이로드가 반환합니다
tool_call = llm_with_tools.invoke([HumanMessage(content=f"What is 2 multiplied by 3", name="Lance")])

tool_call.tool_calls
```

```python
# 출력
# 페이로드에는 호출할 함수의 이름과 입력 스키마와 일치하는 인수가 존재합니다
[{'name': 'multiply',
  'args': {'a': 2, 'b': 3},
  'id': 'call_gMKORxtwRSaLH1gUREPIfuXp',
  'type': 'tool_call'}]
```

### 2.  5.    메시지(Message)를 상태로 사용하기

그래프 상태(graph state)에서 메시지(messages)를 사용할 수 있습니다.

예제에서 `messages` 단일 키로 `TypeDict` 타입으로 `MessageState` 정의합니다.

`messages`는 리스트로 `HumanMessage` 등과 같은 메시지를 사용할 수 있습니다.

```python
# MessagesState를 단일 키 messages를 가진 TypedDict로 정의
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage

class MessagesState(TypedDict):
    messages: list[AnyMessage]
```

### 2.  6.    리듀서(Reducers)

노드는 상태 키인 messages의 새 값을 반환하는데 여기서 문제점이 messages 값을 [재정의](https://langchain-ai.github.io/langgraph/concepts/low_level/#reducers){: target="_blank"}합니다. 즉 값이 덮어집니다.

`Annotated`에서 `add_messages` 리듀서 함수를 메타데이터로 사용하면 기존 messages 상태 키에 메시지가 추가가 됩니다.

```python
# 사전 내장된 add_messages 리듀서를 사용해 메시지 추가하기
from typing import Annotated
from langgraph.graph.message import add_messages

class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
```

그리고 일반적으로 메시지는 목록을 가지고 있어서 LangGraph는 사전 내장된 [MessageState](https://langchain-ai.github.io/langgraph/concepts/low_level/#messagesstate){: target="_blank"}를 사용합니다.

`MessageState`는 사전 내장된 단일키인 `messsages키`, `AnyMessage 오브젝트 목록`, `add_messages 리듀서`로 정의됩니다.

```python
# MessageState 구현
from langgraph.graph import MessagesState

class MessagesState(MessagesState):
    # 여기에 사전 내장된 키 외에 필요한 키를 추가합니다.
    # 예:
    # user_role: str  # 'admin' 또는 'guest' 등 사용자 역할
    pass
```

다음은 `add_messages 리듀서` 사용한 예입니다. 응답에서 `AIMessage` 두 개가 나오는 것을 확인할 수 있습니다.

> 기존 `AIMessage`를 업데이트하는 것이 아닙니다.
{: .prompt-warning }

```python
# 상태 초기화
initial_messages = [AIMessage(content="Hello! How can I assist you?", name="Model"),
                    HumanMessage(content="I'm looking for information on marine biology.", name="Lance")
                   ]

# 기존 메시지에 추가
new_message = AIMessage(content="Sure, I can help with that. What specifically are you interested in?", name="Model")

# 실행
add_messages(initial_messages , new_message)

# 응답
# [AIMessage(content='Hello! How can I assist you?', additional_kwargs={}, response_metadata={}, name='Model', id='4e8da509-8a5f-46c5-bb43-62bc28eed03f'), HumanMessage(content="I'm looking for information on marine biology.", additional_kwargs={}, response_metadata={}, name='Lance', id='217e3314-2295-4d70-a415-beff5d52d9ad'), AIMessage(content='Sure, I can help with that. What specifically are you interested in?', additional_kwargs={}, response_metadata={}, name='Model', id='813435cd-d631-434e-8179-049a4e00b0c5')]
```

다음은 `MessageState`를 사용한 그래프 워크플로를 보여줍니다.

```python

from IPython.display import Image, display
from langgraph.graph import StateGraph, START, END
  
# Node
def tool_calling_llm(state: MessagesState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# Build graph
builder = StateGraph(MessagesState)
builder.add_node("tool_calling_llm", tool_calling_llm)
builder.add_edge(START, "tool_calling_llm")
builder.add_edge("tool_calling_llm", END)
graph = builder.compile()

# View
display(Image(graph.get_graph().draw_mermaid_png()))
```

`LLM`은 입력 또는 작업에 도구 제공이 필요한지 판단합니다. 아래 예는 `Hello!`를 입력하면 도구를 사용하지 않고 출력합니다.

```python
messages = graph.invoke({"messages": HumanMessage(content="Hello!")})
for m in messages['messages']:
    m.pretty_print()

# 출력
# Hi there! How can I assist you today?
```

다음 예는 `LLM`이 도구 사용을 결정하면 출력에서 `Tool Calls` 로그를 확인할 수 있습니다.

```python
messages = graph.invoke({"messages": HumanMessage(content="Multiply 2 and 3")})
for m in messages['messages']:
    m.pretty_print()

# 출력
================================ Human Message =================================

 Multiply 2 and 3
================================== Ai Message ==================================
 Tool Calls:
  multiply (call_KbkBbn3wYjiT6h2Ab9ZkjoNK)
  Call ID: call_KbkBbn3wYjiT6h2Ab9ZkjoNK
  Args:
    a: 2
    b: 3
```

## 3.   라우터(Router)

채팅 모델은 사용자의 입력에 따라 `직접 응답`을 하거나 `도구 호출(Tool Calling)`을 할 수 있게 라우팅합니다.

이전에 `에이전트(Agent)`는 LLM이 자체적 판단하여 `제어 흐름(Control Flow)`을 선택할 수 있다고 설명해 드렸습니다.

다음 예제는 `도구 호출`하는 `노드`를 추가하고 `LLM`이 자체 판단하여 `도구 호출`을 하거나 바로 `종료(END)`하는 `조건부 엣지(Conditional Edge)`를 추가합니다.

```python
%%capture --no-stderr
%pip install --quiet -U langchain_openai langchain_core langgraph langgraph-prebuilt
```

```python
import os, getpass

def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

_set_env("OPENAI_API_KEY")
```

```python
# 도구 호출을 위한 함수 바인딩
from langchain_openai import ChatOpenAI

def multiply(a: int, b: int) -> int:
    """Multiply a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b

llm = ChatOpenAI(model="gpt-4o")
llm_with_tools = llm.bind_tools([multiply])
```

내장된 [ToolNode](https://langchain-ai.github.io/langgraph/reference/agents/#langgraph.prebuilt.tool_node.ToolNode){: target="_blank"}를 사용해 도구 목록을 받고, [tools_condition](https://langchain-ai.github.io/langgraph/reference/agents/#langgraph.prebuilt.tool_node.tools_condition){: target="_blank"}을 조건부 엣지에 사용합니다.

다음 예에서 조건부 엣지의 tool_condtion을 통해 `도구 호출`을 할지 `END`로 갈지 결정합니다.

```python

from IPython.display import Image, display
from langgraph.graph import StateGraph, START, END
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition

# 노드 정의
def tool_calling_llm(state: MessagesState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# 그래프 생성
builder = StateGraph(MessagesState)
builder.add_node("tool_calling_llm", tool_calling_llm)
builder.add_node("tools", ToolNode([multiply]))
builder.add_edge(START, "tool_calling_llm")
builder.add_conditional_edges(
    "tool_calling_llm",
    # Assistant의 최신 메시지(결과)가 도구 호출이면 -> tools_condition이 tools로 라우팅
    # Assistant의 최신 메시지(결과)가 도구 호출이 아니면 -> tools_condition이 END로 라우팅
    tools_condition,
)
builder.add_edge("tools", END)
graph = builder.compile()

# 그래프 이미지 
display(Image(graph.get_graph().draw_mermaid_png()))
```

> `Assistant`는 메시지 모델의 `역할(Role)`입니다.
{: .prompt-info }

다음은 '2 곱하기 2는?' 질문을 `LLM`은 `도구 호출`로 결정하여 결과를 출력합니다.

```python
from langchain_core.messages import HumanMessage
messages = [HumanMessage(content="Hello, what is 2 multiplied by 2?")]
messages = graph.invoke({"messages": messages})
for m in messages['messages']:
    m.pretty_print()

# 출력
# ================================ Human Message =================================
# 
# Hello, what is 2 multiplied by 2?
# ================================== Ai Message ==================================
# Tool Calls:
#   multiply (call_7XJ4HVW9nSLob3KnaVPZVKrl)
#  Call ID: call_7XJ4HVW9nSLob3KnaVPZVKrl
#   Args:
#     a: 2
#     b: 2
# ================================= Tool Message =================================
# Name: multiply
#
# 4


```

## 4.   에이전트(Agent)

`LLM`이 도구 호출을 선택하면 사용자에게 `ToolMessage`를 반환합니다.

그런데 `ToolMessage`를 다시 모델에 전달하면 어떻게 될까요?

다른 도구를 호출하거나 직접 응답을 할 수 있습니다.

이것이 `ReACT(Reasoning and Acting)` 에이전트 아키텍처의 핵심입니다.

* `Act`: 모델이 어떤 도구를 실행하도록 지시합니다.
* `Oberserve`: 도구 실행 결과를 모델에 다시 전달합니다.
* `Reason`: 모델이 도구 결과를 바탕으로 다음 행동(다른 도구 호출 또는 사용자 응답)을 결정합니다.

![ReACT(Reasoning and Acting) 에이전트 아키텍처](assets/drafts/2025-05-07-langgraph-component/20250501_154114_pic_M01_06_01.png)
_ReACT(Reasoning and Acting) 에이전트 아키텍처_

```python
%%capture --no-stderr
%pip install --quiet -U langchain_openai langchain_core langgraph langgraph-prebuilt
```

```python
import os, getpass

def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

_set_env("OPENAI_API_KEY")
```

[추적(logging)](https://docs.smith.langchain.com/observability/concepts){: target="_blank"} 기능을 위해 [LangSmith](https://docs.smith.langchain.com){: target="_blank"}를 사용합니다.

하기 코드 실행 시 Visual Studio Code는 상단에 입력을 위한 프롬프트가 나타납니다.

```python
_set_env("LANGSMITH_API_KEY")
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "langchain-academy"
```

다음 예에서 `multiply`, `add`, `divide` 세 개의 도구를 위한 함수를 생성했습니다.

OpenAI 모델은 효율성을 위해 [병렬 도구 호출(parallel tool calling)](https://python.langchain.com/docs/how_to/tool_calling_parallel/){: target="_blank"}을 사용합니다.

여기서는 `parallel_tool_calls`를 `False` 했습니다.

```python
from langchain_openai import ChatOpenAI

def multiply(a: int, b: int) -> int:
    """Multiply a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b

def add(a: int, b: int) -> int:
    """Adds a and b.

    Args:
        a: first int
        b: second int
    """
    return a + b

def divide(a: int, b: int) -> float:
    """Divide a and b.

    Args:
        a: first int
        b: second int
    """
    return a / b

tools = [add, multiply, divide]
llm = ChatOpenAI(model="gpt-4o")

llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=False)
```

`시스템 메시지(sys_msg)`에서 에이전트 동작을 구성을 합니다.

그러면 `assitant` 노드에서는 `시스템 메시지`와 상태에 `누적된 메시지`와 함께 LLM을 호출합니다.

그리고 반환된 메시지를 다시 상태에 돌려보냅니다.

```python
from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage, SystemMessage

# 시스템 메시지
sys_msg = SystemMessage(content="You are a helpful assistant tasked with performing arithmetic on a set of inputs.")

# 노드
def assistant(state: MessagesState):
   return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}
```

다음 예에서 그래프, 노드와 엣지 생성이 앞서 라우터 예제와 비슷합니다. 차이점은 에이전트 예에서는 `도구 호출`만이 아니라 assistant 노드에서 `ReACT` 패턴을 구현했습니다.

```python
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition
from langgraph.prebuilt import ToolNode
from IPython.display import Image, display

# 그래프 생성
builder = StateGraph(MessagesState)

# 노드 정의
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

# 엣지 정의
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
 END
    # Assistant의 최신 메시지(결과)가 도구 호출이면 -> tools_condition이 tools로 라우팅
    # Assistant의 최신 메시지(결과)가 도구 호출이 아니면 -> tools_condition이 END로 라우팅
    tools_condition,
)
builder.add_edge("tools", "assistant")
react_graph = builder.compile()

# 그래프 이미지
display(Image(react_graph.get_graph(xray=True).draw_mermaid_png()))
```

아래 자연어 질의를 통해 `(3 + 4)` * `2` / `5`에 대해서 `ReACT` 패턴으로 각 도구 호출 및 반환과 누적된 메시지의 최종 결과가 보입니다.

```python
messages = [HumanMessage(content="Add 3 and 4. Multiply the output by 2. Divide the output by 5")]
messages = react_graph.invoke({"messages": messages})

for m in messages['messages']:
    m.pretty_print()
```

```python
# 출력 
================================ Human Message =================================

Add 3 and 4. Multiply the output by 2. Divide the output by 5
================================== Ai Message ==================================
Tool Calls:
  add (call_i8zDfMTdvmIG34w4VBA3m93Z)
 Call ID: call_i8zDfMTdvmIG34w4VBA3m93Z
  Args:
    a: 3
    b: 4
================================= Tool Message =================================
Name: add

7
================================== Ai Message ==================================
Tool Calls:
  multiply (call_nE62D40lrGQC7b67nVOzqGYY)
 Call ID: call_nE62D40lrGQC7b67nVOzqGYY
  Args:
    a: 7
    b: 2
================================= Tool Message =================================
Name: multiply

14
...
2.8
================================== Ai Message ==================================

# The final result after performing the operations \( (3 + 4) \times 2 \div 5 \) is 2.8.
```

## 5.   에이전트 메모리(Agent Memory)

에이전트 예제를 다시 수행하고 하단에 다음 코드를 수행합니다.

`(3 + 4)` * `2` / `5`의 자연어 질의를 단계별로 나눠서 수행합니다.

처음에 3과 4를 더합니다.

```python
messages = [HumanMessage(content="Add 3 and 4.")]
messages = react_graph.invoke({"messages": messages})
for m in messages['messages']:
    m.pretty_print()
```

```python
# 출력
================================ Human Message =================================

Add 3 and 4.
================================== Ai Message ==================================
Tool Calls:
  add (call_gW2DkrKQPW5d3UdNgd4lQt56)
 Call ID: call_gW2DkrKQPW5d3UdNgd4lQt56
  Args:
    a: 3
    b: 4
================================= Tool Message =================================
Name: add

7
================================== Ai Message ==================================

The sum of 3 and 4 is 7.
```

다시 2로 나눕니다. 결과에서 보면 7의 값을 유지하지 않습니다. 상태 유지가 일시적이기 때문에 중단이 발생하거나 여러 단계 대화(multi-turn)를 진행할 수 있는 기능이 제한되는 상황에서는 문제가 발생합니다.

[퍼시스턴스(Persistance)](https://langchain-ai.github.io/langgraph/how-tos/persistence/){: target="_blank"}를 이용해서 이런 문제를 해결할 수 있습니다.

```python
messages = [HumanMessage(content="Multiply that by 2.")]
messages = react_graph.invoke({"messages": messages})
for m in messages['messages']:
    m.pretty_print()
```

```python
# 출력
================================ Human Message =================================

Multiply that by 2.
================================== Ai Message ==================================

It seems like I'm missing the initial value that you want me to multiply by 2. Could you please provide that value?
```

LangGraph는 체크포인터(checkpointer)를 사용하여 각 단계 실행 후에 그래프 상태를 자동으로 저장할 수 있습니다.

내장된 퍼시스턴트 계층은 메모리를 제공하기 때문에 LangGraph가 마지막 상태 업데이트의 데이터를 가져올 수 있습니다.

가장 쉽게 사용할 수 있는 체크포인트는 `인-메모리 키-값 저장소(in-memory key-value store)`인 `MemorySaver`입니다.

사용은 체크포인터로 그래프를 컴파일하기만 하면 그래프에 메모리가 생성됩니다.

```python
from langgraph.checkpoint.memory import MemorySaver
memory = MemorySaver()
react_graph_memory = builder.compile(checkpointer=memory)
```

메모리를 사용하려면 `thread_id`를 지정해야 합니다.

`thread_id`는 그래프 상태 모음(collection of graph state)을 저장합니다.

다음 그림을 통해서 그래프, 슈퍼스텝, 체크포인트, 스레드를 살펴보면,

`그래프`는 노드와 엣지 연결됩니다.

`슈퍼스텝`은 `한 번에 수행되는 작업 묶음` 단위입니다. 순차 노드는 개별 슈퍼스텝이고 병렬 노드는 같은 슈퍼스텝입니다.

`체크포인트`는 각 슈퍼스텝 종료 시점의 패키지 스냅샷으로 `현재 상태`, `다음 실행할 노드`, `메타데이터`(고유 ID, 타임스탬프 등)가 저장됩니다.

`스레드`는 체크포인트의 모음입니다. 즉 컴파일된 그래프 객체를 실제로 호출하여 발생하는 `모든 슈퍼스텝과 체크모인트`를 말합니다.

![그래프, 슈퍼스텝, 체크포인트, 스레드](assets/drafts/2025-05-07-langgraph-component/20250501_134128_pic_M01_07_01.png)
_그래프, 슈퍼스텝, 체크포인트, 스레드_

아래 그림에서는 순차 노드, 병렬 노드에서는 3개의 슈퍼스텝이 존재합니다.

![순차 노드와 병렬노드에서 슈퍼스텝](assets/drafts/2025-05-07-langgraph-component/20250501_135714_pic_M01_07_02.png)
_순차 노드와 병렬노드에서 슈퍼스텝_

`thread_id`를 전달하면 이전에 로깅된 상태 체크포인트부터 진행할 수 있습니다.

```python
# Specify a thread
config = {"configurable": {"thread_id": "1"}}

# Specify an input
messages = [HumanMessage(content="Add 3 and 4.")]

# Run
messages = react_graph_memory.invoke({"messages": messages},config)
for m in messages['messages']:
    m.pretty_print()
```

```python
# 출력
================================ Human Message =================================

Add 3 and 4.
================================== Ai Message ==================================
Tool Calls:
  add (call_2oYEgdji1s5ebAlzsuLhsJ11)
 Call ID: call_2oYEgdji1s5ebAlzsuLhsJ11
  Args:
    a: 3
    b: 4
================================= Tool Message =================================
Name: add

7
================================== Ai Message ==================================

The sum of 3 and 4 is 7.
```

스레드 안에 저장된 대화 이력(체크포인트)이 자동으로 불러옵니다.

기존 스레드에 저장된 `The sum of 3 and 4 is 7` 참조하여 `HumanMessage`의 `Multiply that by 2` 계산하여 14라는 결과를 출력합니다.

```python
messages = [HumanMessage(content="Multiply that by 2.")]
messages = react_graph_memory.invoke({"messages": messages}, config)
for m in messages['messages']:
    m.pretty_print()
```

```python
# 출력
================================ Human Message =================================

Add 3 and 4.
================================== Ai Message ==================================
Tool Calls:
  add (call_2oYEgdji1s5ebAlzsuLhsJ11)
 Call ID: call_2oYEgdji1s5ebAlzsuLhsJ11
  Args:
    a: 3
    b: 4
================================= Tool Message =================================
Name: add

7
================================== Ai Message ==================================

The sum of 3 and 4 is 7.
================================ Human Message =================================

Multiply that by 2.
================================== Ai Message ==================================
Tool Calls:
  multiply (call_NIMq1IkCB0rNzz0ZtGCBw94M)
 Call ID: call_NIMq1IkCB0rNzz0ZtGCBw94M
  Args:
...
14
================================== Ai Message ==================================

The result of multiplying 7 by 2 is 14.
```



## References

* [LangGraph 공식 문서](https://langchain-ai.github.io/langgraph){: target="_blank"}
* [LangChain Academy](https://github.com/langchain-ai/langchain-academy){: target="_blank"}
