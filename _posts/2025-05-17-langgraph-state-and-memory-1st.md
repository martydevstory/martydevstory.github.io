---
title: LangGraph 상태와 메모리 (1)
date: 2025-05-17 11:15:43 +/-TTTT
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
series_order: 3
---
LangGraph의 핵심 구성요소인 상태, 노드, 엣지를 이해했고, 워크플로와 에이전트 요소인 체인, 라우터, 에이전트, 에이전트 메모리에 대해서도 살펴보았습니다.

이번 포스팅에서 노드간의 통신을 위한 `상태 스키마`와 상태 업데이트의 수행 방식을 지정하는 `리듀서`에 대해서 살펴보겠습니다.

> 학습할 리소스는 [LangChain Academy Github](https://github.com/langchain-ai/langchain-academy){: target="_blank"}를 사용합니다.
{: .prompt-info }

## 1. 상태 스키마 (State Schema)

### 1.  1.    스키마 (Schema)

LangGraph는 주요 그래프 클래스인 [StateGraph](https://langchain-ai.github.io/langgraph/concepts/low_level/#stategraph){: target="_blank"}를 정의할 때 [상태 스키마](https://langchain-ai.github.io/langgraph/concepts/low_level/#state){: target="_blank"}를 사용합니다.

상태 스키마는 그래프에서 사용할 데이터 구조(Structure)와 타입을 나타냅니다.

모든 노드는 해당 상태 스키마와 통신을 합니다.

LangGraph는 다양한 Python [타입](https://docs.python.org/3/library/stdtypes.html#type-objects){: target="_blank"} 및 유효성 검사 방식을 지원하여 상태 스키마를 정의하는 데 유연성을 제공합니다.

![graph_state를 상태키로 상태 스키마 정의](assets/posts/2025-05-17-langgraph-state-and-memory-1st/20250508_state_01.png)
_graph_state를 상태키로 상태 스키마 정의_

### 1.  2.    TypedDict

Python에 [typing](https://docs.python.org/3/library/typing.html){: target="_blank"} 모듈에 있는 `TypedDict` 클래스를 사용할 수 있습니다.

이 클래스를 사용하면 키와 해당 값의 타입을 지정할 수 있습니다. 강제성은 없고 `타입 힌트`만 제공합니다.

[mypy](https://github.com/python/mypy){: target="_blank"}와 같은 정적 타입 검사를 통해 IDE에서 코드 실행 전에 잠재적 타입 에러를 잡아낼 수 있습니다.

그러나 런타임에서는 적용되지 않습니다.

```python
# TypedDict 클래스 정의
from typing_extensions import TypedDict

class TypedDictState(TypedDict):
    foo: str
    bar: str
```

더 구체적인 값의 제약 조건을 지정하려면 `Literal` 타입 힌트와 같은 것을 사용할 수 있습니다.

여기서 `mood`는 `happy` 또는 `sad` 중 하나만 가능합니다.

```python
# Literal 타입 힌트 사용
from typing import Literal

class TypedDictState(TypedDict):
    name: str
    mood: Literal["happy","sad"]
```

LangGraph에서 예로 정의된 상태 클래스 `TypedDictState`를 `StateGraph`에 전달하기만 하면 사용할 수 있습니다.

또한, 각 상태 키는 그래프의 `채널(Channel)`이라고 생각할 수 있습니다.

> `채널`의 의미는 그래프 내부에서 흐르는 각 `상태 키`의 역할(입·출력 등)과 경로를 드러내는 개념적 이름입니다.<br/>
> 상태 스키마 TypedDictState 정의하면 그 안에 여러 키(예:foo, bar)가 있고, 각각 하나의 채널이 됩니다.<br/>
> 각 채널은 그래프 시작부터 종료까지 노드 실행에 따라 업데이트 되며 독립적으로 데이터가 이동합니다.
{: .prompt-info }

각 노드에서 `지정된 키` 또는 `채널`의 값을 덮어씁니다.

```python
import random
from IPython.display import Image, display
from langgraph.graph import StateGraph, START, END

def node_1(state):
    print("---Node 1---")
    return {"name": state['name'] + " is ... "}

def node_2(state):
    print("---Node 2---")
    return {"mood": "happy"}

def node_3(state):
    print("---Node 3---")
    return {"mood": "sad"}

def decide_mood(state) -> Literal["node_2", "node_3"]:
  
    # 노드 2, 3을 50:50 확률로 이동시키는 로직
    if random.random() < 0.5:

        # 50%의 확률로 노드 2로 이동
        return "node_2"
  
    # 50% 확률로 노드 3로 이동
    return "node_3"

# 그래프 생성
builder = StateGraph(TypedDictState)
builder.add_node("node_1", node_1)
builder.add_node("node_2", node_2)
builder.add_node("node_3", node_3)

# 엣지 설정
builder.add_edge(START, "node_1")
builder.add_conditional_edges("node_1", decide_mood)
builder.add_edge("node_2", END)
builder.add_edge("node_3", END)

# 컴파일
graph = builder.compile()

# 그래프 이미지
display(Image(graph.get_graph().draw_mermaid_png()))
```

정의된 상태는 딕셔너리 타입(Dict)이므로, 그래프를 호출할 때 딕셔너리 형태로 초기 상태를 전달하여 `name` 키의 초기값을 설정합니다.

```python
graph.invoke({"name":"Lance"})

# 출력
# {'name': 'Lance is ... ', 'mood': 'happy'}
```

### 1.  3.    Dataclass

Python의 [dataclass](https://docs.python.org/3/library/dataclasses.html){: target="_blank"}는 [구조화된 데이터를 정의하는 또 다른 방법](https://www.datacamp.com/tutorial/python-data-classes){: target="_blank"}을 제공합니다.

`dataclass`는 주로 데이터 저장 용도의 클래스를 생성하기 위한 간결한 구문을 제공합니다.

```python
from dataclasses import dataclass

@dataclass
class DataclassState:
    name: str
    mood: Literal["happy","sad"]

```

`dataclass`의 키에 접근하려면 `node_1`에서 사용된 첨자를 수정합니다.

즉, 위의 `TypedDict`에 `state["name"]`를 사용하는 대신 `dataclass`에서는 `state.name`을 사용합니다.

LangGraph는 `dataclass` 등 다른 타입으로 구현되더라도 딕셔니리 타입으로 반환하고 상태 업데이트를 수행할 때 알아서 전체가 아닌 `해당 키 채널만 덮어씁니다`.

```python
def node_1(state):
    print("---Node 1---")
    return {"name": state.name + " is ... "}

# 그래프 생성
builder = StateGraph(DataclassState)
builder.add_node("node_1", node_1)
builder.add_node("node_2", node_2)
builder.add_node("node_3", node_3)

# 엣지 설정
builder.add_edge(START, "node_1")
builder.add_conditional_edges("node_1", decide_mood)
builder.add_edge("node_2", END)
builder.add_edge("node_3", END)

# 그래프 컴파일
graph = builder.compile()

# 그래프 이미지
display(Image(graph.get_graph().draw_mermaid_png()))
```

`dataclass`를 호출하여 상태의 각 키/채널의 초깃값을 설정합니다.

```python
# 그래프 호출
graph.invoke(DataclassState(name="Lance",mood="sad"))
```

```
# 출력

---Node 1---
---Node 2---

{'name': 'Lance is ... ', 'mood': 'happy'}
```

### 1.  4.    Pydantic

`TypedDict`와 `dataclasses`는 타입 힌트를 제공하지만, 런타임 시 타입 강제성은 없습니다.

그러면 잘못된 타입의 값을 할당할 수 있습니다. 아래 코드처럼 `mad`를 할당할 수도 있습니다.

```python
# 잘못된 값을 할당
dataclass_instance = DataclassState(name="Lance", mood="mad")
```

이 문제를 해결할 수 있는 것이 [Pydantic](https://docs.pydantic.dev/latest/api/base_model/){: target="_blank"}입니다.

`Pydantic`은 Python 타입 어노테이션을 사용하는 데이터 검증 및 설정 관리 라이브러리입니다.

검증 기능을 갖추고 있어 [LangGraph에서 상태 스키마를 정의](https://langchain-ai.github.io/langgraph/how-tos/state-model/){: target="_blank"}하는 데 특히 적합합니다.

그리고 런타임 시 지정된 타입 및 제약 조건을 준수하는지 검증할 수 있습니다.

```python
from pydantic import BaseModel, field_validator, ValidationError

class PydanticState(BaseModel):
    name: str
    mood: str # "happy" or "sad" 

    @field_validator('mood')
    @classmethod
    def validate_mood(cls, value):
        # mood는 "happy" 또는 "sad" 만 유지 가능
        if value not in ["happy", "sad"]:
            raise ValueError("Each mood must be either 'happy' or 'sad'")
        return value

try:
    state = PydanticState(name="John Doe", mood="mad")
except ValidationError as e:
    print("Validation Error:", e)

```

```
# 출력

Validation Error: 1 validation error for PydanticState
mood
  Value error, Each mood must be either 'happy' or 'sad' [type=value_error, input_value='mad', input_type=str]
    For further information visit https://errors.pydantic.dev/2.11/v/value_error
```

구현은 다음 `PydanticState`를 사용하여 쉽게 적용할 수 있습니다.

```python
# 그래프 생성
builder = StateGraph(PydanticState) # PydanticState 상태 검증
builder.add_node("node_1", node_1)
builder.add_node("node_2", node_2)
builder.add_node("node_3", node_3)

# 엣지 설정
builder.add_edge(START, "node_1")
builder.add_conditional_edges("node_1", decide_mood)
builder.add_edge("node_2", END)
builder.add_edge("node_3", END)

# 그래프 컴파일
graph = builder.compile()

# 그래프 이미지
display(Image(graph.get_graph().draw_mermaid_png()))
```

```python
# 그래프 호출
graph.invoke(PydanticState(name="Lance",mood="sad"))
```

```
# 출력

---Node 1---
---Node 3---

{'name': 'Lance is ... ', 'mood': 'sad'}
```

## 2. 상태 리듀서 (State Reducer)

상태 스키마의 특정 키/채널에 대한 상태 업데이트가 수행되는 방식을 지정하는 `리듀서(Reducer)`를 살펴보겠습니다.

기본적인 `상태 업데이트`와 `분기`가 수행하는 방법을 보겠습니다.

```python
# 환경 구성
%%capture --no-stderr
%pip install --quiet -U langchain_core langgraph
```

### 2.  1.    기본 상태 덮어쓰기 (Default overwriting state)

LangGraph의 기본적인 상태 업데이트를 위해 상태 스키마로 `TypedDict` 타입을 사용합니다.

```python
from typing_extensions import TypedDict
from IPython.display import Image, display
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    foo: int

def node_1(state):
    print("---Node 1---")
    return {"foo": state['foo'] + 1}

# 그래프 생성
builder = StateGraph(State)
builder.add_node("node_1", node_1)

# 엣지 설정
builder.add_edge(START, "node_1")
builder.add_edge("node_1", END)

# 그래프 컴파일
graph = builder.compile()

# 그래프 이미지
display(Image(graph.get_graph().draw_mermaid_png()))
```

LangGraph는 상태 업데이트에 대한 방법을 지정하지 않으면 `기본적으로 상태 값을 덮어씁니다`.

따라서 `node_1`에 있는 `foo` 값이 `return {"foo": state['foo'] + 1} `을 실행하면

`{'foo': 1}`을 입력으로 전달하고 그래프에서 반환되는 상태는 `{'foo': 2}`가 됩니다.

```python
# 그래프 호출
graph.invoke({"foo" : 1})
```

```
# 출력

---Node 1---

{'foo': 2} # foo 상태 값을 덮어씀
```

### 2.  2.    분기 (Branching)

노드가 분기되는 상황에 대해서도 살펴보겠습니다.

아래 코드에서 `노드 1`은 `노드 2`와 `노드 3`으로 분기됩니다.

`노드 2`와 `노드 3`은 `병렬`로 실행됩니다. 즉, 그래프의 `같은 단계`에서 실행됩니다.

두 노드 모두 같은 단계에서 상태를 덮어쓰려고 시도하면 그래프는 모호해집니다.

그러면 어떻게 상태를 유지해야 할까요.

```python
class State(TypedDict):
    foo: int

def node_1(state):
    print("---Node 1---")
    return {"foo": state['foo'] + 1}

def node_2(state):
    print("---Node 2---")
    return {"foo": state['foo'] + 1}

def node_3(state):
    print("---Node 3---")
    return {"foo": state['foo'] + 1}

# 그래프 생성
builder = StateGraph(State)
builder.add_node("node_1", node_1)
builder.add_node("node_2", node_2)
builder.add_node("node_3", node_3)

# 엣지 설정
builder.add_edge(START, "node_1")
builder.add_edge("node_1", "node_2")
builder.add_edge("node_1", "node_3")
builder.add_edge("node_2", END)
builder.add_edge("node_3", END)

# 그래프 컴파일
graph = builder.compile()

# 그래프 이미지
display(Image(graph.get_graph().draw_mermaid_png()))
```

```python
# 그래프 호출
from langgraph.errors import InvalidUpdateError
try:
    graph.invoke({"foo" : 1})
except InvalidUpdateError as e:
    print(f"InvalidUpdateError occurred: {e}")

```

```
# 출력

---Node 1---
---Node 2---
---Node 3---

# 노드 병렬 수행으로 인한 에러 발생
InvalidUpdateError occurred: At key 'foo': Can receive only one value per step. Use an Annotated key to handle multiple values.
For troubleshooting, visit: https://python.langchain.com/docs/troubleshooting/errors/INVALID_CONCURRENT_GRAPH_UPDATE
```

### 2.  3.    리듀서 (Reducers)

[리듀서](https://langchain-ai.github.io/langgraph/concepts/low_level/#reducers){: target="_blank"}는 위와 같은 문제를 해결하는 일반적인 방법을 제공합니다.

리듀서는 업데이트를 수행하는 방법을 지정합니다.

`Annotated` 타입을 사용하여 리듀서 함수를 지정할 수 있습니다.

각 노드에서 반환된 값을 덮어쓰는 대신 `추가`합니다.

이를 수행할 수 있는 리듀서가 `operator.add`로 Python 내장 연산자 모듈의 함수입니다.

`operator.add`를 리스트에 적용 시 새 요소를 리스트에 이어 붙입니다.

> `Annotated`는 Python의 `typing` 모듈에 있는 제네릭 타입으로, 기존 타입에 `메타데이터`를 붙여주는 역할을 합니다.
{: .prompt-info }

```python
from operator import add
from typing import Annotated

class State(TypedDict):
    foo: Annotated[list[int], add]

def node_1(state):
    print("---Node 1---")
    return {"foo": [state['foo'][0] + 1]}

# 그래프 생성
builder = StateGraph(State)
builder.add_node("node_1", node_1)

# 엣지 설정
builder.add_edge(START, "node_1")
builder.add_edge("node_1", END)

# 그래프 컴파일
graph = builder.compile()

# 그래프 이미지
display(Image(graph.get_graph().draw_mermaid_png()))
```

```python
# 그래프 호출
graph.invoke({"foo" : [1]})
```

```
# 출력

---Node 1---

{'foo': [1, 2]}
```

이제 상태 키 `foo`는 리스트입니다.

`operator.add` 리듀서 함수는 각 노드가 반환하는 값을 기존 리스트에 추가합니다.

```python
def node_1(state):
    print("---Node 1---")
    return {"foo": [state['foo'][-1] + 1]}

def node_2(state):
    print("---Node 2---")
    return {"foo": [state['foo'][-1] + 1]}

def node_3(state):
    print("---Node 3---")
    return {"foo": [state['foo'][-1] + 1]}

# 그래프 생성
builder = StateGraph(State)
builder.add_node("node_1", node_1)
builder.add_node("node_2", node_2)
builder.add_node("node_3", node_3)

# 엣지 설정
builder.add_edge(START, "node_1")
builder.add_edge("node_1", "node_2") # 노드 2와 노드 3이 같은 단계
builder.add_edge("node_1", "node_3") # 노드 2와 노드 3이 같은 단계
builder.add_edge("node_2", END)
builder.add_edge("node_3", END)

# 그래프 컴파일
graph = builder.compile()

# 그래프 이미지
display(Image(graph.get_graph().draw_mermaid_png()))
```

`노드 2`와 `노드 3`의 업데이트는 같은 단계에 있기 때문에 동시에 수행되는 것을 확인할 수 있습니다.

```python
# 그래프 호출
graph.invoke({"foo" : [1]})
```

```
# 출력

---Node 1---
---Node 2---
---Node 3---

{'foo': [1, 2, 3, 3]}
```

추가로 `None`을 `foo`에 전달하면 어떻게 되는지 살펴보겠습니다.

리듀서인 `operator.add`는 `node_1`에 입력으로 전달된 `None`이 `리스트 타입`이 아니기 때문에 오류를 발생합니다.

```python
# 그래프 호출
try:
    graph.invoke({"foo" : None})
except TypeError as e:
    print(f"TypeError occurred: {e}")
```

```
# 출력

TypeError occurred: can only concatenate list (not "NoneType") to list
```

> operator.add는 둘 다 리스트 타입이어야 에러가 발생하지 않습니다.<br/>
> operator.add([1,2], [3,4])    ➞ [1,2,3,4]<br/>
> operator.add([1,2], None)    ➞ TypeError<br/>
{: .prompt-info }

### 2.  4.    커스텀 리듀서 (Custom Reducers)

위의 경우를 해결하기 위해 [커스텀 리듀서](https://langchain-ai.github.io/langgraph/how-tos/subgraph/#custom-reducer-functions-to-manage-state){: target="_blank"}를 정의할 수도 있습니다.

예를 들어, 리스트를 결합하고 입력 중 하나 또는 둘 다 `None`일 수 있는 경우를 처리하는 커스텀 리듀서 로직을 정의해 보겠습니다.

```python
def reduce_list(left: list | None, right: list | None) -> list:
    """두 개의 리스트를 안전하게 결합하고, 두 개의 입력 중 하나 또는 둘 다 None일 수 있는 경우를 처리합니다.

    Args:
        left (list | None): 결합할 첫 번째 리스트 또는 None.
        right (list | None): 결합할 두 번째 리스트 또는 None.

    Returns:
        list: 두 입력 리스트의 모든 요소를 포함하는 새 리스트.
              입력이 None이면 빈 리스트로 처리됩니다.
    """
    if not left:
        left = []
    if not right:
        right = []
    return left + right

class DefaultState(TypedDict):
    foo: Annotated[list[int], add]

class CustomReducerState(TypedDict):
    foo: Annotated[list[int], reduce_list]
```

그리고 `node_1`에서 숫자 2가 들어간 배열을 추가합니다. 그리고 초깃값을 `"foo" : None` 할당합니다.

```python
def node_1(state):
    print("---Node 1---")
    return {"foo": [2]}

# 그래프 생성
builder = StateGraph(DefaultState)
builder.add_node("node_1", node_1)

# 엣지 설정
builder.add_edge(START, "node_1")
builder.add_edge("node_1", END)

# 그래프 컴파일
graph = builder.compile()

# 그래프 이미지
display(Image(graph.get_graph().draw_mermaid_png()))

try:
    print(graph.invoke({"foo" : None}))
except TypeError as e:
    print(f"TypeError occurred: {e}")
```

출력에서 오류가 발생합니다.

```
# 출력

TypeError occurred: can only concatenate list (not "NoneType") to list
```

그러나 `커스텀 리듀서` 사용하면 출력에서 오류가 발생하지 않습니다.

```python
# 그래프 생성
builder = StateGraph(CustomReducerState)
builder.add_node("node_1", node_1)

# 엣지 설정
builder.add_edge(START, "node_1")
builder.add_edge("node_1", END)

# 그래프 컴파일
graph = builder.compile()

# 그래프 이미지
display(Image(graph.get_graph().draw_mermaid_png()))

try:
    print(graph.invoke({"foo" : None}))
except TypeError as e:
    print(f"TypeError occurred: {e}")
```

```
# 출력

---Node 1---
{'foo': [2]}
```

### 2.  5.    메시지 (Messages)

`MessagesState`를 사용하면 사전 정의된 `messages`키와 `add_messages` 리듀서를 해당 키에 자동으로 연결하여 [메시지 작업에 유용](https://langchain-ai.github.io/langgraph/concepts/low_level/#messagesstate){: target="_blank"}한 것을 이전 포스팅에서 확인했습니다.

아래 코드는 `TypedDict`와 `MessagesState`에서도 커스텀과 확장이 가능한 것을 보여줍니다.

`CustomMessageState`는 `messages` 필드에 `add_messages` 리듀서를 지정하고, 여기에 추가 키(`added_key_1`, `added_key_2`)를 덧붙인 사용자 정의 TypedDict를 선언합니다.

그리고 `ExtendedMessagesState`는 이미 준비된 `from langgraph.graph import MessagesState`를 사용하여 중복 코드(Boilerplate)를 제거할 수 있어서 상태 정의가 간결해집니다.

```python
from typing import Annotated
from langgraph.graph import MessagesState
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

# add_messages 리듀서를 사용하여 메시지 목록을 포함하는 사용자 지정 TypedDict를 정의합니다
class CustomMessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    added_key_1: str
    added_key_2: str
    # etc

# add_messages 리듀서를 사용하여 메시지 키를 포함하는 MessagesState를 사용합니다
class ExtendedMessagesState(MessagesState):
    # 내장된 메시지 외에 필요한 키를 추가합니다.
    added_key_1: str
    added_key_2: str
    # etc
```

다음은 `add_messages` 리듀서 `기본 동작 상태`를 확인하기 위해 `상태 초기화`와 `새 메시지 추가`를 합니다.

```python
from langgraph.graph.message import add_messages
from langchain_core.messages import AIMessage, HumanMessage

# 상태 초기화
initial_messages = [AIMessage(content="Hello! How can I assist you?", name="Model"),
                    HumanMessage(content="I'm looking for information on marine biology.", name="Lance")
                   ]

# 새 메시지 추가
new_message = AIMessage(content="Sure, I can help with that. What specifically are you interested in?", name="Model")

# 테스트
add_messages(initial_messages , new_message)
```

`add_messages`를 사용하면 상태에서 `messages` 키에 `메시지를 추가`할 수 있다는 것을 알 수 있습니다.

```
# 출력

[AIMessage(content='Hello! How can I assist you?', additional_kwargs={}, response_metadata={}, name='Model', id='addc6f34-e0be-4ff3-9d6f-117a09008182'),
 HumanMessage(content="I'm looking for information on marine biology.", additional_kwargs={}, response_metadata={}, name='Lance', id='806bfe3a-e690-40f7-93a8-fc4c19908fb4'),
 AIMessage(content='Sure, I can help with that. What specifically are you interested in?', additional_kwargs={}, response_metadata={}, name='Model', id='268b0cb6-4660-40ac-83ea-465f4cfc7d31')]
```

### 2.  6.    메시지 재-작성(Re-writing)

이번에는 `add_messages` 리듀서를 다른 방법의 동작으로 살펴보겠습니다.

`messages` 목록에 있는 기존 메시지와 `동일한 ID(id="1")`를 가진 메시지를 전달하면 `기존 메시지를 덮어씁니다`.

```python
# 상태 초기화 - id 추가
initial_messages = [AIMessage(content="Hello! How can I assist you?", name="Model", id="1"),
                    HumanMessage(content="I'm looking for information on marine biology.", name="Lance", id="2")
                   ]

# 새 메시지 추가
new_message = HumanMessage(content="I'm looking for information on whales, specifically", name="Lance", id="2")

# 테스트
add_messages(initial_messages , new_message)
```

```
# 출력 - 기본 메시지를 덮어씀

[AIMessage(content='Hello! How can I assist you?', additional_kwargs={}, response_metadata={}, name='Model', id='1'),
 HumanMessage(content="I'm looking for information on whales, specifically", additional_kwargs={}, response_metadata={}, name='Lance', id='2')]
```

### 2.  7.    메시지 삭제 (Removal)

`add_messages`는 [메시지 삭제](https://langchain-ai.github.io/langgraph/how-tos/memory/delete-messages/){: target="_blank"}도 할 수 있습니다.

이를 위해 `langchain_core`의 [RemoveMessage](https://api.python.langchain.com/en/latest/messages/langchain_core.messages.modifier.RemoveMessage.html){: target="_blank"}를 사용합니다.

```python
from langchain_core.messages import RemoveMessage

# 메시지 목록
messages = [AIMessage("Hi.", name="Bot", id="1")]
messages.append(HumanMessage("Hi.", name="Lance", id="2"))
messages.append(AIMessage("So you said you were researching ocean mammals?", name="Bot", id="3"))
messages.append(HumanMessage("Yes, I know about whales. But what others should I learn about?", name="Lance", id="4"))

# 삭제할 메세지를 분리
delete_messages = [RemoveMessage(id=m.id) for m in messages[:-2]]
print(delete_messages)
```

`delete_messages`에서 메시지 id 1과 id 2는 리듀서에 의해 삭제가 됩니다.

```
# 출력 - 삭제할 메시지 목록

[RemoveMessage(content='', additional_kwargs={}, response_metadata={}, id='1'), RemoveMessage(content='', additional_kwargs={}, response_metadata={}, id='2')]
```

```python
add_messages(messages , delete_messages)
```

```
# 출력 - 메시지 id 1과 2가 삭제

[AIMessage(content='So you said you were researching ocean mammals?', additional_kwargs={}, response_metadata={}, name='Bot', id='3'),
 HumanMessage(content='Yes, I know about whales. But what others should I learn about?', additional_kwargs={}, response_metadata={}, name='Lance', id='4')]
```

## 정리

모든 노드는 `상태 스키마`로 통신합니다. LangGraph는 키와 값의 타입을 지정하는 `TypedDict`, 첨자를 이용한 간결한 구문을 제공하는 `Dataclass` 및 데이터 검증이 가능한 `Pydanitic` 등 다양한 상태 스키마를 제공합니다.

`상태 리듀서`에서는 상태 업데이트 방식을 지정할 수 있습니다. 기본적으로는 값을 덮어쓰지만 `operator.add` 리듀서 함수를 이용해서 값을 기존 리스트에 추가할 수 있고 `커스텀 리듀서`를 이용하여 출력에서 오류가 발생하지 않게 제어할 수 있습니다.

`메시지`는 `커스텀과 확장`으로 `사용자 지정 TypedDict`를 사용할 수 있고 `기존 MessageState`를 사용하여 코드 중복을 피할 수 있습니다. 그리고 기존 메시지에 동일한 ID를 추가하여 메시지를 덮어쓰거나 삭제할 수 있습니다.

다음 포스팅에서는 `상태와 메모리`의 `다중 스키마`, `메시지 필터링 및 트리밍`, `메시지 요약 및 외부 DB 메모리`를 사용할 수 있는 챗봇을 알아보겠습니다.


## References

* [LangGraph 공식 문서](https://langchain-ai.github.io/langgraph){: target="_blank"}
* [LangChain Academy](https://github.com/langchain-ai/langchain-academy){: target="_blank"}
