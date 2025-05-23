---
title: LangGraph 상태와 메모리 (2)
date: 2025-05-23 11:15:43 +/-TTTT
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
series_order: 4
---

노드 간 통신을 위한 상태 스키마와 리듀서에 대해서 이해했습니다.
다음은 입·출력에서 상태 스키마를 위한 다중 스키마, 그리고 요약 기능과 외부메모리 기능을 가진 챗봇에 대해서 알아보겠습니다. 

> 학습할 리소스는 [LangChain Academy Github](https://github.com/langchain-ai/langchain-academy){: target="_blank"}를 사용합니다.
{: .prompt-info }

## 3.   다중 스키마 (Multiple Schemas)

일반적으로 모든 그래프 노드는 단일 스키마(Single Schema)로 통신합니다.

단일 스키마는 그래프의 입력 및 출력 키/채널을 포함합니다.

노드끼리 필수가 아닌 정보를 전달하고 싶지 않거나 서로 다른 입·출력 스키마를 사용하고 싶을 때 `다중 스키마`를 사용할 수 있습니다.

```python
# 환경 구성
%%capture --no-stderr
%pip install --quiet -U langgraph
```

### 3.  1.  개인 상태 (Private State)

먼저 노드 간에 [개인 상태](https://langchain-ai.github.io/langgraph/how-tos/pass_private_state/){: target="_blank"}를 전달하는 경우를 살펴보겠습니다.

그래프 전체의 입·출력에는 영향을 주지 않지만, 중간 처리 로직에서 필요한 데이터를 주고받을 때 유용합니다.

다음 코드에서 `OverallState`와 `PrivateState` 두 가지 상태 타입을 정의합니다. `PrivateState` 상태 타입의 `baz` 키를 주목해서 보겠습니다.

- `node_1`은 `OverallState`를 입력받아서 `PrivateState`를 반환합니다.
- `baz` 키는 `PrivateState`에만 `포함`됩니다.
- `node_2`는 `PrivateState`를 입력받아서 `OverallState`를 다시 반환합니다.
- `baz` 키는 `OverallState`에서 없으므로 `제외`됩니다.

```python
from typing_extensions import TypedDict
from IPython.display import Image, display
from langgraph.graph import StateGraph, START, END

class OverallState(TypedDict):
    foo: int

class PrivateState(TypedDict):
    baz: int

def node_1(state: OverallState) -> PrivateState:
    print("---Node 1---")
    return {"baz": state['foo'] + 1}

def node_2(state: PrivateState) -> OverallState:
    print("---Node 2---")
    return {"foo": state['baz'] + 1}

# 그래프 생성
builder = StateGraph(OverallState)
builder.add_node("node_1", node_1)
builder.add_node("node_2", node_2)

# 엣지 설정
builder.add_edge(START, "node_1")
builder.add_edge("node_1", "node_2")
builder.add_edge("node_2", END)

# 그래프 컴파일
graph = builder.compile()

# 그래프 이미지
display(Image(graph.get_graph().draw_mermaid_png()))
```

```python
# 그래프 호출
graph.invoke({"foo" : 1})
```

```
# 출력 - baz는 제외됨
---
---Node 2---
{'foo': 3}

```

### 3.  2.  입/출력 스키마

기본적으로 `StateGraph`는 단일 스키마를 받으며 모든 노드는 해당 스키마와 통신해야 합니다.

하지만 [그래프에 대한 명시적인 입력 및 출력 스키마를 정의](https://langchain-ai.github.io/langgraph/how-tos/input_output_schema/?h=input+outp){: target="_blank"}할 수도 있습니다.

이러면 보통 그래프 작업과 관련된 `모든 키`를 포함하는 [내부 스키마(Internal Schema)](https://langchain-ai.github.io/langgraph/concepts/low_level/#schema){: target="_blank"}를 정의합니다.

하지만 입력과 출력을 `제한`하기 위해 특정 `입력` 및 `출력` 스키마를 사용합니다.

첫 번째 예는 단일 스키마만 살펴보겠습니다.

```python
# 단일 스키마만 사용
class OverallState(TypedDict):
    question: str
    answer: str
    notes: str

def thinking_node(state: OverallState):
    return {"answer": "bye", "notes": "... his name is Lance"}

def answer_node(state: OverallState):
    return {"answer": "bye Lance"}

graph = StateGraph(OverallState) # 내부 스키마로 정의
graph.add_node("answer_node", answer_node)
graph.add_node("thinking_node", thinking_node)
graph.add_edge(START, "thinking_node")
graph.add_edge("thinking_node", "answer_node")
graph.add_edge("answer_node", END)

graph = graph.compile()

# 그래프 이미지
display(Image(graph.get_graph().draw_mermaid_png()))

```

```
# 출력

{'question': 'hi', 'answer': 'bye Lance', 'notes': '... his name is Lance'}
```

다음은 그래프에 특정 `input`과 `output` 스키마를 사용해 보겠습니다.

`input` / `output` 스키마는 그래프의 입력과 출력에 허용되는 키를 `필터링`합니다.

또한, `state: InputState` 타입 힌트를 사용하여 각 노드의 입력 스키마를 지정할 수 있습니다.

코드에서 `answer_node`의 출력이 `OutputState`로 `필터링`되는 것을 보여줍니다.

```python
# 특정 입/출력 스키마를 사용
class InputState(TypedDict):
    question: str

class OutputState(TypedDict):
    answer: str

class OverallState(TypedDict):
    question: str
    answer: str
    notes: str

def thinking_node(state: InputState): # InputState 입력 스키마 정의
    return {"answer": "bye", "notes": "... his is name is Lance"}

def answer_node(state: OverallState) -> OutputState: # OutputState로 필터링
    return {"answer": "bye Lance"}

graph = StateGraph(OverallState, input=InputState, output=OutputState)
graph.add_node("answer_node", answer_node)
graph.add_node("thinking_node", thinking_node)
graph.add_edge(START, "thinking_node")
graph.add_edge("thinking_node", "answer_node")
graph.add_edge("answer_node", END)

graph = graph.compile()

# 그래프 이미지
display(Image(graph.get_graph().draw_mermaid_png()))

graph.invoke({"question":"hi"})
```

`output` 스키마는 출력을 `answer` 키로만 제한하는 것을 볼 수 있습니다.

```
 # 출력 - answer 키만 출력됨

{'answer': 'bye Lance'}
```

## 4.   메시지 필터링 및 트리밍

그래프 상태에서 `메시지를 처리하는 방법`에 대해서 좀 더 고급 기능인 `메시지 필터링 및 트리밍`을 살펴보겠습니다.

환경 구성을 위해 이전 포스팅에서 살펴본 `메시지 상태`와 `리듀서`를 간단하게 재구성하겠습니다.

우선, 환경 구성을 실행합니다.

```python
%%capture --no-stderr
%pip install --quiet -U langchain_core langgraph langchain_openai
```

```python
import os, getpass

def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

_set_env("OPENAI_API_KEY")
```

[추적(Traces)](https://docs.smith.langchain.com/observability/concepts#traces){: target="_blank"}을 위해 [LangSmith](https://docs.smith.langchain.com/){: target="_blank"}도 사용합니다.

```python
_set_env("LANGSMITH_API_KEY")
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "langchain-academy"
```

### 2.  1.  메시지 상태 (Messages as state)

먼저 간단한 메시지를 정의합니다.

```python
from pprint import pprint
from langchain_core.messages import AIMessage, HumanMessage
messages = [AIMessage(f"So you said you were researching ocean mammals?", name="Bot")]
messages.append(HumanMessage(f"Yes, I know about whales. But what others should I learn about?", name="Lance"))

for m in messages:
    m.pretty_print()
```

```
# 출력

================================== Ai Message ==================================
Name: Bot

So you said you were researching ocean mammals?
================================ Human Message =================================
Name: Lance

Yes, I know about whales. But what others should I learn about?
```

메시지를 채팅 모델에 전달합니다.

```python
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o")
llm.invoke(messages)
```

```
# 출력

AIMessage(content='In addition to whales, there are several other fascinating ocean mammals that you might find interesting to learn about:\n\n1. **Dolphins**: Highly intelligent and social creatures, dolphins are known for their playful behavior and complex communication skills.\n\n2. **Porpoises**: Similar to dolphins but generally smaller with different facial structures, porpoises are also intelligent and social.\n\n3. **Seals**: These marine mammals are found in various parts of the world and are known for their playful nature and ability to live on both land and sea.\n\n4. **Sea Lions**: Often confused with seals, sea lions are more social and tend to gather in large colonies. They are also known for their ability to "walk" on land using their flippers.\n\n5. **Walruses**: Recognizable by their long tusks and whiskers, walruses are social animals usually found in the Arctic region.\n\n6. **Manatees**: Also known as sea cows, manatees are gentle giants that graze on sea grasses and are primarily found in warm coastal waters.\n\n7. **Dugongs**: Similar to manatees, dugongs are marine herbivores found in warm coastal waters from East Africa to Australia.\n\n8. **Sea Otters**: While not fully aquatic, sea otters spend most of their time in the water and are known for their tool-using behavior, such as using rocks to crack open shellfish.\n\n9. **Polar Bears**: Although primarily land animals, polar bears are excellent swimmers and rely heavily on the ocean for food, primarily hunting seals.\n\nThese animals vary greatly in their behaviors, habitats, and adaptations to the marine environment, providing a rich field of study for anyone interested in ocean mammals.', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 357, 'prompt_tokens': 39, 'total_tokens': 396, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-2024-08-06', 'system_fingerprint': 'fp_a6889ffe71', 'id': 'chatcmpl-BOy5LoxcM3Wv7r2blr1LIE1HOmQ4S', 'finish_reason': 'stop', 'logprobs': None}, id='run-fd560b96-bdba-4742-babf-3c7c1c59dc95-0', usage_metadata={'input_tokens': 39, 'output_tokens': 357, 'total_tokens': 396, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}})
```

`MessagesState`를 사용하여 간단한 그래프로 채팅 모델을 실행합니다.

```python
from IPython.display import Image, display
from langgraph.graph import MessagesState
from langgraph.graph import StateGraph, START, END

# 노드 정의
def chat_model_node(state: MessagesState):
    return {"messages": llm.invoke(state["messages"])}

# 그래프 생성
builder = StateGraph(MessagesState)
builder.add_node("chat_model", chat_model_node)
builder.add_edge(START, "chat_model")
builder.add_edge("chat_model", END)
graph = builder.compile()

# 그래프 이미지
display(Image(graph.get_graph().draw_mermaid_png()))
```

```python
# 그래프 호출
output = graph.invoke({'messages': messages})
for m in output['messages']:
    m.pretty_print()
```

```
# 출력

================================== Ai Message ==================================
Name: Bot

So you said you were researching ocean mammals?
================================ Human Message =================================
Name: Lance

Yes, I know about whales. But what others should I learn about?
================================== Ai Message ==================================

In addition to whales, there are several other ocean mammals you might find interesting to study:

1. **Dolphins**: Highly intelligent and social creatures, dolphins are known for their playful behavior and complex communication. There are numerous species, including the common bottlenose dolphin and the orca, or killer whale, which is actually the largest dolphin species.

2. **Porpoises**: Similar to dolphins, porpoises are generally smaller and have different physical features, such as a more rounded face and triangular dorsal fin. The harbor porpoise is a commonly known species.

3. **Seals**: These are pinnipeds, characterized by their fin-like limbs. Different types of seals include the harbor seal, gray seal, and the enormous elephant seal.

--- 중간 생략 ---

Understanding these diverse marine mammals involves exploring their habitats, behaviors, and the roles they play in their ecosystems, as well as the challenges they face due to human activities and climate change.
```

### 4.  2.  리듀서 (Reducer)

메시지 작업에서 장시간 실행되는 대화를 관리하는 것은 어렵습니다.

메시지 목록이 늘어나거나 토큰 사용량이 높아지고 지연 시간이 발생할 수 있기 때문입니다.

이전 포스팅에서 살펴본 `RemoveMessage`와 `add_messages` 리듀서로 해결할 수 있습니다.

```python
from langchain_core.messages import RemoveMessage

# 노드 정의
def filter_messages(state: MessagesState):
    # 최근 2개 메시지 제외한 모든 메시지 삭제
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
    return {"messages": delete_messages}

def chat_model_node(state: MessagesState):  
    return {"messages": [llm.invoke(state["messages"])]}

# 그래프 생성
builder = StateGraph(MessagesState)
builder.add_node("filter", filter_messages)
builder.add_node("chat_model", chat_model_node)
builder.add_edge(START, "filter")
builder.add_edge("filter", "chat_model")
builder.add_edge("chat_model", END)
graph = builder.compile()

# 그래프 이미지
display(Image(graph.get_graph().draw_mermaid_png()))
```

```python
# 인사말이 추가된 메시지 목록
messages = [AIMessage("Hi.", name="Bot", id="1")]
messages.append(HumanMessage("Hi.", name="Lance", id="2"))
messages.append(AIMessage("So you said you were researching ocean mammals?", name="Bot", id="3"))
messages.append(HumanMessage("Yes, I know about whales. But what others should I learn about?", name="Lance", id="4"))

# 그래프 호출
output = graph.invoke({'messages': messages})
for m in output['messages']:
    m.pretty_print()
```

```
# 출력 - 최근 2개의 메시지를 제외하고 메시지가 모두 삭제되었습니다.

================================== Ai Message ==================================
Name: Bot

So you said you were researching ocean mammals?
================================ Human Message =================================
Name: Lance

Yes, I know about whales. But what others should I learn about?
================================== Ai Message ==================================

In addition to whales, there are several other fascinating ocean mammals you might find interesting:

1. **Dolphins**: Known for their intelligence and playful nature, dolphins are a diverse group with various species like the bottlenose dolphin and the orca (often called killer whale, although it`s actually a dolphin).

2. **Porpoises**: Often confused with dolphins, porpoises are generally smaller and have different dental and fin features. They are less commonly seen but equally intriguing.

3. **Seals**: These are pinnipeds, meaning they have fin-like limbs. Different species include harbor seals and grey seals, known for their curious and often sociable behavior.

-- 중간 생략 --

Exploring these diverse marine mammals will give you a broader understanding of the rich and intricate ecosystems of the oceans.

```

### 4.  3.  필터링 메시지 (Filtering messages)

이제 `필터링 메시지`를 살펴보겠습니다.

특정 메시지 전달을 위해 그래프 상태를 변경 없이 그대로 두고 싶다면 채팅 모델에 전달하는 메시지만 필터링합니다.

예를 들어, `마지막 메시지`만 전달하기 위해서 필터링된 목록인 `llm.invoke(messages[-1:])`를 모델에 전달합니다.

```python
# 노드 정의
def chat_model_node(state: MessagesState):
    return {"messages": [llm.invoke(state["messages"][-1:])]}

# 그래프 생성
builder = StateGraph(MessagesState)
builder.add_node("chat_model", chat_model_node)
builder.add_edge(START, "chat_model")
builder.add_edge("chat_model", END)
graph = builder.compile()

# 그래프 이미지
display(Image(graph.get_graph().draw_mermaid_png()))
```

기존 메시지 목록의 LLM 응답과 후속 질문을 추가합니다.

```python
messages.append(output['messages'][-1])
messages.append(HumanMessage(f"Tell me more about Narwhals!", name="Lance"))
```

```python
for m in messages:
    m.pretty_print()
```

```
# 출력

================================== Ai Message ==================================
Name: Bot

Hi.
================================ Human Message =================================
Name: Lance

Hi.
================================== Ai Message ==================================
Name: Bot

So you said you were researching ocean mammals?
================================ Human Message =================================
Name: Lance

Yes, I know about whales. But what others should I learn about?
================================== Ai Message ==================================

In addition to whales, there are several other fascinating ocean mammals you might find interesting:

1. **Dolphins**: Known for their intelligence and playful nature, dolphins are a diverse group with various species like the bottlenose dolphin and the orca (often called killer whale, although it`s actually a dolphin).

-- 중간 생략 --
...
================================ Human Message =================================
Name: Lance

Tell me more about Narwhals!


```

후속 질문과 관련 LLM 응답이 출력된 것을 확인할 수 있습니다.

```python
# 메시지 필터링을 사용하여 그래프 호출
output = graph.invoke({'messages': messages})
for m in output['messages']:
    m.pretty_print()
```

```
# 출력

================================== Ai Message ==================================
Name: Bot

Hi.
================================ Human Message =================================
Name: Lance

Hi.
================================== Ai Message ==================================
Name: Bot

So you said you were researching ocean mammals?
================================ Human Message =================================
Name: Lance

Yes, I know about whales. But what others should I learn about?
================================== Ai Message ==================================

In addition to whales, there are several other fascinating ocean mammals you might find interesting:

-- 중간 생략 --

================================ Human Message =================================
Name: Lance

Tell me more about Narwhals!
================================== Ai Message ==================================

Narwhals, often referred to as the `unicorns of the sea,` are a unique species of Arctic-dwelling whales known for their distinct long, spiral tusk that protrudes from their heads. Here are some key points about narwhals:

1. **Scientific Classification**: The narwhal’s scientific name is *Monodon monoceros*. They belong to the family Monodontidae, which also includes beluga whales.

-- 중간 생략 --
```

상태에는 모든 메시지가 있지만 [LangSmith](https://smith.langchain.com){: target="_blank"} 추적을 살펴보면 모델 호출이 마지막 메시지만 사용한다는 것을 알 수 있습니다.

![LangSmith 추적](assets/posts/2025-05-23-langgraph-state-and-memory-2nd/langsmith01.png)
_LangSmith Tracing_

### 4.  4.  메시지 트리밍 (Trim Messages)

또 다른 방법은 설정된 토큰 수를 기준으로 [메시지 자르는](https://python.langchain.com/v0.2/docs/how_to/trim_messages/#getting-the-last-max_tokens-tokens){: target="_blank"} 것입니다.

이렇게 하면 메시지 기록이 지정된 토큰 수로 제한됩니다.

필터링은 에이전트 간 처리 후에 메시지 일부만 반환하는 반면, 트리밍은 채팅 모델이 응답하는 데 사용할 수 있는 토큰 수를 제한합니다.

아래 코드에서 `trim_messages`를 참조합니다.

```python
from langchain_core.messages import trim_messages

# 노드 설정
def chat_model_node(state: MessagesState):
    messages = trim_messages(
            state["messages"],
            max_tokens=100,
            strategy="last",
            token_counter=ChatOpenAI(model="gpt-4o"),
            allow_partial=False,
        )
    return {"messages": [llm.invoke(messages)]}

# 그래프 생성
builder = StateGraph(MessagesState)
builder.add_node("chat_model", chat_model_node)
builder.add_edge(START, "chat_model")
builder.add_edge("chat_model", END)
graph = builder.compile()

# 그래프 이미지
display(Image(graph.get_graph().draw_mermaid_png()))
```

기존 메시지를 모두 불러와서 후속 질문을 합니다.

```python
messages.append(output['messages'][-1])
messages.append(HumanMessage(f"Tell me where Orcas live!", name="Lance"))
```

```python
# 메시지 트리밍 예 - 토큰 수를 최대 100개로 제한
trim_messages(
            messages,
            max_tokens=100,
            strategy="last",
            token_counter=ChatOpenAI(model="gpt-4o"),
            allow_partial=False
        )
```

```
# 출력

[HumanMessage(content='Tell me where Orcas live!', additional_kwargs={}, response_metadata={}, name='Lance')]
```

```python
# chat_model_node에서 메시지 트리밍을 사용하여 호출
messages_out_trim = graph.invoke({'messages': messages})
```

LangSmith 추적을 살펴보면 모델 호출을 확인할 수 있습니다.

## 5.   메시지 요약 챗봇

메시지를 자르거나 필터링으로 대화를 제거하는 방법도 있지만

장기간 대화 시 일정 길이 이상이면 이전 메시지를 요약하고 압축하여 보존할 수 있습니다.

또 챗봇에 메모리를 적용하여 높은 토큰 비용이나 지연 없이 장기간 대화할 수 있습니다.

우선 OpenAI 및 LangSmith 환경을 설정합니다.

```python
%%capture --no-stderr
%pip install --quiet -U langchain_core langgraph langchain_openai
```

```python
import os, getpass

def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

_set_env("OPENAI_API_KEY")
```

```python
_set_env("LANGSMITH_API_KEY")
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "langchain-academy"
```

상태에서 이전에 살펴본 `MessagesState`를 사용하여 기본 `messages` 키 외 사용자 지정 `summary` 키를 포함합니다.

```python
from langgraph.graph import MessagesState
class State(MessagesState):
    summary: str
```

`summary(요약)`가 존재하면 이를 프롬프트에 포함하여 LLM을 호출하는 `call_model` 노드를 정의합니다.

```python
from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage

# 모델을 호출하는 로직 정의
def call_model(state: State):
  
    # 요약이 있으면 가져옴
    summary = state.get("summary", "")

    # 요약이 있으면 추가
    if summary:
  
        # 시스템 메시지에 요약 추가
        system_message = f"Summary of conversation earlier: {summary}"

        # 이후 메시지들 앞에 요약 메시지 추가
        messages = [SystemMessage(content=system_message)] + state["messages"]
  
    else:
        messages = state["messages"]
  
    response = model.invoke(messages)
    return {"messages": response}
```

요약을 생성하는 `summarize_conversation` 노드를 정의합니다.

해당 노드는 요약 생성 후 상태를 필터링하기 위해 `RemoveMessage`를 사용합니다.

```python
def summarize_conversation(state: State):
  
    # 기존 요약을 가져옵니다.
    summary = state.get("summary", "")

    # 요약 프롬프트 생성
    if summary:
  
        # 이미 요약이 존재 시 메시지
        summary_message = (
            f"This is summary of the conversation to date: {summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )
  
    else:
        summary_message = "Create a summary of the conversation above:"

    # 프롬프트를 대화 기록에 추가
    messages = state["messages"] + [HumanMessage(content=summary_message)]
    response = model.invoke(messages)
  
    # 최근 메시지 2개를 제외한 나머지 삭제
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
    return {"summary": response.content, "messages": delete_messages}
```

대화 길이에 따라 요약과 생성을 결정하는 `조건부 에지`를 추가합니다.

```python
from langgraph.graph import END
# 대화 종료 또는 요약 여부 결정
def should_continue(state: State):
  
    """Return the next node to execute."""
  
    messages = state["messages"]
  
    # 메시지가 6개를 초과하면 대화를 요약
    if len(messages) > 6:
        return "summarize_conversation"
  
    # 그렇지 않으면 종료
    return END
```

### 5.  1.  메모리 추가

`상태(State)`는 단일 그래프 실행하는 동안에만 [일시적으로 유지](https://github.com/langchain-ai/langgraph/discussions/352#discussioncomment-9291220){: target="_blank"}되기 때문에 다중-턴(Multi-Turn) 대화를 수행하기 어렵습니다.

이전 포스팅에서 설명한 [퍼시스턴스(persistance)](https://langchain-ai.github.io/langgraph/how-tos/persistence/){: target="_blank"}를 사용하여 해결할 수 있습니다.

LangGraph는 `체크포인터(checkpointer)`를 사용해 각 단계가 끝날 때마다 그래프 상태를 자동으로 저장할 수 있습니다.

내장된 `퍼시스턴스 계층`은 메모리를 제공하여, LangGraph가 마지막 상태 업데이트 지점부터 대화를 이어갈 수 있게 해줍니다.

그래프 상태를 위한 인메모리 키-값 저장소인 `MemorySaver`로 쉽게 구현할 수 있습니다.

```python
from IPython.display import Image, display
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START

# 새 그래프 정의
workflow = StateGraph(State)
workflow.add_node("conversation", call_model)
workflow.add_node(summarize_conversation)

# 엔드포인트를 conversation으로 설정
workflow.add_edge(START, "conversation")
workflow.add_conditional_edges("conversation", should_continue)
workflow.add_edge("summarize_conversation", END)

# 컴파일
memory = MemorySaver()
graph = workflow.compile(checkpointer=memory)
display(Image(graph.get_graph().draw_mermaid_png()))
```

### 5.  2.  스레드

`체크포인터`는 매 단계의 상태를 체크포인트로 저장합니다.

저장된 체크포인트들은 대화의 `스레드`로 그룹화할 수 있습니다.

Slack을 예로 들면, 서로 다른 채널이 각각 다른 대화를 담고 있습니다.

스레드는 Slack 채널과 같아서, 대화 등의 상태 집합을 캡처합니다.

아래에서는 `configurable`을 사용해 스레드 ID를 설정합니다.

```python
# 스레드 생성
config = {"configurable": {"thread_id": "1"}}

# 대화 시작
input_message = HumanMessage(content="hi! I'm Lance")
output = graph.invoke({"messages": [input_message]}, config) 
for m in output['messages'][-1:]:
    m.pretty_print()

input_message = HumanMessage(content="what's my name?")
output = graph.invoke({"messages": [input_message]}, config) 
for m in output['messages'][-1:]:
    m.pretty_print()

input_message = HumanMessage(content="i like the 49ers!")
output = graph.invoke({"messages": [input_message]}, config) 
for m in output['messages'][-1:]:
    m.pretty_print()
```

```
# 출력

================================== Ai Message ==================================

Hello Lance! How can I assist you today?
================================== Ai Message ==================================

You mentioned that your name is Lance. How can I help you today?
================================== Ai Message ==================================

That's great! The San Francisco 49ers have a rich history and a passionate fan base. Do you have a favorite player or a memorable game that you particularly enjoyed?
```

아래 코드를 실행시키면 `should_continue`에서 조건부 엣지 설정을 메시지가 6개 미만으로 설정했기 때문에 아직 상태 요약이 없습니다.

```python
graph.get_state(config).values.get("summary","")
```

`thread ID`가 설정된 `config`를 통해 이전에 기록된 상태에서부터 이어갈 수 있습니다.

![그래프, 슈퍼스텝, 체크포인트, 스레드](assets/posts/2025-05-12-langgraph-component/20250501_134128_pic_M01_07_01.png)
_thread_id가 설정된 config를 이전 대화를 불러올 수 있습니다_

```python
input_message = HumanMessage(content="i like Nick Bosa, isn't he the highest paid defensive player?")
output = graph.invoke({"messages": [input_message]}, config) 
for m in output['messages'][-1:]:
    m.pretty_print()
```

```
# 출력

================================== Ai Message ==================================

Yes, as of September 2023, Nick Bosa became the highest-paid defensive player in NFL history. He signed a five-year contract extension with the San Francisco 49ers worth $170 million, with $122.5 million guaranteed. Bosa is known for his exceptional skills as a defensive end and has been a key player for the 49ers.
```

다시 상태 요약을 가져옵니다.

```python
graph.get_state(config).values.get("summary","")
```

```
# 출력 - 요약

'Lance introduced himself and mentioned that he is a fan of the San Francisco 49ers, specifically highlighting his admiration for Nick Bosa. The conversation noted that Nick Bosa became the highest-paid defensive player in NFL history as of September 2023, with a five-year, $170 million contract extension with the 49ers.'
```

## 6.   메시지 요약 및 외부 메모리 DB 기능 적용 챗봇

챗봇이 영구적으로 메모리를 유지해야 한다면 어떻게 해야 할까요?

외부 데이터베이스를 지원하는 체크포인터를 사용하여 유지할 수 있습니다.

코드에서는 [Sqlite를 체크포인터](https://langchain-ai.github.io/langgraph/concepts/low_level/#checkpointer){: target="_blank"}로 사용하지만, [Postgres](https://langchain-ai.github.io/langgraph/how-tos/persistence_postgres/){: target="_blank"} 등 다양한 체크포인터로 사용이 가능합니다.

먼저, 환경을 구성합니다.

```python
%%capture --no-stderr
%pip install --quiet -U langgraph-checkpoint-sqlite langchain_core langgraph langchain_openai
```

```python
import os, getpass

def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

_set_env("OPENAI_API_KEY")
```

### 6.  1.  Sqlite

[SqliteSaver 체크포인터](https://langchain-ai.github.io/langgraph/concepts/low_level/#checkpointer){: target="_blank"}로 구성합니다.

`":memory:"`를 입력하면 메모리 내 Sqlite 데이터베이스가 생성됩니다.

> Sqlite는 [작고 빠르며 매우 인기 있는](https://x.com/karpathy/status/1819490455664685297){: target="_blank"} SQL 데이터베이스입니다.
{: .prompt-info }

```python
import sqlite3
# In memory
conn = sqlite3.connect(":memory:", check_same_thread = False)
```

db 경로를 제공하면 데이터베이스가 생성됩니다.

```python
# 파일이 없으면 다운로드하고 로컬 DB에 연결
!mkdir -p state_db && [ ! -f state_db/example.db ] && wget -P state_db https://github.com/langchain-ai/langchain-academy/raw/main/module-2/state_db/example.db

db_path = "state_db/example.db"
conn = sqlite3.connect(db_path, check_same_thread=False)
```

```python
# 체크포인터 정의
from langgraph.checkpoint.sqlite import SqliteSaver
memory = SqliteSaver(conn)
```

챗봇을 다시 정의합니다. 기존 `MemorySaver` 기반한 챗봇 코드와 동일하고 `Sqlite` 구성만 차이가 있습니다.

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage

from langgraph.graph import END
from langgraph.graph import MessagesState

model = ChatOpenAI(model="gpt-4o",temperature=0)

class State(MessagesState):
    summary: str

# 모델을 호출하는 로직 정의
def call_model(state: State):
  
    # 요약이 있으면 가져오기
    summary = state.get("summary", "")

    # 요약이 있으면 추가
    if summary:
  
        # 시스템 메시지에 요약 추가
        system_message = f"Summary of conversation earlier: {summary}"

        # 이후 메시지들 앞에 요약 메시지 추가
        messages = [SystemMessage(content=system_message)] + state["messages"]
  
    else:
        messages = state["messages"]
  
    response = model.invoke(messages)
    return {"messages": response}

def summarize_conversation(state: State):
  
    # 기존 요약을 가져옵니다.
    summary = state.get("summary", "")

    # 요약 프롬프트 생성 
    if summary:
  
        # 이미 요약이 존재 시 메시지
        summary_message = (
            f"This is summary of the conversation to date: {summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )
  
    else:
        summary_message = "Create a summary of the conversation above:"

    # 프롬프트 대화 기록에 추가
    messages = state["messages"] + [HumanMessage(content=summary_message)]
    response = model.invoke(messages)
  
    # 최신 메시지 2개를 제외한 나머지 삭제
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
    return {"summary": response.content, "messages": delete_messages}

# 대화 종료 또는 요약 여부 결정
def should_continue(state: State):
  
    """Return the next node to execute."""
  
    messages = state["messages"]
  
    # 메시지가 6개를 초과하면 대화를 요약
    if len(messages) > 6:
        return "summarize_conversation"
  
    # 그렇지 않으면 종료
    return END
```

이제 `sqlite 체크포인터`로 다시 컴파일하면 그래프를 여러 번 불러올 수 있습니다.

```python
from IPython.display import Image, display
from langgraph.graph import StateGraph, START

# 새 그래프 정의
workflow = StateGraph(State)
workflow.add_node("conversation", call_model)
workflow.add_node(summarize_conversation)

# conversation 엔트리포인트 구성
workflow.add_edge(START, "conversation")
workflow.add_conditional_edges("conversation", should_continue)
workflow.add_edge("summarize_conversation", END)

# 컴파일
graph = workflow.compile(checkpointer=memory)
display(Image(graph.get_graph().draw_mermaid_png()))
```

이제, 그래프를 여러 번 호출이 가능합니다.

```python
# 스레드 생성
config = {"configurable": {"thread_id": "1"}}

# 대화 시작
input_message = HumanMessage(content="hi! I'm Lance")
output = graph.invoke({"messages": [input_message]}, config) 
for m in output['messages'][-1:]:
    m.pretty_print()

input_message = HumanMessage(content="what's my name?")
output = graph.invoke({"messages": [input_message]}, config) 
for m in output['messages'][-1:]:
    m.pretty_print()

input_message = HumanMessage(content="i like the 49ers!")
output = graph.invoke({"messages": [input_message]}, config) 
for m in output['messages'][-1:]:
    m.pretty_print()
```

```
# 출력

================================== Ai Message ==================================

Hello again, Lance! It's great to hear from you. If there's anything specific you'd like to discuss or any questions you have, feel free to let me know!
================================== Ai Message ==================================

Your name is Lance! If there's anything else you'd like to talk about or explore, just let me know.
================================== Ai Message ==================================

That's awesome! The San Francisco 49ers have a rich history and a passionate fan base. Is there anything specific about the 49ers you'd like to discuss, like their current season, players, or memorable moments?
```

상태가 `로컬(Sqlite DB)`에 저장되어 있는지 확인하면 출력되는 것을 볼 수 있습니다.

```python
config = {"configurable": {"thread_id": "1"}}
graph_state = graph.get_state(config)
graph_state
```

```
# 출력

StateSnapshot(values={'messages': [HumanMessage(content="hi! I'm Lance", additional_kwargs={}, response_metadata={}, id='f5900607-033c-4e0a-b1ee-273c10106af9'), AIMessage(content="Hello again, Lance! It's great to hear from you. If there's anything specific you'd like to discuss or any questions you have, feel free to let me know!", additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 34, 'prompt_tokens': 337, 'total_tokens': 371, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-2024-08-06', 'system_fingerprint': 'fp_d8864f8b6b', 'id': 'chatcmpl-BOyhuchtEnm1L16Q1MSX9Zu1u8yZI', 'finish_reason': 'stop', 'logprobs': None}, id='run-0036777c-5fc1-4555-b89f-2400e587ae43-0', usage_metadata={'input_tokens': 337, 'output_tokens': 34, 'total_tokens': 371, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}}), HumanMessage(content="what's my name?", additional_kwargs={}, response_metadata={}, id='a4c2cec9-2106-4004-8b7f-307c48366f68'), AIMessage(content="Your name is Lance! If there's anything else you'd like to talk about or explore, just let me know.", additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 23, 'prompt_tokens': 204, 'total_tokens': 227, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-2024-08-06', 'system_fingerprint': 'fp_f7a584cf1f', 'id': 'chatcmpl-BOyhxIBPwTFMbnuVqbdROKwGKrWMk', 'finish_reason': 'stop', 'logprobs': None}, id='run-4355d2c6-a11f-44cd-b7e8-6747492428bc-0', usage_metadata={'input_tokens': 204, 'output_tokens': 23, 'total_tokens': 227, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}}), HumanMessage(content='i like the 49ers!', additional_kwargs={}, response_metadata={}, id='f3f2840e-b114-499d-918d-dfa35c80771a'), AIMessage(content="That's awesome! The San Francisco 49ers have a rich history and a passionate fan base. Is there anything specific about the 49ers you'd like to discuss, like their current season, players, or memorable moments?", additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 45, 'prompt_tokens': 241, 'total_tokens': 286, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-2024-08-06', 'system_fingerprint': 'fp_f7a584cf1f', 'id': 'chatcmpl-BOyhyzfIeyAKB96bbC5mW11hQE30d', 'finish_reason': 'stop', 'logprobs': None}, id='run-09a257a5-0161-4815-893d-00ceccee6989-0', usage_metadata={'input_tokens': 241, 'output_tokens': 45, 'total_tokens': 286, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}})], 'summary': "Lance introduced himself multiple times throughout the conversation, consistently expressing his fondness for the San Francisco 49ers football team. The AI assistant acknowledged Lance's name each time and demonstrated a willingness to engage in a discussion about the 49ers, offering to explore various aspects of the team, such as their history, current roster, memorable games, and more. Despite the AI's attempts to delve deeper into the topic, the conversation remained brief and somewhat repetitive, with Lance reintroducing himself at the end without directly engaging with the AI's questions or prompts about the 49ers. The interaction was friendly, but it did not progress beyond initial introductions and expressions of interest in the football team."}, next=(), config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1f01f2ad-9ba1-6b06-801b-bed4edd043b6'}}, metadata={'source': 'loop', 'writes': {'conversation': {'messages': AIMessage(content="That's awesome! The San Francisco 49ers have a rich history and a passionate fan base. Is there anything specific about the 49ers you'd like to discuss, like their current season, players, or memorable moments?", additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 45, 'prompt_tokens': 241, 'total_tokens': 286, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-2024-08-06', 'system_fingerprint': 'fp_f7a584cf1f', 'id': 'chatcmpl-BOyhyzfIeyAKB96bbC5mW11hQE30d', 'finish_reason': 'stop', 'logprobs': None}, id='run-09a257a5-0161-4815-893d-00ceccee6989-0', usage_metadata={'input_tokens': 241, 'output_tokens': 45, 'total_tokens': 286, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}})}}, 'step': 27, 'parents': {}, 'thread_id': '1'}, created_at='2025-04-22T03:35:35.042509+00:00', parent_config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1f01f2ad-8fbc-6a5c-801a-bea5395c98ef'}}, tasks=())
```

### 6.  2.  상태 유지 (Persisting state)

노트북 커널(Kernel)을 `다시 시작`하여 로컬의 `Sqlite DB`에서 로드가 되어 상태 유지되는 것이 확인됩니다.

```python
# 스레드 생성
config = {"configurable": {"thread_id": "1"}}
graph_state = graph.get_state(config)
graph_state
```

```
# 출력

StateSnapshot(values={'messages': [HumanMessage(content="hi! I'm Lance", additional_kwargs={}, response_metadata={}, id='f5900607-033c-4e0a-b1ee-273c10106af9'), AIMessage(content="Hello again, Lance! It's great to hear from you. If there's anything specific you'd like to discuss or any questions you have, feel free to let me know!", additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 34, 'prompt_tokens': 337, 'total_tokens': 371, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-2024-08-06', 'system_fingerprint': 'fp_d8864f8b6b', 'id': 'chatcmpl-BOyhuchtEnm1L16Q1MSX9Zu1u8yZI', 'finish_reason': 'stop', 'logprobs': None}, id='run-0036777c-5fc1-4555-b89f-2400e587ae43-0', usage_metadata={'input_tokens': 337, 'output_tokens': 34, 'total_tokens': 371, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}}), HumanMessage(content="what's my name?", additional_kwargs={}, response_metadata={}, id='a4c2cec9-2106-4004-8b7f-307c48366f68'), AIMessage(content="Your name is Lance! If there's anything else you'd like to talk about or explore, just let me know.", additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 23, 'prompt_tokens': 204, 'total_tokens': 227, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-2024-08-06', 'system_fingerprint': 'fp_f7a584cf1f', 'id': 'chatcmpl-BOyhxIBPwTFMbnuVqbdROKwGKrWMk', 'finish_reason': 'stop', 'logprobs': None}, id='run-4355d2c6-a11f-44cd-b7e8-6747492428bc-0', usage_metadata={'input_tokens': 204, 'output_tokens': 23, 'total_tokens': 227, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}}), HumanMessage(content='i like the 49ers!', additional_kwargs={}, response_metadata={}, id='f3f2840e-b114-499d-918d-dfa35c80771a'), AIMessage(content="That's awesome! The San Francisco 49ers have a rich history and a passionate fan base. Is there anything specific about the 49ers you'd like to discuss, like their current season, players, or memorable moments?", additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 45, 'prompt_tokens': 241, 'total_tokens': 286, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-2024-08-06', 'system_fingerprint': 'fp_f7a584cf1f', 'id': 'chatcmpl-BOyhyzfIeyAKB96bbC5mW11hQE30d', 'finish_reason': 'stop', 'logprobs': None}, id='run-09a257a5-0161-4815-893d-00ceccee6989-0', usage_metadata={'input_tokens': 241, 'output_tokens': 45, 'total_tokens': 286, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}})], 'summary': "Lance introduced himself multiple times throughout the conversation, consistently expressing his fondness for the San Francisco 49ers football team. The AI assistant acknowledged Lance's name each time and demonstrated a willingness to engage in a discussion about the 49ers, offering to explore various aspects of the team, such as their history, current roster, memorable games, and more. Despite the AI's attempts to delve deeper into the topic, the conversation remained brief and somewhat repetitive, with Lance reintroducing himself at the end without directly engaging with the AI's questions or prompts about the 49ers. The interaction was friendly, but it did not progress beyond initial introductions and expressions of interest in the football team."}, next=(), config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1f01f2ad-9ba1-6b06-801b-bed4edd043b6'}}, metadata={'source': 'loop', 'writes': {'conversation': {'messages': AIMessage(content="That's awesome! The San Francisco 49ers have a rich history and a passionate fan base. Is there anything specific about the 49ers you'd like to discuss, like their current season, players, or memorable moments?", additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 45, 'prompt_tokens': 241, 'total_tokens': 286, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-2024-08-06', 'system_fingerprint': 'fp_f7a584cf1f', 'id': 'chatcmpl-BOyhyzfIeyAKB96bbC5mW11hQE30d', 'finish_reason': 'stop', 'logprobs': None}, id='run-09a257a5-0161-4815-893d-00ceccee6989-0', usage_metadata={'input_tokens': 241, 'output_tokens': 45, 'total_tokens': 286, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}})}}, 'step': 27, 'parents': {}, 'thread_id': '1'}, created_at='2025-04-22T03:35:35.042509+00:00', parent_config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1f01f2ad-8fbc-6a5c-801a-bea5395c98ef'}}, tasks=())
```

## 정리

`LangGraph`는 `다중 스키마`를 통해 서로 다른 입출력 스키마를 구성하여 제한을 둘 수 있습니다.

`메시지 필터링`으로 특정 메시지만 모델에 전달할 수 있고 `메시지 트리밍`으로 토큰 수를 제한하여 비용 및 성능을 개선할 수 있습니다.

또한, 챗봇에 `메시지 요약 기능`을 적용하여 기존 전체 대화 내용을 불러오지 않고 대화를 요약하여 높은 토큰 비용이나 지연 없이 장기간 대화가 가능합니다.

인메모리 기반 퍼시스턴스인 `MemorySaver`로 그래프 상태 유지를 쉽게 구성할 수 있고 영구적 저장을 위한 외부 데이터베이스인 `Sqlite`와 같은 다양한 외부데이터베이스를 사용할 수 있습니다.

다음 포스팅에서는 실시간 스트리밍, 일시 정지, 상태 직접 수정 및 피드백과 같은 `사용자 경험(UX)`과 필요시 사람이 직접 개입하거나 수정하는 과정인 `휴먼-인-더-루프`를 알아보겠습니다.

## References

* [LangGraph 공식 문서](https://langchain-ai.github.io/langgraph){: target="_blank"}
* [LangChain Academy](https://github.com/langchain-ai/langchain-academy){: target="_blank"}
