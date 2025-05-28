---
title: LangGraph 사용자 경험(UX)과 휴먼-인-더-루프
date: 2025-05-28 12:15:43 +/-TTTT
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
series_order: 5
---

`그래프 상태와 메모리`를 커스터마이징하는 방법과 장시간 대화를 유지할 수 있는 `외부 메모리 기반 챗봇`을 만들었습니다.

이번 포스팅에서는 `스트리밍`, `브레이크포인트`, `타임트래블` 등 다양한 방식으로 직접 상호작용을 할 수 있는 `사용자 경험`과  `휴먼-인-더-루프`를 살펴보겠습니다.

먼저, 그래프 실행 과정에서 모델 응답 또는 상태 변화를 실시간으로 출력하는 `스트리밍`을 살펴보겠습니다.

> 학습할 리소스는 [LangChain Academy Github](https://github.com/langchain-ai/langchain-academy){: target="_blank"}를 사용합니다.
{: .prompt-info }

```python
# 환경 구성
%%capture --no-stderr
%pip install --quiet -U langgraph langchain_openai langgraph_sdk
```

## 1.   스트리밍 (Streaming)

LangGraph는 스트리밍 기능을 핵심적으로 지원하도록 설계되었습니다.

이전 포스팅에서 만든 챗봇을 다시 구성하고, 그래프 실행 중 출력을 스트리밍하는 다양한 방법을 살펴보겠습니다.

참고로, 토큰 단위 스트리밍을 활성화하기 위해 `call_model`와 함께 `RunnableConfig`를 사용해야 합니다.

이 설정은 Python < 3.11에서만 필요합니다.

노트북이 CoLab(기본 Python 3.x 사용)에서 실행될 경우를 대비한 코드가 포함되었습니다.

```python
import os, getpass

def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

_set_env("OPENAI_API_KEY")
```

```python
from IPython.display import Image, display

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage
from langchain_core.runnables import RunnableConfig

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph import MessagesState

# LLM
model = ChatOpenAI(model="gpt-4o", temperature=0) 

# 상태 (State) 
class State(MessagesState):
    summary: str

# 모델을 호출하는 로직 정의
def call_model(state: State, config: RunnableConfig):
  
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
  
    response = model.invoke(messages, config)
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

# 새 그래프 정의
workflow = StateGraph(State)
workflow.add_node("conversation", call_model)
workflow.add_node(summarize_conversation)

# conversation 엔트리포인트 구성
workflow.add_edge(START, "conversation")
workflow.add_conditional_edges("conversation", should_continue)
workflow.add_edge("summarize_conversation", END)

# 컴파일
memory = MemorySaver()
graph = workflow.compile(checkpointer=memory)
display(Image(graph.get_graph().draw_mermaid_png()))
```

### 1.  1.  전체 상태 스트리밍 (Streaming full state)

이제 [그래프 상태를 스트리밍하는 방법](https://langchain-ai.github.io/langgraph/concepts/low_level/#streaming){: target="_blank"}에 대해 알아보겠습니다.

`.stream`과 `.astream`은 각각 동기(sync) 및 비동기(async) 방식으로 결과를 스트리밍하는 메서드입니다.

LangGraph는 [그래프 상태](https://langchain-ai.github.io/langgraph/how-tos/stream-values/){: target="_blank"}에 대해 몇 가지 [다양한 스트리밍 모드](https://langchain-ai.github.io/langgraph/how-tos/stream-values/){: target="_blank"}를 지원합니다.

* `values`: 각 노드가 실행된 후 그래프의 `전체 상태`를 스트리밍합니다.
* `updates`: 각 노드가 실행된 후 그래프 상태에 `변경이 생긴 부분만` 스트리밍합니다.

![그래프 상태를 스트리밍하는 방법](assets/posts/2025-05-28-langgraph-ux-and-human-in-the-loop/streaming_01.png)
_그래프 상태를 스트리밍하는 방법_

먼저 `stream_mode="updates"`에 대해 살펴보겠습니다.

`updates` 모드로 스트리밍하면, 그래프 내에서 각 노드가 실행된 후 상태에 변화가 있었던 부분만 볼 수 있습니다.

각 `chunk`는 `node_name`을 키로, 업데이트된 상태를 값으로 가지는 딕셔너리 형태로 제공됩니다.

```python
# 스레드 생성
config = {"configurable": {"thread_id": "1"}}

# conversation 시작
for chunk in graph.stream({"messages": [HumanMessage(content="hi! I'm Lance")]}, config, stream_mode="updates"):
    print(chunk)
```

```
# 출력

{'conversation': {'messages': AIMessage(content='Hello Lance! How can I assist you today?', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 11, 'prompt_tokens': 11, 'total_tokens': 22, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-2024-08-06', 'system_fingerprint': 'fp_f7a584cf1f', 'id': 'chatcmpl-BP1EKWZkNHTHK0tvi2cF2KKKLGX8V', 'finish_reason': 'stop', 'logprobs': None}, id='run-c47d3a7c-77f9-4dff-8c71-03442f5697e1-0', usage_metadata={'input_tokens': 11, 'output_tokens': 11, 'total_tokens': 22, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}})}}
```

이제 상태 업데이트된 내용만 출력합니다.

```python
# 대화 시작
for chunk in graph.stream({"messages": [HumanMessage(content="hi! I'm Lance")]}, config, stream_mode="updates"):
    chunk['conversation']["messages"].pretty_print()
```

```
# 출력

================================== Ai Message ==================================

Hi Lance! How's it going? What can I do for you today?
```

`stream_mode="values"`로 설정 시 conversation 노드가 호출 후 `전체 상태(full state)`가 됩니다.

```python
# 새로운 대화를 다시 시작
config = {"configurable": {"thread_id": "2"}}

# 대화 시작
input_message = HumanMessage(content="hi! I'm Lance")
for event in graph.stream({"messages": [input_message]}, config, stream_mode="values"):
    for m in event['messages']:
        m.pretty_print()
    print("---"*25)
```

```
# 출력

================================ Human Message =================================

hi! I'm Lance
---------------------------------------------------------------------------
================================ Human Message =================================

hi! I'm Lance
================================== Ai Message ==================================

Hello Lance! How can I assist you today?
---------------------------------------------------------------------------
```

### 1.  2.  토큰 스트리밍 (Streaming tokens)

그래프 상태 외에 다양한 정보를 스트리밍하고 싶은 경우가 있는데, 특히 채팅 모델 호출에서는 `토큰이 생성되는 즉시 실시간으로 스트리밍`하는 것이 일반적입니다.

이런 경우 `.astream_events` 메서드를 사용하는데 이 메서드는 노드 내에서 발생하는 이벤트를 실시간으로 스트리밍합니다.

각 이벤트는 여러 키를 가진 딕셔너리 타입입니다.

* `event`: 발생한 이벤트의 타입입니다.
* `name`: 이벤트의 이름입니다.
* `data`: 이벤트와 관련된 데이터입니다.
* `metadata`: 이벤트를 발생시킨 노드 정보를 담고 있는 `langgraph_node`를 포함합니다.

실제로 동작하는 과정을 살펴보겠습니다.

```python
config = {"configurable": {"thread_id": "3"}}
input_message = HumanMessage(content="Tell me about the 49ers NFL team")
async for event in graph.astream_events({"messages": [input_message]}, config, version="v2"):
    print(f"Node: {event['metadata'].get('langgraph_node','')}. Type: {event['event']}. Name: {event['name']}")
```

```
# 출력

Node: . Type: on_chain_start. Name: LangGraph
Node: conversation. Type: on_chain_start. Name: conversation
Node: conversation. Type: on_chat_model_start. Name: ChatOpenAI
Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
-- 중간 생략 --
Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
Node: conversation. Type: on_chat_model_end. Name: ChatOpenAI
Node: conversation. Type: on_chain_start. Name: should_continue
Node: conversation. Type: on_chain_end. Name: should_continue
Node: conversation. Type: on_chain_stream. Name: conversation
Node: conversation. Type: on_chain_end. Name: conversation
Node: . Type: on_chain_stream. Name: LangGraph
Node: . Type: on_chain_end. Name: LangGraph
```

핵심은, 그래프 내에서 생성된 챗 모델의 토큰들은 `on_chat_model_stream` 타입의 이벤트로 전달된다는 점입니다.

`event['metadata']['langgraph_node']`를 이용하면 어떤 노드에서 스트리밍되고 있는지 확인할 수 있습니다.

그리고 `event['data']`를 통해 각 이벤트의 실제 데이터(이 경우에는 `AIMessageChunk`)를 가져올 수 있습니다.

```python
node_to_stream = 'conversation'
config = {"configurable": {"thread_id": "4"}}
input_message = HumanMessage(content="Tell me about the 49ers NFL team")
async for event in graph.astream_events({"messages": [input_message]}, config, version="v2"):
    # 특정 노드에서 생성된 챗 모델 토큰을 가져옴
    if event["event"] == "on_chat_model_stream" and event['metadata'].get('langgraph_node','') == node_to_stream:
        print(event["data"])
```

```
# 출력

{'chunk': AIMessageChunk(content='', additional_kwargs={}, response_metadata={}, id='run-7d01ef19-86ae-43de-b12e-16df1db42e9c')}
{'chunk': AIMessageChunk(content='The', additional_kwargs={}, response_metadata={}, id='run-7d01ef19-86ae-43de-b12e-16df1db42e9c')}
{'chunk': AIMessageChunk(content=' San', additional_kwargs={}, response_metadata={}, id='run-7d01ef19-86ae-43de-b12e-16df1db42e9c')}
{'chunk': AIMessageChunk(content=' Francisco', additional_kwargs={}, response_metadata={}, id='run-7d01ef19-86ae-43de-b12e-16df1db42e9c')}
{'chunk': AIMessageChunk(content=' ', additional_kwargs={}, response_metadata={}, id='run-7d01ef19-86ae-43de-b12e-16df1db42e9c')}
{'chunk': AIMessageChunk(content='49', additional_kwargs={}, response_metadata={}, id='run-7d01ef19-86ae-43de-b12e-16df1db42e9c')}
{'chunk': AIMessageChunk(content='ers', additional_kwargs={}, response_metadata={}, id='run-7d01ef19-86ae-43de-b12e-16df1db42e9c')}
{'chunk': AIMessageChunk(content=' are', additional_kwargs={}, response_metadata={}, id='run-7d01ef19-86ae-43de-b12e-16df1db42e9c')}
{'chunk': AIMessageChunk(content=' a', additional_kwargs={}, response_metadata={}, id='run-7d01ef19-86ae-43de-b12e-16df1db42e9c')}
{'chunk': AIMessageChunk(content=' professional', additional_kwargs={}, response_metadata={}, id='run-7d01ef19-86ae-43de-b12e-16df1db42e9c')}
-- 중간 생략 --
{'chunk': AIMessageChunk(content=' to', additional_kwargs={}, response_metadata={}, id='run-7d01ef19-86ae-43de-b12e-16df1db42e9c')}
{'chunk': AIMessageChunk(content=' the', additional_kwargs={}, response_metadata={}, id='run-7d01ef19-86ae-43de-b12e-16df1db42e9c')}
{'chunk': AIMessageChunk(content=' NFL', additional_kwargs={}, response_metadata={}, id='run-7d01ef19-86ae-43de-b12e-16df1db42e9c')}
{'chunk': AIMessageChunk(content="'s", additional_kwargs={}, response_metadata={}, id='run-7d01ef19-86ae-43de-b12e-16df1db42e9c')}
{'chunk': AIMessageChunk(content=' development', additional_kwargs={}, response_metadata={}, id='run-7d01ef19-86ae-43de-b12e-16df1db42e9c')}
{'chunk': AIMessageChunk(content=' and', additional_kwargs={}, response_metadata={}, id='run-7d01ef19-86ae-43de-b12e-16df1db42e9c')}
{'chunk': AIMessageChunk(content=' popularity', additional_kwargs={}, response_metadata={}, id='run-7d01ef19-86ae-43de-b12e-16df1db42e9c')}
{'chunk': AIMessageChunk(content='.', additional_kwargs={}, response_metadata={}, id='run-7d01ef19-86ae-43de-b12e-16df1db42e9c')}
{'chunk': AIMessageChunk(content='', additional_kwargs={}, response_metadata={'finish_reason': 'stop', 'model_name': 'gpt-4o-2024-08-06', 'system_fingerprint': 'fp_f7a584cf1f'}, id='run-7d01ef19-86ae-43de-b12e-16df1db42e9c')}
```

출력에서 확인한 결과 `chunk` 키를 사용하면 `AIMessageChunk`에 바로 접근할 수 있습니다.

다음은 `event["data"]["chunk"].content`만 출력해서 실제 텍스트 토큰만 시각적으로 보여줍니다.

```python
config = {"configurable": {"thread_id": "5"}}
input_message = HumanMessage(content="Tell me about the 49ers NFL team")
async for event in graph.astream_events({"messages": [input_message]}, config, version="v2"):
    # 특정 노드에서 생성된 챗 모델 토큰을 가져옴 
    if event["event"] == "on_chat_model_stream" and event['metadata'].get('langgraph_node','') == node_to_stream:
        data = event["data"]
        print(data["chunk"].content, end="|")
```

```
# 출력

|The| San| Francisco| |49|ers| are| a| professional| American| football| team| based| in| the| San| Francisco| Bay| Area|.| They| compete| in| the| National| Football| League| (|NFL|)| as| a| member| of| the| league|'s| National| Football| Conference| (|N|FC|)| West| division|.| The| team| was| founded| in| |194|6| as| a| charter| member| of| the| All|-Amer|ica| Football| Conference| (|AA|FC|)| and| joined| the| NFL| in| |194|9| when| the| leagues| merged|.

|###| Key| Points|:

|-| **|Team| Name| and| Colors|**|:| The| team| is| named| after| the| prospect|ors| who| arrived| in| Northern| California| during| the| |184|9| Gold| Rush|.| The| team's| colors| are| red|,| gold|,| and| white|.

|-| **|St|adium|**|:| The| |49|ers| play| their| home| games| at| Levi|'s| Stadium| in| Santa| Clara|,| California|,| which| they| moved| to| in| |201|4|.| Before| that|,| they| played| at| Cand|lestick| Park| in| San| Francisco|.

|-| **|Champ|ionship|s|**|:| The| |49|ers| have| won| five| Super| Bowl| titles| (|X|VI|,| XIX|,| XX|III|,| XX|IV|,| and| XX|IX|),| with| their| most| successful| period| being| the| |198|0|s| and| early| |199|0|s|.| They| have| also| won| numerous| division| titles| and| conference| championships|.

|-| **|Not|able| Figures|**|:| The| team| has| had| several| Hall| of| Fame| players|,| including| Joe| Montana|,| Jerry| Rice|,| Steve| Young|,| Ronnie| L|ott|,| and| Charles| Haley|.| Bill| Walsh|,| the| legendary| head| coach|,| is| credited| with| developing| the| West| Coast| offense|,| which| became| a| staple| of| the| team's| success|.

|-| **|R|ival|ries|**|:| The| |49|ers| have| notable| rival|ries| with| the| Seattle| Seahawks|,| Los| Angeles| Rams|,| and| historically| with| the| Dallas| Cowboys| and| Green| Bay| Packers|.

|-| **|Recent| Performance|**|:| In| recent| years|,| the| |49|ers| have| been| competitive|,| reaching| the| Super| Bowl| in| the| |201|9| season| but| losing| to| the| Kansas| City| Chiefs|.| They| have| been| known| for| their| strong| defense| and| innovative| offensive| strategies| under| head| coach| Kyle| Shan|ahan|.

|-| **|Ownership| and| Management|**|:| The| team| is| owned| by| the| York| family|,| with| Jed| York| serving| as| the| CEO|.| The| general| manager| is| John| Lynch|,| a| former| NFL| player| and| Hall| of| F|amer|.

|The| |49|ers| have| a| rich| history| and| a| passionate| fan| base|,| making| them| one| of| the| iconic| franchises| in| the| NFL|.||
```

### 1.  3.  브레이크포인트 (Breakpoints)

`스트리밍` 기능의 기반을 통해 `휴먼-인-더-루프` 상황에서는 그래프가 실행되는 중간중간에 출력 결과를 확인할 수 있습니다.

`휴먼-인-더-루프`의 필요성에 좀 더 자세히 보겠습니다.

- `승인(Approval)` : 에이전트 실행을 중단하고, 현재 상태를 사용자에게 보여주어 사용자가 행동을 `승인(accept)`할 수 있습니다.
- `디버깅(Debugging)` : 그래프 실행을 되돌려(rewind) 문제를 재현하거나 회피할 수 있습니다.
- `수정(Editing)` : 사용자가 상태(state)를 직접 수정할 수 있습니다.

LangGraph는 다양한 `휴먼-인-더-루프` 워크플로를 지원하기 위해 에이전트의 상태를 조회하거나 업데이트할 수 있는 여러 방법을 제공합니다.

먼저, [브레이크포인트](https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/breakpoints/#simple-usage){: target="_blank"}는 그래프 실행 도중 특정 단계에서 일시 정지를 가능하게 하는 간단한 방법입니다.

이를 통해 사용자의 `승인`이 어떻게 구현되는지 살펴보겠습니다.

```python
# 환경 구성
%%capture --no-stderr
%pip install --quiet -U langgraph langchain_openai langgraph_sdk langgraph-prebuilt
```

```python
import os, getpass

def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

_set_env("OPENAI_API_KEY")
```

### 1.  4.  사용자 승인 브레이크포인트 설정

에이전트가 어떤 `도구(Tool)`를 사용할 때마다 사용자의 승인을 받고 싶을 경우 그래프를 컴파일할 때 `interrupt_before=["tools"]` 옵션만 지정하면 됩니다. 여기서 `tools`는 도구를 실행하는 노드를 의미합니다.

이렇게 하면 도구 호출을 실행하는 `tools` 노드 전에 실행이 중단되어, 사용자의 승인을 받을 수 있게 됩니다.

이전 포스팅에서 다뤘던 에이전트를 다시 구성합니다.

```python
from langchain_openai import ChatOpenAI

# 이 함수들이 도구(Tool)가 됩니다
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
    """Divide a by b.

    Args:
        a: first int
        b: second int
    """
    return a / b

tools = [add, multiply, divide]
llm = ChatOpenAI(model="gpt-4o")
llm_with_tools = llm.bind_tools(tools)
```

```python
from IPython.display import Image, display

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition, ToolNode

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# 시스템 메시지
sys_msg = SystemMessage(content="You are a helpful assistant tasked with performing arithmetic on a set of inputs.")

# 노드
def assistant(state: MessagesState):
   return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

# 그래프
builder = StateGraph(MessagesState)

# 노드 정의 : 실제 작업을 수행
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

# 엣지 정의 : 제어 흐름을 결정
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    # Assistant의 최신 메시지(결과)가 도구 호출이면 -> tools_condition이 tools로 라우팅
    # Assistant의 최신 메시지(결과)가 도구 호출이 아니면 -> tools_condition이 END로 라우팅
    tools_condition,
)
builder.add_edge("tools", "assistant")

memory = MemorySaver()
graph = builder.compile(interrupt_before=["tools"], checkpointer=memory)

# 그래프 이미지
display(Image(graph.get_graph(xray=True).draw_mermaid_png()))
```

```python
# 입력
initial_input = {"messages": HumanMessage(content="Multiply 2 and 3")}

# 스레드
thread = {"configurable": {"thread_id": "1"}}

# 첫 번째 브레이크포인트까지 그래프 실행
for event in graph.stream(initial_input, thread, stream_mode="values"):
    event['messages'][-1].pretty_print()
```

```
# 출력

================================ Human Message =================================

Multiply 2 and 3
================================== Ai Message ==================================
Tool Calls:
  multiply (call_loLcP5bomRYH7ZsFVyRVWsb4)
 Call ID: call_loLcP5bomRYH7ZsFVyRVWsb4
  Args:
    a: 2
    b: 3
```

상태(state)를 확인하고, 다음에 호출할 노드를 볼 수 있습니다.

이렇게 하면 그래프가 `중단(interrupted)`되었음을 쉽게 확인할 수 있습니다.

```python
state = graph.get_state(thread)
state.next
```

```
# 출력

('tools',)
```

아래에서 `StateSnapshot`는 LangGraph에서 실행 중간에 저장한 그래프 상태의 체크포인트입니다.

`Graph.get_state()`에서 최근의 그래프 상태를 가져오고 `Graph.get_state_history()`에서 전체 실행 과정에서 저장된 모든 상태(Snapshot)들을 배열로 반환합니다. 그리고 `Graph.stream(None,{thread_id})`는 브레이크포인트 이후, 그래프를 `None`으로 호출(invoke)하면, 마지막 상태 체크포인트(StateSnapshot)에서 바로 이어서 실행됩니다.

![상태 체크포인트](assets/posts/2025-05-28-langgraph-ux-and-human-in-the-loop/state_checkpoint01.png)
_상태 체크포인트 흐름_

명확하게 보여주기 위해, LangGraph는 `도구 호출`이 포함된 `AIMessage`가 있는 현재 상태를 다시 출력합니다.

그리고 그래프의 다음 단계들을 실행하는데, 이 단계는 `도구 노드(tool node)`부터 시작합니다.

도구 노드가 해당 `도구 호출`로 실행되고, 그 결과가 최종 답변을 위해 챗 모델(chat model)로 다시 전달되는 것을 볼 수 있습니다.

```python
for event in graph.stream(None, thread, stream_mode="values"):
    event['messages'][-1].pretty_print()
```

```
# 출력 - 마지막 상태 체크포인트에서 바로 이어서 실행

================================== Ai Message ==================================
Tool Calls:
  multiply (call_loLcP5bomRYH7ZsFVyRVWsb4)
 Call ID: call_loLcP5bomRYH7ZsFVyRVWsb4
  Args:
    a: 2
    b: 3
================================= Tool Message =================================
Name: multiply

6
================================== Ai Message ==================================

The result of multiplying 2 and 3 is 6.
```

사용자 입력을 받아들이는 `승인 단계`와 함께, 앞서 살펴본 내용을 합칩니다.

```python
# 입력
initial_input = {"messages": HumanMessage(content="Multiply 2 and 3")}

# 스레드 설정
thread = {"configurable": {"thread_id": "2"}}

# 첫 번째 중단 지점까지 그래프 실행
for event in graph.stream(initial_input, thread, stream_mode="values"):
    event['messages'][-1].pretty_print()

# 사용자 피드백 받기
user_approval = input("Do you want to call the tool? (yes/no): ")

# 승인 여부 확인
if user_approval.lower() == "yes":
  
    # 승인이 된 경우, 그래프 실행 계속
    for event in graph.stream(None, thread, stream_mode="values"):
        event['messages'][-1].pretty_print()
  
else:
    print("Operation cancelled by user.")
```

```
# 출력

================================ Human Message =================================

Multiply 2 and 3
================================== Ai Message ==================================
Tool Calls:
  multiply (call_958CVku85GEVFNUt99YGK7KP)
 Call ID: call_958CVku85GEVFNUt99YGK7KP
  Args:
    a: 2
    b: 3
================================== Ai Message ==================================
Tool Calls:
  multiply (call_958CVku85GEVFNUt99YGK7KP)
 Call ID: call_958CVku85GEVFNUt99YGK7KP
  Args:
    a: 2
    b: 3
================================= Tool Message =================================
Name: multiply

6
================================== Ai Message ==================================

The product of 2 and 3 is 6.
```

![사용자 입력을 위한 승인 단계](assets/posts/2025-05-28-langgraph-ux-and-human-in-the-loop/breakpoints_01.png)
_사용자 입력을 위한 승인 단계_

## 2.   그래프 상태 수정

그래프 상태를 직접 수정하고, 사용자의 피드백을 반영하는 방법을 살펴보겠습니다.

```python
# 환경 구성
%%capture --no-stderr
%pip install --quiet -U langgraph langchain_openai langgraph_sdk langgraph-prebuilt
```

```python
import os, getpass

def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

_set_env("OPENAI_API_KEY")
```

### 2.  1.  상태 수정하기

이전에 `브레이크포인트(Breakpoints)`에 대해서 살펴보았습니다.

`브레이크포인트`를 사용해 그래프 실행을 중단시키고, 다음 노드를 실행하기 전에 `사용자 승인`을 기다렸습니다.

하지만 `브레이크포인트`는 [그래프 상태를 수정할 기회](https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/edit-graph-state/){: target="_blank"}이기도 합니다.

이제 `assistant` 노드 앞에 `브레이크포인트`를 설정해 보겠습니다.

```python
from langchain_openai import ChatOpenAI

# 이 함수들이 도구(Tool)가 됩니다
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
    """Divide a by b.

    Args:
        a: first int
        b: second int
    """
    return a / b

tools = [add, multiply, divide]
llm = ChatOpenAI(model="gpt-4o")
llm_with_tools = llm.bind_tools(tools)
```

```python
 from IPython.display import Image, display

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition, ToolNode

from langchain_core.messages import HumanMessage, SystemMessage

# 시스템 메시지
sys_msg = SystemMessage(content="You are a helpful assistant tasked with performing arithmetic on a set of inputs.")

# 노드
def assistant(state: MessagesState):
   return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

# 그래프
builder = StateGraph(MessagesState)

# 노드 정의 : 실제 작업을 수행
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

# 엣지 정의 : 제어 흐름을 결정
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    # Assistant의 최신 메시지(결과)가 도구 호출이면 -> tools_condition이 tools로 라우팅
    # Assistant의 최신 메시지(결과)가 도구 호출이 아니면 -> tools_condition이 END로 라우팅 routes to END
    tools_condition,
)
builder.add_edge("tools", "assistant")

memory = MemorySaver()
graph = builder.compile(interrupt_before=["assistant"], checkpointer=memory) # assistant 실행 전에 멈춤

# 그래프 이미지
display(Image(graph.get_graph(xray=True).draw_mermaid_png()))
```

실행하면, 채팅 모델이 응답하기 전에 그래프가 `중단`된 것을 확인할 수 있습니다.

```python
# 입력
initial_input = {"messages": "Multiply 2 and 3"}

# 스레드
thread = {"configurable": {"thread_id": "1"}}

# 첫 번째 브레이크포인트까지 그래프 실행
for event in graph.stream(initial_input, thread, stream_mode="values"): # HumanMessage가 들어간 상태에서 브레이크포인트 멈춤
    event['messages'][-1].pretty_print()
```

```
# 출력

================================ Human Message =================================

Multiply 2 and 3
```

```python
state = graph.get_state(thread)
state
```

```
# 출력

StateSnapshot(values={'messages': [HumanMessage(content='Multiply 2 and 3', additional_kwargs={}, response_metadata={}, id='3144cc67-b0d4-44c5-8f77-c341ae8de898')]}, next=('assistant',), config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1f01f4cb-752d-6ca2-8000-59677f0a62d6'}}, metadata={'source': 'loop', 'writes': None, 'step': 0, 'parents': {}, 'thread_id': '1'}, created_at='2025-04-22T07:38:00.212182+00:00', parent_config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1f01f4cb-752c-60aa-bfff-5322d1ec7b12'}}, tasks=(PregelTask(id='2a9d5404-03c5-0851-7971-7299e4511d74', name='assistant', path=('__pregel_pull', 'assistant'), error=None, interrupts=(), state=None, result=None),))
```

이제 상태를 바로 수정할 수 있습니다.

참고로, `messages` 키에 대한 업데이트는 `add_messages` 리듀서를 사용합니다.

* 기존 메시지를 덮어쓰고 싶다면, 메시지의 `id`를 지정해서 전달하면 됩니다.
* 메시지를 단순히 리스트에 추가(append)하고 싶다면, 아래 예시처럼 `id`를 지정하지 않고 메시지를 전달하면 됩니다.

```python
# 브레이크포인트에서 멈춘 상태에서 수정
graph.update_state(
    thread,
    {"messages": [HumanMessage(content="No, actually multiply 3 and 3!")]}, # add_messages 리듀서가 작동
)
```

```
# 출력

{'configurable': {'thread_id': '1',
  'checkpoint_ns': '',
  'checkpoint_id': '1f01f4cf-512e-637a-8001-435aa2d9dd90'}}
```

위에서, 새로운 메시지와 함께 `update_state`를 호출했습니다.

`add_messages` 리듀서는 이 메시지를 상태의 `messages` 키에 추가(append)합니다.

```python
new_state = graph.get_state(thread).values
for m in new_state['messages']: # 새 메시지가 포함됨
    m.pretty_print()
```

```
# 출력

================================ Human Message =================================

Multiply 2 and 3
================================ Human Message =================================

No, actually multiply 3 and 3!
```

에이전트를 계속 진행합니다.

단순히 `None`을 전달하면, 현재 상태에서부터 다음 단계로 진행됩니다.

현재 상태를 출력한 뒤, 나머지 노드들도 순차적으로 실행됩니다.

```python
# 브레이크포인트에서 None을 넣어서 계속 진행
for event in graph.stream(None, thread, stream_mode="values"):
    event['messages'][-1].pretty_print()
```

```
# 출력 - graph.stream(None, thread, ...) 실행할 때마다 현재 상태에서 이어서 진행

================================ Human Message =================================

No, actually multiply 3 and 3!
================================== Ai Message ==================================
Tool Calls:
  multiply (call_5KOxipxesZ18kHaohSo7vhPp)
 Call ID: call_5KOxipxesZ18kHaohSo7vhPp
  Args:
    a: 3
    b: 3
================================= Tool Message =================================
Name: multiply

9
```

다시 `브레이크포인트`가 설정된 `assistant` 노드에 도달했습니다.

여기서도 `None`을 전달하여 계속 진행(`graph.stream(None, thread, ...) 실행 시`)할 수 있습니다.

```python
for event in graph.stream(None, thread, stream_mode="values"):
    event['messages'][-1].pretty_print()
```

```
# 출력

================================== Ai Message ==================================

The result of multiplying 3 and 3 is 9.
```

## 3.   동적 브레이크포인트 (Dynamic Breakpoints)

`브레이크포인트`는 그래프 컴파일 시 개발자가 특정 노드에 설정합니다.

하지만, 그래프가 실행 중에 동적으로 `스스로 중단(interrupt)`하는 것도 도움이 될 수 있습니다.

이것을 `내부 브레이크포인트(internal breakpoint)`라고 하며, [`NodeInterrupt`](https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/dynamic_breakpoints/#run-the-graph-with-dynamic-interrupt){: target="_blank"}를 사용해서 구현할 수 있습니다.

이 방식의 구체적인 장점은 다음과 같습니다.

1. 개발자가 정의한 논리에 따라 노드 내부에서 `조건부`로 중단시킬 수 있습니다.
2. `NodeInterrupt`에 원하는 데이터를 전달함으로써 `중단 사유`를 사용자에게 전달 할 수 있습니다.

이제 입력 길이에 따라 `NodeInterrupt`를 발생시키는 그래프를 생성합니다.

```python
# 환경 구성
%%capture --no-stderr
%pip install --quiet -U langgraph langchain_openai langgraph_sdk
```

```python
from IPython.display import Image, display

from typing_extensions import TypedDict
from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import NodeInterrupt
from langgraph.graph import START, END, StateGraph

class State(TypedDict):
    input: str

def step_1(state: State) -> State:
    print("---Step 1---")
    return state

def step_2(state: State) -> State:
    # 입력값의 길이가 5자를 초과하면 NodeInterrupt를 발생시킬 수 있습니다.
    if len(state['input']) > 5:
        raise NodeInterrupt(f"Received input that is longer than 5 characters: {state['input']}")
  
    print("---Step 2---")
    return state

def step_3(state: State) -> State:
    print("---Step 3---")
    return state

builder = StateGraph(State)
builder.add_node("step_1", step_1)
builder.add_node("step_2", step_2)
builder.add_node("step_3", step_3)
builder.add_edge(START, "step_1")
builder.add_edge("step_1", "step_2")
builder.add_edge("step_2", "step_3")
builder.add_edge("step_3", END)

# 메모리 설정
memory = MemorySaver()

# 메모리를 적용하여 그래프 컴파일
graph = builder.compile(checkpointer=memory)

# 그래프 이미지
display(Image(graph.get_graph().draw_mermaid_png()))
```

5자보다 긴 입력값을 넣고 그래프를 실행합니다.

```python
initial_input = {"input": "hello world"}
thread_config = {"configurable": {"thread_id": "1"}}

# 첫 번째 중단이 발생할 때까지 그래프를 실행
for event in graph.stream(initial_input, thread_config, stream_mode="values"):
    print(event)
```

```
# 출력

{'input': 'hello world'}
---Step 1---
{'input': 'hello world'}
```

그래프 상태를 확인해 보면, 다음에 실행될 노드가 `step_2`로 설정되어 있습니다.

```python
state = graph.get_state(thread_config)
print(state.next)
```

```
# 출력

('step_2',)
```

`Interrupt`가 상태에 기록된 것도 확인할 수 있습니다.

```python
print(state.tasks)
```

```
# 출력 - step_2에서 중단된 것이 확인, interrupts 통해 NoInterrup 발생 확인, resumable에서 재개 안함.

(PregelTask(id='6b3dcd1b-35e4-94f2-df56-8d8e90bfda9f', name='step_2', path=('__pregel_pull', 'step_2'), error=None, interrupts=(Interrupt(value='Received input that is longer than 5 characters: hello world', resumable=False, ns=None),), state=None, result=None),)
```

그래프를 `브레이크포인트`에서 다시 이어서 실행이 가능합니다.

하지만 이렇게 하면 `동일한 노드가 다시 실행`되기 때문에 상태가 변경되지 않으면 `같은 위치에서 계속 멈춥`니다.

```python
for event in graph.stream(None, thread_config, stream_mode="values"):
    print(event)
```

```
# 출력

{'input': 'hello world'}
```

```python
state = graph.get_state(thread_config)
print(state.next)
```

```
# 출력 - 상태가 변경되지 않으면 같은 위치에서 멈춤

('step_2',)
```

상태를 변경합니다.

```python
# 상태 변경
graph.update_state(
    thread_config,
    {"input": "hi"},
)
```

```
# 출력

{'configurable': {'thread_id': '1',
  'checkpoint_ns': '',
  'checkpoint_id': '1f01f50a-2003-69e0-8002-fdfdec60e96e'}}
```

```python
for event in graph.stream(None, thread_config, stream_mode="values"):
    print(event)
```

```
# 출력 - 상태가 변경됨

{'input': 'hi'}
---Step 2---
{'input': 'hi'}
---Step 3---
{'input': 'hi'}
```

## 4.   타임 트래블 (Time travel)

LangGraph가 [디버깅을 지원](https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/time-travel/){: target="_blank"}하는 방법을 살펴보겠습니다.

과거 상태를 조회하고, 다시 실행하거나, 심지어 이전 상태에서 분기(forking)하는 것도 가능합니다.

이것을 `타임 트래블(time travel)`이라고 부릅니다.

```python
# 환경 구성
%%capture --no-stderr
%pip install --quiet -U langgraph langchain_openai langgraph_sdk langgraph-prebuilt
```

```python
import os, getpass

def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

_set_env("OPENAI_API_KEY")
```

이전 사용했던 코드를 다시 구현합니다.

```python
from langchain_openai import ChatOpenAI

# 이 함수들이 도구(Tool)가 됩니다
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
    """Divide a by b.

    Args:
        a: first int
        b: second int
    """
    return a / b

tools = [add, multiply, divide]
llm = ChatOpenAI(model="gpt-4o")
llm_with_tools = llm.bind_tools(tools)
```

```python
from IPython.display import Image, display

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState
from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt import tools_condition, ToolNode

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# 시스템 메시지
sys_msg = SystemMessage(content="You are a helpful assistant tasked with performing arithmetic on a set of inputs.")

# 노드
def assistant(state: MessagesState):
   return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

# 그래프
builder = StateGraph(MessagesState)

# 노드 정의 : 실제 작업을 수행
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

# 엣지 정의 : 제어 흐름을 결정
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    # Assistant의 최신 메시지(결과)가 도구 호출이면 -> tools_condition이 tools로 라우팅
    # Assistant의 최신 메시지(결과)가 도구 호출이 아니면 -> tools_condition이 END로 라우팅
    tools_condition,
)
builder.add_edge("tools", "assistant")

memory = MemorySaver()
graph = builder.compile(checkpointer=MemorySaver())

# 그래프 이미지
display(Image(graph.get_graph(xray=True).draw_mermaid_png()))
```

이전과 동일하게 실행합니다.

```python
# 입력
initial_input = {"messages": HumanMessage(content="Multiply 2 and 3")}

# 스레드
thread = {"configurable": {"thread_id": "1"}}

# 첫 번째 브레이크포인트까지 그래프 실행
for event in graph.stream(initial_input, thread, stream_mode="values"):
    event['messages'][-1].pretty_print()
```

```
# 출력

================================ Human Message =================================

Multiply 2 and 3
================================== Ai Message ==================================
Tool Calls:
  multiply (call_vk4Szqm7kfXZcQIVV0Fyf7JO)
 Call ID: call_vk4Szqm7kfXZcQIVV0Fyf7JO
  Args:
    a: 2
    b: 3
================================= Tool Message =================================
Name: multiply

6
================================== Ai Message ==================================

The result of multiplying 2 and 3 is 6.
```

### 4.  1.  히스토리 조회 (Browsing History)

`get_state`를 사용하면 `thread_id`를 기준으로 현재 그래프의 상태를 확인할 수 있습니다.

```python
graph.get_state({'configurable': {'thread_id': '1'}})
```

```
# 출력

StateSnapshot(values={'messages': [HumanMessage(content='Multiply 2 and 3', additional_kwargs={}, response_metadata={}, id='bc428612-f369-477c-abdd-41683738fec8'), AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_vk4Szqm7kfXZcQIVV0Fyf7JO', 'function': {'arguments': '{"a":2,"b":3}', 'name': 'multiply'}, 'type': 'function'}], 'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 18, 'prompt_tokens': 131, 'total_tokens': 149, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-2024-08-06', 'system_fingerprint': 'fp_a6889ffe71', 'id': 'chatcmpl-BP2zEr4CHtljWeys85hlx2zhD3fyc', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-19deb1ab-3b6b-45a3-aab6-036a7f129a62-0', tool_calls=[{'name': 'multiply', 'args': {'a': 2, 'b': 3}, 'id': 'call_vk4Szqm7kfXZcQIVV0Fyf7JO', 'type': 'tool_call'}], usage_metadata={'input_tokens': 131, 'output_tokens': 18, 'total_tokens': 149, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}}), ToolMessage(content='6', name='multiply', id='283433e6-64dd-492f-8474-c8df35c76031', tool_call_id='call_vk4Szqm7kfXZcQIVV0Fyf7JO'), AIMessage(content='The result of multiplying 2 and 3 is 6.', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 15, 'prompt_tokens': 156, 'total_tokens': 171, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-2024-08-06', 'system_fingerprint': 'fp_a6889ffe71', 'id': 'chatcmpl-BP2zFO9RO1aX78vPbCLh4hHdtnP6R', 'finish_reason': 'stop', 'logprobs': None}, id='run-0e7d02dd-a762-4c69-988e-ac10ddafd302-0', usage_metadata={'input_tokens': 156, 'output_tokens': 15, 'total_tokens': 171, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}})]}, next=(), config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1f01f512-4e60-60f6-8003-72078ff94ccf'}}, metadata={'source': 'loop', 'writes': {'assistant': {'messages': [AIMessage(content='The result of multiplying 2 and 3 is 6.', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 15, 'prompt_tokens': 156, 'total_tokens': 171, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-2024-08-06', 'system_fingerprint': 'fp_a6889ffe71', 'id': 'chatcmpl-BP2zFO9RO1aX78vPbCLh4hHdtnP6R', 'finish_reason': 'stop', 'logprobs': None}, id='run-0e7d02dd-a762-4c69-988e-ac10ddafd302-0', usage_metadata={'input_tokens': 156, 'output_tokens': 15, 'total_tokens': 171, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}})]}}, 'step': 3, 'parents': {}, 'thread_id': '1'}, created_at='2025-04-22T08:09:42.035035+00:00', parent_config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1f01f512-48bc-633e-8002-a4ceef295474'}}, tasks=())
```

에이전트의 상태 히스토리도 조회할 수 있습니다.

`get_state_history`를 사용하면 이전 모든 단계의 상태를 가져올 수 있습니다.

```python
all_states = [s for s in graph.get_state_history(thread)]
```

```python
len(all_states)
```

```
# 출력

5
```

첫 번째 요소는 `get_state`로 확인한 것과 동일한 현재 상태입니다.

```python
all_states[-2] # HumanMessage - ToolMessage - AIMessage 에서 HumanMessage 선택
```

```
# 출력 - 첫 번째 요소와 동일한 HumanMessage 상태 가져옴, AI 응답을 생성하기 전에 상태를 가져옴

StateSnapshot(values={'messages': [HumanMessage(content='Multiply 2 and 3', additional_kwargs={}, response_metadata={}, id='bc428612-f369-477c-abdd-41683738fec8')]}, next=('assistant',), config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1f01f512-3eb5-603e-8000-7c26db62ead7'}}, metadata={'source': 'loop', 'writes': None, 'step': 0, 'parents': {}, 'thread_id': '1'}, created_at='2025-04-22T08:09:40.392137+00:00', parent_config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1f01f512-3eb3-6068-bfff-fefe64a409f3'}}, tasks=(PregelTask(id='2c9be69d-ca06-f9a4-c182-1288454a36de', name='assistant', path=('__pregel_pull', 'assistant'), error=None, interrupts=(), state=None, result={'messages': [AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_vk4Szqm7kfXZcQIVV0Fyf7JO', 'function': {'arguments': '{"a":2,"b":3}', 'name': 'multiply'}, 'type': 'function'}], 'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 18, 'prompt_tokens': 131, 'total_tokens': 149, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-2024-08-06', 'system_fingerprint': 'fp_a6889ffe71', 'id': 'chatcmpl-BP2zEr4CHtljWeys85hlx2zhD3fyc', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-19deb1ab-3b6b-45a3-aab6-036a7f129a62-0', tool_calls=[{'name': 'multiply', 'args': {'a': 2, 'b': 3}, 'id': 'call_vk4Szqm7kfXZcQIVV0Fyf7JO', 'type': 'tool_call'}], usage_metadata={'input_tokens': 131, 'output_tokens': 18, 'total_tokens': 149, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}})]}),))
```

위에서 설명한 내용을 시각화하면 [상태 체크포인트 흐름](#1--4--사용자-승인-브레이크포인트-설정)의 이미지에서 `Graph.get_state_history()`까지 흐름과 같습니다.

### 4.  2.  재실행 (Replaying)

`재실행`을 통해 이전 실행한 단계 어느 지점이든 에이전트를 다시 실행할 수 있습니다.

먼저 사용자 입력을 받은 단계를 실행합니다.

```python
to_replay = all_states[-2]
```

```python
to_replay
```

```
# 출력

StateSnapshot(values={'messages': [HumanMessage(content='Multiply 2 and 3', additional_kwargs={}, response_metadata={}, id='bc428612-f369-477c-abdd-41683738fec8')]}, next=('assistant',), config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1f01f512-3eb5-603e-8000-7c26db62ead7'}}, metadata={'source': 'loop', 'writes': None, 'step': 0, 'parents': {}, 'thread_id': '1'}, created_at='2025-04-22T08:09:40.392137+00:00', parent_config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1f01f512-3eb3-6068-bfff-fefe64a409f3'}}, tasks=(PregelTask(id='2c9be69d-ca06-f9a4-c182-1288454a36de', name='assistant', path=('__pregel_pull', 'assistant'), error=None, interrupts=(), state=None, result={'messages': [AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_vk4Szqm7kfXZcQIVV0Fyf7JO', 'function': {'arguments': '{"a":2,"b":3}', 'name': 'multiply'}, 'type': 'function'}], 'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 18, 'prompt_tokens': 131, 'total_tokens': 149, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-2024-08-06', 'system_fingerprint': 'fp_a6889ffe71', 'id': 'chatcmpl-BP2zEr4CHtljWeys85hlx2zhD3fyc', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-19deb1ab-3b6b-45a3-aab6-036a7f129a62-0', tool_calls=[{'name': 'multiply', 'args': {'a': 2, 'b': 3}, 'id': 'call_vk4Szqm7kfXZcQIVV0Fyf7JO', 'type': 'tool_call'}], usage_metadata={'input_tokens': 131, 'output_tokens': 18, 'total_tokens': 149, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}})]}),))
```

```python
to_replay.values
```

```
# 출력

{'messages': [HumanMessage(content='Multiply 2 and 3', additional_kwargs={}, response_metadata={}, id='bc428612-f369-477c-abdd-41683738fec8')]}
```

다음에 호출할 노드를 확인할 수 있습니다.

```python
to_replay.next
```

```
# 출력

('assistant',)
```

또한 `checkpoint_id`와 `thread_id`가 포함된 config 정보도 확인할 수 있습니다.

```python
to_replay.config
```

```
# 출력

{'configurable': {'thread_id': '1',
  'checkpoint_ns': '',
  'checkpoint_id': '1f01f512-3eb5-603e-8000-7c26db62ead7'}}
```

여기서부터 다시 실행하려면, `config`를 에이전트에 그대로 전달하면 됩니다.
그래프는 이 체크포인트가 `checkpoint_id`를 확인했기 때문에 이미 실행되었음을 알고 있습니다.
그래서 이 체크포인트부터 다시 실행만 합니다.

```python
# 재실행을 위해 checkpoint_id가 있는 config를 에이전트에 전달 

for event in graph.stream(None, to_replay.config, stream_mode="values"):
    event['messages'][-1].pretty_print()
```

```
# 출력

================================ Human Message =================================

Multiply 2 and 3
================================== Ai Message ==================================
Tool Calls:
  multiply (call_IGmLCLRgJrafOtl15T2dTuyN)
 Call ID: call_IGmLCLRgJrafOtl15T2dTuyN
  Args:
    a: 2
    b: 3
================================= Tool Message =================================
Name: multiply

6
================================== Ai Message ==================================

The result of multiplying 2 and 3 is 6.
```

에이전트가 `재실행`된 후의 상태를 확인할 수 있습니다.

### 4.  3.  분기 (Forking)

만약 같은 단계에서, `다른 입력`으로 실행하고 싶다면 어떻게 할까요?

이것을 가능하게 하는 것이 `분기`입니다.

![분기 흐름](assets/posts/2025-05-28-langgraph-ux-and-human-in-the-loop/forking_01.png)
_분기 흐름_

```python
to_fork = all_states[-2]
to_fork.values["messages"]
```

```python
# 결과

[HumanMessage(content='Multiply 2 and 3', additional_kwargs={}, response_metadata={}, id='bc428612-f369-477c-abdd-41683738fec8')]
```

아래와 같은 `분기`를 위한 `config`가 있습니다.

```python
to_fork.config
```

```python
# 출력

{'configurable': {'thread_id': '1',
  'checkpoint_ns': '',
  'checkpoint_id': '1f01f512-3eb5-603e-8000-7c26db62ead7'}}
```

이 체크포인트에서 상태를 수정합니다.

`checkpoint_id`를 제공하여 `update_state`만 실행하면 됩니다.

`messages`에서 리듀서가 어떻게 동작하는지 [이전의 내용](/posts/langgraph-state-and-memory-1st/#2-상태-리듀서-state-reducer)을 다시 살펴보겠습니다.

* 메시지 ID를 제공하지 않으면, 기존 메시지에 추가됩니다.
* 메시지 ID를 제공하면, 상태에 메시지를 추가하는 대신 덮어쓰기가 됩니다

따라서 메시지를 덮어쓰려면, 우리가 가지고 있는 `to_fork.values["messages"].id`와 같이 `메시지 ID` 를 함께 전달해 주면 됩니다.

```python
fork_config = graph.update_state(
    to_fork.config,
    {"messages": [HumanMessage(content='Multiply 5 and 3', 
                               id=to_fork.values["messages"][0].id)]},
)
```

```python
fork_config
```

```python
# 출력

{'configurable': {'thread_id': '1',
  'checkpoint_ns': '',
  'checkpoint_id': '1f01f51d-c11a-63b6-8001-4850605c9e65'}}

```

이렇게 하면 새로운 `분기된 체크포인트`가 생성됩니다.

하지만, 메타데이터(예: 다음에 이동할 위치 등)는 그대로 유지됩니다.

분기로 인해 에이전트의 현재 상태가 업데이트된 것을 확인할 수 있습니다.

```python
all_states = [state for state in graph.get_state_history(thread) ]
all_states[0].values["messages"]
```

```
# 출력

[HumanMessage(content='Multiply 5 and 3', id='4ee8c440-0e4a-47d7-852f-06e2a6c4f84d')]
```

```python
graph.get_state({'configurable': {'thread_id': '1'}})
```

```
# 출력

StateSnapshot(values={'messages': [HumanMessage(content='Multiply 5 and 3', additional_kwargs={}, response_metadata={}, id='bc428612-f369-477c-abdd-41683738fec8')]}, next=('assistant',), config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1f01f51d-c11a-63b6-8001-4850605c9e65'}}, metadata={'source': 'update', 'writes': {'__start__': {'messages': [HumanMessage(content='Multiply 5 and 3', additional_kwargs={}, response_metadata={}, id='bc428612-f369-477c-abdd-41683738fec8')]}}, 'step': 1, 'parents': {}, 'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1f01f512-3eb5-603e-8000-7c26db62ead7'}, created_at='2025-04-22T08:14:49.344082+00:00', parent_config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1f01f512-3eb5-603e-8000-7c26db62ead7'}}, tasks=(PregelTask(id='b9775a1d-dd76-9b09-6bf7-e0f39e362bb9', name='assistant', path=('__pregel_pull', 'assistant'), error=None, interrupts=(), state=None, result=None),))
```

이제 스트리밍을 하면, 그래프는 이 체크포인트가 한 번도 실행된 적이 없다는 것을 인식합니다.

그래서 그래프는 단순히 재실행하는 것이 아니라, 실제로 실행합니다.

```python
for event in graph.stream(None, fork_config, stream_mode="values"):
    event['messages'][-1].pretty_print()
```

```
# 출력

================================ Human Message =================================

Multiply 5 and 3
================================== Ai Message ==================================
Tool Calls:
  multiply (call_EO0yzh79VkED7Cx7gDkBGnmL)
 Call ID: call_EO0yzh79VkED7Cx7gDkBGnmL
  Args:
    a: 5
    b: 3
================================= Tool Message =================================
Name: multiply

15
================================== Ai Message ==================================

The product of 5 and 3 is 15.
```

이제, 현재 상태가 에이전트 실행의 마지막임을 확인할 수 있습니다.

```python
graph.get_state({'configurable': {'thread_id': '1'}})
```

```
# 출력

StateSnapshot(values={'messages': [HumanMessage(content='Multiply 5 and 3', additional_kwargs={}, response_metadata={}, id='bc428612-f369-477c-abdd-41683738fec8'), AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_EO0yzh79VkED7Cx7gDkBGnmL', 'function': {'arguments': '{"a":5,"b":3}', 'name': 'multiply'}, 'type': 'function'}], 'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 18, 'prompt_tokens': 131, 'total_tokens': 149, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-2024-08-06', 'system_fingerprint': 'fp_a6889ffe71', 'id': 'chatcmpl-BP35usXpIo5A1OL64rmvVkM5SFCAx', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-d6802ef4-de71-4043-afcf-eae6b82d40a8-0', tool_calls=[{'name': 'multiply', 'args': {'a': 5, 'b': 3}, 'id': 'call_EO0yzh79VkED7Cx7gDkBGnmL', 'type': 'tool_call'}], usage_metadata={'input_tokens': 131, 'output_tokens': 18, 'total_tokens': 149, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}}), ToolMessage(content='15', name='multiply', id='286a2a6c-22fd-46d9-bdc7-c9b080bafb57', tool_call_id='call_EO0yzh79VkED7Cx7gDkBGnmL'), AIMessage(content='The product of 5 and 3 is 15.', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 14, 'prompt_tokens': 156, 'total_tokens': 170, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-2024-08-06', 'system_fingerprint': 'fp_a6889ffe71', 'id': 'chatcmpl-BP35uuJF3x4ADqcro1mfGzJsl4EHR', 'finish_reason': 'stop', 'logprobs': None}, id='run-7635bf21-e0bc-46fd-8725-ba8ca9f0fbc6-0', usage_metadata={'input_tokens': 156, 'output_tokens': 14, 'total_tokens': 170, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}})]}, next=(), config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1f01f521-b259-68e6-8004-02d9a1fe5cbb'}}, metadata={'source': 'loop', 'writes': {'assistant': {'messages': [AIMessage(content='The product of 5 and 3 is 15.', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 14, 'prompt_tokens': 156, 'total_tokens': 170, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-2024-08-06', 'system_fingerprint': 'fp_a6889ffe71', 'id': 'chatcmpl-BP35uuJF3x4ADqcro1mfGzJsl4EHR', 'finish_reason': 'stop', 'logprobs': None}, id='run-7635bf21-e0bc-46fd-8725-ba8ca9f0fbc6-0', usage_metadata={'input_tokens': 156, 'output_tokens': 14, 'total_tokens': 170, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}})]}}, 'step': 4, 'parents': {}, 'thread_id': '1'}, created_at='2025-04-22T08:16:35.171326+00:00', parent_config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1f01f521-ac96-6b66-8003-bb860f505c3c'}}, tasks=())
```

## 정리
실시간으로 그래프 상태를 확인할 수 있는 `스트리밍`은 `.stream(동기)`과 `.astream(비동기)`울 통해 출력을 스트리밍하는 방법에 대해서 살펴보았습니다.

`스트리밍 모드`에서 `전체 상태`를 스트리밍하는 `values`, `변경이 생긴 부분`만 스트리밍하는 `updates`를 확인했습니다.
그리고 `.astream_events`를 활용해 노드 내에서 발생하는 이벤트를 `실시간으로 스트리밍`할 수 있었습니다.

`브레이크포인트`를 사용해 그래프 실행 도중 특정 단계에서 `일시 정지` 후 `그래프 상태를 수정`할 수 있었고 이를 `사용자가 승인`할 수 있었습니다.

마지막으로 `타임트래블`에서 `get_state`로 `이전 상태를 조회`가 가능하고, `checkpoint_id`와 `thread_id`가 포함된 `config`를 에이전트에 전달해 `재실행`을 하였고 분기를 위한 `checkpoint_id`가 포함된 `config`를 제공하여 `graph.update_state` 실행하여 `분기`하는 것도 확인했습니다.

`브레이크포인트`, `타임트래블` 기능을 활용하면 디버깅 및 `휴먼-인-더-루프`가 필요한 엔터프라이즈 환경에서도 유연하게 그래프 실행을 관리할 수 있습니다

다음 포스팅에서 `멀티 에이전트 워크플로(multi-agent workflows)`에 대해서 알아보겠습니다.

## References

* [LangGraph 공식 문서](https://langchain-ai.github.io/langgraph){: target="_blank"}
* [LangChain Academy](https://github.com/langchain-ai/langchain-academy){: target="_blank"}
