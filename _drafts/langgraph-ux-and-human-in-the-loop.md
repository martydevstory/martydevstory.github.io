---
title: LangGraph 사용자경험(UX)과 휴먼-인-더-루프
date: 2025-05-11 11:15:43 +/-TTTT
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
series_order: 5
---
> 학습할 리소스는 [LangChain Academy Github](https://github.com/langchain-ai/langchain-academy){: target="_blank"}를 사용합니다.
> {: .prompt-info }

이전 포스팅에서 `그래프 상태와 메모리`를 커스터마이징하는 방법과 장시간 대화를 유지할 수 있는 `외부 메모리 기반 챗봇`을 만들었습니다.

메모리 기능을 기반으로 사용자가 다양한 그래프와 다양한 방식으로 직접 상호 작용할 수 있도록 하는 `휴먼-인-더-루프(human-in-the-loop)`를 살펴보겠습니다.

먼저, 그래프 실행 과정에서 출력(예: 노드 상태 또는 채팅 모델의 토큰 등)을 여러 방법으로 시각화할 수 있는 `스트리밍`을 살펴보겠습니다.

```python
# 환경 구성
%%capture --no-stderr
%pip install --quiet -U langgraph langchain_openai langgraph_sdk
```

## 1.   스트리밍 (Streaming)

LangGraph는 스트리밍 기능을 핵심적으로 지원하도록 설계되었습니다.

이전 포스팅에서 만든 챗봇을 다시 구성하고, 그래프 실행중 출력을 스트리밍하는 다양한 방법을 살펴보겠습니다.

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

### 전체 상태 스트리밍 (Streaming full state)

이제 [그래프 상태를 스트리밍하는 방법](https://langchain-ai.github.io/langgraph/concepts/low_level/#streaming)에 대해 알아보겠습니다.

`.stream`과 `.astream`은 각각 동기(sync) 및 비동기(async) 방식으로 결과를 스트리밍하는 메서드입니다.

LangGraph는 [그래프 상태](https://langchain-ai.github.io/langgraph/how-tos/stream-values/)에 대해 몇 가지 [다양한 스트리밍 모드](https://langchain-ai.github.io/langgraph/how-tos/stream-values/)를 지원합니다.

* `values`: 각 노드가 실행된 후 그래프의 `전체 상태`를 스트리밍합니다.
* `updates`: 각 노드가 실행된 후 그래프 상태에 `변경이 생긴 부분만` 스트리밍합니다.

[이미지 첨부]

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

### 토큰 스트리밍 (Streaming tokens)

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

핵심은, 그래프 내에서 생성된 챗 모델의 토큰들은 `on_chat_model_stream` 타입의 이벤트로 전달된다는 점입니다.

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

### 중단점 (Breakpoints)



ㅇ

## References

* [LangGraph 공식 문서](https://langchain-ai.github.io/langgraph){: target="_blank"}
* [LangChain Academy](https://github.com/langchain-ai/langchain-academy){: target="_blank"}
