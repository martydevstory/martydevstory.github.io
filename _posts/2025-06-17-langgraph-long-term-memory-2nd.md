---
title: LangGraph 장기 메모리 (2)
date: 2025-06-17 10:15:43 +/-TTTT
description : LangGraph 메모리 컬렉션에 대해서 더 자세히 알아보고 시맨틱 메모리 기반 에이전트을 구축해 보겠습니다.
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
series_order: 9
---
챗봇의 기능을 확장해서 문자열 저장에서 메모리가 구조를 가질 수 있게 `시맨틱 메모리`를 단일 [사용자 프로필](https://langchain-ai.github.io/langgraph/concepts/memory/#profile){: target="_blank"}에 저장하도록 만들었습니다.

또한 해당 스키마를 새로운 정보로 업데이트하기 위해 [Trustcall](https://github.com/hinthornw/trustcall){: target="_blank"} 라이브러리를 살펴보았습니다.

이번 포스팅에서는 `컬렉션(Collection)`에 대해서 더 자세히 알아보고 `시맨틱 메모리 기반 에이전트`을 구축해 보겠습니다.

> 학습할 리소스는 [LangChain Academy Github](https://github.com/langchain-ai/langchain-academy){: target="_blank"}를 사용합니다.
{: .prompt-info }

## 3.   컬렉션 스키마 기반 챗봇 구축

때때로 메모리를 `단일 프로필` 대신 [컬렉션](https://docs.google.com/presentation/d/181mvjlgsnxudQI6S3ritg9sooNyu4AcLLFH1UK0kIuk/edit#slide=id.g30eb3c8cf10_0_200){: target="_blank"}에 저장하는 것이 더 적합할 때가 있습니다.

챗봇이 [컬렉션에 메모리를 저장](https://langchain-ai.github.io/langgraph/concepts/memory/#collection){: target="_blank"}하도록 업데이트하고 [Trustcall](https://github.com/hinthornw/trustcall){: target="_blank"}을 사용하여 `컬렉션`을 업데이트하는 방법도 살펴보겠습니다.

```python
# 환경 구성
%%capture --no-stderr
%pip install -U langchain_openai langgraph trustcall langchain_core
```

```python
import os, getpass

def _set_env(var: str):
    # OS 환경 변수에 해당 값이 설정되어 있는지 확인
    env_value = os.environ.get(var)
    if not env_value:
        # 값이 없으면 사용자에게 입력을 요청
        env_value = getpass.getpass(f"{var}: ")
  
    # 현재 프로세스의 환경 변수로 설정
    os.environ[var] = env_value

_set_env("LANGSMITH_API_KEY")
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "langchain-academy"
```

### 3.  1.  컬렉션 스키마 정의하기

사용자 정보를 고정된 `프로필` 구조로 저장하는 대신, 사용자 상호작용에 대한 메모리로 저장하기 위한 유연한 `컬렉션` 스키마를 생성합니다.

저장된 각 메모리는 기억하고자 하는 주요 정보를 담고 있는 단일 `content` 필드를 가지며, 개별 항목으로 저장됩니다.

이 방식을 통해 사용자에 대한 학습, 확장 및 변화가 가능한 개방형 메모리 컬렉션을 구축할 수 있습니다.

컬렉션 스키마는 [Pydantic](https://docs.pydantic.dev/latest/){: target="_blank"} 객체로 정의할 수 있습니다.

```python
from pydantic import BaseModel, Field

class Memory(BaseModel):
    content: str = Field(description="The main content of the memory. For example: User expressed interest in learning about French.")

class MemoryCollection(BaseModel):
    memories: list[Memory] = Field(description="A list of memories about the user.")
```

```python
_set_env("OPENAI_API_KEY")
```

엄격히 구조화된 출력을 위해 LangChain의 [채팅 모델](https://python.langchain.com/docs/concepts/chat_models/){: target="_blank"} 인터페이스에서 제공하는 [`with_structured_output`](https://python.langchain.com/docs/concepts/structured_outputs/#recommended-usage){: target="_blank"} 메서드를 제공합니다.

```python
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

# 모델 초기화
model = ChatOpenAI(model="gpt-4o", temperature=0)

# 모델에 스키마를 바인딩
model_with_structure = model.with_structured_output(MemoryCollection)

# 모델을 호출하여 스키마에 맞는 구조화된 출력 생성
memory_collection = model_with_structure.invoke([HumanMessage("My name is Lance. I like to bike.")])
memory_collection.memories
```

```
# 출력

[Memory(content="User's name is Lance."),
 Memory(content='User likes to bike.')]
```

`Pydantic` 모델 인스턴스를 파이썬 딕셔너리로 직렬화하기 위해 `model_dump()`를 사용합니다.

```python
memory_collection.memories[0].model_dump()
```

```
# 출력
{'content': "User's name is Lance."}
```

그리고, 각 메모리의 딕셔너리를 스토어에 저장합니다.

```python
import uuid
from langgraph.store.memory import InMemoryStore

# 인-메모리 스토어 초기화
in_memory_store = InMemoryStore()

# 메모리를 저장할 네임스페이스 지정
user_id = "1"
namespace_for_memory = (user_id, "memories")

# 키,값으로 네임스페이스에 저장
key = str(uuid.uuid4())
value = memory_collection.memories[0].model_dump()
in_memory_store.put(namespace_for_memory, key, value)

key = str(uuid.uuid4())
value = memory_collection.memories[1].model_dump()
in_memory_store.put(namespace_for_memory, key, value)
```

스토어에서 메모리를 검색합니다.

```python
# 검색 
for m in in_memory_store.search(namespace_for_memory):
    print(m.dict())
```

```
# 출력

{'namespace': ['1', 'memories'], 'key': '4e750e5f-225b-4cb2-bd13-e1eed6d4a9e3', 'value': {'content': "User's name is Lance."}, 'created_at': '2025-04-23T07:31:37.850796+00:00', 'updated_at': '2025-04-23T07:31:37.850800+00:00', 'score': None}
{'namespace': ['1', 'memories'], 'key': '447a6175-6c30-443e-9b51-9b12c3768351', 'value': {'content': 'User likes to bike.'}, 'created_at': '2025-04-23T07:31:37.850910+00:00', 'updated_at': '2025-04-23T07:31:37.850911+00:00', 'score': None}
```

### 3.  2.  컬렉션 스키마 업데이트

이전 포스팅에서 `프로필` 스키마를 업데이트할 때 매번 비효율적으로 다시 생성하는 것에 대한 문제 해결의 대안으로 [Trustcall](https://github.com/hinthornw/trustcall){: target="_blank"}을 설명했습니다.

컬렉션도 마찬가지로 새로운 메모리 추가와 [기존 메모리 업데이트](https://github.com/hinthornw/trustcall?tab=readme-ov-file#simultanous-updates--insertions){: target="_blank"}를 위해 `Trustcall`을 사용하는 방법을 알아보겠습니다.

우선 `Trustcall`을 사용하여 새로운 extractor를 정의하고 이전과 마찬가지로 각 메모리의 스키마인 `Memory`를 제공합니다.

또한, 새 메모리를 컬렉션에 삽입할 수 있도록 `enable_inserts=True` 옵션을 추가할 수 있습니다.

```python
from trustcall import create_extractor

# extractor 생성
trustcall_extractor = create_extractor(
    model,
    tools=[Memory],
    tool_choice="Memory",
    enable_inserts=True, # 새 메모리 삽입
)
```

```python
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# 지침
instruction = """Extract memories from the following conversation:"""

# 대화
conversation = [HumanMessage(content="Hi, I'm Lance."), 
                AIMessage(content="Nice to meet you, Lance."), 
                HumanMessage(content="This morning I had a nice bike ride in San Francisco.")]

# extractor 실행
result = trustcall_extractor.invoke({"messages": [SystemMessage(content=instruction)] + conversation})
```

```python
# 메시지에는 도구 호출이 포함
for m in result["messages"]:
    m.pretty_print()
```

```
# 출력

================================== Ai Message ==================================
Tool Calls:
  Memory (call_GEEkOwDzbrzXvpScifyfYRdy)
 Call ID: call_GEEkOwDzbrzXvpScifyfYRdy
  Args:
    content: Lance had a nice bike ride in San Francisco this morning.
```

```python
# 응답은 스키마에 부합하는 memory가 포함
for m in result["responses"]: 
    print(m)
```

```
# 출력

content='Lance had a nice bike ride in San Francisco this morning.'
```

```python
# 메타데이터에는 도구 호출이 포함
for m in result["response_metadata"]: 
    print(m)
```

```
# 출력

{'id': 'call_GEEkOwDzbrzXvpScifyfYRdy'}
```

```python
# 대화 업데이트
updated_conversation = [AIMessage(content="That's great, did you do after?"), 
                        HumanMessage(content="I went to Tartine and ate a croissant."),                
                        AIMessage(content="What else is on your mind?"),
                        HumanMessage(content="I was thinking about my Japan, and going back this winter!"),]

# 지침 업데이트
system_msg = """Update existing memories and create new ones based on the following conversation:"""

# 기존 메모리를 저장하고 ID, 키(도구 이름), 값을 지정
tool_name = "Memory"
existing_memories = [(str(i), tool_name, memory.model_dump()) for i, memory in enumerate(result["responses"])] if result["responses"] else None
existing_memories
```

```
# 출력

[('0',
  'Memory',
  {'content': 'Lance had a nice bike ride in San Francisco this morning.'})]
```

```python
# 업데이트된 대화와 기존 메모리를 사용하여 extractor를 호출
result = trustcall_extractor.invoke({"messages": updated_conversation, 
                                     "existing": existing_memories})
```

```python
# 모델의 메시지에서 두 개의 도구 호출이 생성됨을 보여줌
for m in result["messages"]:
    m.pretty_print()
```

```
# 출력

================================== Ai Message ==================================
Tool Calls:
  Memory (call_zDisQAPTNekKnxgGxv3HTYja)
 Call ID: call_zDisQAPTNekKnxgGxv3HTYja
  Args:
    content: Lance had a nice bike ride in San Francisco this morning. Then, he went to Tartine and ate a croissant.
  Memory (call_H4RcbWByX6VVDIRUODHTJelH)
 Call ID: call_H4RcbWByX6VVDIRUODHTJelH
  Args:
    content: I was thinking about my trip to Japan, and going back this winter!
```

```python
# 응답은 스키마에 부합하는 memory가 포함
for m in result["responses"]: 
    print(m)
```

```
# 출력

content='Lance had a nice bike ride in San Francisco this morning. Then, he went to Tartine and ate a croissant.'
content='I was thinking about my trip to Japan, and going back this winter!'
```

이것은 우리가 `json_doc_id`를 지정함으로써 컬렉션의 첫 번째 메모리를 업데이트했음을 나타냅니다.

```python
# 메타데이터에는 도구 호출이 포함
for m in result["response_metadata"]: 
    print(m)
```

```
# 출력

{'id': 'call_zDisQAPTNekKnxgGxv3HTYja', 'json_doc_id': '0'}
{'id': 'call_H4RcbWByX6VVDIRUODHTJelH'}
```

### 3.  3.  컬렉션 스키마 업데이트 기반 챗봇

이제 Trustcall을 챗봇에 통합하여 메모리 컬렉션을 생성하고 업데이트합니다.

```python
from IPython.display import Image, display

import uuid

from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.store.memory import InMemoryStore
from langchain_core.messages import merge_message_runs
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables.config import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.base import BaseStore

# 모델 초기화
model = ChatOpenAI(model="gpt-4o", temperature=0)

# Memory 스키마
class Memory(BaseModel):
    content: str = Field(description="The main content of the memory. For example: User expressed interest in learning about French.")

# Trustcall extractor 생성
trustcall_extractor = create_extractor(
    model,
    tools=[Memory],
    tool_choice="Memory",
    # This allows the extractor to insert new memories
    enable_inserts=True,
)

# 챗봇 지침
MODEL_SYSTEM_MESSAGE = """You are a helpful chatbot. You are designed to be a companion to a user. 

You have a long term memory which keeps track of information you learn about the user over time.

Current Memory (may include updated memories from this conversation): 

{memory}"""

# Trustcall 지침
TRUSTCALL_INSTRUCTION = """Reflect on following interaction. 

Use the provided tools to retain any necessary memories about the user. 

Use parallel tool calling to handle updates and insertions simultaneously:"""

def call_model(state: MessagesState, config: RunnableConfig, store: BaseStore):

    """Store에서 Memory를 불러와 챗봇 응답에서 개인화에 활용"""
  
    # 사용자 ID를 config에서 가져옴
    user_id = config["configurable"]["user_id"]

    # 프로필 메모리를 Store에서 검색
    namespace = ("memories", user_id)
    memories = store.search(namespace)

    # 시스템 프롬프트에 메모리를 포맷
    info = "\n".join(f"- {mem.value['content']}" for mem in memories)
    system_msg = MODEL_SYSTEM_MESSAGE.format(memory=info)

    # 메모리와 대화 기록을 사용하여 응답
    response = model.invoke([SystemMessage(content=system_msg)]+state["messages"])

    return {"messages": response}

def write_memory(state: MessagesState, config: RunnableConfig, store: BaseStore):

    """대화 내역을 반영하여 메모리를 스토어에 저장"""
  
    # config에서 사용자 ID를 가져옴
    user_id = config["configurable"]["user_id"]

    # 메모리의 네임스페이스 정의
    namespace = ("memories", user_id)

    # 최신 메모리 가져오기
    existing_items = store.search(namespace)

    # Trustcall extractor에 사용할 기존 기억 포맷팅
    tool_name = "Memory"
    existing_memories = ([(existing_item.key, tool_name, existing_item.value)
                          for existing_item in existing_items]
                          if existing_items
                          else None
                        )

    # 대화 기록과 지침을 병합
    updated_messages=list(merge_message_runs(messages=[SystemMessage(content=TRUSTCALL_INSTRUCTION)] + state["messages"]))

    # extractor 실행
    result = trustcall_extractor.invoke({"messages": updated_messages, 
                                        "existing": existing_memories})

    # Trustcall 메모리를 스토어에 저장
    for r, rmeta in zip(result["responses"], result["response_metadata"]):
        store.put(namespace,
                  rmeta.get("json_doc_id", str(uuid.uuid4())),
                  r.model_dump(mode="json"),
            )

# 그래프 정의
builder = StateGraph(MessagesState)
builder.add_node("call_model", call_model)
builder.add_node("write_memory", write_memory)
builder.add_edge(START, "call_model")
builder.add_edge("call_model", "write_memory")
builder.add_edge("write_memory", END)

# 장기(스레드 간) 메모리용 스토어
across_thread_memory = InMemoryStore()

# 단기(스레드 내) 메모리용 체크포인터
within_thread_memory = MemorySaver()

# 체크포인터와 스토어로 그래프 컴파일
graph = builder.compile(checkpointer=within_thread_memory, store=across_thread_memory)

# 그래프 이미지
display(Image(graph.get_graph(xray=1).draw_mermaid_png()))
```

```python
# 단기(스레드 내) 메모리에는 thread ID를 지정
# 장기(스레드 간) 메모리에는 user ID를 지정
config = {"configurable": {"thread_id": "1", "user_id": "1"}}

# 사용자 입력 메시지
input_messages = [HumanMessage(content="Hi, my name is Lance")]

# 그래프 실행
for chunk in graph.stream({"messages": input_messages}, config, stream_mode="values"):
    chunk["messages"][-1].pretty_print()
```

```
# 출력

================================ Human Message =================================

Hi, my name is Lance
================================== Ai Message ==================================

Hi Lance! It's great to meet you. How can I assist you today?
```

```python
# 사용자 입력 메시지
input_messages = [HumanMessage(content="I like to bike around San Francisco")]

# 그래프 실행
for chunk in graph.stream({"messages": input_messages}, config, stream_mode="values"):
    chunk["messages"][-1].pretty_print()
```

```
# 출력

================================ Human Message =================================

I like to bike around San Francisco
================================== Ai Message ==================================

That sounds like a lot of fun! San Francisco has some beautiful routes for biking. Do you have a favorite trail or area you like to explore?
```

```python
# 메모리 저장을 위한 네임스페이스 지정
user_id = "1"
namespace = ("memories", user_id)
memories = across_thread_memory.search(namespace)
for m in memories:
    print(m.dict())
```

```
# 출력

{'namespace': ['memories', '1'], 'key': 'fc48c426-85c5-469e-a000-5392d14fbbae', 'value': {'content': 'User likes to bike around San Francisco.'}, 'created_at': '2025-04-23T07:37:02.779130+00:00', 'updated_at': '2025-04-23T07:37:02.779131+00:00', 'score': None}
{'namespace': ['memories', '1'], 'key': '1ee46a63-ce12-421e-8fb7-3773d1f57132', 'value': {'content': 'User likes to bike around San Francisco.'}, 'created_at': '2025-04-23T07:37:02.779105+00:00', 'updated_at': '2025-04-23T07:37:02.779106+00:00', 'score': None}
```

```python
# 사용자 입력 메시지
input_messages = [HumanMessage(content="I also enjoy going to bakeries")]

# 그래프 실행
for chunk in graph.stream({"messages": input_messages}, config, stream_mode="values"):
    chunk["messages"][-1].pretty_print()
```

```
# 출력

================================ Human Message =================================

I also enjoy going to bakeries
================================== Ai Message ==================================

Biking and bakeries make a great combination! Do you have a favorite bakery in San Francisco, or are you on the lookout for new ones to try?
```

이전 프로필에서와 마찬가지로 새로운 스레드에서도 컨텍스트를 유지할 수 있습니다.

```python
# 단기(스레드 내) 메모리에는 thread ID를 지정
# 장기(스레드 간) 메모리에는 user ID를 지정
config = {"configurable": {"thread_id": "2", "user_id": "1"}}

# 사용자 입력 메시지
input_messages = [HumanMessage(content="What bakeries do you recommend for me?")]

# 그래프 실행
for chunk in graph.stream({"messages": input_messages}, config, stream_mode="values"):
    chunk["messages"][-1].pretty_print()
```

```
# 출력

================================ Human Message =================================

What bakeries do you recommend for me?
================================== Ai Message ==================================

Since you enjoy biking around San Francisco, you might like to visit some local bakeries that are perfect for a quick stop during your rides. Here are a few recommendations:

1. **Tartine Bakery** - Located in the Mission District, it's famous for its bread and pastries. It's a great spot to grab a morning bun or a croissant.

2. **Arizmendi Bakery** - This worker-owned cooperative in the Inner Sunset offers delicious scones, muffins, and pizza. It's a cozy spot to take a break.

3. **B. Patisserie** - Situated in Lower Pacific Heights, this bakery is known for its kouign-amann and other French pastries. It's a bit of a treat after a long ride.

4. **Mr. Holmes Bakehouse** - In the Tenderloin, this bakery is famous for its cruffins and other inventive pastries. It's a fun place to try something new.

5. **Noe Valley Bakery** - A neighborhood favorite in Noe Valley, offering a variety of classic and seasonal pastries.

These spots are not only delicious but also scattered around the city, giving you a chance to explore different neighborhoods on your bike. Enjoy your rides and treats!
```

## 4.   시맨틱 메모리 기반 에이전트 구축하기

사용자 `프로필`과 `컬렉션` 기반으로 에이전트를 구축할 것입니다.

다시 살펴보면 `시맨틱 메모리`는 `프로필(Profile)`과 `컬렉션(Collection)`의 두 가지 방식으로 관리됩니다.

그리고 AI 에이전트에서는 사용자에 대한 정보(예: 이름, 직책, 선호도 등)를 기억하는 데 사용됩니다.

이는 에이전트가 사용자와의 상호작용을 통해 얻은 정보를 기반으로 응답을 개선하고 개인화하는 데 도움을 줍니다.

> 여기서 `시맨틱` 개념은 `시맨틱 검색(Semantic Search)`과 다릅니다. 시맨틱 검색은 `의미(meaning, 일반적으로 임베딩)`를 사용하여 유사한 콘텐츠를 찾는 기법입니다.
{: .prompt-warning }

장기 메모리 기반 에이전트를 구축하고 `프로필`과 `컬렉션`의 두 스키마를 업데이트하는 방법으로 `Trustcall`을 살펴보겠습니다.

먼저 환경 구성을 합니다.

```python
# 환경 구성
%%capture --no-stderr
%pip install -U langchain_openai langgraph trustcall langchain_core
```

```python
import os, getpass

def _set_env(var: str):
    # OS 환경 변수에 해당 값이 설정되어 있는지 확인
    env_value = os.environ.get(var)
    if not env_value:
        # 값이 없으면 사용자에게 입력을 요청
        env_value = getpass.getpass(f"{var}: ")
  
    # 현재 프로세스의 환경 변수로 설정
    os.environ[var] = env_value

_set_env("LANGSMITH_API_KEY")
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "langchain-academy"
```

```python
_set_env("OPENAI_API_KEY")
```

### 4.  1.  Trustcall 업데이트에 대한 가시성 (Visability)

이전 내용에서 `Trustcall`은 LangGraph 기반 오픈소스 라이브러리로, LLM이 복잡한 구조의 JSON 출력을 생성, 수정할 때 발생하는 오류를 줄입니다.

기존 방식은 전체 JSON을 한 번에 생성하려다 보니 오류가 발생하기 쉬웠습니다.

`Trustcall`은 이러한 문제를 해결하기 위해 LLM에 `JSON Patch` 형식의 수정 지시를 생성하도록 요청합니다.

`JSON Patch`는 부분적으로 수정할 수 있으며, 반복적인 오류 수정이 쉽습니다.

다음 항목에서 `Trustcall` 추적 예를 확인할 수 있습니다.

* [검증 실패 자체 정정](https://smith.langchain.com/public/5cd23009-3e05-4b00-99f0-c66ee3edd06e/r/9684db76-2003-443b-9aa2-9a9dbc5498b7){: target="_blank"}
* [기존 문서 업데이트](https://smith.langchain.com/public/f45bdaf0-6963-4c19-8ec9-f4b7fe0f68ad/r/760f90e1-a5dc-48f1-8c34-79d6a3414ac3){: target="_blank"}

> `JSON Patch`는 JSON 문서의 일부를 추가, 삭제, 교체 등을 통해 부분적으로 수정하는 표준 형식입니다.
> 공식 표준은 [RFC 6902](https://datatracker.ietf.org/doc/html/rfc6902){: target="_blank"}에 정의되어 있습니다.
{: .prompt-info }

이제 Memory 클래스와 컬렉션을 정의합니다.

```python
from pydantic import BaseModel, Field

class Memory(BaseModel):
    content: str = Field(description="The main content of the memory. For example: User expressed interest in learning about French.")

class MemoryCollection(BaseModel):
    memories: list[Memory] = Field(description="A list of memories about the user.")
```

`Spy`는 `Trustcall extractor`에 `리스너(listener)`로 등록되어, 실행이 종료될 때마다 호출됩니다.

이때 전달받은 `실행 정보(run)`를 통해 `Trustcall`이 어떤 `도구 호출(tool call)`을 했는지 추적합니다.

```python
from trustcall import create_extractor
from langchain_openai import ChatOpenAI

# Trustcall에서 도구 호출을 관찰
class Spy:
    def __init__(self):
        self.called_tools = []

    def __call__(self, run):
        # extractor가 수행한 도구 호출 정보를 수집
        q = [run]
        while q:
            r = q.pop()
            if r.child_runs:
                q.extend(r.child_runs)
            if r.run_type == "chat_model":
                self.called_tools.append(
                    r.outputs["generations"][0][0]["message"]["kwargs"]["tool_calls"]
                )

# spy 초기화
spy = Spy()

# 모델 초기화
model = ChatOpenAI(model="gpt-4o", temperature=0)

# extractor 생성
trustcall_extractor = create_extractor(
    model,
    tools=[Memory],
    tool_choice="Memory",
    enable_inserts=True,
)

# 리스너로 spy 등록
trustcall_extractor_see_all_tool_calls = trustcall_extractor.with_listeners(on_end=spy)
```

`trustcall_extractor`를 사용해 invoke로 LLM과 도구 호출 추출을 합니다.

```python
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# 지침
instruction = """Extract memories from the following conversation:"""

# 대화
conversation = [HumanMessage(content="Hi, I'm Lance."), 
                AIMessage(content="Nice to meet you, Lance."), 
                HumanMessage(content="This morning I had a nice bike ride in San Francisco.")]

# extractor 실행
result = trustcall_extractor.invoke({"messages": [SystemMessage(content=instruction)] + conversation})
```

```python
# messages에 도구 호출이 포함
for m in result["messages"]:
    m.pretty_print()
```

`Memory` 도구 호출이 되고 도구 호출에 전달된 `인자`를 확인할 수 있습니다.

```
# 출력

================================== Ai Message ==================================
Tool Calls:
  Memory (call_b38FmEszmh8B8xSlRjj4zzx7)
 Call ID: call_b38FmEszmh8B8xSlRjj4zzx7
  Args:
    content: Lance had a nice bike ride in San Francisco this morning.
```

Memory pydantic 클래스로 출력합니다.

```python
# 응답은 스키마에 부합하는 memory가 포함
for m in result["responses"]: 
    print(m)
```

```
# 출력

content='Lance had a nice bike ride in San Francisco this morning.'
```

`response_metadata`에는 각 도구 호출에 대한 `메타데이터`가 있습니다.

```python
# 메타데이터에는 도구 호출이 포함
for m in result["response_metadata"]: 
    print(m)
```

```
# 출력

{'id': 'call_b38FmEszmh8B8xSlRjj4zzx7'}
```

기존 Memory를 기억하고 새 Memory를 업데이트합니다.

```python
# 대화 업데이트
updated_conversation = [AIMessage(content="That's great, did you do after?"), 
                        HumanMessage(content="I went to Tartine and ate a croissant."),  
                        AIMessage(content="What else is on your mind?"),
                        HumanMessage(content="I was thinking about my Japan, and going back this winter!"),]

# 지침 업데이트
system_msg = """Update existing memories and create new ones based on the following conversation:"""

# 기존 메모리를 저장하고 ID, 키(도구 이름), 값을 지정
tool_name = "Memory"
existing_memories = [(str(i), tool_name, memory.model_dump()) for i, memory in enumerate(result["responses"])] if result["responses"] else None
existing_memories
```

출력에서 보면 추후 업데이트 작업을 위해 기존 Memory를 ID, 도구, 값(딕셔너리) 형태로 관리합니다.

```
# 출력

[('0',
  'Memory',
  {'content': 'Lance had a nice bike ride in San Francisco this morning.'})]
```

새 대화와 함께 기존 Memory를 같이 전달하고 LLM과 도구 호출에 대해 `trustcall extractor`를 실행합니다.

```python
# 업데이트된 대화와 기존 메모리를 사용하여 extractor를 호출
result = trustcall_extractor_see_all_tool_calls.invoke({"messages": updated_conversation, 
                                                        "existing": existing_memories})
```

메타데이터를 호출합니다.

```python
# 메타데이터에는 도구 호출이 포함
for m in result["response_metadata"]: 
    print(m)
```

출력에서 보면 `json_doc_id`는 도구 호출이 기존 Memory('0')를 업데이트한 것임을 명시합니다.

```
# 출력

{'id': 'call_ldn7vwEsmCyD4SImztvdBHr4', 'json_doc_id': '0'}
{'id': 'call_NL6wRA5l2mbO1qWONANAJ0t5'}
{'id': 'call_5Oiko9oLbDIAWylRAnlAaGyf'}
```

```python
# 메세지에는 도구 호출이 포함
for m in result["messages"]:
    m.pretty_print()
```

출력을 살펴보면 `call_ldn7vwEsmCyD4SImztvdBHr4`는 `'json_doc_id' : '0'`이므로 기존 Memory(0번)을 업데이트합니다.

`call id`가 `call_NL6wRA5l2mbO1qWONANAJ0t5`이후 부터는 새로운 Memory입니다.

```
# 출력

================================== Ai Message ==================================
Tool Calls:
  Memory (call_ldn7vwEsmCyD4SImztvdBHr4)
 Call ID: call_ldn7vwEsmCyD4SImztvdBHr4
  Args:
    content: Lance had a nice bike ride in San Francisco this morning. Then, he went to Tartine and ate a croissant. He was also thinking about Japan and planning to go back this winter.
  Memory (call_NL6wRA5l2mbO1qWONANAJ0t5)
 Call ID: call_NL6wRA5l2mbO1qWONANAJ0t5
  Args:
    content: I went to Tartine and ate a croissant.
  Memory (call_5Oiko9oLbDIAWylRAnlAaGyf)
 Call ID: call_5Oiko9oLbDIAWylRAnlAaGyf
  Args:
    content: I was thinking about my Japan, and going back this winter!
```

```python
# 응답을 파싱
for m in result["responses"]:
    print(m)
```

```
# 출력

content='Lance had a nice bike ride in San Francisco this morning. Then, he went to Tartine and ate a croissant. He was also thinking about Japan and planning to go back this winter.'
content='I went to Tartine and ate a croissant.'
content='I was thinking about my Japan, and going back this winter!'
```

Trustcall 실행 중에 발생한 모든 도구 호출 목록을 출력합니다.

```python
# Trustcall에서 도구 호출을 관찰
spy.called_tools
```

`PatchDoc`을 통해 기존 문서('json_doc_id' : '0')를 업데이트하기 위한 도구 호출을 합니다.

그리고 새 Memory를 반영해 기존 Memory를 업데이트합니다. 그 과정에서 `JSON Patch`가 사용되었습니다.

```
# 출력

[[{'name': 'PatchDoc',
   'args': {'json_doc_id': '0',
    'planned_edits': 'Add the new memory of going to Tartine and eating a croissant to the existing memory content. Then, add another memory about thinking about Japan and planning to go back this winter.',
    'patches': [{'op': 'replace',
      'path': '/content',
      'value': 'Lance had a nice bike ride in San Francisco this morning. Then, he went to Tartine and ate a croissant. He was also thinking about Japan and planning to go back this winter.'}]},
   'id': 'call_ldn7vwEsmCyD4SImztvdBHr4',
   'type': 'tool_call'},
  {'name': 'Memory',
   'args': {'content': 'I went to Tartine and ate a croissant.'},
   'id': 'call_NL6wRA5l2mbO1qWONANAJ0t5',
   'type': 'tool_call'},
  {'name': 'Memory',
   'args': {'content': 'I was thinking about my Japan, and going back this winter!'},
   'id': 'call_5Oiko9oLbDIAWylRAnlAaGyf',
   'type': 'tool_call'}]]
```

`extract_tool_info`는 spy.called_tools에서 도구 호출 목록을 받아 도구 호출이 `업데이트(PathDoc)`인지 `새 Memory를 생성`했는지에 대해서 읽기 좋게 변환합니다.

```python
def extract_tool_info(tool_calls, schema_name="Memory"):
    """도구 호출에서 패치(업데이트)와 새로운 기억 생성을 모두 추출
  
    Args:
        tool_calls: 모델에서 발생한 도구 호출 리스트
        schema_name: 스키마 도구의 이름 (예: "Memory", "ToDo", "Profile")
    """

    # changes 리스트를 초기화
    changes = []
  
    for call_group in tool_calls:
        for call in call_group:
            if call['name'] == 'PatchDoc':
                changes.append({
                    'type': 'update',
                    'doc_id': call['args']['json_doc_id'],
                    'planned_edits': call['args']['planned_edits'],
                    'value': call['args']['patches'][0]['value']
                })
            elif call['name'] == schema_name:
                changes.append({
                    'type': 'new',
                    'value': call['args']
                })

    # 결과를 하나의 문자열로 포맷팅
    result_parts = []
    for change in changes:
        if change['type'] == 'update':
            result_parts.append(
                f"Document {change['doc_id']} updated:\n"
                f"Plan: {change['planned_edits']}\n"
                f"Added content: {change['value']}"
            )
        else:
            result_parts.append(
                f"New {schema_name} created:\n"
                f"Content: {change['value']}"
            )
  
    return "\n\n".join(result_parts)

# spy.called_tools를 분석하여 추출 과정에서 실제로 어떤 일이 발생했는지 확인
schema_name = "Memory"
changes = extract_tool_info(spy.called_tools, schema_name)
print(changes)
```

출력에서 문서 Memory 0번이 업데이트되고 새로운 Memory 생성을 읽기 좋게 보여줍니다.

```
# 출력

Document 0 updated:
Plan: Add the new memory of going to Tartine and eating a croissant to the existing memory content. Then, add another memory about thinking about Japan and planning to go back this winter.
Added content: Lance had a nice bike ride in San Francisco this morning. Then, he went to Tartine and ate a croissant. He was also thinking about Japan and planning to go back this winter.

New Memory created:
Content: {'content': 'I went to Tartine and ate a croissant.'}

New Memory created:
Content: {'content': 'I was thinking about my Japan, and going back this winter!'}
```

### 4.  2.  ToDo 리스트 에이전트 구축

`ReAct 에이전트` 기반 ToDo 리스트 생성 및 관리 구현합니다.

이 에이전트는 다음 세 가지 장기 메모리 타입을 업데이트하는 결정을 내립니다.

1. `사용자 프로필 (user)` : 사용자 정보 생성 또는 업데이트
2. `ToDo 리스트 컬렉션 (todo)` : ToDo 항목 추가 또는 업데이트
3. `지침 업데이트 (instrucition)`: ToDo 항목 업데이트 방법 지침

```python
from typing import TypedDict, Literal

# Memory 도구 업데이트하기
class UpdateMemory(TypedDict):
    """ Decision on what memory type to update """
    update_type: Literal['user', 'todo', 'instructions']
```

```python
_set_env("OPENAI_API_KEY")
```

### 4.  3.  그래프 정의

그래프의 전반적인 흐름은 다음과 같습니다.

1. 스키마는 `Profile`과 `ToDo`는 `Pydantic` 기반 구조화, `instruction`은 딕셔너리의 `memory` 필드로 저장됩니다.
2. 주요 노드는 `task_mAIstro`, `update_profile`, `update_todos`, `update_instructions`로 구성됩니다.
3. `task_mAIstro`는 사용자의 입력 메시지를 바탕으로, LLM의 Reasoning을 통해 어떤 타입의 메모리를 업데이트할지 결정하고, `UpdateMemory` 도구 호출을 생성합니다.
4. 조건부 엣지 `route_message`가 도구 호출의 타입(user/todo/instruction)에 따라 해당 노드로 분기합니다.
5. 각 update_xxx 노드에서 `Trustcall Extractor`가 LLM 결과를 구조화하여 장기 메모리 컬렉션에 저장하거나 기존 데이터를 패치합니다.
6. ToDo의 `time_to_complete` 등 세부 필드는 에이전트가 추론해 `자동으로 작성`합니다.
7. 저장된 메모리는 user_id에 기반해 장기적으로 관리되며, 여러 세션과 스레드에서 재활용될 수 있습니다.

아래 그림은 이 전체 프로세스를 간단하게 보여줍니다.

![ReACT ToDo 에이전트 프로세스](assets/posts/2025-06-17-langgraph-long-term-memory-2nd/long-term-momory_2nd_01.png)
_ReACT ToDo 에이전트 프로세스_

```python
import uuid
from IPython.display import Image, display

from datetime import datetime
from trustcall import create_extractor
from typing import Optional
from pydantic import BaseModel, Field

from langchain_core.runnables import RunnableConfig
from langchain_core.messages import merge_message_runs, HumanMessage, SystemMessage

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, MessagesState, END, START
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore

from langchain_openai import ChatOpenAI

# 모델 초기화
model = ChatOpenAI(model="gpt-4o", temperature=0)

# 사용자 프로필 스키마
class Profile(BaseModel):
    """대화하고 있는 사용자의 프로필"""
    name: Optional[str] = Field(description="The user's name", default=None)
    location: Optional[str] = Field(description="The user's location", default=None)
    job: Optional[str] = Field(description="The user's job", default=None)
    connections: list[str] = Field(
        description="Personal connection of the user, such as family members, friends, or coworkers",
        default_factory=list
    )
    interests: list[str] = Field(
        description="Interests that the user has", 
        default_factory=list
    )

# ToDo 스키마
class ToDo(BaseModel):
    task: str = Field(description="The task to be completed.")
    time_to_complete: Optional[int] = Field(description="Estimated time to complete the task (minutes).")
    deadline: Optional[datetime] = Field(
        description="When the task needs to be completed by (if applicable)",
        default=None
    )
    solutions: list[str] = Field(
        description="List of specific, actionable solutions (e.g., specific ideas, service providers, or concrete options relevant to completing the task)",
        min_items=1,
        default_factory=list
    )
    status: Literal["not started", "in progress", "done", "archived"] = Field(
        description="Current status of the task",
        default="not started"
    )

# 사용자 프로필 업데이트용 Trustcall extractor 생성 
profile_extractor = create_extractor(
    model,
    tools=[Profile],
    tool_choice="Profile",
)

# 어떤 도구를 호출하고 어떤 항목을 업데이트할지 선택하는 챗봇 지침 (시스템 메시지)
MODEL_SYSTEM_MESSAGE = """You are a helpful chatbot. 

You are designed to be a companion to a user, helping them keep track of their ToDo list.

You have a long term memory which keeps track of three things:
1. The user's profile (general information about them) 
2. The user's ToDo list
3. General instructions for updating the ToDo list

Here is the current User Profile (may be empty if no information has been collected yet):
<user_profile>
{user_profile}
</user_profile>

Here is the current ToDo List (may be empty if no tasks have been added yet):
<todo>
{todo}
</todo>

Here are the current user-specified preferences for updating the ToDo list (may be empty if no preferences have been specified yet):
<instructions>
{instructions}
</instructions>

Here are your instructions for reasoning about the user's messages:

1. Reason carefully about the user's messages as presented below. 

2. Decide whether any of the your long-term memory should be updated:
- If personal information was provided about the user, update the user's profile by calling UpdateMemory tool with type `user`
- If tasks are mentioned, update the ToDo list by calling UpdateMemory tool with type `todo`
- If the user has specified preferences for how to update the ToDo list, update the instructions by calling UpdateMemory tool with type `instructions`

3. Tell the user that you have updated your memory, if appropriate:
- Do not tell the user you have updated the user's profile
- Tell the user them when you update the todo list
- Do not tell the user that you have updated instructions

4. Err on the side of updating the todo list. No need to ask for explicit permission.

5. Respond naturally to user user after a tool call was made to save memories, or if no tool call was made."""

# Trustcall 지침
TRUSTCALL_INSTRUCTION = """Reflect on following interaction. 

Use the provided tools to retain any necessary memories about the user. 

Use parallel tool calling to handle updates and insertions simultaneously.

System Time: {time}"""

# ToDo 리스트 업데이트를 위한 지침
CREATE_INSTRUCTIONS = """Reflect on the following interaction.

Based on this interaction, update your instructions for how to update ToDo list items. 

Use any feedback from the user to update how they like to have items added, etc.

Your current instructions are:

<current_instructions>
{current_instructions}
</current_instructions>"""

# 노드 정의
def task_mAIstro(state: MessagesState, config: RunnableConfig, store: BaseStore):

    """Store에서 Memory를 불러와 챗봇 응답에서 개인화에 활용"""
  
    # 사용자 ID를 config에서 가져옴
    user_id = config["configurable"]["user_id"]

    # 프로필 메모리를 Store에서 검색
    namespace = ("profile", user_id)
    memories = store.search(namespace)
    if memories:
        user_profile = memories[0].value
    else:
        user_profile = None

    # 작업(ToDo) 메모리를 Store에서 검색
    namespace = ("todo", user_id)
    memories = store.search(namespace)
    todo = "\n".join(f"{mem.value}" for mem in memories)

    # 커스텀 지침 메모리 검색
    namespace = ("instructions", user_id)
    memories = store.search(namespace)
    if memories:
        instructions = memories[0].value
    else:
        instructions = ""
  
    system_msg = MODEL_SYSTEM_MESSAGE.format(user_profile=user_profile, todo=todo, instructions=instructions)

    # 메모리와 대화 기록을 바탕으로 응답 생성
    response = model.bind_tools([UpdateMemory], parallel_tool_calls=False).invoke([SystemMessage(content=system_msg)]+state["messages"])

    return {"messages": [response]}

def update_profile(state: MessagesState, config: RunnableConfig, store: BaseStore):

    """대화 내역을 반영하여 메모리 컬렉션을 업데이트"""
  
    # 사용자 ID를 config에서 가져옴
    user_id = config["configurable"]["user_id"]

    # 프로필 메모리의 네임스페이스 정의
    namespace = ("profile", user_id)

    # 최신 메모리 가져오기
    existing_items = store.search(namespace)

    # Trustcall extractor에 사용할 기존 메모리 포맷팅
    tool_name = "Profile"
    existing_memories = ([(existing_item.key, tool_name, existing_item.value)
                          for existing_item in existing_items]
                          if existing_items
                          else None
                        )

    # 대화 기록과 지침 병합
    TRUSTCALL_INSTRUCTION_FORMATTED=TRUSTCALL_INSTRUCTION.format(time=datetime.now().isoformat())
    updated_messages=list(merge_message_runs(messages=[SystemMessage(content=TRUSTCALL_INSTRUCTION_FORMATTED)] + state["messages"][:-1]))

    # extractor 실행
    result = profile_extractor.invoke({"messages": updated_messages, 
                                         "existing": existing_memories})

    # Trustcall 메모리를 스토어에 저장
    for r, rmeta in zip(result["responses"], result["response_metadata"]):
        store.put(namespace,
                  rmeta.get("json_doc_id", str(uuid.uuid4())),
                  r.model_dump(mode="json"),
            )
    tool_calls = state['messages'][-1].tool_calls
    return {"messages": [{"role": "tool", "content": "updated profile", "tool_call_id":tool_calls[0]['id']}]}

def update_todos(state: MessagesState, config: RunnableConfig, store: BaseStore):

    """대화 내역을 반영하여 ToDo 메모리 컬렉션을 업데이트"""
  
    # 사용자 ID를 config에서 가져옴
    user_id = config["configurable"]["user_id"]

    # ToDo 메모리의 네임스페이스 정의
    namespace = ("todo", user_id)

    # 최신 메모리 가져오기
    existing_items = store.search(namespace)

    # Trustcall extractor에 사용할 기존 기억 포맷팅
    tool_name = "ToDo"
    existing_memories = ([(existing_item.key, tool_name, existing_item.value)
                          for existing_item in existing_items]
                          if existing_items
                          else None
                        )

    # 대화 기록과 지침 병합
    TRUSTCALL_INSTRUCTION_FORMATTED=TRUSTCALL_INSTRUCTION.format(time=datetime.now().isoformat())
    updated_messages=list(merge_message_runs(messages=[SystemMessage(content=TRUSTCALL_INSTRUCTION_FORMATTED)] + state["messages"][:-1]))

    # Trustcall에서 발생한 도구 호출 내역 확인을 위한 Spy 초기화
    spy = Spy()
  
    # ToDo 리스트 업데이트용 Trustcall extractor 생성 
    todo_extractor = create_extractor(
    model,
    tools=[ToDo],
    tool_choice=tool_name,
    enable_inserts=True
    ).with_listeners(on_end=spy)

    # extractor 실행
    result = todo_extractor.invoke({"messages": updated_messages, 
                                    "existing": existing_memories})

    # Trustcall 메모리를 스토어에 저장
    for r, rmeta in zip(result["responses"], result["response_metadata"]):
        store.put(namespace,
                  rmeta.get("json_doc_id", str(uuid.uuid4())),
                  r.model_dump(mode="json"),
            )
  
    # task_mAIstro에서의 도구 호출에 대한 업데이트 응답 확인
    tool_calls = state['messages'][-1].tool_calls

    # Trustcall이 변경된 내용을 추출해서 task_mAIstro에 반환된 ToolMessage를 추가.
    todo_update_msg = extract_tool_info(spy.called_tools, tool_name)
    return {"messages": [{"role": "tool", "content": todo_update_msg, "tool_call_id":tool_calls[0]['id']}]}

def update_instructions(state: MessagesState, config: RunnableConfig, store: BaseStore):

    """대화 내역을 반영하여 지침 메모리 컬렉션을 업데이트"""
  
    # 사용자 ID를 config에서 가져옴
    user_id = config["configurable"]["user_id"]
  
    namespace = ("instructions", user_id)

    existing_memory = store.get(namespace, "user_instructions")
  
    # 시스템 프롬프트에 기존 지침 반영
    system_msg = CREATE_INSTRUCTIONS.format(current_instructions=existing_memory.value if existing_memory else None)
    new_memory = model.invoke([SystemMessage(content=system_msg)]+state['messages'][:-1] + [HumanMessage(content="Please update the instructions based on the conversation")])

    # 스토어에서 기존 메모리를 덮어쓰기
    key = "user_instructions"
    store.put(namespace, key, {"memory": new_memory.content})
    tool_calls = state['messages'][-1].tool_calls
    return {"messages": [{"role": "tool", "content": "updated instructions", "tool_call_id":tool_calls[0]['id']}]}

# 조건부 엣지 설정
def route_message(state: MessagesState, config: RunnableConfig, store: BaseStore) -> Literal[END, "update_todos", "update_instructions", "update_profile"]:

    """메모리와 대화 내역을 반영해 메모리 컬렉션을 업데이트할지 결정"""
    message = state['messages'][-1]
    if len(message.tool_calls) ==0:
        return END
    else:
        tool_call = message.tool_calls[0]
        if tool_call['args']['update_type'] == "user":
            return "update_profile"
        elif tool_call['args']['update_type'] == "todo":
            return "update_todos"
        elif tool_call['args']['update_type'] == "instructions":
            return "update_instructions"
        else:
            raise ValueError

# 그래프 및 모든 노드 생성
builder = StateGraph(MessagesState)

# 메모리 추출 프로세스 흐름 정의
builder.add_node(task_mAIstro)
builder.add_node(update_todos)
builder.add_node(update_profile)
builder.add_node(update_instructions)
builder.add_edge(START, "task_mAIstro")
builder.add_conditional_edges("task_mAIstro", route_message)
builder.add_edge("update_todos", "task_mAIstro")
builder.add_edge("update_profile", "task_mAIstro")
builder.add_edge("update_instructions", "task_mAIstro")

# 장기(스레드 간) 메모리용 저장소
across_thread_memory = InMemoryStore()

# 단기(스레드 내) 메모리용 체크포인터
within_thread_memory = MemorySaver()

# 체크포인터와 저장소로 그래프 컴파일
graph = builder.compile(checkpointer=within_thread_memory, store=across_thread_memory)

# 그래프 이미지
display(Image(graph.get_graph(xray=1).draw_mermaid_png()))
```

![ToDo 에이전트 그래프](assets/posts/2025-06-17-langgraph-long-term-memory-2nd/long-term-momory_2nd_02.png)
_ToDo 에이전트 그래프_

```python

# 단기(스레드 내) 메모리에는 thread_id를 제공
# 장기(스레드 간) 메모리에는 user_id를 제공
config = {"configurable": {"thread_id": "1", "user_id": "Lance"}}

# 프로필 메모리를 생성하기 위한 사용자 입력
input_messages = [HumanMessage(content="My name is Lance. I live in SF with my wife. I have a 1 year old daughter.")]

# 그래프 실행
for chunk in graph.stream({"messages": input_messages}, config, stream_mode="values"):
    chunk["messages"][-1].pretty_print()
```

```
# 출력

================================ Human Message =================================

My name is Lance. I live in SF with my wife. I have a 1 year old daughter.
================================== Ai Message ==================================
Tool Calls:
  UpdateMemory (call_RhTEtKMwTdApzGcCcoVNUYDB)
 Call ID: call_RhTEtKMwTdApzGcCcoVNUYDB
  Args:
    update_type: user
================================= Tool Message =================================

updated profile
================================== Ai Message ==================================

Got it! How can I assist you today?
```

```python
# ToDo 생성을 위한 사용자 입력
input_messages = [HumanMessage(content="My wife asked me to book swim lessons for the baby.")]

# 그래프 실행
for chunk in graph.stream({"messages": input_messages}, config, stream_mode="values"):
    chunk["messages"][-1].pretty_print()
```

```
# 출력

================================ Human Message =================================

My wife asked me to book swim lessons for the baby.
================================== Ai Message ==================================
Tool Calls:
  UpdateMemory (call_WGlwc8gADAQU5NhCnrFMvYmM)
 Call ID: call_WGlwc8gADAQU5NhCnrFMvYmM
  Args:
    update_type: todo
================================= Tool Message =================================

New ToDo created:
Content: {'task': 'Book swim lessons for 1-year-old daughter.', 'time_to_complete': 30, 'solutions': ['Check local swim schools in SF', 'Look for baby swim classes online', 'Ask friends for recommendations'], 'status': 'not started'}
================================== Ai Message ==================================

I've added "Book swim lessons for 1-year-old daughter" to your ToDo list. Is there anything else you'd like to add or update?
```

```python
# ToDo 항목 생성 지침을 업데이트하기 위한 사용자 입력
input_messages = [HumanMessage(content="When creating or updating ToDo items, include specific local businesses / vendors.")]

# 그래프 실행
for chunk in graph.stream({"messages": input_messages}, config, stream_mode="values"):
    chunk["messages"][-1].pretty_print()
```

```
# 출력

================================ Human Message =================================

When creating or updating ToDo items, include specific local businesses / vendors.
================================== Ai Message ==================================
Tool Calls:
  UpdateMemory (call_ipTO0t9XsJ5ZxaqmRIez6N16)
 Call ID: call_ipTO0t9XsJ5ZxaqmRIez6N16
  Args:
    update_type: instructions
================================= Tool Message =================================

updated instructions
================================== Ai Message ==================================

Got it! I'll make sure to include specific local businesses or vendors in San Francisco when creating or updating your ToDo items. Anything else you'd like to do?
```

```python
# 업데이트된 지침 확인
user_id = "Lance"

# 검색 
for memory in across_thread_memory.search(("instructions", user_id)):
    print(memory.value)
```

```
# 출력

{'memory': '<current_instructions>\nWhen creating or updating ToDo list items for Lance, include specific local businesses or vendors in San Francisco. For example, when adding a task like booking swim lessons, suggest specific swim schools or classes available in the area.\n</current_instructions>'}
```

```python
# ToDo 생성을 위한 사용자 입력
input_messages = [HumanMessage(content="I need to fix the jammed electric Yale lock on the door.")]

# 그래프 실행
for chunk in graph.stream({"messages": input_messages}, config, stream_mode="values"):
    chunk["messages"][-1].pretty_print()
```

```
# 출력

================================ Human Message =================================

I need to fix the jammed electric Yale lock on the door.
================================== Ai Message ==================================
Tool Calls:
  UpdateMemory (call_lMEgqvzS3yPxWbOD5P1cMFRx)
 Call ID: call_lMEgqvzS3yPxWbOD5P1cMFRx
  Args:
    update_type: todo
================================= Tool Message =================================

New ToDo created:
Content: {'task': 'Fix the jammed electric Yale lock on the door.', 'time_to_complete': 60, 'solutions': ['Contact a local locksmith in SF', "Check Yale's customer support for troubleshooting", 'Look for repair guides online'], 'status': 'not started'}

Document 4207d807-8b44-458d-acfd-a1d104fe2cc4 updated:
Plan: Add specific local businesses or vendors to the solutions list for booking swim lessons.
Added content: AquaTech Swim School
================================== Ai Message ==================================

I've added "Fix the jammed electric Yale lock on the door" to your ToDo list. If you need any specific recommendations for local locksmiths in SF, just let me know!
```

```python
# 저장할 메모리의 네임스페이스
user_id = "Lance"

# Search 
for memory in across_thread_memory.search(("todo", user_id)):
    print(memory.value)
```

```
# 출력

{'task': 'Book swim lessons for 1-year-old daughter.', 'time_to_complete': 30, 'deadline': None, 'solutions': ['Check local swim schools in SF', 'Look for baby swim classes online', 'Ask friends for recommendations', 'AquaTech Swim School', 'La Petite Baleen Swim Schools', 'San Francisco Recreation and Parks swim classes'], 'status': 'not started'}
{'task': 'Fix the jammed electric Yale lock on the door.', 'time_to_complete': 60, 'deadline': None, 'solutions': ['Contact a local locksmith in SF', "Check Yale's customer support for troubleshooting", 'Look for repair guides online'], 'status': 'not started'}
```

```python
# 기존 ToDo를 업데이트하기 위한 사용자 입력
input_messages = [HumanMessage(content="For the swim lessons, I need to get that done by end of November.")]

# 그래프 실행
for chunk in graph.stream({"messages": input_messages}, config, stream_mode="values"):
    chunk["messages"][-1].pretty_print()
```

```
# 출력

================================ Human Message =================================

For the swim lessons, I need to get that done by end of November.
================================== Ai Message ==================================
Tool Calls:
  UpdateMemory (call_yVL0PsxQakZQYo3dYxC9SaJf)
 Call ID: call_yVL0PsxQakZQYo3dYxC9SaJf
  Args:
    update_type: todo
================================= Tool Message =================================

Document 4207d807-8b44-458d-acfd-a1d104fe2cc4 updated:
Plan: Add a deadline to the task for booking swim lessons, specifying the end of November.
Added content: 2025-11-30T23:59:59
================================== Ai Message ==================================

I've updated the deadline for booking swim lessons to the end of November. If there's anything else you need, feel free to let me know!
```

## 정리

`LangGraph Memory Store`는 `key-value` 기반의 `Store`로, 스레드 간 정보 공유를 위해 `사용자 ID`를 `네임스페이스`로 활용합니다.

이 구조 덕분에 동일 사용자의 정보를 여러 채팅 세션에서 일관되게 관리할 수 있습니다.

시맨틱 메모리는 `단일 객체(프로필)`와 `다수 항목(컬렉션)`으로 관리되며, 이들 스키마의 생성 및 업데이트는 `Trustcall`을 통해 자동화됩니다.

특히 `Trustcall`은 LLM이 구조화 데이터를 생성할 때 발생할 수 있는 JSON 오류나 전체 덮어쓰기 문제를 `JSON Patch` 방식으로 해결하여, 기존 데이터의 필요한 부분만 유연하게 업데이트할 수 있습니다.

마지막으로, 프로필, ToDo, 지침의 업데이트를 ReACT 기반 에이전트로 구현하며, `Trustcall + JSON Patch`의 실제 적용을 살펴보았습니다.

다음 포스팅에서 `배포`에 대해서 자세히 살펴보겠습니다.

## References

* [LangGraph 공식 문서](https://langchain-ai.github.io/langgraph){: target="_blank"}
* [LangChain Academy](https://github.com/langchain-ai/langchain-academy){: target="_blank"}
