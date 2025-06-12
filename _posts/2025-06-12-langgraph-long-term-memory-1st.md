---
title: LangGraph 장기 메모리 (1)
date: 2025-06-12 11:15:43 +/-TTTT
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
series_order: 8
---
[메모리](https://pmc.ncbi.nlm.nih.gov/articles/PMC10410470/){: target="_blank"}는 개인의 정체성 또는 사람들이 현재와 미래를 이해하기 위해 정보를 저장하고, 검색하며, 활용할 수 있게 해 주는 인지 기능으로 AI 애플리케이션에서 사용할 수 있는 [여러 가지 장기 메모리 유형](https://langchain-ai.github.io/langgraph/concepts/memory/#memory){: target="_blank"}(Long-term Memory)이 있습니다.

`장기 메모리`와 일반적인 사실과 개념을 저장하는 인간의 장기 기억 중 하나인 [시맨틱 메모리(Semantic Memory)](https://langchain-ai.github.io/langgraph/concepts/memory/#semantic-memory){: target="_blank"}를 살펴보겠습니다.

> 학습할 리소스는 [LangChain Academy Github](https://github.com/langchain-ai/langchain-academy){: target="_blank"}를 사용합니다.
{: .prompt-info }

## 1.   메모리 기반 챗봇 구축

먼저 스레드 내의 `단기 메모리(short-term)`와 스레드 간의 `장기 메모리(long-term)`를 모두 사용하는 챗봇을 구축하고

`장기 메모리`를 저장하고 검색하는 방법인 [LangGraph Memory Store](https://langchain-ai.github.io/langgraph/reference/store/#langgraph.store.base.BaseStore){: target="_blank"}를 살펴보겠습니다.

`장기 메모리`를 통해 사용자의 정보를 기억하는 개인화된 챗봇을 구축하고 사용자와 대화하는 도중 `핫 패스(Hot Path)` 경로에서 메모리를 저장하도록 구현합니다.

`핫 패스`는 사용자 요청 처리 흐름 안에서 즉시 메모리를 업데이트하기 때문에, 새로운 정보가 다음 상호작용에 바로 반영되는 실시간 일관성을 제공합니다.

`백그라운드 방식`은 메모리 업데이트를 별도의 비동기 작업으로 분리하여 주요 응답 경로의 지연을 제거하고, 애플리케이션 로직과 메모리 관리 로직을 깔끔하게 분리할 수 있습니다.

![에이전트가 메모리를 기록하는 방식](assets/posts/2025-06-12-langgraph-long-term-memory-1st/long-term-momory_1st_02.png)
_에이전트가 메모리를 기록하는 방식_

`핫 패스`는 사용자가 채팅을 진행하는 흐름 속에서 `실시간으로 메모리를 기록`할 수 있습니다.

먼저 환경을 구성합니다.

```python
# 환경 구성
%%capture --no-stderr
%pip install -U langchain_openai langgraph langchain_core
```

```python
import os, getpass

def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

_set_env("LANGSMITH_API_KEY")
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "langchain-academy"
```

### 1.  1.  LangGraph Store 소개

[LangGraph Memory Store](https://langchain-ai.github.io/langgraph/reference/store/#langgraph.store.base.BaseStore){: target="_blank"}는 LangGraph에서 `스레드 간(across threads)` 정보를 저장하고 검색할 수 있는 방법을 제공합니다.

이 `Store`는 영구적인 `key-value` 스토어를 위한 [오픈 소스 베이스 클래스](https://blog.langchain.dev/launching-long-term-memory-support-in-langgraph/){: target="_blank"}입니다.

```python
import uuid
from langgraph.store.memory import InMemoryStore
in_memory_store = InMemoryStore()
```

LangGraph 장기 메모리 객체를 [Store](https://langchain-ai.github.io/langgraph/reference/store/#langgraph.store.base.BaseStore){: target="_blank"}에 저장할 때 다음 항목을 제공합니다

* `디렉터리`와 유사한 튜플인 객체의 `namespace`
* `파일명`과 유사한 객체의 `key`
* `파일 내용`과 유사한 객체의 `value`

[put](https://langchain-ai.github.io/langgraph/reference/store/#langgraph.store.base.BaseStore.put){: target="_blank"} 메서드를 사용하여 `namespace`와 `key`를 통해 객체를 스토어에 저장합니다.

![LangGraph Store 구조](assets/posts/2025-06-12-langgraph-long-term-memory-1st/long-term-momory_1st_01.png)
_LangGraph Store 구조_

```python
# 저장할 메모리의 네임스페이스
user_id = "1"
namespace_for_memory = (user_id, "memories")

# 메모리 저장을 위한 키 생성
key = str(uuid.uuid4())

# 값은 딕셔너리 타입 
value = {"food_preference" : "I like pizza"}

# 메모리 저장
in_memory_store.put(namespace_for_memory, key, value)
```

[search](https://langchain-ai.github.io/langgraph/reference/store/#langgraph.store.base.BaseStore.search){: target="_blank"}를 사용하여 `namespace`로 `Store`에서 객체를 검색하면 리스트를 반환합니다.

```python
# 검색 
memories = in_memory_store.search(namespace_for_memory)
type(memories)
```

```python
# 출력

list
```

```python
# 메타데이터 
memories[0].dict()
```

```
# 출력

{'namespace': ['1', 'memories'],
 'key': '2948c4ce-e68f-4c68-a7bf-0bbe5f135c28',
 'value': {'food_preference': 'I like pizza'},
 'created_at': '2025-04-23T06:35:37.555100+00:00',
 'updated_at': '2025-04-23T06:35:37.555102+00:00',
 'score': None}
```

```python
# key, value 확인

print(memories[0].key, memories[0].value)
```

```
# 출력

2948c4ce-e68f-4c68-a7bf-0bbe5f135c28 {'food_preference': 'I like pizza'}
```

또한 [get](https://langchain-ai.github.io/langgraph/reference/store/#langgraph.store.base.BaseStore.get){: target="_blank"}을 사용하여 `namespace`와 `key`로 객체를 가져올 수 있습니다.

```python
# 네임스페이스와 키로 메모리 가져오기
memory = in_memory_store.get(namespace_for_memory, key)
memory.dict()
```

```
# 출력

{'namespace': ['1', 'memories'],
 'key': '2948c4ce-e68f-4c68-a7bf-0bbe5f135c28',
 'value': {'food_preference': 'I like pizza'},
 'created_at': '2025-04-23T06:35:37.555100+00:00',
 'updated_at': '2025-04-23T06:35:37.555102+00:00'}
```

### 1.  2.  장기 메모리 기반 챗봇 (Chatbot w/ Long-term memory)

[두 가지 유형의 메모리](https://docs.google.com/presentation/d/181mvjlgsnxudQI6S3ritg9sooNyu4AcLLFH1UK0kIuk/edit#slide=id.g30eb3c8cf10_0_156){: target="_blank"} 기반 챗봇을 구축합니다.

1. `단기(스레드 내) 메모리` : 챗봇이 대화 기록을 유지하거나 채팅 세션 중 일시 중단을 허용할 수 있습니다.
2. `장기(스레드 간) 메모리` : 챗봇이 특정 사용자에 대한 정보를 `모든 채팅 세션에 걸쳐` 기억할 수 있습니다.

```python
_set_env("OPENAI_API_KEY")
```

먼저 `단기 메모리`는 [체크포인터](https://langchain-ai.github.io/langgraph/concepts/persistence/#checkpointer-libraries){: target="_blank"}를 사용합니다. 이전 포스팅의 `체크포인터` 개념에 대해 다시 살펴보면

* 그래프 상태를 단계마다 스레드에 기록합니다.
* 스레드에 채팅 기록을 `영속화(persist)`, 즉 스레드별 상태가 데이터베이스 저장됩니다.
* 그래프를 언제든 중단할 수 있고 동일한 `thread_id`로 스레드의 어느 단계든 다시 실행할 수 있습니다.

그리고 `장기 메모리`에는 앞서 소개한 [LangGraph Store](https://langchain-ai.github.io/langgraph/reference/store/#langgraph.store.base.BaseStore){: target="_blank"}를 사용합니다.

```python
# 채팅 모델
from langchain_openai import ChatOpenAI

# LLM 초기화
model = ChatOpenAI(model="gpt-4o", temperature=0) 
```

먼저, 채팅 기록은 `체크포인터`를 사용해 `단기 메모리`에 저장됩니다.

챗봇은 이 채팅 기록을 반영한 뒤, 메모리를 생성하여 [LangGraph Store](https://langchain-ai.github.io/langgraph/reference/store/#langgraph.store.base.BaseStore){: target="_blank"}에 저장합니다.

저장된 이 `장기 메모리`는 향후 모든 채팅 세션에서 활용되어 챗봇의 응답을 개인화하는 데 사용됩니다.

```python
from IPython.display import Image, display

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.store.base import BaseStore

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables.config import RunnableConfig

# 챗봇 지침
MODEL_SYSTEM_MESSAGE = """You are a helpful assistant with memory that provides information about the user. 
If you have memory for this user, use it to personalize your responses.
Here is the memory (it may be empty): {memory}"""

# 채팅 기록과 기존 메모리에서 새로운 메모리를 생성
CREATE_MEMORY_INSTRUCTION = """"You are collecting information about the user to personalize your responses.

CURRENT USER INFORMATION:
{memory}

INSTRUCTIONS:
1. Review the chat history below carefully
2. Identify new information about the user, such as:
   - Personal details (name, location)
   - Preferences (likes, dislikes)
   - Interests and hobbies
   - Past experiences
   - Goals or future plans
3. Merge any new information with existing memory
4. Format the memory as a clear, bulleted list
5. If new information conflicts with existing memory, keep the most recent version

Remember: Only include factual information directly stated by the user. Do not make assumptions or inferences.

Based on the chat history below, please update the user information:"""

def call_model(state: MessagesState, config: RunnableConfig, store: BaseStore):

    """스토어에서 메모리를 불러와 챗봇의 응답을 개인화하는 데 사용"""
  
    # config에서 사용자 ID를 가져옴
    user_id = config["configurable"]["user_id"]

    # 스토어에서 메모리를 가져옴
    namespace = ("memory", user_id)
    key = "user_memory"
    existing_memory = store.get(namespace, key)

    # 메모리가 존재하는 경우 실제 메모리 내용을 추출하고 접두사(prefix)를 추가
    if existing_memory:
        # 값은 'memory' 키가 포함된 딕셔너리
        existing_memory_content = existing_memory.value.get('memory')
    else:
        existing_memory_content = "No existing memory found."

    # 시스템 프롬프트에 메모리를 포맷
    system_msg = MODEL_SYSTEM_MESSAGE.format(memory=existing_memory_content)
  
    # 메모리와 대화 기록을 사용하여 응답
    response = model.invoke([SystemMessage(content=system_msg)]+state["messages"])

    return {"messages": response}

def write_memory(state: MessagesState, config: RunnableConfig, store: BaseStore):

    """채팅 기록을 검토하여 메모리를 생성한 뒤 스토어에 저장"""
  
    # config에서 사용자 ID를 가져옴
    user_id = config["configurable"]["user_id"]

    # 스토어에서 기존 메모리를 가져옴
    namespace = ("memory", user_id)
    existing_memory = store.get(namespace, "user_memory")
  
    # 메모리를 추출함
    if existing_memory:
        existing_memory_content = existing_memory.value.get('memory')
    else:
        existing_memory_content = "No existing memory found."

    # 시스템 프롬프트에 메모리를 포맷
    system_msg = CREATE_MEMORY_INSTRUCTION.format(memory=existing_memory_content)
    new_memory = model.invoke([SystemMessage(content=system_msg)]+state['messages'])

    # 스토어의 기존 메모리를 덮어씀
    key = "user_memory"

    # 'memory' 키가 포함된 딕셔너리로 값을 작성
    store.put(namespace, key, {"memory": new_memory.content})

# 그래프 정의
builder = StateGraph(MessagesState)
builder.add_node("call_model", call_model)
builder.add_node("write_memory", write_memory)
builder.add_edge(START, "call_model")
builder.add_edge("call_model", "write_memory")
builder.add_edge("write_memory", END)

# 장기(스레드 간) 메모리를 위한 스토어
across_thread_memory = InMemoryStore()

# 단기(스레드 내) 메모리를 위한 체크포인터
within_thread_memory = MemorySaver()

# 체크포인터와 스토어를 사용하여 그래프를 컴파일
graph = builder.compile(checkpointer=within_thread_memory, store=across_thread_memory)

# 그래프 이미지
display(Image(graph.get_graph(xray=1).draw_mermaid_png()))
```

챗봇과 상호작용을 할 때 다음 두 가지를 제공합니다.

1. `단기(스레드 내) 메모리` : 채팅 기록을 영속화하기 위한 스레드 ID
2. `장기(스레드 간) 메모리` : 사용자의 장기 메모리에 네임스페이스를 지정하기 위한 사용자 ID

이제 실제로 이들이 어떻게 함께 작동하는지 살펴보겠습니다.

```python
# 단기(스레드 내) 메모리를 위한 스레드 ID를 제공
# 장기(스레드 간) 메모리를 위한 사용자 ID를 제공
config = {"configurable": {"thread_id": "1", "user_id": "1"}}

# 사용자 입력 
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

Hello, Lance! It's nice to meet you. How can I assist you today?
```

```python
# 사용자 입력
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

That sounds like a great way to explore the city, Lance! San Francisco has some beautiful routes and views. Do you have a favorite trail or area you like to bike in?
```

스레드 내(within-thread) 메모리를 위해 `MemorySaver` `체크포인터`를 사용합니다.

`체크포인터`는 채팅 기록을 해당 스레드에 저장합니다.

이제 스레드에 저장된 채팅 기록을 확인할 수 있습니다.

```python
thread = {"configurable": {"thread_id": "1"}}
state = graph.get_state(thread).values
for m in state["messages"]: 
    m.pretty_print()
```

```
# 출력

================================ Human Message =================================

Hi, my name is Lance
================================== Ai Message ==================================

Hello, Lance! It's nice to meet you. How can I assist you today?
================================ Human Message =================================

I like to bike around San Francisco
================================== Ai Message ==================================

That sounds like a great way to explore the city, Lance! San Francisco has some beautiful routes and views. Do you have a favorite trail or area you like to bike in?
```

그래프를 다음과 같이 스토어와 함께 컴파일했습니다.

`across_thread_memory = InMemoryStore()`

그리고 그래프에 채팅 기록을 반영하여 메모리를 스토어에 저장하는 노드(`write_memory`)를 추가했습니다.

이제 메모리가 스토어에 저장되었는지 확인할 수 있습니다.

```python
# 저정할 메모리의 네임스페이스
user_id = "1"
namespace = ("memory", user_id)
existing_memory = across_thread_memory.get(namespace, "user_memory")
existing_memory.dict()
```

```
# 출력

{'namespace': ['memory', '1'],
 'key': 'user_memory',
 'value': {'memory': "**Updated User Information:**\n- User's name is Lance.\n- Likes to bike around San Francisco."},
 'created_at': '2025-04-23T06:58:56.857833+00:00',
 'updated_at': '2025-04-23T06:58:56.857835+00:00'}
```

이제 `같은 사용자 ID`로 `새로운 스레드`를 시작합니다.

챗봇이 사용자의 프로필을 기억하여 응답을 개인화했는지 확인할 수 있습니다.

```python
# 스레드 간 메모리를 위한 사용자 ID와 새로운 스레드 ID를 제공
config = {"configurable": {"thread_id": "2", "user_id": "1"}}

# 사용자 입력
input_messages = [HumanMessage(content="Hi! Where would you recommend that I go biking?")]

# 그래프 실행
for chunk in graph.stream({"messages": input_messages}, config, stream_mode="values"):
    chunk["messages"][-1].pretty_print()
```

출력에서 `스레드가 변경`되어도 `사용자 프로필을 기억`합니다.

```
# 출력

================================ Human Message =================================

Hi! Where would you recommend that I go biking?
================================== Ai Message ==================================

Hi Lance! Since you enjoy biking around San Francisco, there are some fantastic routes you might love. Here are a few recommendations:

1. **Golden Gate Park**: This is a classic choice with plenty of trails and beautiful scenery. You can explore the park's gardens, lakes, and even make your way to Ocean Beach.

2. **The Embarcadero**: Starting from AT&T Park, you can bike along the waterfront, passing by the Ferry Building and Pier 39, all the way to Fisherman's Wharf.

3. **Marin Headlands**: If you're up for a bit of a challenge, you can cross the Golden Gate Bridge and explore the trails in the Marin Headlands. The views of the city and the ocean are breathtaking.

4. **Presidio**: This area offers a mix of forested trails and open spaces with views of the Golden Gate Bridge. It's a great spot for both leisurely rides and more intense biking.

5. **Angel Island**: For a unique experience, take a ferry to Angel Island and bike around the island. You'll get stunning views of the Bay Area from different angles.

Let me know if you want more details on any of these routes!
```

```python
# 사용자 입력 
input_messages = [HumanMessage(content="Great, are there any bakeries nearby that I can check out? I like a croissant after biking.")]

# 그래프 실행
for chunk in graph.stream({"messages": input_messages}, config, stream_mode="values"):
    chunk["messages"][-1].pretty_print()
```

```
# 출력

================================ Human Message =================================

Great, are there any bakeries nearby that I can check out? I like a croissant after biking.
================================== Ai Message ==================================

Absolutely, Lance! Here are a few bakeries in San Francisco where you can enjoy a delicious croissant after your ride:

1. **Tartine Bakery**: Located in the Mission District, Tartine is famous for its pastries, and their croissants are a must-try.

2. **Arsicault Bakery**: Situated in the Richmond District, Arsicault has been praised for its buttery, flaky croissants. It's a bit of a detour from the main biking routes, but definitely worth it.

3. **b. Patisserie**: In the Pacific Heights area, this bakery offers a variety of pastries, including their popular kouign-amann, which is similar to a croissant.

4. **Le Marais Bakery**: With locations in the Marina and Castro, Le Marais offers a charming French café experience with excellent croissants.

5. **Neighbor Bakehouse**: Located in the Dogpatch neighborhood, this spot is known for its creative pastries and delicious croissants.

These spots should provide a delightful treat after your biking adventures. Enjoy your ride and your croissant!
```

## 2.   프로필 스키마 기반 챗봇

위에서 챗봇은 메모리를 문자열로 저장했지만, 실제로는 메모리가 `구조(structure)`를 가지는 것이 더 실용적입니다.

즉, 하나의 사용자 프로필로 정보를 관리하기 위해 지속적으로 업데이트되는 스키마로 만드는 것이 좋습니다.

이제 챗봇을 확장해서 `시맨틱 메모리`를 단일 `사용자 프로필`로 저장하고 스키마에 신규 정보를 업데이트할 수 있도록 `Trustcall`이라는 라이브러리를 적용합니다.

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

### 2.  1.  사용자 프로필 스키마 정의하기

Python에는 `TypedDict`, `Dictionary`, `JSON`, 그리고 [`Pydantic`](https://docs.pydantic.dev/latest/){: target="_blank"}과 같은 [구조화된 데이터](https://python.langchain.com/docs/concepts/structured_outputs/#schema-definition){: target="_blank"}를 위한 다양한 타입들이 있습니다.

이제 `TypedDict`를 사용하여 사용자 프로필 스키마를 정의합니다.

```python
from typing import TypedDict, List

class UserProfile(TypedDict):
    """타입이 지정된 필드를 가지는 사용자 프로필 스키마"""
    user_name: str  # 사용자가 선호하는 이름
    interests: List[str]  # 사용자의 관심사 목록
```

### 2.  2.  스토어에 스키마 저장하기

[LangGraph Store](https://langchain-ai.github.io/langgraph/reference/store/#langgraph.store.base.BaseStore){: target="_blank"}는 `value` 값으로 모든 Python 딕셔너리를 허용합니다.

```python
# TypedDict 인스턴스 생성
user_profile: UserProfile = {
    "user_name": "Lance",
    "interests": ["biking", "technology", "coffee"]
}
user_profile
```

```
# 출력

{'user_name': 'Lance', 'interests': ['biking', 'technology', 'coffee']}
```

[put](https://langchain-ai.github.io/langgraph/reference/store/#langgraph.store.base.BaseStore.put){: target="_blank"} 메서드를 사용하여 `TypedDict`를 스토어에 저장할 수 있습니다.

```python
import uuid
from langgraph.store.memory import InMemoryStore

# 인-메모리 스토어 초기화
in_memory_store = InMemoryStore()

# 메모리를 저장할 네임스페이스 지정
user_id = "1"
user_id = "1"
namespace_for_memory = (user_id, "memory")

# 키,값으로 네임스페이스에 저장
key = "user_profile"
value = user_profile
in_memory_store.put(namespace_for_memory, key, value)
```

[search](https://langchain-ai.github.io/langgraph/reference/store/#langgraph.store.base.BaseStore.search){: target="_blank"} 메서드를 사용하면, 네임스페이스로 스토어(Store)에서 객체들을 검색할 수 있습니다.

```python
# 검색 
for m in in_memory_store.search(namespace_for_memory):
    print(m.dict())
```

```
# 출력

{'namespace': ['1', 'memory'], 'key': 'user_profile', 'value': {'user_name': 'Lance', 'interests': ['biking', 'technology', 'coffee']}, 'created_at': '2025-04-23T07:08:35.959493+00:00', 'updated_at': '2025-04-23T07:08:35.959496+00:00', 'score': None}
```

또, [get](https://langchain-ai.github.io/langgraph/reference/store/#langgraph.store.base.BaseStore.get){: target="_blank"} 메서드를 사용하면, 네임스페이스와 키로 특정 객체를 가져올 수 있습니다.

```python
# 네임스페이스와 키로 프로필 메모리 정보 가져오기
profile = in_memory_store.get(namespace_for_memory, "user_profile")
profile.value
```

```
# 출력

{'user_name': 'Lance', 'interests': ['biking', 'technology', 'coffee']}
```

### 2.  3.  프로필 스키마를 생성하는 챗봇

특정 스키마로 메모리를 생성하는 챗봇을 구축합니다.

여기서 챗봇은 [사용자와의 대화로부터 메모리를 생성](https://langchain-ai.github.io/langgraph/concepts/memory/#profile){: target="_blank"}키 위해 [구조화된 출력(Structured outputs)](https://python.langchain.com/docs/concepts/structured_outputs/#recommended-usage){: target="_blank"}의 개념을 사용합니다.

LangChain의 [채팅 모델](https://python.langchain.com/docs/concepts/chat_models/){: target="_blank"} 인터페이스에는 관련 스키마에 맞춰서 출력을 자동으로 파싱해 주는 [`with_structured_output`](https://python.langchain.com/docs/concepts/structured_outputs/#recommended-usage){: target="_blank"} 메서드가 있어서 구조화된 출력을 쉽게 적용할 수 있습니다.

```python
_set_env("OPENAI_API_KEY")
```

먼저 `UserProfile` 스키마를 `with_structured_output` 메서드에 전달합니다.

그러면 메시지 리스트를 모델에 전달하고, 스키마에 맞는 구조화된 출력을 얻을 수 있습니다.

```python
from pydantic import BaseModel, Field

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

# 모델 초기화
model = ChatOpenAI(model="gpt-4o", temperature=0)

# 모델에 스키마를 바인딩
model_with_structure = model.with_structured_output(UserProfile)

# 모델을 호출하여 스키마에 맞는 구조화된 출력 생성
structured_output = model_with_structure.invoke([HumanMessage("My name is Lance, I like to bike.")])
structured_output
```

```
# 출력

{'user_name': 'Lance', 'interests': ['biking']}
```

Store 기반 챗봇을 구축했을 때 사용했던 `write_memory` 함수에서 model.invoke 대신에 `model_with_structure.invoke`를 사용하면 스키마에 맞는 프로필을 생성할 수 있습니다.

```python
from IPython.display import Image, display

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.store.base import BaseStore

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.runnables.config import RunnableConfig

# 챗봇 지침
MODEL_SYSTEM_MESSAGE = """You are a helpful assistant with memory that provides information about the user. 
If you have memory for this user, use it to personalize your responses.
Here is the memory (it may be empty): {memory}"""

# 기존과 최신 메모리와 채팅 기록으로 새로운 메모리 생성
CREATE_MEMORY_INSTRUCTION = """Create or update a user profile memory based on the user's chat history. 
This will be saved for long-term memory. If there is an existing memory, simply update it. 
Here is the existing memory (it may be empty): {memory}"""

def call_model(state: MessagesState, config: RunnableConfig, store: BaseStore):

    """스토어에서 메모리를 불러와 챗봇 응답에 활용"""
  
    # config에서 사용자 ID 가져옴
    user_id = config["configurable"]["user_id"]

    # 스토어에서 메모리 조회
    namespace = ("memory", user_id)
    existing_memory = store.get(namespace, "user_memory")

    # 시스템 프롬프트용 메모리 포맷팅
    if existing_memory and existing_memory.value:
        memory_dict = existing_memory.value
        formatted_memory = (
            f"Name: {memory_dict.get('user_name', 'Unknown')}\n"
            f"Interests: {', '.join(memory_dict.get('interests', []))}"
        )
    else:
        formatted_memory = None

    # 시스템 메시지로 메모리 전달을 위한 포맷팅
    system_msg = MODEL_SYSTEM_MESSAGE.format(memory=formatted_memory)

    # 메모리와 채팅 기록을 사용하여 응답 생성
    response = model.invoke([SystemMessage(content=system_msg)]+state["messages"])

    return {"messages": response}

def write_memory(state: MessagesState, config: RunnableConfig, store: BaseStore):

    """대화 내역을 반영하여 메모리를 스토어에 저장"""
  
    # config에서 사용자 ID 가져옴
    user_id = config["configurable"]["user_id"]

    # 스토어에서 메모리 조회
    namespace = ("memory", user_id)
    existing_memory = store.get(namespace, "user_memory")

    # 시스템 프롬프트용 메모리 포맷팅
    if existing_memory and existing_memory.value:
        memory_dict = existing_memory.value
        formatted_memory = (
            f"Name: {memory_dict.get('user_name', 'Unknown')}\n"
            f"Interests: {', '.join(memory_dict.get('interests', []))}"
        )
    else:
        formatted_memory = None
  
    # 지침에서 기존 메모리 포맷팅
    system_msg = CREATE_MEMORY_INSTRUCTION.format(memory=formatted_memory)

    # 모델을 호출해 스키마에 맞는 구조화된 메모리 생성
    new_memory = model_with_structure.invoke([SystemMessage(content=system_msg)]+state['messages'])

    # 기존 사용했던 프로필 메모리 덮어쓰기
    key = "user_memory"
    store.put(namespace, key, new_memory)

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
input_messages = [HumanMessage(content="Hi, my name is Lance and I like to bike around San Francisco and eat at bakeries.")]

# 그래프 실행
for chunk in graph.stream({"messages": input_messages}, config, stream_mode="values"):
    chunk["messages"][-1].pretty_print()
```

```
# 출력

================================ Human Message =================================

Hi, my name is Lance and I like to bike around San Francisco and eat at bakeries.
================================== Ai Message ==================================

Hi Lance! It's great to meet you. Biking around San Francisco sounds like a fantastic way to explore the city, and there are so many amazing bakeries to try. Do you have any favorite bakeries or biking routes in the city?

```

스토어에 저장된 메모리를 확인하면 생성한 스키마와 일치하는 딕셔너리 타입을 확인할 수 있습니다.

```python
# 메모리 저장을 위한 네임스페이스 지정
user_id = "1"
namespace = ("memory", user_id)
existing_memory = across_thread_memory.get(namespace, "user_memory")
existing_memory.value
```

```
# 출력

{'user_name': 'Lance', 'interests': ['biking', 'bakeries', 'San Francisco']}
```

### 2.  4.  복잡한 스키마의 구조화된 출력의 실패

아래 대화 예는 일반적이지는 않지만 팀 워크 훈련에서 팀원 간 `신뢰 구축 훈련(Trust fall)`을 위해서 운영자(Operator)에게 훈련 교관(Customer)이 구체적인 신뢰 구축 훈련 방법(사람이 떨어질 때 다이아몬드 모양을 만들어 사람을 받는 훈련)과 그것을 팀에 전달할 통신 방법(전보, 모스 부호, 기호 신호)에 대한 대화입니다.

스키마는 사용자의 Communication 및 Trust fall에 대한 선호도를 나타내는 Pydantic 모델로 구성합니다.

![신뢰 구축 훈련](assets/posts/2025-06-12-langgraph-long-term-memory-1st/long-term-momory_1st_03.png)

```python
from typing import List, Optional

class OutputFormat(BaseModel):
    preference: str
    sentence_preference_revealed: str

class TelegramPreferences(BaseModel):
    preferred_encoding: Optional[List[OutputFormat]] = None
    favorite_telegram_operators: Optional[List[OutputFormat]] = None
    preferred_telegram_paper: Optional[List[OutputFormat]] = None

class MorseCode(BaseModel):
    preferred_key_type: Optional[List[OutputFormat]] = None
    favorite_morse_abbreviations: Optional[List[OutputFormat]] = None

class Semaphore(BaseModel):
    preferred_flag_color: Optional[List[OutputFormat]] = None
    semaphore_skill_level: Optional[List[OutputFormat]] = None

class TrustFallPreferences(BaseModel):
    preferred_fall_height: Optional[List[OutputFormat]] = None
    trust_level: Optional[List[OutputFormat]] = None
    preferred_catching_technique: Optional[List[OutputFormat]] = None

class CommunicationPreferences(BaseModel):
    telegram: TelegramPreferences
    morse_code: MorseCode
    semaphore: Semaphore

class UserPreferences(BaseModel):
    communication_preferences: CommunicationPreferences
    trust_fall_preferences: TrustFallPreferences

class TelegramAndTrustFallPreferences(BaseModel):
    pertinent_user_preferences: UserPreferences

```

전보(Telegram), 모스 부호(Morse Code), 기호 신호(Semaphore)의 복잡한 스키마를 `with_structured_output` 메서드를 활용해 구성했을 때 gpt-4o와 같은 고성능 모델을 사용하더라도 에러가 발생할 수 있습니다.

여기서 에러가 발생한 이유는 대화에서 Semaphore에 대한 언급이 없어서 발생합니다.

```python
from pydantic import ValidationError

# 모델이 스키마를 바인딩
model_with_structure = model.with_structured_output(TelegramAndTrustFallPreferences)

# 대화
conversation = """Operator: How may I assist with your telegram, sir?
Customer: I need to send a message about our trust fall exercise.
Operator: Certainly. Morse code or standard encoding?
Customer: Morse, please. I love using a straight key.
Operator: Excellent. What's your message?
Customer: Tell him I'm ready for a higher fall, and I prefer the diamond formation for catching.
Operator: Done. Shall I use our "Daredevil" paper for this daring message?
Customer: Perfect! Send it by your fastest carrier pigeon.
Operator: It'll be there within the hour, sir."""

# 모델 실행
try:
    model_with_structure.invoke(f"""Extract the preferences from the following conversation:
    <convo>
    {conversation}
    </convo>""")
except ValidationError as e:
    print(e)
```

```
# 출력 - 에러 발생

1 validation error for TelegramAndTrustFallPreferences
pertinent_user_preferences.communication_preferences.semaphore
  Input should be a valid dictionary or instance of Semaphore [type=model_type, input_value=None, input_type=NoneType]
    For further information visit https://errors.pydantic.dev/2.9/v/model_type
```

### 2.  5.  프로필 스키마 생성 및 업데이트를 위한 Trustcall

복잡한 스키마의 사례처럼 단순한 스키마도 업데이트 과정에서 에러가 발생할 수 있는데 복잡한 스키마는 추출 자체가 어려울 수 있습니다.

또한 앞선 챗봇 경우 프로필 스키마를 매번 비효율적으로 다시 생성했습니다.

특히 스키마에 재생성해야 할 정보가 많을 때는 모델의 토큰 낭비가 발생하고, 심각하게는 프로필을 처음부터 다시 생성하면서 정보가 소실될 수 있습니다.

이런 문제들을 해결하기 위한 것이 `Trustcall`입니다.

`Trustcall`은 LangChain 팀의 Will Fu-Hinthorn이 개발한, JSON 스키마를 업데이트하기 위한 오픈 소스 라이브러리입니다.

먼저, 이 메시지 리스트에 대해 `Trustcall`을 사용해 추출을 어떻게 하는지 살펴보겠습니다.

```python
# 대화
conversation = [HumanMessage(content="Hi, I'm Lance."), 
                AIMessage(content="Nice to meet you, Lance."), 
                HumanMessage(content="I really like biking around San Francisco.")]
```

`create_extractor`를 사용할 때, 모델과 스키마(tools)를 함께 전달합니다.

Trustcall에서는 JSON Object, Python Dictionary 및 Pydantic 모델 등 여러 방식으로 스키마를 전달할 수 있습니다.

내부적으로 `Trustcall`은 `도구 호출(tool calling)` 기능을 활용해, 메시지 리스트로부터 구조화된 출력을 생성합니다.

`Trustcall`이 반드시 구조화된 출력을 위해서, `tool_choice` 인자를 통해 스키마 이름을 포함할 수 있습니다.

위의 대화 예를 Trustcall `extractor`를 사용해 호출할 수 있습니다.

```python
from trustcall import create_extractor

# 스키마 
class UserProfile(BaseModel):
    """User profile schema with typed fields"""
    user_name: str = Field(description="The user's preferred name")
    interests: List[str] = Field(description="A list of the user's interests")

# 모델 초기화
model = ChatOpenAI(model="gpt-4o", temperature=0)

# extractor 생성
trustcall_extractor = create_extractor(
    model,
    tools=[UserProfile],
    tool_choice="UserProfile"
)

# 지침
system_msg = "Extract the user profile from the following conversation"

# extractor 실행
result = trustcall_extractor.invoke({"messages": [SystemMessage(content=system_msg)]+conversation})
```

`extractor`를 호출하면 다음 항목을 받습니다.

* `messages` : tool call을 포함한 AIMessages의 리스트
* `responses` : 스키마와 일치하는 파싱된 tool call 결과들
* `response_metadata` : 기존 tool call을 업데이트하는 경우에 적용되며, 각각의 응답이 어떤 기존 객체와 대응하는지 알려줌

```python
for m in result["messages"]: 
    m.pretty_print()
```

```
# 출력

================================== Ai Message ==================================
Tool Calls:
  UserProfile (call_Fjscb9vnTE3EHhk11TFtUuNz)
 Call ID: call_Fjscb9vnTE3EHhk11TFtUuNz
  Args:
    user_name: Lance
    interests: ['biking']
```

```python
schema = result["responses"]
schema
```

```
# 출력

[UserProfile(user_name='Lance', interests=['biking'])]
```

```python
schema[0].model_dump()
```

```
# 출력

{'user_name': 'Lance', 'interests': ['biking']}
```

```python
result["response_metadata"]
```

```
# 출력

[{'id': 'call_Fjscb9vnTE3EHhk11TFtUuNz'}]
```

이제 `Trustcall`을 사용해 프로필을 어떻게 업데이트할 수 있는지 살펴보겠습니다.

업데이트를 위해서 `Trustcall`에 `메시지 리스트`와 `기존 스키마`를 함께 전달합니다.

여기서 핵심은, `Trustcall`은 기존 스키마와 새로 들어온 메시지를 비교해 변경된 부분만 `JSON Patch` 형식으로 출력하도록 모델에 요청합니다.

이 `JSON Patch` 방식은 전체 스키마를 단순히 덮어쓰는 것보다 오류가 적고, 변경된 부분만 생성하므로 훨씬 효율적입니다.

이를 위해 먼저 기존 스키마를 JSON 타입(dict)으로 직렬화해야 합니다.

Pydantic 모델 인스턴스를 딕셔너리로 직렬화하려면 `model_dump()`를 통해 쉽게 dict로 바꿀 수 있습니다.

이렇게 `직렬화한 스키마`(schema[0].model_dump())와 `스키마 이름`(UserProfile)을 `existing` 인자에 함께 전달하면 `Trustcall`은 새 대화 내용과 비교해 자동으로 필요한 변경만 추출합니다.

```python
# 대화 업데이트
updated_conversation = [HumanMessage(content="Hi, I'm Lance."), 
                        AIMessage(content="Nice to meet you, Lance."), 
                        HumanMessage(content="I really like biking around San Francisco."),
                        AIMessage(content="San Francisco is a great city! Where do you go after biking?"),
                        HumanMessage(content="I really like to go to a bakery after biking."),]

# 지침 업데이트
system_msg = f"""Update the memory (JSON doc) to incorporate new information from the following conversation"""

# Invoke the extractor with the updated instruction and existing profile with the corresponding tool name (UserProfile)
# 위에서 만든 지침과, 기존 프로필(schema[0].model_dump()) 및 해당 도구 이름(UserProfile)로 extractor를 실행
result = trustcall_extractor.invoke({"messages": [SystemMessage(content=system_msg)]+updated_conversation}, 
                                    {"existing": {"UserProfile": schema[0].model_dump()}})  
```

```python
for m in result["messages"]: 
    m.pretty_print()
```

```
# 출력

================================== Ai Message ==================================
Tool Calls:
  UserProfile (call_PtdphIPMwe2XPyxTOojaWjoe)
 Call ID: call_PtdphIPMwe2XPyxTOojaWjoe
  Args:
    user_name: Lance
    interests: ['biking', 'visiting bakeries']
```

```python
result["response_metadata"]
```

```python
# 출력

[{'id': 'call_PtdphIPMwe2XPyxTOojaWjoe'}]
```

```python
updated_schema = result["responses"][0]
updated_schema.model_dump()
```

```
# 출력

{'user_name': 'Lance', 'interests': ['biking', 'visiting bakeries']}
```

이제 앞서 살펴본 복잡한 스키마에도 `Trustcall`을 적용해 다시 테스트합니다.

```python
bound = create_extractor(
    model,
    tools=[TelegramAndTrustFallPreferences],
    tool_choice="TelegramAndTrustFallPreferences",
)

# 대화
conversation = """Operator: How may I assist with your telegram, sir?
Customer: I need to send a message about our trust fall exercise.
Operator: Certainly. Morse code or standard encoding?
Customer: Morse, please. I love using a straight key.
Operator: Excellent. What's your message?
Customer: Tell him I'm ready for a higher fall, and I prefer the diamond formation for catching.
Operator: Done. Shall I use our "Daredevil" paper for this daring message?
Customer: Perfect! Send it by your fastest carrier pigeon.
Operator: It'll be there within the hour, sir."""

result = bound.invoke(
    f"""Extract the preferences from the following conversation:
<convo>
{conversation}
</convo>"""
)

# Extract the preferences 추출된 선호도 정보 확인
result["responses"][0]
```

```
# 출력

TelegramAndTrustFallPreferences(pertinent_user_preferences=UserPreferences(communication_preferences=CommunicationPreferences(telegram=TelegramPreferences(preferred_encoding=[OutputFormat(preference='standard encoding', sentence_preference_revealed='standard encoding')], favorite_telegram_operators=None, preferred_telegram_paper=[OutputFormat(preference='Daredevil', sentence_preference_revealed='Daredevil')]), morse_code=MorseCode(preferred_key_type=[OutputFormat(preference='straight key', sentence_preference_revealed='straight key')], favorite_morse_abbreviations=None), semaphore=Semaphore(preferred_flag_color=None, semaphore_skill_level=None)), trust_fall_preferences=TrustFallPreferences(preferred_fall_height=[OutputFormat(preference='higher', sentence_preference_revealed='higher')], trust_level=None, preferred_catching_technique=[OutputFormat(preference='diamond formation', sentence_preference_revealed='diamond formation')])))
```

에러가 발생하지 않고 출력이 되었습니다.

`with_structured_output`과 `Trustcall`은 둘 다 Pydantic 모델을 사용하는데 semaphore 예외 처리가 안 되어서 `with_structured_output`은 에러가 발생했습니다. 이유는 `with_structured_output`은 검증이 엄격합니다.

`Trustcall`은 전체를 재검증하지 않고 부분만 검증하고, optional이면 None을 생성합니다.

### 2.  6.  프로필 스키마 업데이트가 가능한 챗봇

이제 `Trustcall`을 챗봇에 적용해서, 메모리 프로필을 생성하고 업데이트합니다.

```python
from IPython.display import Image, display

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.runnables.config import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.base import BaseStore

# 모델 초기화
model = ChatOpenAI(model="gpt-4o", temperature=0)

# 스키마 
class UserProfile(BaseModel):
    """ 사용자 프로필 """
    user_name: str = Field(description="The user's preferred name")
    user_location: str = Field(description="The user's location")
    interests: list = Field(description="A list of the user's interests")

# extractor 생성
trustcall_extractor = create_extractor(
    model,
    tools=[UserProfile],
    tool_choice="UserProfile", # UserProfile 도구 사용
)

# 챗봇 지침
MODEL_SYSTEM_MESSAGE = """You are a helpful assistant with memory that provides information about the user. 
If you have memory for this user, use it to personalize your responses.
Here is the memory (it may be empty): {memory}"""

# 정보 추출 지침
TRUSTCALL_INSTRUCTION = """Create or update the memory (JSON doc) to incorporate information from the following conversation:"""

def call_model(state: MessagesState, config: RunnableConfig, store: BaseStore):

    """스토어에서 메모리를 불러와 챗봇 응답에 활용"""
  
    # 사용자 ID를 config에서 가져옴
    user_id = config["configurable"]["user_id"]

    # 스토어에서 메모리 불러오기
    namespace = ("memory", user_id)
    existing_memory = store.get(namespace, "user_memory")

    # 시스템 프롬프트용 메모리 포맷팅
    if existing_memory and existing_memory.value:
        memory_dict = existing_memory.value
        formatted_memory = (
            f"Name: {memory_dict.get('user_name', 'Unknown')}\n"
            f"Location: {memory_dict.get('user_location', 'Unknown')}\n"
            f"Interests: {', '.join(memory_dict.get('interests', []))}"  
        )
    else:
        formatted_memory = None

    # 시스템 프롬프트용 메모리 포맷팅
    system_msg = MODEL_SYSTEM_MESSAGE.format(memory=formatted_memory)

    # 메모리와 채팅 기록을 함께 모델에 전달하여 응답 생성
    response = model.invoke([SystemMessage(content=system_msg)]+state["messages"])

    return {"messages": response}

def write_memory(state: MessagesState, config: RunnableConfig, store: BaseStore):

    """채팅 기록을 반영하여 메모리를 스토어에 저장"""
  
    # 사용자 ID를 config에서 가져옴
    user_id = config["configurable"]["user_id"]

    # 스토어에서 기존 메모리 불러오기
    namespace = ("memory", user_id)
    existing_memory = store.get(namespace, "user_memory")
  
    # 리스트에서 값으로 프로필을 가져와, JSON 문서로 변환
    existing_profile = {"UserProfile": existing_memory.value} if existing_memory else None
  
    # extractor 실행
    result = trustcall_extractor.invoke({"messages": [SystemMessage(content=TRUSTCALL_INSTRUCTION)]+state["messages"], "existing": existing_profile})
  
    # 업데이트된 프로필을 JSON 오브젝트로 추출
    updated_profile = result["responses"][0].model_dump()

    # 업데이트된 프로필 저장
    key = "user_memory"
    store.put(namespace, key, updated_profile)

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
# 단기(스레드 내) 메모리를 위한 스레드 ID를 제공
# 장기(스레드 간) 메모리를 위한 사용자 ID를 제공
config = {"configurable": {"thread_id": "1", "user_id": "1"}}

# 사용자 입력
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

Hello, Lance! It's nice to meet you. How can I assist you today?
```

```python
# 사용자 입력 
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

That sounds like a fantastic way to explore the city! San Francisco has some great biking routes. Do you have a favorite trail or area you like to ride in?
```

```python
# 저정할 메모리의 네임스페이스 지정
user_id = "1"
namespace = ("memory", user_id)
existing_memory = across_thread_memory.get(namespace, "user_memory")
existing_memory.dict()
```

```
# 출력

{'namespace': ['memory', '1'],
 'key': 'user_memory',
 'value': {'user_name': 'Lance',
  'user_location': 'San Francisco',
  'interests': ['biking']},
 'created_at': '2025-04-23T07:19:02.018937+00:00',
 'updated_at': '2025-04-23T07:19:02.018939+00:00'}
```

```python
# 사용자 프로필을 JSON 오브젝트로 저장
existing_memory.value
```

```
# 출력

{'user_name': 'Lance',
 'user_location': 'San Francisco',
 'interests': ['biking']}
```

```python
# 사용자 입력
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

San Francisco has some amazing bakeries to explore. Do you have a favorite bakery, or are you looking for recommendations to try on your next ride?
```

새로운 스레드에서 대화를 추가하겠습니다.

`thread_id`를 바꾸면 단기 메모리는 새로 시작되지만, `user_id`를 유지하면 `프로필 기반 장기 메모리`를 사용하여 `컨텍스트`를 유지할 수 있습니다.

```python
# 단기(스레드 내) 메모리를 위한 스레드 ID를 제공
# 장기(스레드 간) 메모리를 위한 사용자 ID를 제공
config = {"configurable": {"thread_id": "2", "user_id": "1"}}

# 사용자 입력
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

Since you're in San Francisco and enjoy going to bakeries, here are a few recommendations you might like:

1. **Tartine Bakery** - Known for its delicious bread and pastries, it's a must-visit for any bakery enthusiast.

2. **B. Patisserie** - Offers a delightful selection of French pastries, including their famous kouign-amann.

3. **Arsicault Bakery** - Renowned for its croissants, which have been praised as some of the best in the country.

4. **Craftsman and Wolves** - Offers innovative pastries and their signature "Rebel Within" muffin, which has a soft-cooked egg inside.

5. **Mr. Holmes Bakehouse** - Famous for their cruffins and a variety of other creative pastries.

These spots should satisfy your bakery cravings while you're biking around the city!
```

## 정리

`LangGraph Memory Store`는 `key-value` 기반의 `Store`로, `스레드 간` 정보 공유를 위해 `사용자 ID`를 `네임스페이스`로 활용합니다.

이 구조 덕분에 동일 사용자의 정보를 여러 채팅 세션에서 일관되게 관리할 수 있습니다.

`시맨틱 메모리`는 `프로필`과 `컬렉션`으로 관리되며, 이들 스키마의 생성 및 업데이트는 `Trustcall`을 통해 자동화됩니다.

특히 `Trustcall`은 LLM이 구조화된 데이터를 생성할 때 발생할 수 있는 JSON 오류나 전체 덮어쓰기 문제를 `JSON Patch` 방식으로 해결하여, 기존 데이터의 필요한 부분만 유연하게 업데이트할 수 있습니다.

다음 포스팅에는 컬렉션을 살펴보고 ReACT 기반 ToDo 리스트 에이전트를 구축해 보겠습니다.

## References

* [LangGraph 공식 문서](https://langchain-ai.github.io/langgraph){: target="_blank"}
* [LangChain Academy](https://github.com/langchain-ai/langchain-academy){: target="_blank"}
