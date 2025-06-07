---
title: LangGraph 장기 메모리 (2)
date: 2025-06-02 12:15:43 +/-TTTT
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

> 학습할 리소스는 [LangChain Academy Github](https://github.com/langchain-ai/langchain-academy){: target="_blank"}를 사용합니다.
> {: .prompt-info }

## 4.   메모리 에이전트 (Memory Agent) 구축 (w/ Semantic Memory)

`시맨틱 메모리`는 `프로필(Profile)`과 `컬렉션(Collection)`의 두 가지 방식으로 관리됩니다.

AI 에이전트에서는 사용자에 대한 정보, 예를 들어 이름, 직책, 선호도 등을 기억하는 데 사용됩니다.

이는 에이전트가 사용자와의 상호작용을 통해 얻은 정보를 기반으로 응답을 개선하고 개인화하는 데 도움을 줍니다.

> 여기서 `시맨틱` 개념은 `시맨틱 검색(Semantic Search)`과 다릅니다. 시맨틱 검색은 `의미(meaning, 일반적으로 임베딩)`를 사용하여 유사한 콘텐츠를 찾는 기법입니다.
{: .prompt-warning }

> [컬렉션](https://langchain-ai.github.io/langgraph/concepts/memory/#collection){: target="_blank"}은 하나의 JSON 문서(프로필) 대신에 여러 개의 ‘작은’ 메모리 항목을 개별 문서로 저장해 나중에 하나의 컬렉션으로 관리하는 방식입니다. `컬렉션`은 추후 더 자세히 살펴보겠습니다.
{: .prompt-info }

장기 메모리 기반 에이전트를 구축하고 `프로필`과 `컬렉션`의 두 스키마를 업데이트하는 방법으로 `Trustcall`을 살펴보겠습니다.

먼저 환경 구성을 합니다.

우리의 에이전트인 task_mAIstro는 ToDo 목록을 관리하는 데 도움을 줄 것입니다!

기존에 만든 챗봇은 항상 대화를 반영하고 메모리를 저장했지만,
task_mAIstro는 언제 메모리(ToDo 항목)를 저장할지 스스로 결정합니다.

기존 챗봇은 한 종류의 메모리(프로필 또는 컬렉션)만 저장했지만,
task_mAIstro는 사용자 프로필 또는 ToDo 항목 컬렉션 중 어느 쪽에 저장할지도 결정할 수 있습니다.

시맨틱 메모리 외에도, task_mAIstro는 절차적 메모리도 관리합니다.
이를 통해 사용자는 ToDo 항목 생성에 대한 선호도를 업데이트할 수 있습니다.

```python
# 환경 구성
%%capture --no-stderr
%pip install -U langchain_openai langgraph trustcall langchain_core
```

```python
import os, getpass

def _set_env(var: str):
    # Check if the variable is set in the OS environment
    env_value = os.environ.get(var)
    if not env_value:
        # If not set, prompt the user for input
        env_value = getpass.getpass(f"{var}: ")
  
    # Set the environment variable for the current process
    os.environ[var] = env_value

_set_env("LANGSMITH_API_KEY")
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "langchain-academy"
```

```python
_set_env("OPENAI_API_KEY")
```

### 4.  1.  Trustcall 업데이트에 대한 가시성 (Visability)

`Trustcall`은 LangGraph 기반 오픈소스 라이브러리로, LLM이 복잡한 구조의 JSON 출력을 생성, 수정할 때 발생하는 오류를 줄입니다.

기존 방식은 전체 JSON을 한 번에 생성하려다 보니 오류가 발생하기 쉬웠습니다.

`Trustcall`은 이러한 문제를 해결하기 위해 LLM에 `JSON Patch` 형식의 수정 지시를 생성하도록 요청합니다.

`JSON Patch`는 부분적으로 수정이 가능하며, 반복적인 오류 수정이 용이합니다.

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

![ReACT ToDo 에이전트 프로세스](assets/drafts/2025-05-28-langgraph-long-term-memory-1st/long-term-momory_03.png)
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

    # Trustcall Trustcall에서 발생한 도구 호출 내역 확인을 위한 Spy 초기화
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

![ToDo 에이전트 그래프](assets/drafts/2025-05-28-langgraph-long-term-memory-1st/long-term-momory_04.png)
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

다음 포스팅에서 `장기 메모리 프로필`과 `컬렉션`에 대해서 자세히 살펴보겠습니다.

## References

* [LangGraph 공식 문서](https://langchain-ai.github.io/langgraph){: target="_blank"}
* [LangChain Academy](https://github.com/langchain-ai/langchain-academy){: target="_blank"}
