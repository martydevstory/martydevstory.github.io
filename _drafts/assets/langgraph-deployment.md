---
title: LangGraph 배포하기
date: 2025-06-10 12:15:43 +/-TTTT
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
series_order: 10
---
이번 포스팅에서는 이제까지 구현한 LangGraph 어플리케이션을 배포해보겠습니다.

> 학습할 리소스는 [LangChain Academy Github](https://github.com/langchain-ai/langchain-academy){: target="_blank"}를 사용합니다.
> {: .prompt-info }

## 1.   배포 생성하기

이전 포스트에서 만든 `task_maistro` 어플리케이션을 배포해보겠습니다.

어플리케이션 샘플 코드는 [`module 5`](https://github.com/langchain-ai/langchain-academy/tree/main/module-5https:/) 디렉터리에 있습니다.

### 1.  1.  코드 구조

LangGraph 플랫폼 배포 생성하기 위해서는 [다음 항목를 제공해야 합니다](https://langchain-ai.github.io/langgraph/concepts/application_structure/):

* LangGraph API 구성 파일 (예: `langgraph.json`)
* 애플리케이션 로직을 구현한 그래프 파일 (예: `task_maistro.py`)
* 애플리케이션 실행에 필요한 의존성을 나열한 파일 (예: `requirements.txt`)
* 애플리케이션 실행에 필요한 환경 변수를 지정하는 파일 (예: `.env` 또는 `docker-compose.yml`)

해당 파일은 [`module-6/deployment`](https://github.com/langchain-ai/langchain-academy/tree/main/module-6/deployment) 디렉터리에 준비되어 있습니다.

### 1.  2.  CLI

[LangGraph CLI](https://langchain-ai.github.io/langgraph/concepts/langgraph_cli/)는 LangGraph 플랫폼 배포를 생성하기 위한 명령줄 인터페이스로 아래와 같이 실행합니다.

```python
%%capture --no-stderr
%pip install -U langgraph-cli
```

[자체 호스팅 배포](https://langchain-ai.github.io/langgraph/how-tos/deploy-self-hosted/#how-to-do-a-self-hosted-deployment-of-langgraph)를 만들기 위해, 다음 단계를 진행합니다.

#### 1. 2.  1.  LangGraph 서버용 Docker 이미지 빌드하기

먼저 LangGraph CLI를 사용하여 [LangGraph 서버](https://docs.google.com/presentation/d/18MwIaNR2m4Oba6roK_2VQcBE_8Jq_SI7VHTXJdl7raU/edit#slide=id.g313fb160676_0_32)용 Docker 이미지를 생성합니다.

이 명령은 그래프와 의존성을 Docker 이미지 하나로 패키징합니다.

Docker 이미지는 애플리케이션 실행에 필요한 코드와 의존성을 포함하는 컨테이너 템플릿입니다.

[Docker](https://docs.docker.com/engine/install/)가 설치되어 있는지 확인한 후, 다음 명령으로 `my-image`라는 이름의 Docker 이미지를 생성합니다:

```bash
$ cd module-6/deployment
$ langgraph build -t my-image
```

#### 1. 2.  2.  Redis 및 PostgreSQL 설정하기

이미 `Redis`와 `PostgreSQL`이 로컬이나 다른 서버에서 실행 중이면, `Redis`와 `PostgreSQL`의 `URI`를 지정하여 LangGraph 서버 컨테이너만 [단독으로 실행](https://langchain-ai.github.io/langgraph/how-tos/deploy-self-hosted/#running-the-application-locally)할 수 있습니다:

```bash
docker run \
    --env-file .env \
    -p 8123:8000 \
    -e REDIS_URI="foo" \
    -e DATABASE_URI="bar" \
    -e LANGSMITH_API_KEY="baz" \
    my-image
```

없다면 제공된 `docker-compose.yml` 파일을 사용하여 세 개의 개별 컨테이너를 생성할 수도 있습니다:

* `langgraph-redis`: 공식 Redis 이미지를 사용하는 컨테이너
* `langgraph-postgres`: 공식 PostgreSQL 이미지를 사용하는 컨테이너
* `langgraph-api`: 미리 빌드한 API 이미지를 사용하는 컨테이너

`docker-compose-example.yml`을 복사한 뒤, 다음 환경 변수를 추가하여 `task_maistro` 어플리케이션을 실행합니다:

* `IMAGE_NAME` (예: `my-image`)
* `LANGSMITH_API_KEY`
* `OPENAI_API_KEY`

다음은 `docker-compose-example.yml` 샘플입니다. 소스 파일에서 확인할 수 있습니다.

```yaml
 # IMAGE_NAME, LANGSMITH_API_KEY, OPENAI_API_KEY 본인에 맞게 설정합니다.
volumes:
    langgraph-data:
        driver: local
services:
    langgraph-redis:
        image: redis:6
        healthcheck:
            test: redis-cli ping
            interval: 5s
            timeout: 1s
            retries: 5
        ports:
            - "6379:6379"
    langgraph-postgres:
        image: postgres:16
        ports:
            - "5432:5432"
        environment:
            POSTGRES_DB: postgres
            POSTGRES_USER: postgres
            POSTGRES_PASSWORD: postgres
        volumes:
            - langgraph-data:/var/lib/postgresql/data
        healthcheck:
            test: pg_isready -U postgres
            start_period: 10s
            timeout: 1s
            retries: 5
            interval: 5s
    langgraph-api:
        image: "my-image" # API 이미지 이름 설정
        ports:
            - "8123:8000"
        depends_on:
            langgraph-redis:
                condition: service_healthy
            langgraph-postgres:
                condition: service_healthy
        environment:
            REDIS_URI: redis://langgraph-redis:6379
            OPENAI_API_KEY: "your_openai_api_key" # OpenAI 설정
            LANGSMITH_API_KEY: "your_langchain_api_key" # LangSmith 설정
            POSTGRES_URI: postgres://postgres:postgres@langgraph-postgres:5432/postgres?sslmode=disable

```

그런 다음, [배포를 시작](https://langchain-ai.github.io/langgraph/how-tos/deploy-self-hosted/#using-docker-compose)합니다:

```bash
$ cd module-6/deployment
$ docker compose up
```

## 2.   LangGraph 플랫폼 배포에 연결

### 2.  1.  배포 생성 확인

`task_maistro` 어플리케이션 배포를 생성했습니다.

그리고 LangGraph 서버와 `task_maistro` 그래프를 포함한 Docker 이미지를 빌드하기 위해 `LangGraph CLI`를 사용했습니다.

제공된 `docker-compose.yml` 파일을 사용하여 정의된 서비스에 따라` langgraph-redis`, `langgraph-postgres`, `langgraph-api` 세 개의 개별 컨테이너를 생성하고 배포를 했습니다.

배포 실행이 완료되면 다음 경로를 통해 배포된 서비스를 이용할 수 있습니다.

* `API` : [http://localhost:8123](http://localhost:8123)
* `Docs` : [http://localhost:8123/docs](http://localhost:8123/docs)
* `LangGraph Studio` : [https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:8123](https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:8123)

### 2.  2.  API 사용하기

LangGraph 서버는 배포된 에이전트와 상호작용을 위한 [다양한 API 엔드포인트](https://github.com/langchain-ai/agent-protocol)를 제공합니다.

이 엔드포인트들은 [공통적인 에이전트 요구 사항을 기반으로 API를 그룹화](https://github.com/langchain-ai/agent-protocol)할 수 있습니다:

* `Runs` : 원자적(단일) 에이전트 실행
* `Threads` : 다중-턴 상호작용 또는 휴먼-인-더-루프
* `Store` : 장기 메모리

그리고 [API 문서 페이지](http://localhost:8123/docs#tag/thread-runs) 직접 요청을 통해 테스트할 수 있습니다.

### 2.  3.  SDK

[LangGraph SDK](https://langchain-ai.github.io/langgraph/concepts/sdk/) (Python 및 JS)는 위에서 설명한 LangGraph Server API와 상호작용할 수 있도록 개발자 친화적인 인터페이스를 제공합니다.

```python
%%capture --no-stderr
%pip install -U langgraph_sdk
```

```python
from langgraph_sdk import get_client

# SDK 연결
url_for_cli_deployment = "http://localhost:8123"
client = get_client(url=url_for_cli_deployment)
```

### 2.  4.  Remote Graph

LangGraph 라이브러리에서 작업하는 경우, [Remote Graph](https://langchain-ai.github.io/langgraph/how-tos/use-remote-graph/)를 사용하면 그래프에 직접 연결할 수 있습니다.

```python
%%capture --no-stderr
%pip install -U langchain_openai langgraph langchain_core
```

```python
from langgraph.pregel.remote import RemoteGraph
from langchain_core.messages import convert_to_messages
from langchain_core.messages import HumanMessage, SystemMessage

# remote graph 연결
url = "http://localhost:8123"
graph_name = "task_maistro" 
remote_graph = RemoteGraph(graph_name, url=url)
```

### 2.  5.  단일 실행 (Runs)

`run`은 그래프의 [단일 실행](https://github.com/langchain-ai/agent-protocol?tab=readme-ov-file#runs-atomic-agent-executions)을 나타냅니다. 클라이언트가 요청을 보낼 때마다 다음 과정이 발생합니다:

1. `HTTP 워커`가 고유한 `run ID`를 생성합니다.
2. 해당 `run`과 그 결과는 `PostgreSQL`에 저장됩니다.
3. 이러한 `run`을 조회하여 다음과 같은 작업을 할 수 있습니다:
   * 상태 확인
   * 결과 조회
   * 실행 이력 추적

여러 종류의 `run`에 대해 자세히 다루는 [How To 가이드](https://langchain-ai.github.io/langgraph/how-tos/#runs)를 확인할 수 있습니다.

이제 `run`을 통해 할 수 있는 몇 가지 흥미로운 작업을 살펴보겠습니다.

#### 2. 5.  1.  백그라운드 실행 (Background Runs)

LangGraph 서버는 두 가지 타입의 run을 지원합니다:

* `Fire and forget` – 백그라운드에서 run을 실행하고, 완료 여부를 기다리지 않음
* `Waiting on a reply (blocking or polling)` – run을 실행하고, 그 출력이 나올 때까지 기다리거나 스트리밍함

두 가지 타입의 `Background run`과 `polling`은 장시간 실행되는 에이전트 작업에서 특히 유용합니다.

어떻게 동작하는지 자세한 내용은 [링크](https://langchain-ai.github.io/langgraph/cloud/how-tos/background_run/#check-runs-on-thread)에서 확인할 수 있습니다.

```python
# 스레드 생성
thread = await client.threads.create()
thread
```

```
# 출력

{'thread_id': '38bffea1-6fab-4bfb-af79-b330fa0c9833',
 'created_at': '2025-04-24T07:21:03.696279+00:00',
 'updated_at': '2025-04-24T07:21:03.696279+00:00',
 'metadata': {},
 'status': 'idle',
 'config': {},
 'values': None,
 'interrupts': {}}
```

```python
# 스레드에 존재하는 기존 run(실행) 목록 확인
thread = await client.threads.create()
runs = await client.runs.list(thread["thread_id"])
print(runs)
```

```
# 출력

[]
```

```python
# ToDo를 생성하고 내 user_id에 저장했는지 확인
user_input = "Add a ToDo to finish booking travel to Hong Kong by end of next week. Also, add a ToDo to call parents back about Thanksgiving plans."
config = {"configurable": {"user_id": "Test"}}
graph_name = "task_maistro" 
run = await client.runs.create(thread["thread_id"], graph_name, input={"messages": [HumanMessage(content=user_input)]}, config=config)
```

```python
# 새로운 스레드와 새로운 run(실행) 시작
thread = await client.threads.create()
user_input = "Give me a summary of all ToDos."
config = {"configurable": {"user_id": "Test"}}
graph_name = "task_maistro" 
run = await client.runs.create(thread["thread_id"], graph_name, input={"messages": [HumanMessage(content=user_input)]}, config=config)
```

```python
# run(실행) 상태 확인
print(await client.runs.get(thread["thread_id"], run["run_id"]))
```

```
# 출력

{'run_id': '1f020dcb-7a71-6e76-a33f-922fec104bd7', 'thread_id': '83e68d06-9f3a-497b-b869-3319c54676c0', 'assistant_id': 'ea4ebafa-a81d-5063-a5fa-67c755d98a21', 'created_at': '2025-04-24T07:21:19.457201+00:00', 'updated_at': '2025-04-24T07:21:21.116719+00:00', 'metadata': {'assistant_id': 'ea4ebafa-a81d-5063-a5fa-67c755d98a21'}, 'status': 'success', 'kwargs': {'input': {'messages': [{'id': None, 'name': None, 'type': 'human', 'content': 'Give me a summary of all ToDos.', 'example': False, 'additional_kwargs': {}, 'response_metadata': {}}]}, 'config': {'metadata': {'created_by': 'system', 'assistant_id': 'ea4ebafa-a81d-5063-a5fa-67c755d98a21'}, 'configurable': {'run_id': '1f020dcb-7a71-6e76-a33f-922fec104bd7', 'user_id': 'Test', 'graph_id': 'task_maistro', 'thread_id': '83e68d06-9f3a-497b-b869-3319c54676c0', 'user-agent': 'langgraph-sdk-py/0.1.63', 'assistant_id': 'ea4ebafa-a81d-5063-a5fa-67c755d98a21', 'langgraph_auth_user': None, 'langgraph_auth_user_id': '', 'langgraph_auth_permissions': []}}, 'command': None, 'webhook': None, 'subgraphs': False, 'temporary': False, 'stream_mode': ['values'], 'feedback_keys': None, 'interrupt_after': None, 'interrupt_before': None}, 'multitask_strategy': 'reject'}
```

아직 실행 중이기 때문에 `'status': 'pending'`임을 알 수 있습니다.

만약 실행이 완료될 때까지 기다리고 싶다면, 즉 블로킹(blocking) 방식으로 `run`을 처리하고 싶다면

`client.runs.join`을 사용하여 실행이 끝날 때까지 대기할 수 있습니다.

이렇게 하면 현재 스레드에서 실행 중인 `run`이 완료될 때까지 새로운 `run`이 시작되지 않도록 보장할 수 있습니다.

```python
# run(실행)이 완료될 때까지 대기
await client.runs.join(thread["thread_id"], run["run_id"])
print(await client.runs.get(thread["thread_id"], run["run_id"]))
```

```
# 출력

{'run_id': '1efa2c00-63e4-6f4a-9c5b-ca3f5f9bff07', 'thread_id': '641c195a-9e31-4250-a729-6b742c089df8', 'assistant_id': 'ea4ebafa-a81d-5063-a5fa-67c755d98a21', 'created_at': '2024-11-14T19:38:29.394777+00:00', 'updated_at': '2024-11-14T19:38:29.394777+00:00', 'metadata': {}, 'status': 'success', 'kwargs': {'input': {'messages': [{'id': None, 'name': None, 'type': 'human', 'content': 'Give me a summary of all ToDos.', 'example': False, 'additional_kwargs': {}, 'response_metadata': {}}]}, 'config': {'metadata': {'created_by': 'system'}, 'configurable': {'run_id': '1efa2c00-63e4-6f4a-9c5b-ca3f5f9bff07', 'user_id': 'Test', 'graph_id': 'task_maistro', 'thread_id': '641c195a-9e31-4250-a729-6b742c089df8', 'assistant_id': 'ea4ebafa-a81d-5063-a5fa-67c755d98a21'}}, 'webhook': None, 'subgraphs': False, 'temporary': False, 'stream_mode': ['values'], 'feedback_keys': None, 'interrupt_after': None, 'interrupt_before': None}, 'multitask_strategy': 'reject'}
```

이제 `run`이 완료되었기 때문에 `'status': 'success'`로 표시됩니다.

### 2.  5.  2.  스트리밍 실행 (Streaming Runs)

클라이언트가 스트리밍 요청을 보낼 때마다 다음과 같은 과정이 발생합니다:

1. `HTTP 워커`가 고유한 `run ID`를 생성합니다.
2. `큐(Queue) 워커`가 해당 `run` 작업을 시작합니다.
3. 실행 중에 `큐 워커`는 `Redis`에 업데이트를 게시합니다.
4. `HTTP 워커`는 해당 `run`에 대한 `Redis`의 업데이트를 구독하고, 이를 클라이언트에게 전달합니다.

이 과정을 통해 스트리밍이 가능해집니다.

[스트리밍](https://langchain-ai.github.io/langgraph/how-tos/#streaming_1)에 대해서는 이전 포스팅에서 살펴보았습니다, 여기서는 그 중 하나인 `토큰 스트리밍` 방식을 살펴보겠습니다.

`토큰`을 클라이언트로 실시간 전송하는 것은, 실행에 시간이 오래 걸릴 수 있는 프로덕션 에이전트와 작업할 때 특히 유용합니다.

`stream_mode="messages-tuple"` 옵션을 사용하여 [토큰을 스트리밍](https://langchain-ai.github.io/langgraph/cloud/how-tos/stream_messages/#setup)합니다.

```python
user_input = "What ToDo should I focus on first."
async for chunk in client.runs.stream(thread["thread_id"], 
                                      graph_name, 
                                      input={"messages": [HumanMessage(content=user_input)]},
                                      config=config,
                                      stream_mode="messages-tuple"):

    if chunk.event == "messages":
        print("".join(data_item['content'] for data_item in chunk.data if 'content' in data_item), end="", flush=True)
```

```
# 출력

You should focus on "Call parents back about Thanksgiving plans" first. It has no specified deadline, but it is a shorter task with an estimated time to complete of 15 minutes. Completing this task quickly will allow you to focus on the more time-consuming task of booking your travel to Hong Kong.
```

### 2.  6.  스레드

`run`이 그래프의 `단일 실행`만을 의미하는 반면, `thread`는 `멀티-턴(multi-turn)`의 상호작용을 지원합니다.

클라이언트가 `thread_id`와 함께 그래프 실행을 요청하면, 서버는 해당 `run`의 모든 [체크포인트](https://langchain-ai.github.io/langgraph/concepts/persistence/#checkpoints) (단계)를 `Postgres` 데이터베이스의 해당 `thread`에 저장합니다.

서버는 [생성된 thread의 상태를 확인](https://langchain-ai.github.io/langgraph/cloud/how-tos/check_thread_status/)할 수 있게 해줍니다.

#### 2. 6.  1.  스레드 상태 확인

또한, 특정 `thread`에 저장된 상태 [체크포인트](https://langchain-ai.github.io/langgraph/concepts/persistence/#checkpoints)에 쉽게 접근할 수 있습니다.

```python
thread_state = await client.threads.get_state(thread['thread_id'])
for m in convert_to_messages(thread_state['values']['messages']):
    m.pretty_print()
```

```
# 출력

================================ Human Message =================================

Give me a summary of all ToDos.
================================== Ai Message ==================================

Here's a summary of your current ToDo list:

1. **Task**: Finish booking travel to Hong Kong
   - **Status**: Not started
   - **Deadline**: May 3, 2025
   - **Solutions**:
     - Check flight prices on Skyscanner
     - Book hotel through Booking.com
     - Arrange airport transfer
   - **Estimated Time to Complete**: 120 minutes

If you need any updates or changes, feel free to let me know!
================================ Human Message =================================

What ToDo should I focus on first.
================================== Ai Message ==================================

You should focus on "Call parents back about Thanksgiving plans" first. It has no specified deadline, but it is a shorter task with an estimated time to complete of 15 minutes. Completing this task quickly will allow you to focus on the more time-consuming task of booking your travel to Hong Kong.
```

#### 2.  6.  2.  스레드 복사

기존의 `thread`를 [복사](https://langchain-ai.github.io/langgraph/cloud/how-tos/copy_threads/)(fork)할 수도 있습니다.

이렇게 하면 기존 `thread`의 히스토리는 그대로 유지되지만, 원래 `thread`에 영향을 주지 않는 독립적인 `run`을 새로 만들 수 있습니다.

```python
# 스레드 복사
copied_thread = await client.threads.copy(thread['thread_id'])
```

```python
# 복사된 스레드의 상태 확인
copied_thread_state = await client.threads.get_state(copied_thread['thread_id'])
for m in convert_to_messages(copied_thread_state['values']['messages']):
    m.pretty_print()
```

```
# 출력

================================ Human Message =================================

Give me a summary of all ToDos.
================================== Ai Message ==================================

Here's a summary of your current ToDo list:

1. **Task**: Finish booking travel to Hong Kong
   - **Status**: Not started
   - **Deadline**: May 3, 2025
   - **Solutions**:
     - Check flight prices on Skyscanner
     - Book hotel through Booking.com
     - Arrange airport transfer
   - **Estimated Time to Complete**: 120 minutes

If you need any updates or changes, feel free to let me know!
================================ Human Message =================================

What ToDo should I focus on first.
================================== Ai Message ==================================

You should focus on "Call parents back about Thanksgiving plans" first. It has no specified deadline, but it is a shorter task with an estimated time to complete of 15 minutes. Completing this task quickly will allow you to focus on the more time-consuming task of booking your travel to Hong Kong.
```

### 2.  7.  휴먼-인-더-루프

이전 포스팅에서 [Human in the loop](https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/)에 대해 다뤘습니다. 서버는 앞서 다뤘던 모든 휴먼-인-더-루프 기능을 지원합니다.

예를 들어, [이전 체크포인트에서 그래프 실행을 검색, 편집, 계속 진행](https://langchain-ai.github.io/langgraph/concepts/persistence/#capabilities)할 수 있습니다.

```python
# 스레드의 히스토리 가져오기
states = await client.threads.get_history(thread['thread_id'])

# 포크할 상태 업데이트 선택
to_fork = states[-2]
to_fork['values']
```

```
# 출력

{'messages': [{'content': 'Give me a summary of all ToDos.',
   'additional_kwargs': {},
   'response_metadata': {},
   'type': 'human',
   'name': None,
   'id': 'fc1895a1-7bae-416c-b7d0-d5c2798338b8',
   'example': False}]}
```

```python
to_fork['values']['messages'][0]['id']
```

```
# 출력

'fc1895a1-7bae-416c-b7d0-d5c2798338b8'
```

```python
to_fork['next']
```

```
# 출력

['task_mAIstro']
```

```python
to_fork['checkpoint_id']
```

```
# 출력

'1f020dcb-7a8c-624d-8000-a673d205212c'
```

상태를 편집합니다. 이전 포스팅에서 `messages`에 대한 `reducer`가 어떻게 동작하는지 다시 확인하면:

* `message ID`를 제공하지 않으면 `메시지가 추가`됩니다.
* `message ID`를 제공하면, 상태에 메시지를 추가하는 것이 아니라 해당 `메시지를 덮어씁니다`.

```python
forked_input = {"messages": HumanMessage(content="Give me a summary of all ToDos that need to be done in the next week.",
                                         id=to_fork['values']['messages'][0]['id'])}

# 상태를 업데이트하여, 스레드에 새로운 체크포인트 생성
forked_config = await client.threads.update_state(
    thread["thread_id"],
    forked_input,
    checkpoint_id=to_fork['checkpoint_id']
)
```

```python
# 스레드에서 새 체크포인트부터 그래프 실행
async for chunk in client.runs.stream(thread["thread_id"], 
                                      graph_name, 
                                      input=None,
                                      config=config,
                                      checkpoint_id=forked_config['checkpoint_id'],
                                      stream_mode="messages-tuple"):

    if chunk.event == "messages":
        print("".join(data_item['content'] for data_item in chunk.data if 'content' in data_item), end="", flush=True)
```

```
# 출력

It looks like there are no tasks in your ToDo list with a deadline within the next week. If you have any tasks that need to be added or updated with specific deadlines, feel free to let me know!
```

### 2.  8.  스레드 간 메모리 (Across-thread memory)

이전 포스팅에서 [LangGraph memory store](https://langchain-ai.github.io/langgraph/concepts/persistence/#memory-store)를 사용해 여러 스레드에 걸쳐 정보를 저장하는 방법을 다루었습니다.

배포된 그래프인 `task_maistro`는 `store`를 활용해 ToDo와 같은 정보를 `user_id` 네임스페이스로 저장합니다.

배포 환경에는 `Postgres` 데이터베이스가 포함되어 있어, 이러한 장기(스레드 간) 메모리를 저장합니다.

LangGraph SDK를 활용해 [스토어와 상호작용할 수 있는 다양한 방법](https://langchain-ai.github.io/langgraph/cloud/reference/sdk/python_sdk_ref/#langgraph_sdk.client.StoreClient)이 제공됩니다.

#### 2. 8.  1.  항목 검색 (Search items)

`task_maistro` 그래프는 기본적으로 (`todo`, `todo_category`, `user_id`)로 네임스페이스가 지정된 ToDo를 `store`에 저장합니다.

`todo_category`는 기본적으로 `general`로 설정되어 있습니다(이는 `deployment/configuration.py`에서 확인할 수 있습니다).

이 튜플을 제공하기만 하면 모든 ToDo를 쉽게 검색할 수 있습니다.

```python
items = await client.store.search_items(
    ("todo", "general", "Test"),
    limit=5,
    offset=0
)
items['items']
```

```
# 출력

[{'namespace': ['todo', 'general', 'Test'],
  'key': '21847225-8c05-432e-8e41-0d7fca53ef1c',
  'value': {'task': 'Call parents back about Thanksgiving plans',
   'status': 'not started',
   'deadline': None,
   'solutions': [],
   'time_to_complete': 15},
  'created_at': '2025-04-24T07:21:21.231052+00:00',
  'updated_at': '2025-04-24T07:21:21.231052+00:00',
  'score': None},
 {'namespace': ['todo', 'general', 'Test'],
  'key': '1d2efb3c-675b-4a78-b00d-7dbd913d1d6f',
  'value': {'task': 'Finish booking travel to Hong Kong',
   'status': 'not started',
   'deadline': '2025-05-03T23:59:59',
   'solutions': ['Check flight prices on Skyscanner',
    'Book hotel through Booking.com',
    'Arrange airport transfer'],
   'time_to_complete': 120},
  'created_at': '2025-04-24T07:21:21.229786+00:00',
  'updated_at': '2025-04-24T07:21:21.229786+00:00',
  'score': None},
 {'namespace': ['todo', 'general', 'Test'],
  'key': '7ad4dedd-6d41-44a8-b8f7-f7ed37349131',
  'value': {'task': 'Finish booking travel to Hong Kong',
   'status': 'not started',
   'deadline': '2025-05-03T23:59:59',
   'solutions': ['Check flight prices on Skyscanner',
    'Book hotel through Booking.com',
    'Arrange airport transfer'],
   'time_to_complete': 120},
  'created_at': '2025-04-24T07:21:18.502633+00:00',
  'updated_at': '2025-04-24T07:21:18.502633+00:00',
  'score': None}]
```

#### 2. 8.  2.  항목 추가 (Add items)

그래프에서는 `put`을 호출하여 항목을 store에 추가합니다.

그래프 외부에서 직접 store에 항목을 추가하고 싶다면, SDK의 [put](https://langchain-ai.github.io/langgraph/cloud/reference/sdk/python_sdk_ref/#langgraph_sdk.client.StoreClient.put_item) 메서드를 사용할 수 있습니다

```python
from uuid import uuid4
await client.store.put_item(
    ("testing", "Test"),
    key=str(uuid4()),
    value={"todo": "Test SDK put_item"},
)
```

```python
items = await client.store.search_items(
    ("testing", "Test"),
    limit=5,
    offset=0
)
items['items']
```

```
# 출력

[{'namespace': ['testing', 'Test'],
  'key': 'ba1fba1f-b10b-48f5-b794-2c7202df6a48',
  'value': {'todo': 'Test SDK put_item'},
  'created_at': '2025-04-24T07:28:26.354556+00:00',
  'updated_at': '2025-04-24T07:28:26.354556+00:00',
  'score': None}]
```

#### 2. 8.  3.  항목 삭제 (Delete items)

SDK를 사용하여 키(key)로 store에서 [항목을 삭제](https://langchain-ai.github.io/langgraph/cloud/reference/sdk/python_sdk_ref/#langgraph_sdk.client.StoreClient.delete_item)할 수 있습니다.

```python
[item['key'] for item in items['items']]
```

```
# 출력

['ba1fba1f-b10b-48f5-b794-2c7202df6a48']
```

```python
await client.store.delete_item(
       ("testing", "Test"),
        key='3de441ba-8c79-4beb-8f52-00e4dcba68d4',
    )
```

```python
items = await client.store.search_items(
    ("testing", "Test"),
    limit=5,
    offset=0
)
items['items']
```

```
# 출력

[{'namespace': ['testing', 'Test'],
  'key': 'ba1fba1f-b10b-48f5-b794-2c7202df6a48',
  'value': {'todo': 'Test SDK put_item'},
  'created_at': '2025-04-24T07:28:26.354556+00:00',
  'updated_at': '2025-04-24T07:28:26.354556+00:00',
  'score': None}]
```

ㅇㅇ

## 정리

ㅇㅇ

## References

* [LangGraph 공식 문서](https://langchain-ai.github.io/langgraph){: target="_blank"}
* [LangChain Academy](https://github.com/langchain-ai/langchain-academy){: target="_blank"}
