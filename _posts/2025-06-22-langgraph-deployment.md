---
title: LangGraph 배포 및 운영
date: 2025-06-22 10:56:43
last_modified_at: 2025-06-22 10:56:43
description : LangGraph 애플리케이션 배포 및 운영에 도움이 되는 이중 입력과 어시스턴트에 대해서 살펴보겠습니다
categories: [AI, LangGraph]
tags: [langgraph, langchain, langsmith, python, llm, generative-ai]
math: true
toc: true
pin: false
image:
    path: assets/posts/2025-05-06-langgraph-introduce/langgraph_logo.png
    alt:
sitemap:
  changefreq: weekly
  priority: 0.5
is_series: true
series_title: "LangGraph"
series_order: 10
---
이번 포스팅에서는 LangGraph 애플리케이션 `배포`와 실제 운영에 도움이 되는 `이중 입력`과 `어시스턴트`에 대해서 살펴보겠습니다.

> 학습할 리소스는 [LangChain Academy Github](https://github.com/langchain-ai/langchain-academy){: target="_blank"}를 사용합니다.
{: .prompt-info }

## 1.   배포 생성하기

이전 포스트에서 만든 `task_maistro` 애플리케이션을 배포해 보겠습니다.

애플리케이션 샘플 코드는 [`module 6`](https://github.com/langchain-ai/langchain-academy/tree/main/module-6/deployment){: target="_blank"} 디렉터리에 있습니다.

### 1.  1.  코드 구조

LangGraph 플랫폼 배포 생성하기 위해서는 [다음 항목을 제공해야 합니다](https://langchain-ai.github.io/langgraph/concepts/application_structure/){: target="_blank"}:

* LangGraph API 구성 파일 (예: `langgraph.json`)
* 애플리케이션 로직을 구현한 그래프 파일 (예: `task_maistro.py`)
* 애플리케이션 실행에 필요한 의존성을 나열한 파일 (예: `requirements.txt`)
* 애플리케이션 실행에 필요한 환경 변수를 지정하는 파일 (예: `.env` 또는 `docker-compose.yml`)

해당 파일은 [`module-6/deployment`](https://github.com/langchain-ai/langchain-academy/tree/main/module-6/deployment){: target="_blank"} 디렉터리에 준비되어 있습니다.

### 1.  2.  CLI

[LangGraph CLI](https://langchain-ai.github.io/langgraph/concepts/langgraph_cli/){: target="_blank"}는 LangGraph 플랫폼 배포를 생성하기 위한 명령줄 인터페이스로 아래와 같이 실행합니다.

```python
%%capture --no-stderr
%pip install -U langgraph-cli
```

[자체 호스팅 배포](https://langchain-ai.github.io/langgraph/how-tos/deploy-self-hosted/#how-to-do-a-self-hosted-deployment-of-langgraph){: target="_blank"}를 만들기 위해, 다음 단계를 진행합니다.

#### 1. 2.  1.  LangGraph 서버용 Docker 이미지 빌드하기

먼저 LangGraph CLI를 사용하여 [LangGraph 서버](https://docs.google.com/presentation/d/18MwIaNR2m4Oba6roK_2VQcBE_8Jq_SI7VHTXJdl7raU/edit#slide=id.g313fb160676_0_32){: target="_blank"}용 Docker 이미지를 생성합니다.

이 명령은 그래프와 의존성을 Docker 이미지 하나로 패키징합니다.

Docker 이미지는 애플리케이션 실행에 필요한 코드와 의존성을 포함하는 컨테이너 템플릿입니다.

[Docker](https://docs.docker.com/engine/install/){: target="_blank"}가 설치되어 있는지 확인한 후, 다음 명령으로 `my-image`라는 이름의 Docker 이미지를 생성합니다:

```bash
$ cd module-6/deployment
$ langgraph build -t my-image
```

#### 1. 2.  2.  Redis 및 PostgreSQL 설정하기

이미 `Redis`와 `PostgreSQL`이 로컬이나 다른 서버에서 실행 중이면, `Redis`와 `PostgreSQL`의 `URI`를 지정하여 LangGraph 서버 컨테이너만 [단독으로 실행](https://langchain-ai.github.io/langgraph/how-tos/deploy-self-hosted/#running-the-application-locally){: target="_blank"}할 수 있습니다:

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

`docker-compose-example.yml`을 복사한 뒤, 다음 환경 변수를 추가하여 `task_maistro` 애플리케이션을 실행합니다:

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

그런 다음, [배포를 시작](https://langchain-ai.github.io/langgraph/how-tos/deploy-self-hosted/#using-docker-compose){: target="_blank"}합니다:

```bash
$ cd module-6/deployment
$ docker compose up
```

## 2.   LangGraph 플랫폼 배포에 연결

### 2.  1.  배포 생성 확인

`task_maistro` 애플리케이션 배포를 생성했습니다.

그리고 LangGraph 서버와 `task_maistro` 그래프를 포함한 Docker 이미지를 빌드하기 위해 `LangGraph CLI`를 사용했습니다.

제공된 `docker-compose.yml` 파일을 사용하여 정의된 서비스에 따라` langgraph-redis`, `langgraph-postgres`, `langgraph-api` 세 개의 개별 컨테이너를 생성하고 배포를 했습니다.

배포 실행이 완료되면 다음 경로를 통해 배포된 서비스를 이용할 수 있습니다.

* `API` : [http://localhost:8123](http://localhost:8123){: target="_blank"}
* `Docs` : [http://localhost:8123/docs](http://localhost:8123/docs){: target="_blank"}
* `LangGraph Studio` : [https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:8123](https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:8123){: target="_blank"}

### 2.  2.  API 사용하기

LangGraph 서버는 배포된 에이전트와 상호작용을 위한 [다양한 API 엔드포인트](https://github.com/langchain-ai/agent-protocol){: target="_blank"}를 제공합니다.

이 엔드포인트들은 [공통적인 에이전트 요구 사항을 기반으로 API를 그룹화](https://github.com/langchain-ai/agent-protocol){: target="_blank"}할 수 있습니다:

* `Runs` : 원자적(단일) 에이전트 실행
* `Threads` : 멀티-턴 상호작용 또는 휴먼-인-더-루프
* `Store` : 장기 메모리

그리고 [API 문서 페이지](http://localhost:8123/docs#tag/thread-runs){: target="_blank"} 직접 요청을 통해 테스트할 수 있습니다.

### 2.  3.  SDK

[LangGraph SDK](https://langchain-ai.github.io/langgraph/concepts/sdk/){: target="_blank"} (Python 및 JS)는 위에서 설명한 LangGraph Server API와 상호작용할 수 있도록 개발자 친화적인 인터페이스를 제공합니다.

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

LangGraph 라이브러리에서 작업하는 경우, [Remote Graph](https://langchain-ai.github.io/langgraph/how-tos/use-remote-graph/){: target="_blank"}를 사용하면 그래프에 직접 연결할 수 있습니다.

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

`run`은 그래프의 [단일 실행](https://github.com/langchain-ai/agent-protocol?tab=readme-ov-file#runs-atomic-agent-executions){: target="_blank"}을 나타냅니다. 클라이언트가 요청을 보낼 때마다 다음 과정이 발생합니다:

1. `HTTP 워커`가 고유한 `run ID`를 생성합니다.
2. 해당 `run`과 그 결과는 `PostgreSQL`에 저장됩니다.
3. 이러한 `run`을 조회하여 다음과 같은 작업을 할 수 있습니다:
   * 상태 확인
   * 결과 조회
   * 실행 이력 추적

여러 종류의 `run`에 대해 자세히 다루는 [How To 가이드](https://langchain-ai.github.io/langgraph/how-tos/#runs){: target="_blank"}를 확인할 수 있습니다.

이제 `run`을 통해 할 수 있는 몇 가지 흥미로운 [작업](https://langchain-ai.github.io/langgraph/cloud/how-tos/background_run/#check-runs-on-thread){: target="_blank"}을 살펴보겠습니다.

> `Run`은 `Agent` 또는 `Graph`가 실제로 동작하는 하나의 실행 인스턴스입니다.
{: .prompt-info }

#### 2. 5.  1.  백그라운드 실행 (Background Runs)

LangGraph 서버는 두 가지 타입의 `run`을 지원합니다:

* `Fire and forget` – 백그라운드에서 `run`을 실행하고, 완료 여부를 기다리지 않음
* `Waiting on a reply (blocking or polling)` – `run`을 실행하고, 그 출력이 나올 때까지 기다리거나 스트리밍함

두 가지 타입의 `Background run`과 `polling`은 장시간 실행되는 에이전트 작업에서 특히 유용합니다.

어떻게 동작하는지 자세한 내용은 [링크](https://langchain-ai.github.io/langgraph/cloud/how-tos/background_run/#check-runs-on-thread){: target="_blank"}에서 확인할 수 있습니다.

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

{'run_id': '1efa2c00-63e4-6f4a-9c5b-ca3f5f9bff07', 'thread_id': '641c195a-9e31-4250-a729-6b742c089df8', 'assistant_id': 'ea4ebafa-a81d-5063-a5fa-67c755d98a21', 'created_at': '2024-11-14T19:38:29.394777+00:00', 'updated_at': '2024-11-14T19:38:29.394777+00:00', 'metadata': {}, 'status': 'pending', 'kwargs': {'input': {'messages': [{'id': None, 'name': None, 'type': 'human', 'content': 'Give me a summary of all ToDos.', 'example': False, 'additional_kwargs': {}, 'response_metadata': {}}]}, 'config': {'metadata': {'created_by': 'system'}, 'configurable': {'run_id': '1efa2c00-63e4-6f4a-9c5b-ca3f5f9bff07', 'user_id': 'Test', 'graph_id': 'task_maistro', 'thread_id': '641c195a-9e31-4250-a729-6b742c089df8', 'assistant_id': 'ea4ebafa-a81d-5063-a5fa-67c755d98a21'}}, 'webhook': None, 'subgraphs': False, 'temporary': False, 'stream_mode': ['values'], 'feedback_keys': None, 'interrupt_after': None, 'interrupt_before': None}, 'multitask_strategy': 'reject'}
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

[스트리밍](https://python.langchain.com/docs/concepts/streaming/){: target="_blank"}에 대해서는 이전 포스팅에서 살펴보았습니다, 여기서는 그중 하나인 `토큰 스트리밍` 방식을 살펴보겠습니다.

`토큰`을 클라이언트로 실시간 전송하는 것은, 실행에 시간이 오래 걸릴 수 있는 프로덕션 에이전트와 작업할 때 특히 유용합니다.

`stream_mode="messages-tuple"` 옵션을 사용하여 [토큰을 스트리밍](https://langchain-ai.github.io/langgraph/cloud/how-tos/stream_messages/#setup){: target="_blank"}합니다.

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

`run`이 그래프의 `단일 실행`만을 의미하는 반면, `스레드`는 `멀티-턴(multi-turn)`의 상호작용을 지원합니다.

클라이언트가 `thread_id`와 함께 그래프 실행을 요청하면, 서버는 해당 `run`의 모든 [체크포인트](https://langchain-ai.github.io/langgraph/concepts/persistence/#checkpoints){: target="_blank"} (단계)를 `Postgres` 데이터베이스의 해당 `스레드`에 저장합니다.

서버는 생성된 스레드의 상태를 확인할 수 있게 해줍니다.

#### 2. 6.  1.  스레드 상태 확인

또한, 특정 `thread`에 저장된 상태 [체크포인트](https://langchain-ai.github.io/langgraph/concepts/persistence/#checkpoints){: target="_blank"}에 쉽게 접근할 수 있습니다.

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

기존의 `스레드`를 복사(fork)할 수도 있습니다.

이렇게 하면 기존 `스레드`의 히스토리는 그대로 유지되지만, 원래 `스레드`에 영향을 주지 않는 독립적인 `run`을 새로 만들 수 있습니다.

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

### 2.  7.  휴먼-인-더-루프 (Human in the loop)

이전 포스팅에서 [휴먼-인-더-루프](https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/){: target="_blank"}에 대해 다뤘습니다. 다시 살펴보면 다음 기능을 지원합니다..

* 체크 포인트 저장
* 과거 상태 검색
* 상태나 결과 편집
* 분기 실행
* 검토 승인 또는 거절
* 유효성 검사 및 재입력 요청

위에서 설명한 기능처럼, [이전 체크포인트에서 그래프 실행을 검색, 편집 또는 계속 진행](https://langchain-ai.github.io/langgraph/concepts/persistence/#capabilities){: target="_blank"}할 수 있습니다.

상태관련 작업을 하겠습니다.

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

이전 포스팅에서 [LangGraph memory store](https://langchain-ai.github.io/langgraph/concepts/persistence/#memory-store){: target="_blank"}를 사용해 여러 스레드에 걸쳐 정보를 저장하는 방법을 다루었습니다.

배포된 그래프인 `task_maistro`는 `store`를 활용해서 ToDo와 같은 정보를 `user_id` 네임스페이스로 저장했습니다.

배포 환경은 `Postgres` 데이터베이스가 있었고, 장기(스레드 간) 메모리를 저장했습니다.

여기서는 `LangGraph SDK`를 활용해 [스토어와 상호작용할 수 있는 다양한 방법](https://langchain-ai.github.io/langgraph/cloud/reference/sdk/python_sdk_ref/#langgraph_sdk.client.StoreClient){: target="_blank"}을 확인할 수 있습니다.

#### 2. 8.  1.  항목 검색 (Search items)

[`module-6/deployment`](https://github.com/langchain-ai/langchain-academy/tree/main/module-6/deployment){: target="_blank"}의 `task_maistro` 그래프는 기본적으로 `todo`, `todo_category`, `user_id`를 사용해서 네임스페이스로 지정된 ToDo를 `store`에 저장합니다.
이전 포스팅에서는 `task_maistro` 그래프는 `todo_category`는 없었지만, 업무용, 개인용, 일반적인 용도로 구분하는 것을 추가했습니다.

`todo_category`는 기본적으로 `general`로 설정되어 있습니다. 이는 `deployment/configuration.py`에서 확인할 수 있습니다.

해당 소스는 모든 ToDo를 쉽게 검색할 수 있게 아래처럼 3개의 값을 묶어 튜플을 제공합니다.

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

그래프에서는 `put`을 호출하여 항목을 `store`에 추가합니다.

그래프 외부에서 직접 `store`에 항목을 추가하고 싶다면, SDK의 [put](https://langchain-ai.github.io/langgraph/cloud/reference/sdk/python_sdk_ref/#langgraph_sdk.client.StoreClient.put_item){: target="_blank"} 메서드를 사용할 수 있습니다.

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

SDK를 사용하여 키(key)로 `store`에서 [항목을 삭제](https://langchain-ai.github.io/langgraph/cloud/reference/sdk/python_sdk_ref/#langgraph_sdk.client.StoreClient.delete_item){: target="_blank"}할 수 있습니다.

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

## 3.   Double Texting (이중 입력)

프로덕션 환경에서 챗봇 애플리케이션 사용 시 [Double Texting (이중 입력)](https://langchain-ai.github.io/langgraph/concepts/double_texting/){: target="_blank"}을 원활하게 처리하는 것은 중요합니다.

사용자는 이전 `run`이 완료되기도 전에, 메시지를 연달아 보낼 수 있으며, 이를 원활하게 처리해야 합니다.

> 사용자가 첫 번째 실행이 완료되기 전에 그래프를 두 번째로 호출할 수 있습니다. 이를 `Double Texting (이중 입력)`이라고 합니다.
{: .prompt-info }

### 3.  1.  Reject

가장 쉬운 방법은, [Reject](https://langchain-ai.github.io/langgraph/cloud/how-tos/reject_concurrent/){: target="_blank"} 전략을 통해 현재 `run`이 완료될 때까지, 새로운 `run`을 `모두 거부`하는 것입니다.

```python
# 환경 구성
%%capture --no-stderr
%pip install -U langgraph_sdk
```

```python
from langgraph_sdk import get_client
url_for_cli_deployment = "http://localhost:8123"
client = get_client(url=url_for_cli_deployment)
```

```python
import httpx
from langchain_core.messages import HumanMessage

# 스레드 생성
thread = await client.threads.create()

# ToDo 생성
user_input_1 = "Add a ToDo to follow-up with DI Repairs."
user_input_2 = "Add a ToDo to mount dresser to the wall."
config = {"configurable": {"user_id": "Test-Double-Texting"}}
graph_name = "task_maistro" 

run = await client.runs.create(
    thread["thread_id"],
    graph_name,
    input={"messages": [HumanMessage(content=user_input_1)]}, 
    config=config,
)
try:
    await client.runs.create(
        thread["thread_id"],
        graph_name,
        input={"messages": [HumanMessage(content=user_input_2)]}, 
        config=config,
        multitask_strategy="reject", # Double texting 처리를 위한 reject 전략
    )
except httpx.HTTPStatusError as e:
    print("Failed to start concurrent run", e)
```

```
# 출력

Failed to start concurrent run Client error '409 Conflict' for url 'http://localhost:8123/threads/7fd9711b-30eb-4fe7-a2df-a3e663e27108/runs'
For more information check: https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/409

```

```python
from langchain_core.messages import convert_to_messages

# 원래 run이 완료될 때까지 대기
await client.runs.join(thread["thread_id"], run["run_id"])

# 스레드의 상태 가져오기
state = await client.threads.get_state(thread["thread_id"])
for m in convert_to_messages(state["values"]["messages"]):
    m.pretty_print()
```

```
# 출력 - reject으로 첫 번째 run만 존재

================================ Human Message =================================

Add a ToDo to follow-up with DI Repairs.
================================== Ai Message ==================================

It looks like the task "Follow-up with DI Repairs" is already on your ToDo list. Is there anything else you'd like to add or modify?
```

### 3.  2.  Enqueue

[enqueue](https://langchain-ai.github.io/langgraph/cloud/how-tos/enqueue_concurrent/){: target="_blank"}를 사용하면, 현재 `run`이 끝날 때까지 새로운 실행을 `대기열(queue)`에 추가할 수 있습니다.

```python
# 새 스레드 생성
thread = await client.threads.create()

# 새로운 ToDo 생성
user_input_1 = "Send Erik his t-shirt gift this weekend."
user_input_2 = "Get cash and pay nanny for 2 weeks. Do this by Friday."
config = {"configurable": {"user_id": "Test-Double-Texting"}}
graph_name = "task_maistro" 

first_run = await client.runs.create(
    thread["thread_id"],
    graph_name,
    input={"messages": [HumanMessage(content=user_input_1)]}, 
    config=config,
)

second_run = await client.runs.create(
    thread["thread_id"],
    graph_name,
    input={"messages": [HumanMessage(content=user_input_2)]}, 
    config=config,
    multitask_strategy="enqueue", # Double texting 처리를 위한 enqueue 전략
)

# 두 번째 run이 완료될 때까지 대기
await client.runs.join(thread["thread_id"], second_run["run_id"])

# 스레드의 상태 가져오기
state = await client.threads.get_state(thread["thread_id"])
for m in convert_to_messages(state["values"]["messages"]):
    m.pretty_print()
```

```
# 출력 - queue 순서로 첫 번째 run 종료 후 두 번째 run이 완료됨

================================ Human Message =================================

Send Erik his t-shirt gift this weekend.
================================== Ai Message ==================================
Tool Calls:
  UpdateMemory (call_yG7CzJQnEgnY292XSGRzIAyd)
 Call ID: call_yG7CzJQnEgnY292XSGRzIAyd
  Args:
    update_type: todo
================================= Tool Message =================================

New ToDo created:
Content: {'task': 'Send Erik his t-shirt gift', 'time_to_complete': 30, 'deadline': '2025-04-27T23:59:59', 'solutions': ['Package the t-shirt', "Get Erik's address", 'Visit the post office', 'Send via courier service'], 'status': 'not started'}

Document ac2569eb-8840-4ee4-818b-5a33cdb69c42 updated:
Plan: Update the status of the task 'Follow-up with DI Repairs' to 'in progress'.
Added content: in progress
================================== Ai Message ==================================

I've updated your ToDo list to send Erik his t-shirt gift this weekend. If there's anything else you need, feel free to let me know!
================================ Human Message =================================

Get cash and pay nanny for 2 weeks. Do this by Friday.
================================== Ai Message ==================================
Tool Calls:
  UpdateMemory (call_kQn0A4sSBBujMpjHv8dFnNF7)
 Call ID: call_kQn0A4sSBBujMpjHv8dFnNF7
  Args:
    update_type: todo
================================= Tool Message =================================

New ToDo created:
Content: {'task': 'Get cash and pay nanny for 2 weeks', 'time_to_complete': 20, 'deadline': '2025-04-25T23:59:59', 'solutions': ['Withdraw cash from ATM', 'Calculate 2 weeks of pay', 'Hand cash to nanny'], 'status': 'not started'}

Document ac2569eb-8840-4ee4-818b-5a33cdb69c42 updated:
Plan: Update the status of the task 'Follow-up with DI Repairs' to 'in progress'.
Added content: in progress
================================== Ai Message ==================================

I've updated your ToDo list to ensure you get cash and pay the nanny for 2 weeks by Friday. Let me know if there's anything else you need help with!
```

### 3.  3.  Interrupt

[interrupt](https://langchain-ai.github.io/langgraph/cloud/how-tos/interrupt_concurrent/){: target="_blank"}를 사용하면, 현재 `run`을 중단하고, 지금까지 진행된 작업 내역은 저장할 수 있습니다.

```python
import asyncio

# 새 스레드 생성
thread = await client.threads.create()

# 새로운 ToDo 생성
user_input_1 = "Give me a summary of my ToDos due tomrrow."
user_input_2 = "Never mind, create a ToDo to Order Ham for Thanksgiving by next Friday."
config = {"configurable": {"user_id": "Test-Double-Texting"}}
graph_name = "task_maistro" 

interrupted_run = await client.runs.create(
    thread["thread_id"],
    graph_name,
    input={"messages": [HumanMessage(content=user_input_1)]}, 
    config=config,
)

# 첫 번째 run이 어느 정도 진행될 때까지 대기하여 스레드에서 볼 수 있도록 함
await asyncio.sleep(1)

second_run = await client.runs.create(
    thread["thread_id"],
    graph_name,
    input={"messages": [HumanMessage(content=user_input_2)]}, 
    config=config,
    multitask_strategy="interrupt", # Double texting 처리를 위한 interrupt 전략
)

# 두 번째 run이 완료될 때까지 대기
await client.runs.join(thread["thread_id"], second_run["run_id"])

# 스레드의 상태 가져오기
state = await client.threads.get_state(thread["thread_id"])
for m in convert_to_messages(state["values"]["messages"]):
    m.pretty_print()
```

```
# 출력

================================ Human Message =================================

Give me a summary of my ToDos due tomrrow.
================================ Human Message =================================

Never mind, create a ToDo to Order Ham for Thanksgiving by next Friday.
================================== Ai Message ==================================
Tool Calls:
  UpdateMemory (call_9Zncl75ucEXLueJofgm3QbQx)
 Call ID: call_9Zncl75ucEXLueJofgm3QbQx
  Args:
    update_type: todo
================================= Tool Message =================================

New ToDo created:
Content: {'task': 'Order Ham for Thanksgiving', 'time_to_complete': 10, 'deadline': '2025-05-02T23:59:59', 'solutions': ['Choose a ham supplier', 'Decide on the type of ham', 'Place the order online or by phone', 'Confirm delivery date']}

Document 6cf82f18-0ac4-44f4-bceb-bafe6dbad89b unchanged:
The task 'Get cash and pay nanny for 2 weeks' is due tomorrow, 2025-04-25. No changes are needed to the existing task details.
================================== Ai Message ==================================

I've updated your ToDo list with the task to "Order Ham for Thanksgiving" by next Friday. If you need anything else, feel free to ask!
```

초기 `run`이 저장되어 있고, 상태는 `interrupted`로 표시된 것을 볼 수 있습니다.

```python
# 첫 번째 실행이 interrupted 되었는지 확인
print((await client.runs.get(thread["thread_id"], interrupted_run["run_id"]))["status"])
```

```
# 출력

interrupted
```

### 3.  4.  Rollback

[rollback](https://langchain-ai.github.io/langgraph/cloud/how-tos/rollback_concurrent/){: target="_blank"}을 사용하면, 그래프의 이전 실행을 중단하고 삭제한 뒤, 새 입력으로 새로운 `run`을 시작합니다.

```python
# 새 스레드 생성
thread = await client.threads.create()

# 새로운 ToDo 생성
user_input_1 = "Add a ToDo to call to make appointment at Yoga."
user_input_2 = "Actually, add a ToDo to drop by Yoga in person on Sunday."
config = {"configurable": {"user_id": "Test-Double-Texting"}}
graph_name = "task_maistro" 

rolled_back_run = await client.runs.create(
    thread["thread_id"],
    graph_name,
    input={"messages": [HumanMessage(content=user_input_1)]}, 
    config=config,
)

second_run = await client.runs.create(
    thread["thread_id"],
    graph_name,
    input={"messages": [HumanMessage(content=user_input_2)]}, 
    config=config,
    multitask_strategy="rollback", # Double texting 처리를 위한 rollback 전략
)

# 두 번째 run이 완료될 때까지 대기
await client.runs.join(thread["thread_id"], second_run["run_id"])

# 스레드의 상태 가져오기
state = await client.threads.get_state(thread["thread_id"])
for m in convert_to_messages(state["values"]["messages"]):
    m.pretty_print()
```

```
# 출력

================================ Human Message =================================

Actually, add a ToDo to drop by Yoga in person on Sunday.
================================== Ai Message ==================================

It looks like you already have a task to "Drop by Yoga in person" on your ToDo list with a deadline of April 27, 2025. Would you like me to update the deadline to the upcoming Sunday, or is there something else you'd like to change about this task?
```

초기 `run`이 삭제되었습니다.

```python
# 원래 run이 삭제되었는지 확인
try:
    await client.runs.get(thread["thread_id"], rolled_back_run["run_id"])
except httpx.HTTPStatusError as _:
    print("Original run was correctly deleted")
```

```
# 출력

Original run was correctly deleted
```

### 3.  5.  Double texting (이중 입력) 전략 요약

[Double texting 전략에 대한 요약](https://langchain-ai.github.io/langgraph/concepts/double_texting/){: target="_blank"}:

![Double texting 전략](assets/posts/2025-06-22-langgraph-deployment/langgraph-deployment_01.png)
_Double texting 전략_

그림에서 특히 Enequeue에 `Run 2`가 `Human message`가 점선인 이유는 `Run 1`의 `AI message`를 대기하다가 완료되면 `Run 2`가 실행됩니다.
그리고 `Interrupt`의 `Run 1`도 `Run 2`가 실행되자마자 중단됩니다.

## 4.   어시스턴트 (Assistants)

[어시스턴트](https://langchain-ai.github.io/langgraph/concepts/assistants/#resources){: target="_blank"}는 에이전트를 빠르게 생성하고 다양한 방식으로 실험하며, 필요에 따라 수정하고 버전 관리를 할 수 있도록 돕는 LangGraph의 추상화 계층입니다.

> `어시스턴트`, `에이전트`, `도구`에 개념을 다시 살펴보면
> `어시스턴트`는 특정 목적을 가진 대화형 에이전트 시스템으로 사용자와 소통하는 챗봇 전체를 의미합니다.
> `에이전트`는 어시스턴트 내에서 상태를 보고 판단하여 행동을 결정합니다.
> `도구`는 함수, 기능 또는 외부 API입니다.
{: .prompt-info }

### 4. 1.   그래프에 설정값 전달하기

[`task_maistro`](https://github.com/langchain-ai/langchain-academy/tree/main/module-6/deployment){: target="_blank"} 그래프는 이미 어시스턴트를 사용할 수 있도록 설정되어 있습니다.

이 그래프에는 `configuration.py` 파일이 정의되어 있고, 그래프 내에서 불러와서 사용합니다.

그래프 노드 내부에서는 설정 가능한 필드(`user_id`, `todo_category`, `task_maistro_role`)에 접근할 수 있습니다.

### 4.  2.  어시스턴트 생성하기

`task_maistro` 애플리케이션에서 어시스턴트 활용은 `개인용` 및 `업무용` 작업을 위한 어시스턴트를 각각 구성하는 것처럼 다양한 카테고리의 작업에 대해 별도의 `ToDo 리스트`를 가질 수 있습니다.

이처럼 서로 다른 어시스턴트는 `todo_category`와 `task_maistro_role`과 같은 설정 가능한 필드를 사용해 손쉽게 만들 수 있습니다.

![개인용 및 업무용 작업을 위한 어시스턴트를 각각 구성](assets/posts/2025-06-22-langgraph-deployment/langgraph-deployment_02.png)
_개인용 및 업무용 작업을 위한 어시스턴트를 각각 구성_

```python
%%capture --no-stderr
%pip install -U langgraph_sdk
```

그래프를 배포할 때 생성한 기본 어시스턴트입니다.

```python
from langgraph_sdk import get_client
url_for_cli_deployment = "http://localhost:8123"
client = get_client(url=url_for_cli_deployment)
```

### 4.  3.  개인 어시스턴트 (Personal assistant)

이것은 개인적인 작업을 관리하기 위해 사용할 `개인 어시스턴트`입니다.

```python
personal_assistant = await client.assistants.create(
    # task_maistro는 배포한 그래프의 이름입니다
    "task_maistro", 
    config={"configurable": {"todo_category": "personal"}}
)
print(personal_assistant)
```

```
# 출력

{'assistant_id': 'ccaa907b-1faf-4873-8aa3-752412505884', 'graph_id': 'task_maistro', 'created_at': '2025-04-24T08:07:31.751394+00:00', 'updated_at': '2025-04-24T08:07:31.751394+00:00', 'config': {'configurable': {'todo_category': 'personal'}}, 'metadata': {}, 'version': 1, 'name': 'Untitled', 'description': None}

```

이제, 이 어시스턴트에 내 `user_id`를 추가해서 편리하게 사용할 수 있도록 `update`를 사용하여 [새 버전을 만듭니다](https://langchain-ai.github.io/langgraph/cloud/how-tos/assistant_versioning/#create-a-new-version-for-your-assistant){: target="_blank"}.

```python
task_maistro_role = """You are a friendly and organized personal task assistant. Your main focus is helping users stay on top of their personal tasks and commitments. Specifically:

- Help track and organize personal tasks
- When providing a 'todo summary':
  1. List all current tasks grouped by deadline (overdue, today, this week, future)
  2. Highlight any tasks missing deadlines and gently encourage adding them
  3. Note any tasks that seem important but lack time estimates
- Proactively ask for deadlines when new tasks are added without them
- Maintain a supportive tone while helping the user stay accountable
- Help prioritize tasks based on deadlines and importance

Your communication style should be encouraging and helpful, never judgmental. 

When tasks are missing deadlines, respond with something like "I notice [task] doesn't have a deadline yet. Would you like to add one to help us track it better?"""

configurations = {"todo_category": "personal", 
                  "user_id": "lance",
                  "task_maistro_role": task_maistro_role}

personal_assistant = await client.assistants.update(
    personal_assistant["assistant_id"],
    config={"configurable": configurations}
)
print(personal_assistant)
```

```
# 출력

{'assistant_id': 'ccaa907b-1faf-4873-8aa3-752412505884', 'graph_id': 'task_maistro', 'created_at': '2025-04-24T08:07:49.565100+00:00', 'updated_at': '2025-04-24T08:07:49.565100+00:00', 'config': {'configurable': {'user_id': 'lance', 'todo_category': 'personal', 'task_maistro_role': 'You are a friendly and organized personal task assistant. Your main focus is helping users stay on top of their personal tasks and commitments. Specifically:\n\n- Help track and organize personal tasks\n- When providing a \'todo summary\':\n  1. List all current tasks grouped by deadline (overdue, today, this week, future)\n  2. Highlight any tasks missing deadlines and gently encourage adding them\n  3. Note any tasks that seem important but lack time estimates\n- Proactively ask for deadlines when new tasks are added without them\n- Maintain a supportive tone while helping the user stay accountable\n- Help prioritize tasks based on deadlines and importance\n\nYour communication style should be encouraging and helpful, never judgmental. \n\nWhen tasks are missing deadlines, respond with something like "I notice [task] doesn\'t have a deadline yet. Would you like to add one to help us track it better?'}}, 'metadata': {}, 'version': 2, 'name': 'Untitled', 'description': None}
```

### 4.  4.  업무용 어시스턴트 (Work assistant)

이제 `업무용 어시스턴트`를 만들어 보겠습니다. 이 어시스턴트는 업무 관련 작업을 관리할 때 사용합니다.

```python
task_maistro_role = """You are a focused and efficient work task assistant. 

Your main focus is helping users manage their work commitments with realistic timeframes. 

Specifically:

- Help track and organize work tasks
- When providing a 'todo summary':
  1. List all current tasks grouped by deadline (overdue, today, this week, future)
  2. Highlight any tasks missing deadlines and gently encourage adding them
  3. Note any tasks that seem important but lack time estimates
- When discussing new tasks, suggest that the user provide realistic time-frames based on task type:
  • Developer Relations features: typically 1 day
  • Course lesson reviews/feedback: typically 2 days
  • Documentation sprints: typically 3 days
- Help prioritize tasks based on deadlines and team dependencies
- Maintain a professional tone while helping the user stay accountable

Your communication style should be supportive but practical. 

When tasks are missing deadlines, respond with something like "I notice [task] doesn't have a deadline yet. Based on similar tasks, this might take [suggested timeframe]. Would you like to set a deadline with this in mind?"""

configurations = {"todo_category": "work", 
                  "user_id": "lance",
                  "task_maistro_role": task_maistro_role}

work_assistant = await client.assistants.create(
    # task_maistro는 배포한 그래프의 이름입니다
    "task_maistro", 
    config={"configurable": configurations}
)
print(work_assistant)
```

```
# 출력

{'assistant_id': '9a4a7b55-8246-43f5-81a3-45db1e893847', 'graph_id': 'task_maistro', 'created_at': '2025-04-24T08:08:07.786433+00:00', 'updated_at': '2025-04-24T08:08:07.786433+00:00', 'config': {'configurable': {'user_id': 'lance', 'todo_category': 'work', 'task_maistro_role': 'You are a focused and efficient work task assistant. \n\nYour main focus is helping users manage their work commitments with realistic timeframes. \n\nSpecifically:\n\n- Help track and organize work tasks\n- When providing a \'todo summary\':\n  1. List all current tasks grouped by deadline (overdue, today, this week, future)\n  2. Highlight any tasks missing deadlines and gently encourage adding them\n  3. Note any tasks that seem important but lack time estimates\n- When discussing new tasks, suggest that the user provide realistic time-frames based on task type:\n  • Developer Relations features: typically 1 day\n  • Course lesson reviews/feedback: typically 2 days\n  • Documentation sprints: typically 3 days\n- Help prioritize tasks based on deadlines and team dependencies\n- Maintain a professional tone while helping the user stay accountable\n\nYour communication style should be supportive but practical. \n\nWhen tasks are missing deadlines, respond with something like "I notice [task] doesn\'t have a deadline yet. Based on similar tasks, this might take [suggested timeframe]. Would you like to set a deadline with this in mind?'}}, 'metadata': {}, 'version': 1, 'name': 'Untitled', 'description': None}
```

### 4.  5.  어시스턴트 활용하기

어시스턴트는 배포 환경에서 `Postgres`에 저장되고 `SDK`를 사용하면 [검색](https://langchain-ai.github.io/langgraph/cloud/how-tos/configuration_cloud/){: target="_blank"}을 통해 어시스턴트를 쉽게 찾을 수 있습니다.

```python
assistants = await client.assistants.search()
for assistant in assistants:
    print({
        'assistant_id': assistant['assistant_id'],
        'version': assistant['version'],
        'config': assistant['config']
    })
```

```
# 출력

{'assistant_id': '9a4a7b55-8246-43f5-81a3-45db1e893847', 'version': 1, 'config': {'configurable': {'user_id': 'lance', 'todo_category': 'work', 'task_maistro_role': 'You are a focused and efficient work task assistant. \n\nYour main focus is helping users manage their work commitments with realistic timeframes. \n\nSpecifically:\n\n- Help track and organize work tasks\n- When providing a \'todo summary\':\n  1. List all current tasks grouped by deadline (overdue, today, this week, future)\n  2. Highlight any tasks missing deadlines and gently encourage adding them\n  3. Note any tasks that seem important but lack time estimates\n- When discussing new tasks, suggest that the user provide realistic time-frames based on task type:\n  • Developer Relations features: typically 1 day\n  • Course lesson reviews/feedback: typically 2 days\n  • Documentation sprints: typically 3 days\n- Help prioritize tasks based on deadlines and team dependencies\n- Maintain a professional tone while helping the user stay accountable\n\nYour communication style should be supportive but practical. \n\nWhen tasks are missing deadlines, respond with something like "I notice [task] doesn\'t have a deadline yet. Based on similar tasks, this might take [suggested timeframe]. Would you like to set a deadline with this in mind?'}}}
{'assistant_id': 'ccaa907b-1faf-4873-8aa3-752412505884', 'version': 2, 'config': {'configurable': {'user_id': 'lance', 'todo_category': 'personal', 'task_maistro_role': 'You are a friendly and organized personal task assistant. Your main focus is helping users stay on top of their personal tasks and commitments. Specifically:\n\n- Help track and organize personal tasks\n- When providing a \'todo summary\':\n  1. List all current tasks grouped by deadline (overdue, today, this week, future)\n  2. Highlight any tasks missing deadlines and gently encourage adding them\n  3. Note any tasks that seem important but lack time estimates\n- Proactively ask for deadlines when new tasks are added without them\n- Maintain a supportive tone while helping the user stay accountable\n- Help prioritize tasks based on deadlines and importance\n\nYour communication style should be encouraging and helpful, never judgmental. \n\nWhen tasks are missing deadlines, respond with something like "I notice [task] doesn\'t have a deadline yet. Would you like to add one to help us track it better?'}}}
{'assistant_id': 'ea4ebafa-a81d-5063-a5fa-67c755d98a21', 'version': 1, 'config': {}}
```

이처럼 SDK를 통해 쉽게 관리가 가능합니다. 더 이상 사용하지 않는 어시스턴트도 쉽게 삭제할 수 있습니다.

```python
await client.assistants.delete("ea4ebafa-a81d-5063-a5fa-67c755d98a21") #assistant_id
```

이제 사용할 `개인용(personal)`과 `업무용(work)` 어시스턴트의 `assistant ID`를 설정해 보겠습니다.

```python
work_assistant_id = assistants[0]['assistant_id']
personal_assistant_id = assistants[1]['assistant_id']
```

### 4.  6.  업무용 어시스턴트

업무용 어시스턴트에 할 일(ToDo)을 몇 개 추가해 보겠습니다.

```python
from langchain_core.messages import HumanMessage
from langchain_core.messages import convert_to_messages

user_input = "Create or update few ToDos: 1) Re-film Module 6, lesson 5 by end of day today. 2) Update audioUX by next Monday."
thread = await client.threads.create()
async for chunk in client.runs.stream(thread["thread_id"], 
                                      work_assistant_id,
                                      input={"messages": [HumanMessage(content=user_input)]},
                                      stream_mode="values"):

    if chunk.event == 'values':
        state = chunk.data
        convert_to_messages(state["messages"])[-1].pretty_print()
```

```
# 출력

================================ Human Message =================================

Create or update few ToDos: 1) Re-film Module 6, lesson 5 by end of day today. 2) Update audioUX by next Monday.
================================== Ai Message ==================================
Tool Calls:
  UpdateMemory (call_RyngIB97HjX3QqyRWBcqIu0o)
 Call ID: call_RyngIB97HjX3QqyRWBcqIu0o
  Args:
    update_type: todo
================================= Tool Message =================================

New ToDo created:
Content: {'task': 'Re-film Module 6, lesson 5', 'time_to_complete': 120, 'deadline': '2025-04-24T23:59:00', 'solutions': ['Book a studio', 'Prepare script', 'Check equipment'], 'status': 'not started'}
================================== Ai Message ==================================
Tool Calls:
  UpdateMemory (call_4UFdYqpJTd4UhBzXk6jf5itx)
 Call ID: call_4UFdYqpJTd4UhBzXk6jf5itx
  Args:
    update_type: todo
================================= Tool Message =================================

Document 9b9135ad-3be0-4aee-8da3-c790f89a56f9 unchanged:
The task 'Re-film Module 6, lesson 5' is already present with the correct deadline of '2025-04-24T23:59:00'. No updates are needed for this task.

New ToDo created:
Content: {'task': 'Update audioUX', 'deadline': '2025-04-28T23:59:00', 'status': 'not started'}
================================== Ai Message ==================================

I've updated your ToDo list with the tasks:

1. **Re-film Module 6, lesson 5** - Due by the end of today.
2. **Update audioUX** - Due by next Monday.

If you need any further assistance or adjustments, feel free to let me know!
```

```python
user_input = "Create another ToDo: Finalize set of report generation tutorials."
thread = await client.threads.create()
async for chunk in client.runs.stream(thread["thread_id"], 
                                      work_assistant_id,
                                      input={"messages": [HumanMessage(content=user_input)]},
                                      stream_mode="values"):

    if chunk.event == 'values':
        state = chunk.data
        convert_to_messages(state["messages"])[-1].pretty_print()
```

```
# 출력

================================ Human Message =================================

Create another ToDo: Finalize set of report generation tutorials.
================================== Ai Message ==================================

I notice the task "Finalize set of report generation tutorials" doesn't have a deadline yet. Based on similar tasks, this might take around 2 days. Would you like to set a deadline with this in mind?
```

이 어시스턴트는 해당 `지침(instructions)`을 사용하여, `마감일(deadline)`이 설정되어 있지 않기 때문에 `Ai Message`는 작업 생성 과정에서` 마감일`을 지정해 달라고 요청합니다.

```python
user_input = "OK, for this task let's get it done by next Tuesday."
async for chunk in client.runs.stream(thread["thread_id"], 
                                      work_assistant_id,
                                      input={"messages": [HumanMessage(content=user_input)]},
                                      stream_mode="values"):

    if chunk.event == 'values':
        state = chunk.data
        convert_to_messages(state["messages"])[-1].pretty_print()
```

```
# 출력

================================ Human Message =================================

OK, for this task let's get it done by next Tuesday.
================================== Ai Message ==================================
Tool Calls:
  UpdateMemory (call_aDyLLqLjaYnk366gSSXtur4T)
 Call ID: call_aDyLLqLjaYnk366gSSXtur4T
  Args:
    update_type: todo
================================= Tool Message =================================

New ToDo created:
Content: {'task': 'Finalize set of report generation tutorials', 'time_to_complete': 240, 'deadline': '2025-04-29T23:59:00', 'solutions': ['Outline tutorial content', 'Create video scripts', 'Record tutorial videos', 'Edit and finalize videos'], 'status': 'not started'}
================================== Ai Message ==================================

I've updated the task "Finalize set of report generation tutorials" with a deadline of next Tuesday, April 29, 2025. If there's anything else you'd like to add or adjust, just let me know!
```

### 4.  7.  개인용 어시스턴트

마찬가지로, 개인 어시스턴트에도 할 일(ToDo)을 추가할 수 있습니다.

```python
user_input = "Create ToDos: 1) Check on swim lessons for the baby this weekend. 2) For winter travel, check AmEx points."
thread = await client.threads.create()
async for chunk in client.runs.stream(thread["thread_id"], 
                                      personal_assistant_id,
                                      input={"messages": [HumanMessage(content=user_input)]},
                                      stream_mode="values"):

    if chunk.event == 'values':
        state = chunk.data
        convert_to_messages(state["messages"])[-1].pretty_print()
```

```
# 출력

================================ Human Message =================================

Create ToDos: 1) Check on swim lessons for the baby this weekend. 2) For winter travel, check AmEx points.
================================== Ai Message ==================================
Tool Calls:
  UpdateMemory (call_wNMv6ky7tAGRgp7kiD230yBI)
 Call ID: call_wNMv6ky7tAGRgp7kiD230yBI
  Args:
    update_type: todo
================================= Tool Message =================================

New ToDo created:
Content: {'task': 'Check on swim lessons for the baby this weekend', 'time_to_complete': 30, 'deadline': '2025-04-26T00:00:00', 'solutions': ['Call local swimming pools', 'Check online for baby swim classes', 'Ask friends for recommendations'], 'status': 'not started'}
================================== Ai Message ==================================

I've added the task to check on swim lessons for the baby this weekend to your list. 

For the second task, "For winter travel, check AmEx points," would you like to add a deadline to help us track it better?
```

```python
user_input = "Give me a todo summary."
thread = await client.threads.create()
async for chunk in client.runs.stream(thread["thread_id"], 
                                      personal_assistant_id,
                                      input={"messages": [HumanMessage(content=user_input)]},
                                      stream_mode="values"):

    if chunk.event == 'values':
        state = chunk.data
        convert_to_messages(state["messages"])[-1].pretty_print()
```

```
# 출력

================================ Human Message =================================

Give me a todo summary.
================================== Ai Message ==================================

Here's your current todo summary:

**This Week:**
- **Check on swim lessons for the baby this weekend**
  - Deadline: April 26, 2025
  - Solutions: Call local swimming pools, Check online for baby swim classes, Ask friends for recommendations
  - Estimated time to complete: 30 minutes

I notice all tasks have deadlines and time estimates, which is great for staying organized! If you have any new tasks to add or need help prioritizing, feel free to let me know.
```

## 정리

LangGraph 플랫폼 배포 생성을 위한 항목(`LangGraph API 구성 파일`, `그래프 파일`, `의존성 파일`, `환경 변수 파일`)을 살펴보았습니다. 그리고 `LangGraph CLI`를 사용해서 LangGraph 서버용 Docker 이미지를 생성하고 docker-compose를 구성하여 배포했습니다. 

LangGraph 서버는 배포된 에이전트와 상호작용을 위한 `API`와 `SDK`를 제공합니다.

그래프의 단일 실행인 `run`은 실행 후 완료 여부를 기다리지 않거나 출력이 나올 때까지 기다리는 두 가지 타입을 지원합니다.

또한 `토큰 스트리밍`을 통해 실행이 오래 걸릴 수 있는 프로덕션 에이전트와 작업의 유용성을 보았습니다.

멀티-턴 상호작용을 지원하는 `스레드`에서 상태 확인 및 복사가 가능했고, `휴먼-인-더-루프`를 통해 이전 체크포인트에서 그래프 실행을 검색, 편집 또는 계속 실행이 가능했습니다.

`LangGraph SDK`를 사용하여 `항목 검색`, `추가`, `삭제`와 같이 `장기메모리`와 다양한 상호 작용을 알아보았습니다.

또, `이중 입력` 처리를 위한 `거부`, `대기열 추가`, `중단 후 작업 내역을 저장` 및 `삭제`할 수 있는 전략도 살펴보았습니다.

마지막으로 에이전트를 빠르게 생성하고 다양한 방식의 실험 및 수정과 버전관리를 돕는 `어시스턴트`를 `개인용`과 `업무용` 작업을 위한 `어시스턴트`로 각각 구성하였습니다.

10개의 포스팅에서 우리는 LangGraph의 다양한 기능들을 살펴보았습니다. LangGraph는 추론 흐름을 제어하는 상태 머신 프레임워크로서 우리의 완전 자율 에이전트의 높은 제어 수준에서 신뢰성을 개선할 수 있게 도와줄 것입니다.


## References

* [LangGraph 공식 문서](https://langchain-ai.github.io/langgraph){: target="_blank"}
* [LangChain Academy](https://github.com/langchain-ai/langchain-academy){: target="_blank"}
