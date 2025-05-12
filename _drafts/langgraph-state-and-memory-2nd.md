---
title: LangGraph 상태와 메모리 (2)
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
series_order: 4
---
> 학습할 리소스는 [LangChain Academy Github](https://github.com/langchain-ai/langchain-academy){: target="_blank"}를 사용합니다.
> {: .prompt-info }

## 1. 다중 스키마 (Multiple Schemas)

일반적으로 모든 그래프 노드는 단일 스키마(Single Schema)로 통신합니다.

단일 스키마는 그래프의 입력 및 출력 키/채널을 포함합니다.

노드끼리 필수가 아닌 정보를 전달하고 싶지 않거나 서로 다른 입/출력 스키마를 사용하고 싶을 때 `다중 스키마`를 사용할 수 있습니다.

```python
# 환경 구성
%%capture --no-stderr
%pip install --quiet -U langgraph
```

### 비공개 상태 (Private State)

먼저 노드 간에 [비공개 상태](https://langchain-ai.github.io/langgraph/how-tos/pass_private_state/)를 전달하는 경우를 살펴보겠습니다.

이는 그래프의 중간 작업 로직의 일부로 필요한 모든 경우에 유용하지만, 전체 그래프 입력이나 출력과는 관련이 없습니다.

`OverallState`와 `PrivateState`를 정의하겠습니다.

`node_2`는 `PrivateState`를 입력으로 사용하지만, `OverallState`에 데이터를 씁니다.

`baz`는 `PrivateState`에만 포함됩니다.

`node_2`는 `PrivateState`를 입력으로 사용하지만 `OverallState`에 출력합니다.

따라서 `baz`는 `OverallState`에 없으므로 그래프 출력에서 제외됨을 알 수 있습니다.

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
