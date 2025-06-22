---
title: LangGraph와 MCP
date: 2025-06-18 11:56:43 +/-TTTT
description : LangGraph와 MCP에 대해서 살펴보겠습니다.
tags: [langgraph, langchain, langsmith, python, llm, generative-ai, mcp]
math: true
toc: true
pin: false
# image:
#     path: assets/posts/2025-05-06-langgraph-introduce/langgraph_logo.png
#     alt:
is_series: false
series_title: "LangGraph"
series_order: 10
---
이번 포스팅에서는 LangGraph 애플리케이션 `배포`와 실제 운영에 도움이 되는 `이중 입력`과 `어시스턴트`에 대해서 살펴보겠습니다.

> 학습할 리소스는 [LangChain Academy Github](https://github.com/langchain-ai/langchain-academy){: target="_blank"}를 사용합니다.
> {: .prompt-info }

## MCP (Model Context Protocol) 소개

MCP는 애플리케이션이 LLM에 컨텍스트를 제공하는 방식을 표준화하는 개방형 프로토콜입니다.

기존에 도구 호출(Tool Calling)을 통해 비슷한 기능을 제공했습니다. 그러면 도구 호출과 차이는 무엇일까요?

MCP는 AI 애플리케이션을 위한 USB-C 포트와 같습니다. USB-C가 다양한 주변 기기 및 액세서리에 기기를 연결하는 표준화된 방식을 제공하는 것처럼, MCP는 AI 모델을 다양한 데이터 소스 및 도구에 연결하는 표준화된 방식을 제공합니다.

### 왜 MCP인가?

MCP는 LLM을 기반으로 에이전트와 복잡한 워크플로를 구축하는 데 도움을 줍니다. LLM은 데이터 및 도구와 통합해야 하는 경우가 많으며, MCP는 다음과 같은 기능을 제공합니다.

- LLM이 직접 연결할 수 있는 미리 구축된 통합 목록이 점점 증가
- LLM 공급자와 벤더 간 전환의 유연성
- 인프라 내에서 데이터를 보호하기 위한 모범 사례

### MCP 범용 아키텍처

MCP는 본질적으로 호스트 애플리케이션이 여러 서버에 연결할 수 있는 클라이언트-서버 아키텍처를 따릅니다.

- MCP 호스트 : MCP를 통해 데이터에 액세스하려는 Claude Desktop, IDE 또는 AI 도구와 같은 프로그램
- MCP 클라이언트 : 서버와 1:1 연결을 유지하는 프로토콜 클라이언트
- MCP 서버 : 표준화된 MCP를 통해 특정 기능을 제공하는 경량화 프로그램
- 로컬 데이터 소스 : MCP 서버가 안전하게 액세스할 수 있는 컴퓨터의 파일, 데이터베이스 및 서비스
- 원격 서비스 : MCP 서버가 연결할 수 있는 인터넷(예: API를 통해)을 통해 사용 가능한 외부 시스템

![MCP 범용 아키텍처](assets/drafts/2025-06-17-langgraph-mcp/langgraph-mcp_general_arch.png)
_MCP 범용 아키텍처

## 핵심 아키텍처

### 클라이언트-서버 아키텍처

`MCP`는 `클라이언트-서버` 아키텍처를 구조입니다.

* `호스트` : 연결을 시작하는 Claude Desktop 또는 IDE와 같은 LLM 애플리케이션을 말합니다.
* `클라이언트` : 호스트 애플리케이션 내부에서 서버와 1:1 연결을 유지합니다.
* `서버` : 클라이언트에게 컨텍스트, 도구 및 프롬프트를 제공합니다

### 핵심 구성 요소

- `프로토콜 계층` : MCP 메시지와 명령을 정의합니다.
- `전송 계층` : 클라이언트-서버 간의 데이터 전송 방식으로 로컬 프로세스에 이상적인 `stdio`와 스트리밍을 위한 `streamable HTTP` 사용합니다.
- `메시지 유형` : 클라이언트와 서버가 주고받는 `요청(request)`, `결과(result)`, `오류(error)`, `알림(notification)` 데이터 형식을 정의합니다.

### 연결 수명 주기 (LifeCycle)

* `초기화 (Initialization)` : 클라이언트는 버전과 기능을 요청하면 서버에서 응답 후 클라이언트로 확인 보냄
* `메시지 교환 (Message Exchange)` : `요청과 응답` 및 일방적인 `알림`을 통해 실제로 데이터를 주고받는 단계
* `종료 (Termination)` : 통신이 끝났을 때 연결 종료

[그림]

### 에러 처리

요청, 전송, 프로토콜 수준의 에러를 처리합니다.

그리고 SDK와 애플리케이션을 사용해서 `-32000 이상` `JSON-RPC` 표준으로 `자신만의 에러 코드`를 정의할 수 있습니다.

```typescript
enum ErrorCode {
  // Standard JSON-RPC error codes
  ParseError = -32700,
  InvalidRequest = -32600,
  MethodNotFound = -32601,
  InvalidParams = -32602,
  InternalError = -32603,
}
```

## 주요 개념

MCP에서 다루는 주요 개념인 `리소스`, `프롬프트`, `도구`, `샘플링`, `루트`, `전송`에 대해서 알아보겠습니다.

### 리소스

리소스는 서버의 데이터와 콘텐츠를 LLM이 사용할 수 있고 MCP 서버가 클라이언트에 제공하려는 모든 종류의 데이터를 나타냅니다.

리소스가 제공하는 데이터는 `파일 콘텐츠`, `데이터베이스 레코드`, `API 응답`, `라이브 시스템 데이터`, `스크린샷 및 이미지`, `로그 파일` 등이 있습니다.

각 리소스는 고유한 URI로 식별되며 텍스트 또는 이진 데이터를 포함할 수 있습니다.

```
# 형식
[protocol]://[host]/[path]

# 예
file:///home/user/documents/report.pdf
postgres://database/customers/schema
screen://localhost/display1
```

#### 리소스 유형

리소스는 `텍스트`와 `바이너리` 두 가지 유형의 콘텐츠가 포함됩니다.

`텍스트`는 `UTF-8`로 인코딩 된 텍스트 데이터로 소스코드, 구성 파일, 로그 파일, JSON/XML 데이터, 일반 텍스트 용도로 적합하고

`바이너리`는 `base64`로 인코딩된 원시 바이너리 데이터로 이미지, PDF, 오디오/비디오 파일, 텍스트 형식이 아닌 용도에 적합합니다.

#### 리소스 탐색

리소스 탐색에서 `직접` 또는 `리소스 템플릿`을 통해 사용 가능한 리소스를 찾을 수 있습니다.

#### 리소스 읽기

리소스를 읽기 위해서는 클라이언트가 resources/read 리소스 URI로 요청합니다.

```json
{
  contents: [
    {
      uri: string;        // The URI of the resource
      mimeType?: string;  // Optional MIME type

      // One of:
      text?: string;      // For text resources
      blob?: string;      // For binary resources (base64 encoded)
    }
  ]
}
```

#### 리소스 업데이트

MCP는 목록과 콘텐츠가 업데이트 되면 `nofifications/resources/list_changed` 알림을 통해 클라이언트에게 알릴 수 있습니다.

특정 콘텐츠 업데이트가 되면 다음과 같은 순서로 `구독` 및 `구독 취소`가 가능합니다.

1. 클라이언트가 `resources/subscribe`리소스 URI와 함께 전송
2. `notifications/resources/updated`리소스가 변경되면 서버가 전송
3. `resources/read`로 클라이언트는 최신 콘텐츠를 가져옴
4. `resources/unsubscribe`로 클라이언트는 구독 취소 가능

다음은 리소스 업데이트 예제입니다.

```python
app = Server("example-server")

@app.list_resources()
async def list_resources() -> list[types.Resource]:
    return [
        types.Resource(
            uri="file:///logs/app.log",
            name="Application Logs",
            mimeType="text/plain"
        )
    ]

@app.read_resource()
async def read_resource(uri: AnyUrl) -> str:
    if str(uri) == "file:///logs/app.log":
        log_contents = await read_log_file()
        return log_contents

    raise ValueError("Resource not found")

# Start server
async with stdio_server() as streams:
    await app.run(
        streams[0],
        streams[1],
        app.create_initialization_options()
    )
```

### 프롬프트

재사용 가능한 프롬프트 템플릿 및 워크플로 만들기

프롬프트를 사용하면 서버가 재사용 가능한 프롬프트 템플릿과 워크플로를 정의하여 클라이언트가 사용자와 LLM에 쉽게 제공할 수 있습니다. 또한, 공통 LLM 상호작용을 표준화하고 공유하는 강력한 방법을 제공합니다.

개요

MCP의 프롬프트는 다음을 수행할 수 있는 미리 정의된 템플릿입니다.

* 동적 인수 허용
* 리소스의 컨텍스트 포함
* 여러 상호 작용 연결
* 특정 워크플로 가이드
* UI 요소(슬래시 명령 등)로 표면화

### 도구 (Tools)

ㅇㅇ

### 샘플링 (Sampling)

ㅇㅇ

### 루트 (Roots)

ㅇㅇ

### 전송 (Transport)

ㅇㅇ

ㅇ

## LangGraph와 MCP 통합

LangGraph 에이전트는 langchain-mcp-adapters 라이브러리를 통해 MCP 서버에 정의된 도구를 사용할 수 있습니다.

`langchain-mcp-adapters`LangGraph에서 MCP 도구를 사용하려면 라이브러리를 설치하세요 .

```bash
pip install langchain-mcp-adapters
```

```python
# MCP 서버에 정의된 도구를 사용하는 에이전트
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent

client = MultiServerMCPClient(
    {
        "math": {
            "command": "python",
            # Replace with absolute path to your math_server.py file
            "args": ["/path/to/math_server.py"],
            "transport": "stdio",
        },
        "weather": {
            # Ensure your start your weather server on port 8000
            "url": "http://localhost:8000/mcp",
            "transport": "streamable_http",
        }
    }
)
tools = await client.get_tools()
agent = create_react_agent(
    "anthropic:claude-3-7-sonnet-latest",
    tools
)
math_response = await agent.ainvoke(
    {"messages": [{"role": "user", "content": "what's (3 + 5) x 12?"}]}
)
weather_response = await agent.ainvoke(
    {"messages": [{"role": "user", "content": "what is the weather in nyc?"}]}
)
```

### 커스텀 MCP 서버

자체 MCP 서버를 만들려면 mcp라이브러리를 사용할 수 있습니다. 이 라이브러리는 도구를 정의하고 서버로 실행하는 간단한 방법을 제공합니다.

MCP 라이브러리를 설치하세요:

```bash
pip install mcp
```

```python
# 예제 수학 서버(stdio 전송)
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Math")

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

@mcp.tool()
def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b

if __name__ == "__main__":
    mcp.run(transport="stdio")
```

```python
# 예시 날씨 서버(스트리밍 가능한 HTTP 전송)
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Weather")

@mcp.tool()
async def get_weather(location: str) -> str:
    """Get weather for location."""
    return "It's always sunny in New York"

if __name__ == "__main__":
    mcp.run(transport="streamable-http")
```

dd

## 정리

## References

* [LangGraph 공식 문서](https://langchain-ai.github.io/langgraph){: target="_blank"}
* [MCP](https://langchain-ai.github.io/langgraph/agents/mcp/){: target="_blank"}
* [MCP](https://modelcontextprotocol.io/introduction)
