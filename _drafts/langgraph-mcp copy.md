---
title: MCP (Model Context Protocol) 응용
date: 2025-06-18 11:56:43 +/-TTTT
description : MCP에 대해서 아키텍처 및 기능에 대해서 살펴보겠습니다.
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
