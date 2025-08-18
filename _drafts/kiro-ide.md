---
title: AI 에이전트 개발 환경 Kiro 소개
date: 2025-07-18 11:56:43 +/-TTTT
description : AWS에서 2025년 7월 스펙 기반 개발 환경인 Kiro를 발표했습니다.
tags: [kiro, llm, generative-ai, mcp, ide]
math: true
toc: true
pin: false
# image:
#     path: assets/posts/2025-05-06-langgraph-introduce/langgraph_logo.png
#     alt:
# is_series: false
# series_title: "LangGraph"
# series_order: 10
---
이번 포스팅에서는 LangGraph 애플리케이션 `배포`와 실제 운영에 도움이 되는 `이중 입력`과 `어시스턴트`에 대해서 살펴보겠습니다.

> 학습할 리소스는 [LangChain Academy Github](https://github.com/langchain-ai/langchain-academy){: target="_blank"}를 사용합니다.
> {: .prompt-info }


## 스펙 기반 개발 환경 "Kiro" 

AWS에서 2025년 7월 AI 기반 통합 개발 환경(IDE)인 `Kiro` preview를 발표했습니다.
기존 AI 기반 IDE는 `Cursor AI`, `Windsurf`가 있었습니다.
`Kiro`는 다른 IDE와 차이점이 스펙 기반 개발(spec-driven development) 환경을 지원한다는 것입니다.

## Kiro의 특징

### 스펙 기반 개발 워크플로우
`Kiro`의 가장 독특한 특징은 개발자의 단일 프롬프트를 구조화된 `요구사항`, `기술 설계`, 그리고 `구현 작업`으로 변환하는 능력입니다. 예를 들어 "제품 리뷰 시스템 추가"라는 간단한 요청을 입력하면, `Kiro`는 다음과 같이 작업을 수행합니다:

`요구사항 분석 (Requirements)`: 사용자 스토리와 EARS(Easy Approach to Requirements Syntax) 형식의 승인 기준을 생성

`기술 설계 (Design)`: 데이터 플로우 다이어그램, TypeScript 인터페이스, 데이터베이스 스키마, API 엔드포인트를 포함한 설계 문서 생성

`작업 구현 (Tasks)`: 종속성에 따라 순서가 지정된 작업과 하위 작업을 생성하며, 각 작업에는 단위 테스트, 통합 테스트, 접근성 요구사항 등이 포함

### AI 에이전트 훅(Agent Hooks)
**훅(Hooks)**은 Kiro의 또 다른 핵심 기능으로, 파일 저장, 생성, 삭제 등의 이벤트에 대응하여 자동으로 실행되는 이벤트 기반 자동화 도구입니다. 이를 통해 다음과 같은 작업들을 자동화할 수 있습니다:

- 문서 자동 생성 및 업데이트
- API 동기화
- 보안 검사 실행
- 테스트 파일 업데이트





오히려 스펙 기반으로 선택 시 Plan - Design - Task를 자동으로 구성이 되기 때문에
Cursor AI를 사용할 때 처럼 Taskmanager와 같은 서비스를 연동할 필요가 없어 편했다.

## 아직 개선은 필요
프리뷰가 나오자마자 인스타그램 MVP 기반으로 바이브코딩을 했을 때 가장 많이 나오는 문제가 
코딩하면서 밤을 세우겠다가 아니라 Retry이 하다가 지쳤다.

나중에는 Amazon Bedrock 연동하여 유료로 진행할 예정으로 알고 있다.
AWS 수익을 위해서 개발했겠지만 Cursor AI, Windsurf 처럼 범용적인
AI IDE 기반 서비스로서 길을 갔으면 한다.



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

dd

## 정리

## References
* [Kiro official site](https://kiro.dev/blog/introducing-kiro/)
* [AWS Korea Blog](https://aws.amazon.com/ko/blogs/korea/introducing-kiro/)
* [한국 사용자를 위한 가이드][https://whchoi98.notion.site/Kiro-23104ef7e60e80d3b838e13d2d65498e]
* [한국어 설정 및 한국어 응답 설정][https://whchoi98.notion.site/Kiro-23204ef7e60e802c970cc669d62c1540]
* [해커톤][https://kiro.devpost.com/?trk=dccd318a-a012-40c6-bffb-bd0a6216646d&sc_channel=el]