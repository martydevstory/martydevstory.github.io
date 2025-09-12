---
title: Claude Code 소개
date: 2025-08-18 11:56:43
last_modified_at: 2025-08-18 11:56:43
description : 앤트로픽에서 발표한 Claude Code가 무엇인지 또 어떻게 사용하고 최대한 활용할 수 있는지 알아보겠습니다.
categories: [AI, Vibe-Coding]
tags: [claude-code, llm, generative-ai, claude, vibe-coding]
math: true
toc: true
pin: false
image:
    path: assets/posts/2025-08-18-claude-code-introduce/claude-code-logo.png
    alt:
catagories: blog
sitemap:
  changefreq: weekly
  priority: 0.5
is_series: true
series_title: "Claude Code"
series_order: 1
---

> 학습할 리소스는 [Anthropic Academy](https://www.anthropic.com/learn){: target="_blank"}를 사용합니다.
{: .prompt-info }

Anthropic의 에이전트 코딩 도구인 Claude Code는 터미널에서 실행되며 이전보다 훨씬 빠르게 아이디어를 코드로 변환할 수 있도록 도와줍니다.

포스팅에서는 앞으로 설명할 내용은 다음과 같습니다.

1. 코딩 어시스턴트(Coding Assistant)와 Claude Code를 통해 도구를 사용하는 이유
2. 프로젝트에서 Claude Code를 사용법과 Claude Code를 최대한 활용하는 방법
3. 훅(Hook)과 SDK


## 1. 코딩 어시스턴트(Coding Assistant)란

`코딩 어시스턴트`는 단순하게 코드를 작성하는 도구보다는
복잡한 프로그래밍 작업을 처리하기 위해 `언어 모델(LLM)`을 사용하는 수준 높은 시스템입니다.

### 1.1 코딩 어시스턴트의 작동 방식

오류 메시지를 기반으로 버그를 수정하는 것과 같은 작업을 코딩 어시스턴트에게 요청하면, 개발자가 문제에 접근하는 방식과 유사한 과정을 따릅니다:

![코딩 어시스턴트의 작동 방식](assets/posts/2025-08-18-claude-code-introduce/claude-code-01.png)
_코딩 어시스턴트의 작동 방식_

- `컨텍스트 수집` : 오류가 나타내는 의미와 코드베이스의 어느 부분이 영향을 받는지 또 어떤 파일이 관련되어 있는지 이해합니다.
- `계획 수립` : 코드를 변경하고 테스트를 실행하여 수정사항을 검증하는 등 문제를 해결하는 방법 결정합니다.
- `행동 실행` : 파일을 업데이트하고 명령을 실행하여 설루션 구현합니다.

여기서 핵심은 첫 단계와 마지막 단계에서 어시스턴트가 외부(파일 읽기, 문서 가져오기, 명령 실행하기, 또는 코드 편집하기 등)와 상호작용을 해야 합니다.

### 1.2 코딩 어시스턴트의 도구 사용 (Tool Use)

`언어 모델` 자체는 텍스트만 처리하고 텍스트만 반환할 수 있고 실제로 `파일을 읽거나 명령을 실행할 수는 없습니다`.

코딩 어시스턴트는 이 문제를 `도구 사용`으로 해결합니다.

### 1.3 도구 사용의 작동 방식

`코딩 어시스턴트`에게 요청을 보내면, `언어 모델(LLM)`에게 행동을 요청하는 방법을 가르치는 지시 사항을 자동으로 메시지에 추가합니다.

예를 들어, *"파일을 읽고 싶다면 'ReadFile: 파일명'으로 응답하세요"*와 같은 텍스트를 추가할 수 있습니다.

전체 흐름은 다음과 같습니다:

1. 사용자 요청: *"main.go 파일에 어떤 코드가 작성되어 있나요?"*
2. `코딩 어시스턴트`가 요청에 도구 지시 사항을 추가
3. `언어 모델` 응답: *"ReadFile: main.go"*
4. `코딩 어시스턴트`가 실제 파일을 읽고 내용을 모델에 다시 전송
5. `언어 모델`이 파일 내용을 기반으로 최종 답변 제공

이 시스템을 통해 `언어 모델`은 실제로는 포맷된 텍스트 응답을 생성하는 것뿐이지만, 효과적으로 "파일 읽기", "코드 작성", "명령 실행"을 할 수 있습니다.

### 1.4 Claude Code의 도구 사용이 중요한 이유

모든 `언어 모델`이 도구를 똑같이 잘 사용하는 것은 아닙니다.

Claude 시리즈 모델(Opus, Sonnet, Haiku)은 도구가 무엇을 하는지 이해하고 복잡한 작업을 완료하기 위해 효과적으로 사용하는 데 특히 뛰어납니다.

도구 사용에서의 이러한 강점은 `Claude Code`에 여러 가지 주요 이점을 제공합니다.

![도구 사용](assets/posts/2025-08-18-claude-code-introduce/tool-use.png)
_도구 사용_

#### 1.5 Claude Code의 강력한 도구 사용의 이점

`더 어려운 작업 처리` : Claude는 다양한 도구를 결합하여 복잡한 작업을 처리할 수 있으며, 이전에 본 적이 없는 도구도 사용할 수 있습니다.

`확장할 수 있는 플랫폼` : Claude Code에 새로운 도구를 쉽게 추가할 수 있으며, Claude는 워크플로가 발전함에 따라 이를 사용하도록 적응합니다.

`더 나은 보안` : Claude Code는 인덱싱 없이도 코드베이스를 탐색할 수 있어, 전체 코드베이스를 외부 서버로 보내지 않아도 되는 경우가 많습니다.

### 1.6 Claude Code가 사용할 수 있는 도구

| 도구 | 설명 | 
|------|------| 
| Bash | 환경에서 셸 명령을 실행합니다 | 
| Edit | 특정 파일에 대상 편집을 수행합니다 | 
| Glob | 패턴 매칭을 기반으로 파일을 찾습니다 | 
| Grep | 파일 내용에서 패턴을 검색합니다 | 
| LS | 파일과 디렉토리를 나열합니다 | 
| MultiEdit | 단일 파일에서 여러 편집을 원자적으로 수행합니다 | 
| NotebookEdit | Jupyter 노트북 셀을 수정합니다 | 
| NotebookRead | Jupyter 노트북 내용을 읽고 표시합니다 | 
| Read | 파일의 내용을 읽습니다 | 
| Task | 복잡한 다단계 작업을 처리하기 위해 서브에이전트를 실행합니다 | 
| TodoWrite | 구조화된 작업 목록을 생성하고 관리합니다 | 
| WebFetch | 지정된 URL에서 콘텐츠를 가져옵니다 | 
| WebSearch | 도메인 필터링으로 웹 검색을 수행합니다 | 
| Write | 파일을 생성하거나 덮어씁니다 |

### 1.7 사용 사례

#### 사용 사례 1 : 라이브러리 최적화

자바스크립트 [chalk 라이브러리](https://github.com/chalk/chalk){: target="_blank"} 기반 최적화하는 사용 사례입니다.

라이브러리에서 느려지는 현상이 나타나는 문제에 대한 Claude Code가 분석, 구현, 검증을 실행합니다.

![라이브러리 최적화](assets/posts/2025-08-18-claude-code-introduce/usecase-01.png)
_라이브러리 최적화_

![최적화 실행](assets/posts/2025-08-18-claude-code-introduce/usecase-01-02.png)
_최적화 실행_

#### 사용 사례 2 : 데이터 분석 작업

스트리밍 데이터 분석을 통해 사용자의 이탈률을 노트북에서 분석하고 필요한 경우 차트를 생성하여 시각화를 제공합니다.

![스트리밍 데이터 분석 작업](assets/posts/2025-08-18-claude-code-introduce/usecase-02-01.png)
_스트리밍 데이터 분석 작업_

![스트리밍 데이터 분석 실행](assets/posts/2025-08-18-claude-code-introduce/usecase-02-02.png)
_스트리밍 데이터 분석 실행_

#### 사용 사례 3 : UI 스타일링 작업
`React` 기반 웹의 UI 개선을 위해 Claud Code와 `Playwright` MCP를 사용합니다. 
[`Playwright`](https://playwright.dev/){: target="_blank"}는 Microsoft에서 개발한 웹 애플리케이션 테스트 자동화 도구입니다.

![UI 스타일링 작업](assets/posts/2025-08-18-claude-code-introduce/usecase-03-01.png)
_UI 스타일링 작업_

![UI 스타일링 실행](assets/posts/2025-08-18-claude-code-introduce/usecase-03-02.png)
_UI 스타일링 실행_

#### 사용 사례 4 : Github 통합

Claude Code는 Github Pull Request를 통해 변경된 코드를 읽고 보안에 문제가 되는 사항을 검토합니다.

예로 이메일과 같은 개인정보가 노출되었을 경우, 검토하고 보고서를 작성합니다.

![Github 통합](assets/posts/2025-08-18-claude-code-introduce/usecase-04-01.png)
_Github 통합_

![Claude Code PR](assets/posts/2025-08-18-claude-code-introduce/usecase-04-02.png)
_Claude Code PR Review_

### 1.8 핵심 요점

`코딩 어시스턴트`를 이해하는 것은 몇 가지 핵심 사항으로 요약됩니다:

- `코딩 어시스턴트`는 `언어 모델`을 사용하여 다양한 작업을 완료합니다.

- `언어 모델`은 대부분의 실제 프로그래밍 작업을 처리하기 위해 도구가 필요합니다.

모든 언어 모델이 같은 수준의 기술로 도구를 사용하는 것은 아닙니다.

Claude의 강력한 도구 사용 능력은 `Claude Code`에서 더 나은 보안, 사용자 정의 및 지속성을 가능하게 합니다.

이러한 도구 사용 기능은 단순한 텍스트 생성 모델이 파일을 읽고, 코드베이스를 이해하며, 프로젝트에 의미 있는 변경을 가할 수 있는 강력한 `코딩 어시스턴트`로 변화시키는 것입니다.


다음 포스팅에서는 프로젝트를 구축하고 Claude Code 도구들을 사용하는 방법을 알아보겠습니다.

## References
* [Anthropic 공식](https://www.anthropic.com/){: target="_blank"}
* [Anthropic Academy](https://www.anthropic.com/learn){: target="_blank"}