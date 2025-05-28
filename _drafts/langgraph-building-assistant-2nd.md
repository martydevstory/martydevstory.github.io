---
title: LangGraph 어시스턴트 구축하기 (2)
date: 2025-05-25 12:15:43 +/-TTTT
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
series_order: 7
---
LangGraph에서 `메모리`, `휴먼-인-더-루프`, `제어 가능성`에 대해서 살펴보았습니다.

이 개념들 바탕으로 챗 모델 기반 경량 멀티 에이전트 시스템 구성하고 [리서치 및 보고서 생성 자동화 워크플로](https://jxnl.co/writing/2024/06/05/predictions-for-the-future-of-rag/#reports-over-rag)를 구축하겠습니다.

## 4.   리서치 어시스턴트 구축 (실습)

리서치 어시스턴트 구축을 위해 아래 단계로 구성을 진행합니다.

- `소스 선택` : 사용자가 리서치에 사용할 입력 소스를 선택
- `기획` : 사용자는 리서치 주제를 입력하고 시스템은 하위 주제별 AI 분석가 팀을 구성, 하위 주제는 휴먼-인-더-루프로 검토 및 수정
- `LLM 사용` : 각 분석가는 하위 주제별 AI 분석가와 심층 인터뷰 진행, [STORM 논문](https://github.com/langchain-ai/langgraph/blob/main/examples/storm/storm.ipynb)의 방식과 유사하게 입력 소스 바탕으로 다회차 대화, 인터뷰는 각각 `서브그래프(sub-graph)` 내에서 상태를 가지며 수행
- `리서치 수행` : 전문가들을 병렬로 질문에 대한 정보 수집, 전체 인터뷰는 맵리듀스로 동시 진행
- `출력 타입` : 인터뷰에서 수집된 정보는 최종 보고서로 통합, 보고서는 맞춤형 프롬프트를 통해 다양한 출력 타입으로 생성

![리서치 어시스턴트 흐름](assets/drafts/2025-05-21-langgraph-building-assistant-2nd/research_assistant_01.png)
_리서치 어시스턴트 흐름_

### 4.  1.  설정

```python
# 환경 구성
%%capture --no-stderr
%pip install --quiet -U langgraph langchain_openai langchain_community langchain_core tavily-python wikipedia
```

```python
import os, getpass

def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

_set_env("OPENAI_API_KEY")
```

```python
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o", temperature=0) 
```

```python
_set_env("LANGSMITH_API_KEY")
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "langchain-academy"
```

### 4.  2.  분석가 생성 및 휴먼-인-더-루프

분석가를 생성하고 휴먼-인-더-루프를 통해 검토를 합니다.

```python
from typing import List
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

class Analyst(BaseModel):
    affiliation: str = Field(
        description="Primary affiliation of the analyst.",
    )
    name: str = Field(
        description="Name of the analyst."
    )
    role: str = Field(
        description="Role of the analyst in the context of the topic.",
    )
    description: str = Field(
        description="Description of the analyst focus, concerns, and motives.",
    )
    @property
    def persona(self) -> str:
        return f"Name: {self.name}\nRole: {self.role}\nAffiliation: {self.affiliation}\nDescription: {self.description}\n"

class Perspectives(BaseModel):
    analysts: List[Analyst] = Field(
        description="Comprehensive list of analysts with their roles and affiliations.",
    )

class GenerateAnalystsState(TypedDict):
    topic: str # 연구 주제
    max_analysts: int # 분석가 수
    human_analyst_feedback: str # 휴먼 피드백
    analysts: List[Analyst] # 질문하는 분석가들
```

```python
from IPython.display import Image, display
from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

analyst_instructions="""You are tasked with creating a set of AI analyst personas. Follow these instructions carefully:

1. First, review the research topic:
{topic}
  
2. Examine any editorial feedback that has been optionally provided to guide creation of the analysts: 
  
{human_analyst_feedback}
  
3. Determine the most interesting themes based upon documents and / or feedback above.
  
4. Pick the top {max_analysts} themes.

5. Assign one analyst to each theme."""

def create_analysts(state: GenerateAnalystsState):
  
    """ 분석가 생성 """
  
    topic=state['topic']
    max_analysts=state['max_analysts']
    human_analyst_feedback=state.get('human_analyst_feedback', '')
  
    # 구조화된 출력 형식을 강제로 적용
    structured_llm = llm.with_structured_output(Perspectives)

    # 시스템 메시지
    system_message = analyst_instructions.format(topic=topic,
                                                            human_analyst_feedback=human_analyst_feedback, 
                                                            max_analysts=max_analysts)

    # 질문 생성
    analysts = structured_llm.invoke([SystemMessage(content=system_message)]+[HumanMessage(content="Generate the set of analysts.")])
  
    # 분석가 목록을 상태에 기록
    return {"analysts": analysts.analysts}

def human_feedback(state: GenerateAnalystsState):
    """ 중단이 설계된 동작이 없는 노드(no-op) """
    pass

def should_continue(state: GenerateAnalystsState):
    """ 다음 실행될 노드 반환 """

    # 휴먼 피드백 여부 확인
    human_analyst_feedback=state.get('human_analyst_feedback', None)
    if human_analyst_feedback:
        return "create_analysts"
  
    # 그렇지 않으면 종료
    return END

# 노드 및 엣지 설정
builder = StateGraph(GenerateAnalystsState)
builder.add_node("create_analysts", create_analysts)
builder.add_node("human_feedback", human_feedback)
builder.add_edge(START, "create_analysts")
builder.add_edge("create_analysts", "human_feedback")
builder.add_conditional_edges("human_feedback", should_continue, ["create_analysts", END])

# 컴파일
memory = MemorySaver()
graph = builder.compile(interrupt_before=['human_feedback'], checkpointer=memory)

# 그래프 이미지
display(Image(graph.get_graph(xray=1).draw_mermaid_png()))
```

```python
# 입력
max_analysts = 3 
topic = "The benefits of adopting LangGraph as an agent framework"
thread = {"configurable": {"thread_id": "1"}}

# 첫 번째 중단 지점까지 그래프 실행
for event in graph.stream({"topic":topic,"max_analysts":max_analysts,}, thread, stream_mode="values"):
    # 검토
    analysts = event.get('analysts', '')
    if analysts:
        for analyst in analysts:
            print(f"Name: {analyst.name}")
            print(f"Affiliation: {analyst.affiliation}")
            print(f"Role: {analyst.role}")
            print(f"Description: {analyst.description}")
            print("-" * 50)  
```

```
# 출력 - 질문을 할 다양한 관점의 분석가들이 생성됨

Name: Dr. Emily Carter
Affiliation: Tech Innovators Inc.
Role: Technology Adoption Specialist
Description: Dr. Carter focuses on the strategic benefits of adopting new technologies like LangGraph. She is particularly interested in how LangGraph can streamline processes, improve efficiency, and provide a competitive edge to organizations. Her analysis often includes case studies and data-driven insights to support the adoption of innovative frameworks.
--------------------------------------------------
Name: Mr. Raj Patel
Affiliation: Data Security Solutions
Role: Cybersecurity Analyst
Description: Mr. Patel is concerned with the security implications of adopting new frameworks such as LangGraph. His focus is on understanding how LangGraph can enhance or compromise data security within organizations. He evaluates the framework's security features, potential vulnerabilities, and compliance with industry standards.
--------------------------------------------------
Name: Dr. Lisa Nguyen
Affiliation: AI Ethics Consortium
Role: Ethical AI Researcher
Description: Dr. Nguyen examines the ethical considerations of implementing AI frameworks like LangGraph. She is interested in how LangGraph addresses issues such as bias, transparency, and accountability in AI systems. Her work involves assessing the ethical implications of AI technologies and advocating for responsible AI development and deployment.
--------------------------------------------------
```

```python
# 상태를 가져오고 다음 노드 확인
state = graph.get_state(thread)
state.next
```

```python
# 출력

('human_feedback',)
```

```python
# human_feedback 노드인 것처럼 상태를 업데이트
graph.update_state(thread, {"human_analyst_feedback": 
                            "Add in someone from a startup to add an entrepreneur perspective"}, as_node="human_feedback")
```

```
# 출력

{'configurable': {'thread_id': '1',
  'checkpoint_ns': '',
  'checkpoint_id': '1f0200b4-4f8a-6564-8002-6a6a846eef99'}}
```

```python
# 그래프 실행을 계속 진행
for event in graph.stream(None, thread, stream_mode="values"):
    # Review
    analysts = event.get('analysts', '')
    if analysts:
        for analyst in analysts:
            print(f"Name: {analyst.name}")
            print(f"Affiliation: {analyst.affiliation}")
            print(f"Role: {analyst.role}")
            print(f"Description: {analyst.description}")
            print("-" * 50) 
```

```
# 출력 - 휴먼 피드백에서 명시된 것 처럼 기업 관점에서 분석가 추가

Name: Dr. Emily Carter
Affiliation: Tech Innovators Inc.
Role: Technology Adoption Specialist
Description: Dr. Carter focuses on the strategic benefits of adopting new technologies like LangGraph. She is particularly interested in how LangGraph can streamline processes, improve efficiency, and provide a competitive edge to organizations. Her analysis often includes case studies and data-driven insights to support the adoption of innovative frameworks.
--------------------------------------------------
Name: Mr. Raj Patel
Affiliation: Data Security Solutions
Role: Cybersecurity Analyst
Description: Mr. Patel is concerned with the security implications of adopting new frameworks such as LangGraph. His focus is on understanding how LangGraph can enhance or compromise data security within organizations. He evaluates the framework's security features, potential vulnerabilities, and compliance with industry standards.
--------------------------------------------------
Name: Dr. Lisa Nguyen
Affiliation: AI Ethics Consortium
Role: Ethical AI Researcher
Description: Dr. Nguyen examines the ethical considerations of implementing AI frameworks like LangGraph. She is interested in how LangGraph addresses issues such as bias, transparency, and accountability in AI systems. Her work involves assessing the ethical implications of AI technologies and advocating for responsible AI development and deployment.
--------------------------------------------------
Name: Alex Johnson
Affiliation: Tech Innovators Inc.
Role: Startup Entrepreneur
Description: Alex is a co-founder of a tech startup focused on developing AI-driven solutions for small businesses. He is interested in how adopting LangGraph as an agent framework can provide competitive advantages, streamline development processes, and reduce costs for startups. His focus is on practical implementation and scalability of LangGraph in a fast-paced startup environment.
--------------------------------------------------
Name: Dr. Emily Chen
Affiliation: AI Research Lab, University of California
Role: AI Researcher
Description: Dr. Chen is a leading researcher in artificial intelligence and machine learning. Her focus is on the technical benefits of adopting LangGraph, such as its ability to enhance agent communication, improve learning efficiency, and facilitate complex problem-solving. She is interested in how LangGraph can advance academic research and contribute to the development of more sophisticated AI models.
--------------------------------------------------
Name: Michael Thompson
Affiliation: Global Tech Solutions
Role: Enterprise Technology Strategist
Description: Michael is a technology strategist at a large enterprise, responsible for evaluating and integrating new technologies into the company's operations. He is focused on the strategic benefits of LangGraph, including its potential to improve operational efficiency, enhance data integration, and support digital transformation initiatives. Michael is interested in how LangGraph can be leveraged to drive innovation and maintain a competitive edge in the market.
--------------------------------------------------
```

```python
# 만족 시 피드백 없이 진행
further_feedack = None
graph.update_state(thread, {"human_analyst_feedback": 
                            further_feedack}, as_node="human_feedback")
```

```
# 출력

{'configurable': {'thread_id': '1',
  'checkpoint_ns': '',
  'checkpoint_id': '1f0200b5-cd9b-6452-8004-b6f5e600605c'}}
```

```python
# 그래프 실행을 끝까지 계속 진행
for event in graph.stream(None, thread, stream_mode="updates"):
    print("--Node--")
    node_name = next(iter(event.keys()))
    print(node_name)
```

```python
final_state = graph.get_state(thread)
analysts = final_state.values.get('analysts')
```

```python
final_state.next
```

```
# 출력 - 마지막이라 다음 단계가 없음

()
```

```python
for analyst in analysts:
    print(f"Name: {analyst.name}")
    print(f"Affiliation: {analyst.affiliation}")
    print(f"Role: {analyst.role}")
    print(f"Description: {analyst.description}")
    print("-" * 50) 
```

```
# 출력 - stream_mode의 updates에서 변경된 값만 출력

Name: Alex Johnson
Affiliation: Tech Innovators Inc.
Role: Startup Entrepreneur
Description: Alex is a co-founder of a tech startup focused on developing AI-driven solutions for small businesses. He is interested in how adopting LangGraph as an agent framework can provide competitive advantages, streamline development processes, and reduce costs for startups. His focus is on practical implementation and scalability of LangGraph in a fast-paced startup environment.
--------------------------------------------------
Name: Dr. Emily Chen
Affiliation: AI Research Lab, University of California
Role: AI Researcher
Description: Dr. Chen is a leading researcher in artificial intelligence and machine learning. Her focus is on the technical benefits of adopting LangGraph, such as its ability to enhance agent communication, improve learning efficiency, and facilitate complex problem-solving. She is interested in how LangGraph can advance academic research and contribute to the development of more sophisticated AI models.
--------------------------------------------------
Name: Michael Thompson
Affiliation: Global Tech Solutions
Role: Enterprise Technology Strategist
Description: Michael is a technology strategist at a large enterprise, responsible for evaluating and integrating new technologies into the company's operations. He is focused on the strategic benefits of LangGraph, including its potential to improve operational efficiency, enhance data integration, and support digital transformation initiatives. Michael is interested in how LangGraph can be leveraged to drive innovation and maintain a competitive edge in the market.
--------------------------------------------------
```

### 4.  3.  인터뷰 진행

분석가가 전문가에게 질문을 합니다.
코드를 보시면 페르소나를 가진 `분석가`와 달리 `전문가`는 별도 `페르소나 없이 답변`만 합니다.
나중에 해당 부분은 성능을 염두하여 확장은 가능합니다.

```python
import operator
from typing import  Annotated
from langgraph.graph import MessagesState

class InterviewState(MessagesState):
    max_num_turns: int # 대화 횟수
    context: Annotated[list, operator.add] # 출처 문서
    analyst: Analyst # 질문하는 분석가
    interview: str # 인터뷰 대화록
    sections: list # Send() API를 위한 외부 상태로 복제되는 최종 키

class SearchQuery(BaseModel):
    search_query: str = Field(None, description="Search query for retrieval.")
```

```python
question_instructions = """You are an analyst tasked with interviewing an expert to learn about a specific topic. 

Your goal is boil down to interesting and specific insights related to your topic.

1. Interesting: Insights that people will find surprising or non-obvious.
  
2. Specific: Insights that avoid generalities and include specific examples from the expert.

Here is your topic of focus and set of goals: {goals}
  
Begin by introducing yourself using a name that fits your persona, and then ask your question.

Continue to ask questions to drill down and refine your understanding of the topic.
  
When you are satisfied with your understanding, complete the interview with: "Thank you so much for your help!"

Remember to stay in character throughout your response, reflecting the persona and goals provided to you."""

def generate_question(state: InterviewState):
    """ 질문을 생성하는 노드 """

    # 상태 가져오기
    analyst = state["analyst"]
    messages = state["messages"]

    # 질문 생성
    system_message = question_instructions.format(goals=analyst.persona)
    question = llm.invoke([SystemMessage(content=system_message)]+messages)
  
    # 메시지를 상태에 기록
    return {"messages": [question]}
```

### 4.  4.  답변 생성 (병렬화)

전문가는 다음과 같은 다양한 출처에서 정보를 병렬로 수집하여 답변합니다.

- 특정 웹사이트 (예: [WebBaseLoader](https://python.langchain.com/v0.2/docs/integrations/document_loaders/web_base/))
- 색인된 문서 (예: [RAG](https://python.langchain.com/v0.2/docs/tutorials/rag/))
- 웹 검색
- 위키백과 검색

[Tavily](https://tavily.com/)와 같은 웹 검색 도구를 사용하여 웹과 위키백과를 `검색하는 노드`를 생성합니다.

그리고 분석가의 `질문에 답변`하는 노드도 생성합니다.

마지막으로 전체 인터뷰를 저장하고 `인터뷰 요약`("section")을 작성하는 노드도 생성합니다.

```python
def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

_set_env("TAVILY_API_KEY")
```

```python
# 웹 검색 도구
from langchain_community.tools.tavily_search import TavilySearchResults
tavily_search = TavilySearchResults(max_results=3)
```

```python
# 위키피디어 검색 도구
from langchain_community.document_loaders import WikipediaLoader
```

```python
from langchain_core.messages import get_buffer_string

# 웹 검색 쿼리 작성
search_instructions = SystemMessage(content=f"""You will be given a conversation between an analyst and an expert. 

Your goal is to generate a well-structured query for use in retrieval and / or web-search related to the conversation.
  
First, analyze the full conversation.

Pay particular attention to the final question posed by the analyst.

Convert this final question into a well-structured web search query""")

def search_web(state: InterviewState):
  
    """ 웹 검색에서 문서 검색 """

    # 검색 쿼리 생성
    structured_llm = llm.with_structured_output(SearchQuery)
    search_query = structured_llm.invoke([search_instructions]+state['messages'])
  
    # 검색 실행
    search_docs = tavily_search.invoke(search_query.search_query)

    # 문서 포맷
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document href="{doc["url"]}"/>\n{doc["content"]}\n</Document>'
            for doc in search_docs
        ]
    )

    return {"context": [formatted_search_docs]} 

def search_wikipedia(state: InterviewState):
  
    """ 위키피디아에서 문서 검색 """

    # 검색 쿼리 생성
    structured_llm = llm.with_structured_output(SearchQuery)
    search_query = structured_llm.invoke([search_instructions]+state['messages'])
  
    # 검색 실행
    search_docs = WikipediaLoader(query=search_query.search_query, 
                                  load_max_docs=2).load()

    # 문서 포맷
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}\n</Document>'
            for doc in search_docs
        ]
    )

    return {"context": [formatted_search_docs]} 

answer_instructions = """You are an expert being interviewed by an analyst.

Here is analyst area of focus: {goals}. 
  
You goal is to answer a question posed by the interviewer.

To answer question, use this context:
  
{context}

When answering questions, follow these guidelines:
  
1. Use only the information provided in the context. 
  
2. Do not introduce external information or make assumptions beyond what is explicitly stated in the context.

3. The context contain sources at the topic of each individual document.

4. Include these sources your answer next to any relevant statements. For example, for source # 1 use [1]. 

5. List your sources in order at the bottom of your answer. [1] Source 1, [2] Source 2, etc
  
6. If the source is: <Document source="assistant/docs/llama3_1.pdf" page="7"/>' then just list: 
  
[1] assistant/docs/llama3_1.pdf, page 7 
  
And skip the addition of the brackets as well as the Document source preamble in your citation."""

def generate_answer(state: InterviewState):
  
    """ 질문에 답변하는 노드 """

    # 상태 가져오기
    analyst = state["analyst"]
    messages = state["messages"]
    context = state["context"]

    # 질문에 답변 생성
    system_message = answer_instructions.format(goals=analyst.persona, context=context)
    answer = llm.invoke([SystemMessage(content=system_message)]+messages)
  
    # 답변자 이름을 expert로 설정
    answer.name = "expert"
  
    # 상태에 추가하기
    return {"messages": [answer]}

def save_interview(state: InterviewState):
  
    """ 인터뷰 저장 """

    # 메시지 가져오기
    messages = state["messages"]
  
    # 인터뷰를 문자열로 변환
    interview = get_buffer_string(messages)
  
    # interview 상태키에 저장
    return {"interview": interview}

def route_messages(state: InterviewState, 
                   name: str = "expert"):

    """ 질문과 답변 간 라우팅 수행 """
  
    # 메시지 가져오기
    messages = state["messages"]
    max_num_turns = state.get('max_num_turns',2)

    # 전문가 응답 횟수 확인
    num_responses = len(
        [m for m in messages if isinstance(m, AIMessage) and m.name == name]
    )

    # 설정된 최대 응답 횟수 이상이면 종료
    if num_responses >= max_num_turns:
        return 'save_interview'

    # 해당 라우터는 각 질문과 답변 쌍이 실행된 후 실행
    # 마지막 질문을 받아 토론 종료를 알리는 시그널인지 확인
    last_question = messages[-2]
  
    if "Thank you so much for your help" in last_question.content:
        return 'save_interview'
    return "ask_question"

section_writer_instructions = """You are an expert technical writer. 
  
Your task is to create a short, easily digestible section of a report based on a set of source documents.

1. Analyze the content of the source documents: 
- The name of each source document is at the start of the document, with the <Document tag.
  
2. Create a report structure using markdown formatting:
- Use ## for the section title
- Use ### for sub-section headers
  
3. Write the report following this structure:
a. Title (## header)
b. Summary (### header)
c. Sources (### header)

4. Make your title engaging based upon the focus area of the analyst: 
{focus}

5. For the summary section:
- Set up summary with general background / context related to the focus area of the analyst
- Emphasize what is novel, interesting, or surprising about insights gathered from the interview
- Create a numbered list of source documents, as you use them
- Do not mention the names of interviewers or experts
- Aim for approximately 400 words maximum
- Use numbered sources in your report (e.g., [1], [2]) based on information from source documents
  
6. In the Sources section:
- Include all sources used in your report
- Provide full links to relevant websites or specific document paths
- Separate each source by a newline. Use two spaces at the end of each line to create a newline in Markdown.
- It will look like:

### Sources
[1] Link or Document name
[2] Link or Document name

7. Be sure to combine sources. For example this is not correct:

[3] https://ai.meta.com/blog/meta-llama-3-1/
[4] https://ai.meta.com/blog/meta-llama-3-1/

There should be no redundant sources. It should simply be:

[3] https://ai.meta.com/blog/meta-llama-3-1/
  
8. Final review:
- Ensure the report follows the required structure
- Include no preamble before the title of the report
- Check that all guidelines have been followed"""

def write_section(state: InterviewState):

    """ 질문에 답하는 노드 생성 """

    # 상태 가져오기
    interview = state["interview"]
    context = state["context"]
    analyst = state["analyst"]
   
    # 인터뷰 소스 문서 또는 인터뷰 자체를 사용하여 섹션을 작성
    system_message = section_writer_instructions.format(focus=analyst.description)
    section = llm.invoke([SystemMessage(content=system_message)]+[HumanMessage(content=f"Use this source to write your section: {context}")]) 
  
    # 상태에 추가하기
    return {"sections": [section.content]}

# 노드와 엣지 설정
interview_builder = StateGraph(InterviewState)
interview_builder.add_node("ask_question", generate_question)
interview_builder.add_node("search_web", search_web)
interview_builder.add_node("search_wikipedia", search_wikipedia)
interview_builder.add_node("answer_question", generate_answer)
interview_builder.add_node("save_interview", save_interview)
interview_builder.add_node("write_section", write_section)

# 그래프 흐름
interview_builder.add_edge(START, "ask_question")
interview_builder.add_edge("ask_question", "search_web")
interview_builder.add_edge("ask_question", "search_wikipedia")
interview_builder.add_edge("search_web", "answer_question")
interview_builder.add_edge("search_wikipedia", "answer_question")
interview_builder.add_conditional_edges("answer_question", route_messages,['ask_question','save_interview'])
interview_builder.add_edge("save_interview", "write_section")
interview_builder.add_edge("write_section", END)

# 인터뷰 
memory = MemorySaver()
interview_graph = interview_builder.compile(checkpointer=memory).with_config(run_name="Conduct Interviews")

# 그래프 이미지
display(Image(interview_graph.get_graph().draw_mermaid_png()))
```

```python
# 분석가 한 명 선택
analysts[0]
```

```
# 출력

Analyst(affiliation='Tech Innovators Inc.', name='Alex Johnson', role='Startup Entrepreneur', description='Alex is a co-founder of a tech startup focused on developing AI-driven solutions for small businesses. He is interested in how adopting LangGraph as an agent framework can provide competitive advantages, streamline development processes, and reduce costs for startups. His focus is on practical implementation and scalability of LangGraph in a fast-paced startup environment.')
```

주제와 관련된 `llama3.1 논문 인덱스`를 전달하여 인터뷰를 실행합니다.

```python
from IPython.display import Markdown
messages = [HumanMessage(f"So you said you were writing an article on {topic}?")]
thread = {"configurable": {"thread_id": "1"}}
interview = interview_graph.invoke({"analyst": analysts[0], "messages": messages, "max_num_turns": 2}, thread)
Markdown(interview['sections'][0])
```

```
# 출력

## Leveraging LangGraph for AI-Driven Solutions in Startups

### Summary

In the fast-paced environment of tech startups, the adoption of advanced frameworks like LangGraph can provide significant competitive advantages. LangGraph is a flexible framework designed for building stateful applications, particularly beneficial for startups aiming to streamline development processes and reduce costs. It offers a robust solution for managing complex workflows involving multiple agents, which is crucial for startups that need to scale quickly and efficiently.

LangGraph stands out due to its ability to handle complex scenarios with multiple agents and facilitate human-agent collaboration. It provides built-in statefulness, human-in-the-loop workflows, and first-class streaming support, making it an ideal choice for startups looking to implement dynamic and complex tasks [1]. Unlike LangChain, which may become limiting as projects grow in complexity, LangGraph offers enhanced control and flexibility, allowing for the implementation of loops and conditional logic within workflows [1].

One of the most novel aspects of LangGraph is its ability to maintain a persistent shared state, which accumulates information as the agent progresses. This feature is particularly useful for startups as it enables memory within an agent’s workflow, allowing for the carrying over of conversation context or intermediate results between nodes [2]. This persistent state is supported by a built-in persistence layer, which is crucial for long-running processes and continuity across sessions [2].

The real-world adoption of LangGraph by companies like Replit, Uber, and AppFolio highlights its effectiveness in providing control and reliability, which are paramount in customer-facing or mission-critical applications [2]. These companies initially experimented with more autonomous approaches but ultimately chose LangGraph for its predictability and fine-grained control. This trend suggests that while true agent autonomy may be appealing, structured workflow orchestration often proves more practical in production settings [2].

LangGraph also excels in human-agent collaboration, allowing agents to write drafts for review and await approval before acting. This feature, combined with the ability to inspect the agent’s actions and "time-travel" to roll back and take different actions, provides startups with the flexibility to correct course as needed [3]. Additionally, LangGraph's native token-by-token streaming bridges user expectations and agent capabilities, enhancing user experience design [3].

In summary, LangGraph offers a scalable and practical solution for startups looking to leverage AI-driven solutions. Its ability to manage complex workflows, maintain state, and facilitate human-agent collaboration makes it a valuable tool for startups aiming to streamline development processes and reduce costs.

### Sources
[1] https://www.getzep.com/ai-agents/langchain-agents-langgraph  
[2] https://medium.com/@saeedhajebi/langgraph-is-not-a-true-agentic-framework-3f010c780857  
[3] https://www.langchain.com/langgraph
```

### 4.  5.  인터뷰 병렬화 (맵리듀스)

`Send()` API를 사용해 인터뷰를 병렬화하며, 이는 맵 단계에 해당합니다.

리듀스 단계에서는 이 인터뷰들을 결합하여 보고서 본문을 만듭니다.

### 4.  6.  마무리(Finalize)

마지막 단계에서는 보고서의 서론(introduction)과 결론(conclusion)을 작성합니다.

```python
import operator
from typing import List, Annotated
from typing_extensions import TypedDict

class ResearchGraphState(TypedDict):
    topic: str # 연구 주제
    max_analysts: int # 분석가 수
    human_analyst_feedback: str # 인간 피드백
    analysts: List[Analyst] # 질문하는 분석가들
    sections: Annotated[list, operator.add] # Send() API key
    introduction: str # 최종 보고서 서론
    content: str # 최종 보고서 본문
    conclusion: str # 최종 보고서 결론
    final_report: str # 최종 보고서
```

```python
from langgraph.constants import Send

def initiate_all_interviews(state: ResearchGraphState):
    """ 각 인터뷰 서브 그래프를 Send API로 실행하는 맵 단계 """  

    # 사람이 피드백을 제공했는지 확인
    human_analyst_feedback=state.get('human_analyst_feedback')
    if human_analyst_feedback:
        # Return to create_analysts
        return "create_analysts"

    # 그렇지 않으면 Send() API를 통해 인터뷰를 병렬로 시작
    else:
        topic = state["topic"]
        return [Send("conduct_interview", {"analyst": analyst,
                                           "messages": [HumanMessage(
                                               content=f"So you said you were writing an article on {topic}?"
                                           )
                                                       ]}) for analyst in state["analysts"]]

report_writer_instructions = """You are a technical writer creating a report on this overall topic: 

{topic}
  
You have a team of analysts. Each analyst has done two things: 

1. They conducted an interview with an expert on a specific sub-topic.
2. They write up their finding into a memo.

Your task: 

1. You will be given a collection of memos from your analysts.
2. Think carefully about the insights from each memo.
3. Consolidate these into a crisp overall summary that ties together the central ideas from all of the memos. 
4. Summarize the central points in each memo into a cohesive single narrative.

To format your report:
 
1. Use markdown formatting. 
2. Include no pre-amble for the report.
3. Use no sub-heading. 
4. Start your report with a single title header: ## Insights
5. Do not mention any analyst names in your report.
6. Preserve any citations in the memos, which will be annotated in brackets, for example [1] or [2].
7. Create a final, consolidated list of sources and add to a Sources section with the `## Sources` header.
8. List your sources in order and do not repeat.

[1] Source 1
[2] Source 2

Here are the memos from your analysts to build your report from: 

{context}"""

def write_report(state: ResearchGraphState):
    # 모든 섹션들을 가져오기
    sections = state["sections"]
    topic = state["topic"]

    # 모든 섹션을 문자열로 합치기
    formatted_str_sections = "\n\n".join([f"{section}" for section in sections])
  
    # 최종 보고서 요약 생성
    system_message = report_writer_instructions.format(topic=topic, context=formatted_str_sections)  
    report = llm.invoke([SystemMessage(content=system_message)]+[HumanMessage(content=f"Write a report based upon these memos.")]) 
    return {"content": report.content}

intro_conclusion_instructions = """You are a technical writer finishing a report on {topic}

You will be given all of the sections of the report.

You job is to write a crisp and compelling introduction or conclusion section.

The user will instruct you whether to write the introduction or conclusion.

Include no pre-amble for either section.

Target around 100 words, crisply previewing (for introduction) or recapping (for conclusion) all of the sections of the report.

Use markdown formatting. 

For your introduction, create a compelling title and use the # header for the title.

For your introduction, use ## Introduction as the section header. 

For your conclusion, use ## Conclusion as the section header.

Here are the sections to reflect on for writing: {formatted_str_sections}"""

def write_introduction(state: ResearchGraphState):
    # 전체 섹션 모음
    sections = state["sections"]
    topic = state["topic"]

    # 모든 섹션들을 합치기
    formatted_str_sections = "\n\n".join([f"{section}" for section in sections])
  
    # 최종 보고서 요약 생성 - 서론
  
    instructions = intro_conclusion_instructions.format(topic=topic, formatted_str_sections=formatted_str_sections)  
    intro = llm.invoke([instructions]+[HumanMessage(content=f"Write the report introduction")]) 
    return {"introduction": intro.content}

def write_conclusion(state: ResearchGraphState):
    # 전체 섹션 모음
    sections = state["sections"]
    topic = state["topic"]

    # 모든 섹션들을 합치기
    formatted_str_sections = "\n\n".join([f"{section}" for section in sections])
  
    # 최종 보고서 요약 생성 - 결론
  
    instructions = intro_conclusion_instructions.format(topic=topic, formatted_str_sections=formatted_str_sections)  
    conclusion = llm.invoke([instructions]+[HumanMessage(content=f"Write the report conclusion")]) 
    return {"conclusion": conclusion.content}

def finalize_report(state: ResearchGraphState):
    """ The is the "reduce" step where we gather all the sections, combine them, and reflect on them to write the intro/conclusion """
    # 전체 보고서 저장
    content = state["content"]
    if content.startswith("## Insights"):
        content = content.strip("## Insights")
    if "## Sources" in content:
        try:
            content, sources = content.split("\n## Sources\n")
        except:
            sources = None
    else:
        sources = None

    final_report = state["introduction"] + "\n\n---\n\n" + content + "\n\n---\n\n" + state["conclusion"]
    if sources is not None:
        final_report += "\n\n## Sources\n" + sources
    return {"final_report": final_report}

# 노드와 엣지 설정
builder = StateGraph(ResearchGraphState)
builder.add_node("create_analysts", create_analysts)
builder.add_node("human_feedback", human_feedback)
builder.add_node("conduct_interview", interview_builder.compile())
builder.add_node("write_report",write_report)
builder.add_node("write_introduction",write_introduction)
builder.add_node("write_conclusion",write_conclusion)
builder.add_node("finalize_report",finalize_report)

# 조건 분기 로직 설정
builder.add_edge(START, "create_analysts")
builder.add_edge("create_analysts", "human_feedback")
builder.add_conditional_edges("human_feedback", initiate_all_interviews, ["create_analysts", "conduct_interview"])
builder.add_edge("conduct_interview", "write_report")
builder.add_edge("conduct_interview", "write_introduction")
builder.add_edge("conduct_interview", "write_conclusion")
builder.add_edge(["write_conclusion", "write_report", "write_introduction"], "finalize_report")
builder.add_edge("finalize_report", END)

# 컴파일
memory = MemorySaver()
graph = builder.compile(interrupt_before=['human_feedback'], checkpointer=memory)
display(Image(graph.get_graph(xray=1).draw_mermaid_png()))
```

![그래프 시각화](assets/drafts/2025-05-21-langgraph-building-assistant-2nd/research_assistant_02.png)
_그래프 시각화_

LangGraph에 개방형 질문을 합니다.

```python
# 입력값 설정
max_analysts = 3 
topic = "The benefits of adopting LangGraph as an agent framework"
thread = {"configurable": {"thread_id": "1"}}

# 첫 번째 중단 지점까지 그래프 실행
for event in graph.stream({"topic":topic,
                           "max_analysts":max_analysts}, 
                          thread, 
                          stream_mode="values"):
  
    analysts = event.get('analysts', '')
    if analysts:
        for analyst in analysts:
            print(f"Name: {analyst.name}")
            print(f"Affiliation: {analyst.affiliation}")
            print(f"Role: {analyst.role}")
            print(f"Description: {analyst.description}")
            print("-" * 50)  
```

```
# 출력

Name: Dr. Emily Carter
Affiliation: Tech Innovators Inc.
Role: Technology Adoption Specialist
Description: Dr. Carter focuses on the strategic benefits of adopting new technologies like LangGraph. She is particularly interested in how LangGraph can streamline processes, improve efficiency, and provide a competitive edge to organizations. Her analysis often includes case studies and data-driven insights to support the adoption of innovative frameworks.
--------------------------------------------------
Name: Mr. Raj Patel
Affiliation: Data Security Solutions
Role: Cybersecurity Analyst
Description: Mr. Patel is concerned with the security implications of adopting new frameworks such as LangGraph. His focus is on understanding how LangGraph can enhance or compromise data security within organizations. He evaluates the framework's security features, potential vulnerabilities, and compliance with industry standards.
--------------------------------------------------
Name: Ms. Sarah Nguyen
Affiliation: GreenTech Advisors
Role: Sustainability Consultant
Description: Ms. Nguyen examines the environmental impact of adopting technologies like LangGraph. She is interested in how such frameworks can contribute to sustainable practices by reducing resource consumption and improving energy efficiency. Her analysis includes the long-term environmental benefits and the role of technology in achieving sustainability goals.
--------------------------------------------------
```

```python
# We now update the state as if we are the human_feedback node
graph.update_state(thread, {"human_analyst_feedback": 
                                "Add in the CEO of gen ai native startup"}, as_node="human_feedback")
```

```python
# 출력

{'configurable': {'thread_id': '1',
  'checkpoint_ns': '',
  'checkpoint_id': '1f0200c1-f790-6102-8002-08921c73cea2'}}
```

```python
# 체크
for event in graph.stream(None, thread, stream_mode="values"):
    analysts = event.get('analysts', '')
    if analysts:
        for analyst in analysts:
            print(f"Name: {analyst.name}")
            print(f"Affiliation: {analyst.affiliation}")
            print(f"Role: {analyst.role}")
            print(f"Description: {analyst.description}")
            print("-" * 50)  
```

```
# 출력

Name: Dr. Emily Carter
Affiliation: Tech Innovators Inc.
Role: Technology Adoption Specialist
Description: Dr. Carter focuses on the strategic benefits of adopting new technologies like LangGraph. She is particularly interested in how LangGraph can streamline processes, improve efficiency, and provide a competitive edge to organizations. Her analysis often includes case studies and data-driven insights to support the adoption of innovative frameworks.
--------------------------------------------------
Name: Mr. Raj Patel
Affiliation: Data Security Solutions
Role: Cybersecurity Analyst
Description: Mr. Patel is concerned with the security implications of adopting new frameworks such as LangGraph. His focus is on understanding how LangGraph can enhance or compromise data security within organizations. He evaluates the framework's security features, potential vulnerabilities, and compliance with industry standards.
--------------------------------------------------
Name: Ms. Sarah Nguyen
Affiliation: GreenTech Advisors
Role: Sustainability Consultant
Description: Ms. Nguyen examines the environmental impact of adopting technologies like LangGraph. She is interested in how such frameworks can contribute to sustainable practices by reducing resource consumption and improving energy efficiency. Her analysis includes the long-term environmental benefits and the role of technology in achieving sustainability goals.
--------------------------------------------------
Name: Dr. Emily Carter
Affiliation: Tech Innovators Inc.
Role: AI Framework Specialist
Description: Dr. Carter focuses on the technical advantages of adopting LangGraph as an agent framework. She is particularly interested in how LangGraph can enhance the efficiency and scalability of AI systems. Her analysis includes a deep dive into the framework's architecture and its ability to integrate with existing technologies.
--------------------------------------------------
Name: Michael Thompson
Affiliation: FutureTech Ventures
Role: Investment Analyst
Description: Michael evaluates the economic and market potential of LangGraph as an agent framework. He is concerned with the return on investment for companies adopting this technology and how it positions them competitively in the AI landscape. His insights are driven by market trends and financial forecasts.
--------------------------------------------------
Name: Sophia Zhang
Affiliation: AI Pioneers Ltd.
Role: CEO of Gen AI Native Startup
Description: Sophia leads a startup that is native to generative AI technologies. Her focus is on the practical implementation and real-world benefits of LangGraph in startup environments. She is motivated by the framework's potential to accelerate product development and innovation, providing a competitive edge in the fast-paced tech industry.
--------------------------------------------------
```

```python
# 만족한지 확인
graph.update_state(thread, {"human_analyst_feedback": 
                            None}, as_node="human_feedback")
```

```
# 출력

{'configurable': {'thread_id': '1',
  'checkpoint_ns': '',
  'checkpoint_id': '1f0200c2-c647-6dc4-8004-c56f7591cf00'}}
```

```python
# 계속 진행
for event in graph.stream(None, thread, stream_mode="updates"):
    print("--Node--")
    node_name = next(iter(event.keys()))
    print(node_name)
```

```
# 출력

--Node--
conduct_interview
--Node--
conduct_interview
--Node--
conduct_interview
--Node--
write_conclusion
--Node--
write_introduction
--Node--
write_report
--Node--
finalize_report
```

```python
from IPython.display import Markdown
final_state = graph.get_state(thread)
report = final_state.values.get('final_report')
Markdown(report)
```

```
# 출력

# Embracing LangGraph: A New Era in AI Agent Frameworks

## Introduction

In the dynamic realm of AI and machine learning, the demand for efficient and scalable frameworks is ever-increasing. LangGraph, an innovative open-source framework by LangChain, emerges as a transformative tool for developing complex AI agent workflows. By leveraging a graph-based architecture, LangGraph enhances efficiency, scalability, and modularity, making it ideal for production-grade applications. Its seamless integration with existing technologies has already gained the trust of major enterprises. As the global conversational AI market expands, LangGraph offers a robust solution for companies seeking to harness AI's potential, particularly in startups aiming to accelerate innovation and maintain a competitive edge.

---



LangGraph emerges as a transformative framework in the realm of AI and machine learning, offering significant benefits in terms of efficiency, scalability, and market potential. As an open-source framework developed by LangChain, LangGraph introduces a graph-based architecture that models agent steps as nodes in a directed acyclic graph (DAG). This innovative approach provides developers with precise control over branching and error handling, making it particularly suitable for complex, multi-step tasks [1][2].

The architecture of LangGraph enhances the efficiency and scalability of AI systems by managing the intricate relationships between various components of an AI agent workflow. This graph-based programming model allows for flexible workflow definition and precise control over execution paths, making it ideal for production-grade applications with complex decision logic, such as those requiring sophisticated state management, memory persistence, and human-in-the-loop capabilities [1][3]. Its seamless integration with existing technologies has been proven effective in real-world applications by major enterprises like LinkedIn, Uber, Klarna, and GitLab, allowing developers to focus on agent behavior rather than infrastructure concerns [1][4].

LangGraph's modularity, reusability, and scalability are further enhanced by its graph-based approach, which maintains clarity in complex workflows. This makes it a game-changer for AI engineers and developers working on multi-step applications, whether building conversational agents or orchestrating AI-powered pipelines [5]. The framework's robust architecture can handle a high volume of interactions and complex workflows, enabling the development of scalable systems that can grow with organizational needs [6].

Economically, LangGraph is positioned as a promising investment for companies aiming to capitalize on the expanding conversational AI market, projected to grow significantly by 2030 [1]. Its graph-based architecture is particularly effective for applications requiring contextual coherence, such as chatbots and virtual assistants, making it attractive for businesses involved in content creation and marketing [1]. The framework's ability to provide fine-grained control over agent workflows ensures compliance and reliability, crucial in industries like healthcare and finance [2]. Additionally, LangGraph facilitates cost management by optimizing resource allocation and reducing operational costs [2].

LangGraph's adaptability to domain-specific success stories, such as coding assistance and internal data queries, underscores the effectiveness of structured workflow orchestration over true agent autonomy [3]. This aligns with LangGraph's design philosophy, which emphasizes structured control for better outcomes [3]. Successful implementations by companies like Replit, Uber, and AppFolio demonstrate its effectiveness in enhancing user experience, automating processes, and providing valuable insights, ultimately leading to greater ROI for organizations [4].

For startups, LangGraph offers a powerful solution to accelerate product development and innovation. Its low-level, extensible framework allows for the customization of agent roles and workflows, providing startups with the flexibility to tailor AI solutions to their specific needs [1]. The integration with platforms like Amazon Bedrock enhances its capabilities for multi-agent system development, addressing challenges in state management, agent coordination, and workflow orchestration [2]. Real-world implementations by companies such as Uber and Elastic highlight LangGraph's potential to drive business value by automating complex tasks and enhancing decision-making accuracy [3].

In conclusion, LangGraph provides a robust, scalable, and flexible framework for enterprises and startups alike, seeking to enhance the efficiency and scalability of their AI systems. Its graph-based architecture simplifies development and supports large-scale multi-agent applications, making it a critical tool for organizations aiming to gain a competitive edge in the rapidly evolving AI landscape.


---

## Conclusion

LangGraph emerges as a transformative framework in the AI landscape, offering unparalleled efficiency and scalability for complex agent workflows. Its graph-based architecture provides developers with precise control over execution paths, making it ideal for production-grade applications. The framework's seamless integration with existing technologies and its adoption by major enterprises like LinkedIn and Uber underscore its real-world effectiveness. LangGraph's ability to streamline workflows and optimize resource allocation positions it as a valuable asset for businesses aiming to enhance operational capabilities. For startups, its flexibility and scalability accelerate innovation, providing a competitive edge in the rapidly evolving tech industry.

## Sources
[1] https://developer.ibm.com/articles/awb-comparing-ai-agent-frameworks-crewai-langgraph-and-beeai  
[2] https://langfuse.com/blog/2025-03-19-ai-agent-comparison  
[3] https://medium.com/towards-data-science/choosing-between-llm-agent-frameworks-69019493b259  
[4] https://sunnychopper.medium.com/understanding-langgraph-key-concepts-for-developing-agentic-ai-systems-74e537e95a48  
[5] https://medium.com/@noorfatimaafzalbutt/langgraph-streamlining-workflow-design-with-graph-based-ai-applications-6ecefc2c437f  
[6] https://www.datacamp.com/tutorial/langgraph-tutorial  
[7] https://oyelabs.com/langgraph-vs-crewai-vs-openai-swarm-ai-agent-framework/  
[8] https://medium.com/@adnanmasood/the-agentic-imperative-series-part-3-langchain-langgraph-building-dynamic-agentic-workflows-7184bad6b827  
[9] https://medium.com/@saeedhajebi/langgraph-is-not-a-true-agentic-framework-3f010c780857  
[10] https://www.rapidinnovation.io/post/ai-agents-in-langgraph  
[11] https://langchain-ai.github.io/langgraph/  
[12] https://aws.amazon.com/blogs/machine-learning/build-multi-agent-systems-with-langgraph-and-amazon-bedrock/  
[13] https://blog.langchain.dev/is-langgraph-used-in-production/
```

## 정리

먼저 `create_analysts` 노드를 통해 `Role`이 다른 여러 분석가를 생성 후, `human_feedback` 노드를 통해 또 다른 관점의 분석가도 추가했습니다.

그리고 `conduct_interview` 서브 그래프를 `Send API` 호출로 답변을 생성하기 위해 `병렬 실행`하고 웹과 위키피디아에서 검색 결과를 수집했습니다.

또한 각 인터뷰는 `write_node`에서 통합하여 요약하고, `generate_introduction`, `generate_conclusion` 노드에서 서론과 결말을 생성했습니다.

마지막으로 `finalize_report` 노드에서 서론, 본론, 결말의 모든 내용을 합쳐 단일 보고서로 `리듀스` 작업을 수행했습니다.


다음 포스팅에서는 대화 정보를 장기 저장하고 재사용 가능한 `장기 기억 메모리`에 대해서 알아보겠습니다.


## References

* [LangGraph 공식 문서](https://langchain-ai.github.io/langgraph){: target="_blank"}
* [LangChain Academy](https://github.com/langchain-ai/langchain-academy){: target="_blank"}
