# Import langchain
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.tools import Tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing import Dict, Any

# Import state, prompts, tools 
from state import AgentGraphState
from prompts import router_prompt, investment_strategy_prompt, research_prompt, final_text_formatter, rag_caller_prompt, rag_caller_json
from tools import get_stock_analysis, generate_rag_queries, set_openai_api_key

# Import json
import json


class Agent:
    def __init__(self, state: AgentGraphState):
        self.state = state
        self.openai_api_key = state.get("api_key", "")
        self.openai_version = "gpt-4o"

    def get_llm(self) -> ChatOpenAI:
        return ChatOpenAI(model=self.openai_version, openai_api_key=self.openai_api_key, temperature=0.6)

    def update_state(self, key: str, value: Any) -> AgentGraphState:
        self.state[key] = value
        return self.state

class TradingAgent(Agent):
    def __init__(self, state: AgentGraphState):
        super().__init__(state)
        
        set_openai_api_key(self.openai_api_key)
        
        self.market_research_tool = self._create_market_research_tool()
        self.rag_tool = self._create_rag_tool()
        
        self.rag_caller_json = None
    
    def _create_market_research_tool(self) -> Tool:
        """Create a tool for market research using SerpAPI"""

        return Tool(
            name="market_research",
            description="Useful for getting real-time information about stocks, market trends, company news, and financial data. Input should be a stock ticker symbol or a specific market research question.",
            func=get_stock_analysis
        )
    
    def _create_rag_tool(self) -> Tool:
        """Create a RAG tool for retrieving information from the knowledge base"""

        return Tool(
            name="knowledge_base",
            description="Useful for retrieving historical information about investment strategies, market research, company performance, and financial concepts from the knowledge base. Input should be a specific question.",
            func=generate_rag_queries
        )
    
    def _format_chat_history(self) -> str:
        """Format the chat history as context for the agents"""
        messages = self.state.get("messages", [])
        
        if not messages:
            return "No previous conversation."
        
        formatted_history = "Previous conversation:\n"
 
        for msg in messages[-10:]:
            role = "User" if msg["role"] == "user" else "Assistant"
            formatted_history += f"{role}: {msg['content']}\n\n"
        
        return formatted_history
    
    def router_agent(self) -> AgentGraphState:
        """Router agent that determines which agent to use based on the user's query"""
 
        user_query = self.state["human_input"]
    
        chat_context = self._format_chat_history()
        system_prompt = router_prompt.format(chat_context=chat_context)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ]
        
        llm = self.get_llm()
        ai_msg = llm.invoke(messages)
        response = ai_msg.content.strip()
        
        self.update_state("router_response", response)
        
        return self.state
    
    def rag_caller_agent(self) -> AgentGraphState:
        """RAG caller agent that determines if RAG is necessary and formulates a query"""

        user_query = self.state["human_input"]

        chat_context = self._format_chat_history()
        
        system_prompt = rag_caller_prompt.format(question=user_query, chat_context=chat_context)

        messages = [
            {"role": "system", "content": system_prompt}
        ]
        llm = self.get_llm()
        
        # Use the guided_json configuration if available
        if hasattr(self, 'rag_caller_json'):
            llm = llm.with_structured_output(self.rag_caller_json)
            
        ai_msg = llm.invoke(messages)
        
        # If structured output is used, the response is already parsed
        if hasattr(self, 'rag_caller_json'):
            parsed_response: Dict[str, Any] = ai_msg
        else:
            response = ai_msg.content
            parsed_response = json.loads(response)
        
        # If RAG is needed, execute the RAG query immediately
        if parsed_response.get("need_rag", False):
            rag_results = generate_rag_queries(parsed_response["rag_query"])
            parsed_response["rag_results"] = rag_results
        else:
            parsed_response["rag_results"] = ""

        self.update_state("rag_caller_response", parsed_response)
        
        return self.state
    
    def investment_strategy_agent(self) -> AgentGraphState:
        """Investment strategy agent that provides personalized advice on asset allocation"""

        user_query = self.state["human_input"]

        rag_response = self.state.get("rag_caller_response", {})
        rag_results = rag_response.get("rag_results", "")

        chat_context = self._format_chat_history()

        system_prompt = investment_strategy_prompt.format(
            question=user_query,
            rag_query=rag_results,
            chat_context=chat_context
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ]
        
        llm = self.get_llm()
        ai_msg = llm.invoke(messages)
        response = ai_msg.content
        
        self.update_state("agent_response", response)
        
        return self.state

    def research_agent(self) -> AgentGraphState:
        """Research agent that provides real-time market research and stock analysis"""

        tools = [self.market_research_tool]
        
        user_query = self.state["human_input"]
        chat_context = self._format_chat_history()
        
        system_prompt = research_prompt.format(query=user_query, chat_context=chat_context)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        llm = self.get_llm()
        agent = create_openai_tools_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
        
        response = agent_executor.invoke({"input": user_query})["output"]

        self.update_state("agent_response", response)
        
        return self.state
    
    def end_agent(self) -> AgentGraphState:
        """End agent that formats the response using the formatter prompt"""

        user_query = self.state.get("human_input", "")
        agent_response = self.state.get("agent_response", "")
        router_response = self.state.get("router_response", "")
        
        agent_type = "general"
        if router_response == "investment_strategy_agent":
            agent_type = "investment_strategy"
        elif router_response == "research_agent":
            agent_type = "research"
            
        formatted_prompt = final_text_formatter.format(
            user_query=user_query,
            agent_response=agent_response,
            agent_type=agent_type
        )
        
        messages = [
            {"role": "system", "content": formatted_prompt}
        ]
        
        llm = self.get_llm()
        ai_msg = llm.invoke(messages)
        formatted_response = ai_msg.content
        
        self.update_state("formatted_response", formatted_response)
        self.update_state("agent_response", formatted_response)  
        self.update_state("original_response", agent_response)  
        self.update_state("end_chain", "end_chain")
        
        return self.state