# Import langgraph
from langgraph.graph import StateGraph

# Import state, agents and prompts
from state import AgentGraphState
from agents import TradingAgent
from prompts import rag_caller_json

class Graph:
    def __init__(self, trading_agent: TradingAgent) -> None:
        self.trading_agent = trading_agent
        self.graph = StateGraph(AgentGraphState)
        self.debug = True  

    def _track_state(self, state: AgentGraphState, node_name: str) -> AgentGraphState:
        """Track the state at each node for debugging purposes"""
        if not self.debug:
            return state
            
        if "execution_path" not in state:
            state["execution_path"] = []
        if "state_history" not in state:
            state["state_history"] = []

        state["current_node"] = node_name
        state["execution_path"].append(node_name)
        
        state_snapshot = {k: v for k, v in state.items() 
                         if k not in ["execution_path", "state_history", "current_node"]}
        
        # Add the snapshot to the history
        state["state_history"].append({
            "node": node_name,
            "state": state_snapshot
        })
        
        return state
    
    def _initialize_memory(self, state: AgentGraphState) -> AgentGraphState:
        """Initialize memory for conversation history if it doesn't exist"""

        if "messages" not in state:
            state["messages"] = []
        return state
    
    def _add_human_message(self, state: AgentGraphState) -> AgentGraphState:
        """Add the human message to the conversation history"""
        
        if "messages" not in state:
            state["messages"] = []
            
        # Add the human message to the conversation history
        human_input = state.get("human_input", "")
        if human_input:
            state["messages"].append({"role": "user", "content": human_input})
            
        return state
    
    def _add_ai_message(self, state: AgentGraphState) -> AgentGraphState:
        """Add the AI message to the conversation history"""

        if "messages" not in state:
            state["messages"] = []
            
        # Add the AI message to the conversation history
        ai_response = state.get("agent_response", "")
        if ai_response:
            state["messages"].append({"role": "assistant", "content": ai_response})
            
        return state
    
    def router_node(self, state: AgentGraphState) -> AgentGraphState:
        """Call the router agent and update state"""


        state = self._track_state(state, "router_before")
        
        # Call the agent - pass the state to the agent
        self.trading_agent.state = state
        updated_state = self.trading_agent.router_agent()
        
        # Track state after execution
        updated_state = self._track_state(updated_state, "router_after")
        
        return updated_state
    
    def _route_based_on_response(self, state: AgentGraphState) -> str:
        """Determine which node to route to based on the router response"""

        if state.get("router_response") == "investment_strategy_agent":
            return "rag_caller"
        elif state.get("router_response") == "research_agent":
            return "research"
        else:
            return "research"      
    
    def rag_caller_node(self, state: AgentGraphState) -> AgentGraphState:
        """Call the RAG caller agent and update state"""

        state = self._track_state(state, "rag_caller_before")
        
        # Call the agent - pass the state to the agent
        self.trading_agent.state = state

        # Pass the guided_json configuration to the agent
        self.trading_agent.rag_caller_json = rag_caller_json
        updated_state = self.trading_agent.rag_caller_agent()
        
        updated_state = self._track_state(updated_state, "rag_caller_after")
        
        updated_state["next_node"] = "investment_strategy"
        
        return updated_state
    
    def investment_strategy_node(self, state: AgentGraphState) -> AgentGraphState:
        """Call the investment strategy agent and update state"""

        state = self._track_state(state, "investment_strategy_before")
        
        # Call the agent - pass the state to the agent
        self.trading_agent.state = state
        updated_state = self.trading_agent.investment_strategy_agent()
        
        # Track state after execution
        updated_state = self._track_state(updated_state, "investment_strategy_after")
        
        return updated_state
    
    def research_node(self, state: AgentGraphState) -> AgentGraphState:
        """Call the research agent and update state"""

        state = self._track_state(state, "research_before")
        self.trading_agent.state = state
        updated_state = self.trading_agent.research_agent()
        
        # Track state after execution
        updated_state = self._track_state(updated_state, "research_after")
        
        return updated_state
    
    def end_node(self, state: AgentGraphState) -> AgentGraphState:
        """Call the end agent and update state"""

        # Track state before execution
        state = self._track_state(state, "end_before")
        
        # Call the agent - pass the state to the agent
        self.trading_agent.state = state
        updated_state = self.trading_agent.end_agent()
        
        # Track state after execution
        updated_state = self._track_state(updated_state, "end_after")
        
        return updated_state
    
    def build(self) -> StateGraph:
        """Build the graph"""

        # Add memory management nodes
        self.graph.add_node("initialize_memory", self._initialize_memory)
        self.graph.add_node("add_human_message", self._add_human_message)
        self.graph.add_node("add_ai_message", self._add_ai_message)
        self.graph.add_node("router", self.router_node)

        self.graph.add_node(
            "rag_caller",
            self.rag_caller_node
        )
        
        self.graph.add_node("research", self.research_node)
        self.graph.add_node("investment_strategy", self.investment_strategy_node)
        self.graph.add_node("end", self.end_node)

        self.graph.set_entry_point("initialize_memory")
        self.graph.set_finish_point("add_ai_message")

        self.graph.add_edge("initialize_memory", "add_human_message")
        self.graph.add_edge("add_human_message", "router")

        self.graph.add_conditional_edges(
            "router",
            self._route_based_on_response,
            {
                "rag_caller": "rag_caller",
                "research": "research"
            }
        )

        self.graph.add_edge("rag_caller", "investment_strategy")

        self.graph.add_edge("research", "end")
        self.graph.add_edge("investment_strategy", "end")
        self.graph.add_edge("end", "add_ai_message")
        
        # Compile the workflow
        return self.graph.compile()