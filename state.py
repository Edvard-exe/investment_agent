# Import typing
from typing import TypedDict, Optional, List, Dict, Any

class AgentGraphState(TypedDict, total=False):
    """State for the agent graph"""

    human_input: str
    api_key: str
    router_response: Optional[str]
    rag_caller_response: Optional[Dict[str, Any]]
    agent_response: Optional[str]
    end_chain: Optional[str]

    # Chat history
    messages: Optional[List[Dict[str, str]]]
    
    # State tracking fields
    current_node: Optional[str]
    execution_path: List[str]
    state_history: List[Dict[str, Any]]