# Import streamlit 
import streamlit as st
from typing import Optional

# Import agents and infrastructure
from agents import TradingAgent
from state import AgentGraphState
from graph import Graph

class StockMarketAssistantApp:
    def __init__(self) -> None:
        """Initialize the application and set up the session state."""

        st.set_page_config(page_title="Stock Market Assistant", page_icon="💬", layout="wide")
        
        # Initialize session state
        if "api_key" not in st.session_state:
            st.session_state.api_key = ""
        
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        if "graph_state" not in st.session_state:
            st.session_state.graph_state = None
    
    def setup_ui(self) -> None:
        """Set up the user interface including sidebar and main content area."""

        st.title("Stock Market Assistant")
        
        # Sidebar for API key input and controls
        with st.sidebar:
            st.title("Configuration")
            api_key = st.text_input(
                "Enter your OpenAI API key:", 
                type="password", 
                value=st.session_state.api_key
            )
            
            if api_key:
                st.session_state.api_key = api_key
                st.success("API key set!")
            
            st.divider()
            
            # Clear chat history button
            if st.button("Clear Chat History"):
                st.session_state.messages = []
                st.session_state.graph_state = None
                st.rerun()
            
            st.markdown("This app uses OpenAI's API. Please provide your own API key.")
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Welcome message if no messages yet and API key is provided
        if not st.session_state.messages and st.session_state.api_key:
            with st.chat_message("assistant"):
                st.markdown("Welcome! I'm your Stock Market Assistant. How can I help you today?")
        
        # Show welcome info if no API key
        if not st.session_state.api_key:
            self._show_welcome_info()
    
    def handle_user_input(self) -> None:
        """Handle user input and generate responses."""

        user_input: Optional[str] = None
        if st.session_state.api_key:
            user_input = st.chat_input("Ask about stocks, investment strategies, or market research...")
        
        if user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(user_input)
            
            # Display assistant thinking
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                message_placeholder.markdown("Thinking...")
    
                # Create initial state
                if st.session_state.graph_state:
                    initial_state = AgentGraphState(
                        human_input=user_input,
                        api_key=st.session_state.api_key,
                        messages=st.session_state.graph_state.get("messages", [])
                    )
                else:
                    initial_state = AgentGraphState(
                        human_input=user_input,
                        api_key=st.session_state.api_key
                    )

                # Process with agent system
                trading_agent = TradingAgent(initial_state)
                graph = Graph(trading_agent)
                workflow = graph.build()
                final_state = workflow.invoke(initial_state)

                # Update UI and session state
                response = final_state.get("agent_response", "I'm sorry, I couldn't process your request.")
                message_placeholder.markdown(response)
                st.session_state.messages = final_state.get("messages", [])
                st.session_state.graph_state = final_state 
    
    def _show_welcome_info(self) -> None:
        """Show welcome information when no API key is provided."""

        st.info("""
        ### Welcome to the Stock Market Assistant!
        
        **How to get started:**
        1. Enter your OpenAI API key in the sidebar to activate the chatbot
        2. Once activated, you'll see a welcome message and can start asking questions
        
        **How to use the chatbot:**
        The assistant uses specialized agents to provide you with expert financial advice:
        
        **Investment Strategy Agent:**
        - Provides personalized advice on asset allocation
        - Helps with portfolio rebalancing strategies
        - Offers trading strategies and investment recommendations
        - Assists with long-term financial planning
        
        **Market Research Agent:**
        - Delivers real-time market research and stock analysis
        - Provides news analysis and market trends
        - Tracks funds and positions
        - Offers insights on company performance and financial data
        
        **Examples of questions you can ask:**
        - "What investment strategy would you recommend for a beginner?" (Investment Strategy)
        - "How should I allocate my portfolio between stocks and bonds?" (Investment Strategy)
        - "Can you analyze the performance of Tesla stock over the past year?" (Market Research)
        - "What are the latest market trends in the tech sector?" (Market Research)
        - "Explain the concept of dollar-cost averaging" (Investment Strategy)
        
        Your conversations will be maintained throughout your session, and you can clear the chat history using the button in the sidebar.
        """)
    
    def run(self) -> None:
        """Run the Streamlit application."""

        self.setup_ui()
        self.handle_user_input()

if __name__ == "__main__":
    app = StockMarketAssistantApp()
    app.run()