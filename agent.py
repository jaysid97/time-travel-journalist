from adk import Agent, SequentialAgent, ToolContext
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

# Tool setup for real-time history retrieval
wiki_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

def save_era_to_state(tool_context: ToolContext, era: str):
    """Saves the user's requested time period to the agent state."""
    tool_context.state["requested_era"] = era
    return {"status": "era_saved"}

# Agent 1: The Historian Researcher (Fact-Finder)
historian_researcher = Agent(
    name="historian_researcher",
    model="gemini-1.5-flash",
    instructions="""Search Wikipedia for key facts about the user's requested era. 
    Identify 3 major events, 2 key historical figures, and 1 interesting cultural fact about everyday life back then.""",
    tools=[wiki_tool],
    output_key="historical_data"
)

# Agent 2: The Time Journalist (Creative Reporter)
time_journalist = Agent(
    name="time_journalist",
    model="gemini-1.5-flash",
    instructions="""As the Time Journalist, write an immersive, first-person 'Live Dispatch'. 
    Use the 'historical_data' to describe the sights, sounds, and atmosphere of the moment as if you are standing there.""",
)

# Multi-Agent Orchestration
app = SequentialAgent(
    agents=[historian_researcher, time_journalist]
)
