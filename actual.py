from langchain.agents import initialize_agent
from langchain.chat_models import ChatOpenAI
import chainlit as cl
from langchain.chat_models import ChatOpenAI
from langchain.agents import load_tools, initialize_agent, AgentType

# Initializes a ChatOpenAI model, loads tools, and sets up an agent chain for zero-shot reaction and description tasks
@cl.on_chat_start
def start():
    llm = ChatOpenAI(temperature=0.5, streaming=True)
    tools = load_tools(
        ["arxiv"]
    )

    agent_chain = initialize_agent(
        tools,
        llm,
        max_iterations=10,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True, 
    )

    cl.user_session.set("agent", agent_chain)

# Processes incoming messages using a pre-initialized agent with asynchronous execution and final answer streaming
@cl.on_message
async def main(message):
    agent = cl.user_session.get("agent")
    cb = cl.LangchainCallbackHandler(stream_final_answer=True)

    await cl.make_async(agent.run)(message, callbacks=[cb])