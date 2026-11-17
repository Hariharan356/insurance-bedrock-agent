from typing import Annotated
from typing_extensions import TypedDict

from langchain_aws import ChatBedrockConverse
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_chroma import Chroma

from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from bedrock_embeddings import BedrockTitanEmbeddings


# 1. Initialize ChromaDB
embeddings = BedrockTitanEmbeddings()

vector_store = Chroma(
    collection_name="vehicle_insurance_policy",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)


# 2. RAG Retrieval Tool
@tool
def search_policy(question: str) -> str:
    """Search vehicle insurance policy documents for relevant clauses."""

    results = vector_store.similarity_search(question, k=3)

    if not results:
        return "No relevant policy information found."

    response = []

    for i, doc in enumerate(results, start=1):
        response.append(
            f"Policy Reference {i}:\n"
            f"{doc.page_content}\n"
            f"Source: {doc.metadata.get('source', 'Unknown')}"
        )

    return "\n\n".join(response)


# 3. Claim Eligibility Checker
@tool
def check_claim_eligibility(
    policy_active: bool,
    incident_covered: bool,
    claim_within_policy_period: bool,
    exclusion_applies: bool
) -> str:
    """Check claim eligibility using supplied policy conditions."""

    if not policy_active:
        return "NOT ELIGIBLE: Policy is inactive."

    if not incident_covered:
        return "NOT ELIGIBLE: Incident is not covered."

    if not claim_within_policy_period:
        return "NOT ELIGIBLE: Accident is outside the policy period."

    if exclusion_applies:
        return "NOT ELIGIBLE: A policy exclusion applies."

    return "ELIGIBLE: All supplied demo conditions are satisfied."


# 4. Claim Amount Calculator
@tool
def calculate_claim_amount(
    repair_amount: float,
    deductible: float,
    depreciation: float
) -> str:
    """Calculate illustrative payable amount after deductible and depreciation."""

    if repair_amount < 0 or deductible < 0 or depreciation < 0:
        return "Error: Amounts cannot be negative."

    payable = max(0, repair_amount - deductible - depreciation)

    return (
        f"Repair Amount: Rs. {repair_amount}\n"
        f"Deductible: Rs. {deductible}\n"
        f"Depreciation: Rs. {depreciation}\n"
        f"Illustrative Payable Amount: Rs. {payable}"
    )


# 5. Register exactly three tools
tools = [
    search_policy,
    check_claim_eligibility,
    calculate_claim_amount
]


# 6. Initialize AWS Bedrock
llm = ChatBedrockConverse(
    model="au.anthropic.claude-haiku-4-5-20251001-v1:0",
    region_name="ap-southeast-2",
    temperature=0,
    max_tokens=800
)

llm_with_tools = llm.bind_tools(tools)


# 7. Define LangGraph State
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


# 8. Agent Node
def agent(state: AgentState):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


# 9. Build LangGraph
graph = StateGraph(AgentState)

graph.add_node("agent", agent)
graph.add_node("tools", ToolNode(tools))

graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", tools_condition)
graph.add_edge("tools", "agent")

app = graph.compile()


# 10. Test Scenario
if __name__ == "__main__":

    question = """
    My vehicle repair cost is Rs. 50000.
    My policy deductible is Rs. 5000.
    Assume depreciation is Rs. 2000.

    My policy is active.
    The accident is covered.
    The accident happened within the policy period.
    No exclusion applies.

    Use the policy retrieval tool to verify relevant policy clauses.
    Then check claim eligibility and calculate the illustrative payable amount.

    Use all three tools. Do not invent policy conditions.
    Explain that the calculation is illustrative.
    """

    result = app.invoke({
        "messages": [HumanMessage(content=question)]
    })

    print("\\nInsurance Agent Response:\\n")
    print(result["messages"][-1].content)

    print("\\nTool Execution Details:")

    for message in result["messages"]:
        if getattr(message, "tool_calls", None):
            for call in message.tool_calls:
                print("Tool called:", call["name"])
