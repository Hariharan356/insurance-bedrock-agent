import streamlit as st

from langchain_core.messages import HumanMessage
from insurance_agent import app as insurance_graph


st.set_page_config(
    page_title="Insurance Claim Assistant",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Insurance Claim Assistant")
st.caption("AI-powered Vehicle Insurance Policy and Claim Analysis")

st.info(
    "This demo uses AWS Bedrock Claude Haiku, Titan Embeddings, "
    "ChromaDB, LangGraph and LangSmith."
)

st.header("Claim Details")

with st.form("claim_form"):

    col1, col2 = st.columns(2)

    with col1:
        repair_amount = st.number_input(
            "Repair Amount (₹)",
            min_value=0.0,
            value=50000.0,
            step=1000.0
        )

        deductible = st.number_input(
            "Deductible (₹)",
            min_value=0.0,
            value=5000.0,
            step=500.0
        )

        depreciation = st.number_input(
            "Depreciation (₹)",
            min_value=0.0,
            value=2000.0,
            step=500.0
        )

    with col2:
        policy_active = st.checkbox(
            "Policy is active",
            value=True
        )

        incident_covered = st.checkbox(
            "Accident is covered",
            value=True
        )

        within_period = st.checkbox(
            "Accident occurred within policy period",
            value=True
        )

        exclusion_applies = st.checkbox(
            "A policy exclusion applies",
            value=False
        )

    question = st.text_area(
        "Ask your insurance question",
        value="Review my vehicle insurance claim and explain the relevant policy clauses."
    )

    submitted = st.form_submit_button(
        "Analyze Insurance Claim",
        type="primary",
        use_container_width=True
    )


if submitted:

    prompt = f"""
    Analyze this vehicle insurance claim.

    User-provided claim details:
    Repair amount: Rs. {repair_amount}
    Deductible: Rs. {deductible}
    Depreciation: Rs. {depreciation}

    User-provided conditions:
    Policy active: {policy_active}
    Incident covered: {incident_covered}
    Accident within policy period: {within_period}
    Exclusion applies: {exclusion_applies}

    User question: {question}

    Instructions:
    1. Use search_policy to retrieve relevant policy clauses.
    2. Use check_claim_eligibility with the supplied conditions.
    3. Use calculate_claim_amount with the supplied amounts.
    4. Use all three tools.
    5. Do not invent policy conditions or claim facts.
    6. Clearly explain that eligibility is based on supplied assumptions.
    7. Clearly state that the calculated amount is illustrative only.
    8. Mention relevant policy references in your answer.
    """

    with st.spinner("Analyzing your insurance claim using AI..."):

        try:
            result = insurance_graph.invoke({
                "messages": [HumanMessage(content=prompt)]
            })

            final_answer = result["messages"][-1].content

            st.success("Claim analysis completed!")

            st.header("Insurance Agent Response")
            st.markdown(final_answer)

            st.header("Tool Execution Details")

            executed_tools = []

            for message in result["messages"]:
                if getattr(message, "tool_calls", None):
                    for call in message.tool_calls:
                        executed_tools.append(call["name"])

            for tool_name in executed_tools:
                st.write("✅", tool_name)

        except Exception as error:
            st.error(f"Agent execution failed: {error}")


st.divider()
st.caption(
    "Demo application | AWS Bedrock | LangGraph | ChromaDB | LangSmith"
)
