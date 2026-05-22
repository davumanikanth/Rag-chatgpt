import os
import streamlit as st
from dotenv import load_dotenv
from rag import search_duckduckgo, rag_chain
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

st.set_page_config(page_title="Anime BOT", layout="centered")

st.title("Rag Chatbot")
st.caption("Enter an Rag URL and ask questions")

try:
    st.image("MCP-vs-RAG-image1.png", width=700)
except:
    st.warning("Image not found. App will still work.")

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "processed_url" not in st.session_state:
    st.session_state.processed_url = ""

user_url = st.text_input("Enter Rag URL", key="url_input")

if user_url:
    if user_url != st.session_state.processed_url:
        with st.spinner("Processing URL..."):
            retriever = rag_chain(user_url)

            if retriever:
                st.session_state.retriever = retriever
                st.session_state.processed_url = user_url
                st.success("URL processed successfully!")
            else:
                st.error("Failed to process URL. Please check the URL.")
                st.session_state.retriever = None
                st.session_state.processed_url = ""

    if st.session_state.retriever:
        st.markdown("---")

        question = st.text_input(
            "Ask your question about the uploaded url:",
            key=f"query_{st.session_state.processed_url}"
        )

        if question:
            with st.spinner("Thinking..."):
                try:
                    retrieved_docs = st.session_state.retriever.invoke(question)

                    context = "\n".join(
                        [doc.page_content for doc in retrieved_docs]
                    )

                    duckduckgo_res = search_duckduckgo(question)

                    combined_context = f"""
RAG Context:
{context}

Web Search Context:
{duckduckgo_res}
"""

                    llm = ChatGroq(
                        model="llama-3.3-70b-versatile",
                        temperature=0.3,
                        api_key=os.getenv("API")
                    )

                    prompt = ChatPromptTemplate.from_template(
                        """
Answer the question using the given context.

Rules:
- Answer in bullet points only.
- Mention source as [RAG] or [Search].
- If answer is not available, say "I couldn't find it in the given context."

Context:
{context}

Question:
{input}

Answer:
"""
                    )

                    formatted_prompt = prompt.format_prompt(
                        input=question,
                        context=combined_context
                    )

                    response = llm.invoke(formatted_prompt.to_messages())

                    st.markdown("### Answer")
                    st.success(response.content)

                except Exception as e:
                    st.error(f"Error: {e}")

else:
    st.info("Enter an Anime URL to continue.")

st.markdown("---")
st.caption("Developed by 5132K")