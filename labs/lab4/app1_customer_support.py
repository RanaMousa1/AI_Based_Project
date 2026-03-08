"""
Enterprise Customer Support System
====================================
A LangChain-based customer support chatbot that uses Retrieval-Augmented Generation
(RAG) over product knowledge bases, conversation memory, and specialized support
tools (order tracking, stock checking) integrated through a LangChain Agent.

Technical features:
  - Document vectorization with RecursiveCharacterTextSplitter chunking
  - 3 distinct product knowledge bases (product_a, product_b, company_policy)
  - ConversationBufferMemory for multi-turn context retention
  - 2 specialized tools integrated via initialize_agent
  - Structured response formatting with category/answer/sources/actions sections
  - Error handling and edge-case management
"""

import os
from dotenv import load_dotenv

load_dotenv()

from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.agents import Tool, initialize_agent, AgentType
from langchain_groq import ChatGroq


# ---------------------------------------------------------------------------
# 1. Document Vectorization (3 Knowledge Bases)
# ---------------------------------------------------------------------------

def load_knowledge_base(knowledge_dir: str = "./knowledge") -> FAISS:
    """Load all .txt files from *knowledge_dir*, split them into chunks,
    and return a FAISS vector store for retrieval."""
    loader = DirectoryLoader(knowledge_dir, glob="./*.txt", loader_cls=TextLoader)
    docs = loader.load()
    if not docs:
        raise FileNotFoundError(
            f"No .txt documents found in '{knowledge_dir}'. "
            "Please add at least 3 product knowledge base files."
        )
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-mpnet-base-v2")
    return FAISS.from_documents(chunks, embeddings)


# ---------------------------------------------------------------------------
# 2. Specialized Support Tools
# ---------------------------------------------------------------------------

def track_order(order_id: str) -> str:
    """Simulate looking up an order by its ID and returning its shipping status."""
    # In production this would query a real order-management system.
    dummy_orders = {
        "12345": "Shipped — expected delivery on March 12, 2026.",
        "67890": "Processing — payment confirmed, preparing for shipment.",
        "11111": "Delivered — received on March 5, 2026.",
    }
    order_id = order_id.strip()
    if order_id in dummy_orders:
        return f"Order #{order_id}: {dummy_orders[order_id]}"
    return f"Order #{order_id}: No record found. Please verify the order number."


def check_stock(item: str) -> str:
    """Check the current stock availability for a given product name."""
    # In production this would query an inventory database.
    stock_db = {
        "technova x200": "In Stock — 142 units available. Ships within 1-2 business days.",
        "edubook 14": "In Stock — 58 units available. Ships within 1-2 business days.",
        "slim sleeve": "In Stock — 300+ units available.",
        "usb-c hub": "Low Stock — only 12 units remaining. Order soon!",
        "wireless mouse m100": "Out of Stock — expected restock on March 20, 2026.",
    }
    item_lower = item.strip().lower()
    for product_name, status in stock_db.items():
        if product_name in item_lower or item_lower in product_name:
            return f"'{item.strip()}': {status}"
    return f"'{item.strip()}': Product not found in inventory. Please check the product name."


# ---------------------------------------------------------------------------
# 3. Structured Response Formatter
# ---------------------------------------------------------------------------

def format_response(category: str, answer: str, sources: str = "", actions: str = "") -> str:
    """Wrap a raw answer into a structured support response template."""
    divider = "-" * 50
    sections = [
        divider,
        f" Category   : {category}",
        divider,
        f" Answer     :\n{answer}",
    ]
    if sources:
        sections += [divider, f" Sources    : {sources}"]
    if actions:
        sections += [divider, f" Suggested Actions : {actions}"]
    sections.append(divider)
    return "\n".join(sections)


# ---------------------------------------------------------------------------
# 4. Build the Agent with RAG + Tools
# ---------------------------------------------------------------------------

def build_support_agent():
    """Construct and return the customer-support agent with RAG retrieval,
    order tracking, and stock checking tools."""

    # Load vector store and build retrieval chain
    vector_db = load_knowledge_base()
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True, output_key="output"
    )

    # RAG chain (used as a tool by the agent)
    rag_chain = ConversationalRetrievalChain.from_llm(
        llm,
        vector_db.as_retriever(search_kwargs={"k": 3}),
        memory=ConversationBufferMemory(
            memory_key="chat_history", return_messages=True, output_key="answer"
        ),
        return_source_documents=True,
    )

    def knowledge_search(query: str) -> str:
        """Search the product knowledge bases for an answer."""
        result = rag_chain.invoke({"question": query})
        answer = result.get("answer", "No answer found.")
        sources = set()
        for doc in result.get("source_documents", []):
            src = doc.metadata.get("source", "unknown")
            sources.add(os.path.basename(src))
        source_str = ", ".join(sorted(sources)) if sources else "N/A"
        return format_response(
            category="Knowledge Base Query",
            answer=answer,
            sources=source_str,
            actions="Ask a follow-up question or type 'quit' to exit.",
        )

    # Define tools the agent can choose from
    tools = [
        Tool(
            name="KnowledgeBase",
            func=knowledge_search,
            description=(
                "Use this tool to answer questions about TechNova products, "
                "company policies, returns, warranties, shipping, or billing. "
                "Input should be the customer's question."
            ),
        ),
        Tool(
            name="OrderTracker",
            func=lambda oid: format_response(
                category="Order Tracking",
                answer=track_order(oid),
                actions="If the status seems wrong, contact support@technova.com.",
            ),
            description=(
                "Use this tool to track an order by its order ID. "
                "Input should be the order number/ID (e.g. '12345')."
            ),
        ),
        Tool(
            name="StockChecker",
            func=lambda item: format_response(
                category="Stock Availability",
                answer=check_stock(item),
                actions="Visit technova.com to place an order.",
            ),
            description=(
                "Use this tool to check if a product is currently in stock. "
                "Input should be the product name (e.g. 'TechNova X200')."
            ),
        ),
    ]

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        memory=memory,
        verbose=False,
        handle_parsing_errors=True,
    )
    return agent


# ---------------------------------------------------------------------------
# 5. Interactive CLI Loop
# ---------------------------------------------------------------------------

def main():
    """Run the customer support chatbot in an interactive loop."""
    print("=" * 55)
    print("  Welcome to TechNova Customer Support ")
    print("  Type your question below. Type 'quit' to exit.")
    print("=" * 55)

    try:
        agent = build_support_agent()
    except Exception as e:
        print(f" Failed to initialize support system: {e}")
        return

    while True:
        try:
            query = input("\n You: ").strip()
            if not query:
                continue
            if query.lower() in ("quit", "exit", "q"):
                print("\nThank you for contacting TechNova Support. Goodbye! ")
                break

            response = agent.invoke({"input": query})
            output = response.get("output", "Sorry, I could not process your request.")
            print(f"\n Support:\n{output}")

        except KeyboardInterrupt:
            print("\n\nSession ended. Goodbye! ")
            break
        except Exception as e:
            print(f"\n Support Error: {e}")
            print("Please try rephrasing your question.")


if __name__ == "__main__":
    main()
