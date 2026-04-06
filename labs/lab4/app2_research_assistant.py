"""
Academic Research Assistant
============================
A LangChain-based research system that collects information from Wikipedia and ArXiv,
filters it for relevance, synthesises it through an LLMChain, and generates structured
reports in configurable formats (brief, detailed, academic).

Technical features:
  - 2 research tools/sources: Wikipedia and ArXiv
  - Information synthesis via LLMChain (not raw llm.predict)
  - Configurable report formats: brief, detailed, academic
  - Source tracking with real ArXiv metadata (title, authors, URL, date)
  - Keyword-based relevance filtering beyond simple length checks
  - Error handling and edge-case management
"""

import os
from dotenv import load_dotenv

load_dotenv()

from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.document_loaders import ArxivLoader
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import LLMChain


# ---------------------------------------------------------------------------
# Prompt Templates for Each Report Format
# ---------------------------------------------------------------------------

BRIEF_TEMPLATE = PromptTemplate(
    input_variables=["topic", "wiki_data", "arxiv_data", "references"],
    template="""You are an academic research assistant. Based on the sources below,
write a BRIEF research summary on "{topic}".

Structure your response EXACTLY as follows:
## Summary
(2-3 paragraph overview of key findings)

## Key Takeaways
(3-5 bullet points)

## References
{references}

--- Sources ---
WIKIPEDIA:
{wiki_data}

ARXIV PAPERS:
{arxiv_data}
""",
)

DETAILED_TEMPLATE = PromptTemplate(
    input_variables=["topic", "wiki_data", "arxiv_data", "references"],
    template="""You are an academic research assistant. Based on the sources below,
write a DETAILED research report on "{topic}".

Structure your response EXACTLY as follows:
## Introduction
(Background and importance of the topic)

## Literature Overview
(Discussion of existing work from the provided sources)

## Key Findings
(Detailed analysis of the most important points)

## Discussion
(Implications, limitations, and future directions)

## References
{references}

--- Sources ---
WIKIPEDIA:
{wiki_data}

ARXIV PAPERS:
{arxiv_data}
""",
)

ACADEMIC_TEMPLATE = PromptTemplate(
    input_variables=["topic", "wiki_data", "arxiv_data", "references"],
    template="""You are an academic research assistant. Based on the sources below,
write a FORMAL ACADEMIC report on "{topic}".

Structure your response EXACTLY as follows:
## Abstract
(Concise summary of the report in one paragraph)

## 1. Introduction
(Context, motivation, and research questions)

## 2. Methodology
(Describe the information-gathering approach: Wikipedia for general context, ArXiv for peer-reviewed research)

## 3. Findings
(Detailed synthesis of the collected information, organised by theme)

## 4. Analysis & Discussion
(Critical analysis, comparison of sources, gaps in current knowledge)

## 5. Conclusion
(Summary of insights and suggested future research)

## References
{references}

--- Sources ---
WIKIPEDIA:
{wiki_data}

ARXIV PAPERS:
{arxiv_data}
""",
)

FORMAT_MAP = {
    "brief": BRIEF_TEMPLATE,
    "detailed": DETAILED_TEMPLATE,
    "academic": ACADEMIC_TEMPLATE,
}


# ---------------------------------------------------------------------------
# Source Tracking — ArXiv Metadata Extraction
# ---------------------------------------------------------------------------

def extract_arxiv_metadata(docs: list) -> tuple[str, str]:
    """Extract structured metadata from ArXiv document objects.

    Returns:
        - combined_text: concatenated page content from all papers
        - references_block: formatted reference list with titles, authors, URLs
    """
    if not docs:
        return "", ""

    texts = []
    references = []
    for i, doc in enumerate(docs, start=1):
        meta = doc.metadata
        title = meta.get("Title", "Untitled")
        authors = meta.get("Authors", "Unknown")
        published = meta.get("Published", "N/A")
        entry_id = meta.get("entry_id", "")
        # Build a clean URL from the entry_id (ArXiv format)
        url = entry_id if entry_id.startswith("http") else f"https://arxiv.org/abs/{entry_id}"

        texts.append(f"[Paper {i}: {title}]\n{doc.page_content[:2000]}")
        references.append(
            f"[{i}] {title}. {authors}. Published: {published}. URL: {url}"
        )

    return "\n\n".join(texts), "\n".join(references)


# ---------------------------------------------------------------------------
# Relevance Filtering
# ---------------------------------------------------------------------------

def compute_relevance(text: str, topic: str) -> float:
    """Compute a simple keyword-overlap relevance score between *text* and *topic*.

    The score is the fraction of topic keywords that appear in the text (case-insensitive).
    Returns a float between 0.0 and 1.0.
    """
    if not text:
        return 0.0
    topic_keywords = set(topic.lower().split())
    text_lower = text.lower()
    if not topic_keywords:
        return 0.0
    matches = sum(1 for kw in topic_keywords if kw in text_lower)
    return matches / len(topic_keywords)


def filter_sources(wiki_data: str, arxiv_docs: list, topic: str,
                   threshold: float = 0.3) -> tuple[str, list]:
    """Filter Wikipedia and ArXiv sources by keyword relevance.

    Sources whose relevance score falls below *threshold* are discarded.
    Also applies a minimum-length check (< 100 chars) for Wikipedia data.

    Returns the (possibly empty) filtered wiki text and filtered ArXiv docs list.
    """
    # Filter Wikipedia
    if len(wiki_data) < 100 or compute_relevance(wiki_data, topic) < threshold:
        wiki_data = ""

    # Filter ArXiv papers individually
    filtered_arxiv = [
        doc for doc in arxiv_docs
        if compute_relevance(doc.page_content, topic) >= threshold
    ]
    return wiki_data, filtered_arxiv


# ---------------------------------------------------------------------------
# Main Research Function
# ---------------------------------------------------------------------------

def research(topic: str, report_format: str = "detailed") -> str:
    """Collect information from Wikipedia and ArXiv, filter for relevance,
    synthesise via an LLMChain, and return a structured report.

    Args:
        topic: The research topic to investigate.
        report_format: One of 'brief', 'detailed', or 'academic'.

    Returns:
        A formatted research report as a string.
    """
    report_format = report_format.lower().strip()
    if report_format not in FORMAT_MAP:
        return (
            f" Invalid format '{report_format}'. "
            f"Choose from: {', '.join(FORMAT_MAP.keys())}"
        )

    try:
        # ----- Fetch from sources -----
        print(f" Searching Wikipedia for '{topic}'...")
        wiki_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
        wiki_data = wiki_tool.run(topic)

        print(f" Searching ArXiv for '{topic}'...")
        arxiv_docs = ArxivLoader(query=topic, load_max_docs=3).load()

        # ----- Relevance filtering -----
        wiki_data, arxiv_docs = filter_sources(wiki_data, arxiv_docs, topic)

        if not wiki_data and not arxiv_docs:
            return (
                " No sufficiently relevant information found for this topic. "
                "Try a more specific or well-known research topic."
            )

        # ----- Source tracking -----
        arxiv_text, arxiv_references = extract_arxiv_metadata(arxiv_docs)
        wiki_ref = "- Wikipedia: https://en.wikipedia.org/wiki/" + topic.replace(" ", "_")
        all_references = wiki_ref
        if arxiv_references:
            all_references += "\n" + arxiv_references

        # ----- Synthesis via LLMChain -----
        llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.3)
        prompt_template = FORMAT_MAP[report_format]
        chain = LLMChain(llm=llm, prompt=prompt_template)

        print(f"  Generating {report_format} report...")
        result = chain.invoke({
            "topic": topic,
            "wiki_data": wiki_data if wiki_data else "(No relevant Wikipedia data found)",
            "arxiv_data": arxiv_text if arxiv_text else "(No relevant ArXiv papers found)",
            "references": all_references,
        })

        return result.get("text", str(result))

    except Exception as e:
        return f" Research Error: {e}"


# ---------------------------------------------------------------------------
# Interactive CLI
# ---------------------------------------------------------------------------

def main():
    """Run the research assistant interactively."""
    print("=" * 55)
    print("  Academic Research Assistant ")
    print("  Formats: brief | detailed | academic")
    print("  Type 'quit' to exit.")
    print("=" * 55)

    while True:
        try:
            topic = input("\n Enter research topic: ").strip()
            if not topic:
                continue
            if topic.lower() in ("quit", "exit", "q"):
                print("Goodbye! ")
                break

            fmt = input(" Report format (brief/detailed/academic) [detailed]: ").strip()
            if not fmt:
                fmt = "detailed"

            report = research(topic, report_format=fmt)
            print(f"\n{'=' * 55}")
            print(report)
            print(f"{'=' * 55}")

        except KeyboardInterrupt:
            print("\n\nSession ended. Goodbye! ")
            break


if __name__ == "__main__":
    main()
