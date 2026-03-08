import os
from dotenv import load_dotenv
load_dotenv()

from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.document_loaders import ArxivLoader
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

# 1. Tools
wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
arxiv = lambda q: ArxivLoader(query=q, load_max_docs=2).load()

# 2. Synthesis Logic
def research(topic):
    try:
        wiki_data = wiki.run(topic)
        arxiv_data = arxiv(topic)
        
        # 3. Filtering Mechanism (Simple Length check)
        if len(wiki_data) < 100 and not arxiv_data: return "Topic not found or lack of sufficient data."

        llm = ChatGroq(model="llama3-8b-8192")
        data = f"WIKI: {wiki_data}\nARXIV: {arxiv_data}"
        
        prompt = PromptTemplate.from_template("Generate a report with deep synthesis and clear source citations: {data}")
        return llm.predict(prompt.format(data=data))
    except Exception as e:
        return f"Research Error: {e}"

if __name__ == "__main__":
    print(research("AI in healthcare"))
