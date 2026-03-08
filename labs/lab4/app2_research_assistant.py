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
    llm = ChatGroq(model="llama3-8b-8192")
    data = f"WIKI: {wiki.run(topic)}\nARXIV: {arxiv(topic)}"
    
    prompt = PromptTemplate.from_template("Synthesize into a formal report with citations: {data}")
    return llm.predict(prompt.format(data=data))

if __name__ == "__main__":
    print(research("AI in healthcare"))
