"""Multi-Agent Interview Coach System."""

from .manager import InterviewManager
from .logger import InterviewLogger
import os
import requests
from bs4 import BeautifulSoup
from typing import List, Literal, Union, Dict, Callable
from pydantic import BaseModel, Field
from pprint import pprint
from langchain.tools import tool
from langgraph.graph import MessagesState
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    FewShotPromptTemplate
)

from langchain_core.runnables import RunnableLambda
from langchain.agents import create_agent
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.runnables import chain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from IPython.display import Image, display

import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
api_key = os.environ.get("OPENAI_API_KEY")
base_url = os.environ.get("OPENAI_BASE_URL")
model = os.environ.get('OPENAI_MODEL')
llm = ChatOpenAI(model=model)


__all__ = ["InterviewManager", "InterviewLogger"]
