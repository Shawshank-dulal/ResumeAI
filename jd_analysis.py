from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.chains import create_extraction_chain

# Standard Helpers
import pandas as pd
import requests
import time
import json
from datetime import datetime

# Text Helpers
from bs4 import BeautifulSoup
from markdownify import markdownify as md

# For token counting
from langchain.callbacks import get_openai_callback

from dotenv import load_dotenv


def printOutput(output):
    print(json.dumps(output, sort_keys=True, indent=3))


load_dotenv()


llm = ChatOpenAI(
    # model_name="gpt-3.5-turbo",
    model="gpt-3.5-turbo",
    temperature=0,
    max_tokens=2000,
    openai_api_key=openai_api_key,
)


def entity_recognition(schema, data):
    chain = create_extraction_chain(schema, llm)
    response = chain.run(data)
    return response
