import os

from langchain_ollama import ChatOllama
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_openai import ChatOpenAI

from utils.fs_utils import load_api_key

os.environ['OPENAI_API_KEY'] = load_api_key('openai.api_key')
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_API_KEY'] = load_api_key('langchain.api_key')
os.environ['LANGCHAIN_PROJECT'] = 'langchain-academy'


# Initialize the rate limiter to allow 3 requests per minute
rate_limiter = InMemoryRateLimiter(
    requests_per_second=3/60,  # 3 requests per 60 seconds
    check_every_n_seconds=1,   # Check every second
    max_bucket_size=3          # Allow bursts of up to 3 requests
)

# LLM models
llm_4o = ChatOpenAI(model='gpt-4o', temperature=0, rate_limiter=rate_limiter)
llm_4o_mini = ChatOpenAI(model='gpt-4o-mini', temperature=0, rate_limiter=rate_limiter)
llm_3_5_turbo = ChatOpenAI(model='gpt-3.5-turbo', temperature=0, rate_limiter=rate_limiter)
llm_o1_mini = ChatOpenAI(model='o1-mini', temperature=0, rate_limiter=rate_limiter)

# ollama run --keepalive 30m llama3.2:3b-instruct-q8_0
llm_llama3_2_3b = ChatOllama(model='llama3.2:3b-instruct-q8_0', temperature=0)
# ollama run --keepalive 30m llama3.1:8b-instruct-q8_0
llm_llama3_1_8b = ChatOllama(model='llama3.1:8b-instruct-q8_0', temperature=0)