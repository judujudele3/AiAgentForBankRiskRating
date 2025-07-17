from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms.base import LLM
from utils.utils import load_prompt, summarize_dataframe
import pandas as pd

def detect_issues_and_suggest_actions(llm: LLM, df: pd.DataFrame) -> str:
    prompt_text = load_prompt("data_exploration_prompt.txt")
    df_summary = summarize_dataframe(df)

    prompt = PromptTemplate(
        input_variables=["data_summary"],
        template=prompt_text
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.run(data_summary=df_summary)
    return response

def identify_risks_and_targets(llm: LLM, df: pd.DataFrame) -> str:
    prompt_text = load_prompt("risk_identification_prompt.txt")
    df_summary = summarize_dataframe(df)

    prompt = PromptTemplate(
        input_variables=["data_summary"],
        template=prompt_text
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.run(data_summary=df_summary)
    return response
