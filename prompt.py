from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain

get_answer_prompt = """
    You are expert in financial statements and auditing.
    You will be given document segments from a financial document.

    Given two segment or chunk of document as chunk1 and chunk2,
    you need to output whether chunk2 is a continuation to chunk1 or not.
    
    If chunk2 is a continuation to chunk1 then your output should be True.
    If chunk2 is not a continuation to chunk1, or chunk2 is a different section, 
    then your output should be False.

    Your output should be either True or False, with no clarifying information.

    example:
    chunk1: some texts.
    chunk2: some texts in continuation.
    output: True

    chunk1: {chunk1}
    chunk2: {chunk2}
    output: 

"""
