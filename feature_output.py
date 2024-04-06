from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain


from app import llm

FEATURES_PROMPT="""
Based on the idea description of an application provided below, generate features for the same based on the Instructions provided below.

<Instructions>
1. Description of Idea: Describe the idea in detail. Include its purpose, target audience and unique selling points of the application.
2. Generate Key Features: Utilize the description to brainstorm and generate a list of key features that the application should have. Consider the target audience, their needs/preferences while generating features.
    2.1 Include both essential functionalities and any additional features that could enhance the application's value proposition or user experience.
    2.2 Identify and highlight any unique or innovative features that set the application apart from existing solutions in the market.
        Emphasize how these features address specific pain points or offer distinct advantages to users.
3. Consider Technical Feasibility: While generating features, consider the technical feasibility of implementing each feature within the application.
4. Provide a detailed list of features for the application based on the provided idea description.
5. Organize the features in a structured format, such as bullet points or a list
6. Provide a catchy title for the application.
</Instructions>

Idea Description: {idea}

"""

TITLE_PROMPT="""
Based on the Idea Description and the Key Features of the application mentioned below.
Return a catchy title of the application.
Idea Description:{idea}
Key Features:{features}
"""

Example={"Title":"","Key_Features":"","Application Description":""}

def get_features(idea,memory):
    prompt=PromptTemplate(
        input_variables=['idea'],template=FEATURES_PROMPT
    )

    llm_chain=LLMChain(llm=llm,prompt=prompt,memory=memory)
    res=llm_chain({"idea":idea})
    return res['text'].split("Idea Description:")[1]

def get_title(idea,features):
    prompt=PromptTemplate(
        input_variables=['idea','features'],template=TITLE_PROMPT
    )
    llm_chain=LLMChain(llm=llm,prompt=prompt)
    res=llm_chain({"idea":idea,"features":features})
    return res

# idea="Learn to fly"
# features=get_features(idea)
# print(features)

