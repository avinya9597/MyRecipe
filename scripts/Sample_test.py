#!/usr/bin/env python
# coding: utf-8

# In[25]:


# # install required libraries
# !pip install openai
# !pip install langchain
# !pip install --upgrade langchain openai -q
# !pip install sentence-transformers


# In[8]:


# importing libraries
import os
import openai
from pinecone import Pinecone
import langchain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI


# In[48]:


index_name = "semanticsearch"
token = token
llm = OpenAI(temperature=0.0, api_key=token)


# In[49]:


API_KEY= api_key
ENVIRONMENT = 'gcp-starter'

from pinecone import Pinecone, ServerlessSpec
pc = Pinecone(api_key= API_KEY, environment = ENVIRONMENT)


# In[50]:


myindex = pc.Index(host= host_name)


# In[51]:


from sentence_transformers import SentenceTransformer
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=device)
model


# In[60]:


rquery = 'food_category: Breakfast,preparation_time: 00:00:15, included_ingredients: eggs, bacon,excluded_ingredients: milk,description: I am in the mood for a quick breakfast'
# create the query vector
xq = model.encode(rquery).tolist()

# now query
xc = myindex.query(vector=xq, top_k=3, include_metadata=True)


# In[80]:


from langchain.prompts import PromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.chains import LLMChain, SequentialChain

def build_query_chain(llm):  # Ensure llm is of type Runnable
    output_parser = StructuredOutputParser.from_response_schemas([ResponseSchema(name="query_vector", description="The vector used to query the vector database in order to find the most suitable recipes")])

    response_format = output_parser.get_format_instructions()

    prompt = PromptTemplate.from_template(
        template="""
        You are a food recommender bot. Your task is to pick from a recipes database the recipe that better fits the user preference.
        
        Take your time to understand the following user preferences:
        'food_category':{food_category}
        'preparation_time':{preparation_time}
        'included_ingredients':{included_ingredients}
        'excluded_ingredients':{excluded_ingredients}
        'description':{description}
        
        Now take your time to gather those preferences and create a string to perform a similarity search on a vector database containing food recipes.
        The query must be clear and specific, utilizing relevant features.
        
        {response_format}
        """
    )

    query_chain = LLMChain(llm=llm, prompt=prompt, output_key='query_vector')

    chain = SequentialChain(
        chains=[query_chain],
        input_variables=['food_category', 'preparation_time', 'included_ingredients', 'excluded_ingredients', 'description'] + ['response_format'],
        output_variables=['query_vector'],
        verbose=False
    )

    return chain, response_format, output_parser


# In[81]:


# Now, let's create the query chain
chain, response_format, output_parser = build_query_chain(llm)


# In[82]:


output_parser


# In[85]:


# Now, let's run the query chain
response = chain.run(food_category="Breakfast",
                      preparation_time="00:00:15",
                      included_ingredients="eggs, bacon",
                      excluded_ingredients="milk",
                      description="I am in the mood for a quick breakfast",
                      response_format=response_format)

xq = model.encode(response).tolist()

# now query
xd = myindex.query(vector=xq, top_k=3, include_metadata=True)
# Now, let's query the index with the obtained vector
print(xd)


# In[90]:


def fix_prep_time(prep_time):
    # Your implementation to fix prep time if necessary
    return prep_time

recipe_options = [
    {
        "name": match['id'],
        "time": fix_prep_time(match['metadata']['TotalTime']),
        "carbohydrates": match['metadata']['CarbohydratePercentage'],
        "protein": match['metadata']['ProteinPercentage'],
        "fat": match['metadata']['FatPercentage'],
        "sugar": match['metadata']['SugarPercentage'],
        "Cholestrol": match['metadata']['CholesterolContent'],
        "instructions": match['metadata']['RecipeInstructions'].replace('\n', '').split('.')[:-1],
        "ingredients": eval(match['metadata']['Ingredients'])
    }
    for match in xd['matches']
]


# In[92]:


recipe_options[0]


# In[ ]:


#*************************************************************************************************************************#

