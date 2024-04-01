#!/usr/bin/env python
# coding: utf-8

# In[1]:


from langchain.prompts import PromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.chains import LLMChain, SequentialChain


# In[2]:


output_parser = StructuredOutputParser.from_response_schemas([ResponseSchema(name="query_vector", description="The vector used to query the vector database in order to find the most suitable recipes")])

   response_format = output_parser.get_format_instructions()

   prompt = PromptTemplate.from_template(
       template="""
       You are a food recommender bot. Your task is to pick from a recipes database the recipe that better fits the user preference.
       
       Take your time to understand the following user preferences:
       'food_category':{food_category}
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
       input_variables=['food_category', 'included_ingredients', 'excluded_ingredients', 'description'] + ['response_format'],
       output_variables=['query_vector'],
       verbose=False
   )

   return chain, response_format


# In[ ]:


#*************************************************************************************************************************#

