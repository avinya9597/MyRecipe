#!/usr/bin/env python
# coding: utf-8

# In[9]:


# !pip install chain
# !pip install import-ipynb chain


# In[24]:


import datetime
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.vectorstores import Pinecone
import streamlit as st
from Recipe_LangChain import build_query_chain

from sentence_transformers import SentenceTransformer
import torch


# In[33]:


index_name = os.environ(pinecone_indexname)
API_KEY= os.environ(pinecone_key)
ENVIRONMENT = 'gcp-starter'
pc = Pinecone(api_key= API_KEY, environment = ENVIRONMENT)
myindex = pc.Index(host= os.environ(host_name)


# In[34]:


token = os.environ(api_token)
llm = OpenAI(temperature=0.0, api_key=token)


# In[ ]:


## Generate embeddings
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=device)


# In[ ]:


def main():
    
   
    st.markdown(
    """
        <style>
    
        .st-emotion-cache-16txtl3 {
                padding: 2rem 1.5rem;
            }
        
         .reportview-container .sidebar-content {{
                    padding-top: {1}rem;
                }}
                .reportview-container .main .block-container {{
                    padding-top: {1}rem;
                }}
        section[data-testid="stSidebar"] {
        top: 0%; 
        height: 100% !important;
      }
        div[class^="block-container"] {
            padding-top: 0rem;
            padding-left: 0rem;
        }
        </style>
        """,
            unsafe_allow_html=True,
        )
    st.sidebar.title('MyRecipe Bot')

   
    st.title("Make my Recipe!")

    st.write("Fill in the form in the sidebar to get a recipes")

    # Ask the questions and store answers
    st.sidebar.subheader("1. Meal")
    meal = st.sidebar.selectbox(
        "Type of meal", ["Select an option", "Breakfast", "Lunch", "Dinner", "Dessert", "Snack", "Drinks"]
    )

    st.sidebar.subheader("2. Ingredients to Include")
    included_ingredients = st.sidebar.text_input(
        "Include those ingredients",
        value=""
    )

    st.sidebar.subheader("3. Ingredients to Exclude")
    excluded_ingredients = st.sidebar.text_input(
        "Exclude those ingredients",
        value=""
    )

    


    st.sidebar.subheader("4. Describe how you want your meal to be like")
    description = st.sidebar.text_input("Complement the answers above")

    chain, response_format = build_query_chain(llm)

    if st.button("### Recommend Recipes 👩‍🍳"):
                # Now, let's run the query chain
        response = chain.run(food_category= meal,
                            included_ingredients= included_ingredients,
                            excluded_ingredients= excluded_ingredients,
                            description= description,
                            response_format=response_format)

        xq = model.encode(response).tolist()

        # now query
        docs = myindex.query(vector=xq, top_k=5, include_metadata=True)

        def fix_prep_time(prep_time):
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
            for match in docs['matches']
        ]


        recipe_1 = recipe_options[0]
        recipe_2 = recipe_options[1]
        recipe_3 = recipe_options[2]
        # recipe_4 = recipe_options[3]
        # recipe_5 = recipe_options[4]
        
        st.write("---")
        
        st.write("### **Enjoy your meal 😋**")
        
        
        
        try:
            
            tab1, tab2, tab3 = st.tabs([f"Option {i+1}" for i in range(3)])
            
            with tab1:
                
                st.write(
                    f"""
                    <div style="font-size: 20px;">
                        <strong>{recipe_1['name']}</strong>
                        <br>
                        Cooking Time: {recipe_1['time']}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.text("")

                col1, col2, col3, col4 = st.columns(4)
                
                col1.metric("Carbohydrates", f"{round(100*recipe_1['carbohydrates'], 2)}%")
                col2.metric("Protein", f"{round(100*recipe_1['protein'])}%")
                col3.metric("Fat", f"{round(100*recipe_1['fat'])}%")
                col4.metric("Sugar", f"{round(100*recipe_1['sugar'])}%")
                
                ingredientsCol, instructionsCol = st.columns(2)
                
                formatIngredients = [f"* {key}: {value}\n" for key, value in recipe_1['ingredients'].items()]
                formatInstructions = [f"1. {step}.\n" for step in recipe_1['instructions']]
                
                with ingredientsCol:
                    st.header("Ingredients")
                    st.write("\n".join(formatIngredients))

                with instructionsCol:
                    st.header("Instructions")
                    st.write("\n".join(formatInstructions))
                
            with tab2:
                
                st.write(
                    f"""
                    <div style="font-size: 20px;">
                        <strong>{recipe_2['name']}</strong>
                        <br>
                        Cooking Time: {recipe_2['time']}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                st.text("")

                col1, col2, col3, col4,col5 = st.columns(5)
                
                col1.metric("Carbohydrates", f"{round(100*recipe_2['carbohydrates'], 2)}%")
                col2.metric("Protein", f"{round(100*recipe_2['protein'])}%")
                col3.metric("Fat", f"{round(100*recipe_2['fat'])}%")
                col4.metric("Sugar", f"{round(100*recipe_2['sugar'])}%")
                
                ingredientsCol, instructionsCol = st.columns(2)
                
                formatIngredients = [f"* {key}: {value}\n" for key, value in recipe_2['ingredients'].items()]
                formatInstructions = [f"1. {step}\n" for step in recipe_2['instructions']]
                
                with ingredientsCol:
                    st.header("Ingredients")
                    st.write("\n".join(formatIngredients))

                with instructionsCol:
                    st.header("Instructions")
                    st.write("\n".join(formatInstructions))
                
            with tab3:
                
                st.write(
                    f"""
                    <div style="font-size: 20px;">
                        <strong>{recipe_3['name']}</strong>
                        <br>
                        Cooking Time: {recipe_3['time']}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                st.text("")

                col1, col2, col3, col4, col5 = st.columns(5)
                
                col1.metric("Carbohydrates", f"{round(100*recipe_1['carbohydrates'], 2)}%")
                col2.metric("Protein", f"{round(100*recipe_1['protein'])}%")
                col3.metric("Fat", f"{round(100*recipe_1['fat'])}%")
                col4.metric("Sugar", f"{round(100*recipe_1['sugar'])}%")
                
                ingredientsCol, instructionsCol = st.columns(2)
                
                formatIngredients = [f"* {key}: {value}\n" for key, value in recipe_3['ingredients'].items()]
                formatInstructions = [f"1. {step}\n" for step in recipe_3['instructions']]
                
                with ingredientsCol:
                    st.header("Ingredients")
                    st.write("\n".join(formatIngredients))

                with instructionsCol:
                    st.header("Instructions")
                    st.write("\n".join(formatInstructions))
           
                
        except:
            st.write(
                f"""
                    
                    """
            )


if __name__ == "__main__":
    main()


# In[ ]:


#************************************************************************************************************************#

