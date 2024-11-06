import streamlit as st
from search_core import make_search_index, make_search_index_qa, search_query_all, answer_question
import pandas as pd
import json

# Setting page layout
st.set_page_config(
    page_title="Поиск по публикации/вопросы-ответы",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)
# Main page heading
#st.title("Поиск изображений Google Open Images по текстовому запросу")

searchStarted= False
qaStarted= False

# Sidebar
with st.sidebar:
    tab1, tab2 = st.sidebar.tabs(["Поиск по публикации", "Вопросы-ответы"])

with tab1:
    st.header("Поиск по публикации")
    search_input = st.text_input(label='Введите запрос:', value='аккумуляторная батарея')
    searchStarted = st.button('Искать')

with tab2:
    st.header("Вопросы-ответы")
    #qa_input = st.text_input(label='Введите вопрос:', value='Какой ресурс до первого ремонта?')
    #qa_input = st.text_input(label='Введите вопрос:', value='Что входит в состав системы предпускового подогрева?')
    qa_input = st.text_input(label='Введите вопрос:', value='Для чего нужен нагреватель с нагнетателем воздуха?')
    
    qaStarted = st.button('Узнать ответ')
 
if searchStarted==True:
    st.header("Результаты поиска")
    search_result= search_query_all(search_input)
    df = pd.DataFrame(pd.json_normalize(search_result))
    df.columns=['Параграф модуля данных', 'Код МД']
    st.table(df) 
if qaStarted==True:
    st.header("Ответ")
    mode_string = 'strict'
    model_string = '1'
    answer= answer_question(qa_input, mode_string, model_string)
    df = pd.DataFrame(pd.json_normalize(answer))
    df.columns=['Уверенность', 'Ответ', 'Код МД']
    st.table(df)    




