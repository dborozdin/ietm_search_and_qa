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

LANGUAGE= 'language'
if LANGUAGE not in st.session_state:
    st.session_state[LANGUAGE]= 'Русский'
    
if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"
    st.session_state.disabled = False
    st.session_state.horizontal = True

with st.sidebar:
    st.session_state[LANGUAGE]= st.sidebar.radio(
            "Язык/Language",
            ["Русский", "English"],
            key="Русский",
            label_visibility=st.session_state.visibility,
            disabled=st.session_state.disabled,
            horizontal=st.session_state.horizontal,
        )
    if st.session_state[LANGUAGE]== 'Русский':    
        st.sidebar.subheader('Демо-публикация: "Урал 44202-80М", 74 модуля данных, русский язык')
        tab1, tab2 = st.sidebar.tabs(["Поиск по публикации", "Вопросы-ответы"])
    else:
        st.sidebar.subheader('Publication asset: "S1000D release 5.0 bike example", 101 data module, english language')
        tab1, tab2 = st.sidebar.tabs(["Indexed search", "Question answering"])

with tab1:
    if st.session_state[LANGUAGE]== 'Русский':
        st.header("Поиск по публикации")
        search_input = st.text_input(label='Введите запрос:', value='аккумуляторная батарея')
        searchStarted = st.button('Искать')
    else:
        st.header("Publication content search")
        search_input = st.text_input(label='Enter query:', value='bicycle wheel')
        searchStarted = st.button('Search')

with tab2:
    if st.session_state[LANGUAGE]== 'Русский':
        st.header("Вопросы-ответы")
        qa_input = st.text_input(label='Введите вопрос:', value='Какой ресурс до первого ремонта?')
        #qa_input = st.text_input(label='Введите вопрос:', value='Что входит в состав системы предпускового подогрева?')
        #qa_input = st.text_input(label='Введите вопрос:', value='Для чего нужен нагреватель с нагнетателем воздуха?')
        qaStarted = st.button('Узнать ответ')
    else:
        st.header("Question answering")
        qa_input = st.text_input(label='Enter question:', value='How many brake pads on the bicycle?')
        qaStarted = st.button('Find out')
 
if searchStarted==True:
    if st.session_state[LANGUAGE]== 'Русский':
        st.header("Результаты поиска")
        search_result= search_query_all(search_input, language="ru")
        df = pd.DataFrame(pd.json_normalize(search_result))
        df.columns=['Параграф модуля данных', 'Код МД']
        st.table(df) 
    else:
        st.header("Search results")
        search_result= search_query_all(search_input, language="en")
        df = pd.DataFrame(pd.json_normalize(search_result))
        df.columns=['Data module paragraph', 'Data module code']
        st.table(df) 
if qaStarted==True:
    if st.session_state[LANGUAGE]== 'Русский':
        st.header("Ответ")
        mode_string = 'strict'
        model_string = '1'
        answer= answer_question(qa_input, mode_string, model_string, language="ru")
        df = pd.DataFrame(pd.json_normalize(answer))
        df.columns=['Уверенность', 'Ответ', 'Код МД']
        st.table(df)    
    else:
        st.header("Answer")
        mode_string = 'strict'
        model_string = '1'
        answer= answer_question(qa_input, mode_string, model_string, language="en")
        df = pd.DataFrame(pd.json_normalize(answer))
        df.columns=['Score', 'Answer', 'Data module code']
        st.table(df)   



