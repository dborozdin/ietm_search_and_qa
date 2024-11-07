import os
import lxml.etree as ET
import pandas as pd

import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
import spacy
from tqdm import tqdm
import configparser
import pickle
import re
from transformers import pipeline
import torch
from tqdm import tqdm

import Stemmer
global stemmer
import json

#exclude_tags=['graphic', 'figure']  
#include_tags=['note', 'notePara', 'para']

exclude_tags=['graphic']  
include_tags=['note', 'notePara', 'para', 'title', 'warningAndCautionPara', 'techName', 'infoName']
add_colon_tags=['title', 'techName']
make_lower_parent_tags=['listItemDefinition']
PARSE_PATHS=['//dmodule/content[last()]/procedure[last()]/preliminaryRqmts[last()]',
             '//dmodule/content[last()]/procedure[last()]/mainProcedure[last()]',
             '//dmodule/content[last()]/description[last()]',
             '//dmodule/content[last()]/crew[last()]/crewRefCard[last()]/crewDrill[last()]',
             '//dmodule/identAndStatusSection[last()]/dmAddress[last()]/dmAddressItems[last()]/dmTitle[last()]']

PERSCENTAGE_IN_RATIO=0.5
THRESHOLD=0.1
BATCH_SIZE=8

global nlp, tokenizer_search, tokenizer_qa, device
global search_df, qa_df, SEARCH_DATA
global index_data_loaded, qa_index_data_loaded, qa_model_initialized
global qa_model, qa_model_num

PUBLICATION_DEMO_RU_PATH="publications/Demo publication in Russian"
PUBLICATION_DEMO_EN_PATH="publications/Bike Data Set for Release number 5.0"
PUBLICATION_PATH=PUBLICATION_DEMO_RU_PATH
TOKENIZER_SEARCH_FILENAME='tokenizer_search.pickle'
TOKENIZER_QA_FILENAME='tokenizer_qa.pickle'
INDEX_FOLDER= PUBLICATION_PATH+ os.sep+ "index"
INDEX_FOLDER_RU= PUBLICATION_DEMO_RU_PATH+ os.sep+ "index"
INDEX_FOLDER_EN= PUBLICATION_DEMO_EN_PATH+ os.sep+ "index"
#print('INDEX_FOLDER:', INDEX_FOLDER)
TOKENIZER_SEARCH_PATH= INDEX_FOLDER+ os.sep+ TOKENIZER_SEARCH_FILENAME
TOKENIZER_SEARCH_PATH_RU= INDEX_FOLDER_RU+ os.sep+ TOKENIZER_SEARCH_FILENAME
TOKENIZER_SEARCH_PATH_EN= INDEX_FOLDER_EN+ os.sep+ TOKENIZER_SEARCH_FILENAME
TOKENIZER_QA_PATH= INDEX_FOLDER+ os.sep+ TOKENIZER_QA_FILENAME
TOKENIZER_QA_PATH_RU= INDEX_FOLDER_RU+ os.sep+ TOKENIZER_QA_FILENAME
TOKENIZER_QA_PATH_EN= INDEX_FOLDER_EN+ os.sep+ TOKENIZER_QA_FILENAME
#print('TOKENIZER_SEARCH_PATH:', TOKENIZER_SEARCH_PATH)
PUBLICATION_LANGUAGE="ru"

nlp=None  
search_df=None
qa_df=None
index_data_loaded=False
qa_index_data_loaded=False
SEARCH_DATA= None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
qa_model_initialized=False
 

def get_xpath_one(tree, xpath):
    res = tree.xpath(xpath)
    if res:
        return res[0]
        
def get_dmc(doc):
    dmc=""
    node= get_xpath_one(doc, '//dmCode')
    dmc='DMC-'+'-'.join([node.get('modelIdentCode'), \
                         node.get('itemLocationCode'), \
                         node.get('systemCode'), \
                         node.get('subSystemCode')+node.get('subSubSystemCode'), \
                         node.get('assyCode'),\
                         node.get('disassyCode')+node.get('disassyCodeVariant'),\
                         node.get('infoCode')+node.get('infoCodeVariant'),\
                         node.get('systemDiffCode')])
                       
    #print('dmc:                  ', dmc)
    return dmc
    
def is_float(string):
    if string.replace(".", "").replace(",", "").replace("+", "").replace("-", "").isnumeric():
        return True
    else:
        return False
        
    
def stringify_children(node, texts, pis, excludeDigits=True):
    s = node.text
    if (s != None) and (s.isspace()==False):
        if excludeDigits:
            if is_float(s)==False:
               texts.add(s) 
        else:
            texts.add(s)
    for child in node:
        if child.tag not in exclude_tags:
            if child not in pis:
                stringify_children(child, texts, pis)
    return 
    
def stringify_children_incl(node, texts, pis, make_lower=False):
    ET.strip_tags(node, 'internalRef')
    ET.strip_tags(node, 'emphasis')
    s = node.text
    if s and make_lower==True:
        s= s.lower()
    if s and node.tag in add_colon_tags:
        s=s+':'
    #print('s', s)
    clear_s= clear_text(s)
    if (s != None) and (s.isspace()==False) and (clear_s!='') and (clear_s):
        print('s:', s)
        print('clear_text(s):', clear_text(s))
        texts.append(s) 

    for child in node:
        #print('child.tag:', child.tag)
        if (len(child.getchildren())>0) or (child.tag in include_tags):
            if (child not in pis) and (child.tag not in exclude_tags):
                make_lower=False
                if node.tag in make_lower_parent_tags:
                    make_lower=True
                stringify_children_incl(child, texts, pis, make_lower)
    return 
    
def clear_text(text):
    #print('clear_text!')
    clean_text = re.sub(r'(?:(?!\u0301)[\W\d_])+', ' ', str(text).lower())
    return clean_text

def lemmatize_and_stemm(df_r):
    global nlp, stemmer
    #print('lemmatize_and_stemm!')
    disabled_pipes = [ "parser",  "ner"]
    if PUBLICATION_LANGUAGE=="ru":
        nlp = spacy.load('ru_core_news_sm', disable=disabled_pipes)
        stemmer= Stemmer.Stemmer('ru')#russian
    else:
        nlp = spacy.load('en_core_web_sm', disable=disabled_pipes)
        stemmer= Stemmer.Stemmer('en')#english

    lemm_texts = []
    stem_texts=[]

    for doc in tqdm(nlp.pipe(df_r['lemm_text'].values, disable = disabled_pipes), total=df_r.shape[0]):
        lemm_text = " ".join([i.lemma_ for i in doc])    
        lemm_texts.append(lemm_text) 
        stem_text = " ".join([stemmer.stemWord(i.text) for i in doc])  
        stem_texts.append(stem_text) 

    df_r['lemm_text']= lemm_texts
    df_r['stem_text']= stem_texts
    df_r=df_r.drop_duplicates()
    #print('lemmatization and stemming success!')
    return 
    
def tokenize_text(df_r, save_filename):
    #global tokenizer_search
    #print('tokenize_text!')
    
    #try:
        #with open('tokenizer.pickle', 'rb') as handle:
            #tokenizer = pickle.load(handle)
            #print('tokenizer loaded from file')
    #except Exception as e:
    tokenizer = Tokenizer(oov_token='<oov>') 
    print('tokenizer created')
        
    texts= pd.concat([df_r['lemm_text'],df_r['stem_text']])
    tokenizer.fit_on_texts(texts)
    total_words = len(tokenizer.word_index) + 1
    print("Total number of words: ", total_words) 
    with open(save_filename, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)    
    return tokenizer
    
def make_final_index(df_r, tokenizer, index_filename='search_index.csv', includePlainText=True):
    print('make_final_index!')
    tokens=[]
    labels=[]
    dmcs=[]
    texts=[]
    for index, row in tqdm(df_r.iterrows()):
        #print('row:', row)
        text= row['text']
        lemm_token= tokenizer.texts_to_sequences([row['lemm_text']])[0]
        stem_token= tokenizer.texts_to_sequences([row['stem_text']])[0]
        dmc= row['DMC']
        #print(str(row['label_enc'])+':'+dmc)
        tokens.append(lemm_token)
        labels.append(row['label_enc'])
        dmcs.append(dmc)
        texts.append(text)
        tokens.append(stem_token)
        labels.append(row['label_enc'])
        dmcs.append(dmc)
        texts.append(text)
    columns= ['tokens', 'labels', 'DMC']
    data= {'tokens': tokens, 'labels': labels, 'DMC': dmcs}
    if includePlainText==True:
        columns= ['tokens', 'labels', 'text', 'DMC']
        data= {'tokens': tokens, 'labels': labels, 'text': texts, 'DMC': dmcs}
    s_df= pd.DataFrame(columns=columns, data= data)  
    s_df= s_df.loc[s_df.astype(str).drop_duplicates().index]
    print('final index info:')
    print(s_df.info())
    s_df.to_csv(index_filename, sep=';', index=False)
    #print(f'results saved to {index_filename}')
    return s_df
        
def make_search_index(path): 
    global nlp, tokenizer_search, search_df, index_data_loaded  
    #print('make_search_index!')
    directory= path.replace('"', '')
    #print(f'path: {directory}')
    df_r= pd.DataFrame(columns=['text'])     

    for file in os.listdir(directory):
        filename = file#os.fsdecode(file)
        if 'PMC' in filename:
            continue
        #print('filename: ', filename)
        if filename.lower().endswith(".xml")==False: 
            continue
        filepath= directory+ os.sep+ filename
        print('filepath:', filepath)
        
        doc= ET.parse(filepath)
        dmc= get_dmc(doc)

        file_texts=set()
        pis = doc.xpath("//processing-instruction()")
        for node in doc.xpath('//dmodule'):
            stringify_children(node, file_texts, pis)

        #print('file_texts:', file_texts)
        df= pd.DataFrame(columns=['text'], data= file_texts)
        df['DMC']= dmc
        df_r= pd.concat([df_r, df], ignore_index=True)
    print('parsing results:')
    print(df_r.info())
    #PARSING_INDEX_FILENAME='strings_with_DMC.csv'
    #print(f'parsing results saved to: {PARSING_INDEX_FILENAME}')
    #df_r.to_csv(PARSING_INDEX_FILENAME, index=False, sep = ';')
    
    df_r['lemm_text']=df_r['text'].apply(clear_text) 
    lemmatize_and_stemm(df_r)
    df_r= df_r.reset_index(drop=True)
    df_r['label_enc']= df_r.index
    tokenizer_search= tokenize_text(df_r, TOKENIZER_SEARCH_PATH)
    #print('tokenizer before make_final_index:', tokenizer_search)
    search_df= make_final_index(df_r, tokenizer_search)
    index_data_loaded= True
    return len(search_df)
    
def make_search_index_qa(path): 
    global nlp, tokenizer_qa, qa_df, qa_index_data_loaded  
    #print('make_search_index_qa!')
    directory= path.replace('"', '')
    #print(f'path: {directory}')
    df_r= pd.DataFrame(columns=['text'])     

    for file in os.listdir(directory):
        filename = file#os.fsdecode(file)
        if 'PMC' in filename:
            continue
        #print('filename: ', filename)
        if filename.lower().endswith(".xml")==False: 
            continue
        filepath= directory+ os.sep+ filename
        #print('filepath:', filepath)
        
        doc= ET.parse(filepath)
        dmc= get_dmc(doc)

        paths= PARSE_PATHS
        
        pis = doc.xpath("//processing-instruction()")
        for pi in pis:
            if pi.getparent()!=None:
                ET.strip_tags(pi.getparent(), pi.tag)
        
        cntr=1
        for expr in paths:
            try:
                x_path_result = doc.xpath(expr)
            except ET.XPathEvalError:
                continue
            
            if not x_path_result:
                continue
            file_texts=[]    
            dmc_with_chapter= f'{dmc}({cntr})'    
            for node in x_path_result:#doc.xpath(expr):
                stringify_children_incl(node, file_texts, pis)
            cntr=cntr+1
            #print('file_texts:',file_texts)
            #print('file_texts len:',len(file_texts))
            if len(file_texts)==0:
                continue
            concat_texts=[' \n '.join(file_texts)]    
            #print('file_texts:', file_texts)    

            #df= pd.DataFrame(columns=['text'], data= file_texts)
            df= pd.DataFrame(columns=['text'], data= concat_texts)
            df['DMC']= dmc_with_chapter
            df_r= pd.concat([df_r, df], ignore_index=True)
    #print('parsing results:')
    #print(df_r.info())
    #PARSING_INDEX_FILENAME='strings_with_DMC.csv'
    #print('parsing results saved to: {PARSING_INDEX_FILENAME}')
    #df_r.to_csv(PARSING_INDEX_FILENAME, index=False, sep = ';')
    
    df_r['lemm_text']=df_r['text'].apply(clear_text) 
    lemmatize_and_stemm(df_r)
    df_r= df_r.reset_index(drop=True)
    df_r['label_enc']= df_r.index
    tokenizer_qa= tokenize_text(df_r, TOKENIZER_QA_PATH)
    qa_df= make_final_index(df_r, tokenizer_qa, index_filename='qa_index.csv')
    qa_index_data_loaded= True
    return len(qa_df)

def convert2list(string):
    x = json.loads(string)
    lst=[]
    for n in x:
        #print(x)
        lst.append(int(n))
    return lst
    
def load_index_data():
    global nlp, tokenizer_search, search_df, index_data_loaded
    print('load_index_data!')
    print('PUBLICATION_LANGUAGE:', PUBLICATION_LANGUAGE)
    #spacy    
    disabled_pipes = [ "parser",  "ner"]
    if PUBLICATION_LANGUAGE=="ru":
        nlp = spacy.load('ru_core_news_sm', disable=disabled_pipes)
        stemmer= Stemmer.Stemmer('ru')#russian
    else:
        nlp = spacy.load('en_core_web_sm', disable=disabled_pipes)
        stemmer= Stemmer.Stemmer('en')#english
    #print('spacy loaded:', nlp)
    #tokenizer
    if PUBLICATION_LANGUAGE=="ru":
        with open(TOKENIZER_SEARCH_PATH_RU, 'rb') as handle:
            tokenizer_search = pickle.load(handle)
    else:
        with open(TOKENIZER_SEARCH_PATH_EN, 'rb') as handle:
            tokenizer_search = pickle.load(handle)
    #print('tokenizer loaded:', tokenizer)
    #index
    if PUBLICATION_LANGUAGE=="ru":
        search_index_path= INDEX_FOLDER_RU+os.sep+'search_index.csv'
    else:
        search_index_path= INDEX_FOLDER_EN+os.sep+'search_index.csv'
    search_df= pd.read_csv(search_index_path, sep=';')
    print('index file loaded:', search_df.info())    
    search_df['tokens']= search_df['tokens'].apply(convert2list)
    index_data_loaded= True
    return nlp, tokenizer_search, search_df
    
def load_index_data_qa():
    global nlp, tokenizer_qa, qa_df, qa_index_data_loaded, stemmer
    #print('load_index_data_qa!')
    #spacy    
    disabled_pipes = [ "parser",  "ner"]
    if PUBLICATION_LANGUAGE=="ru":
        nlp = spacy.load('ru_core_news_sm', disable=disabled_pipes)
        stemmer= Stemmer.Stemmer('ru')#russian
    else:
        nlp = spacy.load('en_core_web_sm', disable=disabled_pipes)
        stemmer= Stemmer.Stemmer('en')#english
    print('spacy loaded:', nlp)
    #tokenizer
    if PUBLICATION_LANGUAGE=="ru":
        with open(TOKENIZER_QA_PATH_RU, 'rb') as handle:
            tokenizer_qa = pickle.load(handle)
    else:
        with open(TOKENIZER_QA_PATH_EN, 'rb') as handle:
            tokenizer_qa = pickle.load(handle)
    #print('tokenizer loaded:', tokenizer_qa)
    #index
    if PUBLICATION_LANGUAGE=="ru":
        qa_index_path= INDEX_FOLDER_RU+os.sep+'qa_index.csv'
    else:
        qa_index_path= INDEX_FOLDER_EN+os.sep+'qa_index.csv'
    qa_df= pd.read_csv(qa_index_path, sep=';')
    #print('index qa file loaded:', qa_df.info())    
    qa_df['tokens']= qa_df['tokens'].apply(convert2list)
    qa_index_data_loaded= True
    return nlp, tokenizer_qa, qa_df    
 
def customIsIn(x , tokens):
    result= False
    cnt_in=0
    for val in x:
        if val in tokens:
            cnt_in+=1
            PERSCENTAGE_IN= cnt_in/len(tokens)
            if PERSCENTAGE_IN>=PERSCENTAGE_IN_RATIO:
                return True
    return result

def get_lemmed_stemmed_text(text):
    global nlp, stemmer
     #print('nlp loaded or not:', nlp)
    if PUBLICATION_LANGUAGE=="ru":
        spacy_stopwords = spacy.lang.ru.stop_words.STOP_WORDS #russian  
        stemmer= Stemmer.Stemmer('ru')#russian
    else:
        spacy_stopwords = nlp.Defaults.stop_words #english  
        stemmer= Stemmer.Stemmer('en')#english
    #print('spacy_stopwords:', spacy_stopwords)
    doc = nlp(clear_text(text))
    # Remove stop words
    doc_cleared = [token for token in doc if not token.is_stop]
    #print('doc_cleared:', doc_cleared)
    lemm_text = " ".join([i.lemma_ for i in doc_cleared if not i.lemma_ in spacy_stopwords])  
    print(f'lemm_text: {lemm_text}')
    stem_text = " ".join([stemmer.stemWord(i.text) for i in doc_cleared if not stemmer.stemWord(i.text) in spacy_stopwords])  
    print(f'stem_text: {stem_text}')
    return lemm_text, stem_text

def search_query_any(query, df=None, tokenizer=None):
    global SEARCH_DATA, search_df, index_data_loaded, stemmer
    print('search_query_any!')
    print(f'query: {query}')
    if index_data_loaded==False:
        load_index_data()
    SEARCH_DATA= df
    if df is None:
        if index_data_loaded==False:
            load_index_data()
            SEARCH_DATA=search_df
    lemm_text, stem_text= get_lemmed_stemmed_text(query)
    if tokenizer==None:
        tokenizer= tokenizer_search   
    token_list = tokenizer.texts_to_sequences([lemm_text])[0]
    #print(f'token_list: {token_list}')
    token_list_stem = tokenizer.texts_to_sequences([stem_text])[0]
    #print(f'token_list stem: {token_list_stem}')
    
    mask1 = SEARCH_DATA.tokens.apply(lambda x: customIsIn(x, token_list))
    indexes1= SEARCH_DATA[mask1]['labels'].unique()
    mask2= SEARCH_DATA.tokens.apply(lambda x: customIsIn(x, token_list_stem))
    indexes2= SEARCH_DATA[mask2]['labels'].unique()
    indexes= np.concatenate((indexes1, indexes2), axis=None)
    results_df= SEARCH_DATA[SEARCH_DATA['labels'].isin(indexes)].drop(['tokens', 'labels'], axis=1)
    results_df= results_df.drop_duplicates()
    result=[]
    regex = re.compile(r'\([^)]*\)')
    for index, row in results_df.iterrows():
        text= row['text']
        dmc= row['DMC'] 
        dmc= re.sub(regex, '', dmc)
        result.append({'text': text, 'DMC':dmc})
    return result

def search_query_all(query, df=None, tokenizer=None, language="ru"):
    global SEARCH_DATA, search_df, index_data_loaded, PUBLICATION_LANGUAGE
    print('search_query_all!')
    print(f'query: {query}')
    old_publication_language= PUBLICATION_LANGUAGE
    PUBLICATION_LANGUAGE= language
    print('PUBLICATION_LANGUAGE:', PUBLICATION_LANGUAGE)
    SEARCH_DATA= df
    if df is None:
        if index_data_loaded==False or language!=old_publication_language:
            load_index_data()
        SEARCH_DATA=search_df
        print('SEARCH_DATA:', SEARCH_DATA.head())
        
    print('nlp loaded or not:', nlp)
    
    doc = nlp(clear_text(query))
    lemm_text, stem_text= get_lemmed_stemmed_text(query)
    if tokenizer==None:
        tokenizer= tokenizer_search
    token_list = tokenizer.texts_to_sequences([lemm_text])[0]
    print(f'token_list: {token_list}')
    token_list_stem = tokenizer.texts_to_sequences([stem_text])[0]
    print(f'token_list stem: {token_list_stem}')
    
    mask1= SEARCH_DATA['tokens'].map(set(token_list).issubset)
    mask2= SEARCH_DATA['tokens'].map(set(token_list_stem).issubset)
    indexes1= SEARCH_DATA[mask1]['labels'].unique()
    indexes2= SEARCH_DATA[mask2]['labels'].unique()
    indexes= np.concatenate((indexes1, indexes2), axis=None)
    results_df= SEARCH_DATA[SEARCH_DATA['labels'].isin(indexes)].drop(['tokens', 'labels'], axis=1)
    results_df= results_df.drop_duplicates()
    result=[]
    regex = re.compile(r'\([^)]*\)')
    for index, row in results_df.iterrows():
        text= row['text']
        dmc= row['DMC'] 
        dmc= re.sub(regex, '', dmc)
        result.append({'text': text, 'DMC':dmc})
    return result

def concat_by_DMC(s_df):
    #print('concat_by_DMC!')
    #print(s_df.head())
    #объединяем лемматизированную и стеммизированную часть датасета
    concat_tokens=[]
    for label in s_df['labels'].unique():
        tokens_lists= s_df[s_df['labels']==label]['tokens'].to_list()
        joined_lst=[]
        for lst in tokens_lists:
            joined_lst+= lst
        concat_tokens.append(joined_lst)
    #print(concat_tokens[:5])
    df= s_df.drop('tokens', axis=1)
    df= df.drop_duplicates()
    df['tokens']=concat_tokens

    #объединяем тексты и токены по DMC
    concat_tokens=[]
    DMCs=[]
    texts=[]
    for dmc_code in df['DMC'].unique():
        DMCs.append(dmc_code)
        #объединяем списки токенов для одного модуля данных (DMC)
        tokens_lists= df[df['DMC']==dmc_code]['tokens'].to_list()
        joined_token_lst=[]
        for lst in tokens_lists:
            joined_token_lst+= lst
        concat_tokens.append(joined_token_lst)
        #объединяем тексты
        text_list= df[df['DMC']==dmc_code]['text'].to_list()
        concat_text=' \n '.join(str(txt) for txt in text_list)
        texts.append(concat_text)
    #print('concat_tokens',len(concat_tokens))
    #print('DMCs',len(DMCs))
    #print('texts',len(texts))
    df= pd.DataFrame(columns=['DMC'], data=DMCs)  
    df['text']= texts
    df['tokens']= concat_tokens
    df['labels']= df.index
    #print(df.head())
    return df


def initialize_qa_model(model):
    global qa_df, qa_model, qa_model_num
    qa_model_num= model
    print('initialize_qa_model!')
    if model==1 or str(model)=="1":
        qa_model= pipeline("question-answering", "dmibor/ietm_search_and_qa", device=device)
        print('initialized model number 1!')
    else:#model==2 (базовая)
        qa_model= pipeline("question-answering", "timpal0l/mdeberta-v3-base-squad2", device=device)
        print('initialized model number 2!')
    #if qa_index_data_loaded==False:
    load_index_data_qa()
    #print('len(qa_df)', len(qa_df))
    qa_df= concat_by_DMC(qa_df)   
    #qa_df.to_csv('concat_index.csv', sep=';', index=False)    
    #print('concat_by_DMC len(qa_df)', len(qa_df))
    qa_model_initialized=True
 
def get_best_and_longest_result(model_results, threshold, mode):
    print('get_best_and_longest_result!')
    print('mode:', mode)
    best_result=None
    longest_result=None
    if(type(model_results)!= list):
        return best_result, longest_result
    best_result= model_results[0]
    best_result_answer= best_result['answer']
    print('best_result_answer: ',best_result_answer)
    best_answer_cleaned= (re.sub(r"[\W\d_]+$", "", best_result_answer)).strip()
    print('best_answer_cleaned: ',best_answer_cleaned)
    longest_answer=''
    longest_answer_len= len(best_answer_cleaned)
    longest_result= best_result
    print("type(mode)", type(mode))
    print("mode=='strict'", mode=='strict')
    print("mode==\"strict\"", mode=="strict")
    if mode=='strict':
        return best_result, longest_result
    if best_result['score']>=threshold:
        print('best_result_answer: ',best_answer_cleaned)
        print('best_result score:', best_result['score'])
        for result in model_results:
            answer= result['answer']
            answer_cleaned= re.sub(r"[\W\d_]+$", "", answer).strip()
            #print('answer_cleaned: ',answer_cleaned)
            if best_answer_cleaned in answer_cleaned:
                if len(answer_cleaned)>longest_answer_len:
                    print('new longest answer: ',answer_cleaned)
                    print('longest score:', result['score'])
                    print()
                    longest_answer= answer_cleaned
                    longest_answer_len= len(answer_cleaned)
                    longest_result= result
    #print('longest_answer:' , longest_answer)
    return best_result, longest_result
 
def find_answer(inputs, threshold, max_answer_len=1000, top_k=20, verbose=True, mode='strict'):
    print('find_answer!')
    print('mode:', mode)
    found_answer=False
    #print('qa_model', qa_model)
    model_results= qa_model([{"question": q["question"], "context": q["context"]} for q in inputs], batch_size=BATCH_SIZE, max_answer_len=max_answer_len, top_k=top_k)
    #print('model_results type:', type(model_results))
    if isinstance(model_results, dict):
        tmp= model_results
        model_results= list()
        model_results.append(tmp)
    #print('model_results:', model_results)
    # Добавляем индексы обратно в результаты
    best_score=0
    best_result=None
    longest_result=None
    for i, result in enumerate(model_results):#для каждого документа (модуля данных) свой список результатов
        dmc_value= inputs[i]["DMC"]
        #print('dmc_value:', dmc_value)
        if isinstance(result, dict):
            tmp= result
            result= list()
            result.append(tmp)
        for r in result:#это список результатов для одного модуля данных
            #print('r:', r)
            r["DMC"] = dmc_value
        #print(model_results)
        best_doc_result, longest_doc_result= get_best_and_longest_result(result, threshold, mode)
        if best_doc_result["score"]>best_score:
            best_score= best_doc_result["score"]
            best_result= best_doc_result
            longest_result= longest_doc_result
    #print('longest_result', longest_result)
    if best_result['score']>=threshold:
        longest_answer= longest_result['answer']
        answer_cleaned= re.sub(r"[\W\d_]+$", '', longest_answer).strip()
        if verbose==True:
            prob_value= round(model_result['score'], 2)
            print(f'Answer (score= {prob_value}): {answer_cleaned}')
        longest_result['answer']= answer_cleaned
        found_answer=True
    if found_answer==False and verbose==True:
        print('Answer not found!')
    model_result= best_result 
    model_result['answer']= longest_result['answer']
    return model_result
    
def answer_question(question, mode, model=1, language="ru"):
    global qa_model_initialized, qa_model_num, tokenizer_qa, PUBLICATION_LANGUAGE
    print('answer_question!')
    old_publication_language= PUBLICATION_LANGUAGE
    PUBLICATION_LANGUAGE= language
    print('PUBLICATION_LANGUAGE:', PUBLICATION_LANGUAGE)
    if qa_model_initialized==False or model!= qa_model_num or old_publication_language!= language:
        initialize_qa_model(model)
    print(f'question: {question}')
    print(f'mode: {mode}')
    print(f'model: {qa_model}')
    
    filtered_index= search_query_all(question, qa_df, tokenizer_qa)
    threshold= THRESHOLD
    #print('filtered_index все слова:', len(filtered_index))
    if len(filtered_index)<1:
        filtered_index= search_query_any(question, qa_df, tokenizer_qa)
        threshold= THRESHOLD
    #print('filtered_index:', filtered_index)

    inputs = [{"question": question, "context": indx["text"], "DMC": indx["DMC"]} for indx in filtered_index]
    #print('qa model inputs', inputs)
    top_k=1
    if mode!="strict":
        top_k=len(filtered_index)
    result= find_answer(inputs, threshold=threshold, max_answer_len=1000, top_k=top_k, verbose=False, mode=mode)

    if result!= None:
        best_answer= result['answer']
        best_score= result['score']
        best_DMC= result['DMC']
        regex = re.compile(r'\([^)]*\)')
        best_DMC= re.sub(regex, '', best_DMC)
        result= [{'score': best_score, 'answer': best_answer, 'DMC': best_DMC}]
    return result
    
