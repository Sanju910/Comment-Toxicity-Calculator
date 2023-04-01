import streamlit as st
# UTILITY
import joblib
import pickle
from joblib import load
# NLP
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk import SnowballStemmer
from gensim.models.doc2vec import Doc2Vec
import warnings

warnings.filterwarnings('ignore')
comment = ""
tresh1 = 0.500
tresh2 = 0.937
tresh3 = 0.999


# set page setting
st.set_page_config(page_title='Toxic Comments')

with open('styles.css') as f:
    st.markdown(f'<style>{f.read()}</style>',unsafe_allow_html=True)

# set history var
if 'history' not in st.session_state:
    st.session_state.history = []

# import similarity (to be cached)
@st.cache_data(persist=True)
def importModel(filename):
    model = load(filename)
    return model

@st.cache_data(persist=True) #allow_output_mutation=True)
def importD2V(filename):
    model = Doc2Vec.load(filename)
    return model

@st.cache_data(persist=True)
def loadPickle(filename):
    file = pickle.load(open(filename, 'rb'))
    return file

@st.cache_data(persist=True)
def loadPunkt():
    nltk.download('punkt')

with st.spinner('Loading the models, this could take some time...'):
    loadPunkt()
    normalizer = importModel("normalizerD2V.joblib")
    classifier = importModel("toxicCommModel.joblib")
    model_d2v = importD2V("d2v_comments.model")


# REGEX
def apply_regex(corpus):
    corpus = re.sub("\S*\d\S*"," ", corpus) 
    corpus = re.sub("\S*@\S*\s?"," ", corpus)
    corpus = re.sub("\S*#\S*\s?"," ", corpus)
    corpus = re.sub(r'http\S+', ' ', corpus)
    corpus = re.sub(r'[^a-zA-Z0-9 ]', ' ',corpus)
    corpus = corpus.replace(u'\ufffd', '8')
    corpus = re.sub(' +', ' ', corpus)
    return corpus

# TOKENIZE TEXT - we use the Spacy library stopwords
stop_words = loadPickle("stop_words.pkl")

# TOKENIZE TEXT and STOP WORDS REMOVAL - execution (removes also the words shorter than 2 and longer than 15 chars)
def tokenize(doc):
    tokens_1 = word_tokenize(str(doc))
    return [word.lower() for word in tokens_1 if len(word) > 1 and len(word) < 15 and word not in stop_words and not word.isdigit()]

# STEMMING
stemmer = SnowballStemmer(language="english")
def applyStemming(listOfTokens):
    return [stemmer.stem(token) for token in listOfTokens]

# PROBS TO CLASS
def probs_to_prediction(probs, threshold):
    pred=[]
    for x in probs[:,1]:
        if x>=threshold:
            pred = 1
        else:
            pred = 0
    return pred


# PROCESSING
def compute(comment, tresh):
    global preds
    global probs
    global stems
    stems = ""
    preds = []
    comment = apply_regex(comment)
    comment = tokenize(comment)
    comment = applyStemming(comment)
    stems = comment

    vectorizedComment =  model_d2v.infer_vector(comment, epochs=70)
    normComment = normalizer.transform([vectorizedComment])
    probs = classifier.predict_proba(normComment)
    val = probs[0][1]
    message = ""
    if 0<=val<0.2:
        message += "This comment is friendly, respectful, and free of harmful language or intent."
    elif 0.2<=val<0.4:
        message += "This comment has a slightly negative tone, and may contain potentially harmful language or intent."
    elif 0.4<=val<0.7:
        message += "This comment contains a moderate amount of harmful language or intent, and may be perceived as offensive or hurtful by some readers."
    elif 0.7<=val<0.9:
        message += "This comment contains significant harmful language or intent that is upsetting or offensive to many readers."
    else:
        message += "This comment contains highly harmful language or intent that is deeply offensive or disturbing to readers, and may even incite hatred or violence."

    st.write(f"{message}")

    st.markdown("""---""") 
    # display()
    return None

def display():
    with st.container():
        st.write("#### Different classification outputs at different threshold values:")
        col3, col4, col5 = st.columns(3)
        col3.metric("", "TOXIC" if preds[0]==1 else "NON TOXIC", delta = 0.500)
        col4.metric("", "TOXIC" if preds[1]==1 else "NON TOXIC", delta = 0.937)
        col5.metric("", "TOXIC" if preds[2]==1 else "NON TOXIC", delta = 0.999)
    st.markdown("""---""")
    with st.container():
        st.write("#### Result of the NLP Pipeline:")
        st.write(stems)
    return None

# TITLE
st.write("# ☢️ Toxic Comments Classification")

st.write("\n")
with st.form("my_form",clear_on_submit=True):
    comment = st.text_input(label='Enter your comment:')
    thresh = st.slider('Set the Threshold, default 0.5', 0.000, 1.000, step=0.0001, value=0.500)

    submitted = st.form_submit_button("Submit")
    if submitted:
        compute(comment,thresh)

# # INPUT TEXTBOX
# comment = st.text_area('', "Drop your comment here! 😎")
    
# # IMPUT THRESHOLD
# thresh = st.slider('Set the Threshold, default 0.5', 0.000, 1.000, step=0.0001, value=0.500)


# sidebar
st.sidebar.write("""
Our real-time Toxic Comment Classifier combines tokenization, stemming, and Doc2Vec encoding to identify harmful comments. A customised logistic regression model predicts toxicity using a vast corpus of comments. We utilise a threshold value to group comments into toxicity levels to simplify the interpretation of predicted probability and make it easier to discover and manage dangerous content. Our model moderates online conversations quickly and accurately to keep users safe and respectful.
""")
