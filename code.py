import json
import os
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import streamlit as st


with open(os.path.join(os.path.dirname(__file__), 'faqs.json'), 'r') as f:
    data = json.load(f)

q = [faq['question'] for faq in data['faqs']]
a = [faq['answer'] for faq in data['faqs']]

lem = WordNetLemmatizer()
stop = set(stopwords.words('english'))

def clean(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [lem.lemmatize(w) for w in words if w not in stop]
    return ' '.join(words)

q_clean = [clean(i) for i in q]

vec = TfidfVectorizer()
mat = vec.fit_transform(q_clean)


def get_ans(user_q):
    u = clean(user_q)
    u_vec = vec.transform([u])
    scores = cosine_similarity(u_vec, mat)
    best = scores.argmax()
    if scores[0][best] < 0.1:
        return "sorry i don't have an answer for that"
    return a[best]

st.title("PC Building Helper Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages=[]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user_input = st.chat_input("Ask me anything about building a PC")

if user_input:
    with st.chat_message("user"):
        st.write(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    res = get_ans(user_input)

    with st.chat_message("assistant"):
        st.write(res)
    st.session_state.messages.append({"role": "assistant", "content": res})