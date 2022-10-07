# Front end app to interact with api.
# To use this, initialize the api first.
# Then type in cmd: streamlit run <name_of_this_file>
#
# Exemple:
# streamlit run scripts/app.py


import streamlit as st
import requests



def main():
    url = 'http://127.0.0.1:8000/predict_tweet/'
    #url = 'http://20100oc.eu.pythonanywhere.com/'
    
    st.title('Sentiment analysis')
    tweet = st.text_input(label='Insert tweet here', max_chars=280)

    predict_btn = st.button('Prédire')
    if predict_btn:
        keys = {'tweet': tweet}
        raw_res = requests.get(url, params=keys)
        #raw_res = requests.post(url, params=keys)
        res = raw_res.json()

        st.write('Tweet:', res['tweet'])
        st.write('Sentiment:', res['sentiment'])
        st.write('Probability:', res['probability'])
        



if __name__ == '__main__':
    main()
