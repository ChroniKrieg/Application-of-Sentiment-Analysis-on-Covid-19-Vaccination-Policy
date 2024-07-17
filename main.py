import streamlit as st
import pandas as pd
import numpy as np
from streamlit_option_menu import option_menu
import pickle
from PIL import Image

# Navigation sidebar
with st.sidebar:
    selected = option_menu('MENU',
                           ['Sentiment Analysis', 'About The Data'],
                           default_index=0)

# Page: Sentiment Analysis
if selected == 'Sentiment Analysis':
    st.title('Aplikasi Sentimen Analisis Kebijakan Vaksinasi Covid-19 dengan Model Regresi Logistik')
    st.write('This is a sentiment analysis app for Covid-19 Vaccination Policy on Twitter')

    # Load dataset
    df = pd.read_csv('G:\\TA\\TA\\dataset\\Covid-19 Vaccine Tweets\\Covid-19 Vaccine Tweets with Sentiment Annotation\\cleaned_data_tweet_only.csv',
                     encoding='unicode_escape')

    # Assuming 'label' column has 1 for positive and 0 for negative sentiments
    positif = df[df['label'] == 1]
    negatif = df[df['label'] == 0]
    # print(positif)
    # Display a random positive tweet
    if len(positif) > 0:
        idx_positif = np.random.randint(0, len(positif))
        print(idx_positif)
        contoh_positif = pd.DataFrame(positif[['cleaned_tweet']].iloc[idx_positif])
        contoh_positif.columns = ['Example for positive tweet']
        st.write(contoh_positif)

    # Display a random negative tweet
    if len(negatif) > 0:
        idx_negatif = np.random.randint(0, len(negatif))
        print(idx_negatif)
        contoh_negatif = pd.DataFrame(negatif[['cleaned_tweet']].iloc[idx_negatif])
        contoh_negatif.columns = ['Example for negative tweet']
        st.write(contoh_negatif)

    # User input for tweet
    kalimat = st.text_area("Input your tweet:")

    # Model selection with other models indicated as unavailable
    option_model = st.selectbox(
        'the model used:',
        ['Logistic Regression'],
        index=0
    )
    # Load vectorizer and model
    vector = pickle.load(open('G:\\TA\\TA\\streamlit\\tfidf_vectorizer.sav', 'rb'))
    if option_model == 'Logistic Regression' :
        model = pickle.load(open('G:\\TA\\TA\\streamlit\\logregr.sav', 'rb'))

    # Transform user input using the vectorizer
    kalimat_transformed = vector.transform([kalimat])


    # Predict sentiment
    prediksi = model.predict(kalimat_transformed)
    prediksi_proba = model.predict_proba(kalimat_transformed)

    # Display prediction results
    if st.button("Analyze Tweet"):
        if kalimat:
            sentiment = 'positive' if prediksi == 1 else 'negative'
            probability = prediksi_proba[0][prediksi][0] * 100
            message = f"The sentiment of your tweet is {sentiment} with probability {probability:.2f}%"
            if prediksi == 1:
                st.success(message)
            else:
                st.error(message)
        else:
            st.write("Please input your tweet")

# Page: About The Data
if selected == 'About The Data':
    st.title('About The Data')
    # Load dataset
    df = pd.read_csv(
        'G:\\TA\\TA\\dataset\\Covid-19 Vaccine Tweets\\Covid-19 Vaccine Tweets with Sentiment Annotation\\cleaned_data_tweet_only.csv',
        encoding='unicode_escape')

    # data review
    st.write('Sentiment Analysis of Covid-19 Vaccination Policy by the Public on Twitter')
    st.write(df)

    # Display a bar chart of tweet labels
    label_counts = df['label'].value_counts().rename({0: 'Negative', 1: 'Positive'})
    st.bar_chart(label_counts)

    # wordcloud
    st.write('Wordcloud from the all tweets: ')
    wordcloud_review = Image.open('G:\TA\TA\dataset\Covid-19 Vaccine Tweets\Covid-19 Vaccine Tweets with Sentiment Annotation\worldcloud.png')
    st.image(wordcloud_review)
