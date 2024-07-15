import streamlit as st
import pandas as pd

# Title and description
st.title("BOW - A Standardised Lexicon of Body Odour Words")
st.write(
    "Let's start building! For help and inspiration, head over to [docs.streamlit.io](https://docs.streamlit.io/)."
)

# Load the dataset
file_path = 'BOW-data.xlsx'
df = pd.read_excel(file_path)

# Display the dataset
st.write("Dataset Preview:")
st.dataframe(df.head())

# Filtering options
country = st.selectbox("Select Country", df['country'].unique())
language = st.selectbox("Select Language", df['language'].unique())
conditions = st.multiselect("Select Conditions", df['condition'].unique())

# Filter dataset based on selections
filtered_df = df[(df['country'] == country) & (df['language'] == language)]
if conditions:
    filtered_df = filtered_df[filtered_df['condition'].isin(conditions)]

# Calculate frequencies
frequency_df = filtered_df['full description'].value_counts().reset_index()
frequency_df.columns = ['Description', 'Frequency']
frequency_df['Percentage'] = (frequency_df['Frequency'] / frequency_df['Frequency'].sum()) * 100

# Sort by frequency
frequency_df = frequency_df.sort_values(by='Frequency', ascending=False)

# Display filtered dataset and frequencies
st.write("Filtered Dataset:")
st.dataframe(filtered_df)

st.write("Word Frequency:")
st.dataframe(frequency_df)
