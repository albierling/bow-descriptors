import streamlit as st
import pandas as pd

# Title and description
st.title("BOW - A Standardised Lexicon of Body Odour Words")
st.write(
    "Let's start building! For help and inspiration, head over to [docs.streamlit.io](https://docs.streamlit.io/)."
)

# Load the dataset
file_path = 'BOW-data.xlsx'  # Ensure this path is correct relative to app.py
df = pd.read_excel(file_path)

# Filtering options
languages_with_multiple_countries = ['English', 'German', 'Spanish']
language = st.selectbox("Select Language", languages_with_multiple_countries)

if language == 'English':
    countries = ['USA', 'UK', 'AUS']
elif language == 'German':
    countries = ['GER', 'AUT', 'SUI']
elif language == 'Spanish':
    countries = ['ESP', 'MEX', 'ARG']
else:
    countries = df[df['language'] == language]['country'].unique()

country = st.selectbox("Select Country", countries)
conditions = st.multiselect("Select Conditions", df['condition'].unique())

# Filter dataset based on selections
filtered_df = df[(df['country'] == country) & (df['language'] == language)]
if conditions:
    filtered_df = filtered_df[filtered_df['condition'].isin(conditions)]

# Calculate frequencies using the lemma column
frequency_df = filtered_df['lemma'].value_counts().reset_index()
frequency_df.columns = ['Description', 'Frequency']
frequency_df['Percentage'] = (frequency_df['Frequency'] / frequency_df['Frequency'].sum()) * 100

# Sort by frequency
frequency_df = frequency_df.sort_values(by='Frequency', ascending=False)

# Display filtered dataset and frequencies
st.write("Word Frequency:")
st.dataframe(frequency_df)

# Plotting age distribution
st.write("Age Distribution")
plt.figure(figsize=(10, 4))
plt.hist(filtered_df['age'], bins=20, color='skyblue', edgecolor='black')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
st.pyplot(plt)
