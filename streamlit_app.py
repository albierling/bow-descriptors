import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Title and description
st.title("BOW - A Standardised Lexicon of Body Odour Words")
st.write(
    "Let's start building! For help and inspiration, head over to [docs.streamlit.io](https://docs.streamlit.io/)."
)

# Load the dataset
file_path = 'BOW-data.xlsx'  # Ensure this path is correct relative to app.py
df = pd.read_excel(file_path)

# Fill missing values in 'translation_lemmatized' with a placeholder
df['translation_lemmatized'] = df['translation_lemmatized'].fillna('Missing')

# Filtering options
languages_with_multiple_countries = ['English', 'German', 'Spanish']
unique_languages = df['language'].unique()

language = st.selectbox("Select Language", unique_languages)

if language in languages_with_multiple_countries:
    if language == 'English':
        available_countries = df[df['language'] == 'English']['country'].unique()
    elif language == 'German':
        available_countries = df[df['language'] == 'German']['country'].unique()
    elif language == 'Spanish':
        available_countries = df[df['language'] == 'Spanish']['country'].unique()
    available_countries = list(available_countries)
    available_countries.insert(0, "All")
    country = st.multiselect("Select Country", available_countries, default="All")
else:
    country = df[df['language'] == language]['country'].unique()
    st.write(f"Country: {country[0]}")

conditions = st.multiselect("Select Conditions", df['condition'].unique())

# Filter dataset based on selections
if language in languages_with_multiple_countries:
    if "All" in country:
        filtered_df = df[df['language'] == language]
    else:
        filtered_df = df[(df['country'].isin(country)) & (df['language'] == language)]
else:
    filtered_df = df[(df['country'] == country[0]) & (df['language'] == language)]

if conditions:
    filtered_df = filtered_df[filtered_df['condition'].isin(conditions)]

# Calculate frequencies using the lemma column and include lemmatized translation
frequency_df = filtered_df.groupby(['lemma', 'translation_lemmatized']).size().reset_index(name='Frequency')

# Calculate the percentage and cumulative percentage
total_entries = frequency_df['Frequency'].sum()
frequency_df['Percentage'] = (frequency_df['Frequency'] / total_entries) * 100
frequency_df = frequency_df.sort_values(by='Frequency', ascending=False)
frequency_df['Cumulative Percentage'] = frequency_df['Percentage'].cumsum()

# Round the percentage and cumulative percentage to two decimal places
frequency_df['Percentage'] = frequency_df['Percentage'].round(2)
frequency_df['Cumulative Percentage'] = frequency_df['Cumulative Percentage'].round(2)

# Add Rank column
frequency_df['Rank'] = range(1, len(frequency_df) + 1)

# Reorder columns to have Rank first
frequency_df = frequency_df[['Rank', 'lemma', 'translation_lemmatized', 'Frequency', 'Percentage', 'Cumulative Percentage']]

# Style the "Missing" values
def style_missing(val):
    if val == 'Missing':
        return 'color: grey; font-style: italic;'
    return ''

styled_frequency_df = frequency_df.style.applymap(style_missing, subset=['translation_lemmatized'])

# Display filtered dataset and frequencies without index
st.write("Word Frequency:")
st.dataframe(frequency_df.style.applymap(style_missing, subset=['translation_lemmatized']), height=400)

# Plotting age distribution
st.write("Age Distribution")
fig, ax = plt.subplots(figsize=(10, 4))
ax.hist(filtered_df['age'], bins=20, color='skyblue', edgecolor='black')
ax.set_title('Age Distribution')
ax.set_xlabel('Age')
ax.set_ylabel('Frequency')
st.pyplot(fig)
