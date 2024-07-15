import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pydeck as pdk

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

# Display map with selected countries
st.write("Map of Selected Countries")

# Mapping country codes to coordinates (latitude and longitude)
country_coords = {
    'Germany': [51.1657, 10.4515],
    'Great-Britain': [55.3781, -3.4360],
    'Chile': [-35.6751, -71.5430],
    'Colombia': [4.5709, -74.2973],
    'Italy': [41.8719, 12.5674],
    'Poland': [51.9194, 19.1451],
    'Czech Republic': [49.8175, 15.4730],
    'Norway': [60.4720, 8.4689],
    'Finland': [61.9241, 25.7482],
    'Turkey': [38.9637, 35.2433],
    'Israel': [31.0461, 34.8516],
    'Hong Kong': [22.3193, 114.1694],
    'Vanuatu': [-15.3767, 166.9592],
    'India': [20.5937, 78.9629]
}

# Get coordinates for the selected countries
selected_coords = []
if language in languages_with_multiple_countries:
    if "All" not in country:
        selected_coords = [country_coords[c] for c in country if c in country_coords]
else:
    selected_coords = [country_coords[country[0]]]

# Create a DataFrame for the selected coordinates
map_df = pd.DataFrame(selected_coords, columns=['lat', 'lon'])

# Create a pydeck map
layer = pdk.Layer(
    "ScatterplotLayer",
    map_df,
    get_position="[lon, lat]",
    get_radius=500000,
    get_color=[255, 0, 0],
    pickable=True
)

view_state = pdk.ViewState(
    latitude=20.0,
    longitude=0.0,
    zoom=1
)

map = pdk.Deck(
    layers=[layer],
    initial_view_state=view_state,
    tooltip={"text": "{lat}, {lon}"}
)

st.pydeck_chart(map)
