import streamlit as st
import pandas as pd
import numpy as np
import os
from os import path

import plotly.express as px
import pydeck as pdk

from wordcloud import WordCloud
from PIL import Image
from matplotlib import font_manager


def select_font(language, display_fontfile=False):
    # rather hacky way to select the right font
    noto = 'NotoSans-Regular'
    if language == 'Chinese':
        noto = 'NotoSansCJK-Regular'
    elif language == 'Hebrew':
        noto = 'NotoSansHebrew-Regular'
    elif language == 'Hindi':
        noto = 'NotoSansDevanagari-Regular'
    
    flist = font_manager.findSystemFonts(fontpaths=None, fontext='ttf')
    fn_noto = ''
    for fn in flist:
        if noto in fn:
            fn_noto = fn
            break
    
    ## select font for word cloud
    try:
        font_file = font_manager.findfont('Arial Unicode MS', fallback_to_default=False)
    except:
    #    font_search = font_manager.FontProperties(family='sans-serif', weight='normal')
        font_search = font_manager.FontProperties(fname=fn_noto)
        font_file = font_manager.findfont(font_search)

    if display_fontfile:
        st.write('Font: ' + font_file)
    
    return font_file

# Logo and title
col1, col2 = st.columns([0.3, 0.7])
with col1:
    st.text('')
    st.image('blue-Logos-smellodi-01.png', width=170)
with col2:
    st.markdown('## A Standardised Lexicon of Body Odour Words')

st.write(
    ""
)

# Load the dataset
#file_path = 'BOW-data.xlsx' 
#file_path = 'BOW-df.xlsx'
file_path = 'BOW_df_v2.xlsx'
df = pd.read_excel(file_path)

# Remove white spaces at the end of entries in the 'lemma' column
df['lemma'] = df['lemma'].str.strip()

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
st.dataframe(frequency_df.style.applymap(style_missing, subset=['translation_lemmatized']), height=400,hide_index=True)

# Wordcloud
st.subheader("Word cloud")
    
## get number of words for wordcloud
col1, col2 = st.columns([0.3, 0.7])
with col1:
    wc_translated = st.toggle("Use translation")

if wc_translated:
    max_no = frequency_df['translation_lemmatized'].count()
    font_file = select_font('English')
else:
    max_no = frequency_df['lemma'].count()
    font_file = select_font(language)

with col2:
    wc_no = st.slider("No of words", 1, max_no, 24)

## Generate a word cloud image
d = path.dirname(__file__) if "__file__" in locals() else os.getcwd()
mask = np.array(Image.open(path.join(d, 'mask.png')))

if wc_translated:
    word_list = list(frequency_df['translation_lemmatized'][0:wc_no-1].values)
else:
    word_list = list(frequency_df['lemma'][0:wc_no-1].values)

freq_list = list(frequency_df['Frequency'][0:wc_no-1].values)
freq_dict = dict(zip(word_list, freq_list))

wordcloud = WordCloud(font_path=font_file, mask=mask, contour_width=3, contour_color='steelblue', background_color='white').generate_from_frequencies(freq_dict) #.generate(text)

st.image(wordcloud.to_array(), use_column_width='always', caption='word cloud')

# New subheading
st.subheader("Country Details and Visualizations")

# Display map with selected countries
st.write("Map of All Countries")

# Mapping country codes to coordinates (latitude and longitude)
country_coords = {
    'Germany': [51.1657, 10.4515],
    'United Kingdom': [55.3781, -3.4360],
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
    'India': [20.5937, 78.9629],
    'Austria': [47.5162, 14.5501],
    'United Kingdom': [55.3781, -3.4360],
    'Canada': [56.1304, -106.3468],
    'Sweden': [60.1282, 18.6435]
}

# Number of participants by country
participants = {
    'Germany': 333,
    'Austria': 189,
    'Hong Kong': 374,
    'Israel': 254,
    'United Kingdom': 60,
    'Canada': 71,
    'Sweden': 117,
    'Chile': 128,
    'Colombia': 57,
    'Finland': 174,
    'Norway': 157,
    'Czech Republic': 143,
    'India': 122,
    'Turkey': 117,
    'Italy': 116,
    'Poland': 116,
    'Vanuatu': 100
}

# Create a DataFrame for the selected coordinates
map_df = pd.DataFrame.from_dict(country_coords, orient='index', columns=['lat', 'lon']).reset_index()
map_df.columns = ['country', 'lat', 'lon']

# Add number of participants to the DataFrame
map_df['participants'] = map_df['country'].map(participants)

# Create a pydeck map
layer = pdk.Layer(
    "ScatterplotLayer",
    map_df,
    get_position="[lon, lat]",
    get_radius=200000,
    get_color=[0, 128, 255],
    pickable=True,
    auto_highlight=True
)

view_state = pdk.ViewState(
    latitude=20.0,
    longitude=0.0,
    zoom=1
)

map = pdk.Deck(
    layers=[layer],
    initial_view_state=view_state,
    tooltip={"text": "{country}\nParticipants: {participants}"}
)

st.pydeck_chart(map)

# Show details for the selected country
selected_country = st.selectbox("Select a country to view details", map_df['country'].unique())

if selected_country:
    st.write(f"Number of Participants: {participants[selected_country]}")
    
    # Filter data for the selected country
    country_df = df[df['country'] == selected_country].drop_duplicates(subset=['no'])
    
    # Show gender distribution
    gender_counts = country_df['gender'].value_counts()
    gender_pie = px.pie(
        gender_counts,
        values=gender_counts.values,
        names=gender_counts.index,
        title=f'Gender Distribution in {selected_country}',
        color_discrete_sequence=px.colors.sequential.Blues
    )
    st.plotly_chart(gender_pie)
    
    # Show age distribution
    age_hist = px.histogram(
        country_df,
        x='age',
        nbins=20,
        title=f'Age Distribution in {selected_country}',
        labels={'age': 'Age'},
        color_discrete_sequence=['skyblue']
    )
    st.plotly_chart(age_hist)
