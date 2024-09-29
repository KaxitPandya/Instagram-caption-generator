import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
import time
import base64
# import os
# from dotenv import load_dotenv, find_dotenv

# # Load environment variables
# load_dotenv(find_dotenv(), override=True)

# Custom CSS
st.markdown("""
    <style>
    .title {
        font-size: 2.5em;
        color: black;
        text-align: center;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .stButton button {
        background-color: #FF6347;
        color: white;
        font-size: 1.2em;
        border-radius: 5px;
    }
    .stButton button:hover {
        background-color: #FF4500;
        color: white;
    }
    .spinner {
        text-align: center;
        font-size: 1.5em;
        color: #4B0082;
        margin-top: 20px;
    }
""", unsafe_allow_html=True)

# Function to select the appropriate model based on user's choice
def llm_invoke(option, api_key=api_key):
    if option == "Explain Scenario using text to get the caption":
        llm = ChatGoogleGenerativeAI(model='gemini-1.5-pro', temperature=0, max_output_tokens=None, api_key=api_key)
    else:
        llm = ChatGoogleGenerativeAI(model='gemini-1.5-flash', api_key=api_key)
    return llm

# Function to create prompt for the selected model
def generate_message(llm, totalCaptions, language, captionTone, captionLength):
    if llm.dict()['model'].split('/')[1] == 'gemini-1.5-pro':
        scene = st.text_input("Explain the scenario for which you want the caption:")
        if scene:
            template = PromptTemplate(
                input_variables=["language", "totalCaptions", "scenario", "captionTone", "captionLength"],
                template="I want {totalCaptions} alternative caption(s) for the following scenario: \"{scenario}\" in {language} language, and the tone should be {captionTone} and the caption length should be Instagram {captionLength} size"
            )
            formatted_prompt = template.format(
                totalCaptions=captions,
                scenario=scene,
                language=lang,
                captionTone=tone,
                captionLength=length
            )
            return formatted_prompt
        else:
            st.warning('🚨Please explain the scenario first...')
    else:
        uploaded_file = st.file_uploader("Upload an image:", type=['jpg', 'jpeg', 'png'])
        if uploaded_file:
            img_data = uploaded_file.read()
            image_data = base64.b64encode(img_data).decode('utf-8')
            text = f'You are a helpful assistant that helps people generate their Instagram story and post captions. I want {totalCaptions} alternative caption(s) for the following image in {language} language, and the tone should be {captionTone} and the caption length should be Instagram {captionLength} size'
            prompt = HumanMessage(content=[{'type': 'text', 'text': text}, {'type': 'image_url', 'image_url': {'url': f"data:image/jpeg;base64,{image_data}"} }])
            return prompt


st.markdown('<div class="title">Instagram Caption Generator</div>', unsafe_allow_html=True)

# add your Google API key in the secrets.toml file as:
# GOOGLE_API_KEY = 'your_api_key'
# this key will be used when deploying your Streamlit app.
api_key = st.secrets["AIzaSyAAzlvmMFqDSmQa-sKfhbE5XRnHk7YAO1Q"]
lang = st.selectbox("Select Language:", ['english(default)', 'French'], index=None)
option = st.radio("Choose input method: ", ['Explain Scenario using text to get the caption', 'Upload image to get the caption'], index=None)


if lang and option:
    llm = llm_invoke(option)
    captions = st.selectbox('Select Number of Alternatives you want: ', [1, 2, 3, 4, 5], index=None)
    tone = st.selectbox("Select Tone:", ['Creative', 'Humorous', 'Funny', 'Humorous and funny', 'Conversational', 'genZ language'], index=None)
    length = st.selectbox('Select caption Type:', ['Story caption (short)', 'Post caption (Short)', 'Story caption (long)', 'Post caption (long)'], index=None)
    prompt = generate_message(llm, captions, lang, tone, length)
    
    placeholder = st.empty()

    # enable the button only if all conditions are met
    generate_button = st.button("Generate Caption", disabled=not (prompt and captions and tone and length))
    
    if generate_button:
        content = ""
        # stream the model's output content character by character for either scenario (text or image input).
        with st.spinner("Generating captions..."):
            if llm.dict()['model'].split('/')[1] == 'gemini-1.5-pro':
                for chunk in llm.stream(prompt):
                    for char in chunk.content:
                        content += char
                        placeholder.markdown(content)
                        time.sleep(0.005)
            else:
                for chunk in llm.stream([prompt]):
                    for char in chunk.content:
                        content += char
                        placeholder.markdown(content)
                        time.sleep(0.005)

        # store the generated content in session state
        st.session_state['generated_content'] = content

    # display the clear button if content exists
    flag = 'generated_content' in st.session_state
    clear_button = st.button("Clear Output", disabled= not flag)

    if clear_button:
        # clear the generated content and remove it from the session state
        st.session_state.pop('generated_content', None)
        placeholder.empty()
        st.query_params.clear()
        st.rerun()

