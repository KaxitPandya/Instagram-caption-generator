import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
import time
import base64
import os
from dotenv import load_dotenv
from PIL import Image
import io

# To run the app locally, store the GOOGLE_API_KEY in your .env file, and comment out the following line:
# api_key = st.secrets["GOOGLE_API_KEY"]
# Additionally, uncomment the following lines to load the API key from the .env file:
# import os
# from dotenv import load_dotenv, find_dotenv
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
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #f1f1f1;
        color: black;
        text-align: center;
        padding: 10px 0;
        font-size: 0.9em;
        border-top: 1px solid #ddd;
    }

""", unsafe_allow_html=True)

st.markdown('<div class="title">Instagram Caption Generator</div>', unsafe_allow_html=True)

api_key = st.secrets["GOOGLE_API_KEY"]

lang = st.selectbox("Select Language:", ['English(Default)', 'French'], index=None)
option = st.radio("Choose input method: ", ['Explain Scenario using text to get the caption', 'Upload image to get the caption'], index=None)


# select the appropriate model based on the user's choice in the 'option' input.
# Note: Google has updated model names. Using 'gemini-pro' which is widely available.
def llm_invoke(option, api_key=api_key):
    # Use gemini-pro for both text and image scenarios
    # gemini-pro supports both text generation and vision capabilities
    if option == "Explain Scenario using text to get the caption":
        llm = ChatGoogleGenerativeAI(model='gemini-pro', temperature=0, max_output_tokens=None, api_key=api_key)
    else:
        # For image uploads, gemini-pro also supports vision
        llm = ChatGoogleGenerativeAI(model='gemini-pro', api_key=api_key)
    return llm
    return llm

def generate_message(option, llm, totalCaptions, language, captionTone, captionLength):
    # Determine input type based on the selected option
    if option == "Explain Scenario using text to get the caption":
        # Text input mode
        scene = st.text_input("Explain the scenario for which you want the caption: ")
        if scene:
            template = PromptTemplate(
                input_variables=["language", "totalCaptions", "scenario", "captionTone", "captionLength"],
                template="I want {totalCaptions} alternative caption(s) for the following scenario: \"{scenario}\" in {language} language, and the tone should be {captionTone} and the caption length should be Instagram {captionLength} size"
                )
            formatted_prompt = template.format(
                totalCaptions=totalCaptions,
                scenario=scene,
                language=language,
                captionTone=captionTone,
                captionLength=captionLength
            )
            # Use proper LangChain message objects
            prompt = [
                SystemMessage(content='You are a helpful assistant that helps people generate their instagram story and post captions'),
                HumanMessage(content=formatted_prompt)
            ]
            return prompt
        else:
            st.warning('🚨Please explain the scenario first...')
            return None
    else:
        # Image upload mode
        uploaded_file = st.file_uploader("Upload an image:", type=['jpg', 'jpeg', 'png'])
        if uploaded_file is not None:
            # Read image data and ensure it's in RGB mode (required by Gemini)
            img_data = uploaded_file.read()
            image = Image.open(io.BytesIO(img_data))
            
            # Convert to RGB if necessary (some images might be RGBA or other modes)
            # This is important for Gemini API compatibility
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            text = f'You are a helpful assistant that helps people generate their instagram story and post captions. I want {totalCaptions} alternative caption(s) for the following image in {language} language, and the tone should be {captionTone} and the caption length should be Instagram {captionLength} size'
            
            # For ChatGoogleGenerativeAI with images, create HumanMessage
            # Include system instruction in the text, as SystemMessage might cause conversion issues
            full_text = f'You are a helpful assistant that helps people generate their instagram story and post captions. {text}'
            
            # Use model_construct to bypass Pydantic validation
            # ChatGoogleGenerativeAI will handle the PIL Image conversion internally
            try:
                # Create HumanMessage with text (including system instruction) and image
                # Don't use SystemMessage separately as it might cause conversion issues
                human_msg = HumanMessage.model_construct(content=[full_text, image])
            except Exception as e:
                st.error(f"Error creating HumanMessage: {str(e)}")
                return None
            
            # Return just the HumanMessage (without SystemMessage) for image inputs
            # This might help avoid the conversion error
            prompt = [human_msg]
            return prompt
        else:
            st.warning('🚨Please upload an image.....')
            return None

if lang and option:
    llm = llm_invoke(option)
    captions = st.selectbox('Select Number of Alternatives you want: ', [1, 2, 3, 4, 5], index=None)
    tone = st.selectbox("Select Tone:", ['Creative', 'Humorous', 'Funny', 'Humorous and funny', 'Conversational', 'genZ language'], index=None)
    length = st.selectbox('Select caption Type:', ['Story caption (short)', 'Post caption (Short)', 'Story caption (long)', 'Post caption (long)'], index=None)
    prompt = generate_message(option, llm, captions, lang, tone, length)

    placeholder = st.empty()
    generate_button = st.button("Generate Caption", disabled=not (prompt and captions and tone and length))

    if generate_button:
        content = ""
        with st.spinner("Generating captions..."):
            # Stream the response - gemini-pro works the same way for both text and images
            # Both use the same message format now
            try:
                # Try streaming with the prompt directly (works for both text and image prompts)
                for chunk in llm.stream(prompt):
                    if hasattr(chunk, 'content') and chunk.content:
                        for char in chunk.content:
                            content += char
                            placeholder.markdown(content)
                            time.sleep(0.005)
            except Exception as e:
                st.error(f"Error generating caption: {str(e)}")
                st.info("Please check your API key and model availability.")
        st.session_state['generated_content'] = content

    flag = 'generated_content' in st.session_state
    clear_button = st.button("Clear Output", disabled= not flag)

    if clear_button:
        st.session_state.pop('generated_content', None)
        placeholder.empty()
        st.query_params.clear()
        st.rerun()



