# import streamlit as st
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_core.prompts import PromptTemplate
# from langchain_core.messages import HumanMessage, SystemMessage
# import time
# import base64
# import os
# from dotenv import load_dotenv
# from PIL import Image
# import io

# # To run the app locally, store the GOOGLE_API_KEY in your .env file, and comment out the following line:
# # api_key = st.secrets["GOOGLE_API_KEY"]
# # Additionally, uncomment the following lines to load the API key from the .env file:
# # import os
# # from dotenv import load_dotenv, find_dotenv
# # load_dotenv(find_dotenv(), override=True)


# # Custom CSS
# st.markdown("""
#     <style>
#     .title {
#         font-size: 2.5em;
#         color: black;
#         text-align: center;
#         font-weight: bold;
#         margin-bottom: 10px;
#     }
#     .stButton button {
#         background-color: #FF6347;
#         color: white;
#         font-size: 1.2em;
#         border-radius: 5px;
#     }
#     .stButton button:hover {
#         background-color: #FF4500;
#         color: white;
#     }
#     .spinner {
#         text-align: center;
#         font-size: 1.5em;
#         color: #4B0082;
#         margin-top: 20px;
#     }
#     .footer {
#         position: fixed;
#         left: 0;
#         bottom: 0;
#         width: 100%;
#         background-color: #f1f1f1;
#         color: black;
#         text-align: center;
#         padding: 10px 0;
#         font-size: 0.9em;
#         border-top: 1px solid #ddd;
#     }

# """, unsafe_allow_html=True)

# st.markdown('<div class="title">Instagram Caption Generator</div>', unsafe_allow_html=True)

# api_key = st.secrets["GOOGLE_API_KEY"]

# lang = st.selectbox("Select Language:", ['English(Default)', 'French'], index=None)
# option = st.radio("Choose input method: ", ['Explain Scenario using text to get the caption', 'Upload image to get the caption'], index=None)


# # select the appropriate model based on the user's choice in the 'option' input.
# # Note: Google has updated model names. Using 'gemini-1.5-pro' and 'gemini-1.5-flash'
# def llm_invoke(option, api_key=api_key):
#     # Use gemini-1.5-pro for text scenarios (more powerful)
#     # Use gemini-1.5-flash for image scenarios (faster, supports vision)
#     if option == "Explain Scenario using text to get the caption":
#         llm = ChatGoogleGenerativeAI(model='gemini-1.5-pro', temperature=0, max_output_tokens=None, api_key=api_key)
#     else:
#         # For image uploads, use gemini-1.5-flash which supports vision
#         llm = ChatGoogleGenerativeAI(model='gemini-1.5-flash', api_key=api_key)
#     return llm

# def generate_message(option, llm, totalCaptions, language, captionTone, captionLength):
#     # Determine input type based on the selected option
#     if option == "Explain Scenario using text to get the caption":
#         # Text input mode
#         scene = st.text_input("Explain the scenario for which you want the caption: ")
#         if scene:
#             template = PromptTemplate(
#                 input_variables=["language", "totalCaptions", "scenario", "captionTone", "captionLength"],
#                 template="I want {totalCaptions} alternative caption(s) for the following scenario: \"{scenario}\" in {language} language, and the tone should be {captionTone} and the caption length should be Instagram {captionLength} size"
#                 )
#             formatted_prompt = template.format(
#                 totalCaptions=totalCaptions,
#                 scenario=scene,
#                 language=language,
#                 captionTone=captionTone,
#                 captionLength=captionLength
#             )
#             # Use proper LangChain message objects
#             prompt = [
#                 SystemMessage(content='You are a helpful assistant that helps people generate their instagram story and post captions'),
#                 HumanMessage(content=formatted_prompt)
#             ]
#             return prompt
#         else:
#             st.warning('🚨Please explain the scenario first...')
#             return None
#     else:
#         # Image upload mode
#         uploaded_file = st.file_uploader("Upload an image:", type=['jpg', 'jpeg', 'png'])
#         if uploaded_file is not None:
#             # Read image data and ensure it's in RGB mode (required by Gemini)
#             img_data = uploaded_file.read()
#             image = Image.open(io.BytesIO(img_data))
            
#             # Convert to RGB if necessary (some images might be RGBA or other modes)
#             # This is important for Gemini API compatibility
#             if image.mode != 'RGB':
#                 image = image.convert('RGB')
            
#             text = f'You are a helpful assistant that helps people generate their instagram story and post captions. I want {totalCaptions} alternative caption(s) for the following image in {language} language, and the tone should be {captionTone} and the caption length should be Instagram {captionLength} size'
            
#             # For ChatGoogleGenerativeAI with images, use the correct dict format
#             # Convert image to base64 string (Pydantic accepts this format)
#             from io import BytesIO
#             img_bytes_io = BytesIO()
#             image.save(img_bytes_io, format='JPEG')
#             img_bytes = img_bytes_io.getvalue()
#             img_base64 = base64.b64encode(img_bytes).decode('utf-8')
            
#             # Include system instruction in the text
#             full_text = f'You are a helpful assistant that helps people generate their instagram story and post captions. {text}'
            
#             # Use the dict format that Pydantic accepts and ChatGoogleGenerativeAI expects
#             # This is the standard format: list of dicts with type and content
#             human_msg = HumanMessage(
#                 content=[
#                     {"type": "text", "text": full_text},
#                     {
#                         "type": "image_url",
#                         "image_url": f"data:image/jpeg;base64,{img_base64}"
#                     }
#                 ]
#             )
            
#             # Return just the HumanMessage (without SystemMessage) for image inputs
#             prompt = [human_msg]
#             return prompt
#         else:
#             st.warning('🚨Please upload an image.....')
#             return None

# if lang and option:
#     llm = llm_invoke(option)
#     captions = st.selectbox('Select Number of Alternatives you want: ', [1, 2, 3, 4, 5], index=None)
#     tone = st.selectbox("Select Tone:", ['Creative', 'Humorous', 'Funny', 'Humorous and funny', 'Conversational', 'genZ language'], index=None)
#     length = st.selectbox('Select caption Type:', ['Story caption (short)', 'Post caption (Short)', 'Story caption (long)', 'Post caption (long)'], index=None)
#     prompt = generate_message(option, llm, captions, lang, tone, length)

#     placeholder = st.empty()
#     generate_button = st.button("Generate Caption", disabled=not (prompt and captions and tone and length))

#     if generate_button:
#         content = ""
#         with st.spinner("Generating captions..."):
#             # Stream the response - both gemini-1.5-pro and gemini-1.5-flash work the same way
#             # Both use the same message format now
#             try:
#                 # Try streaming with the prompt directly (works for both text and image prompts)
#                 for chunk in llm.stream(prompt):
#                     if hasattr(chunk, 'content') and chunk.content:
#                         for char in chunk.content:
#                             content += char
#                             placeholder.markdown(content)
#                             time.sleep(0.005)
#             except Exception as e:
#                 st.error(f"Error generating caption: {str(e)}")
#                 st.info("Please check your API key and model availability.")
#         st.session_state['generated_content'] = content

#     flag = 'generated_content' in st.session_state
#     clear_button = st.button("Clear Output", disabled= not flag)

#     if clear_button:
#         st.session_state.pop('generated_content', None)
#         placeholder.empty()
#         st.query_params.clear()
#         st.rerun()




import base64
import os
import time
from importlib import metadata

import streamlit as st
from PIL import Image
import io

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage


def join_with_hyphen(*parts: str) -> str:
    return chr(45).join(parts)


def require_recent_langchain_google() -> None:
    dist = join_with_hyphen("langchain", "google", "genai")
    try:
        ver = metadata.version(dist)
    except metadata.PackageNotFoundError:
        st.error("Missing langchain google genai package")
        st.stop()

    def to_ints(v: str) -> tuple[int, int, int]:
        core = v.split(".")[:3]
        nums = []
        for x in core:
            s = "".join(ch for ch in x if ch.isdigit())
            nums.append(int(s) if s else 0)
        while len(nums) < 3:
            nums.append(0)
        return tuple(nums)

    if to_ints(ver) < (4, 0, 0):
        st.error("Update langchain google genai to version 4.0.0 or newer")
        st.stop()


require_recent_langchain_google()

st.set_page_config(page_title="Instagram Caption Generator", page_icon="📷", layout="centered")

st.markdown(
    """
    <style>
    .title { font-size: 2.2em; text-align: center; font-weight: 700; margin-bottom: 10px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="title">Instagram Caption Generator</div>', unsafe_allow_html=True)

api_key = st.secrets.get("GOOGLE_API_KEY") or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
if not api_key:
    st.error("Set GOOGLE_API_KEY or GEMINI_API_KEY")
    st.stop()

lang = st.selectbox("Select Language", ["English", "French"], index=0)
mode = st.radio(
    "Choose input method",
    ["Explain scenario using text", "Upload image"],
    index=0,
)

quality = st.selectbox("Model quality", ["Fast", "Best"], index=0)

model_fast = join_with_hyphen("gemini", "2.5", "flash", "lite")
model_best = join_with_hyphen("gemini", "2.5", "pro")

model_id = model_best if quality == "Best" else model_fast

llm = ChatGoogleGenerativeAI(
    model=model_id,
    api_key=api_key,
    temperature=0.4,
)

captions = st.selectbox("Number of alternatives", [1, 2, 3, 4, 5], index=0)
tone = st.selectbox(
    "Tone",
    ["Creative", "Humorous", "Funny", "Conversational", "Gen Z"],
    index=0,
)
length = st.selectbox(
    "Caption type",
    ["Story short", "Post short", "Story long", "Post long"],
    index=0,
)

template = PromptTemplate(
    input_variables=["language", "total", "tone", "length", "scenario"],
    template=(
        "Generate {total} alternative Instagram captions in {language}. "
        "Tone is {tone}. Length style is {length}. "
        "Scenario or context is {scenario}. "
        "Return one caption per line."
    ),
)

placeholder = st.empty()
generate_button = st.button("Generate captions")

def stream_to_placeholder(messages: list) -> str:
    content = ""
    for chunk in llm.stream(messages):
        piece = getattr(chunk, "content", "")
        if isinstance(piece, list):
            text_parts = []
            for p in piece:
                if isinstance(p, str):
                    text_parts.append(p)
                elif isinstance(p, dict) and "text" in p:
                    text_parts.append(str(p["text"]))
            piece = "".join(text_parts)
        if piece:
            content += str(piece)
            placeholder.markdown(content)
            time.sleep(0.003)
    return content

if generate_button:
    if mode == "Explain scenario using text":
        scene = st.text_input("Explain the scenario", value="", placeholder="Example A beach sunset with friends")
        if not scene:
            st.warning("Please enter a scenario")
            st.stop()

        prompt_text = template.format(
            language=lang,
            total=captions,
            tone=tone,
            length=length,
            scenario=scene,
        )

        messages = [
            SystemMessage(content="You write strong Instagram captions."),
            HumanMessage(content=prompt_text),
        ]

        with st.spinner("Generating"):
            final_text = stream_to_placeholder(messages)
            st.session_state["generated_content"] = final_text

    else:
        uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if uploaded is None:
            st.warning("Please upload an image")
            st.stop()

        img_bytes = uploaded.read()
        image = Image.open(io.BytesIO(img_bytes))
        if image.mode != "RGB":
            image = image.convert("RGB")

        buf = io.BytesIO()
        image.save(buf, format="JPEG")
        jpg_bytes = buf.getvalue()
        img_b64 = base64.b64encode(jpg_bytes).decode("utf-8")

        scenario = (
            f"Create {captions} alternative Instagram captions in {lang}. "
            f"Tone is {tone}. Length style is {length}. "
            "Use the image as the main context. Return one caption per line."
        )

        messages = [
            HumanMessage(
                content=[
                    {"type": "text", "text": scenario},
                    {
                        "type": "image",
                        "base64": img_b64,
                        "mime_type": "image/jpeg",
                    },
                ]
            )
        ]

        with st.spinner("Generating"):
            final_text = stream_to_placeholder(messages)
            st.session_state["generated_content"] = final_text

if st.button("Clear output"):
    st.session_state.pop("generated_content", None)
    placeholder.empty()
    st.rerun()
