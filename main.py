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
import io
import os
from importlib import metadata

import streamlit as st
from PIL import Image

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage


def join_with_dash(*parts: str) -> str:
    return chr(45).join(parts)


def require_langchain_google_genai_v4() -> None:
    dist = join_with_dash("langchain", "google", "genai")
    try:
        ver = metadata.version(dist)
    except metadata.PackageNotFoundError:
        st.error("Missing langchain google genai package")
        st.stop()

    def to_ints(v: str) -> tuple[int, int, int]:
        raw = v.split(".")[:3]
        nums: list[int] = []
        for x in raw:
            digits = "".join(ch for ch in x if ch.isdigit())
            nums.append(int(digits) if digits else 0)
        while len(nums) < 3:
            nums.append(0)
        return (nums[0], nums[1], nums[2])

    if to_ints(ver) < (4, 0, 0):
        st.error("Please upgrade langchain_google_genai to version 4 point 0 point 0 or newer")
        st.stop()


@st.cache_resource
def get_llm(model_id: str, api_key: str, temperature: float) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=model_id,
        api_key=api_key,
        temperature=temperature,
    )


def model_candidates() -> list[str]:
    gemini = "gemini"
    a = join_with_dash(gemini, "2.5", "flash", "lite")
    b = join_with_dash(gemini, "2.5", "flash")
    return [a, b]


def select_model_id(choice: str) -> str:
    cands = model_candidates()
    if choice == "Best":
        return cands[1]
    return cands[0]


def image_to_data_uri(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf8")
    return "data:image/jpeg;base64," + b64


def stream_answer(llm: ChatGoogleGenerativeAI, messages: list, out_box) -> str:
    text = ""
    for chunk in llm.stream(messages):
        piece = getattr(chunk, "content", "")
        if isinstance(piece, list):
            joined = ""
            for p in piece:
                if isinstance(p, str):
                    joined += p
                elif isinstance(p, dict) and "text" in p:
                    joined += str(p["text"])
            piece = joined
        if piece:
            text += str(piece)
            out_box.markdown(text)
    return text


require_langchain_google_genai_v4()

st.set_page_config(page_title="Instagram Caption Generator", page_icon="📷", layout="centered")
st.title("Instagram Caption Generator")

api_key = (
    st.secrets.get("GOOGLE_API_KEY")
    or os.environ.get("GOOGLE_API_KEY")
    or os.environ.get("GEMINI_API_KEY")
)
if not api_key:
    st.error("Set GOOGLE_API_KEY or GEMINI_API_KEY in Streamlit secrets or environment")
    st.stop()

st.caption("Text mode uses a fast chat model, image mode uses the same model with vision input")

out_box = st.empty()

if "generated" not in st.session_state:
    st.session_state.generated = ""

with st.form("caption_form"):
    col_a, col_b = st.columns(2)
    with col_a:
        language = st.selectbox("Language", ["English", "French"], index=0)
        mode = st.radio("Input", ["Text scenario", "Image upload"], index=0)
        quality = st.selectbox("Quality", ["Fast", "Best"], index=0)
    with col_b:
        count = st.selectbox("Alternatives", [1, 2, 3, 4, 5], index=0)
        tone = st.selectbox("Tone", ["Creative", "Humorous", "Funny", "Conversational", "Gen Z"], index=0)
        length = st.selectbox("Length", ["Story short", "Post short", "Story long", "Post long"], index=0)

    temperature = st.slider("Creativity", 0.0, 1.0, 0.6, 0.05)

    scenario_text = ""
    uploaded_file = None

    if mode == "Text scenario":
        scenario_text = st.text_area(
            "Scenario",
            placeholder="Example Sunset at the beach with friends, candid vibes",
            height=120,
        )
    else:
        uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

    submitted = st.form_submit_button("Generate")

if submitted:
    model_id = select_model_id(quality)
    llm = get_llm(model_id, api_key, temperature)

    system = SystemMessage(content="You write strong Instagram captions. Return one caption per line.")

    if mode == "Text scenario":
        if not scenario_text.strip():
            st.warning("Please enter a scenario")
            st.stop()

        user_prompt = (
            "Generate "
            + str(count)
            + " alternative Instagram captions in "
            + language
            + ". Tone is "
            + tone
            + ". Length style is "
            + length
            + ". Scenario is "
            + scenario_text.strip()
            + ". Return one caption per line."
        )

        messages = [system, HumanMessage(content=user_prompt)]

        with st.spinner("Generating"):
            st.session_state.generated = stream_answer(llm, messages, out_box)

    else:
        if uploaded_file is None:
            st.warning("Please upload an image")
            st.stop()

        img_bytes = uploaded_file.read()
        image = Image.open(io.BytesIO(img_bytes))
        if image.mode != "RGB":
            image = image.convert("RGB")

        st.image(image, caption="Uploaded image", use_container_width=True)

        user_text = (
            "Generate "
            + str(count)
            + " alternative Instagram captions in "
            + language
            + ". Tone is "
            + tone
            + ". Length style is "
            + length
            + ". Use the image as the main context. Return one caption per line."
        )

        data_uri = image_to_data_uri(image)

        messages = [
            system,
            HumanMessage(
                content=[
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": data_uri},
                ]
            ),
        ]

        with st.spinner("Generating"):
            st.session_state.generated = stream_answer(llm, messages, out_box)

if st.session_state.generated:
    st.divider()
    st.subheader("Output")
    st.text(st.session_state.generated)

col1, col2 = st.columns(2)
with col1:
    if st.button("Clear"):
        st.session_state.generated = ""
        out_box.empty()
        st.rerun()
with col2:
    st.download_button(
        "Download",
        data=st.session_state.generated or "",
        file_name="captions.txt",
        mime="text/plain",
        disabled=not bool(st.session_state.generated),
    )

