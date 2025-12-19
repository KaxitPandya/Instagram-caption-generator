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
        nums = []
        for x in raw:
            digits = "".join(ch for ch in x if ch.isdigit())
            nums.append(int(digits) if digits else 0)
        while len(nums) < 3:
            nums.append(0)
        return (nums[0], nums[1], nums[2])

    if to_ints(ver) < (4, 0, 0):
        st.error("Upgrade langchain google genai to version 4 point 0 point 0 or newer")
        st.stop()


@st.cache_resource
def get_llm(model_id: str, api_key: str, temperature: float) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(model=model_id, api_key=api_key, temperature=temperature)


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
    st.error("Set GOOGLE_API_KEY or GEMINI_API_KEY")
    st.stop()

if "generated" not in st.session_state:
    st.session_state.generated = ""

out_box = st.empty()

col_left, col_right = st.columns(2)

with col_left:
    language = st.selectbox("Language", ["English", "French"], index=0)
    mode = st.radio("Input", ["Text scenario", "Image upload"], index=0)
    quality = st.selectbox("Quality", ["Fast", "Best"], index=0)

with col_right:
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
        key="scenario_text",
    )
else:
    uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"], key="image_file")

gemini = "gemini"
model_fast = join_with_dash(gemini, "2.5", "flash", "lite")
model_best = join_with_dash(gemini, "2.5", "flash")
model_id = model_best if quality == "Best" else model_fast

generate = st.button("Generate")
clear = st.button("Clear")

if clear:
    st.session_state.generated = ""
    out_box.empty()
    st.rerun()

if generate:
    llm = get_llm(model_id, api_key, temperature)
    system = SystemMessage(content="You write strong Instagram captions. Return one caption per line.")

    if mode == "Text scenario":
        if not scenario_text.strip():
            st.warning("Please enter a scenario")
            st.stop()

        user_text = (
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

        messages = [system, HumanMessage(content=user_text)]
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

        st.image(image, use_container_width=True)

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
    st.subheader("Output")
    st.text(st.session_state.generated)
    st.download_button(
        "Download",
        data=st.session_state.generated,
        file_name="captions.txt",
        mime="text/plain",
    )

# import base64
# import io
# import os
# from importlib import metadata

# import streamlit as st
# from PIL import Image

# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_core.messages import HumanMessage, SystemMessage


# def join_with_dash(*parts: str) -> str:
#     return chr(45).join(parts)


# def require_langchain_google_genai_v4() -> None:
#     dist = join_with_dash("langchain", "google", "genai")
#     try:
#         ver = metadata.version(dist)
#     except metadata.PackageNotFoundError:
#         st.error("Missing langchain google genai package")
#         st.stop()

#     def to_ints(v: str) -> tuple[int, int, int]:
#         raw = v.split(".")[:3]
#         nums: list[int] = []
#         for x in raw:
#             digits = "".join(ch for ch in x if ch.isdigit())
#             nums.append(int(digits) if digits else 0)
#         while len(nums) < 3:
#             nums.append(0)
#         return (nums[0], nums[1], nums[2])

#     if to_ints(ver) < (4, 0, 0):
#         st.error("Please upgrade langchain_google_genai to version 4 point 0 point 0 or newer")
#         st.stop()


# @st.cache_resource
# def get_llm(model_id: str, api_key: str, temperature: float) -> ChatGoogleGenerativeAI:
#     return ChatGoogleGenerativeAI(
#         model=model_id,
#         api_key=api_key,
#         temperature=temperature,
#     )


# def model_candidates() -> list[str]:
#     gemini = "gemini"
#     a = join_with_dash(gemini, "2.5", "flash", "lite")
#     b = join_with_dash(gemini, "2.5", "flash")
#     return [a, b]


# def select_model_id(choice: str) -> str:
#     cands = model_candidates()
#     if choice == "Best":
#         return cands[1]
#     return cands[0]


# def image_to_data_uri(image: Image.Image) -> str:
#     buf = io.BytesIO()
#     image.save(buf, format="JPEG")
#     b64 = base64.b64encode(buf.getvalue()).decode("utf8")
#     return "data:image/jpeg;base64," + b64


# def stream_answer(llm: ChatGoogleGenerativeAI, messages: list, out_box) -> str:
#     text = ""
#     for chunk in llm.stream(messages):
#         piece = getattr(chunk, "content", "")
#         if isinstance(piece, list):
#             joined = ""
#             for p in piece:
#                 if isinstance(p, str):
#                     joined += p
#                 elif isinstance(p, dict) and "text" in p:
#                     joined += str(p["text"])
#             piece = joined
#         if piece:
#             text += str(piece)
#             out_box.markdown(text)
#     return text


# require_langchain_google_genai_v4()

# st.set_page_config(page_title="Instagram Caption Generator", page_icon="📷", layout="centered")
# st.title("Instagram Caption Generator")

# api_key = (
#     st.secrets.get("GOOGLE_API_KEY")
#     or os.environ.get("GOOGLE_API_KEY")
#     or os.environ.get("GEMINI_API_KEY")
# )
# if not api_key:
#     st.error("Set GOOGLE_API_KEY or GEMINI_API_KEY in Streamlit secrets or environment")
#     st.stop()

# st.caption("Text mode uses a fast chat model, image mode uses the same model with vision input")

# out_box = st.empty()

# if "generated" not in st.session_state:
#     st.session_state.generated = ""

# with st.form("caption_form"):
#     col_a, col_b = st.columns(2)
#     with col_a:
#         language = st.selectbox("Language", ["English", "French"], index=0)
#         mode = st.radio("Input", ["Text scenario", "Image upload"], index=0)
#         quality = st.selectbox("Quality", ["Fast", "Best"], index=0)
#     with col_b:
#         count = st.selectbox("Alternatives", [1, 2, 3, 4, 5], index=0)
#         tone = st.selectbox("Tone", ["Creative", "Humorous", "Funny", "Conversational", "Gen Z"], index=0)
#         length = st.selectbox("Length", ["Story short", "Post short", "Story long", "Post long"], index=0)

#     temperature = st.slider("Creativity", 0.0, 1.0, 0.6, 0.05)

#     scenario_text = ""
#     uploaded_file = None

#     if mode == "Text scenario":
#         scenario_text = st.text_area(
#             "Scenario",
#             placeholder="Example Sunset at the beach with friends, candid vibes",
#             height=120,
#         )
#     else:
#         uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

#     submitted = st.form_submit_button("Generate")

# if submitted:
#     model_id = select_model_id(quality)
#     llm = get_llm(model_id, api_key, temperature)

#     system = SystemMessage(content="You write strong Instagram captions. Return one caption per line.")

#     if mode == "Text scenario":
#         if not scenario_text.strip():
#             st.warning("Please enter a scenario")
#             st.stop()

#         user_prompt = (
#             "Generate "
#             + str(count)
#             + " alternative Instagram captions in "
#             + language
#             + ". Tone is "
#             + tone
#             + ". Length style is "
#             + length
#             + ". Scenario is "
#             + scenario_text.strip()
#             + ". Return one caption per line."
#         )

#         messages = [system, HumanMessage(content=user_prompt)]

#         with st.spinner("Generating"):
#             st.session_state.generated = stream_answer(llm, messages, out_box)

#     else:
#         if uploaded_file is None:
#             st.warning("Please upload an image")
#             st.stop()

#         img_bytes = uploaded_file.read()
#         image = Image.open(io.BytesIO(img_bytes))
#         if image.mode != "RGB":
#             image = image.convert("RGB")

#         st.image(image, caption="Uploaded image", use_container_width=True)

#         user_text = (
#             "Generate "
#             + str(count)
#             + " alternative Instagram captions in "
#             + language
#             + ". Tone is "
#             + tone
#             + ". Length style is "
#             + length
#             + ". Use the image as the main context. Return one caption per line."
#         )

#         data_uri = image_to_data_uri(image)

#         messages = [
#             system,
#             HumanMessage(
#                 content=[
#                     {"type": "text", "text": user_text},
#                     {"type": "image_url", "image_url": data_uri},
#                 ]
#             ),
#         ]

#         with st.spinner("Generating"):
#             st.session_state.generated = stream_answer(llm, messages, out_box)

# if st.session_state.generated:
#     st.divider()
#     st.subheader("Output")
#     st.text(st.session_state.generated)

# col1, col2 = st.columns(2)
# with col1:
#     if st.button("Clear"):
#         st.session_state.generated = ""
#         out_box.empty()
#         st.rerun()
# with col2:
#     st.download_button(
#         "Download",
#         data=st.session_state.generated or "",
#         file_name="captions.txt",
#         mime="text/plain",
#         disabled=not bool(st.session_state.generated),
#     )


