import streamlit as st
import pytesseract
from transformers import pipeline
from PIL import Image
from pdf2image import convert_from_bytes

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

st.set_page_config(
    page_title="AI Document QA",
    page_icon="📄",
    layout="wide"
)

st.markdown("""
<style>

.main-title{
background:linear-gradient(90deg,#2563eb,#1d4ed8);
padding:20px;
border-radius:15px;
text-align:center;
color:white;
font-size:38px;
font-weight:bold;
margin-bottom:20px;
}

.answer-box{
background:#e8f5e9;
padding:20px;
border-radius:10px;
border-left:8px solid green;
font-size:22px;
font-weight:bold;
}

.footer{
text-align:center;
padding:20px;
color:gray;
font-size:15px;
}

</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return pipeline(
        "document-question-answering",
        model="impira/layoutlm-document-qa"
    )

nlp = load_model()

st.sidebar.title("🤖 Project Information")

st.sidebar.success("AI Document Question Answering")

st.sidebar.write("### Features")

st.sidebar.write("✅ Image Upload")
st.sidebar.write("✅ PDF Upload")
st.sidebar.write("✅ AI Question Answering")
st.sidebar.write("✅ Confidence Score")
st.sidebar.write("✅ Question History")
st.sidebar.write("✅ Download Report")

st.sidebar.write("---")

st.sidebar.write("Developer")
st.sidebar.info("Mubasharunnisa")

st.sidebar.write("Version")
st.sidebar.success("v2.0")

st.markdown("""
<div class="main-title">
📄 AI Document Question Answering System
</div>
""", unsafe_allow_html=True)

st.write("Upload a document and ask questions using Artificial Intelligence.")

if "history" not in st.session_state:
    st.session_state.history = []

uploaded_file = st.file_uploader(
    "📂 Upload Document",
    type=["png","jpg","jpeg","pdf"]
)

if uploaded_file is not None:

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Document", uploaded_file.name)

    with col2:
        st.metric("Size", f"{round(uploaded_file.size/1024,1)} KB")

    with col3:
        st.metric("Type", uploaded_file.type.split("/")[-1].upper())

    if uploaded_file.type == "application/pdf":

        pages = convert_from_bytes(uploaded_file.read())

        page_number = st.selectbox(
            "Select PDF Page",
            range(1,len(pages)+1)
        )

        image = pages[page_number-1]

    else:

        page_number = 1
        image = Image.open(uploaded_file)

    st.image(
        image,
        caption=f"Page {page_number}",
        use_container_width=True
    )

    image.save("temp.png")

    question = st.text_input(
        "❓ Ask your question"
    )

    if st.button("🔍 Analyze Document", use_container_width=True):

        with st.spinner("🤖 AI is analyzing your document..."):

            result = nlp("temp.png", question)

        answer = result[0]["answer"]

        score = round(result[0]["score"]*100,2)

        st.markdown(
            f"""
            <div class="answer-box">
            ✅ Answer <br><br>
            {answer}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.metric(
            "Confidence",
            f"{score}%"
        )

        st.session_state.history.append(
            {
                "Question":question,
                "Answer":answer
            }
        )

if len(st.session_state.history)>0:

    st.write("---")

    st.subheader("📜 Question History")

    history_text=""

    for i,item in enumerate(st.session_state.history,start=1):

        st.write(f"**Question {i}:** {item['Question']}")
        st.write(f"**Answer:** {item['Answer']}")
        st.write("---")

        history_text += (
            f"Question {i}: {item['Question']}\n"
            f"Answer: {item['Answer']}\n\n"
        )

    st.download_button(
        "📥 Download Report",
        history_text,
        "answers.txt",
        "text/plain",
        use_container_width=True
    )

st.markdown("""
<div class="footer">

Developed using ❤️ Streamlit | Transformers | LayoutLM | Tesseract OCR

</div>
""", unsafe_allow_html=True)
