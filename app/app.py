import streamlit as st
import requests
import time
import tiktoken
import pandas as pd
import matplotlib.pyplot as plt

BASE_URL = "https://api-695260164759.us-central1.run.app"
##BASE_URL = "http://127.0.0.1:8000"

# Sidebar Navigation
st.sidebar.title("Navigation")
selected_tab = st.sidebar.radio("Go to", ["Extraction", "LLM Processing"])

def count_tokens(text, model):
    """Counts tokens in a given text based on the selected model."""
    try:
        if model in ["GPT-4o", "GPT-3.5"]:
            enc = tiktoken.encoding_for_model("gpt-4")
            return len(enc.encode(text))
        elif model in ["Claude", "Claude-3"]:
            return int(len(text.split()) * 1.2)
        elif model in ["Gemini-Flash", "Gemini-Pro"]:
            return int(len(text.split()) * 1.15)
        elif model in ["DeepSeek"]:
            return int(len(text.split()) * 1.1)
        elif model in ["Grok"]:
            return int(len(text.split()) * 1.05)
        else:
            return len(text.split())
    except Exception:
        return len(text.split())

MODEL_PRICING = {
    "GPT-4o": {"input_price": 5.00, "output_price": 10.00}, 
    "Claude": {"input_price": 3.00, "output_price": 15.00},
    "Gemini-Flash": {"input_price": 0.10, "output_price": 0.40},
    "DeepSeek": {"input_price": 0.27, "output_price": 1.10},
    "Grok": {"input_price": 2.00, "output_price": 10.00}
}


# **Extraction Tab**
if selected_tab == "Extraction":
    st.title("Extract Text from PDF")

    uploaded_file = st.file_uploader("Upload a PDF ", type="pdf", key="pdf_extraction")

    if uploaded_file:
     if st.button("Extract Text ", use_container_width=True):
        with st.spinner(" Extracting text from PDF..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
            response = requests.post(f"{BASE_URL}/upload_pdf/", files=files)

            if response.status_code == 200:
                scraped_file = response.json()["filename"]
                s3_url = response.json()["s3_url"]  # ✅ Get S3 URL
                st.success(f" PDF extracted! Saved as `{scraped_file}` in S3.")

                # 🔍 Debug: Display S3 URL
                st.write(f"S3 File URL: {s3_url}")

                # ✅ Fetch Extracted Text from FastAPI
                text_response = requests.get(f"{BASE_URL}/get_extracted_text/{scraped_file}")

                if text_response.status_code == 200:
                    extracted_text = text_response.json()["extracted_text"]
                    st.session_state["extracted_text"] = extracted_text
                else:
                    st.error(f"❌ Failed to retrieve extracted text. Error Code: {text_response.status_code}")

            else:
                st.error(f"❌ Upload failed: {response.text}")

# ✅ Show Extracted Text
    if "extracted_text" in st.session_state:
        st.markdown("### Extracted Text:")
        st.text_area("Extracted Text", st.session_state["extracted_text"], height=300)
# **🤖 LLM Processing Tab**
elif selected_tab == "LLM Processing":
    st.title("LLM Document Processing")

    # 1️⃣ **Model Selection**
    st.markdown("<h2 style='text-align: center; color: Black;'>Select Model</h2>", unsafe_allow_html=True)
    model = st.selectbox("Choose a model", ["GPT-4o", "Gemini-Flash", "DeepSeek", "Claude", "Grok"], key="model")
    
    # 3️⃣ **List Available Processed Files**
    st.markdown("<h2 style='text-align: center; color: Black;'>Select Processed File</h2>", unsafe_allow_html=True)

    response = requests.get(f"{BASE_URL}/select_pdfcontent/")

    if response.status_code == 200:
        markdown_files = response.json().get("markdowns", [])
    
        if markdown_files:
            selected_markdown = st.selectbox("Select a Processed Markdown File ", markdown_files)
        else:
            st.warning("⚠️ No processed Markdown files found. Upload a PDF first.")
            selected_markdown = None
    else:
        st.error(" Failed to fetch processed Markdown files.")
        selected_markdown = None
        
    # 4️⃣ **Summarization Section**
# ✅ Ensure session state has "summary" key
    if "summary" not in st.session_state:
        st.session_state["summary"] = None

# Summarization Section
    st.markdown("<h2 style='text-align: center; color: black;'>Summarization</h2>", unsafe_allow_html=True)

    if selected_markdown and st.button("Summarize Document ", use_container_width=True):
        with st.spinner("Summarizing document... Please wait."):
            response = requests.post(
                f"{BASE_URL}/summarize/",
                data={"pdf_name": selected_markdown, "llm": model}
        )

        if response.status_code == 200:
            task_id = response.json().get("task_id")
            elapsed_time = 0
            max_wait_time = 600  # 10 minutes timeout

            while elapsed_time < max_wait_time:
                time.sleep(2)  # Reduce API load
                elapsed_time += 2
                result_response = requests.get(f"{BASE_URL}/get_result/{task_id}")

                try:
                    result_data = result_response.json()
                    if "result" in result_data:
                        # ✅ Store summary in session state
                        st.session_state["summary"] = result_data["result"]
                        break
                except requests.exceptions.JSONDecodeError:
                    st.error(" Failed to retrieve a valid response from the server.")
                    break
            else:
                st.error(" Summarization took too long. Please try again later.")
        else:
            st.error(f" Summarization request failed: {response.text}")
    
    
# ✅ Display stored summary even after refresh
    if st.session_state["summary"]:
        st.markdown("<h3 style='text-align: center; color: black;'>Summarization Result:</h3>", unsafe_allow_html=True)
        st.write(st.session_state["summary"])
# ✅ Ensure session state has "answer" key
    if "answer" not in st.session_state:
        st.session_state["answer"] = None

    st.markdown("<h2 style='text-align: center; color: black;'>Ask a Question</h2>", unsafe_allow_html=True)

# ✅ Use st.text_input() instead of st.text_area() so Enter submits automatically
    question = st.text_input("Enter your question and press Enter:")

# ✅ Create an empty placeholder for answer display
    answer_placeholder = st.empty()
    
    if selected_markdown and question and st.button("Get Answer 💡", use_container_width=True):
        with st.spinner("Fetching answer... Please wait."):
            response = requests.post(
                f"{BASE_URL}/ask_question/",
                data={"pdf_name": selected_markdown, "llm": model, "question": question}
        )

        if response.status_code == 200:
            task_id = response.json().get("task_id")

            while True:
                time.sleep(2)
                result_response = requests.get(f"{BASE_URL}/get_result/{task_id}")
                result_data = result_response.json()

                if "result" in result_data:
                    # ✅ Store answer in session state
                    st.session_state["answer"] = result_data["result"]
                    break
        else:
            st.error(f" Failed to submit question request: {response.text}")
     
     # ✅ Display stored answer without affecting summary
    if st.session_state["answer"]:

        answer_placeholder.write(st.session_state["answer"])

    if selected_markdown:
    # Fetch the file content
        file_response = requests.get(f"{BASE_URL}/download_markdown/{selected_markdown}")

        if file_response.status_code == 200:
            file_content = file_response.text
            pdf_token_count = count_tokens(file_content, model)  # ✅ Dynamic token count
            st.session_state["pdf_token_count"] = pdf_token_count
        else:
            st.session_state["pdf_token_count"] = 0
            st.error(" Failed to fetch file content.")

# ✅ Display token count for the selected PDF
    if "pdf_token_count" in st.session_state:
        st.markdown(f" **Token Count for Selected PDF**: {st.session_state['pdf_token_count']} tokens")

# ✅ Display stored summary even after refresh
    if st.session_state["summary"]:
        summary_token_count = count_tokens(st.session_state["summary"], model)  # ✅ Model-aware token count
        st.session_state["summary_token_count"] = summary_token_count
        st.markdown(f" **Token Count for Summarization**: {st.session_state['summary_token_count']} tokens")

    if question:
        question_token_count = count_tokens(question, model)  # ✅ Count question tokens per model
        st.session_state["question_token_count"] = question_token_count
        st.markdown(f" **Token Count for Question**: {st.session_state['question_token_count']} tokens")

    if st.session_state["answer"]:
        answer_token_count = count_tokens(st.session_state["answer"], model)  # ✅ Count answer tokens per model
        st.session_state["answer_token_count"] = answer_token_count

        answer_placeholder.write(st.session_state["answer"])
        st.markdown(f" **Token Count for Answer**: {st.session_state['answer_token_count']} tokens")

    if selected_markdown and model:
    # Fetch the file content only when both file and model are selected
        file_response = requests.get(f"{BASE_URL}/download_markdown/{selected_markdown}")

        if file_response.status_code == 200:
            file_content = file_response.text
            pdf_token_count = count_tokens(file_content, model)  # ✅ Dynamic token count
            st.session_state["pdf_token_count"] = pdf_token_count
        else:
            st.session_state["pdf_token_count"] = 0
            st.error(" Failed to fetch file content.")

        # ✅ Display token count and cost for the selected PDF
        pdf_price = (st.session_state["pdf_token_count"] / 1_000_000) * MODEL_PRICING[model]["input_price"]
        st.markdown(f" **Cost for pdf:** ${pdf_price:.5f}")


    if question and model:
        question_token_count = count_tokens(question, model)  # ✅ Count question tokens per model
        st.session_state["question_token_count"] = question_token_count
        question_price = (question_token_count / 1_000_000) * MODEL_PRICING[model]["input_price"]
        
        st.markdown(f"**Cost for Question:** ${question_price:.5f}")

    # ✅ Display stored summary even after refresh
    if st.session_state["summary"]:
        summary_token_count = count_tokens(st.session_state["summary"], model)  # ✅ Model-aware token count
        st.session_state["summary_token_count"] = summary_token_count
        summary_price = (summary_token_count / 1_000_000) * MODEL_PRICING[model]["output_price"]
        st.markdown(f"**Cost for Summarisation:** ${summary_price:.5f}")


    if st.session_state["answer"]:
        answer_token_count = count_tokens(st.session_state["answer"], model)  # ✅ Count answer tokens per model
        st.session_state["answer_token_count"] = answer_token_count
        answer_price = (answer_token_count / 1_000_000) * MODEL_PRICING[model]["output_price"]

        answer_placeholder.write(st.session_state["answer"])
        st.markdown(f"**Cost for Answer:** ${answer_price:.5f}")

    if "pdf_token_count" in st.session_state and "question_token_count" in st.session_state and model:
        total_input_tokens = st.session_state["pdf_token_count"] + st.session_state["question_token_count"]
        total_input_cost = (total_input_tokens / 1_000_000) * MODEL_PRICING[model]["input_price"]
    else:
        total_input_tokens = 0
        total_input_cost = 0

    if "summary_token_count" in st.session_state and "answer_token_count" in st.session_state and model:
        total_output_tokens = st.session_state["summary_token_count"] + st.session_state["answer_token_count"]
        total_output_cost = (total_output_tokens / 1_000_000) * MODEL_PRICING[model]["output_price"]
    else:
        total_output_tokens = 0
        total_output_cost = 0

    import pandas as pd  # Ensure pandas is imported

    if model and selected_markdown:  # ✅ Ensure both model and file are selected
        total_cost = total_input_cost + total_output_cost

        # ✅ Replace "-" with 0 for numerical values to prevent errors
        cost_data = pd.DataFrame({
            "Category": [" Input Tokens (PDF + Question)", "Output Tokens (Summarization + Answer)", "Total Cost"],
            "Token Count": [total_input_tokens if total_input_tokens > 0 else 0, 
                            total_output_tokens if total_output_tokens > 0 else 0, 
                            None],  # Keep "None" for total cost to avoid errors
            "Cost (USD)": [total_input_cost if total_input_cost > 0 else 0, 
                        total_output_cost if total_output_cost > 0 else 0, 
                        total_cost if total_cost > 0 else 0]  # Ensure numerical values
        })

        # ✅ Convert "Token Count" to integers to avoid serialization error
        cost_data["Token Count"] = cost_data["Token Count"].astype("Int64")

        # ✅ Display table in Streamlit
        st.markdown("### **Total Token Usage & Cost (Based on 1M Token Pricing)**")
        st.table(cost_data)


    # ✅ Create a bar chart for token usage
    if selected_markdown and model:
        fig, ax = plt.subplots(figsize=(8, 5))
        labels = ["PDF Tokens", "Summary Tokens", "Question Tokens", "Answer Tokens"]
        values = [
            st.session_state.get("pdf_token_count", 0),
            st.session_state.get("summary_token_count", 0),
            st.session_state.get("question_token_count", 0),
            st.session_state.get("answer_token_count", 0),
        ]
        ax.bar(labels, values)
        ax.set_ylabel("Token Count")
        ax.set_title("Token Usage Breakdown")

        # ✅ Display the chart in Streamlit
        st.pyplot(fig)

    # ✅ Reset stored values when the user changes the model or file
    if "prev_model" not in st.session_state:
        st.session_state["prev_model"] = model

    if "prev_file" not in st.session_state:
        st.session_state["prev_file"] = selected_markdown
    if "prev_question" not in st.session_state:
        st.session_state["prev_question"] = ""
    
    if model != st.session_state["prev_model"]:
        st.session_state["summary"] = None  # Clear summary
        st.session_state["prev_model"] = model  # Update the tracking model
        st.rerun()

    # ✅ If the user selects a new model or file, reset session state
    if model != st.session_state["prev_model"] or selected_markdown != st.session_state["prev_file"] or st.session_state["prev_question"] != question:
        
        st.session_state["answer"] = None
        st.session_state["question"] = ""
        st.session_state["question_token_count"] = 0
        st.session_state["summary_token_count"] = 0
        st.session_state["answer_token_count"] = 0
        st.session_state["pdf_token_count"] = 0
        st.session_state["prev_model"] = model
        st.session_state["prev_file"] = selected_markdown
        st.session_state["prev_question"] = question
        st.rerun() 
