
import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from datetime import datetime
import pandas as pd

# Load the fine-tuned model and tokenizer
model_name_or_path = "./fine_tuned_model"
model = GPT2LMHeadModel.from_pretrained(model_name_or_path)
tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path)

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define the text generation function
def generate_text(seed_text, max_length=100, temperature=1.0, num_return_sequences=1):
    input_ids = tokenizer.encode(seed_text, return_tensors='pt').to(device)
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=max_length,
            temperature=temperature,
            num_return_sequences=num_return_sequences,
            do_sample=True,
            top_k=50,
            top_p=0.95,
        )
    generated_texts = [tokenizer.decode(output[i], skip_special_tokens=True) for i in range(num_return_sequences)]
    return generated_texts

# Function to log user information
def log_user_info(username, email):
    login_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_data = {"Username": [username], "Email": [email], "Login Time": [login_time]}
    log_df = pd.DataFrame(log_data)
    log_df.to_csv("user_log.csv", mode='a', header=False, index=False)
    st.success(f"User {username} logged in at {login_time}")

# Streamlit App
st.set_page_config(page_title="Text Generation App", layout="centered")

# Initialize session state for login status
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

# Login Page
if not st.session_state.logged_in:
    st.title("Text Generation App")
    st.header("Login")

    # User authentication
    username = st.text_input("Username")
    email = st.text_input("Email")
    login_button = st.button("Login")

    if login_button and username and email:
        st.session_state.logged_in = True
        st.session_state.username = username
        st.session_state.email = email
        log_user_info(username, email)

# Text Generation Page
if st.session_state.logged_in:
    st.title("Text Generation App")
    st.header("Generate Text")
    prompt = st.text_area("Enter your prompt here:")
    max_length = st.slider("Max Length", min_value=50, max_value=500, value=100)
    temperature = st.slider("Temperature", min_value=0.1, max_value=1.0, value=1.0)
    generate_button = st.button("Generate")

    if generate_button and prompt:
        with st.spinner("Generating text..."):
            generated_texts = generate_text(prompt, max_length=max_length, temperature=temperature)
            st.success("Text generated successfully!")
            for i, text in enumerate(generated_texts):
                st.text_area(f"Generated Text {i + 1}:", value=text, height=300)

    # Logout button
    if st.button("Logout"):
        st.session_state.logged_in = False

    # Display logged user information (optional)
    if st.checkbox("Show Logged User Info"):
        try:
            user_log_df = pd.read_csv("user_log.csv", names=["Username", "Email", "Login Time"])
            st.dataframe(user_log_df)
        except FileNotFoundError:
            st.info("No login data available yet.")
