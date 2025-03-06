import streamlit as st
from transformers import BertForQuestionAnswering, BertTokenizer
import torch

st.set_page_config(page_title="Fine-Tuned BERT Question Answering System")

# Function to load the fine-tuned BERT model and tokenizer from local directory
# Cached using st.cache_resource to avoid reloading on each rerun (improves performance)
@st.cache_resource
def load_model():
    try:
        # Load the fine-tuned model and tokenizer from local files
        model = BertForQuestionAnswering.from_pretrained("./fine-tuned-qa-model")
        tokenizer = BertTokenizer.from_pretrained("./fine-tuned-qa-tokenizer")
        return model, tokenizer
    except Exception as e:
        # Handle loading errors (e.g., missing files, wrong paths)
        st.error(f"Error loading model or tokenizer: {e}")
        return None, None

# Load model and tokenizer once (due to caching)
model, tokenizer = load_model()

# Streamlit app interface
st.title("Question Answering System with Fine-Tuned BERT")

# Introduction and usage instructions for users
st.write("""
Welcome to the Question Answering System! This tool uses a fine-tuned BERT model to answer questions based on the context you provide.

### How to use:
1. **Enter the Context**: Paste or type the text you want to ask questions about in the "Context" box.
2. **Ask a Question**: Type your question in the "Question" box.
3. **Get the Answer**: Click the "Get Answer" button, and the system will predict the answer based on the context.

**Note**: The model works best with clear and concise contexts and questions.
""")

# User inputs: context and question
context = st.text_area("Context", "Enter the context here...")
question = st.text_input("Question", "Enter your question here...")

# Button to trigger prediction process
if st.button("Get Answer"):
    # Validate inputs
    if not context.strip() or not question.strip():
        st.warning("Please provide both a context and a question.")
    # Check if the model and tokenizer loaded successfully
    elif model is None or tokenizer is None:
        st.error("The model or tokenizer failed to load. Please check the model files and try again.")
    else:
        try:
            # Tokenize context and question together (required input format for BERT QA)
            inputs = tokenizer(question, context, return_tensors="pt", max_length=512, truncation=True)
            
            # Perform inference with the BERT model (disable gradient computation for efficiency)
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Extract the predicted start and end token positions for the answer
            start_pred = torch.argmax(outputs.start_logits)
            end_pred = torch.argmax(outputs.end_logits)
            
            # Convert token IDs back to human-readable text (answer span)
            answer = tokenizer.convert_tokens_to_string(
                tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][start_pred:end_pred + 1])
            )
            
            # Display the predicted answer (or show a warning if it's empty)
            if answer.strip():
                st.success(f"**Answer:** {answer}")
            else:
                st.warning("The model could not find an answer. Please check the context and question.")
        except Exception as e:
            # Catch unexpected errors during processing
            st.error(f"An error occurred while processing your request: {e}")
