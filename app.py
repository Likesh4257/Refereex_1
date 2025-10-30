import os
import shutil
import numpy as np
import torch
import faiss
import pymupdf  # PyMuPDF for PDF extraction
from transformers import AutoTokenizer, AutoModel
import streamlit as st

# Directories
DATASET_DIR = "dataset"
REMOVED_DIR = "removed"
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(REMOVED_DIR, exist_ok=True)

# Load embedding model
MODEL_NAME = "distilbert-base-uncased"  # Lightweight model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# FAISS vector database with ID mapping
dimension = 768  # DistilBERT's output dimension
index = faiss.IndexIDMap(faiss.IndexFlatL2(dimension))
paper_metadata = {}  # id -> {"author": ..., "paper": ...}
next_id = 0


# --- Utils --- #
def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from a PDF using PyMuPDF"""
    with pymupdf.open(file_path) as doc:
        text = ""
        for page in doc:
            text += page.get_text()
    return text.strip()


def embed_text(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings.cpu().numpy()


def add_paper_embedding(author: str, paper_name: str):
    global next_id
    paper_path = os.path.join(DATASET_DIR, author, paper_name)
    text = extract_text_from_pdf(paper_path)
    embedding = embed_text(text)
    paper_id = next_id
    next_id += 1
    index.add_with_ids(embedding, np.array([paper_id]))
    paper_metadata[paper_id] = {"author": author, "paper": paper_name}


def rebuild_index():
    """Rebuild index from dataset (only if needed)."""
    global index, paper_metadata, next_id
    index = faiss.IndexIDMap(faiss.IndexFlatL2(dimension))
    paper_metadata = {}
    next_id = 0
    for author in os.listdir(DATASET_DIR):
        author_dir = os.path.join(DATASET_DIR, author)
        if os.path.isdir(author_dir):
            for paper in os.listdir(author_dir):
                add_paper_embedding(author, paper)


# Streamlit UI Components

def upload_paper():
    st.subheader("Add a Paper")
    author_name = st.text_input("Author Name")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:
        if author_name:
            author_dir = os.path.join(DATASET_DIR, author_name)
            os.makedirs(author_dir, exist_ok=True)
            file_path = os.path.join(author_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            add_paper_embedding(author_name, uploaded_file.name)
            st.success(f"Paper {uploaded_file.name} added for {author_name}.")
        else:
            st.error("Please enter an author name.")


def list_reviewers():
    st.subheader("List of Reviewers and Their Papers")
    reviewers = {}
    for meta in paper_metadata.values():
        reviewers.setdefault(meta["author"], []).append(meta["paper"])
    st.write(reviewers)


def remove_reviewer():
    st.subheader("Remove Reviewer")
    author_name = st.text_input("Enter the name of the reviewer to remove")

    if author_name:
        to_remove_ids = [pid for pid, meta in paper_metadata.items() if meta["author"] == author_name]
        if to_remove_ids:
            index.remove_ids(np.array(to_remove_ids))
            for pid in to_remove_ids:
                del paper_metadata[pid]

            author_dir = os.path.join(DATASET_DIR, author_name)
            if os.path.exists(author_dir):
                shutil.move(author_dir, os.path.join(REMOVED_DIR, author_name))

            st.success(f"Reviewer {author_name} and their papers removed.")
        else:
            st.error(f"No papers found for reviewer {author_name}.")


def remove_paper():
    st.subheader("Remove a Paper")
    paper_name = st.text_input("Enter the name of the paper to remove")

    if paper_name:
        pid_to_remove = None
        for pid, meta in paper_metadata.items():
            if meta["paper"] == paper_name:
                pid_to_remove = pid
                break
        if pid_to_remove is None:
            st.error("Paper not found.")
        else:
            # Remove from FAISS
            index.remove_ids(np.array([pid_to_remove]))
            meta = paper_metadata.pop(pid_to_remove)

            # Move file to removed directory
            author_dir = os.path.join(DATASET_DIR, meta["author"])
            removed_dir = os.path.join(REMOVED_DIR, meta["author"])
            os.makedirs(removed_dir, exist_ok=True)
            shutil.move(os.path.join(author_dir, paper_name), os.path.join(removed_dir, paper_name))

            st.success(f"Paper {paper_name} removed.")


def get_reviewers_for_new_paper():
    st.subheader("Get Recommended Reviewers for New Paper")
    uploaded_file = st.file_uploader("Upload the new paper (PDF)", type="pdf")

    if uploaded_file is not None:
        text = extract_text_from_pdf(uploaded_file)
        embedding = embed_text(text)

        k = st.slider("Select number of reviewers", 1, 10, 3)
        D, I = index.search(embedding, k * 5)
        authors_seen = set()
        selected_authors = []
        for idx in I[0]:
            author = paper_metadata[idx]["author"]
            if author not in authors_seen:
                authors_seen.add(author)
                selected_authors.append(author)
            if len(selected_authors) >= k:
                break

        st.write("Recommended Reviewers:")
        st.write(selected_authors)


# Main Streamlit App Logic
def main():
    st.title("RefereeX Paper Review Management")

    rebuild_index()  # Rebuild the FAISS index at the start

    menu = ["Upload Paper", "List Reviewers", "Remove Reviewer", "Remove Paper", "Get Reviewers for New Paper"]
    choice = st.sidebar.selectbox("Select an Action", menu)

    if choice == "Upload Paper":
        upload_paper()
    elif choice == "List Reviewers":
        list_reviewers()
    elif choice == "Remove Reviewer":
        remove_reviewer()
    elif choice == "Remove Paper":
        remove_paper()
    elif choice == "Get Reviewers for New Paper":
        get_reviewers_for_new_paper()


if __name__ == "__main__":
    main()
