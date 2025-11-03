# Refereex_1
1. Setup Directories and Variables

    Create two directories: dataset (to store papers) and removed (for removed papers).
    Initialize the model and tokenizer (distilbert-base-uncased for embeddings).
    Set up the FAISS vector index to store embeddings and the associated metadata.

2. Helper Functions

    extract_text_from_pdf(file_path): Extracts text from PDF using PyMuPDF.
    embed_text(text): Generates embedding for the extracted text using the DistilBERT model.
    add_paper_embedding(author, paper_name):
        Extracts the paper from dataset/author/paper_name,
        Creates an embedding for the text,
        Adds the embedding to the FAISS index and stores metadata.
    rebuild_index(): Rebuilds the FAISS index from all papers in the dataset (useful for re-indexing when needed).

3. Streamlit UI Components

    Upload Paper:
        Users provide an author name and upload a PDF.
        Paper is stored in the dataset directory and embedded for indexing.
    List Reviewers:
        Displays a list of authors (reviewers) and their associated papers.
    Remove Reviewer:
        Users enter an author name, and if papers exist for the author, their embeddings are removed from the FAISS index and the papers are moved to the removed directory.
    Remove Paper:
        Users enter a paper name, and if found, it is removed from the FAISS index and moved to the removed directory.
    Get Reviewers for New Paper:
        Users upload a new paper PDF.
        The paper is embedded, and the system returns a list of recommended reviewers based on similarity (FAISS search for top k similar authors).

4. Main Function (main)
    The app is titled "RefereeX Paper Review Management".
    The FAISS index is rebuilt at the start of the app.
    The user selects an action from the sidebar:
        Upload Paper: Add a new paper.
        List Reviewers: View all reviewers and their papers.
        Remove Reviewer: Remove an author and their papers.
        Remove Paper: Remove a specific paper.
        Get Reviewers for New Paper: Get recommended reviewers for a new paper.

5. Flow of Paper Embeddings and FAISS Search

    Paper text is extracted and embedded as a vector.
    The embedding is added to FAISS with metadata (author and paper name).
    When a new paper is uploaded, its embedding is compared to the existing ones, and recommended reviewers (authors) are returned by FAISS based on cosine similarity.

   Team-members:
   Likesh Koya - SE22UARI210
   Vivek Vardhan Pasupuleti - SE22UCSE199
   Menda Anish - SE22UCSE168
