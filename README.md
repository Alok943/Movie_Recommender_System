Movie Recommendation System (Hybrid Approach)

This project implements a scalable movie recommendation engine that combines Content-Based Filtering (NLP-driven metadata similarity) with Popularity Metrics to balance relevance and quality. The system fetches pre-computed TF-IDF artifacts from Hugging Face, enabling fast inference and typo-tolerant, fuzzy-matched movie suggestions.
🚀 Features

    Remote Artifact Loading: Downloads versioned datasets and similarity matrices directly from the Hugging Face Hub using huggingface_hub, decoupling model artifacts from code.

    Fuzzy Title Matching: Uses the thefuzz library to handle typos and partial search terms, ensuring robust title resolution (e.g., "Avata" → "Avatar").

    Hybrid Scoring Logic: Ranks candidate movies using a weighted hybrid score.

    Scalable Performance: Utilizes numpy.argpartition for partial sorting, enabling efficient top-N retrieval across large datasets (~500k movies).

🛠️ Tech Stack

    Language: Python

    Libraries:

        Pandas: Data manipulation and Parquet handling

        Scikit-Learn: TF-IDF vectorization and cosine similarity computation

        TheFuzz: Fuzzy string matching

        Joblib: Loading serialized matrix files

        HuggingFace Hub: Cloud storage for model artifacts

📦 Installation

To run the notebook locally, install the required dependencies:
Bash

pip install huggingface_hub pandas joblib scikit-learn thefuzz numpy

📋 How to Use

The notebook automatically connects to the Alok8732/Movie_Recommnedation_Artifacts repository to download cleaned movie metadata and pre-computed TF-IDF artifacts.

    Initialize the engine: Run the artifact loading cells to pull data from Hugging Face.

    Generate Recommendations: Use the get_recommendations() function:

Python

# Example Usage
results = get_recommendations("Inception", n=12)
display(results)

🧠 Recommendation Logic

The system calculates a hybrid score for every movie relative to your search:
Score=(α×Cosine Similarity)+(β×Normalized Log Popularity)

    Content Similarity (α=0.85): Prioritizes movies with similar metadata and narrative structure using cosine similarity.

    Popularity Prior (β=0.15): Applies a log-normalized vote-count boost to favor well-received and widely engaged titles.

🛠️ Engineering Advancements (Beyond the Basics)

Unlike a standard tutorial-based recommender, this implementation addresses real-world deployment challenges:

    Decoupled Architecture: Instead of bundling heavy .csv or .pkl files with the code, this system uses a Remote Artifact Strategy. By fetching versioned models from Hugging Face, the repository stays lightweight and CI/CD-friendly.

    Log-Normalized Hybrid Scoring: Standard systems often suffer from "popularity bias" or "obscure results." This engine uses a logarithmic normalization of vote counts to ensure that popular movies get a fair boost without drowning out relevant niche content.

    Fuzzy Search Resolution: Most basic engines fail if the user makes a typo. This system implements a Scalable Fuzzy Matching layer, making the interface robust against human error and partial titles.

    Optimized Top-N Retrieval: Instead of a full sort() on the entire similarity matrix (which is O(nlogn)), it utilizes np.argpartition for a Linear Time O(n) partial sort, significantly reducing latency for the 500k+ movie dataset.

🚀 Upcoming Features & Roadmap

The project is currently evolving from a content-matching script into a production-ready engine. The following features are in active development:

    TMDB API Poster Integration: Fetching high-quality, real-time movie posters directly from The Movie Database (TMDB) to provide a visually rich user experience.

    Interactive Streamlit Frontend: Developing a web-based dashboard for real-time movie discovery, featuring dynamic search bars and responsive UI components.

    Sequential Recommendation Logic: Enhancing the engine to process lists of watched movies (user history) by calculating weighted average vectors for more personalized session-based output.

    Collaborative Filtering Integration: Incorporating Matrix Factorization (SVD) to capture "users who liked this also liked" patterns.

    Vector Database Migration: Transitioning similarity storage to ChromaDB or Pinecone to ensure O(1) search latency as the dataset scales.

👤 Author

Alok Singh BTech AI & ML, Amity University Gurugram
