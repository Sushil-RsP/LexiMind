<<<<<<< HEAD
🧠 LexiMind: Legal Document Analysis Engine

[Download 1950_data.zip](https://drive.google.com/file/d/1srTT31jDVDEEqya8iFTNAB67MVS7DZz-/view?usp=sharing)

LexiMind is a powerful web application built with Flask that performs intelligent analysis on legal documents. Users can either paste text or upload a PDF of a legal judgment to receive an AI-powered classification and find semantically similar cases from a database.

The application is designed to be efficient, processing files in memory and loading machine learning models on startup from Google Drive.

🌟 Key Features

Dual Input: Accepts direct text input or PDF file uploads.

PDF Text Extraction: Reliably extracts text from uploaded PDF documents without saving them to disk.

Document Classification: Automatically predicts the legal category of the submitted document using a pre-trained model.

Semantic Search: Finds the top 5 most relevant existing cases by comparing meaning, not just keywords.

On-Demand Model Loading: Downloads and loads all necessary model files from Google Drive on the first run.

Privacy Focused: Processes data in memory and includes cleanup steps to remove sensitive text after analysis.

💻 Tech Stack & Libraries

This project combines a web backend with modern natural language processing libraries:

Backend: Flask

Machine Learning:

sentence-transformers: For generating text embeddings and performing semantic search.

scikit-learn (joblib): For loading the classification model.

Data Handling: Pandas, NumPy

PDF Processing: pdfplumber, PyPDF2

Model Downloading: gdown for fetching files from Google Drive.

🚀 How to Run Locally

Follow these instructions to get the LexiMind web application running on your machine.

1. Clone the Repository

git clone [https://github.com/Sushil-RsP/LexiMind.git](https://github.com/Sushil-RsP/LexiMind.git)
cd LexiMind


2. Create and Activate a Virtual Environment
It's highly recommended to use a virtual environment to manage dependencies.

# Create the environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate


3. Install Dependencies
All the required packages are listed in the requirements.txt file. Install them using the following command:

pip install -r requirements.txt


4. Run the Application
Execute the main script from your terminal:

python app.py


Important: The very first time you run the app, it will download several large model files from Google Drive. This may take a few minutes depending on your internet connection. Subsequent startups will be much faster as the files will be cached locally.

5. Open in Browser
Once the server is running, open your web browser and navigate to http://127.0.0.1:7860/.

📁 File Structure

app.py: The core Flask application. It contains all the routes, data processing logic, and prediction functions.

templates/index.html: (Implied) The HTML file that renders the user interface for input and results.

requirements.txt: A file listing all the Python dependencies for the project.

Downloaded Files: Upon first run, the following files will be downloaded into your project directory:

judgment_texts.pkl: The text data of existing legal cases.

model.pkl: The Sentence Transformer model for embedding.

case_names.pkl: The names corresponding to the legal cases.

embeddings.pkl: Pre-computed embeddings of the judgment texts.

modellog.pkl: The pre-trained classification model.

# You can download all pkl files by run model.ipynb but first you need to dowanload database from my drive

