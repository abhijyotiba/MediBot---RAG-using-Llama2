# 🩺 MediBot – Medical Chatbot using RAG + LLaMA2

MediBot is an AI-powered medical chatbot built using a **Retrieval-Augmented Generation (RAG)** pipeline with **LLaMA2**, **LangChain**, and **Pinecone**. It allows users to ask medical questions and get contextually accurate answers sourced from a medical book (PDF).

---

## 💡 Features

- 🔍 Vector-based search with **Pinecone**
- 🧠 Embedding generation using **HuggingFace Embeddings**
- 📄 PDF document ingestion & chunking
- 🤖 Language generation using **LLaMA2** via `ctransformers`
- 🌐 Flask web interface (chat UI)
- 🧑‍⚕️ Tailored for medical Q&A

---


### Workflow
<p align="center">
  <img src="https://i.postimg.cc/63FZ932s/Editor-Mermaid-Chart-2025-06-12-142525.png" width="500"/>
</p>


## 🛠️ Tech Stack

- **LangChain**
- **LLaMA2 (CTransformers)**
- **HuggingFace Sentence Transformers**
- **Pinecone** (Vector DB)
- **Flask** (Backend)
- **HTML/CSS/JS** (Frontend)

---


---

## ⚙️ Setup Instructions

### 1. Clone the Repo
```bash
git clone https://github.com/abhijyotiba/MediBot---RAG-using-Llama2.git
```
### 2. Create and Activate virtual environment
```bash
python -m venv venv
venv\Scripts\activate     # On Windows
# source venv/bin/activate   # On Linux/macOS
```
### 3. install Requirenments
```bash
pip install -r requirements.txt
```

### 4. Replace Pinecone API in all the files
```bash
trails.py 
app.py
store.py
```

### 5. Generate Vector store
```bash
python store.py
```

### 6. Run the App
```bash
python app.py
```

Then visit http://127.0.0.1:8080 to chat!

<p align="center">
  <img src="https://i.postimg.cc/8chm9WyX/Screenshot-2025-06-20-004114.png" width="700"/>
</p>
