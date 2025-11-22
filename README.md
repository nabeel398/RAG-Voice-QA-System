# 🗣️ RAG + Voice QA API
### Retrieval-Augmented Generation with Speech-to-Text & ElevenLabs TTS (FastAPI)

This project provides an end-to-end **voice question-answering API** built with:

- **FastAPI**
- **FAISS** vector search
- **SentenceTransformers** embeddings
- **Groq Llama 3.1** for language reasoning
- **SpeechRecognition + pydub** for STT
- **ElevenLabs** for voice output (TTS)

Upload documents → Ask a question by audio → API retrieves context → Llama answers → ElevenLabs returns spoken audio.

---

## 🚀 Features

### 🔍 Retrieval-Augmented Generation (RAG)
- Upload **PDF, DOCX, PPTX, TXT** documents  
- Automatic chunking + storage in FAISS index  
- Query retrieves the most relevant chunks  

### 🎙️ Voice Input (Speech-to-Text)
- Accepts: `.wav`, `.mp3`, `.m4a`, `.flac`  
- Converts audio → transcribes using Google STT  

### 🤖 Llama 3.1 on Groq
- Fast, powerful model for answering questions  
- Uses document context only (true RAG behavior)  

### 🔊 ElevenLabs Text-to-Speech
- Converts LLM answer to speech  
- Streams back an MP3 response  
- Uses “Rachel” voice automatically if available  

---

## 📦 Tech Stack

| Component | Technology |
|----------|------------|
| Web Framework | FastAPI |
| Embeddings | SentenceTransformer (all-MiniLM-L6-v2) |
| Vector DB | FAISS |
| LLM | Groq API (Llama 3.1) |
| Speech-to-Text | SpeechRecognition + pydub |
| Text-to-Speech | ElevenLabs |
| Deployment | Uvicorn |

---

🔧 Setup & Installation
-----------------------

### 1\. Clone the repository

`   git clone https://github.com//.git`
`   cd <your-repo>` 

### 2\. Create & activate virtual environment

`   python3 -m venv venv  `
`   source venv/bin/activate   # Linux/macOS`
`   venv\Scripts\activate      # Windows   `

### 3\. Install dependencies

`   pip install -r requirements.txt   `

### 4\. Set environment variables

Create a .env file in the project root:

`   ELEVENLABS_API_KEY=your_elevenlabs_key_here<br>`
`   GROQ_API_KEY=your_groq_key_here   `

### 5\. Run the API

`   uvicorn main:app --host 0.0.0.0 --port 8000 --reload   `

---

## 🧠 How It Works (Pipeline)

### **1. Upload Document**
- Extract text from: **PDF / DOCX / PPTX / TXT**
- Split into overlapping **500-character chunks**
- Embed chunks using **SentenceTransformer**
- Store embeddings in **FAISS** for retrieval

---

### **2. Voice Question**
- Receive audio input  
- Convert to WAV when needed  
- Transcribe speech using **Google Speech Recognition**

---

### **3. RAG Query**
- Embed the user's question  
- Retrieve the top relevant chunks from FAISS  

---

### **4. LLM Answer (Groq)**
- **Llama 3.1** generates a concise answer based on retrieved context  
- If missing info → responds with:  
  *“I cannot find the answer in the provided documents.”*

---

### **5. ElevenLabs TTS**
- Convert the generated answer to natural speech  
- Stream back the MP3 audio response to the client  

---


