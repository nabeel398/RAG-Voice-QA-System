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

## 🔧 Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
