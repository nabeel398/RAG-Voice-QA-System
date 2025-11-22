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

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`git clone https://github.com//.git  cd` 

### 2\. Create & activate virtual environment

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   python3 -m venv venv  source venv/bin/activate   # Linux/macOS  venv\Scripts\activate      # Windows   `

### 3\. Install dependencies

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   pip install -r requirements.txt   `

### 4\. Set environment variables

Create a .env file in the project root:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   ELEVENLABS_API_KEY=your_elevenlabs_key_here  GROQ_API_KEY=your_groq_key_here   `

### 5\. Run the API

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   uvicorn main:app --host 0.0.0.0 --port 8000 --reload   `
