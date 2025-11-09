# Localized voice-interactive assistant
### https://nanobots-ai.com/ Powered by open-source LLMs, custom data ingestion (RAG), and expressive text-to-speech (TTS) + speech-to-text (STT) pipelines. It can be trained on business documents and deployed to engage users through natural voice conversations.
#### Structure
voice-salesbot/  
├── app.py # Main voice-chat loop  
├── .env # Environment variables  
├── requirements.txt # All required packages    
├── venv/ # Python virtual environment    
├── data/ # Folder to store uploaded PDFs/texts  
├── llm/  
│ ├── chat.py # Wrapper for local LLM (Ollama Mistral)  
│ └── prompts.py # System/user prompts  
├── rag/  
│ ├── ingest.py # Load, chunk, and embed files  
│ └── retriever.py # Vector store querying  
├── speech/  
│ ├── stt.py # Voice recording + Whisper STT  
│ ├── tts.py # Legacy TTS (Tacotron2)  
│ └── tts_runner.py # CLI script to speak response  

