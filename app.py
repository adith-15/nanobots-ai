from llm.chat import get_chat_completion
from rag.retriever import query_index
import subprocess

SYSTEM_PROMPT = """You are an expert sales assistant.
Use the provided context to answer the customer's question clearly and persuasively.
If you don't know the answer, say so honestly.
"""
def record_audio_subprocess():
    subprocess.run(["python", "speech/stt.py", "--record_only"])

def main():
    print("\n==============================")
    print("      Nanobot - Your Sales Expert   ")
    print("==============================")
    print("Type 'speak' to use voice input.")
    print("Type 'exit' or 'quit' to end.\n")
    print("---------------------")
    while True:
        user_input = input("Your input (or 'speak'): ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break
        
        if user_input.lower() == "speak":
            print("Recording audio...")
            result = subprocess.run(
                ["python", "speech/stt.py"],
                capture_output=True,
                text=True
            )
            user_query = result.stdout.strip()
            print(f"\nTranscribed: {user_query}\n")

        else:
            user_query = user_input

        if not user_query:
            start_response = "Hi there, Go ahead and ask me your question. Let me help you with that."
            subprocess.run(["python", "speech/tts_runner.py", start_response])
            continue

        # Retrieve relevant chunks
        retrievals = query_index(user_query, top_k=3)
        context_texts = [r["text"] for r in retrievals]
        context_combined = "\n\n".join(context_texts)

        # Get response from LLM
        response = get_chat_completion(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_query,
            context=context_combined
        )

    
        subprocess.run(["python", "speech/tts_runner.py", response])

if __name__ == "__main__":
    main()