import requests

def get_response(prompt):
    payload = {
        "model": "mistral",
        "prompt": prompt,
        "stream": False
    }
    response = requests.post("http://localhost:11434/api/generate", json=payload)
    response.raise_for_status()
    return response.json()["response"]

def get_chat_completion(system_prompt, user_prompt, context=""):
    final_prompt = f"{system_prompt}\n\nContext:\n{context}\n\nUser:\n{user_prompt}"
    payload = {
        "model": "mistral",
        "prompt": final_prompt,
        "stream": False
    }
    response = requests.post("http://localhost:11434/api/generate", json=payload)
    response.raise_for_status()
    return response.json()["response"]

if __name__ == "__main__":
    print(get_response("Explain why our product is better than the competition."))