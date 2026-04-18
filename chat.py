import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    print("ERROR: GROQ_API_KEY not found.")
    print("Make sure you have a .env file with: GROQ_API_KEY=your_key_here")
    exit(1)

client = Groq(api_key=GROQ_API_KEY)

SYSTEM_PROMPT = """You are GitWhisper, an AI assistant that helps 
developers understand code and repositories. Be helpful, 
concise, and technical. If you don't know something, say so."""


history = []


def chat(user_message):
    """Send a message and get a reply. History is updated automatically."""

    history.append({
        "role": "user",
        "content": user_message
    })

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            *history   
        ],
        temperature=0.7,  
        max_tokens=1024   
    )

    
    reply = response.choices[0].message.content

    
    history.append({
        "role": "assistant",
        "content": reply
    })

    return reply


def main():
    print()
    print("=" * 50)
    print("  GitWhisper — AI Chat")
    print("  Type 'exit' to quit")
    print("  Type 'clear' to reset conversation")
    print("=" * 50)
    print()

    while True:
        try:
            user_input = input("You: ").strip()
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        if user_input.lower() == "clear":
            history.clear()
            print("Conversation cleared.\n")
            continue

        try:
            reply = chat(user_input)
            print(f"\nGitWhisper: {reply}\n")
        except Exception as e:
            print(f"\nError: {e}\n")


if __name__ == "__main__":
    main()