from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="llama3.1:8b",
    temperature=0.0,

)

messages = [
    ("system", "You are a helpful assistant that translates from English to French. Translate the user sentence into French."),
    ("user", "I am hoping to build a chatbot that can help me with my homework.")
]

response = llm.invoke(messages)

print(response.content)