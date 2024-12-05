from flask import Flask, render_template, request
from langchain_ollama import OllamaLLM
from langchain.prompts import ChatPromptTemplate

app = Flask(__name__)

# Initialize the LLM and prompt template
model = OllamaLLM(model="llama3.2")
template = """Answer the question below.

Here is the conversation so far:
{history}

question: {question}
answer :"""
prompt = ChatPromptTemplate.from_template(template)
llm_chain = prompt | model

conversation_history = []

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_message = request.form['user_input']
        conversation_history.append(("User:", user_message))
        render_template('index.html', messages=conversation_history)

        llm_response = llm_chain.invoke({"history": conversation_history , "question": user_message})
        conversation_history.append(("AI:", llm_response))

    return render_template('index.html', messages=conversation_history)

if __name__ == '__main__':
    app.run(debug=True)