from flask import Flask, render_template, jsonify, request
from src.helper import load_embedding_model
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
from langchain.chains import LLMChain
from pinecone import Pinecone
import os

app = Flask(__name__)

PINECONE_API_KEY ='pcsk_2U7yy5_HDTJQWg9WaRgBXi6crVgEk8tzF8oLtEhUeKeAr1rToQNfJk6kxEtiu4FrNnfxxx'

embeddings = load_embedding_model()

#Initializing the Pinecone

pc = Pinecone(api_key=PINECONE_API_KEY)

index = pc.Index("chatbot2-test")


# === LLM Setup ===
llm = CTransformers(
    model="models/llama-2-7b-chat.ggmlv3.q4_0.bin",
    model_type="llama",
    config={
        "max_new_tokens": 200,
        "temperature": 0.7,
        "top_p": 0.95,
        "repetition_penalty": 1.1
    }
)

# === Prompt Template ===
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a helpful medical assistant. Using the context below, give a clear and concise answer to the question.
If the answer is not found in the context, reply: "Sorry, I couldn't find an answer."

Context:
{context}

Question:
{question}
"""
)

qa_chain = prompt_template | llm


# === Search Pinecone ===
def search_pinecone(query, index, embedding_model, top_k=2):
    query_vector = embedding_model.embed_query(query)
    results = index.query(vector=query_vector, top_k=top_k, include_metadata=True)
    return results


# === Build Prompt & Generate Answer ===
def get_answer(user_query):
    search_results = search_pinecone(user_query, index, embeddings)
    contexts = [match['metadata'].get("text", "") for match in search_results["matches"]]
    context_text = "\n\n".join(contexts)
    prompt = prompt_template.format(context=context_text, question=user_query)
    answer = qa_chain.invoke({"context": context_text, "question": user_query})
    return answer


# === Routes ===
@app.route("/")
def home():
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.values.get("msg")  # works for both GET and POST
    if not msg:
        return jsonify({"error": "No message provided"}), 400

    try:
        answer = get_answer(msg)
        return jsonify({"answer": answer})
    except Exception as e:
        print("❌ ERROR:", e)
        return jsonify({"error": str(e)}), 500
    

# === Run App ===
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)    