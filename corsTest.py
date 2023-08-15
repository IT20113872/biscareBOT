from flask import Flask, request, jsonify
from flask_cors import CORS
from llama_index import SimpleDirectoryReader, GPTListIndex, readers, GPTSimpleVectorIndex, LLMPredictor, PromptHelper, ServiceContext
from langchain import OpenAI
import logging
import sys
import os
from IPython.display import Markdown, display

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Set the log level for the llama_index module to ERROR
logging.getLogger("llama_index").setLevel(logging.ERROR)
# api = "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"
# api = "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"
api = "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"
os.environ["OPENAI_API_KEY"] = api


def construct_index(directory_path):
    # set maximum input size
    max_input_size = 4096
    num_outputs = 2000
    max_chunk_overlap = 20
    chunk_size_limit = 600

    prompt_helper = PromptHelper(
        max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

    llm_predictor = LLMPredictor(llm=OpenAI(
        temperature=0.5, model_name="text-davinci-003", max_tokens=num_outputs))

    documents = SimpleDirectoryReader(directory_path).load_data()

    service_context = ServiceContext.from_defaults(
        llm_predictor=llm_predictor, prompt_helper=prompt_helper)
    index = GPTSimpleVectorIndex.from_documents(
        documents, service_context=service_context)

    index.save_to_disk('index.json')

    return index
 

@app.route('/')
def hello_world():
    return 'Ask Any Question..............'


@app.route('/<question>')
def ask_ai(question):
    index = GPTSimpleVectorIndex.load_from_disk('index.json')
    while True:
        query = question
        try: 
            response = index.query(query)
            print(response)
            # Wrap the response in a JSON object
            return jsonify({"response": response})
        except Exception as e:
            print(f"Error occurred: {e}")


if __name__ == '__main__':
    # Construct index before running the app
    construct_index("context_data-main\data")
    app.run()
