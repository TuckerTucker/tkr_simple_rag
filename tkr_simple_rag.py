import os
import numpy as np
from tkr_utils.helper_openai import OpenAIHelper
from tkr_utils import setup_logging
from sentence_transformers import SentenceTransformer

# Disable tokenizers parallelism to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Create the logger using the filename
logger = setup_logging(__file__)

# Define your document
# read from a file on disk to get your specific content
doc = """
This is the first paragraph of your document.
It can contain multiple lines.

This is the second paragraph. It also has multiple lines.

This is the third paragraph.
"""

# Split the document into paragraphs
paragraphs = doc.split('\n\n')  # Adjust the delimiter as needed

# Load the sentence transformer model with force download
model = SentenceTransformer('Alibaba-NLP/gte-base-en-v1.5', trust_remote_code=True)

# Encode the paragraphs into embeddings
docs_embed = model.encode(paragraphs, normalize_embeddings=True)

# Define your query
query = "what is the topic of the second paragraph?"
query_embed = model.encode(query, normalize_embeddings=True)

# Calculate similarities between the document embeddings and the query embedding
similarities = np.dot(docs_embed, query_embed.T)

# Get the indices (vector embeddings) of the 3 paragraphs most similar to the query
top_3_idx = np.argsort(similarities, axis=0)[-3:][::-1].tolist()

# Extract the most similar paragraphs
most_similar_documents = [paragraphs[idx] for idx in top_3_idx]

# Create an OpenAI client
client = OpenAIHelper()

# Define the prompt based on the most similar documents
prompt = [{"role": "system", "content": "You are a helpful assistant."}] + [{"role": "user", "content":query}]+[{"role": "user", "content":doc} for doc in most_similar_documents]

# Send a chat completion request with the prompt
response = client.send_message(prompt)

# Print the response to the terminal
logger.info(response.choices[0].message.content)
logger.info(response)
