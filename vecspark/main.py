import pyspark.sql.functions as spark_func
import numpy as np
from pyspark.sql.types import  FloatType
from PyPDF2 import PdfReader
import tiktoken
import csv

@spark_func.udf(returnType=FloatType())
def cosine(v1, v2) -> float:
    """
    Computes the Cosine Distance

    :param v1: vector1
    :param v2: vector2
    """
    vec1 = np.array(v1)  
    vec2 = np.array(v2)
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return float(dot_product / (norm1 * norm2))

@spark_func.udf(returnType=FloatType())
def dot(v1, v2) -> float:
    """
    Computes the Dot Product

    :param v1: vector1
    :param v2: vector2
    """
    vec1 = np.array(v1)  
    vec2 = np.array(v2)
    return float(np.dot(vec1, vec2))

@spark_func.udf(returnType=FloatType())
def manhattan(v1, v2) -> float:
    """
    Computes the Manhatten Distance

    :param v1: vector1
    :param v2: vector2
    """
    vec1 = np.array(v1)  
    vec2 = np.array(v2)
    return float(np.sum(np.abs(vec1 - vec2)))

@spark_func.udf(returnType=FloatType())
def euclidean(v1, v2) -> float:
    """
    Computes the Euclidean Distance

    :param v1: vector1
    :param v2: vector2
    """
    vec1 = np.array(v1)  
    vec2 = np.array(v2)
    return float(np.sqrt(np.sum((vec1 - vec2) ** 2)))

@spark_func.udf(returnType=FloatType())
def chebyshev(v1, v2) -> float:
    """
    Computes the Chebyshev Distance

    :param v1: vector1
    :param v2: vector2
    """
    vec1 = np.array(v1)
    vec2 = np.array(v2)
    return float(np.max(np.abs(vec1 - vec2)))

@spark_func.udf(returnType=FloatType())
def bhattacharyya(v1, v2) -> float:
    """
    Computes the Bhattacharyya Distance

    :param v1: vector1
    :param v2: vector2
    """
    vec1 = np.array(v1)
    vec2 = np.array(v2)
    return float(-np.log(np.sum(np.sqrt(vec1 * vec2))))

@spark_func.udf(returnType=FloatType())
def hamming(v1, v2) -> float:
    """
    Computes the Hamming Distance

    :param v1: vector1
    :param v2: vector2
    """
    vec1 = np.array(v1)
    vec2 = np.array(v2)
    return float(np.sum(vec1 != vec2))

@spark_func.udf(returnType=FloatType())
def minkowski(v1, v2, p: int) -> float:
    """
    Computes the Minkowski Distance

    :param v1: vector1
    :param v2: vector2
    :param p: norm parameter for minkowski
    """
    vec1 = np.array(v1)
    vec2 = np.array(v2)
    return float(np.sum(np.abs(vec1 - vec2) ** p) ** (1 / p))

@spark_func.udf(returnType=FloatType())
def pearson(v1, v2) -> float:
    """
    Computes the Pearson Distance

    :param v1: vector1
    :param v2: vector2
    """
    vec1 = np.array(v1)
    vec2 = np.array(v2)
    return float(np.corrcoef(vec1, vec2)[0, 1])

def load_and_chunk_pdf(file_path, max_tokens=8191, model="cl100k_base"):
    """
    Load a PDF file, process its content, and split it into chunks suitable for vector database embeddings.

    :param file_path: Path to the PDF file.
    :param max_tokens: Maximum number of tokens per chunk (default: 8191 for OpenAI's text-embedding-ada-002).
    :param model: Tokenizer model (default: "cl100k_base" for OpenAI).
    :return: A list of text chunks, each within the max token limit.
    """
    try:
        # Read the PDF content
        reader = PdfReader(file_path)
        content = ""
        for page in reader.pages:
            content += page.extract_text() + "\n"

        print(f"PDF loaded successfully. Total length: {len(content)} characters.")
        
        # Initialize the tokenizer
        tokenizer = tiktoken.get_encoding(model)
        
        # Split text into chunks
        words = content.split()
        chunks = []
        current_chunk = []
        current_token_count = 0

        for word in words:
            token_count = len(tokenizer.encode(word))
            if current_token_count + token_count > max_tokens:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_token_count = 0
            current_chunk.append(word)
            current_token_count += token_count

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        print(f"Content divided into {len(chunks)} chunks, each fitting within {max_tokens} tokens.")
        return chunks

    except Exception as e:
        print(f"Error processing the file: {e}")
        return None

def load_and_chunk_csv(file_path, max_tokens=8191, model="cl100k_base"):
    """
    Load a CSV file, process its content, and split it into chunks suitable for vector database embeddings.

    :param file_path: Path to the CSV file.
    :param max_tokens: Maximum number of tokens per chunk (default: 8191 for OpenAI's text-embedding-ada-002).
    :param model: Tokenizer model (default: "cl100k_base" for OpenAI).
    :return: A list of text chunks, each within the max token limit.
    """
    try:
        # Read the CSV content
        content = ""
        with open(file_path, mode="r", encoding="utf-8") as file:
            reader = csv.reader(file)
            for row in reader:
                content += ", ".join(row) + "\n"

        print(f"CSV loaded successfully. Total length: {len(content)} characters.")

        # Initialize the tokenizer
        tokenizer = tiktoken.get_encoding(model)
        
        # Split text into chunks
        words = content.split()
        chunks = []
        current_chunk = []
        current_token_count = 0

        for word in words:
            token_count = len(tokenizer.encode(word))
            if current_token_count + token_count > max_tokens:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_token_count = 0
            current_chunk.append(word)
            current_token_count += token_count

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        print(f"Content divided into {len(chunks)} chunks, each fitting within {max_tokens} tokens.")
        return chunks

    except Exception as e:
        print(f"Error processing the file: {e}")
        return None


def load_and_chunk_txt(file_path, max_tokens=8191, model="cl100k_base"):
    """
    Load a TXT file, process its content, and split it into chunks suitable for vector database embeddings.

    :param file_path: Path to the TXT file.
    :param max_tokens: Maximum number of tokens per chunk (default: 8191 for OpenAI's text-embedding-ada-002).
    :param model: Tokenizer model (default: "cl100k_base" for OpenAI).
    :return: A list of text chunks, each within the max token limit.
    """
    try:
        # Read the TXT content
        with open(file_path, mode="r", encoding="utf-8") as file:
            content = file.read()

        print(f"TXT file loaded successfully. Total length: {len(content)} characters.")
        
        # Initialize the tokenizer
        tokenizer = tiktoken.get_encoding(model)
        
        # Split text into chunks
        words = content.split()
        chunks = []
        current_chunk = []
        current_token_count = 0

        for word in words:
            token_count = len(tokenizer.encode(word))
            if current_token_count + token_count > max_tokens:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_token_count = 0
            current_chunk.append(word)
            current_token_count += token_count

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        print(f"Content divided into {len(chunks)} chunks, each fitting within {max_tokens} tokens.")
        return chunks

    except Exception as e:
        print(f"Error processing the file: {e}")
        return None
