# VecSpark

VecSpark is a library designed to leverage the power of PySpark for handling vector embeddings at scale. It provides efficient methods for:

- **Calculating similarity scores** using various metrics.
- **Chunking large text** for database storage and further processing.

Built on top of PySpark, VecSpark enables seamless distributed computation of vector operations, making it ideal for large-scale NLP and analytics applications.

---

## Features

1. **Similarity Calculations**:
   - Supports multiple metrics, including Cosine, Euclidean, Manhattan, Minkowski, Pearson, Hamming, Bhattacharyya, and Chebyshev distances.
2. **Text Chunking**:
   - Breaks down text into manageable chunks, ready for database insertion or vector processing.

---

## Installation

```bash
pip install vecspark
```

---

## Usage

### 1. Initialize PySpark
```python
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("VecSpark Application") \
    .config("spark.master", "local[*]") \
    .getOrCreate()
```

### 2. Chunk Text from Files
```python
from vecspark import load_and_chunk_txt

# Chunk a text file into segments of 200 tokens
chunks = load_and_chunk_txt("sample.txt", max_tokens=200)

# Display chunks
for idx, chunk in enumerate(chunks):
    print(f"Chunk {idx + 1}: {chunk}")
```

### 3. Calculate Similarities
```python
spark = SparkSession.builder.appName("TopNSimilar").getOrCreate()

# Example DataFrame
data = [
    ("record1", [1.0, 2.0, 3.0]),
    ("record2", [4.0, 2.0, 1.0]),
    ("record3", [3.0, 5.0, 6.0])
]

schema = StructType([
    StructField("data", StringType(), True),
    StructField("features", ArrayType(FloatType()), True)
])

df = spark.createDataFrame(data, schema)

df.show()

vector = [2.0,3.0,6.0]

newdf = df.withColumn('Cosine', cosine(F.col('features'),F.lit(vector))).withColumn('Dot', dot(F.col('features'),F.lit(vector))).withColumn('Eucledian', euclidean(F.col('features'),F.lit(vector))).withColumn('Manhatten', manhattan(F.col('features'),F.lit(vector))).withColumn('Minkowski', minkowski(F.col('features'),F.lit(vector),F.lit(1))).withColumn('Pearson', pearson(F.col('features'),F.lit(vector))).withColumn('Hamming', hamming(F.col('features'),F.lit(vector))).withColumn('Bhattacharya', bhattacharyya(F.col('features'),F.lit(vector))).withColumn('Chebyshev', chebyshev(F.col('features'),F.lit(vector)))

newdf.show()

```

Output:

```bash 
Spark version: 3.5.3
+-------+---------------+
|   data|       features|
+-------+---------------+
|record1|[1.0, 2.0, 3.0]|
|record2|[4.0, 2.0, 1.0]|
|record3|[3.0, 5.0, 6.0]|
+-------+---------------+

+-------+---------------+----------+----+---------+---------+---------+----------+-------+------------+---------+
|   data|       features|    Cosine| Dot|Eucledian|Manhatten|Minkowski|   Pearson|Hamming|Bhattacharya|Chebyshev|
+-------+---------------+----------+----+---------+---------+---------+----------+-------+------------+---------+
|record1|[1.0, 2.0, 3.0]| 0.9926846|26.0|3.3166249|      5.0|      5.0|0.96076894|    3.0|   -2.092647|      3.0|
|record2|[4.0, 2.0, 1.0]|0.62347966|20.0| 5.477226|      8.0|      8.0|-0.8910421|    3.0|  -2.0447733|      5.0|
|record3|[3.0, 5.0, 6.0]| 0.9732576|57.0| 2.236068|      3.0|      3.0| 0.8910421|    2.0|  -2.5114248|      2.0|
+-------+---------------+----------+----+---------+---------+---------+----------+-------+------------+---------+
```

## Example

### Using in a GenAi fucntion with ollama
I am repeating the code from Ollama All-MiniLm Docs with just using PySpark and Vecspark instead of ChromaDB

```Python
import ollama
documents = [
  "Llamas are members of the camelid family meaning they're pretty closely related to vicuñas and camels",
  "Llamas were first domesticated and used as pack animals 4,000 to 5,000 years ago in the Peruvian highlands",
  "Llamas can grow as much as 6 feet tall though the average llama between 5 feet 6 inches and 5 feet 9 inches tall",
  "Llamas weigh between 280 and 450 pounds and can carry 25 to 30 percent of their body weight",
  "Llamas are vegetarians and have very efficient digestive systems",
  "Llamas live to be about 20 years old, though some only live for 15 years and others live to be 30 years old",
]

def embedding(doc):
    return ollama.embeddings(model="all-minilm",prompt=doc)["embedding"]

rows = [
    Row(document=[doc], embedding=embedding(doc)) for doc in documents
]

schema = StructType([
    StructField("document", StringType(), nullable=False),
    StructField("embedding", ArrayType(FloatType()), nullable=False)
])

sc = SparkSession.builder.appName("Example").config("spark.driver.memory", "4g").config("spark.executor.memory", "4g").getOrCreate()

df = sc.createDataFrame(rows, schema=schema)

## Data Stored in Vector DB
df.show()

prompt = "What animals are llamas related to?"

response = ollama.embeddings(
  prompt=prompt,
  model="all-minilm"
)["embedding"]


finalDB = df.withColumn("Cosine", cosine(F.col('embedding'), F.lit(response)))

finalDB.show()

data = finalDB.orderBy(F.col("Cosine").desc()).select("document").first()

print(data)

output = ollama.generate(
  model="nemotron-mini",
  prompt=f"Using this data: {data}. Respond to this prompt: {prompt}"
)

print(output['response'])
```
Output:
```bash
+--------------------+--------------------+
|            document|           embedding|
+--------------------+--------------------+
|[Llamas are membe...|[0.048099015, 0.1...|
|[Llamas were firs...|[0.17195219, 0.16...|
|[Llamas can grow ...|[0.009301755, 0.1...|
|[Llamas weigh bet...|[0.24956073, 0.17...|
|[Llamas are veget...|[0.3719531, -0.03...|
|[Llamas live to b...|[0.11673717, 0.16...|
+--------------------+--------------------+

+--------------------+--------------------+----------+
|            document|           embedding|    Cosine|
+--------------------+--------------------+----------+
|[Llamas are membe...|[0.048099015, 0.1...| 0.7713934|
|[Llamas were firs...|[0.17195219, 0.16...| 0.6688401|
|[Llamas can grow ...|[0.009301755, 0.1...| 0.5910876|
|[Llamas weigh bet...|[0.24956073, 0.17...|  0.654605|
|[Llamas are veget...|[0.3719531, -0.03...| 0.7110807|
|[Llamas live to b...|[0.11673717, 0.16...|0.66238075|
+--------------------+--------------------+----------+

Row(document="[Llamas are members of the camelid family meaning they're pretty closely related to vicuñas and camels]")
 Llamas are closely related to vicuñas and camels, as they belong to the same family.
```
---

## License

VecSpark is licensed under the [BDL License](LICENSE).

The License while a BDL is available for all uses with only exception of conversion into a managed Database as a Service. 
If you are intreseted in this use 

---