from setuptools import setup, find_packages

setup(
    name="vecspark",
    version="0.1.0",
    description="A PySpark-based library for vector similarity and distance computations.",
    author="Aditya Bharadwaj, Vivek Garg",
    author_email="adityabharadwaj47@gmail.com, gargvivek2003@gmail.com",  
    url="https://github.com/Adi-ayro/vecspark",
    packages=find_packages(),
    install_requires=[
        "pyspark>=3.0.0",
        "numpy>=1.21.0",
        "tiktoken>=0.8.0",
        "PyPDF2>=3.0.1"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: Other/Proprietary License",  
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    license="VecSpark Business Developer License",  
)
