from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from duckduckgo_search import DDGS

load_dotenv()


def search_duckduckgo(query):
    results = []

    try:
        with DDGS() as ddgs:
            for result in ddgs.text(query, max_results=3):
                if "body" in result:
                    results.append(result["body"])

        return "\n".join(results)

    except Exception as e:
        print(f"Error during DuckDuckGo search: {e}")
        return ""


def rag_chain(url):
    try:
        loader = WebBaseLoader(url)
        docs = loader.load()

        if not docs:
            return None

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        texts = text_splitter.split_documents(docs)

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        vectorstore = FAISS.from_documents(texts, embeddings)

        retriever = vectorstore.as_retriever(
            search_kwargs={"k": 5}
        )

        return retriever

    except Exception as e:
        print(f"Error in rag_chain: {e}")
        return None