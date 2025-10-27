"""Load html from files, clean up, split, ingest into Weaviate."""

import logging
import os

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_postgres import PGVectorStore, PGEngine
from langchain_postgres.v2.indexes import IVFFlatIndex, HNSWIndex

from backend.avma_sitemaploader import AVMASitemapLoader

from backend.embeddings import get_embeddings_model
from backend.parser import avma_docs_extractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# def metadata_extractor(
#     meta: dict, soup: BeautifulSoup, title_suffix: Optional[str] = None
# ) -> dict:
#     title_element = soup.find("title")
#     description_element = soup.find("meta", attrs={"name": "description"})
#     html_element = soup.find("html")
#     title = title_element.get_text() if title_element else ""
#     if title_suffix is not None:
#         title += title_suffix

#     return {
#         "source": meta["loc"],
#         "title": title,
#         "description": description_element.get("content", "")
#         if description_element
#         else "",
#         "language": html_element.get("lang", "") if html_element else "",
#         **meta,
#     }


def load_avma_docs():
    return AVMASitemapLoader(
        "https://avmajournals.avma.org/sitemap.xml",
        filter_urls=[r"^https?://[^/]*\.avma\.org/.*\.xml$"],
        parsing_function=avma_docs_extractor,
        default_parser="lxml",
        # bs_kwargs={
        #     "parse_only": SoupStrainer(
        #         name=("article", "title", "html", "lang", "content")
        #     ),
        # },
        # meta_function=metadata_extractor,
    ).load()


def ingest_docs():
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    embedding = get_embeddings_model()
    embedding_dimensions = 1536
    index_name = "petopeta-hnsw-index"

    table_name = os.environ["VECTOR_TABLE_NAME"]
    pg_engine = PGEngine.from_connection_string(url=os.environ["VECTOR_DB_URL"])
    # pg_engine.init_vectorstore_table(table_name=table_name, vector_size=embedding_dimensions)

    vectorstore = PGVectorStore.create_sync(
        engine=pg_engine,
        table_name=table_name,
        embedding_service=embedding,
    )
    # vectorstore.apply_vector_index(HNSWIndex(name=index_name))

    docs_from_documentation = load_avma_docs()
    logger.info(f"Loaded {len(docs_from_documentation)} docs from documentation")

    docs_transformed = text_splitter.split_documents(docs_from_documentation)
    docs_transformed = [doc for doc in docs_transformed if len(doc.page_content) > 10]

    # We try to return 'source' and 'title' metadata when querying vector store and
    # Weaviate will error at query time if one of the attributes is missing from a
    # retrieved document.
    for doc in docs_transformed:
        if "source" not in doc.metadata:
            doc.metadata["source"] = ""
        if "title" not in doc.metadata:
            doc.metadata["title"] = ""

    logger.info(f"About to add docs of {len(docs_transformed)}")

    batch_size = 256
    for i in range(int(len(docs_transformed) / batch_size) + 1):
        logger.info(f"adding docs from {batch_size * i} to {batch_size * (i + 1)}")
        vectorstore.add_documents(
            docs_transformed[batch_size * i : batch_size * (i + 1)]
        )

    vectorstore.reindex(index_name)


if __name__ == "__main__":
    ingest_docs()
