# PetoPeta

This repo is an implementation of a ai Agent/Chatbot specifically focused on question answering over Pet and Animal.
Built with [LangChain](https://github.com/langchain-ai/langchain/), [LangGraph](https://github.com/langchain-ai/langgraph/), and [Next.js](https://nextjs.org).

The app leverages LangChain and LangGraph's streaming support and async API to update the page in real time for multiple users.

## Running locally
```bash
# for backend
langgraph dev

# for frontend
cd ./frontend
yarn dev
```

## Technical description

There are two components: ingestion and question-answering.

Ingestion has the following steps:

1. Load html with a modified version of LangChain's [SitemapLoader](https://python.langchain.com/docs/integrations/document_loaders/sitemap)
2. Split documents with LangChain's [RecursiveCharacterTextSplitter](https://api.python.langchain.com/en/latest/text_splitter/langchain.text_splitter.RecursiveCharacterTextSplitter.html)
3. Create a vectorstore of embeddings, using LangChain's [Postgres vectorstore wrapper](https://python.langchain.com/docs/integrations/vectorstores/pgvector) (with OpenAI's embeddings).

Question-Answering has the following steps:

1. Given the chat history and new user input, determine what a standalone question would be using an LLM.
2. Given that standalone question, look up relevant documents from the vectorstore.
3. Pass the standalone question and relevant documents to the model to generate and stream the final answer.
4. Generate a trace URL for the current chat session, as well as the endpoint to collect feedback.

## Documentation

- **[Concepts](./CONCEPTS.md)**: A conceptual overview of the different components of Petopeta. Goes over features like ingestion, vector stores, query analysis, etc.

## Preview

![PetoPeta Screenshot](./assets/images/screenshot1.png)
