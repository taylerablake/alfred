import datetime
from langchain.document_loaders import ReadTheDocsLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import openai
import pinecone
import tiktoken
from time import sleep
from tqdm.auto import tqdm
from uuid import uuid4


class SmartAssistant:
    def __init__(self, openai_api_key: str, pinecone_api_key: str, pinecone_env: str) -> None:
        self.openai_api_key = openai_api_key
        self.openai_pinecone_key = pinecone_api_key
        self.pinecone_env = pinecone_env

    def _load_docs(self, docs_path: str):
        loader = ReadTheDocsLoader('rtdocs')
        docs = loader.load()
        data = []

        for doc in docs:
            data.append({
                'url': doc.metadata['source'].replace('rtdocs/', 'https://'),
                'text': doc.page_content
            })
        return data

    def tiktoken_len(self, text, encoding_name='p50k_base'):

        tokenizer = tiktoken.get_encoding(encoding_name)
        tokens = tokenizer.encode(
            text,
            disallowed_special=()
        )
        return len(tokens)

    def _process_data(self, data: list):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=20,
            length_function=self.tiktoken_len,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = []

        for idx, record in enumerate(tqdm(data)):
            texts = text_splitter.split_text(record['text'])
        chunks.extend([{
            'id': str(uuid4()),
            'text': texts[i],
            'chunk': i,
            'url': record['url']
        } for i in range(len(texts))])
        return chunks


    def _initialize_index(self) :
        openai.api_key = self.openai_api_key

        embed_model = "text-embedding-3-small"
        res = openai.Embedding.create(
            input=[
                "Sample document text goes here",
                "there will be several phrases in each batch"
            ], engine=embed_model
        )
        index_name = 'gpt-4-langchain-docs'

        # initialize connection to pinecone
        pinecone.init(
            api_key=self.pinecone_api_key,  # app.pinecone.io (console)
            environment=self.pinecone_env  # next to API key in console
        )

        # check if index already exists (it shouldn't if this is first time)
        if index_name not in pinecone.list_indexes():
            # if does not exist, create index
            pinecone.create_index(
                index_name,
                dimension=len(res['data'][0]['embedding']),
                metric='dotproduct'
            )
        # connect to index
        index = pinecone.GRPCIndex(index_name)
        return index

    def encode_docs(self, docs_path: str, embed_model='text-embedding-3-small') -> bool:

        docs = self._load_docs(docs_path)
        data = self._process_data(docs)
        pinecone_index = self._initialize_index()

        batch_size = 100  # how many embeddings we create and insert at once

        for i in tqdm(range(0, len(data), batch_size)):
            # find end of batch
            i_end = min(len(data), i+batch_size)
            meta_batch = data[i:i_end]
            # get ids
            ids_batch = [x['id'] for x in meta_batch]
            # get texts to encode
            texts = [x['text'] for x in meta_batch]
            # create embeddings (try-except added to avoid RateLimitError)
            try:
                res = openai.Embedding.create(input=texts, engine=embed_model)
            except:
                done = False
                while not done:
                    sleep(5)
                    try:
                        res = openai.Embedding.create(input=texts, engine=embed_model)
                        done = True
                    except:
                        pass
            embeds = [record['embedding'] for record in res['data']]
            # cleanup metadata
            meta_batch = [{
                'text': x['text'],
                'chunk': x['chunk'],
                'url': x['url']
            } for x in meta_batch]
            to_upsert = list(zip(ids_batch, embeds, meta_batch))
            # upsert to Pinecone
            pinecone_index.upsert(vectors=to_upsert)
            self.pinecone_index = pinecone_index




