import os
import intel_extension_for_pytorch as intel_ipex
import torch
import argparse
import time
from langchain_community.embeddings import HuggingFaceEmbeddings, HuggingFaceInstructEmbeddings, HuggingFaceBgeEmbeddings
#from langchain.embeddings import HuggingFaceEmbeddings, HuggingFaceInstructEmbeddings
#from langchain.embeddings import HuggingFaceBgeEmbeddings, GooglePalmEmbeddings
#from langchain.vectorstores import Chroma
from langchain_community.vectorstores import Chroma, PGVector
#from langchain.vectorstores.pgvector import PGVector
from langchain.docstore.document import Document

def persist_embedding(documents, persist_directory, embeddings, batch_size):
    #try:
    #    if "instruct" in model_path:
    #        embeddings = HuggingFaceInstructEmbeddings(model_name=model_path)
    #    elif "bge" in model_path:
    #        embeddings = HuggingFaceBgeEmbeddings(
    #            model_name=model_path,
    #            encode_kwargs={'normalize_embeddings': True},
    #            query_instruction="Represent this sentence for searching relevant passages:")
    #    elif "Google" == model_path:
    #        embeddings = GooglePalmEmbeddings()
    #    else:
    #        embeddings = HuggingFaceEmbeddings(
    #            model_name=model_path,
    #            encode_kwargs={"normalize_embeddings": True},
    #        )
    #except:
    #    print("persist_embedding exception, please handle it.")
    print("batch_size", batch_size)
    vectordb = Chroma.from_documents(documents=documents,embedding=embeddings, persist_directory=persist_directory,
            collection_metadata={"hnsw:space": "cosine", "max_batch_size": batch_size})
    #time_start = time.time()
    vectordb.persist()
    #time_end = time.time()
    #print("vectordb persist cost {} seconds in total.".format(time_end - time_start), flush=True)
    vectordb = None

def prepare_sample_documents(num_instances):
    print("num_instances: ", num_instances)
    documents = [{
            "instruction": "If you are a doctor, \
                            please answer the medical questions based on the patient's description.",
            "input": "sir, MY uncle has ILD-Interstitial Lung disease.from my research \
                      over google i found that the cause is important to be known. the cause wat \
                      i feel is due to breakage of acid bottle in bathroom which caused lot of smoke \
                      and that time he wasnot able to be normal for 15-20 minutes. then he was normal .\
                      may be that would have been the reason for this ILD. Please suggest wat we should do",
            "output": "Thanks for your question on Chat Doctor. I can understand your concern. \
                       OLD (interstitial lung disease) can not be caused by single exposure of any chemical. \
                       Chronic exposure (for years) of chemicals can cause OLD. It is not possible to find out \
                       cause for each and every OLD patient. Some patients can develop OLD just because of aging \
                       (old age). So your uncle is not having OLD due to breakage of acid bottle. \
                       He might be having age related OLD. Better to start Perfinodone and N acetyl cysteine (NAC) \
                       for OLD rather than searching for cause. Hope I have solved your query. I will be happy to \
                       help you further. Wishing good health to your uncle. Thanks."
      }]
    sample_documents = documents * num_instances
    return sample_documents

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # embedding model: "BAAI/bge-base-en-v1.5" or "hkunlp/instructor-large"
    parser.add_argument('--embedding_model', type=str, help='Select which model to embed the content.',
                        default='sentence-transformers/paraphrase-multilingual-mpnet-base-v2')   #'BAAI/bge-small-en-v1.5')
    parser.add_argument('--output_path', type=str, help='Where to save the embedding.', default='./db/')
    parser.add_argument('--count', type=int, help='Embedding Model Samples in VectorDB.',
                        default=10000)
    parser.add_argument('--batch_size', type=int, help='Number of batch size.',
                        default=500)
    # vector database: "chroma" or "pgvector"
    parser.add_argument('--vector_database', type=str, help='The vector database type, could be chroma or pgvector',
                        default="chroma")
    # search type: "mmr" or "similarity_score_threshold"
    parser.add_argument('--search_type', type=str,
                        help='The retrieval type, could be mmr or similarity_score_threshold',
                        default="mmr")
    # if search type is "mmr", the search_kwargs should be {"k": 1, "fetch_k": 5}
    # if the search type is similarity_score_threshold, the search_kwargs should be {"score_threshold": 0.1, "k": 1}.
    parser.add_argument('--search_kwargs', type=dict, help='Set the number of the retrieval database.',
                        default={"k": 1, "fetch_k": 5})
    # the max length in the instance
    parser.add_argument('--max_length', type=int, help='The length in the instance.', default=512)
    args = parser.parse_args()

    sample_documents = prepare_sample_documents(args.count)
    print("phase 1")
    new_sens = []
    for sub in sample_documents:
        content = sub['instruction']+ sub['input'] + sub['output']
        if content=="":
            continue
        content = content[:512]
        new_sens.append(content)

    print("phase 2")
    time_start = time.time()
    documents = []
    for paragraph in new_sens:
        new_doc = Document(page_content=paragraph)
        documents.append(new_doc)
    time_end = time.time()
    print("phase 2 takes {}".format(time_end - time_start), flush=True)

    print("phase 3")

    iters = args.count // args.batch_size
    COLLECTION_NAME = "pg_stat_statements"
    CONNECTION_STRING = "postgresql://postgres:postgres@localhost:5432/mydb"
    CONNECTION1_STRING = "postgresql://postgres:postgres@localhost:5432/mydb1"
    CONNECTIONS_TABLE = {'BAAI/bge-small-en-v1.5': CONNECTION_STRING,
                         'BAAI/bge-base-en-v1.5': CONNECTION1_STRING}
    construct_db_time = 0
    retrieval_time = 0
    #if "instruct" in args.embedding_model:
    #    embeddings = HuggingFaceInstructEmbeddings(model_name=args.embedding_model)
    #elif "bge" in args.embedding_model:
    #    embeddings = HuggingFaceBgeEmbeddings(model_name = args.embedding_model, \
    #            encode_kwargs={'normalize_embeddings': True}, \
    #            query_instruction="Represent this sentence for searching relevant passages:")
    #try:
    model_path = args.embedding_model
    if "instruct" in model_path:
        embeddings = HuggingFaceInstructEmbeddings(model_name=model_path)
    elif "bge" in model_path:
        embeddings = HuggingFaceBgeEmbeddings(
            model_name=model_path,
            encode_kwargs={'normalize_embeddings': True},
            query_instruction="Represent this sentence for searching relevant passages:")
    elif "Google" == model_path:
        embeddings = GooglePalmEmbeddings()
    else:
        embeddings = HuggingFaceEmbeddings(
            model_name=model_path,
            encode_kwargs={"normalize_embeddings": True},
        )
    #except:
    #    print("persist_embedding exception, please handle it.")
    print("phase 4")

    embeddings.client = intel_ipex.optimize(embeddings.client.eval(), dtype=torch.bfloat16, inplace=True)
    with torch.cpu.amp.autocast(enabled=True, dtype=torch.bfloat16):
        if args.vector_database == "pgvector":
            print("Start to process the pgvector database... \n ", flush=True)
        else:
            print("Start to process the chroma database... \n ", flush=True)
        if args.vector_database == "pgvector":
            # construct database
            #for i in range(iters):
            time_start = time.time()
            db = PGVector.from_documents(
                embedding=embeddings,
                documents=documents,
                collection_name=COLLECTION_NAME,
                connection_string=CONNECTIONS_TABLE[args.embedding_model],
            )
            time_end = time.time()
            print("The time for construct the pgvector database cost {} seconds".\
                format(time_end-time_start, flush=True))
            # retrieval
            retriever = db.as_retriever(search_type=args.search_type, search_kwargs=args.search_kwargs)
            for i in range(10000):
                time_start = time.time()
                docs = retriever.get_relevant_documents(
                    "If you are a doctor, please answer the medical questions based on the patient's description.")
                time_end = time.time()
                retrieval_time += time_end-time_start
                if i % 200 == 0:
                  print("The retrieval for pgvector cost {} seconds for iterarion {}".\
                      format(time_end-time_start, i+1), flush=True)
            print("The retrieval for pgvector cost {} seconds in average.".format(retrieval_time/10000), flush=True)
        else:
            # construct database
            #for i in range(iters):
            output_path = os.path.join(args.output_path, str(time.time()).replace('.', ''))
            import sys
            print("Start the embedding, the number of the instance in the database is {} with size {}.".format(
                 len(sample_documents), sys.getsizeof(sample_documents)), flush=True)
            time_start = time.time()
            persist_embedding(documents, output_path, embeddings, args.batch_size)
            time_end = time.time()
            print("The time for construct the chroma database cost {} seconds".\
                format(time_end-time_start), flush=True)

            # retrieval
            vectordb = Chroma(persist_directory=args.output_path, embedding_function=embeddings,
                              collection_metadata={"hnsw:space": "cosine"})
            retriever = vectordb.as_retriever(search_type=args.search_type, search_kwargs=args.search_kwargs)
            for i in range(10000):
                time_start = time.time()
                docs = retriever.get_relevant_documents("If you are a doctor, \
                    please answer the medical questions based on the patient's description.")
                time_end = time.time()
                retrieval_time += time_end-time_start
                if i % 200 == 0:
                    print("The retrieval for chroma cost {} seconds for iterarion {}".\
                        format(time_end-time_start, i+1), flush=True)
            print("The retrieval for chroma cost {} seconds in average.".format(retrieval_time/10000), flush=True)



