from langchain_community.document_loaders import UnstructuredFileLoader, TextLoader, DirectoryLoader
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from LoadFile import load_file
from langchain_community.embeddings import QianfanEmbeddingsEndpoint
from config import qianfan_ak,qianfan_sk,model_embed,endpoint_embed
from langchain_community.embeddings import ModelScopeEmbeddings
from langchain_community.vectorstores import Chroma


class DocumentService(object):
    def __init__(self,docs_path,vector_store_path):

        self.embeddings = QianfanEmbeddingsEndpoint(
        qianfan_ak=qianfan_ak,
        qianfan_sk=qianfan_sk,
        model=model_embed, endpoint=endpoint_embed
        )

        # self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_model_dir)
        # self.embeddings = embedding
        self.docs_path = docs_path
        self.vector_store_path = vector_store_path
        # self.vector_store = None

    def init_source_vector(self):
        """
        初始化本地知识库向量
        :return:
        """
        split_text = load_file(self.docs_path)
        print(split_text)
        db = Chroma.from_documents(
            documents=split_text, 
            embedding=self.embeddings, 
            persist_directory=self.vector_store_path
        )
        # # 采用embeding模型对文本进行向量化
        # self.vector_store = FAISS.from_documents(split_text, self.embeddings)
        # # 把结果存到faiss索引里面
        # self.vector_store.save_local(self.vector_store_path)

    # def load_vector_store(self):
    #     self.vector_store = FAISS.load_local(self.vector_store_path, self.embeddings)

    def load_vector_store(self):
        vectordb = Chroma(
            persist_directory=self.vector_store_path,
            embedding_function=self.embeddings
            )
        top_k = 5
        retriever = vectordb.as_retriever(search_kwargs={"k": top_k})
        return retriever

if __name__ == '__main__':
    doc_service = DocumentService()
    ###将文本分块向量化存储起来
    doc_service.init_source_vector()