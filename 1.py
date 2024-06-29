# Embedding模型API调用与测试
from langchain_community.embeddings import QianfanEmbeddingsEndpoint
from config import qianfan_ak,qianfan_sk,model_embed,endpoint_embed
# os.environ["QIANFAN_AK"] = "your_ak"
# os.environ["QIANFAN_SK"] = "your_sk"
# 从千帆云平台获取Emb模型API

embed = QianfanEmbeddingsEndpoint(
    qianfan_ak=qianfan_ak,
    qianfan_sk=qianfan_sk,
    model=model_embed, endpoint=endpoint_embed
)
res = embed.embed_documents(["hi", "world"])

for r in res:
    print(r[:8])