
import os
import qianfan

# 使用安全认证AK/SK鉴权，通过环境变量方式初始化；替换下列示例中参数，安全认证Access Key替换your_iam_ak，Secret Key替换your_iam_sk
os.environ["QIANFAN_ACCESS_KEY"] = "pG5Eq6zo1p66o4bg3UY5fDne"
os.environ["QIANFAN_SECRET_KEY"] = "S5ZJ6WaIyi5nu6lYiF2NMGUfJnGFxvoa"

emb = qianfan.Embedding()

resp = emb.do(model="tao-8k", texts=[ 
    "推荐一些美食"
])
print(resp["body"])