# 虚拟环境：vllm
# pip install PyPDF2

import gradio as gr
import requests
import os
import json
from datetime import datetime

# 确保中文显示正常
import matplotlib.pyplot as plt

import os
import tiktoken  # 替换 langchain 的分块器
from typing import List, Dict, Any
from chromadb import PersistentClient, Collection
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import re
from openai import OpenAI
import time
from rank_bm25 import BM25Okapi
import PyPDF2
import pandas as pd
import traceback


plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]


# 配置参数
CHROMA_DB_PATH = "/home/dell/Documents/chroma_local_db"
EMBEDDING_MODEL = "/home/dell/Downloads/bge-m3/"
CHUNK_SIZE = 500  # 文本块字符数
CHUNK_OVERLAP = 0  # 块重叠字符数
TOP_K = 5

def deepseek(msg, retry_num=1, temperature=0.2):
    for _ in range(retry_num):  # Retry
        try:
            model_name = "deepseek-chat"
            client = OpenAI(api_key=os.environ.get("DEEPSEEK_KEY"), base_url="https://api.deepseek.com")
            completion = client.chat.completions.create(
                model=model_name,
                messages=msg,
                stream=False,
                temperature=temperature
                # response_format={"type": "json_object"},
            )

            res = completion.choices[0].message.content
            # if res[-1]!="}":
            #     return None
            # res = json.loads(completion.choices[0].message.content)
            return res
        except Exception as e:
            print(f"Error processing question: {e}")
            time.sleep(10)
    return None


def deepseek_post(msg, retry_num=1, temperature=0.5,api_key='sk-**',api_base='https://api.deepseek.com/v1/chat/completions'):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    data = {
        "model": "deepseek-chat",
        "messages": msg,
        "temperature": temperature,
        "max_tokens": 5000,
        "stream": True  # 启用流式输出
    }
    
    response = requests.post(
        api_base,
        headers=headers,
        data=json.dumps(data),
        stream=True  # 启用流式接收
    )        
    return  response

def get_keywords(query):
    messages=[
        {"role": "system", "content": "你是一个信息抽取工具"},
        {"role": "user", "content": f'“{query}”\n请提取上面这句话中与消费维权有关的关键词， 提取的关键词数量不要超过3个，选择最优代表性的关键词，用list数据格式展示，如 ["关键词1", "关键词2"]'},
    ] 
    return deepseek(messages)



class ChromaKBManager:
    def __init__(self):
        self.client = PersistentClient(path=CHROMA_DB_PATH)
        self.embedding_function = SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL,
            device="cuda"
        )
        # 初始化 tiktoken 分块器（支持中英文）
        self.encoder = tiktoken.get_encoding("cl100k_base")  # 通用编码模型
        self.id_url_map= self.get_reference()

    def list_all_collections(self) -> List[str]:
        """获取所有集合的名称列表"""
        collections = self.client.list_collections()
        return [col.name for col in collections]

    def get_collection_document_count(self, collection_name: str) -> int:
        """获取指定集合中的文档数量"""
        try:
            collection = self.client.get_collection(name=collection_name)
            return collection.count()  # 返回文档总数
        except ValueError:
            print(f"集合 {collection_name} 不存在")
            return 0

    def create_collection(self, collection_name: str) -> Collection:
        return self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function
        )

    def get_max_doc_id(self, collection: Collection) -> int:
        """获取集合中现有文档的最大ID序号（假设ID格式为 doc_数字）"""
        # 获取所有文档ID
        all_ids = collection.get()["ids"]
        if not all_ids:
            return -1  # 若集合为空，从0开始

        # 提取ID中的数字（例如从 "doc_3" 中提取 3）
        max_id = -1
        pattern = re.compile(r"doc_(\d+)")  # 匹配 "doc_数字" 格式
        for doc_id in all_ids:
            match = pattern.match(doc_id)
            if match:
                current_id = int(match.group(1))
                if current_id > max_id:
                    max_id = current_id
        return max_id

    def split_text(self, text: str) -> List[str]:
        """使用 tiktoken 分块（按字符数拆分，保留语义）"""
        chunks = text.split('\n\n')
        chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
        return chunks

    # 以下方法与之前相同（省略，保持不变）
    def add_texts_to_collection(self, collection: Collection, texts: List[str],
                                metadatas: List[Dict[str, Any]] = None) -> None:
        # 获取当前最大ID序号
        max_id = self.get_max_doc_id(collection)

        ids = [f"doc_{max_id + 1 + i}" for i in range(len(texts))]
        collection.add(
            documents=texts,
            metadatas=metadatas if metadatas else [{} for _ in range(len(texts))],
            ids=ids
        )
        print(f"成功写入 {len(texts)} 条文本数据到集合 {collection.name}")

    def delete_documents_by_ids(self, collection: Collection, ids: List[str]) -> None:
        """根据ID删除文档（支持单个或多个ID）"""
        if not ids:
            print("请提供要删除的文档ID列表")
            return
        # 执行删除
        collection.delete(ids=ids)
        print(f"成功删除 {len(ids)} 条文档（ID: {ids}）")

    def load_all_documents(self, collection: Collection) -> List[Dict[str, Any]]:
        result = collection.get()
        return [
            {"id": id, "document": doc, "metadata": meta}
            for id, doc, meta in zip(result["ids"], result["documents"], result["metadatas"])
        ]

    def get_document_vectors(self, collection: Collection, ids: List[str]) -> List[Dict[str, Any]]:
        """根据文档ID获取对应的向量及文档内容"""
        if not ids:
            return []
        # 调用get方法，指定需要返回embeddings（向量）和documents（文档内容）
        result = collection.get(ids=ids, include=["embeddings", "documents"])
        # 格式化结果：每个元素包含id、document、embedding
        return [
            {
                "id": doc_id,
                "document": doc,
                "embedding": embedding  # 向量列表（float类型）
            } for doc_id, doc, embedding in zip(
                result["ids"],
                result["documents"],
                result["embeddings"]
            )
        ]

    def similarity_search(self, collection: Collection, query: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
        results = collection.query(
            query_texts=[query],
            n_results=top_k,
            include=["documents", "metadatas", "distances"] # 
        )
        return [
            {
                "document": doc,
                "metadata": meta,
                "distance": dist,
                "similarity": 1 - dist if dist <= 1 else 0
            } for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            )
        ]

    def get_reference(self):
        path='/home/dell/Documents/问答对-链接2.csv'
        df=pd.read_csv(path)
        return dict(zip(df['id'],df['url']))

    def getNumbers(self, text):
        # 先找整块，再在块里提取所有数字
        blocks = re.findall(r'检索结果[、,，\s\d]*', text)
        numbers = [] 
        for block in blocks: # ['检索结果4', '检索结果1、4', '检索结果5']
            nums = block.replace('检索结果','')
            numbers.append(nums)

        print(numbers)  # ['4', '1、4', '5'] 
        return numbers[0]


    def replace_ref(self,recall_map,txt):
        recall_map={k.replace('检索结果',''):v for k,v in recall_map.items()}
        print(recall_map)

        old_ids=[]
        match = re.findall(r'\（.*检索结果.*\）', txt)  # 处理（如检索结果4案例）
        for item in match:
            # ids=item[item.find('检索结果')+4:-1]
            ids=self.getNumbers(item)

            old_ids.append(ids.split('、'))

        # print('old_ids',old_ids)

        ids_flat = [item for sublist in old_ids for item in sublist] # 嵌套list展开
        id_set=[]
        for item in ids_flat:
            if item not in id_set:
                id_set.append(item)  # 去除重复元素

        id_newOrder=list(range(1,len(id_set)+1))
        map_dict=dict(zip(id_set,id_newOrder)) # 生成连续编号  改进字典设计
        map_dict # {'1': 1, '4': 2, '7': 3, '3': 4, '5': 5} 
        print(old_ids) # [['4'], ['1', '7'], ['3', '5']]
        print('map_dict',map_dict)

        try:
            ref_urls=[]
            for oid in old_ids:
                ref_url=''
                for id in oid:
                    ref_url+=f'[{map_dict[id]}]({recall_map[id]})、' # 将检索结果编号 替换为 新的编号，同时 添加链接
                ref_urls.append(f'[{ref_url[:-1]}]')

            print(ref_urls)
            print(match)

            for i,ref_txt in enumerate(match): # 整个文本替换
                txt=txt.replace(ref_txt,ref_urls[i])
        except Exception as e:
            traceback.print_exc()


        return txt


class DeepSeekChatBot:
    def __init__(self):
        """初始化DeepSeek聊天机器人"""
        # self.api_key = os.getenv("DEEPSEEK_API_KEY")
        self.api_key = 'sk-**'
        self.api_base = "https://api.deepseek.com/v1/chat/completions"
        self.chat_history = []
        self.file_cache = {}  # 用于缓存上传的文件内容
        self.kb_manager = ChromaKBManager()
        self.collection=self.kb_manager.create_collection('weiquan')
        self.questions=self.getQuesitons()
        self.temp_msg=''

    def getQuesitons(self):
        with open(r'/home/dell/Documents/questions2.txt', 'r', encoding='utf-8') as f:
             questions = f.readlines()
        return questions

    def set_api_key(self, api_key):
        """设置API密钥"""
        self.api_key = api_key
        return "API密钥已更新"

    def process_file(self, file):
        """处理上传的文件并提取内容"""
        if not file:
            return None, "没有上传文件"

        try:
            file_ext = os.path.splitext(file.name)[1].lower()
            file_content = ""

            # 根据文件类型读取内容
            if file_ext in ['.txt', '.md', '.py', '.html', '.css', '.js']:
                with open(file.name, 'r', encoding='utf-8') as f:
                    file_content = f.read(10000)  # 限制读取大小，避免内容过多
                file_content = f"文件内容 ({file.name}):\n{file_content[:10000]}"

            elif file_ext in ['.pdf']:
                try:
                    import PyPDF2
                    reader = PyPDF2.PdfReader(file.name)
                    text = ""
                    for page in reader.pages[:5]:  # 只读取前5页
                        text += page.extract_text() or ""
                    file_content = f"PDF文件内容 ({file.name}，前5页):\n{text[:10000]}"
                except ImportError:
                    file_content = f"PDF文件 ({file.name}) 已上传，但需要安装PyPDF2才能解析内容。请安装: pip install PyPDF2"
                except Exception as e:
                    file_content = f"解析PDF文件时出错: {str(e)}"

            else:
                file_content = f"不支持的文件类型: {file_ext}，文件名为: {file.name}"

            # 缓存文件内容
            file_id = f"file_{datetime.now().timestamp()}"
            self.file_cache[file_id] = file_content
            return file_id, f"文件处理成功: {file.name}"

        except Exception as e:
            return None, f"文件处理错误: {str(e)}"

    def searchBM25(self,query):
        keywords=get_keywords(query)
        print('提取的关键词：',keywords)
        all_docs = self.kb_manager.load_all_documents(self.collection)

        corpus = []
        ids = []
        docs = []
        sources = []
        doc_sim=[]
        for doc in all_docs:
            ids.append(doc['id'])
            sources.append(doc['metadata']['source'])
            docs.append(doc['document'])
            corpus.append(doc['metadata']['words'])

        bm25 = BM25Okapi(corpus)
        # query = '假一罚十'
        keywords=eval(keywords)
        if len(keywords)==0: return
        print('提取关键词：',keywords)
        for kw in keywords:
            doc_scores = bm25.get_scores(kw)
            # print(doc_scores)

            # 查找最相似的文档
            doc_with_scores = list(zip(range(len(corpus)), doc_scores))
            # 按分数降序排序
            sorted_docs = sorted(doc_with_scores, key=lambda x: x[1], reverse=True)

            for idx, score in sorted_docs[:2]: # 每个关键词提取几个文档
                doc_sim.append(docs[idx])
                # print(f"文档{idx + 1}（分数：{score:.4f}）：{docs[idx]}")
        return doc_sim

    def chat_with_deepseek(self, message, file):
        """与DeepSeek API进行交互 - 流式输出"""
        if not self.api_key:
            yield "", self._format_chat_history()
            return

        if not message and not file:
            yield "", self._format_chat_history()
            return

        # 处理文件
        file_content = ""
        if file:
            file_id, file_msg = self.process_file(file)
            if file_id:
                file_content = self.file_cache[file_id]
            else:
                # 如果文件处理失败，仍然返回当前历史
                yield "", self._format_chat_history()
                return

        # 构建完整消息
        full_message = message
        print(message)
        if file_content:
            full_message += f"\n\n{file_content}"

        results = self.kb_manager.similarity_search(self.collection, message, top_k=5) # not include file content
        retrieve_info = ''
        recall_map=dict()

        for i, res in enumerate(results, 1): # 相似文档
            retrieve_info += f"检索结果{i}： \n{res['document']}\n\n"
            match = re.search(r"\[(\d+)\]", res['document']) #
            if match:
                ref_id = match.group(1)
                recall_map[f"检索结果{i}"]=self.kb_manager.id_url_map[int(ref_id)]  # 根据id获取url

        # 添加BM25查询
        doc_sim=self.searchBM25(message) # 关键词查找
        # print('bm25\n',doc_sim)
        
        for i, doc_content in enumerate(doc_sim, len(results)+1): # 合并检索结果
            retrieve_info += f"检索结果{i}： \n{doc_content}\n\n"
            match = re.search(r"\[(\d+)\]", doc_content)
            if match:
                ref_id = match.group(1)
                recall_map[f"检索结果{i}"]=self.kb_manager.id_url_map[int(ref_id)]  # 根据id获取url


        prompt = f'''
## 用户问题是：{full_message}

## 针对此问题，在知识库中找到如下信息：
{retrieve_info}

## 回答要求：
- 生成回复时，只能基于上述提供的检索结果，绝对不能使用外部知识，并以“（检索结果4、8）”的形式在引用内容的后面进行标注
- 对上述检索结果进行系统化整理，使回复内容更加有条理
- 回复内容的最后要提供一个非常简短的“总结”，说明操作流程'''
        # print(query)

        # 添加用户消息到历史记录
        self.chat_history.append({"role": "user", "content": prompt})

        try:
            response=''
            local_llm=True # 使用在线的deepseek还是本地模型

            if local_llm: # 使用本地部署的ollama
                url = "http://localhost:11434/api/chat"

                headers = { "Content-Type": "application/json"}

                # 请求参数
                data = {
                    "model": "qwen3:14b",  # 模型名称（需与本地拉取的一致）
                    "messages": self.chat_history,
                    "temperature": 0.7,
                    "max_tokens": 5000,
                    "stream": True  # 流式响应
                }
                response = requests.post(url, data=json.dumps(data), headers=headers, stream=True)
                # 处理流式响应
                assistant_reply = ""
                for line in response.iter_lines():
                    if line:
                        try:
                            # 解析每行 JSON 数据
                            line_data = json.loads(line.decode('utf-8'))
                            # 提取响应内容（流式返回时，content 可能为 null，需判断）
                            if "message" in line_data and line_data["message"]["content"]:
                                content = line_data["message"]["content"] #.replace('</think>','\n</think>')  # 注意这里
                                if content.find('<think>')>=0:
                                    content=content.replace('<think>','<details open><summary><strong>💭 查看思考过程</strong></summary><pre style="background-color: #f5f5f5; padding: 10px; border-radius: 5px; margin-top: 10px;">')
                                elif content.find('</think>')>=0:
                                    content=content.replace('</think>','</pre></details>')

                                assistant_reply += content
                                # print(content, end="", flush=True)
                                # 实时更新聊天历史显示
                                temp_history = self.chat_history.copy() # 不修改chat_history 中的内容
                                temp_history.append({"role": "assistant", "content": assistant_reply})
                                yield "", self._format_chat_history(temp_history)

                        except json.JSONDecodeError:
                            continue
            else: # 调用在线的deepseek
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}"
                }

                data = {
                    "model": "deepseek-chat",
                    "messages": self.chat_history,
                    "temperature": 0.7,
                    "max_tokens": 5000,
                    "stream": True  # 启用流式输出
                }

                response = requests.post(
                    self.api_base,
                    headers=headers,
                    data=json.dumps(data),
                    stream=True  # 启用流式接收
                )
                assistant_reply = ""
                for line in response.iter_lines(): # deepseek chat模式 没有thinking过程
                    if line:
                        line = line.decode('utf-8')
                        if line.startswith('data: '):
                            data_str = line[6:]  # 移除 'data: ' 前缀

                            if data_str == '[DONE]':
                                break

                            try:
                                chunk = json.loads(data_str)
                                if 'choices' in chunk and len(chunk['choices']) > 0:
                                    delta = chunk['choices'][0].get('delta', {})
                                    content = delta.get('content', '')
                                    print(content, end="", flush=True)
                                    if content:
                                        assistant_reply += content
                                        # 实时更新聊天历史显示
                                        temp_history = self.chat_history.copy()
                                        temp_history.append({"role": "assistant", "content": assistant_reply})
                                        yield "", self._format_chat_history(temp_history)

                            except json.JSONDecodeError:
                                continue
            
            # 替换参考文献链接
            assistant_reply=self.kb_manager.replace_ref(recall_map, assistant_reply) 

            # 将完整的助手回复添加到历史记录
            # print('第一次回复：',assistant_reply)  # 显示完整的回复信息
            for i in range(2):
                if i==0: # 获取相关问题
                    # assistant_reply+='\n'+self.get_follow_quesetion(full_message)
                    # temp_history = self.chat_history.copy()
                    # temp_history.append({"role": "assistant", "content": assistant_reply})
                    # output=self._format_chat_history(history=temp_history)
                    # yield "", output
                    
                    assistant_reply += '\n\n'
                    response=self.get_follow_quesetion(full_message,assistant_reply) # 获取相关问题
                    for line in response.iter_lines(): # deepseek chat模式 没有thinking过程
                        if line:
                            line = line.decode('utf-8')
                            if line.startswith('data: '):
                                data_str = line[6:]  # 移除 'data: ' 前缀

                                if data_str == '[DONE]':
                                    break

                                try:
                                    chunk = json.loads(data_str)
                                    if 'choices' in chunk and len(chunk['choices']) > 0:
                                        delta = chunk['choices'][0].get('delta', {})
                                        content = delta.get('content', '')
                                        print(content, end="", flush=True)
                                        if content:
                                            assistant_reply += content
                                            # 实时更新聊天历史显示
                                            temp_history = self.chat_history.copy()
                                            temp_history.append({"role": "assistant", "content": assistant_reply})
                                            yield "", self._format_chat_history(temp_history)

                                except json.JSONDecodeError:
                                    continue

                else:  # 完整输出  
                    if assistant_reply:
                        self.chat_history.append({"role": "assistant", "content": assistant_reply}) # 最终消息添加到历史中
                        print('完整回复：',assistant_reply)
                    yield "", self._format_chat_history(isEnd=True) # 形成最后的输出                    

        except Exception as e:
            error_msg = f"发生错误: {str(e)}"
            self.chat_history.append({"role": "assistant", "content": error_msg})
            yield "", self._format_chat_history()
    
    # 调用deepseek模型
    def get_follow_quesetion(self, query, assistant_reply):
        # messages=[
        #     {"role": "system", "content": "你是一个问答助手"},
        #     {"role": "user", "content": f"用户当前的问题是：{query}，请推测用户接下来可能问的问题，问题描述尽可能简洁"},
        # ]
        # return deepseek(msg=messages)

        match = re.search(r'.*总结(.*)', assistant_reply)
        if match:
            summary = match.group(1)
        else:
            summary=query

        # 流式调用
        msg=[{"role": "system", "content": "你是一个维权问题分析助手"},
                {"role": "user", "content": '''用户当前的问题是：{query}，系统给出的建议是：{summary}，请推测用户接下来可能问的问题，最多列出5个问题，直接用编号列出问题，不需要详细展开，
                 并以markdown链接的形式展示每个问题，链接网址为：https://www.baidu.com'''}]
        return  deepseek_post(msg=msg, api_key=self.api_key,api_base=self.api_base)

    def _format_chat_history(self, history=None, isEnd=False):
        """将历史记录转换为Gradio所需的格式"""
        if history is None:
            history = self.chat_history

        gradio_history = []
        for i in range(0, len(history), 2): # 每次处理一对用户和助手消息
            if i + 1 < len(history):
                user_msg = history[i]["content"]  # 用户消息
                if isEnd:  # 对最后一条助手消息进行处理，关闭details标签
                    assistant_msg = history[i + 1]["content"].replace('<details open>', '<details>')  # 助手消息
                else:
                    assistant_msg = history[i + 1]["content"]

                gradio_history.append((user_msg, assistant_msg)) # 用户消息与助手消息开存储
            elif history[i]["role"] == "user":
                # 如果只有用户消息，显示为未完成的对话
                gradio_history.append((history[i]["content"], None))

        return gradio_history

    def clear_chat(self):
        """清空聊天历史"""
        self.chat_history = []
        self.file_cache = {}
        return "聊天历史已清空", []


def create_interface():
    """创建Gradio界面"""
    bot = DeepSeekChatBot()

    with gr.Blocks(title="DeepSeek聊天助手", theme=gr.themes.Soft(),css=".my-input {height: 75vh !important;}") as demo:
        with gr.Row():
            with gr.Column(scale=1):
                api_key_input = gr.Textbox(
                    label="DeepSeek API密钥",
                    type="password",
                    placeholder="请输入你的DeepSeek API密钥"
                )
                set_api_btn = gr.Button("设置API密钥")
                api_status = gr.Textbox(label="API状态", value="未设置API密钥", interactive=False)

                clear_btn = gr.Button("清空聊天历史")
                clear_status = gr.Textbox(label="操作状态", interactive=False)

                file_input = gr.File(label="上传文件 (可选)")

                gr.Markdown("### 支持的文件类型")
                gr.Markdown("- 文本文件: .txt, .md, .py等")

                gr.Markdown("### ✨ 新特性")
                gr.Markdown("- 🔄 流式输出：实时显示AI回复\n- ⚡ 更快的响应体验")

            with gr.Column(scale=4):
                chat_history = gr.Chatbot(label="聊天历史", elem_classes="my-input")
                message = gr.Textbox(label="输入消息", placeholder="请输入你的消息...", elem_id="my_textbox")
                send_btn = gr.Button("发送", variant="primary")

        # 设置事件
        set_api_btn.click(
            fn=bot.set_api_key,
            inputs=[api_key_input],
            outputs=[api_status]
        )

        clear_btn.click(
            fn=bot.clear_chat,
            inputs=[],
            outputs=[clear_status, chat_history]
        )

        # 流式输出需要使用 .then() 链式调用
        send_btn.click(
            fn=bot.chat_with_deepseek,
            inputs=[message, file_input],
            outputs=[message, chat_history]
        )

        # 支持回车发送
        message.submit(
            fn=bot.chat_with_deepseek,
            inputs=[message, file_input],
            outputs=[message, chat_history]
        )

    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.launch(server_name="0.0.0.0", server_port=7862)