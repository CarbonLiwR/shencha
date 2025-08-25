## 审查智能体

通过fast api提供server 服务

### 智能体

复制.nev.example为.env
并按照格式填写api_key

再创建一个.env1，你可以在env1里自由配置，配置的apikey用于意图识别与信息提取智能体
在env文件里的配置用于图片识别智能体


### 后端

```commandline
pip install -r requirements.txt
python app.py
```


### 前端

```commandline
cd frontend
npm install
npm run dev
```

### 测试用例使用

```commandline
python test_api_all.py
```
