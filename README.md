## 审查智能体

通过fast api提供server 服务

运行
```commandline
uvicorn app:app --reload --host 0.0.0.0 --port 8000
docker build -t shencha-agent:latest .
docker images shencha-agent
docker save shencha-agent:latest -o shencha/shencha-deploy/docker-images/shencha-agent.tar
```
### 智能体
