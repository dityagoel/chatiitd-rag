# chatiitd-rag

## How to run
1) Run qdrant using docker:
```
sudo docker run -d -p 6333:63333 -p 6334:6334 --restart unless-stopped qdrant/qdrant
```

2) Go to http://localhost:6333/dashboard and upload the 2 snapshots from https://drive.google.com/drive/folders/17WIUqR5XkIPWMJ-Gsb_P-yfqFgu4QXQj?usp=sharing

3) Create a .env file in the root directory with the following content:
```
GOOGLE_API_KEY="Gemini api key"
QDRANT_URL="http://localhost:6333"
```

4) Run the script
```
cd agentic_chatbot
python agent.py
``` 