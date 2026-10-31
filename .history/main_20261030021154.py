import uvicorn

if __name__ == "__main__":
    print("啟動 Omni-Autonomous AI Enterprise v2 控制台：http://127.0.0.1:8000")
    uvicorn.run("ui.app:app", host="127.0.0.1", port=8000, reload=False)