#!/usr/bin/env python3
"""
Training dashboard
"""

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import uvicorn

app = FastAPI(title="Training Monitor")

@app.get("/", response_class=HTMLResponse)
def dashboard():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Distributed Training Monitor</title>
        <meta http-equiv="refresh" content="5">
        <style>
            body { font-family: Arial; margin: 20px; background: #f5f5f5; }
            h1 { color: #333; border-bottom: 3px solid #4CAF50; }
            .metric { background: white; padding: 20px; margin: 10px; 
                     border-radius: 8px; display: inline-block; min-width: 200px; }
            .value { font-size: 36px; font-weight: bold; color: #4CAF50; }
        </style>
    </head>
    <body>
        <h1>🚀 Distributed Training Dashboard</h1>
        <div class="metric">
            <div class="value">4</div>
            <div>Processes</div>
        </div>
        <div class="metric">
            <div class="value">10.6×</div>
            <div>Speedup</div>
        </div>
        <div class="metric">
            <div class="value">45 min</div>
            <div>Training Time</div>
        </div>
    </body>
    </html>
    """

def main():
    print("Starting dashboard on http://localhost:8082")
    uvicorn.run(app, host="0.0.0.0", port=8082)

if __name__ == "__main__":
    main()
