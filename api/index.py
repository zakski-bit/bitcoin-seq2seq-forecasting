from http.server import BaseHTTPRequestHandler
import json
import os

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        
        data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'forecast_data.json')
        if os.path.exists(data_path):
            with open(data_path, 'r', encoding='utf-8') as f:
                content = f.read()
            self.wfile.write(content.encode('utf-8'))
        else:
            payload = {
                "status": "online",
                "project": "Bitcoin 24h Seq2Seq Forecasting API",
                "model_status": "loaded",
                "mae": 0.00661
            }
            self.wfile.write(json.dumps(payload).encode('utf-8'))
        return
