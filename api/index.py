from http.server import BaseHTTPRequestHandler
import json
import os
import mimetypes

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        url_path = self.path.split('?')[0]
        
        # API endpoint
        if url_path == '/api' or url_path.startswith('/api/'):
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            payload = {
                "status": "online",
                "project": "Bitcoin 24h Multi-Horizon Price Forecasting",
                "model": "Seq2Seq LSTM with Custom Self-Attention",
                "mae_norm": 0.00661,
                "horizon": "24 hours",
                "lookback": "48 hours"
            }
            self.wfile.write(json.dumps(payload, indent=2).encode('utf-8'))
            return
        
        # Route root / or /index.html to index.html
        if url_path == '/' or url_path == '/index.html':
            target_file = os.path.join(ROOT_DIR, 'index.html')
        else:
            # Strip leading slash
            rel_path = url_path.lstrip('/')
            target_file = os.path.join(ROOT_DIR, rel_path)
        
        # Serve static file if exists
        if os.path.exists(target_file) and os.path.isfile(target_file):
            content_type, _ = mimetypes.guess_type(target_file)
            if not content_type:
                if target_file.endswith('.js'):
                    content_type = 'application/javascript'
                elif target_file.endswith('.json'):
                    content_type = 'application/json'
                elif target_file.endswith('.css'):
                    content_type = 'text/css'
                else:
                    content_type = 'text/plain'
            
            with open(target_file, 'rb') as f:
                content = f.read()
            
            self.send_response(200)
            self.send_header('Content-type', content_type)
            self.send_header('Content-Length', str(len(content)))
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(content)
            return
        
        # Fallback to index.html
        fallback_file = os.path.join(ROOT_DIR, 'index.html')
        if os.path.exists(fallback_file):
            with open(fallback_file, 'rb') as f:
                content = f.read()
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.send_header('Content-Length', str(len(content)))
            self.end_headers()
            self.wfile.write(content)
            return

        self.send_response(404)
        self.end_headers()
        self.wfile.write(b'404 Not Found')
