import json
import os
import signal
import sys
import time
import io
import random
import http.server
import urllib.parse
from datetime import datetime
from typing import List

from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn


class UserResForHTTPGet:
    def __init__(self, id: str, name: str, age: int):
        self.id = id
        self.name = name
        self.age = age


class NewUserNameAge:
    def __init__(self, name: str, age: int):
        self.name = name
        self.age = age


class NewUserId:
    def __init__(self, id: str):
        self.id = id


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    pass


class UserRequestHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path.startswith("/user"):
            content_type = self.headers.get("Content-Type", "")

            if content_type != "application/json":
                self.send_response(400)
                self.end_headers()
                return

            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            data = json.loads(body.decode())

            if "name" not in data or "age" not in data:
                self.send_response(400)
                self.end_headers()
                return

            name = data["name"]
            age = data["age"]

            if not name or not age:
                self.send_response(400)
                self.end_headers()
                return

            user_id = create_user(name, age)

            if user_id is None:
                self.send_response(500)
                self.end_headers()
                return

            response_data = {"id": user_id}
            response_body = json.dumps(response_data).encode()

            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(response_body)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header(
            "Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS"
        )
        self.end_headers()


def main():
    port = 8000
    server = ThreadedHTTPServer(("", port), UserRequestHandler)
    print(f"Server running on port {port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
