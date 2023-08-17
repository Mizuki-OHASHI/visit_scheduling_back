from http import HTTPStatus
import json
from datetime import datetime

from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

from optimize import workflow


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    pass


# Access_Control_Allow_Origin = "https://visit-scheduling-front.vercel.app"
Access_Control_Allow_Origin = "http://localhost:3000"


class UserRequestHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path.startswith("/v1"):
            content_type = self.headers.get("Content-Type", "")

            if content_type != "application/json":
                self.send_response(HTTPStatus.BAD_REQUEST)
                self.end_headers()
                return

            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            data = json.loads(body.decode())

            if (
                "chouseisan" not in data
                or "memberinfo" not in data
                or "candidateinfo" not in data
                or "considergender" not in data
            ):
                self.send_response(HTTPStatus.BAD_REQUEST)
                self.end_headers()
                return

            chouseisan = data["chouseisan"]
            memberInfo = data["memberinfo"]
            candidateInfo = data["candidateinfo"]
            consider_gender = bool(data["considergender"])

            start = datetime.now()
            # try:
            response_data = workflow(
                chouseisan, memberInfo, candidateInfo, consider_gender
            )
            # except:
            #     response_data = {"status": "error", "detail": traceback.format_exc()}
            #     print(traceback.format_exc())

            response_body = json.dumps(response_data).encode("utf-8")

            delta = datetime.now() - start
            # print("==" * 64)
            print(f"実行時間\t: {delta.seconds}.{delta.microseconds} 秒\n\n")

            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response_body)))
            self.send_header(
                "Access-Control-Allow-Origin", Access_Control_Allow_Origin
            )  # Add this line
            self.end_headers()
            self.wfile.write(response_body)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Access-Control-Allow-Origin", Access_Control_Allow_Origin)
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.end_headers()


def main():
    port = 8080
    server = ThreadedHTTPServer(("", port), UserRequestHandler)
    print(f"Server running on port {port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
