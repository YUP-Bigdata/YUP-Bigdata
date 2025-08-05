# import sys

# from PySide6.QtWidgets import QApplication

# from ui.chat_window import ChatWindow

import uvicorn
from api.api_server import app

# def main():
#     app = QApplication(sys.argv)
#     window = ChatWindow()
#     window.show()
#     sys.exit(app.exec())


if __name__ == "__main__":
    # main()
    uvicorn.run(
        "api.api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True  # 개발 중 코드 변경 시 자동 리로드
    )
