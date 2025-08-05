from elasticsearch import AsyncElasticsearch
from datetime import datetime
from typing import List, Optional


class ChatStorage:
    def __init__(self):
        # ES 연결 설정 수정
        self.es = AsyncElasticsearch(
            hosts=['http://192.168.10.19:9201'],
            basic_auth=('elastic', 'Bhas62FOcDkAqxqh=QPr'),
            verify_certs=False,
            request_timeout=30,
            retry_on_timeout=True,
            max_retries=3
        )
        self.index = "chat_history"

    async def init(self):
        """인덱스 초기화"""
        try:
            # 서버 연결 상태 확인
            info = await self.es.info()
            print(f"Elasticsearch 서버 연결 성공")

            # # 인덱스 존재 여부 확인
            # exists = await self.es.indices.exists(index=self.index)
            
            # if exists:
            #     print(f"'{self.index}' 인덱스가 이미 존재합니다.")
            #     # 기존 매핑 확인
            #     mapping = await self.es.indices.get_mapping(index=self.index)
            #     print(f"현재 매핑: {mapping}")
            #     return

            # # 새 인덱스 생성
            # create_response = await self.es.indices.create(
            #     index=self.index,
            #     settings={
            #         "number_of_shards": 1,
            #         "number_of_replicas": 0
            #     },
            #     mappings={
            #         "properties": {
            #             "title": {"type": "text"},
            #             "messages": {
            #                 "type": "nested",
            #                 "properties": {
            #                     "role": {"type": "keyword"},
            #                     "content": {"type": "text"},
            #                     "timestamp": {"type": "date"}
            #                 }
            #             },
            #             "created_at": {"type": "date"},
            #             "updated_at": {"type": "date"}
            #         }
            #     }
            # )
            
            # if create_response.get("acknowledged"):
            #     # print(f"새로운 '{self.index}' 인덱스가 생성되었습니다.")
            #     # 생성된 매핑 확인
            #     mapping = await self.es.indices.get_mapping(index=self.index)
            #     # print(f"생성된 매핑: {mapping}")
            # else:
            #     raise Exception("인덱스 생성이 승인되지 않았습니다.")
                
        except Exception as e:
            print(f"인덱스 초기화 중 오류 발생: {str(e)}")
            # 상세 에러 정보 출력
            import traceback
            print(traceback.format_exc())
            raise

    async def create_chat(self, title: str, messages: List[str] = None) -> str:
        """새 채팅 생성"""
        def parse_message(msg: str) -> dict:
            if ": " in msg:
                role, content = msg.split(": ", 1)
            else:
                role, content = "unknown", msg
            return {
                "role": role.lower(),
                "content": content,
                "timestamp": datetime.now().isoformat()
            }

        doc = {
            "title": title,
            "messages": [parse_message(msg) for msg in (messages or [])],
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        result = await self.es.index(index=self.index, document=doc)
        return result["_id"]


    async def update_chat(self, chat_id: str, messages: List[str]):
        """채팅 업데이트 - 기존 메시지에 추가"""
        def parse_message(msg: str) -> dict:
            if ": " in msg:
                role, content = msg.split(": ", 1)
            else:
                role, content = "unknown", msg
            return {
                "role": role.lower(),
                "content": content,
                "timestamp": datetime.now().isoformat()
            }

        try:
            result = await self.es.get(index=self.index, id=chat_id)
            existing_messages = result["_source"].get("messages", [])
            new_messages = [parse_message(msg) for msg in messages]
            combined_messages = existing_messages + new_messages

            await self.es.update(
                index=self.index,
                id=chat_id,
                body={
                    "doc": {
                        "messages": combined_messages,
                        "updated_at": datetime.now().isoformat()
                    }
                }
            )
        except Exception as e:
            print(f"업데이트 중 오류 발생: {str(e)}")


    async def get_chat(self, chat_id: str):
        """채팅 조회"""
        result = await self.es.get(index=self.index, id=chat_id)
        chat = result["_source"]
        return {
            "title": chat["title"],
            "messages": [
                f"{msg['role'].capitalize()}: {msg['content']}"
                for msg in chat["messages"]
            ]
        }

    async def list_chats(self, from_: int = 0, size: int = 10):
        """채팅 목록 조회"""
        result = await self.es.search(
            index=self.index,
            body={
                "sort": [{"updated_at": "desc"}],
                "from": from_,
                "size": size
            }
        )
        return [
            {
                "id": hit["_id"],
                "title": hit["_source"]["title"],
                "updated_at": hit["_source"]["updated_at"]
            }
            for hit in result["hits"]["hits"]
        ]

    async def delete_chat(self, chat_id: str):
        """채팅 삭제"""
        await self.es.delete(index=self.index, id=chat_id)

    async def close(self):
        """연결 종료"""
        await self.es.close()