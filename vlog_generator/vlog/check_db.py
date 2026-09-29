import os
from neo4j import GraphDatabase

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "playtaiwan2026")

def check_database():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    with driver.session() as session:
        # 1. 統計所有標籤與數量
        result = session.run("MATCH (n) RETURN labels(n) AS labels, count(n) AS total")
        print("📊 【資料庫節點統計】")
        for record in result:
            labels = record["labels"]
            total = record["total"]
            print(f"  - 標籤 {labels}: {total} 筆")

        # 2. 抽樣檢查有沒有餐廳 (Restaurant) 或旅宿 (Hotel)
        print("\n🏨 【餐廳與旅宿抽樣檢查】")
        res_sample = session.run("MATCH (r:Restaurant) RETURN r.name AS name LIMIT 3")
        restaurants = [r["name"] for r in res_sample]
        print(f"  - 餐廳 (Restaurant) 範例: {restaurants if restaurants else '❌ 目前尚無資料'}")

        hotel_sample = session.run("MATCH (h:Hotel) RETURN h.name AS name LIMIT 3")
        hotels = [h["name"] for h in hotel_sample]
        print(f"  - 旅宿 (Hotel) 範例: {hotels if hotels else '❌ 目前尚無資料'}")

    driver.close()

if __name__ == "__main__":
    check_database()
