import os
from neo4j import GraphDatabase

# 連線設定 (對應你的 core/config 或環境變數)
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "playtaiwan2026")

def run_cypher_file(file_path: str):
    if not os.path.exists(file_path):
        print(f"❌ 找不到檔案: {file_path}")
        return

    print(f"📂 正在讀取並執行: {file_path} ...")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 將 cypher 檔以分號 (;) 切割成個別指令執行
    statements = [stmt.strip() for stmt in content.split(";") if stmt.strip()]

    with driver.session() as session:
        for idx, query in enumerate(statements):
            try:
                session.run(query)
                print(f"  [✔] 執行第 {idx+1}/{len(statements)} 道指令成功")
            except Exception as e:
                print(f"  [⚠️] 第 {idx+1} 道指令執行失敗 (可能已存在或語法不相容): {e}")

    driver.close()
    print(f"✨ 檔案 {file_path} 匯入完成！\n")

if __name__ == "__main__":
    dbs_dir = os.path.expanduser("~/playtaiwan/dbs")
    
    # 依序匯入餐廳與旅宿
    run_cypher_file(os.path.join(dbs_dir, "restaurant.cypher"))
    run_cypher_file(os.path.join(dbs_dir, "hotel.cypher"))
