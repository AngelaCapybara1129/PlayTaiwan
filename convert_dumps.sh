#!/bin/bash

# 定義你的 4 個 dump 檔案名稱
DUMPS=(
  "attraction-2026-08-20T10-08-48.dump"
  "hotel-2026-08-15T04-43-46.dump"
  "event-2026-08-15T04-43-46.dump"
  "restaurant-2026-08-15T04-47-34.dump"
)

NEO4J_IMAGE="neo4j:enterprise"

for DUMP_FILE in "${DUMPS[@]}"; do
    PREFIX=$(echo $DUMP_FILE | cut -d'-' -f1)
    echo "=================================================="
    echo "🔄 開始處理: $PREFIX ($DUMP_FILE)"
    
    # 1. 準備暫存資料夾
    sudo rm -rf neo4j_temp_data temp_import
    mkdir -p neo4j_temp_data temp_import

    # 2. 複製 dump 檔並改名為 neo4j.dump
    cp "dbs/$DUMP_FILE" "temp_import/neo4j.dump"

    # 3. 載入 Dump
    echo "📦 載入 Dump 檔案至暫存空間 (企業版模式)..."
    docker run --rm \
        -e NEO4J_ACCEPT_LICENSE_AGREEMENT=yes \
        -v $(pwd)/temp_import:/import \
        -v $(pwd)/neo4j_temp_data:/data \
        ${NEO4J_IMAGE} neo4j-admin database load neo4j --from-path=/import --overwrite-destination=true

    # 4. 啟動暫時的 Neo4j 容器 (不掛載 export，直接寫在容器內部)
    echo "🚀 啟動暫時的 Neo4j 容器進行匯出..."
    docker run -d --name neo4j-exporter \
        -e NEO4J_ACCEPT_LICENSE_AGREEMENT=yes \
        -e NEO4J_AUTH=none \
        -e NEO4J_PLUGINS='["apoc"]' \
        -e NEO4J_apoc_export_file_enabled=true \
        -v $(pwd)/neo4j_temp_data:/data \
        ${NEO4J_IMAGE}
    
    echo "⏳ 等待資料庫啟動 (約 30 秒)..."
    sleep 30

    # 5. 匯出至容器內部的 /var/lib/neo4j/import 目錄 (絕對不會有權限問題)
    echo "📝 匯出為 $PREFIX.cypher 腳本..."
    echo "CALL apoc.export.cypher.all('${PREFIX}.cypher', {format: 'cypher-shell'});" | docker exec -i neo4j-exporter cypher-shell --non-interactive

    # 6. 大絕招：使用 docker cp 把檔案從容器裡面直接複製到我們主機的 dbs 資料夾
    echo "🚚 使用 docker cp 將腳本抓取至 dbs 目錄..."
    docker cp neo4j-exporter:/var/lib/neo4j/import/${PREFIX}.cypher dbs/

    # 7. 清理
    echo "🧹 清理暫時容器..."
    docker rm -f neo4j-exporter
    sudo rm -rf neo4j_temp_data temp_import

    echo "✅ $PREFIX 處理完成！"
done

echo "=================================================="
echo "🎉 恭喜！所有 Dump 檔案皆已成功轉換為 Cypher 腳本！"
