from __future__ import annotations

from typing import Any, Optional

from visitor_analysis.neo4j_client import Neo4jClient, pick, serialize_analysis


class AudienceService:
    def __init__(self, neo4j: Neo4jClient):
        self._neo4j = neo4j

    def list_attractions(
        self,
        keyword: Optional[str] = None,
        page: int = 1,
        page_size: int = 20,
    ) -> dict[str, Any]:
        page = max(1, page)
        page_size = min(max(1, page_size), 100)
        skip = (page - 1) * page_size
        keyword = (keyword or "").strip()

        query = """
        MATCH (a:Attraction)
        WHERE ($keyword = '' OR
               toLower(coalesce(a.name, '')) CONTAINS toLower($keyword) OR
               toLower(coalesce(a.address, a.city, a.county, a.location, '')) CONTAINS toLower($keyword))
        WITH
          coalesce(a.name, '') AS name,
          coalesce(a.address, a.city, a.county, a.location, '') AS address,
          collect(DISTINCT a.uid) AS uids,
          collect(a) AS nodes
        WITH name, address, uids, nodes,
             any(n IN nodes WHERE EXISTS { MATCH (n)-[:HAS_ANALYSIS]->(:VisitorAnalysis) }) AS has_analysis
        RETURN name, address, uids, has_analysis
        ORDER BY name, address
        SKIP $skip LIMIT $limit
        """

        count_query = """
        MATCH (a:Attraction)
        WHERE ($keyword = '' OR
               toLower(coalesce(a.name, '')) CONTAINS toLower($keyword) OR
               toLower(coalesce(a.address, a.city, a.county, a.location, '')) CONTAINS toLower($keyword))
        WITH
          coalesce(a.name, '') AS name,
          coalesce(a.address, a.city, a.county, a.location, '') AS address
        WITH DISTINCT name, address
        RETURN count(*) AS total
        """

        params = {"keyword": keyword, "skip": skip, "limit": page_size}
        count_rows = self._neo4j.run(count_query, {"keyword": keyword})
        total = int(count_rows[0]["total"]) if count_rows else 0

        rows = self._neo4j.run(query, params)
        items = []
        for row in rows:
            uids = [str(u) for u in (row["uids"] or []) if u]
            items.append(
                {
                    "name": row["name"],
                    "address": row["address"],
                    "uid": uids[0] if uids else None,
                    "uids": uids,
                    "has_analysis": bool(row["has_analysis"]),
                }
            )

        return {
            "total": total,
            "page": page,
            "page_size": page_size,
            "items": items,
        }

    def get_audience(self, attraction_uid: str) -> dict[str, Any]:
        uid = (attraction_uid or "").strip()
        if not uid:
            raise ValueError("attraction_id 不可為空白")

        query = """
        MATCH (a:Attraction {uid: $uid})
        OPTIONAL MATCH (a)-[:HAS_ANALYSIS]->(d:VisitorAnalysis {visitor_type: 'domestic'})
        OPTIONAL MATCH (a)-[:HAS_ANALYSIS]->(i:VisitorAnalysis {visitor_type: 'inbound'})
        RETURN a AS attraction, d AS domestic, i AS inbound
        LIMIT 1
        """

        rows = self._neo4j.run(query, {"uid": uid})
        if not rows:
            return {
                "found": False,
                "attraction": None,
                "domestic": None,
                "inbound": None,
            }

        row = rows[0]
        attraction = Neo4jClient.as_props(row, "attraction") or {}
        domestic = serialize_analysis(Neo4jClient.as_props(row, "domestic"))
        inbound = serialize_analysis(Neo4jClient.as_props(row, "inbound"))

        return {
            "found": True,
            "attraction": {
                "uid": pick(attraction, "uid", default=uid),
                "name": pick(attraction, "name"),
                "address": pick(
                    attraction, "address", "city", "county", "location", default=None
                ),
                "city": pick(attraction, "city", "county", default=None),
            },
            "domestic": domestic,
            "inbound": inbound,
        }
