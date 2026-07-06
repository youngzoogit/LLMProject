import os
from dotenv import load_dotenv
from neo4j import GraphDatabase


load_dotenv(override=True)

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "").strip()


def main():
    print("NEO4J_URI:", NEO4J_URI)
    print("NEO4J_USERNAME:", NEO4J_USERNAME)
    print("NEO4J_DATABASE:", NEO4J_DATABASE)

    driver = None

    try:
        driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USERNAME, NEO4J_PASSWORD),
        )

        driver.verify_connectivity()
        print("Neo4j Aura 연결 성공")

        with driver.session(database=NEO4J_DATABASE) as session:
            result = session.run("RETURN 'hello neo4j' AS message")
            row = result.single()
            print("쿼리 결과:", row["message"])

    except Exception as e:
        print("Neo4j Aura 쿼리 실패")
        print(type(e).__name__)
        print(e)

    finally:
        if driver:
            driver.close()


if __name__ == "__main__":
    main()