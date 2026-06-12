import asyncio
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
from sqlalchemy import text

async def init():
    dl = SQLAlchemyDataLayer(conninfo="sqlite+aiosqlite:///./chainlit.db", ssl_require=False)
    async with dl.engine.begin() as conn:
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                identifier TEXT UNIQUE NOT NULL,
                "createdAt" TEXT,
                metadata TEXT NOT NULL
            )
        """))
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS threads (
                id TEXT PRIMARY KEY,
                "createdAt" TEXT,
                name TEXT,
                "userId" TEXT,
                "userIdentifier" TEXT,
                tags TEXT,
                metadata TEXT,
                FOREIGN KEY ("userId") REFERENCES users(id)
            )
        """))
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS steps (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                type TEXT NOT NULL,
                "threadId" TEXT NOT NULL,
                "parentId" TEXT,
                "disableFeedback" INTEGER NOT NULL,
                streaming INTEGER NOT NULL,
                waiting INTEGER NOT NULL,
                "isError" INTEGER NOT NULL,
                metadata TEXT,
                tags TEXT,
                input TEXT,
                output TEXT,
                "createdAt" TEXT,
                start TEXT,
                "end" TEXT,
                FOREIGN KEY ("threadId") REFERENCES threads(id)
            )
        """))
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS feedbacks (
                id TEXT PRIMARY KEY,
                "forId" TEXT NOT NULL,
                "threadId" TEXT NOT NULL,
                value INTEGER NOT NULL,
                comment TEXT
            )
        """))
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS elements (
                id TEXT PRIMARY KEY,
                "threadId" TEXT,
                type TEXT,
                url TEXT,
                "chainlitKey" TEXT,
                name TEXT NOT NULL,
                display TEXT,
                language TEXT,
                page INTEGER,
                size TEXT,
                "forId" TEXT,
                mime TEXT
            )
        """))
        print("Tables créées avec succès !")

asyncio.run(init())