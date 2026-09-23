# history.py(检索历史记录模块)：用轻量 SQLite 单文件数据库保存每一次图片检索的日志，让"系统被用过"可回溯。
# 解决两个核心问题:
#  1. 记录:每次 /api/search 成功后调 record_search 落库(查询词/模式/结果数/耗时/Mock 标记)
#  2. 查询:GET /api/history 给前端历史面板,点击历史条目可一键重发
# 存储位置 DATA_DIR/history.db;sqlite3 连接按需开关,无长连接、无外部依赖
# 调用方:routers/search.py 写入;前端 SearchPage.tsx 读取

import sqlite3
import time
from pathlib import Path

from fastapi import APIRouter

from backend.config import DATA_DIR

# 创建独立路由实例，相当于一个子路由容器
router = APIRouter(prefix="/api/history", tags=["Search History"])

# 拼接数据库文件路径：data/history.db
DB_PATH = Path(DATA_DIR) / "history.db"


# _connect(取连接)：建立sqlite数据库短连接，返回数据库连接对象conn
def _connect() -> sqlite3.Connection:

    # 打开sqlite文件；timeout=3：如果数据库被锁住，最多等待3秒再抛异常
    conn = sqlite3.connect(DB_PATH, timeout=3)

    # 开启WAL预写日志模式，是sqlite的高级特性
    conn.execute("PRAGMA journal_mode=WAL")

    # 返回：数据库连接对象conn
    return conn


# _ensure_table(建表)：幂等建表函数：检查表是否存在，不存在就创建；存在就什么都不做
def _ensure_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS searches (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at REAL NOT NULL,
            query TEXT NOT NULL,
            mode TEXT NOT NULL,
            result_count INTEGER NOT NULL,
            total_time REAL NOT NULL,
            mock INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    conn.commit()


# record_search(写入一条检索记录)：保存单次检索的记录到 sqlite，在search.py检索成功后调用
# 任何异常都不抛——历史记录是锦上添花,绝不能让它弄挂检索主流程
def record_search(query: str, mode: str, result_count: int, total_time: float, mock: bool) -> None:

    # query：用户搜索文本（图片检索时 query 为空）
    # mode：搜索模式（文本搜图 / 以图搜图）
    # result_count：这次检索返回多少张图片
    # total_time：检索耗时，单位秒
    # mock：是否是 mock 模拟数据模式

    try:
        # 拿到 sqlite 连接；with上下文管理器，自动关闭数据库连接，不用手动 close
        with _connect() as conn:

            # 调用之前写的函数：检查表是否存在，不存在自动建表
            _ensure_table(conn)

            # SQL 插入语句，往 searches 表新增一行
            conn.execute(
                "INSERT INTO searches (created_at, query, mode, result_count, total_time, mock) VALUES (?,?,?,?,?,?)",
                (time.time(), query or "(图片检索)", mode, result_count, total_time, 1 if mock else 0),
            )

            # 提交事务
            conn.commit()

    # 捕获所有异常：数据库文件损坏、磁盘满、读写权限不足等等
    except Exception as e:
        print(f"⚠️ 检索历史写入失败(不影响主流程): {e}")


# list_history(最近检索 /api/history)：查询检索历史，默认 20 条,新的在前
@router.get("")
def list_history(limit: int = 20):

    # 参数防护，把 limit 限制在1~100。防止前端传入超大数字，一次性读取海量数据压垮 sqlite
    limit = max(1, min(limit, 100))

    # 打开数据库连接
    with _connect() as conn:
        # 检查表存在
        _ensure_table(conn)

        # 执行查询 SQL，按 id 倒序，取 limit 条
        rows = conn.execute(
            "SELECT id, created_at, query, mode, result_count, total_time, mock "
            "FROM searches ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        # `.fetchall()`：取出全部查询结果，存到 rows。rows 是元组列表，每一行是一个元组

    # 返回 json 给前端
    return {
        "total": len(rows),
        "items": [
            {
                "id": r[0],
                "createdAt": r[1],
                "query": r[2],
                "mode": r[3],
                "resultCount": r[4],
                "totalTime": r[5],
                "mock": bool(r[6]),
            }
            for r in rows
        ],
    }
 
# clear_history(清空历史 /api/history)：清空全部历史记录,最干净
@router.delete("")
def clear_history():
    # 获取数据库连接
    with _connect() as conn:
        # 检查表
        _ensure_table(conn)
        # 删除 searches 表里所有行
        conn.execute("DELETE FROM searches")
        # 提交删除操作
        conn.commit()

    # 返回成功标识
    return {"status": "ok"}
