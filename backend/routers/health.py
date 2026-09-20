"""
简化的健康检查路由 - 用于快速测试前后端连接
"""
# health.py(健康检查):最简单的 FastAPI 路由样板，用于前后端联通性测试与容器探活，不涉及任何业务逻辑
# 解决两个场景:
#  1. 前后端联调:前端启动时 ping /api/health 确认后端在线
#  2. 容器探活:K8s/Docker 健康检查通过 /api/health 判断服务是否就绪
# 调用方:前端启动脚本、Docker healthcheck、CI 联调测试
from fastapi import APIRouter

# 创建独立路由实例，相当于一个子路由容器
# prefix="/api" 让所有路由统一加 /api 前缀，tags=["Health"] 在 OpenAPI 文档分组显示
router = APIRouter(prefix="/api", tags=["Health"])

# health_check(健康检查 /api/health):返回固定 ok 状态,用于 K8s liveness/readiness probe
@router.get("/health")
def health_check():
    return {"status": "ok", "service": "multimodal-rag-backend"}

# 健康检查接口的真实用途（工程层面）：
# 1.开发调试：启动后端之后，先访问/api/health，快速确认服务有没有正常跑起来，不用调用复杂的检索 / Agent 接口（检索接口要加载大模型，很慢）。
# 2.前端预检：前端页面加载时，先发请求 ping 这个接口，判断后端服务是否在线，给用户提示 “后端未连接”。
# 3.容器 / 云服务存活探测（K8s/docker）：容器编排平台会周期性访问/health，如果接口返回失败，判定服务挂掉，自动重启容器

# 【进阶优化思路，复刻项目可以升级】
# 真正生产级健康检查，要在接口内部增加判断：
#  1.检索引擎是否初始化完成
#  2.Milvus 连接是否正常
#  3.Agent 是否就绪
# 一旦模型 / 向量库异常，就返回 500 错误，而不是永远 ok

# ping(联调探活 /api/ping):极简返回，经典 ping-pong 测试接口，极简连通性测试，前端发 GET 请求，后端返回 pong，用来验证网络通不通、跨域有没有问题
@router.get("/ping")
def ping():
    return {"message": "pong"}
