import gymnasium as gym
import ale_py

# 这一行是保险：如果环境版本对齐了，没有它也能跑；
# 如果环境还是报错找不到 ALE，加上它能强制让 Gymnasium 识别驱动。
try:
    gym.register_envs(ale_py)
except:
    pass

# 创建 Boxing 环境 (ALE/Boxing-v5 是标准 ID)
env = gym.make("ALE/Boxing-v5", render_mode="human")

observation, info = env.reset()

print("✅ 游戏窗口应已弹出，正在运行随机动作...")

for _ in range(1000):
    # 随机采取一个动作
    action = env.action_space.sample()

    # 执行动作 (注意：Gymnasium 返回 5 个值)
    observation, reward, terminated, truncated, info = env.step(action)

    # 渲染画面
    env.render()

    # 如果游戏结束或达到限制，重置环境
    if terminated or truncated:
        observation, info = env.reset()

env.close()
print("🎮 测试运行完成。")