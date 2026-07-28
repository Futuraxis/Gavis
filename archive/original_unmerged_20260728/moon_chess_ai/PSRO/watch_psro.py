# watch_psro.py
import numpy as np
from moon_env_wrapper import MoonChessEnv
from agent import Agent
import time

# 加载训练数据
data = np.load('Qh.npy', allow_pickle=True).item()
nash_pi = data['nash']          # 纳什均衡策略
pi = data['pi']                 # 策略池

# 构建最终策略（按纳什权重混合）
nash_agent = Agent(nash_pi)

env = MoonChessEnv()

for episode in range(5):
    obs, _ = env.reset()
    done = False
    step = 0
    print(f"\n========== 第 {episode+1} 局 ==========")
    
    while not done:
        mask = env.available_actions()
        action = nash_agent.step(obs, Amask=mask)
        obs, r, done, _, _ = env.step(action)
        step += 1
        time.sleep(0.5)
        # 可选：解码状态观察棋盘（如需要）
        print(f"步 {step}: 下在位置 {action}, 奖励 {r}")
    
    print(f"游戏结束，共 {step} 步，最后奖励 {r}")
