# train.py - 训练月亮棋AI的核心脚本
from moon_chess_env import MoonChessEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback

# 1. 创建我们的月亮棋环境
env = MoonChessEnv()

# 2. 创建 PPO 算法模型
# MlpPolicy 指使用多层神经网络（深度学习）
# verbose=1 让训练过程打印进度条
model = PPO('MlpPolicy', env, verbose=1, 
            learning_rate=0.0003,  # 学习率
            n_steps=2048,          # 每批训练步数
            batch_size=64,         
            n_epochs=10)           # 每批数据反复学习10次

print("\n🚀 开始训练 AI，让它学会下月亮棋！")
print("⏳ CPU 训练大约需要 2-3 分钟，请耐心等待...\n")

# 3. 开始训练！总步数设为 50000 步（对于月亮棋足够学会基本策略）
model.learn(total_timesteps=50000)

# 4. 保存训练好的模型（下次直接用，不用重新训练）
model.save("moon_chess_ai_model")
print("\n✅ 训练完成！AI模型已保存为 'moon_chess_ai_model.zip'")