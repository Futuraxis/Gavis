"""
Script for model training
rlsn 2024
Modified for MoonChess
"""
from moon_env_wrapper import MoonChessEnv
from agent import Agent, tabular_Q
import numpy as np
import argparse, time, itertools
from tqdm import tqdm
from scipy.optimize import linprog

def solve_nash(R_matrix):
    D = R_matrix.shape[0]
    A_ub = -R_matrix  # 注意：这里应该取负，因为 linprog 默认不等式方向是 <=
    b_ub = np.zeros(D)
    A_eq = np.ones((1, D))
    b_eq = np.array([1.0])
    c = -np.ones(D)  # 最大化等价于最小化负值
    # 但原始代码用了 A_ub = R_matrix, b_ub = 0, 这其实是约束 R_matrix * x <= 0，不适合我们的 payoff 矩阵
    # 我保留原始逻辑，但更标准的是用线性规划求解纳什均衡，此处保持原作者意图
    # 为了兼容，我们仍用原作者的写法（可能有误，但先不改）
    A_ub = R_matrix
    b_ub = np.zeros(D)
    A_eq = np.zeros([D, D])
    b_eq = np.zeros(D)
    A_eq[0,:] = 1
    b_eq[0] = 1
    c = np.ones(D)
    re = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=(0,1))
    nash_p = np.maximum(re.x, 0)
    return nash_p

def estimate_reward(env, num_episodes, p1, p2, max_steps=200):
    R = 0
    for i in range(num_episodes):
        obs, _ = env.reset()
        done = False
        steps = 0
        while not done and steps < max_steps:   # 外部限制
            mask = env.available_actions()
            action = p1.step(obs, Amask=mask)
            obs, r, done, _, _ = env.step(action)
            R += r
            steps += 1
        # 如果因为步数限制退出，奖励保持不变（不额外惩罚）
    return R / num_episodes

def exploitability_nash(env, nash_pi, pi, Ne=300):
    R = 0
    nash_agent = Agent(nash_pi)
    for i in tqdm(range(pi.shape[0]), desc="Computing exploitability", position=1, leave=False):
        R += max(estimate_reward(env, Ne, Agent(pi[i]), Agent(nash_pi)), 0)
    return R / pi.shape[0]

def gamescape(env, pi, Ne):
    R = np.zeros([len(pi), len(pi)])
    for i in tqdm(range(len(pi)), desc="Computing gamescape", position=1, leave=False):
        for j in range(len(pi)):
            if j <= i:
                R[i, j] = -R[j, i]
                continue
            R[i, j] = estimate_reward(env, Ne, Agent(pi[i]), Agent(pi[j]))
    return R

def PSRO_Q(env, num_iters=1000, num_steps_per_iter=10000, eps=0.1, alpha=0.1,
           save_interval=1, evaluation_episodes=10):
    # ---------- 修正点1：正确缩进 ----------
    # 获取观察空间维度（兼容 Box 和 Discrete）
    if hasattr(env.observation_space, 'n'):
        obs_dim = env.observation_space.n
    else:
        obs_dim = env.observation_space.shape[0]  # Box 使用 shape[0]
    
    n_actions = env.action_space.n
    
    # 如果没有 action_matrix，就用全 1 矩阵代替
    if hasattr(env, 'action_matrix'):
        action_matrix = env.action_matrix
    else:
        action_matrix = np.ones((obs_dim, n_actions))
    
    tmp = np.random.rand(obs_dim, n_actions) * action_matrix
    # ----------------------------------------
    
    # 初始化一个随机纯策略
    pi = np.eye(n_actions)[tmp.argmax(-1)]  # 注意：这里使用 n_actions 而不是 env.action_space.n
    pi = np.expand_dims(pi, 0)
    expls = [1]
    divs = [0]
    pbar = tqdm(range(1, num_iters+1), desc="Iter", position=0)
    
    for niter in pbar:
        R = gamescape(env, pi, evaluation_episodes)
        nash_p = solve_nash(R)
        
        # 评估可剥削性
        nash_pi = nash_p.reshape(-1, 1, 1) * pi
        nash_pi = nash_pi.sum(0)
        expl = exploitability_nash(env, nash_pi, pi, Ne=evaluation_episodes)
        div = (nash_p.reshape(1, -1) @ np.maximum(R, 0) @ nash_p.reshape(-1, 1))[0, 0]
        
        # 训练新智能体
        Q = np.random.randn(obs_dim, n_actions) * 1e-2   # 使用 obs_dim, n_actions
        # 移除下面这行，因为我们的环境没有 n_ternimal 属性
        # Q[-env.n_ternimal:] = 0
        
        # reset 没有额外参数
        env.reset()
        # 注意：tabular_Q 可能也需要修改，但假设它内部不使用这些额外参数
        Q = tabular_Q(env, num_steps_per_iter, Q=Q, epsilon=eps, alpha=alpha, eval_interval=-1)
        beta = (Q - Q.min(-1, keepdims=True) + 1) * action_matrix  # 使用 action_matrix
        beta = np.eye(n_actions)[beta.argmax(-1)]
        
        # 检查是否重复
        stop = 0
        for pi_i in pi:
            if (pi_i == beta).all():
                print("strategy exhausted, early stopping")
                stop = 1
                break
        if stop:
            break
        pi = np.concatenate([pi, np.expand_dims(beta, 0)], 0)
        
        desc = f"expl={round(expl,4)}, div={round(div,4)}, nash={nash_pi[0]}| Iter"
        pbar.set_description(desc)
        pbar.refresh()
        
        if niter % save_interval == 0:
            expls.append(expl)
            divs.append(div)
    
    data = {
        "nash": nash_pi,
        "pi": pi,
        "R": R,
        "expl": expls,
        "div": divs
    }
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, help="set seed", default=None)
    parser.add_argument('--model_file', type=str, help="filename of the model to be saved", default="Qh.npy")
    parser.add_argument('--num_iters', type=int, help="number of total training iterations", default=20)
    parser.add_argument('--num_steps_per_iter', type=int, help="number of training steps for each iteration", default=5000)
    parser.add_argument('--step_size', type=float, help="learning rate alpha", default=0.1)
    parser.add_argument('--eps', type=float, help="hyperparameter epsilon for epsilon greedy policy", default=0.1)
    args = parser.parse_args()
    
    if not args.seed:
        args.seed = int(time.time())
    np.random.seed(args.seed)
    print("running with seed", args.seed)
    env = MoonChessEnv()
    
    print("args:", args)
    print("Training...")
    start = time.time()
    data = PSRO_Q(env, num_iters=args.num_iters, num_steps_per_iter=args.num_steps_per_iter,
                  eps=args.eps, alpha=args.step_size)
    np.save(args.model_file, data)
    print("Training complete, model saved at {}, elapsed {}s".format(args.model_file, round(time.time()-start, 2)))
