"""
参数调优与神经网络训练脚本

该脚本通过网格搜索和交叉验证找到波士顿房价数据集的最佳神经网络参数，
并保存结果供后续使用。
"""
import numpy as np
import pickle
import time
import matplotlib.pyplot as plt
from main import grid_search, evaluate_model

# 设置随机种子以便结果可复现
np.random.seed(42)

# 加载数据
print("加载数据...")
xTr = np.loadtxt("data/xTr.csv", delimiter=",")
yTr = np.loadtxt("data/yTr.csv", delimiter=",").reshape(1, -1)
xTe = np.loadtxt("data/xTe.csv", delimiter=",")
yTe = np.loadtxt("data/yTe.csv", delimiter=",").reshape(1, -1)

# 完整参数网格
full_param_grid = {
    'TRANSNAME': ['sigmoid', 'tanh', 'relu', 'leaky_relu', 'gelu'],
    'ROUNDS': [100, 200, 300],
    'ITER': [30, 50, 70],
    'STEPSIZE': [0.005, 0.01, 0.05],
    'hidden_layers': [
        [20],
        [20, 20],
        [20, 20, 20],
        [30, 20, 10],
        [10, 20, 10]
    ]
}

# 小规模参数网格（用于快速测试）
small_param_grid = {
    'TRANSNAME': ['sigmoid', 'relu', 'leaky_relu'],
    'ROUNDS': [200],
    'ITER': [50],
    'STEPSIZE': [0.01],
    'hidden_layers': [
        [20, 20],
        [20, 20, 20]
    ]
}

# 选择使用哪个参数网格
print("请选择参数搜索模式：")
print("1. 快速模式（小规模参数网格，用于测试）")
print("2. 完整模式（全参数网格，耗时较长）")
mode = input("请输入选择（1或2）：")

if mode == "2":
    active_grid = full_param_grid
    print("\n使用完整参数网格进行搜索，这可能需要较长时间...")
else:
    active_grid = small_param_grid
    print("\n使用小规模参数网格进行快速测试...")

# 记录开始时间
start_time = time.time()

# 执行参数搜索
print("\n开始参数搜索...")
best_params = grid_search(xTr, yTr, xTe, yTe, active_grid, use_cv=True, k_folds=5, verbose=True)

# 计算搜索耗时
search_time = time.time() - start_time
print(f"\n参数搜索完成！总耗时: {search_time:.2f}秒")

# 使用最佳参数在完整测试集上评估
print("\n使用最佳参数评估最终模型...")
train_err, test_err, train_time = evaluate_model(xTr, yTr, xTe, yTe, best_params, verbose=True)

print(f"\n最终结果:")
print(f"训练RMSE: {train_err:.4f}")
print(f"测试RMSE: {test_err:.4f}")
print(f"训练时间: {train_time:.2f}秒")

# 保存最佳参数
with open('best_parameters.pickle', 'wb') as f:
    pickle.dump(best_params, f)

print("\n最佳参数已保存至'best_parameters.pickle'")

# 可视化结果
d = xTr.shape[0]
print(f"\n最佳参数摘要:")
print(f"激活函数: {best_params['TRANSNAME']}")
print(f"训练轮次: {best_params['ROUNDS']}")
print(f"每轮迭代: {best_params['ITER']}")
print(f"学习率: {best_params['STEPSIZE']}")
hidden_structure = best_params['wst'].tolist()
hidden_structure = hidden_structure[1:-1]  # 移除输入和输出层
print(f"隐藏层结构: {hidden_structure}")

# 创建网络结构可视化
plt.figure(figsize=(10, 6))
layers = [d] + hidden_structure + [1]  # 完整网络结构

# 绘制网络结构
for i, layer_size in enumerate(layers):
    # 绘制节点
    for j in range(layer_size):
        # 调整节点位置
        y_pos = (j - layer_size/2) * 0.5 + 0.5
        plt.scatter(i, y_pos, s=100, color='blue')
        
        # 如果不是最后一层，绘制连接到下一层的线
        if i < len(layers) - 1:
            for k in range(layers[i+1]):
                y_next = (k - layers[i+1]/2) * 0.5 + 0.5
                plt.plot([i, i+1], [y_pos, y_next], 'gray', alpha=0.1)

plt.title(f'最佳神经网络结构 (激活函数: {best_params["TRANSNAME"]})')
plt.xlabel('层')
plt.ylabel('节点')
plt.grid(False)
plt.tight_layout()
plt.savefig('best_network_structure.png')
plt.show()

print("\n网络结构图已保存至'best_network_structure.png'")