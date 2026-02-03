import pickle
import numpy as np
import time
from preprocess import preprocess
from initweights import initweights
from grdescent import grdescent
from deepnet import deepnet
from get_transition_func import get_transition_func

def evaluate_model(xTr, yTr, xTe, yTe, params, verbose=False):
    """
    评估给定参数下的模型性能
    
    参数:
    xTr, yTr: 训练数据
    xTe, yTe: 测试数据
    params: 参数字典，包含TRANSNAME, ROUNDS, ITER, STEPSIZE, wst
    verbose: 是否打印详细信息
    
    返回:
    train_err, test_err: 训练和测试误差
    training_time: 训练时间（秒）
    """
    # 解包参数
    TRANSNAME = params['TRANSNAME']
    ROUNDS = params['ROUNDS']
    ITER = params['ITER']
    STEPSIZE = params['STEPSIZE']
    wst = params['wst']
    
    # 预处理数据
    [xTr_prep, xTe_prep, _, _] = preprocess(xTr, xTe)
    
    # 初始化权重
    w = initweights(wst)
    
    # 训练模型并记录时间
    start_time = time.time()
    f = lambda w: deepnet(w, xTr_prep, yTr, wst, TRANSNAME)
    
    for i in range(ROUNDS):
        w = grdescent(f, w, STEPSIZE, ITER, 1e-8)
    
    training_time = time.time() - start_time
    
    # 评估性能
    predTr = deepnet(w, xTr_prep, [], wst, TRANSNAME)
    predTe = deepnet(w, xTe_prep, [], wst, TRANSNAME)
    
    train_err = np.sqrt(np.mean((predTr - yTr) ** 2))
    test_err = np.sqrt(np.mean((predTe - yTe) ** 2))
    
    if verbose:
        print(f"Parameters: {params}")
        print(f"Training RMSE: {train_err:.4f}")
        print(f"Test RMSE: {test_err:.4f}")
        print(f"Training time: {training_time:.2f} seconds")
        print("-" * 50)
    
    return train_err, test_err, training_time

def cross_validate(xTr, yTr, params, k_folds=5, verbose=False):
    """
    使用k折交叉验证评估模型性能
    
    参数:
    xTr, yTr: 训练数据
    params: 参数字典
    k_folds: 折数
    verbose: 是否打印详细信息
    
    返回:
    mean_valid_error: 平均验证误差
    """
    # 解包参数
    TRANSNAME = params['TRANSNAME']
    ROUNDS = params['ROUNDS']
    ITER = params['ITER']
    STEPSIZE = params['STEPSIZE']
    wst = params['wst']
    
    # 获取数据维度
    n = xTr.shape[1]
    fold_size = n // k_folds
    valid_errors = []
    
    # 随机打乱数据索引
    indices = np.random.permutation(n)
    
    for i in range(k_folds):
        # 创建验证集索引
        valid_idx = indices[i * fold_size:(i + 1) * fold_size]
        train_idx = np.setdiff1d(indices, valid_idx)
        
        # 分割数据
        x_train, y_train = xTr[:, train_idx], yTr[:, train_idx]
        x_valid, y_valid = xTr[:, valid_idx], yTr[:, valid_idx]
        
        # 预处理数据
        [x_train_prep, x_valid_prep, _, _] = preprocess(x_train, x_valid)
        
        # 初始化权重
        w = initweights(wst)
        
        # 训练模型
        f = lambda w: deepnet(w, x_train_prep, y_train, wst, TRANSNAME)
        
        for j in range(ROUNDS):
            w = grdescent(f, w, STEPSIZE, ITER, 1e-8)
        
        # 评估验证集性能
        pred_valid = deepnet(w, x_valid_prep, [], wst, TRANSNAME)
        valid_err = np.sqrt(np.mean((pred_valid - y_valid) ** 2))
        valid_errors.append(valid_err)
        
        if verbose:
            print(f"Fold {i+1}/{k_folds}, Validation RMSE: {valid_err:.4f}")
    
    mean_valid_error = np.mean(valid_errors)
    std_valid_error = np.std(valid_errors)
    
    if verbose:
        print(f"Mean Validation RMSE: {mean_valid_error:.4f} ± {std_valid_error:.4f}")
    
    return mean_valid_error

def grid_search(xTr, yTr, xTe, yTe, param_grid, use_cv=True, k_folds=5, verbose=True):
    """
    网格搜索找到最佳参数
    
    参数:
    xTr, yTr: 训练数据
    xTe, yTe: 测试数据
    param_grid: 参数网格
    use_cv: 是否使用交叉验证
    k_folds: 交叉验证折数
    verbose: 是否打印详细信息
    
    返回:
    best_params: 最佳参数
    """
    d = xTr.shape[0]  # 输入特征维度
    
    # 生成所有参数组合
    all_params = []
    for transname in param_grid['TRANSNAME']:
        for rounds in param_grid['ROUNDS']:
            for iter_count in param_grid['ITER']:
                for stepsize in param_grid['STEPSIZE']:
                    for hidden_layers in param_grid['hidden_layers']:
                        # 构建完整的网络结构
                        wst = np.array([1] + hidden_layers + [d])
                        
                        params = {
                            'TRANSNAME': transname,
                            'ROUNDS': rounds,
                            'ITER': iter_count,
                            'STEPSIZE': stepsize,
                            'wst': wst
                        }
                        
                        all_params.append(params)
    
    if verbose:
        print(f"Total parameter combinations: {len(all_params)}")
    
    # 评估所有参数组合
    best_error = float('inf')
    best_params = None
    
    for i, params in enumerate(all_params):
        if verbose:
            print(f"\nEvaluating combination {i+1}/{len(all_params)}:")
        
        if use_cv:
            # 使用交叉验证评估
            error = cross_validate(xTr, yTr, params, k_folds, verbose)
        else:
            # 直接在测试集上评估
            _, error, _ = evaluate_model(xTr, yTr, xTe, yTe, params, verbose)
        
        # 更新最佳参数
        if error < best_error:
            best_error = error
            best_params = params
            
            if verbose:
                print(f"New best parameters found! Error: {best_error:.4f}")
                print(f"Parameters: {best_params}")
    
    return best_params

# 扩展get_transition_func函数来支持Leaky ReLU和GELU
def add_activation_functions():
    """
    向get_transition_func.py添加新的激活函数定义
    """
    # 这个函数只是为了显示代码，实际使用时需要修改get_transition_func.py文件
    leaky_relu_code = """
    elif transType.lower() == 'leaky_relu':
        alpha = 0.01  # 泄漏系数
        trans_func = lambda z: np.maximum(alpha * z, z)
        trans_func_der = lambda z: np.where(z > 0, 1, alpha)
    """
    
    gelu_code = """
    elif transType.lower() == 'gelu':
        # 高斯误差线性单元激活函数 (GELU)
        trans_func = lambda z: 0.5 * z * (1 + np.tanh(np.sqrt(2 / np.pi) * (z + 0.044715 * z**3)))
        # GELU的近似导数
        trans_func_der = lambda z: 0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (z + 0.044715 * z**3))) + \
                                 0.5 * z * (1 - np.tanh(np.sqrt(2 / np.pi) * (z + 0.044715 * z**3))**2) * \
                                 np.sqrt(2 / np.pi) * (1 + 3 * 0.044715 * z**2)
    """
    
    print("需要添加到get_transition_func.py的代码:")
    print(leaky_relu_code)
    print(gelu_code)

if __name__ == '__main__':
    # 加载数据
    xTr = np.loadtxt("data/xTr.csv", delimiter=",")
    yTr = np.loadtxt("data/yTr.csv", delimiter=",").reshape(1, -1)
    xTe = np.loadtxt("data/xTe.csv", delimiter=",")
    yTe = np.loadtxt("data/yTe.csv", delimiter=",").reshape(1, -1)
    
    # 定义参数网格
    param_grid = {
        'TRANSNAME': ['sigmoid', 'tanh', 'relu', 'leaky_relu', 'gelu'],  # 包括新添加的激活函数
        'ROUNDS': [100, 200, 300],
        'ITER': [30, 50, 70],
        'STEPSIZE': [0.005, 0.01, 0.05],
        'hidden_layers': [
            [20],             # 1层隐藏层
            [20, 20],         # 2层隐藏层
            [20, 20, 20],     # 3层隐藏层
            [30, 20, 10],     # 递减宽度
            [10, 20, 10]      # 沙漏形
        ]
    }
    
    # 为了快速测试，可以使用一个小的参数子集
    # 下面是一个更小的参数网格示例
    small_param_grid = {
        'TRANSNAME': ['sigmoid', 'relu'],
        'ROUNDS': [200],
        'ITER': [50],
        'STEPSIZE': [0.01],
        'hidden_layers': [
            [20, 20],
            [20, 20, 20]
        ]
    }
    
    # 选择是否使用小参数网格（用于快速测试）
    use_small_grid = True
    active_grid = small_param_grid if use_small_grid else param_grid
    
    print("开始参数搜索...")
    best_params = grid_search(xTr, yTr, xTe, yTe, active_grid, use_cv=True, k_folds=5, verbose=True)
    
    print("\n最佳参数:")
    print(best_params)
    
    # 使用最佳参数在完整测试集上评估
    train_err, test_err, train_time = evaluate_model(xTr, yTr, xTe, yTe, best_params, verbose=True)
    
    print(f"\n使用最佳参数的最终结果:")
    print(f"训练RMSE: {train_err:.4f}")
    print(f"测试RMSE: {test_err:.4f}")
    print(f"训练时间: {train_time:.2f}秒")
    
    # 保存最佳参数
    with open('best_parameters.pickle', 'wb') as f:
        pickle.dump(best_params, f)
    
    print("\n最佳参数已保存至'best_parameters.pickle'")