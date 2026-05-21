import numpy as np
import matplotlib.pyplot as plt
import stumpy


def compute_matrix_profile(ts, window_size, excl_zone=None):
    """
    手动计算矩阵轮廓（Matrix Profile）和索引（Matrix Profile Index）
    参数：
        ts: 时间序列（1D numpy数组）
        window_size: 子序列窗口大小
        excl_zone: 排除区域（默认window_size//2）
    返回：
        mp: 矩阵轮廓数组
        mpi: 矩阵轮廓索引数组
    """
    n = len(ts)
    num_subseq = n - window_size + 1
    if excl_zone is None:
        excl_zone = window_size // 2

    mp = np.full(num_subseq, np.inf)
    mpi = np.full(num_subseq, -1)

    for i in range(num_subseq):
        subseq = ts[i:i + window_size]
        min_dist = np.inf
        min_idx = -1

        for j in range(num_subseq):
            # 跳过排除区域和自身匹配（核心：excl_zone生效关键）
            if abs(i - j) < excl_zone or i == j:
                continue
            comp_subseq = ts[j:j + window_size]
            dist = np.sqrt(np.sum((subseq - comp_subseq) ** 2))
            if dist < min_dist:
                min_dist = dist
                min_idx = j

        mp[i] = min_dist
        mpi[i] = min_idx

    return mp, mpi


def compute_arc_curve(mpi, num_subseq, excl_zone):
    """
    计算弧曲线（Arc Curve）
    参数：
        mpi: 矩阵轮廓索引数组
        num_subseq: 子序列总数
        excl_zone: 排除区域（来自excl_factor）
    返回：
        ac: 弧曲线数组
    """
    ac = np.zeros(num_subseq)
    for i in range(num_subseq):
        j = int(mpi[i])
        # 跳过无效匹配（mpi=-1）和排除区域内的匹配
        if j == -1 or abs(i - j) < excl_zone:
            continue
        ac[j] += 1
    return ac


def compute_cac(ac):
    """
    计算校正弧曲线（Corrected Arc Curve）
    参数：
        ac: 弧曲线数组
    返回：
        cac: 校正弧曲线数组（0-1区间，越小越可能是边界）
    """
    max_ac = np.max(ac) if np.max(ac) != 0 else 1  # 避免除零
    cac = 1 - (ac / max_ac)
    return cac


def find_boundaries(window_size, ac, cac, n_regimes, excl_zone):
    """
    改进：加入排除区域逻辑，确保边界点之间距离≥excl_zone
    参数：
        window_size: 窗口大小
        ac: 弧曲线数组
        cac: 校正弧曲线数组
        n_regimes: 期望分割段数
        excl_zone: 排除区域（关联excl_factor）
    返回：
        segments: 分割边界的索引（时间序列中的位置）
    """
    # 复制cac，后续用于标记已选中的排除区域
    cac_copy = cac.copy()
    boundary_indices = []

    for _ in range(n_regimes - 1):
        # 找到当前cac最小值的位置（排除已标记的区域）
        min_idx = np.argmin(cac_copy)
        if np.isinf(cac_copy[min_idx]):  # 无更多有效边界
            break
        boundary_indices.append(min_idx)

        # 标记该边界的排除区域，避免后续选中过近的点（核心改进）
        start_excl = max(0, min_idx - excl_zone)
        end_excl = min(len(cac_copy), min_idx + excl_zone)
        cac_copy[start_excl:end_excl] = np.inf  # 排除区域设为无穷大，不再被选中

    # 输出调试信息
    print("Boundary indices (子序列起始位置):", boundary_indices)
    for idx in boundary_indices:
        start = max(0, idx - 5)
        end = min(len(ac), idx + 6)
        print(f"Index {idx} 周围AC值: {ac[start:end]}")
        print(f"Index {idx} 周围CAC值: {cac[start:end]}")

    # 转换为时间序列中的绝对位置（子序列中心）
    segments = np.array(boundary_indices) + (window_size // 2)
    segments = np.sort(segments)
    return segments


# def fluss(ts, window_size, n_regimes=3, excl_factor=1):
#     # 1. 计算排除区域（excl_factor直接生效）
#     excl_zone = window_size * excl_factor
#     print(f"当前排除区域大小: {excl_zone} (window_size={window_size}, excl_factor={excl_factor})")
#
#     # 2. 计算矩阵轮廓和索引
#     mp, mpi = compute_matrix_profile(ts, window_size, excl_zone)
#
#     # 3. 计算弧曲线和校正弧曲线
#     num_subseq = len(mp)
#     ac = compute_arc_curve(mpi, num_subseq, excl_zone)
#     cac = compute_cac(ac)
#
#     # 4. 改进：传递excl_zone给边界选择，确保参数生效
#     segments = find_boundaries(window_size, ac, cac, n_regimes, excl_zone)
#     print(f"FLUSS检测的边界（时间序列位置）：{segments}")
#
#     # 5. 可视化（修正参数顺序）
#     fluss_visualize(ts, mp, mpi, ac, cac, segments, window_size)
#     return ts, segments

def fluss(ts, window_size, n_regimes=3, excl_factor=1):
    # 2. 计算 Matrix Profile
    # stumpy.stump 返回一个矩阵，第一列是距离(MP)，第二列是索引(MPI)
    mp_res = stumpy.stump(ts, window_size)

    # 提取 Matrix Profile Index (MPI)，这是 FLUSS 的输入
    mpi = mp_res[:, 1]
    mp = mp_res[:, 0]

    # 3. 计算 FLUSS
    # L: 子序列长度 (通常设为 m)
    # n_regimes: 你期望找到几个分割点 (如果不确定，可以看曲线最低点)
    # excl_factor: 排除区域因子 (默认 5)
    cac, regime_locations = stumpy.fluss(mpi, L=window_size, n_regimes=n_regimes, excl_factor=excl_factor)
    print(f"检测到的分割点位置: {regime_locations}")
    fluss_visualize(ts, mp, mpi, None, cac, regime_locations, window_size)
    return ts, regime_locations


def fluss_visualize(ts, mp=None, mpi=None, ac=None, cac=None, segments=None, window_size=None):
    """根据参数是否为None来控制打印子图的数量"""
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
    plt.rcParams['axes.unicode_minus'] = False

    # 计算实际需要显示的子图数量
    subplot_count = 1  # 至少显示原始时间序列
    if mp is not None:
        subplot_count += 1
    if mpi is not None:
        subplot_count += 1
    if ac is not None:
        subplot_count += 1
    if cac is not None:
        subplot_count += 1

    plt.figure(figsize=(12, 3 * subplot_count))
    plt.suptitle('FLUSS时间序列分割空调状态', fontsize=16, fontweight='bold')

    current_subplot = 1

    # 子图1：原始时间序列 + 边界
    plt.subplot(subplot_count, 1, current_subplot)
    plt.plot(ts, label='原始时间序列')
    if segments is not None:
        for seg in segments:
            label = '边界' if seg == segments[0] else ""
            plt.axvline(x=seg, color='red', linestyle='--', linewidth=2, label=label)
    plt.title('分割结果')
    plt.legend()
    current_subplot += 1

    # 子图2：矩阵轮廓（如果提供了mp数据）
    if mp is not None:
        x_offset = window_size // 2 if window_size is not None else 0
        x_indices = np.arange(len(mp)) + x_offset
        plt.subplot(subplot_count, 1, current_subplot)
        plt.plot(x_indices, mp, label='矩阵轮廓（MP）', color='blue')
        plt.title('Matrix Profile')
        if len(x_indices) > 0:
            plt.xlim(x_indices[0], x_indices[-1])
        plt.legend()
        current_subplot += 1

    # 子图3：矩阵轮廓索引（如果提供了mpi数据）
    if mpi is not None:
        plt.subplot(subplot_count, 1, current_subplot)
        plt.plot(mpi, label='矩阵轮廓索引（MPI）', color='green')
        plt.title('Matrix Profile Index')
        plt.legend()
        current_subplot += 1

    # 子图4：弧曲线（如果提供了ac数据）
    if ac is not None:
        x_offset = window_size // 2 if window_size is not None else 0
        x_indices = np.arange(len(ac)) + x_offset
        plt.subplot(subplot_count, 1, current_subplot)
        plt.plot(x_indices, ac, label='弧曲线（AC）', color='orange')
        plt.title('Arc Curve')
        if len(x_indices) > 0:
            plt.xlim(x_indices[0], x_indices[-1])
        plt.legend()
        current_subplot += 1

    # 子图5：校正弧曲线（如果提供了cac数据）
    if cac is not None:
        x_offset = window_size // 2 if window_size is not None else 0
        x_indices = np.arange(len(cac)) + x_offset
        plt.subplot(subplot_count, 1, current_subplot)
        plt.plot(x_indices, cac, label='校正弧曲线（CAC）', color='purple')
        plt.title('Corrected Arc Curve')
        if len(x_indices) > 0:
            plt.xlim(x_indices[0], x_indices[-1])
        plt.legend()

    plt.tight_layout()

    # 捕获键盘输入，询问是否保存结果
    user_input = input("是否保存可视化图片到 analyze_result 文件夹？(y/n): ")
    if user_input.lower() == 'y':
        import os
        from datetime import datetime

        # 创建保存目录
        save_dir = r'../../ukdale_disaggregate/analyze_result'
        os.makedirs(save_dir, exist_ok=True)

        # 生成时间戳用于文件名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # 保存图片
        filename = os.path.join(save_dir, f'fluss_analysis_{timestamp}.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()

        print(f"可视化图片已保存到 {filename}")
    else:
        plt.show()



# 测试代码（验证excl_factor生效）
if __name__ == "__main__":
    # 生成测试时间序列（含3个分段）
    np.random.seed(3407)
    ts1 = np.random.normal(loc=5, scale=0.5, size=200)
    ts2 = np.random.normal(loc=10, scale=0.8, size=300)
    ts3 = np.random.normal(loc=3, scale=0.3, size=200)
    ts = np.concatenate([ts1, ts2, ts3])

    # 测试不同excl_factor（观察边界变化）
    segment = fluss(ts, window_size=100, n_regimes=3, excl_factor=1)  # 排除区域20
    fluss(ts, window_size=20, n_regimes=3, excl_factor=2)  # 排除区域40（边界会变化）
    # cac, regime_locs = stumpy.fluss(ts, 20, n_regimes=3, excl_factor=1)

    # # 注意：stumpy.fluss返回的cac已经是计算好的，可以直接可视化
    # plt.figure(figsize=(12, 6))
    # plt.plot(cac, color='purple')
    # plt.title('Corrected Arc Curve (CAC) - StuMPy版本')
    # plt.xlabel('子序列索引')
    # plt.ylabel('CAC值')
    # plt.grid(True, alpha=0.3)
    # plt.show()
