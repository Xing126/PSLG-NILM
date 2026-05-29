# 基于双阶段无监督聚类的电器工作状态自适应检测算法说明书

在处理用电设备的功率时间序列时，传统的“硬阈值”判定方法常面临两个核心痛点：
1. **静默/休眠功率不为0**：电器在休眠或待机状态下仍有微弱功率波动，无法用 `Power > 0` 直接切分。
2. **工作区间功率短暂跌落**：由于电器内部的物理特性（如压缩机间歇停机、温控棒到温断电、洗涤间歇停转），工作期间的功率会短时跌落至待机水平，导致识别出的工作区间断断续续、严重碎片化。

本算法提供了一套完整的**无监督、数据驱动（Data-Driven）的自适应解决方案**。算法分为两个阶段：
* **第一阶段（幅值域聚类）**：基于一维 K-Means 自动学习静态功率阈值，分离“工作”与“休眠”状态。
* **第二阶段（时域聚类）**：提取破碎区间的时域特征，利用高斯混合模型（GMM）自动学习“断开容忍时间（$T_{drop}$）”与“最小有效工作时间（$T_{min\_work}$）”，并在时间轴上进行形态学平滑，输出稳定、连续的电器工作区间。

---

## 架构总览与工作流

本算法的完整数据管道（Pipeline）如下图所示。通过将幅值域聚类与时域聚类级联，实现从原始嘈杂功率序列到精确工作状态区间的端到端提取。



核心步骤如下：
1. **数据预处理**：对原始一维功率序列进行移动平均（SMA），消除瞬时高频高斯噪声。
2. **幅值域 K-Means 聚类**：将平滑后的功率数据视作一维特征空间，训练 $K=2$ 的 K-Means 模型，自动计算分类阈值。
3. **时域特征提取**：利用幅值阈值对序列进行初次硬二值化切分，提取所有连续低于阈值的“零区间（Zero Segments）”长度和连续高于阈值的“一区间（One Segments）”长度。
4. **时域 GMM 聚类**：
   - 对“零区间”长度进行 $K=2$ 的高斯混合模型（GMM）聚类，区分“正常间歇跌落”与“真正停机休眠”，从而自适应学习出 **$T_{drop}$**。
   - 对“一区间”长度进行 $K=2$ 的 GMM 聚类，区分“瞬时电涌/脉冲噪声”与“真正开机工作”，从而自适应学习出 **$T_{min\_work}$**。
5. **形态学时域平滑**：利用学习到的 $T_{drop}$ 和 $T_{min\_work}$ 参数，对初次二值化信号进行一维形态学闭运算（Closing Operation）及长度清洗，最终输出稳定连续的工作区间。

---

## 数学原理

### 1. 幅值域一维 K-Means 聚类
假定输入的平滑功率时间序列为 $X = \{x_1, x_2, \dots, x_N\}$。算法通过迭代将数据划分为两个簇 $C_0$（休眠簇）和 $C_1$（工作簇），其目标是最小化组内平方误差（SSE）：

$$\arg\min_{C} \sum_{i=0}^{1} \sum_{x \in C_i} ||x - \mu_i||^2$$

其中 $\mu_0$ 和 $\mu_1$ 分别为休眠簇和工作簇的中心点（满足 $\mu_0 < \mu_1$）。
自动生成的**动态功率分类阈值（Threshold）**定义为两类中心的几何中点：

$$\text{Threshold} = \frac{\mu_0 + \mu_1}{2}$$

### 2. 时域高斯混合模型（GMM）聚类
以学习 $T_{drop}$ 为例。由于时间跨度通常呈现出偏态分布（短时间的间歇跌落可能集中在几秒到十几秒，而长时间的休眠通常长达数千秒），算法首先对提取的时间长度集合 $D_{zero}$ 进行对数变换 $Y = \ln(D_{zero} + 1)$，以提高分布的紧凑度。

随后，使用两个高斯分布组成的混合模型对 $Y$ 的概率密度进行拟合：

$$p(y) = \pi_A \mathcal{N}(y | \mu_A, \sigma_A^2) + \pi_B \mathcal{N}(y | \mu_B, \sigma_B^2)$$

其中 $\pi$ 为混合系数，$\mathcal{N}$ 为高斯概率密度函数。设 $\mu_A < \mu_B$，则簇 $A$ 代表“间歇性短时跌落”，簇 $B$ 代表“真正停机休眠”。
通过将簇 $A$ 中样本对应回原始时间域，其最大值即定义为算法自适应学到的**断开容忍时间（$T_{drop}$）**：

$$T_{drop} = \max \left( \{d \in D_{zero} \mid y = \ln(d+1) \in \text{Cluster } A\} \right)$$

同理，对高于阈值的持续时间集合 $D_{one}$ 运行相同的流程，识别出短时噪声簇，该簇的最大值即定义为**最小有效工作时间（$T_{min\_work}$）**。

### 3. 一维形态学闭运算约束
形态学闭运算（Closing）在数学上定义为**先膨胀（Dilation）后腐蚀（Erosion）**。

对于初次二值化后的状态信号 $B(t) \in \{0, 1\}$，使用长度为 $T_{drop}$ 的结构元素 $S$ 进行处理：
* **膨胀阶段**：$B \oplus S$。凡是处于工作状态（1）的点，其边界在时间轴上向外扩展。如果两个工作状态之间的“0（跌落空隙）”小于 $T_{drop}$，它们将被连接合并。
* **腐蚀阶段**：$(B \oplus S) \ominus S$。将合并后的信号边界整体向内等距离缩减，确保不人为夸大电器整体的工作时长。

---

## 完整 Python 代码实现

以下代码包含了完整的算法管道：从模拟一段带有**噪声、待机功率不为0、且包含间歇跌落**的复杂功率序列开始，自动训练并提取出最终的工作区间。

```python
import numpy as np
import scipy.ndimage as ndimage
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

class AdaptiveApplianceDetector:
    def __init__(self, fs=1):
        """
        自适应电器状态检测器
        :param fs: 数据采样率 (Hz)，即每秒采集多少个数据点
        """
        self.fs = fs
        self.threshold = None
        self.t_drop = None
        self.t_min_work = None
        
    def _moving_average(self, data, window_size=5):
        """一维滑动平均平滑"""
        return np.convolve(data, np.ones(window_size)/window_size, mode='same')

    def fit_parameters(self, raw_power_series):
        """
        双阶段聚类训练阶段：自动学习幅值阈值和时域容忍度参数
        """
        print("开始双阶段聚类自适应学习...")
        
        # 1. 预处理平滑
        smoothed_power = self._moving_average(raw_power_series, window_size=5)
        
        # 2. 阶段一：一维 K-Means 学习功率阈值
        X_amp = smoothed_power.reshape(-1, 1)
        kmeans = KMeans(n_clusters=2, random_state=42, n_init='auto').fit(X_amp)
        centers = np.sort(kmeans.cluster_centers_.flatten())
        self.threshold = (centers[0] + centers[1]) / 2
        print(f"  [幅值域] 学习完成。自动分类阈值设定为: {self.threshold:.2f} W")
        
        # 初次硬二值化切分
        binary_status = (smoothed_power > self.threshold).astype(int)
        
        # 3. 提取时间片段集合
        # 提取低于阈值的零片段
        labeled_zeros, num_zeros = ndimage.label(1 - binary_status)
        zero_durations = np.array([np.sum(labeled_zeros == i) / self.fs for i in range(1, num_zeros + 1)])
        
        # 提取高于阈值的一片段
        labeled_ones, num_ones = ndimage.label(binary_status)
        one_durations = np.array([np.sum(labeled_ones == i) / self.fs for i in range(1, num_ones + 1)])
        
        # 4. 阶段二(A)：GMM 聚类学习 T_drop (分析零区间)
        if len(zero_durations) >= 2:
            X_zero = np.log1p(zero_durations).reshape(-1, 1) # 对数化消除长尾偏态影响
            gmm_zero = GaussianMixture(n_components=2, random_state=42).fit(X_zero)
            labels_zero = gmm_zero.predict(X_zero)
            
            # 均值小的是“间歇跌落簇”，均值大的是“停机休眠簇”
            short_cluster_idx = np.argmin(gmm_zero.means_.flatten())
            short_drop_durations = zero_durations[labels_zero == short_cluster_idx]
            
            self.t_drop = np.max(short_drop_durations) if len(short_drop_durations) > 0 else 5
        else:
            self.t_drop = 10 # 样本不足时的鲁棒保底值
            
        # 5. 阶段二(B)：GMM 聚类学习 T_min_work (分析一区间)
        if len(one_durations) >= 2:
            X_one = np.log1p(one_durations).reshape(-1, 1)
            gmm_one = GaussianMixture(n_components=2, random_state=42).fit(X_one)
            labels_one = gmm_one.predict(X_one)
            
            # 均值小的是“噪声/电涌簇”，均值大的是“真正工作簇”
            noise_cluster_idx = np.argmin(gmm_one.means_.flatten())
            noise_durations = one_durations[labels_one == noise_cluster_idx]
            
            self.t_min_work = np.max(noise_durations) if len(noise_durations) > 0 else 5
        else:
            self.t_min_work = 15 # 保底值
            
        print(f"  [时域] 学习完成。断开容忍时间 (T_drop): {self.t_drop:.2f} 秒")
        print(f"  [时域] 学习完成。最小有效工作时间 (T_min_work): {self.t_min_work:.2f} 秒")
        return self

    def detect(self, raw_power_series):
        """
        在线/离线状态推理阶段：提取最终稳定连续的工作区间
        """
        if self.threshold is None or self.t_drop is None or self.t_min_work is None:
            raise ValueError("检测器尚未进行参数训练，请先调用 fit_parameters()。")
            
        points_drop = int(self.t_drop * self.fs)
        points_min_work = int(self.t_min_work * self.fs)
        
        # 1. 基础二值化
        smoothed_power = self._moving_average(raw_power_series, window_size=5)
        binary_status = (smoothed_power > self.threshold).astype(int)
        
        # 2. 时域约束：一维形态学闭运算桥接短时跌落
        structuring_element = np.ones(max(1, points_drop))
        smoothed_status = ndimage.binary_closing(binary_status, structure=structuring_element).astype(int)
        
        # 3. 长度清洗：消除短时高频噪声区间
        labeled_segments, num_features = ndimage.label(smoothed_status)
        final_status = np.zeros_like(smoothed_status)
        intervals = []
        
        for i in range(1, num_features + 1):
            segment_indices = np.where(labeled_segments == i)[0]
            
            # 仅保留长度大于等于最小有效工作时间的区间
            if len(segment_indices) >= points_min_work:
                final_status[segment_indices] = 1
                intervals.append({
                    "start_time_sec": float(segment_indices[0] / self.fs),
                    "end_time_sec": float(segment_indices[-1] / self.fs),
                    "duration_sec": float(len(segment_indices) / self.fs)
                })
                
        return final_status, intervals

# ==========================================
# 算法运行示例与模拟验证
# ==========================================
if __name__ == "__main__":
    # 设定采样率为 1Hz (每秒一个采样点)
    FS = 1 
    
    print("--- 1. 模拟生成复杂的智能电表功率数据 ---")
    np.random.seed(42)
    
    # 基础基线：模拟电器休眠状态（基线功率在 15W 左右波动）
    total_duration = 4000
    timeline = np.ones(total_duration) * 15 + np.random.normal(0, 1, total_duration)
    
    # 注入真实的连续工作区间 1：时间[500-1500]秒，功率150W。
    # 模拟其在 [800-815]秒（持续15秒）发生间歇性功率暴跌至10W（模拟温控断电）
    timeline[500:1500] = 150 + np.random.normal(0, 5, 1000)
    timeline[800:1515] = 12 + np.random.normal(0, 1, 15)
    
    # 注入真实的连续工作区间 2：时间[2500-3500]秒，功率150W。
    # 模拟其在 [3000-3020]秒（持续20秒）发生间歇性功率暴跌至12W
    timeline[2500:3500] = 150 + np.random.normal(0, 5, 1000)
    timeline[3000:3020] = 12 + np.random.normal(0, 1, 20)
    
    # 注入瞬时高频尖峰噪声（模拟电涌：持续3秒的160W高功率）
    timeline[1800:1803] = 160
    timeline[2100:2102] = 140

    print(f"数据模拟生成完毕，时间序列总长度: {len(timeline)} 秒。")
    print("-" * 50)

    # --- 2. 初始化检测器并训练 ---
    detector = AdaptiveApplianceDetector(fs=FS)
    detector.fit_parameters(timeline)
    
    print("-" * 50)
    print("--- 3. 执行状态推理与区间提取 ---")
    status_series, work_intervals = detector.detect(timeline)
    
    print(f"\n检测完成！共成功定位 {len(work_intervals)} 个平滑有效的工作区间：")
    for idx, interval in enumerate(work_intervals):
        print(f" └─ 区间 {idx+1}: 开始时间 = {interval['start_time_sec']:.1f}s, "
              f"结束时间 = {interval['end_time_sec']:.1f}s, "
              f"实际计算工作长度 = {interval['duration_sec']:.1f}秒")