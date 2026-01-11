import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

class SystemModel:
    def __init__(self, num_users, num_tasks, num_servers):
        self.num_users = num_users
        self.num_tasks = num_tasks
        self.num_servers = num_servers
        
        # 初始化系统参数
        self.initialize_parameters()
    
    def initialize_parameters(self):
        # 用户设备参数 - 进一步降低本地计算能力，增加本地执行难度
        self.f_local = np.random.uniform(0.05, 0.2, (self.num_users,)) * 10**9  # 本地计算能力 (CPU cycles/s)
        self.P_local = np.random.uniform(0.8, 2.0, (self.num_users,))  # 本地计算功率 (W)，增加本地能耗
        
        # 任务参数 - 进一步增加任务复杂度，使得本地执行更困难
        self.data_sizes = np.random.uniform(2, 10, (self.num_users, self.num_tasks)) * 10**6  # 任务数据量 (bits)，适当降低数据量
        self.cpu_cycles = np.random.uniform(800, 2000, (self.num_users, self.num_tasks)) * 10**6  # 任务所需CPU周期，增加复杂度
        self.deadlines = np.random.uniform(100, 500, (self.num_users, self.num_tasks)) * 10**-3  # 任务截止时间 (s)
        
        # 边缘服务器参数 - 大幅增强边缘计算能力
        self.f_edge = np.random.uniform(80, 150, (self.num_servers,)) * 10**9  # 边缘计算能力 (CPU cycles/s)
        self.P_transmit = np.random.uniform(0.05, 0.2, (self.num_users, self.num_servers))  # 传输功率 (W)，降低传输能耗
        
        # 通信参数 - 大幅提高带宽和信道增益，降低传输成本
        self.bandwidth = np.random.uniform(150, 300, (self.num_users, self.num_servers)) * 10**6  # 带宽 (Hz)，增加带宽
        self.channel_gains = np.random.uniform(5e-8, 5e-6, (self.num_users, self.num_servers))  # 信道增益，增加增益
        self.noise_power = 10**-13  # 噪声功率 (W/Hz)
        
        # 边缘服务器单位计算价格 (初始值) - 降低价格，使得边缘计算更有吸引力
        # 1 CPU hour = 3.6e12 cycles (1 GHz for 1 hour)
        # 价格范围为约$0.01-$0.1/CPU hour，使边缘计算成本更低
        self.prices = np.random.uniform(2.78e-10, 2.78e-9, (self.num_servers,))  # 价格 ($/CPU cycle)

class User:
    def __init__(self, user_id, system_model):
        self.user_id = user_id
        self.system = system_model
        
    def calculate_local_cost(self, task_id):
        """计算本地执行成本"""
        f = self.system.f_local[self.user_id]
        P = self.system.P_local[self.user_id]
        C = self.system.cpu_cycles[self.user_id, task_id]
        D = self.system.data_sizes[self.user_id, task_id]
        
        # 计算时间 (秒)
        t_local = C / f
        # 能耗 (焦耳)
        E_local = P * t_local
        
        # 成本函数：时间+加权能耗（能耗权重设为1e-3，平衡时间和能耗的影响）
        cost_local = t_local + 1e-3 * E_local
        
        return cost_local
    
    def calculate_edge_cost(self, task_id, server_id, price):
        """计算边缘执行成本"""
        # 传输时间和能耗
        B = self.system.bandwidth[self.user_id, server_id]
        h = self.system.channel_gains[self.user_id, server_id]
        P_tx = self.system.P_transmit[self.user_id, server_id]
        N0 = self.system.noise_power
        D = self.system.data_sizes[self.user_id, task_id]
        
        # 传输速率 (Shannon公式)
        R = B * np.log2(1 + (P_tx * h) / (B * N0))
        # 传输时间 (秒)
        t_transmit = D / R
        # 传输能耗 (焦耳)
        E_transmit = P_tx * t_transmit
        
        # 边缘计算时间和成本
        f_edge = self.system.f_edge[server_id]
        C = self.system.cpu_cycles[self.user_id, task_id]
        t_compute = C / f_edge  # 计算时间 (秒)
        # 计算成本 ($)
        cost_compute = price * C
        
        # 总边缘成本：时间+加权能耗+计算成本（能耗权重设为1e-3，平衡各成本项）
        cost_edge = t_transmit + t_compute + 1e-3 * E_transmit + cost_compute
        
        return cost_edge
    
    def optimize_offloading(self, prices):
        """优化用户的卸载决策"""
        offloading_decision = np.zeros((self.system.num_tasks, self.system.num_servers + 1))  # 0表示本地，1~num_servers表示边缘服务器
        
        for task_id in range(self.system.num_tasks):
            # 计算本地执行成本
            local_cost = self.calculate_local_cost(task_id)
            
            # 计算各边缘服务器执行成本
            edge_costs = []
            for server_id in range(self.system.num_servers):
                edge_cost = self.calculate_edge_cost(task_id, server_id, prices[server_id])
                edge_costs.append(edge_cost)
            
            # 找到最小成本的执行位置
            all_costs = [local_cost] + edge_costs
            min_cost_idx = np.argmin(all_costs)
            
            # 设置卸载决策
            offloading_decision[task_id, min_cost_idx] = 1
        
        return offloading_decision

class EdgeServer:
    def __init__(self, server_id, system_model):
        self.server_id = server_id
        self.system = system_model
    
    def calculate_revenue(self, prices, all_offloading_decisions):
        """计算边缘服务器的收益"""
        revenue = 0.0
        
        for user_id in range(self.system.num_users):
            for task_id in range(self.system.num_tasks):
                if all_offloading_decisions[user_id][task_id, self.server_id + 1] == 1:  # 检查是否卸载到本服务器
                    C = self.system.cpu_cycles[user_id, task_id]
                    revenue += prices[self.server_id] * C
        
        return revenue
    
    def calculate_profit(self, prices, all_offloading_decisions):
        """计算边缘服务器的利润"""
        # 计算收益
        revenue = self.calculate_revenue(prices, all_offloading_decisions)
        
        # 计算成本：仅考虑维护成本，简化能源成本模型
        # 服务器维护成本：与计算资源使用成正比，但比例降低
        maintenance_cost = 0.0
        
        for user_id in range(self.system.num_users):
            for task_id in range(self.system.num_tasks):
                if all_offloading_decisions[user_id][task_id, self.server_id + 1] == 1:
                    C = self.system.cpu_cycles[user_id, task_id]
                    # 维护成本：0.05 * 收益（降低比例，使利润为正）
                    maintenance_cost += 0.05 * prices[self.server_id] * C
        
        profit = revenue - maintenance_cost
        
        return profit

class StackelbergGame:
    def __init__(self, system_model):
        self.system = system_model
        self.users = [User(user_id, system_model) for user_id in range(system_model.num_users)]
        self.servers = [EdgeServer(server_id, system_model) for server_id in range(system_model.num_servers)]
    
    def followers_response(self, prices):
        """跟随者（用户）的响应：根据价格优化卸载决策"""
        all_offloading_decisions = []
        for user in self.users:
            offloading_decision = user.optimize_offloading(prices)
            all_offloading_decisions.append(offloading_decision)
        return all_offloading_decisions
    
    def leaders_problem(self, prices):
        """领导者（边缘服务器）的问题：优化价格最大化利润"""
        all_offloading_decisions = self.followers_response(prices)
        total_profit = 0.0
        
        for server in self.servers:
            profit = server.calculate_profit(prices, all_offloading_decisions)
            total_profit += profit
        
        # 由于minimize函数是求最小值，所以返回负的利润
        return -total_profit
    
    def solve(self, max_iterations=500, tolerance=1e-8):
        """求解Stackelberg均衡"""
        # 初始价格
        current_prices = self.system.prices.copy()
        
        # 迭代求解
        for iteration in range(max_iterations):
            # 求解领导者问题
            result = minimize(self.leaders_problem, current_prices, method='SLSQP', 
                           bounds=[(1e-10, 1e-8)]*self.system.num_servers, 
                           options={'maxiter': 100, 'disp': False})
            new_prices = result.x
            
            # 检查收敛性
            price_change = np.linalg.norm(new_prices - current_prices)
            if price_change < tolerance:
                print(f"价格收敛，迭代次数: {iteration+1}")
                break
            
            current_prices = new_prices.copy()
        else:
            print(f"达到最大迭代次数 {max_iterations}")
        
        # 最终的卸载决策
        final_offloading_decisions = self.followers_response(current_prices)
        
        return current_prices, final_offloading_decisions
    
    def evaluate_performance(self, prices, offloading_decisions):
        """评估系统性能"""
        total_cost = 0.0
        total_profit = 0.0
        total_offloaded_tasks = 0
        server_utilization = np.zeros(self.system.num_servers)
        
        # 计算用户总成本和卸载任务数
        for user_id, user in enumerate(self.users):
            for task_id in range(self.system.num_tasks):
                decision = offloading_decisions[user_id][task_id]
                if decision[0] == 1:  # 本地执行
                    cost = user.calculate_local_cost(task_id)
                else:  # 边缘执行
                    server_id = np.where(decision[1:] == 1)[0][0]
                    cost = user.calculate_edge_cost(task_id, server_id, prices[server_id])
                    total_offloaded_tasks += 1
                    # 计算服务器资源利用率
                    server_utilization[server_id] += self.system.cpu_cycles[user_id, task_id]
                total_cost += cost
        
        # 计算服务器资源利用率（归一化到总计算能力）
        total_server_capacity = np.sum(self.system.f_edge) * 1e-9  # GHz
        for server_id in range(self.system.num_servers):
            # 将CPU周期转换为GHz秒
            server_utilization[server_id] = (server_utilization[server_id] / 1e9) / total_server_capacity
        
        # 计算边缘服务器总利润
        for server in self.servers:
            total_profit += server.calculate_profit(prices, offloading_decisions)
        
        return {
            'total_cost': total_cost,
            'total_profit': total_profit,
            'total_offloaded_tasks': total_offloaded_tasks,
            'server_utilization': server_utilization,
            'offloading_rate': total_offloaded_tasks / (self.system.num_users * self.system.num_tasks)
        }

# 可视化结果
class Visualizer:
    @staticmethod
    def plot_offloading_decisions(offloading_decisions, num_users, num_tasks, num_servers):
        """绘制卸载决策矩阵"""
        fig, axes = plt.subplots(num_users, 1, figsize=(12, 3*num_users))
        if num_users == 1:
            axes = [axes]
        
        for user_id in range(num_users):
            ax = axes[user_id]
            decision_matrix = offloading_decisions[user_id][:, 1:]  # 只显示边缘服务器决策
            im = ax.imshow(decision_matrix, cmap='Blues', aspect='auto', vmin=0, vmax=1)
            
            ax.set_title(f'User {user_id+1} Offloading Decisions', fontsize=13, fontweight='bold')
            ax.set_xlabel('Edge Servers', fontsize=11)
            ax.set_ylabel('Tasks', fontsize=11)
            
            # 设置坐标轴刻度
            ax.set_xticks(np.arange(num_servers))
            ax.set_xticklabels([f'Server {i+1}' for i in range(num_servers)], fontsize=10)
            ax.set_yticks(np.arange(num_tasks))
            ax.set_yticklabels([f'Task {i+1}' for i in range(num_tasks)], fontsize=10)
            
            # 添加颜色条
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_ticks([0, 1])
            cbar.set_ticklabels(['Not Offloaded', 'Offloaded'])
            
            # 在每个单元格中显示数值
            for i in range(num_tasks):
                for j in range(num_servers):
                    text = ax.text(j, i, int(decision_matrix[i, j]), ha='center', va='center', 
                                  color='white' if decision_matrix[i, j] == 1 else 'black', 
                                  fontweight='bold', fontsize=10)
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_prices(initial_prices, final_prices):
        """绘制价格变化"""
        num_servers = len(initial_prices)
        x = np.arange(num_servers)
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 6))
        bars1 = ax.bar(x - width/2, initial_prices, width, label='Initial Prices', color='#3498db')
        bars2 = ax.bar(x + width/2, final_prices, width, label='Final Prices', color='#2ecc71')
        
        ax.set_title('Edge Server Prices Comparison', fontsize=14, fontweight='bold')
        ax.set_xlabel('Edge Servers', fontsize=12)
        ax.set_ylabel('Price ($/CPU cycle)', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels([f'Server {i+1}' for i in range(num_servers)], fontsize=11)
        ax.legend(fontsize=11)
        
        # 在柱状图上添加数值标签
        def add_labels(bars):
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.2e}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=9)
        
        add_labels(bars1)
        add_labels(bars2)
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_performance(initial_performance, final_performance):
        """绘制性能对比"""
        categories = ['Total User Cost', 'Total Server Profit']
        initial_values = list(initial_performance)
        final_values = list(final_performance)
        
        x = np.arange(len(categories))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x - width/2, initial_values, width, label='Initial')
        ax.bar(x + width/2, final_values, width, label='After Optimization')
        
        ax.set_title('System Performance Comparison')
        ax.set_ylabel('Value')
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend()
        
        plt.tight_layout()
        plt.show()

# 主函数
def main():
    # 初始化系统参数
    num_users = 3
    num_tasks = 4
    num_servers = 2
    
    # 创建系统模型
    system = SystemModel(num_users, num_tasks, num_servers)
    
    # 初始化Stackelberg博弈
    game = StackelbergGame(system)
    
    # 初始性能评估
    initial_prices = system.prices.copy()
    initial_offloading = game.followers_response(initial_prices)
    initial_performance = game.evaluate_performance(initial_prices, initial_offloading)
    
    # 求解Stackelberg均衡
    final_prices, final_offloading = game.solve()
    
    # 最终性能评估
    final_performance = game.evaluate_performance(final_prices, final_offloading)
    
    # 打印结果
    print("=== Stackelberg博弈结果 ===")
    print(f"初始价格: {initial_prices}")
    print(f"最终价格: {final_prices}")
    print(f"初始用户总成本: {initial_performance['total_cost']:.4f}")
    print(f"最终用户总成本: {final_performance['total_cost']:.4f}")
    print(f"初始服务器总利润: {initial_performance['total_profit']:.6f}")
    print(f"最终服务器总利润: {final_performance['total_profit']:.6f}")
    print(f"初始卸载率: {initial_performance['offloading_rate']:.2%}")
    print(f"最终卸载率: {final_performance['offloading_rate']:.2%}")
    print(f"初始卸载任务数: {initial_performance['total_offloaded_tasks']}")
    print(f"最终卸载任务数: {final_performance['total_offloaded_tasks']}")
    print(f"初始服务器利用率: {initial_performance['server_utilization']}")
    print(f"最终服务器利用率: {final_performance['server_utilization']}")
    
    # 打印卸载决策
    print("\n=== 最终卸载决策 ===")
    for user_id in range(num_users):
        print(f"用户 {user_id+1}:")
        for task_id in range(num_tasks):
            decision = final_offloading[user_id][task_id]
            if decision[0] == 1:
                print(f"  任务 {task_id+1}: 本地执行")
            else:
                server_id = np.where(decision[1:] == 1)[0][0]
                print(f"  任务 {task_id+1}: 卸载到服务器 {server_id+1}")
    
    # 可视化结果
    visualizer = Visualizer()
    visualizer.plot_prices(initial_prices, final_prices)
    visualizer.plot_offloading_decisions(final_offloading, num_users, num_tasks, num_servers)
    visualizer.plot_performance(
        (initial_performance['total_cost'], initial_performance['total_profit']), 
        (final_performance['total_cost'], final_performance['total_profit'])
    )

if __name__ == "__main__":
    main()