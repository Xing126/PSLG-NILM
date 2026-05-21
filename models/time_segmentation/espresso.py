import numpy as np
try:
    from pyeda.inter import *
    HAS_PYEDA = True
except ImportError:
    HAS_PYEDA = False
    print("Warning: pyeda library not found. Please install it using 'pip install pyeda'.")

class EspressoMinimizer:
    """
    ESPRESSO 算法实现类，用于逻辑化简。
    支持布尔表达式化简和带“无关项（Don't Care）”的真值表化简。
    """
    def __init__(self):
        if not HAS_PYEDA:
            raise ImportError("pyeda is required for EspressoMinimizer. Install it with 'pip install pyeda'.")

    def minimize_expression(self, expr_str):
        """
        直接对布尔表达式进行化简。
        
        参数:
            expr_str (str): 布尔表达式字符串，例如 "a & b | a & ~b"
            
        返回:
            expr: 化简后的 pyeda 表达式对象
        """
        try:
            f = expr(expr_str)
            if f.is_zero() or f.is_one():
                return f
            # espresso_exprs 返回一个元组，取第一个
            f_min = espresso_exprs(f.to_dnf())[0]
            return f_min
        except Exception as e:
            print(f"Error minimizing expression: {e}")
            return None

    def minimize_truth_table(self, num_vars, entries):
        """
        对真值表进行化简，支持“无关项”。
        
        参数:
            num_vars (int): 输入变量的数量
            entries (list): 真值表条目列表。
                           每个条目是一个元组 (inputs, outputs)
                           inputs: 长度为 num_vars 的字符串，如 '01-' ('-' 表示 don't care)
                           outputs: 长度为输出数量的字符串，如 '10'
        
        返回:
            list: 化简后的逻辑表示（通常为 DNF 形式的表达式列表）
        """
        try:
            # 创建输入变量
            X = exprvars('x', num_vars)
            
            # 构造真值表
            # pyeda 的 truthtable 构造较为复杂，通常建议直接使用 espresso_tts
            # espresso_tts 接受 (num_inputs, num_outputs, table_data)
            
            # 转换 entries 为 espresso_tts 格式
            # table_data 是一个 list of tuples: (input_cube, output_cube)
            # input_cube: tuple of (0, 1, 2) where 2 is don't care
            # output_cube: tuple of (0, 1, 2)
            
            formatted_data = []
            num_outputs = len(entries[0][1]) if entries else 1
            
            for in_str, out_str in entries:
                in_cube = tuple(1 if c == '1' else 0 if c == '0' else 2 for c in in_str)
                out_cube = tuple(1 if c == '1' else 0 if c == '0' else 2 for c in out_str)
                formatted_data.append((in_cube, out_cube))
            
            # 调用 pyeda 的 espresso 接口
            # 注意：espresso_tts 是直接针对真值表数据的
            results = espresso_tts(num_vars, num_outputs, formatted_data)
            return results
        except Exception as e:
            print(f"Error minimizing truth table: {e}")
            return None

def espresso_minimize(data, mode='expr', **kwargs):
    """
    便捷函数，支持两种模式的化简。
    """
    minimizer = EspressoMinimizer()
    if mode == 'expr':
        return minimizer.minimize_expression(data)
    elif mode == 'tt':
        num_vars = kwargs.get('num_vars')
        return minimizer.minimize_truth_table(num_vars, data)
    else:
        raise ValueError("Invalid mode. Use 'expr' or 'tt'.")

if __name__ == "__main__":
    if not HAS_PYEDA:
        print("Skipping examples because pyeda is not installed.")
    else:
        # 示例 1: 布尔表达式化简
        print("--- 示例 1: 布尔表达式化简 ---")
    expr_to_min = "a & b | a & ~b"
    print(f"原始表达式: {expr_to_min}")
    min_expr = espresso_minimize(expr_to_min, mode='expr')
    print(f"化简后: {min_expr}")

    # 示例 2: 真值表化简 (包含 Don't Care)
    print("\n--- 示例 2: 真值表化简 (包含 Don't Care) ---")
    # 3变量输入，1变量输出
    # 假设输入为 '000' -> '0', '001' -> '1', '010' -> '1', '1--' -> '0' (don't care)
    # 实际上 ESPRESSO 通常用于处理 ON-set, OFF-set 和 DC-set
    # 这里我们演示一个简单的格式
    tt_data = [
        ('000', '0'),
        ('001', '1'),
        ('010', '1'),
        ('011', '0'),
        ('100', '1'),
        ('101', '1'),
        ('110', '0'),
        ('111', '0')
    ]
    # 如果想体现 Don't Care，可以在 entries 中使用 '-' 或在逻辑中定义
    # pyeda 的 espresso_tts 期望完整的真值表描述或特定的 ON/DC 集合
    
    # 重新定义一个简单的例子
    # f(a, b) = a & b, 但我们不知道 (1, 0) 的情况 (Don't Care)
    tt_data_dc = [
        ('00', '0'),
        ('01', '0'),
        ('11', '1'),
        ('10', '-') # Don't Care
    ]
    print(f"真值表数据: {tt_data_dc}")
    # results = espresso_minimize(tt_data_dc, mode='tt', num_vars=2)
    # print(f"化简结果: {results}")
