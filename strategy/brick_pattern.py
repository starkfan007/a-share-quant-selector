"""
超短线砖型图策略 - 通达信公式 Python 实现

指标定义：
1. 白线 (WHITELINE) = EMA(EMA(CLOSE, 10), 10)
   - 短期趋势线，反映股价短期波动方向

2. 黄线 (YELLOWLINE) = (MA(CLOSE, M1) + MA(CLOSE, M2) + MA(CLOSE, M3) + MA(CLOSE, M4)) / 4
   - 多空线，反映股价中长期强弱界限

3. 砖型图指标：
   - VAR1A = (HHV(HIGH, 4) - CLOSE) / (HHV(HIGH, 4) - LLV(LOW, 4)) * 100 - 90
   - VAR2A = SMA(VAR1A, 4, 1) + 100
   - VAR3A = (CLOSE - LLV(LOW, 4)) / (HHV(HIGH, 4) - LLV(LOW, 4)) * 100
   - VAR4A = SMA(VAR3A, 6, 1)
   - VAR5A = SMA(VAR4A, 6, 1) + 100
   - VAR6A = VAR5A - VAR2A
   - 砖型图 = IF(VAR6A > 4, VAR6A - 4, 0)

选股条件：
4. 砖型转折 (CC > 0)：
   - AA = (REF(砖型图, 1) < 砖型图)
   - CC = REF(AA, 1) == 0 AND AA == 1 (即砖型图由平/降转为升的第一天)

5. 趋势占优：白线 > 黄线 (WHITELINE > YELLOWLINE)

6. 股价位置：股价在黄线上 (CLOSE > YELLOWLINE)

7. 动量加速：今天砖体 / 前一天砖体 > 0.66
   - 砖体 = ABS(砖型图 - REF(砖型图, 1))

8. 超买约束：J < 64
   - 避免在 KDJ 指标极度超买区域介入

9. 最终选股信号 = 砖型转折 AND 趋势占优 AND 股价在黄线上 AND 动量加速 AND 超买约束
"""
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from strategy.base_strategy import BaseStrategy
from utils.technical import (
    SMA, LLV, HHV, REF, KDJ, calculate_zhixing_trend
)


class BrickPatternStrategy(BaseStrategy):
    """超短线砖型图策略 - 基于高低收平滑的转折点选股"""
    
    def __init__(self, params=None):
        """
        初始化策略
        :param params: 参数字典，包含周期设置、J值上限、动量比率等
        """
        # 默认参数
        default_params = {
            'N1': 4,              # 砖型图计算周期
            'N2': 6,              # 平滑周期
            'M1': 14,             # 多空线周期1
            'M2': 28,             # 多空线周期2
            'M3': 57,             # 多空线周期3
            'M4': 114,            # 多空线周期4
            'J_LIMIT': 64,        # J值上限约束，避免极度超买
            'BRICK_RATIO': 0.66   # 砖体增长比率（动量比率）
        }
        
        # 合并用户参数
        if params:
            default_params.update(params)
        
        super().__init__("超短线砖型图策略", default_params)
    
    def calculate_indicators(self, df) -> pd.DataFrame:
        """
        计算砖型图策略所需的所有指标
        :param df: 原始股票数据 (DataFrame)
        :return: 包含所有技术指标的 DataFrame
        """
        result = df.copy()
        n1 = self.params.get('N1', 4)
        n2 = self.params.get('N2', 6)
        
        # 1. 计算白线和黄线 (WHITELINE, YELLOWLINE)
        # 使用知行多空线通用计算模块
        trend_df = calculate_zhixing_trend(
            result, 
            m1=self.params['M1'],
            m2=self.params['M2'],
            m3=self.params['M3'],
            m4=self.params['M4']
        )
        result['whiteline'] = trend_df['short_term_trend']
        result['yellowline'] = trend_df['bull_bear_line']
        
        # 2. 计算 KDJ 指标的 J 值
        # 用于过滤极度超买状态
        kdj_df = KDJ(result, n=9, m1=3, m2=3)
        result['j_val'] = kdj_df['J']
        
        # 3. 计算砖型图指标
        # 核心算法：通过对高低收价格的多级平滑计算转折能量
        
        # 为了计算 SMA，我们需要将数据转为正序（时间从旧到新）
        df_sorted = result.sort_index(ascending=False)
        
        high = df_sorted['high']
        low = df_sorted['low']
        close = df_sorted['close']
        
        # 计算 4 周期最高/最低价
        hhv4 = high.rolling(window=n1).max()
        llv4 = low.rolling(window=n1).min()
        
        # 核心变量计算 (VAR1A - VAR6A)
        denom1 = hhv4 - llv4
        var1a = (hhv4 - close) / denom1.replace(0, 0.001) * 100 - 90
        
        # 使用通达信风格 SMA (4,1)
        var2a = SMA(var1a.fillna(0), n1, 1) + 100
        
        # 价格位置变量
        var3a = (close - llv4) / denom1.replace(0, 0.001) * 100
        
        # 多级平滑
        var4a = SMA(var3a.fillna(0), n2, 1)
        var5a = SMA(var4a.fillna(0), n2, 1) + 100
        
        # 差值计算
        var6a = var5a - var2a
        
        # 生成砖型图指标
        brick_chart = var6a.apply(lambda x: x - 4 if x > 4 else 0)
        
        # 将结果合并回原 DataFrame（恢复原始倒序索引）
        df_sorted['brick_chart'] = brick_chart
        result['brick_chart'] = df_sorted['brick_chart']
        
        return result
    
    def select_stocks(self, df, stock_name='') -> list:
        """
        执行选股判断逻辑
        :param df: 包含所有指标的 DataFrame
        :param stock_name: 股票名称
        :return: 符合条件的选股信号列表
        """
        if df is None or len(df) < 5:
            return []
            
        try:
            # 获取最近三天的数据 (iloc[0]=最新, iloc[1]=昨天, iloc[2]=前天)
            
            # 1. 砖型图转折条件 (CC > 0)
            # 逻辑：昨天砖型图未升，今天砖型图升，构成转折向上
            brick_0 = df['brick_chart'].iloc[0] # 今天
            brick_1 = df['brick_chart'].iloc[1] # 昨天
            brick_2 = df['brick_chart'].iloc[2] # 前天
            
            aa_0 = brick_1 < brick_0 # 今天砖型图上升
            aa_1 = brick_2 < brick_1 # 昨天砖型图上升
            cc_0 = (not aa_1) and aa_0 # 转折向上信号
            
            if not cc_0:
                return []
            
            # 2. 趋势条件 (WHITELINE > YELLOWLINE)
            # 逻辑：白线在黄线上方，处于短期强势通道
            white_0 = df['whiteline'].iloc[0]
            yellow_0 = df['yellowline'].iloc[0]
            trend_ok = white_0 > yellow_0
            
            # 3. 股价位置条件 (股价在黄线上: CLOSE > YELLOWLINE)
            # 逻辑：价格回踩多空线未跌破，或在多空线上方震荡
            close_0 = df['close'].iloc[0]
            price_above = close_0 > yellow_0
            
            # 4. 动量条件 (大于三分之二: 今天砖体/前一天砖体 > 0.66)
            # 逻辑：今天指标上升的动量必须达到昨天的 2/3 以上，保证反转力度
            prev_brick_body = abs(brick_1 - brick_2)
            curr_brick_body = abs(brick_0 - brick_1)
            
            # 防止除零错误处理
            if prev_brick_body == 0:
                ratio_ok = curr_brick_body > 0
            else:
                ratio_ok = (curr_brick_body / prev_brick_body) > self.params['BRICK_RATIO']
            
            # 5. KDJ 约束 (J < 64)
            # 逻辑：防止追高极度超买的股票
            j_val = df['j_val'].iloc[0]
            j_ok = j_val < self.params['J_LIMIT']
            
            # 最终汇总判断
            if trend_ok and price_above and ratio_ok and j_ok:
                reasons = []
                reasons.append(f"砖型图趋势转折(值:{round(brick_0, 2)})")
                reasons.append(f"白线>黄线")
                reasons.append(f"股价在黄线上")
                reasons.append(f"动量比:{round(curr_brick_body/prev_brick_body if prev_brick_body!=0 else 0, 2)}>0.66")
                reasons.append(f"J值:{round(j_val, 1)}<64")
                
                return [{
                    'category': 'brick_pattern_signal',
                    'close': round(close_0, 2),
                    'brick_val': round(brick_0, 2),
                    'j_val': round(j_val, 1),
                    'reasons': " + ".join(reasons)
                }]
                
        except Exception as e:
            # 异常记录，防止单只股票处理失败导致整个扫描中断
            # print(f"分析 {stock_name} 时发生错误: {e}")
            pass
            
        return []
