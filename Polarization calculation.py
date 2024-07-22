import numpy as np

# 用户输入斯托克斯参数
s0 = float(input("请输入s0: "))
s1 = float(input("请输入s1: "))
s2 = float(input("请输入s2: "))
s3 = float(input("请输入s3: "))

# 计算偏振度P
P = np.sqrt(s1**2 + s2**2 + s3**2) / s0

# 计算线偏振度DoLP
DoLP = np.sqrt(s1**2 + s2**2) / s0

# 计算圆偏振度DoCP
DoCP = np.abs(s3) / s0

# 计算椭圆率
# 椭圆角β，它与s3关系为tan(2β) = s3 / sqrt(s1^2 + s2^2)，椭圆率ε = tan(β)
# 当需要输出椭圆率本身时，求数学表达中β的一半即为椭圆率。
if s1 == 0 and s2 == 0:  # 避免除以零
    ellipse_ratio = 0
else:
    ellipse_ratio1=np.sqrt(s1**2 + s2**2)+np.sqrt(s1**2 + s2**2 + s3**2)
    ellipse_ratio = np.arctan(np.abs(s3) / ellipse_ratio1)
    ellipse_ratio = np.tan(ellipse_ratio)
    # 计算方位角

    azimuth = np.arctan2(s2, s1) / 2

# 输出结果
print(f"偏振度P: {P}")
print(f"线偏振度DoLP: {DoLP}")
print(f"圆偏振度DoCP: {DoCP}")
print(f"椭圆率: {ellipse_ratio}")
print(f"方位角（弧度）: {azimuth}")

# 如果您希望以度为单位显示方位角
azimuth_degrees = np.degrees(azimuth)
print(f"方位角（度）: {azimuth_degrees}")