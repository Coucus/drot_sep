import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from scipy.optimize import fsolve
#构造目标函数和初始条件
#多项式拟合函数=ax+bx^2+cx^3拟合sinx
xtest_sample=np.dot(np.pi,[-1,-1/2,-1/3,0,1/3,1/2,1])
ytest_sample=np.sin(xtest_sample)
#test_matrix=np.matrix([[2,2],[2,34]])
#test_matrix2=[[0,-48]]
def function_matrix(x):
    return [x,np.power(x,2),np.power(x,3)]
a=sp.symbols('a')
b=sp.symbols('b')
c=sp.symbols('c')
start=np.matrix([[a,b,c]])
costfunction=0
for i in range(len(xtest_sample)):
 costfunction += np.power(function_matrix(xtest_sample[i])*start.T-ytest_sample[i],2)
#start=np.matrix([[-10,-10]])

expression = np.sum(costfunction)
expanded_expression = sp.expand(expression)


# 提取系数
coeff_a2 = float(expanded_expression.coeff(a**2))
coeff_b2 = float(expanded_expression.coeff(b**2))
coeff_c2 = float(expanded_expression.coeff(c**2))
coeff_ab = float(expanded_expression.coeff(a*b))
coeff_bc = float(expanded_expression.coeff(b*c))
coeff_ac = float(expanded_expression.coeff(a*c))
coeff_a = float(expanded_expression.coeff(a)-coeff_ab*b-coeff_ac*c)
coeff_b = float(expanded_expression.coeff(b)-coeff_ab*a-coeff_bc*c)
coeff_c = float(expanded_expression.coeff(c)-coeff_ac*a-coeff_ab*b)
print(expanded_expression)
print("系数 a^2:", coeff_a2)
print("系数 b^2:", coeff_b2)
print("系数 c^2:", coeff_c2)
print("系数 ab:", coeff_ab)
print("系数 ac:", coeff_bc)
print("系数 bc:", coeff_ac)
print("系数 a:", coeff_a)
print("系数 b:", coeff_b)
print("系数 c:", coeff_c)


test_matrix=np.matrix([[2*coeff_a2,coeff_ab,coeff_ac],[coeff_ab,2*coeff_b2,coeff_bc],[coeff_ac,coeff_bc,2*coeff_c2]])
test_matrix2=np.matrix([[coeff_a,coeff_b,coeff_c]])
first_step=np.matrix([[2,0.3,1]])
cost_function=first_step*test_matrix*first_step.T+test_matrix2*first_step.T+3.5
#采用FR共轭梯度算法，目标函数x1^2+x2^2+x1x2一定有最小值，即阀值为梯度为0
g=first_step*test_matrix+test_matrix2
d=-g#确定共轭方向
alpha=g*g.T/(d*test_matrix*d.T)#初始步长
step=[1,5,100]
cost_matrix=[]#记录数值
alpha_matrix=[]
x=np.linspace(-np.pi,np.pi,50)
fig = plt.figure(figsize=(20, 12))
for j in range(len(step)):
#while cost_function>11:
  alpha_matrix.clear()
  for k in range(step[j]):
   cost_matrix.append(cost_function)
   first_step_terminal=first_step
   first_step=first_step_terminal+alpha*d
   g_terminal=g
   g=first_step*test_matrix+test_matrix2
   beta=g*g.T/(g_terminal*g_terminal.T)#共轭系数
   d_terminal=d
   d =-g+beta*d_terminal#确定共轭方向
   cost_function = first_step*test_matrix*first_step.T+3.5
   alpha = g * g.T / (d * test_matrix * d.T)
   alpha_matrix.append(alpha[0,0])
  plt.subplot(2, 2, j+1)
  plt.plot(x, first_step[0,0]*x + first_step[0,1]*np.power(x,2) + first_step[0,2]*np.power(x,3), label='FR-result \n steps={}'.format(step[j]))

  plt.plot(x,np.sin(x),c='r',label='sinx')
  plt.legend(fontsize=15)
  plt.xlabel('x')
  plt.ylabel('fitting curve')
  plt.grid()
plt.subplot(2,2,4)
plt.plot(range(len(alpha_matrix)),alpha_matrix,c='b',label='step value')
plt.legend(fontsize=15)
plt.xlabel('steps')
plt.grid()
#ax1.plot(x,first_step[0,0]*x+first_step[0,1]*np.power(x,2)+first_step[0,2]*np.power(x,3),label='FR-result')
#ax1.plot(x,np.sin(x),c='r',label='sinx')
#print(alpha_matrix)
plt.show()
