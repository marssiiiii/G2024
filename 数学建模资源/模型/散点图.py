import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体
sns.set(font='SimHei')  # 使用你系统中支持中文的字体

# 读取Excel文件
file_path = r'C:\Users\USER\OneDrive\桌面\4-5-散点图.xlsm'
df = pd.read_excel(file_path)

# 创建图表
plt.figure(figsize=(10, 6))
ax = plt.gca()

# 绘制散点图
scatter = sns.scatterplot(x='评论数量/销量', y='满意度', data=df, hue='细分类目', s=100)

# 添加标题和标签，调整位置和字号
plt.title('消费者满意度和销量关系分析', loc='center', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('销售额', fontsize=14, fontweight='bold', labelpad=10)
plt.ylabel('满意度', fontsize=14, fontweight='bold', labelpad=10)

# 设置横轴范围从0开始
ax.set_xlim(0)

# 调整图例位置
legend = plt.legend(title='服装类目', bbox_to_anchor=(0.97, 1), loc='upper left')

# 设置坐标轴颜色为黑色
ax.spines['bottom'].set_color('black')
ax.spines['left'].set_color('black')
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')

# 设置网格线颜色为灰色
ax.xaxis.grid(True, color='gray', linestyle='dashed', linewidth=0.5)
ax.yaxis.grid(True, color='gray', linestyle='dashed', linewidth=0.5)

# 设置图表背景色为白色
ax.set_facecolor('white')

# 调整整体图像向左移动
plt.subplots_adjust(left=0.1)

# 减小坐标轴线段长度
ax.spines['bottom'].set_bounds(0, 4780)  # 设置x轴的范围
ax.spines['left'].set_bounds(3.825, 4.38)  # 设置y轴的范围

# 在横坐标轴末端添加箭头
ax.annotate('', xy=(1, 0), xycoords='axes fraction', xytext=(0.99, 0),
            arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))

# 在纵坐标轴末端添加箭头
ax.annotate('', xy=(0, 1), xycoords='axes fraction', xytext=(0, 0.99),
            arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))

# 显示图形
plt.show()
