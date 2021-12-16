from matplotlib import pyplot as plt
plt.rcParams['font.sans-serif']=['kaiti'] #用来正常显示中文标签
plt.plot([1,2,3], 'o-', color='pink', label='搭风格')
plt.legend()
plt.show()