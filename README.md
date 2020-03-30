# 2020华为软件精英挑战赛热身赛-方案记录
最后版本是 [Main8_2.cpp](https://github.com/drawepen/HWSoftwareChallenge2020/blob/master/Main8-2.cpp)---0.0306，准确率81%，排名32。
思路：
 * -test文件应该没有符号，格式统一，可以定位跳转；
 * -train文件有未归一化数据，存在-号和>1，测试集没有，去除含-号的数据//不理想，跳过还会影响速度，最后不跳过
 * -多进程-预测集16个，训练集6个，训练集读的比较少，进程太多可能弥补不了启动进程的时间，最佳个数应该不是这个，但到比赛尾声了，没有提交次数用于测试了，只能猜测；
 * -线程创建的进程执行更多行
 * -只使用小数点后一位数字//效果更好，这并非偶然，当算法很模糊的时候，需要样本特征更加明显才能比较好点的分类，特征最好分级，0-9显然比0.001到0.999样本差别更大
 * -训练集01分类求和取平均，求欧拉距离，这个算法也是这个比赛的常识了，O(n)的时间复杂度，目前还不知道更好的算法，（单个特征取闸值的话，特征选取、闸值训练避免不了来回处理数据，很难比这个算法更好，如果直接给定就硬编码了），但要问为什么这么模糊方法准确率能轻松达到80%,抬头望青天；
 * -特征分级为1,2,4,8，以便只用移位运算；///失败
 * -特征权值紧挨,连续特征权值,减少换页；//效果不大
 * -以7\*recNum估计一行大小//改为以recNum\*49/8估计大小，实验表明训练数据越连续预测效果越好，原因不详，大概是训练集没有打乱
 * -使用arm_neon
 * -设置cpu亲和性///失败
 * -赛后增加-来自大佬分享-特征值可以直接使用char的ASCII，不用减'0'!!! |( T﹏T )这么明显竟没想到。
