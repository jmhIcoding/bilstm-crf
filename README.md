# bilstm-crf
双向lstm+crf 序列标注,国科大自然语言处理课程作业
完成地名、人名、组织名的标注
# 思路
1. 使用source_data.txt里面的语料训练词向量
2. 设计一个双向的lstm神经网络训练得到各节点特征函数
3. lstm的末尾挂上一层crf网络,用以训练得到边特征函数
4. 定义损失函数为log
5. 划分，验证集
6. 训练
7. 优化调参
8. 测试，计算F1
# 数据集的描述
(1) 标签集
{  
B-PER	：人名开始 ； I-PER：人名中间 ；
B-LOC：	地名开始， 
I-LOC：地名中间； 
B-ORG：机构名开始 ；I-ORG：机构名中间；
O ：其他 }
(2) 训练语料
目录data 中 Source_data.txt 文件为训练文本 
目录data 中 Source_label.txt  文件为训练文本对应的每个字的标签 
如周恩来对应B-PER, I-PER, I-PER
(3) 测试语料
目录data 中 Test_data.txt 文件为测试文本 ，Test_label.txt 文件为对应的标签 （格式同训练语料）
(4)
bilstm-crf.py :网络代码
utils.py	  :数据集处理函数
