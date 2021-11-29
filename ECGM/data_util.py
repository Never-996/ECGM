
# coding=utf-8

import os
import getConfig

import re
#结巴是国内的一个分词python库，分词效果非常不错。pip3 install jieba安装

gConfig = {}

gConfig=getConfig.get_config()

#gConfig['resource_data']='train_data/xiaohuangji50w_nofenci.conv'
conv_path = gConfig['resource_data']
 
if not os.path.exists(conv_path):	
	exit()
#将训练集的数据识别读取并存入一个List中，分以下几步
#a、打开文件 
#b、读取文件中的内容，并对文件的数据进行初步处理
#c、找出我们想要的数据存储下来
convs = []  # 用于存储对话的列表，[['question','answer'],['question','answer']...]
with open(conv_path,encoding='utf-8') as f:
	one_conv = []        # 存储一次完整对话
	for line in f:
		line = line.strip('\n')#去除换行符，并将原文件中已经分词的标记去掉，重新用结巴分词.
		#用zhong.hanzi中的punctuation将所有标点符号清除
		line=re.sub(r"[%s]+" %punctuation, "",line)
		if line == '':
			continue
		#如果是E(对话分隔符)
		if line[0] == gConfig['e']:
			if one_conv:
				convs.append(one_conv)
			one_conv = []
		#如果是M(对话)
		elif line[0] == gConfig['m']:
			one_conv.append(line.split(' ')[1])#将一次完整的对话存储下来
#print(convs)
#对训练集的对话进行分类，分为问和答(或者叫上文、下文)，作为encoder和decoder的熟练数据
#分为以下步骤
#1、初始化变量，ask response为List
#2、按照语句的顺序来分为问句和答句，根据行数的奇偶性来判断
#3、在存储语句的时候对语句使用结巴分词，jieba.cut

# 把对话分成问与答两个部分
seq = []        

for conv in convs:
	# 默认一问一答，需要对数进行粗略裁剪，对话行数要变成偶数
	if len(conv) == 1:#只有一条，直接舍弃
		continue
	if len(conv) % 2 != 0:#有单数且多条，则舍弃最后一条
		conv = conv[:-1]
	for i in range(len(conv)):
		if i % 2 == 0:
			conv[i]=" ".join(jieba.cut(conv[i]))#使用jieba分词器进行分词
			conv[i+1]=" ".join(jieba.cut(conv[i+1]))
			seq.append(conv[i]+'\t'+conv[i+1])#i从0开始，偶数行为问句，奇数行为答句

#seq_data = train_data/seq.data
seq_train = open(gConfig['seq_data'],'w',encoding='utf-8') 

for i in range(len(seq)):
   seq_train.write(seq[i]+'\n') 
   if i % 1000 == 0:
      print(len(range(len(seq))), '处理进度：', i)
 
seq_train.close()



