import tensorflow as tf
from tqdm import tqdm
import io
import getConfig
import numpy as np

def preprocess(filename1,filename2):
    save_list=[]
    with open(filename1) as file_object:
        for line in tqdm(file_object):
            line_in_split=line.split(',')
            #print(line_in_split)
            save_list.append(line_in_split[0].split(':')[2]+',')
            save_list.append(line_in_split[1]+',')
            save_list.append(line_in_split[2]+',')
            save_list.append(line_in_split[5])
            save_list.append('\n')
            #print(save_list)
    with open(filename2,'w') as file_object:
        for string in save_list:
            file_object.write(string)


#filename1='train_data/Eng_conv/train.csv'
#filename2='train_data/Eng_conv/save.csv'
#preprocess(filename1,filename2)

def statistic_len(path,num_examples):
    #统计数据集中 句子的长度分布
    len0_20=0
    len20_40=0
    len40_60=0
    len60_80=0
    len80_100=0
    len100_120=0
    len_others=0
    lines=io.open(path).read().strip().split('\n')
    for l in lines[1:num_examples]:
        l_len=len(l.split(',')[3].split(' '))
        #print("l_len=",l_len)
        if (l_len>0)and(l_len<=20):
            len0_20+=1
        elif (l_len>20)and(l_len<=40):
            len20_40+=1
        elif (l_len>40)and(l_len<=60):
            len40_60+=1
        elif (l_len>60)and(l_len<=80):
            len60_80+=1
        elif (l_len>80)and(l_len<=100):
            len80_100+=1
        elif (l_len>100)and(l_len<=120):
            len100_120+=1
        else:
            len_others+=1
    print("len0_20=",len0_20)
    print("len20_40=",len20_40)
    print("len40_60=",len40_60)
    print("len60_80=",len60_80)
    print("len80_100=",len80_100)
    print("len100_120=",len100_120)
    print("len_others=",len_others)

statistic_len('train_data/Eng_conv/save.csv',84169)

def get_emo_num(path,num_examples):
    list1=[]
    lines=io.open(path).read().strip().split('\n')
    for l in lines[1:num_examples]:
        list1.append(l.split(',')[2])
    list2=list(set(list1))
    print(list2)
    print("emo_num:",len(list2))

#get_emo_num('train_data/Eng_conv/save.csv', 10000)


def preprocess_embeddings():
    save_list=[]
    with open('vectors/glove.6B.100d.txt',encoding='utf-8') as file_object:
        
        num_list=[]
        word_list=[]
        for line in file_object:            
            word_list.append(line.split(' ')[0])
            emb_string_list=line.split(' ')[1:]
            emb_num_list=[]
            for emb_string in emb_string_list:
                emb_num_list.append(float(emb_string))
            num_list.append(emb_num_list)
        #print("num_list:\n",num_list)
        emb_tensor=tf.constant(num_list)
        #print("emb_tensor:\n",emb_tensor)
        Dense_layer=tf.keras.layers.Dense(128)
        emb_dense=Dense_layer(emb_tensor)
        emb_dense=emb_dense.numpy()
        #print(emb_dense)
        length=len(word_list)
        for i in range(0,length):
            emb_dense_string=""
            for j in emb_dense[i]:
                emb_dense_string+=str(j)
                emb_dense_string+=' '
            save_list.append(word_list[i]+' '+str(emb_dense_string)+'\n')
    print("read over!")
    with open('vectors/glove.128d.txt','w',encoding='utf-8') as file_object:
        for string in save_list:
            file_object.write(string)
    
