# -*- coding:utf-8 -*-
import os
import sys
import time
import tensorflow as tf
import empatheticModel
import getConfig
import io
import numpy as np

gConfig = {}

gConfig=getConfig.get_config(config_file='seq2seq.ini')

vocab_inp_size = gConfig['enc_vocab_size']#20000
vocab_tar_size = gConfig['dec_vocab_size']#20000
embedding_dim=gConfig['emp_embedding_dim']#100
units=gConfig['emp_layer_size']#200
BATCH_SIZE=gConfig['batch_size']#448
max_length_inp,max_length_tar=61,61
emotions=['surprised', 'confident', 'content', 'apprehensive', 
          'nostalgic', 'anxious', 'hopeful', 'impressed', 
          'jealous', 'angry', 'faithful', 'embarrassed', 
          'sentimental', 'excited', 'joyful', 'furious', 
          'grateful', 'lonely', 'anticipating', 'trusting', 
          'sad', 'disgusted', 'prepared', 'proud', 
          'guilty', 'terrified', 'annoyed', 'caring', 
          'disappointed', 'ashamed', 'afraid', 'devastated']
positive_emotions=[0,1,2,6,7,10,13,14,16,19,22,23,27]
negative_emotions=[3,4,5,8,9,11,12,15,17,18,20,21,24,25,26,28,29,30,31]

with tf.compat.v1.Session() as sess:
    with tf.device('/gpu:1'):
        a = tf.compat.v1.placeholder(tf.int32)
        b = tf.compat.v1.placeholder(tf.int32)
        add = tf.add(a, b)
        sum = sess.run(add, feed_dict={a: 3, b: 4})
        print(sum)

def preprocess_sentence(w):
    w ='start '+ w + ' end'
    return w

def create_dataset(path, num_examples):
    #lines列表，按行保存文件
    lines = io.open(path).read().strip().split('\n')
    A_text_list=[]
    text_list=[]
    l_temp=lines[0]
    for l in lines[1:num_examples]:
        if l_temp.split(',')[0]==l.split(',')[0]:
            A_text_list.append(l_temp.split(',')[2]+' '+preprocess_sentence(l_temp.split(',')[3]))
            A_text_list.append(l.split(',')[2]+' '+preprocess_sentence(l.split(',')[3]))
            #A_emo_pair.append()
            text_list.append(A_text_list)            
        A_text_list=[]
        l_temp=l        
    #[['input1','target1'],['input2','target2'],...]
    #text_list = [[preprocess_sentence(w)for w in l.split('\t')] for l in lines[:num_examples]]
    #zip()将分散的东西按对应的位置组合在一起，zip(*）则是相反的操作。
    #将问答对 处理成 input_lang target_lang
    return zip(*text_list)

def max_length(tensor):
    return max(len(t) for t in tensor)

def read_data(path,num_examples):
    #['input1','input2','input3',...]  ['target1','target2','target3',...]
    input_lang,target_lang=create_dataset(path,num_examples)

    input_tensor,input_token=tokenize(input_lang)
    target_tensor,target_token=tokenize(target_lang)
    #tensor是词向量，token是Tokenizer类对象
    return input_tensor,input_token,target_tensor,target_token

def tokenize(lang):#lang是文本元组，('string1','string2',...)
    #Tokenizer是一个类。用于文本语料的向量化。
    #初始化一个Tokenizer
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=gConfig['enc_vocab_size'],#保留的最大词数，基于词频，保留前num_words-1个 
                                                           oov_token=3)#它将被添加到word_index中，并用于在text_to_sequence调用期间替换词汇外的单词
    #根据文本列表，更新内部词汇表，在使用texts_to_sequences之前必须调用
    #lang_tokenizer中的self.index_word被改变
    lang_tokenizer.fit_on_texts(lang)
    #将texts中的每个text转化为int型序列，sequence=[[1],[2,3],[4,5,6],...]
    tensor = lang_tokenizer.texts_to_sequences(lang)
    #用PAD补全，maxlen是将每一个sequence补全到多长，padding=post是从sequence后面补全（默认是从前面补全）
    #max_length_inp=20
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, maxlen=max_length_inp,padding='post')
    #返回一个tensor，外加一个Tokenizer类对象
    return tensor, lang_tokenizer

input_tensor,input_token,target_tensor,target_token= read_data(gConfig['seq_data'], gConfig['max_train_data_size'])

def train_MIME():
    print("Preparing data in %s" % gConfig['train_data'])
    #每个step完成一batch（128），每个epoch完成steps_per_epoch个step
    steps_per_epoch = len(input_tensor) // gConfig['batch_size']
    print("steps_per_epoch=",steps_per_epoch)
    enc_hidden = empatheticModel.context_encoder.initialize_hidden_state()
    checkpoint_dir = gConfig['model_data']
    #list_dir:返回目录中包含的条目列表
    ckpt=tf.io.gfile.listdir(checkpoint_dir)
    if ckpt:
        print("reload pretrained model")
        #载入之前保存的参数
        empatheticModel.checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    BUFFER_SIZE = len(input_tensor)
    #from_tensor_slices()将dataset转化为一问一答形式
    dataset = tf.data.Dataset.from_tensor_slices((input_tensor,target_tensor)).shuffle(BUFFER_SIZE)
    #每BATCH_SIZE个为一批
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    checkpoint_dir = gConfig['model_data']
    #将目录和文件名合成一个路径
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    start_time = time.time()
    current_epoch=0
    #每个epoch就是把全部数据集轮一次
    while True:
        current_epoch+=1
        print("current_epoch=",current_epoch)
        #time.time()返回当前的时间戳
        start_time_epoch = time.time()
        total_loss = 0
        #每个dataset是一个input_tensor和一个target_tensor
        #dataset已经分成了batchs，取504个batch，即全部轮一次作为一个epoch
        for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
            #循环504次
            batch_loss = empatheticModel.train_step(inp, targ, input_token, target_token, enc_hidden)
            total_loss += batch_loss
            print(batch_loss.numpy())

        #完成上面for循环的总时间 除以 每个周期的步数 等于 该周期内每一步消耗的平均时长
        step_time_epoch = (time.time() - start_time_epoch) / steps_per_epoch
        step_loss = total_loss / steps_per_epoch
        current_steps = +steps_per_epoch
        step_time_total = (time.time() - start_time) / current_steps

        print('训练总步数: {} 每步耗时: {}  最新每步耗时: {} 最新每步loss {:.4f}'.format(current_steps, step_time_total, step_time_epoch,
                                                                      step_loss.numpy()))
        #保存模型
        empatheticModel.checkpoint.save(file_prefix=checkpoint_prefix)

        sys.stdout.flush()
def predict(sentence):#sentence是input
    checkpoint_dir = gConfig['model_data']
    #加载模型
    empatheticModel.checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    emotion=sentence.split(' ')[0]
    #预处理，加start和end
    temp=''
    for word in sentence.split(' ')[1:]:
        temp+=word
        temp+=' '
    sentence = emotion+' '+preprocess_sentence(temp)
    print("sentence:",sentence)
    #word_index是个字典，{'word1':1,'word20:2',...}
    #找到sentence中每一个word在word_index中对应的序号，构成词向量
    inputs = [input_token.word_index.get(i,3) for i in sentence.split(' ')]
    #PAD补齐,max_length_inp=60
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],maxlen=max_length_inp,padding='post')
    inputs = tf.convert_to_tensor(inputs)
    result = ''
    hidden = [tf.zeros((1, units))]
    enc_out, enc_hidden, emo_embeddings = empatheticModel.context_encoder(inputs, input_token, hidden)
    
    enc_output, enc_hidden = empatheticModel.emotion_encoder(emo_embeddings, enc_out, enc_hidden)
    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([target_token.word_index['start']], 0)

    for t in range(max_length_tar):
        predictions, dec_hidden, attention_weights = empatheticModel.context_decoder(dec_input, target_token, dec_hidden, enc_output)
        predicted_id = tf.argmax(predictions[0]).numpy()
        if target_token.index_word[predicted_id] == 'end':
            break
        elif target_token.index_word[predicted_id] == 'start':
            continue
        if target_token.index_word[predicted_id] == 'comma':
            result += ','
        else:
            result += target_token.index_word[predicted_id] + ' '      
        dec_input = tf.expand_dims([predicted_id], 0)

    return result

if __name__ == '__main__':
    #statistic_len('train_data/Eng_conv/save.csv', gConfig['max_train_data_size'])
    if len(sys.argv) - 1:
        gConfig = getConfig.get_config(sys.argv[1])
    else:
        gConfig = getConfig.get_config()
    print('\n>> Mode : %s\n' %(gConfig['mode']))
    if gConfig['mode'] == 'train':  
        train_MIME()
    elif gConfig['mode'] == 'serve':   
        print('Serve Usage : >> python3 app.py')


