import tensorflow as tf
import numpy as np
import getConfig
import io

"""
Model由layers构成，
layers在Model的__init__中被初始化，
并在call中以layer(x)的形式调用
"""
gConfig = {}
gConfig=getConfig.get_config(config_file='seq2seq.ini')
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
embeddings_in_dict={}
emo_embeddings_in_dict={}
def get_embeddings_in_dict(path,num_examples):    
    lines = io.open(path).read().strip().split('\n')    
    text_list=[]
    for l in lines[:num_examples]:
        text_list.append('start '+l.split(',')[3]+' end')                       
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=gConfig['enc_vocab_size'],#保留的最大词数，基于词频，保留前num_words-1个 
                                                           oov_token=3)#它将被添加到word_index中，并用于在text_to_sequence调用期间替换词汇外的单词
    #根据文本列表，更新内部词汇表，在使用texts_to_sequences之前必须调用
    #lang_tokenizer中的self.index_word被改变
    lang_tokenizer.fit_on_texts(text_list)
    word_num=lang_tokenizer.num_words
    
    embeddings=np.zeros((word_num,100))
    for line in open(gConfig['emb_file'],encoding='utf-8').readlines():
      sp=line.split()
      if(len(sp)==101):
        if sp[0] in lang_tokenizer.word_index:
          embeddings[lang_tokenizer.word_index[sp[0]]]=[float(x) for x in sp[1:]]
      else:
        print(sp[0])
    for index,word in lang_tokenizer.index_word.items():
        embeddings_in_dict[word]=embeddings[lang_tokenizer.word_index[word]].tolist()          
    return

def get_emo_embeddings_in_dict():
  for line in open(gConfig['emo_emb_file'],encoding='utf-8').readlines():
    sp=line.split()
    if(len(sp)==101):
      emo_embeddings_in_dict[sp[0]]=[float(x) for x in sp[1:]] 
    else:
      print(sp[0])
  return

get_embeddings_in_dict(gConfig['seq_data'],gConfig['max_train_data_size'])
get_emo_embeddings_in_dict()


def get_embeddings(input_tensor,vocab):
  #vocab是token
  #embeddings=np.random.randn(vocab.num_words,300)*0.01
  #embeddings=np.zeros((vocab.num_words,100))
  if gConfig['emb_file'] is not None:
    
    inp_embeddings=[]
    emo_embeddings=[]
    for inp in input_tensor:
      inp_num=inp.numpy()
      
      emotion=vocab.index_word[inp_num[0]]
      if emotion in emotions:
        emo_embeddings.append(emo_embeddings_in_dict[emotion])
      else: 
        emo_embeddings.append(emo_embeddings_in_dict['caring'])
      temp=[]
      #print(inp_num)
      for i in inp_num[1:]:
        if i == 0:
          temp.append([0.00]*100)
        else:
          temp.append(embeddings_in_dict[vocab.index_word[i]])
      inp_embeddings.append(temp)
  inp_embeddings=tf.convert_to_tensor(inp_embeddings)
  emo_embeddings=tf.convert_to_tensor(emo_embeddings)
  return inp_embeddings,emo_embeddings  

def get_embeddings_for_targ(input_tensor,vocab):
  #vocab是token
  #embeddings=np.random.randn(vocab.num_words,300)*0.01
  #embeddings=np.zeros((vocab.num_words,100))
  if gConfig['emb_file'] is not None:
    
    inp_embeddings=[]
    for inp in input_tensor:      
      temp=[]
      for i in inp.numpy():
        if i == 0:
          temp.append([0.00]*100)
        else:
          temp.append(embeddings_in_dict[vocab.index_word[i]])
      inp_embeddings.append(temp)
  inp_embeddings=tf.convert_to_tensor(inp_embeddings) 
  return inp_embeddings 

def extend_emo_embeddings(emo_embeddings):
  emo_embeddings=np.array(emo_embeddings)
  extend_emo_embeddings=[]
  for emo_embedding in emo_embeddings:
    temp=[]
    for i in range(60):
      temp.append(emo_embedding)
    extend_emo_embeddings.append(temp)
  extend_emo_embeddings=tf.convert_to_tensor(extend_emo_embeddings)
  return extend_emo_embeddings
  

class ContextEncoder(tf.keras.Model):
  #两层，第一层Embedding，第二层GRU
  def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
    super(ContextEncoder, self).__init__()
    self.batch_sz = batch_sz
    self.enc_units = enc_units
    #将正整数（索引）转换为固定大小的密集向量，只能用作模型的第一层
    #vocab_size=20000,embedding_dim=100
    #self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(self.enc_units,
                                   return_sequences=True,#return the last output in the output sequence, or the full sequence.
                                   return_state=True,#return the last state in addition to the output.
                                   recurrent_initializer='glorot_uniform')
    
  #调用
  def call(self, inp, token, hidden):
    #x = self.embedding(x)
    x,emo_embeddings=get_embeddings(inp,token)
    print("emo_embeddings:{}\nemo_embeddings_shape:{}".format(emo_embeddings,tf.shape(emo_embeddings)))
    extend_emo_embeddings1=extend_emo_embeddings(emo_embeddings)
    output, state = self.gru(x, initial_state = hidden)
    return output, state, extend_emo_embeddings1

  def initialize_hidden_state(self):
    #返回一个零矩阵，维度为batch_sz * enc_units
    return tf.zeros((self.batch_sz, self.enc_units))


class EmotionEncoder(tf.keras.Model):    
  def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
    super(EmotionEncoder,self).__init__()
    
    self.gru = tf.keras.layers.GRU(enc_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')

  def call(self, extend_emo_embeddings, encoder_outputs, mask_src):
    
    #transformer模型encoder输出的output(即隐藏层)，与emotion_embedding进行拼接
    #shape=(batch_sz,max_len,units)第0维是batch，第1维是每个单词，第2维是单词的词向量表示
    #在第二维将词向量与情感标注向量进行拼接
    hidden_state_with_emo=tf.concat([encoder_outputs, extend_emo_embeddings],2)
    #拼接后的hidden_state_with_emo作为EmotionInputEncoder的输入
    output, state = self.gru(hidden_state_with_emo)
    return output,state
    

class BahdanauAttention(tf.keras.Model):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    #Dense全连接层，output = activation(dot(input, kernel) + bias)，将input变成维度为units的向量
    #units:输出的维度大小，改变inputs的最后一维
    #inputs=tf.ones([2,3,4,20])
    #a=tf.layers.dene(inputs,60)
    #print(a.get_shape())------->(2,3,4,60)
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, query, values):#
    # hidden shape == (batch_size, hidden size)
    # hidden_with_time_axis shape == (batch_size, 1, hidden size)
    # we are doing this to perform addition to calculate the score
    # expand_dims()，在指定维度上，为tensor加一个维度
    hidden_with_time_axis = tf.expand_dims(query, 1)
    # score shape == (batch_size, max_length, hidden_size)
    # score = W3(tanh(W1(EncoderOutput)+W2(DecoderHidden)))
    # 意义:query*key=score，即以点乘的方法，用query去查询每一个key，从而得到query与每一个key的关联程度
    score = self.V(tf.nn.tanh(self.W1(values) + self.W2(hidden_with_time_axis)))
    # attention_weights shape == (batch_size, max_length, 1)
    # we get 1 at the last axis because we are applying score to self.V
    attention_weights = tf.nn.softmax(score, axis=1)
    # context_vector shape after sum == (batch_size, hidden_size)
    context_vector = attention_weights * values
    context_vector = tf.reduce_sum(context_vector, axis=1)
    return context_vector, attention_weights



class ContextDecoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
    #vocab_size=20000   embedding_dim=100   dec_units=200   batch_sz=128
    super(ContextDecoder, self).__init__()
    self.batch_sz = batch_sz
    self.dec_units = dec_units
    self.gru = tf.keras.layers.GRU(self.dec_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
    self.fc = tf.keras.layers.Dense(vocab_size)  
    self.attention = BahdanauAttention(self.dec_units)

  def call(self, dec_input, token, hidden, enc_output):
    """
    execute.py  seq2seqModel.decoder(dec_input, dec_hidden, enc_out)
    x是存有'start'序号的列表，hidden是encoder输出的state，enc_output是encoder输出的output
    """
    context_vector, attention_weights = self.attention(hidden, enc_output)  
    #x作为decoder的输入向量，通过embedding转换成128维
    x=get_embeddings_for_targ(dec_input,token)
    #extend_emo_embeddings1=extend_emo_embeddings(emo_embeddings)
    x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)  
    output, state = self.gru(x)
    output = tf.reshape(output, (-1, output.shape[2]))    
    x = self.fc(output)
    return x, state, attention_weights


def get_mime_emotions(emo_embeddings):
  mime_emotions=[]
  emo_embeddings=emo_embeddings.numpy()
  for emo_embedding in emo_embeddings:
    for emotion,embedding in emo_embeddings_in_dict:
      if emo_embedding == embedding:
        return 



class PredictEncoder(tf.keras.Model):
  #两层，第一层Embedding，第二层GRU
  def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
    super(PredictEncoder, self).__init__()
    self.batch_sz = batch_sz
    self.enc_units = enc_units
    #将正整数（索引）转换为固定大小的密集向量，只能用作模型的第一层
    #vocab_size=20000,embedding_dim=100
    self.gru = tf.keras.layers.GRU(self.enc_units,
                                   return_sequences=True,#return the last output in the output sequence, or the full sequence.
                                   return_state=True,#return the last state in addition to the output.
                                   recurrent_initializer='glorot_uniform')
    
  #调用
  def call(self, inp, token, hidden):
    x,emo_embeddings=get_embeddings(inp,token)    
    extend_emo_embeddings1=extend_emo_embeddings(emo_embeddings)
    output, state = self.gru(x, initial_state = hidden)
    return output, state, extend_emo_embeddings1

  def initialize_hidden_state(self):
    #返回一个零矩阵，维度为batch_sz * enc_units
    return tf.zeros((self.batch_sz, self.enc_units))


vocab_inp_size = gConfig['enc_vocab_size']#20000
vocab_tar_size = gConfig['dec_vocab_size']#20000
embedding_dim=gConfig['emp_embedding_dim']#100
units=gConfig['emp_layer_size']#200
BATCH_SIZE=gConfig['batch_size']

context_encoder = ContextEncoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)

context_decoder = ContextDecoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE)

emotion_encoder=EmotionEncoder(vocab_inp_size,embedding_dim,units,BATCH_SIZE)



optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)
  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask

  return tf.reduce_mean(loss_)

#变量的保存与恢复。tf.train.Checkpoint接受的初始化参数比较特殊，是一个**kwargs，
#即一系列键值对，键名（等号左黄色）随意取，值（等号右）为需要保存的对象
checkpoint = tf.train.Checkpoint(optimizer=optimizer,context_encoder=context_encoder,emotion_encoder=emotion_encoder,context_decoder=context_decoder)

#@tf.function
def train_step(inp, targ, inp_lang, targ_lang, enc_hidden):
  #target_lang是target_token
  loss = 0
  #GradientTape()自动监视可训练的变量
  with tf.GradientTape() as tape:
    enc_output, enc_hidden, emo_embeddings = context_encoder(inp, inp_lang, enc_hidden)
    enc_output, enc_hidden = emotion_encoder(emo_embeddings, enc_output, enc_hidden)

    dec_hidden = enc_hidden
    #[[2],[2],[2],...] shape=(32, 1)
    dec_input = tf.expand_dims([targ_lang.word_index['start']] * BATCH_SIZE, 1)
    for t in range(1, targ.shape[1]):   
      predictions, dec_hidden, _ = context_decoder(dec_input, targ_lang, dec_hidden, enc_output)
      loss += loss_function(targ[:, t], predictions)
      #targ shape=(batch_sz,max_lenth),该步是获取每一个batch的第t个元素
      #[[num],[num],[num],...]  shape=(batch_sz,1) 
      dec_input = tf.expand_dims(targ[:, t], 1)

  batch_loss = (loss / int(targ.shape[1]))
  variables = context_encoder.trainable_variables + context_decoder.trainable_variables
  #默认情况下，只要调用GradientTape.gradient()方法，就会释放GradientTape拥有的资源
  gradients = tape.gradient(loss, variables)
  optimizer.apply_gradients(zip(gradients, variables))
  return batch_loss

  