from tqdm import tqdm
import io
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

def get_emo_words(filename):
    pos_list=[]
    neg_list=[]
    with open(filename) as file_object:
        for line in tqdm(file_object):
            line_in_split=line.split(',')
            words=line_in_split[3].split(' ')
            if emotions.index(line_in_split[2]) in positive_emotions:
                pos_list.extend(words)
                pos_list=list(set(pos_list))
            else:
                neg_list.extend(words)
                neg_list=list(set(neg_list))
    pos_list1=[i for i in pos_list if i not in neg_list]
    neg_list1=[i for i in neg_list if i not in pos_list]
    print("pos_list:\n{}neg_list:\n{}".format(pos_list1,neg_list1))
    return pos_list1,neg_list1


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

"""
filename1='train_data/Eng_conv/train.csv'
filename2='train_data/Eng_conv/save.csv'
preprocess(filename1,filename2)
"""
filename2='train_data/Eng_conv/save.csv'
get_emo_words(filename2)
