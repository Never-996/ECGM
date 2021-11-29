
from configparser import SafeConfigParser

def get_config(config_file='seq2seq.ini'):
    #初始化一个SafeConfigParser类对象，对指定的配置文件做增删改查操作
    parser = SafeConfigParser()
    parser.read(config_file,encoding='utf-8')
    # get the ints, floats and strings
    _conf_ints = [ (key, int(value)) for key,value in parser.items('ints') ]
    #_conf_floats = [ (key, float(value)) for key,value in parser.items('floats') ]
    _conf_strings = [ (key, str(value)) for key,value in parser.items('strings') ]
    #返回一个字典，字典中存配置，{[(key,value),(key,value),()()...]}
    return dict(_conf_ints  + _conf_strings)