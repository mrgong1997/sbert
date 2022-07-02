import os
import random
from sentence_transformers import  InputExample
read_data_dir= r'D:\PycharmProjects\104\lilo\sbert\data'
output_data_dir= r'D:\PycharmProjects\104\lilo\sbert\Ddata'
def __B2D__data(readfile,writefile):
    if not os.path.exists(os.path.join(read_data_dir, readfile)):
        raise Exception("Not Found file:{} Error".format(os.path.join(read_data_dir, readfile)))
    if not os.path.exists(os.path.join(output_data_dir, writefile)):
        raise Exception("Not Found file:{} Error".format(os.path.join(output_data_dir, writefile)))

    with open(os.path.join(read_data_dir, readfile), mode='r', encoding='utf-8') as in_file:
        with open(os.path.join(output_data_dir, writefile), mode='w', encoding='utf-8') as out_file:
            while True:
                readline = in_file.readline()
                if readline == '':
                    break
                _sentence_pairs = readline.strip().split('\t')
                _sentence_one = _sentence_pairs[0].strip()
                _sentence_two = _sentence_pairs[1].strip()
                _label = float(_sentence_pairs[2].strip())
                if _label==1:
                    _label=round(random.uniform(0.7, 1),2)
                elif _label==0:
                    _label=round(random.uniform(0,0.3),2)
                dataline=[_sentence_one+"\t",_sentence_two+"\t",str(_label)+'\n']
                out_file.writelines(dataline)
__B2D__data('test.txt','test.txt')