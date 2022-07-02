import os
import json
import csv
read_data_dir= r'F:\PycharmProject\kbqa\CBLUE-main\MagicChangeData\genData\KUAKE-QIC'
output_data_dir= r'F:\PycharmProject\sbert\Data_cblue'

label_list = [line.strip() for line in open('cblue_label','r',encoding='utf8')]
print(label_list)
label2id = {label:idx for idx,label in enumerate(label_list)}
# print(label2id['治疗方案'])
def __B2D__data(readfile,writefile):
    if not os.path.exists(os.path.join(read_data_dir, readfile)):
        raise Exception("Not Found file:{} Error".format(os.path.join(read_data_dir, readfile)))

    with open(os.path.join(read_data_dir, readfile), 'rt', encoding="utf-8") as readfile:
        with open(os.path.join(output_data_dir, writefile), mode='w', encoding='utf-8',newline='') as out_file:
            #读取所有数据到data
            data = json.load(readfile)
            #建立writer对象
            writer = csv.writer(out_file)
            for item in data:
                #把json数据中list列表转换成想要的格式
                linex = [item['query'],item['label'],1]
                writer.writerow(linex)
                line=[0,1,2,3,4,5,6,7,8,9,10]
                n=0
                for label in label_list:
                    if item['label']!=label:
                        line[n]=[item['query'],label,0]
                        writer.writerow(line[n])
                        n = n + 1
        out_file.close()
    readfile.close()
    return 0

def sbert4softmax(readfile,writefile):
    if not os.path.exists(os.path.join(read_data_dir, readfile)):
        raise Exception("Not Found file:{} Error".format(os.path.join(read_data_dir, readfile)))

    with open(os.path.join(read_data_dir, readfile), 'rt', encoding="utf-8") as readfile:
        with open(os.path.join(output_data_dir, writefile), mode='w', encoding='utf-8',newline='') as out_file:
            #读取所有数据到data
            data = json.load(readfile)
            #建立writer对象
            writer = csv.writer(out_file)
            for item in data:
                #把json数据中list列表转换成想要的格式
                linex = [item['query'],item['label'],label2id[item['label']]]
                writer.writerow(linex)
        out_file.close()
    readfile.close()
    return 0
def sbert4Triplet(readfile,writefile):
    if not os.path.exists(os.path.join(read_data_dir, readfile)):
        raise Exception("Not Found file:{} Error".format(os.path.join(read_data_dir, readfile)))

    with open(os.path.join(read_data_dir, readfile), 'rt', encoding="utf-8") as readfile:
        with open(os.path.join(output_data_dir, writefile), mode='w', encoding='utf-8',newline='') as out_file:
            #读取所有数据到data
            data = json.load(readfile)
            #建立writer对象
            writer = csv.writer(out_file)
            for item in data:
                #把json数据中list列表转换成想要的格式
                positive_label = item['label']
                for label in label_list:
                    if item['label'] != label:
                        negative_label = label
                        linex = [item['query'] ,positive_label,negative_label]
                        writer.writerow(linex)
        out_file.close()
    readfile.close()
    return 0
#把cblue验证集中的label删掉
def dev2test(readfile,writefile):
    if not os.path.exists(os.path.join(read_data_dir, readfile)):
        raise Exception("Not Found file:{} Error".format(os.path.join(read_data_dir, readfile)))

    with open(os.path.join(read_data_dir, readfile), 'rt', encoding="utf-8") as readfile:
        with open(os.path.join(output_data_dir, writefile), mode='w', encoding='utf-8',newline='') as out_file:
            #读取所有数据到data
            data = json.load(readfile)
            data2 = []
            for item in data:
                item2 = {}
                item2['id'] = item['id']
                item2['query'] = item['query']
                data2.append(item2)

            json.dump(data2, out_file, ensure_ascii=False, indent=2)
        out_file.close()
    readfile.close()
    return 0
#找出模型预测错误的label
def findwrong(readfile,readfile2,writefile):
    if not os.path.exists(os.path.join(read_data_dir, readfile)):
        raise Exception("Not Found file:{} Error".format(os.path.join(read_data_dir, readfile)))

    with open(os.path.join(read_data_dir, readfile), 'rt', encoding="utf-8") as readfile:
        with open(os.path.join(output_data_dir, readfile2), 'rt', encoding="utf-8") as readfile2:
            with open(os.path.join(output_data_dir, writefile), mode='w', encoding='utf-8',newline='') as out_file:

                #读取所有数据到data
                data = json.load(readfile)
                data2 = json.load(readfile2)
                data3 = []
                for item in data:
                    for item2 in data2:
                        item3 = {}
                        if item['id'] == item2['id'] and item['label']!=item2['label']:
                            item3['id'] = item['id']
                            item3['query'] = item['query']
                            item3['sbert'] = item2['label']
                            item3['true'] = item['label']
                            data3.append(item3)

                json.dump(data3, out_file, ensure_ascii=False, indent=2)
            out_file.close()
        readfile2.close()
    readfile.close()
    return 0
#11:两个模型都预测对了，10：sbert对bert错，00：都错
def count11x10x00(truefile,sbertfile,bertfile,writefile):

    with open(os.path.join(read_data_dir, truefile), 'rt', encoding="utf-8") as truefile:
        with open(os.path.join(output_data_dir, sbertfile), 'rt', encoding="utf-8") as sbertfile:
            with open(os.path.join(output_data_dir, bertfile), 'rt', encoding="utf-8") as bertfile:
                with open(os.path.join(output_data_dir, writefile), mode='w', encoding='utf-8',newline='') as out_file:

                    #读取所有数据到data
                    data1 = json.load(truefile)
                    data2 = json.load(sbertfile)
                    data3 = json.load(bertfile)
                    data4 = []
                    for i in range(1955):
                        item4 = {}
                        if data1[i]['label'] == data2[i]['label'] and data1[i]['label'] == data3[i]['label']:
                            item4['id11'] = data1[i]['id']
                            item4['query'] = data1[i]['query']
                            item4['label'] = data1[i]['label']
                            item4['sbert'] = data2[i]['label']
                            item4['bert'] = data3[i]['label']
                            data4.append(item4)
                        elif data1[i]['label'] == data2[i]['label'] and data1[i]['label'] != data3[i]['label']:
                            item4['id10'] = data1[i]['id']
                            item4['query'] = data1[i]['query']
                            item4['label'] = data1[i]['label']
                            item4['sbert'] = data2[i]['label']
                            item4['bert'] = data3[i]['label']
                            data4.append(item4)
                        elif data1[i]['label'] != data2[i]['label'] and data1[i]['label'] != data3[i]['label']:
                            item4['id00'] = data1[i]['id']
                            item4['query'] = data1[i]['query']
                            item4['label'] = data1[i]['label']
                            item4['sbert'] = data2[i]['label']
                            item4['bert'] = data3[i]['label']
                            data4.append(item4)
                    json.dump(data4, out_file, ensure_ascii=False, indent=2)
                out_file.close()
            bertfile.close()
        sbertfile.close()
    truefile.close()
    return 0
def showcblue(readfile):
    if not os.path.exists(os.path.join(read_data_dir, readfile)):
        raise Exception("Not Found file:{} Error".format(os.path.join(read_data_dir, readfile)))

    with open(os.path.join(read_data_dir, readfile), 'rt', encoding="utf-8") as readfile:
            #读取所有数据到data
            data = json.load(readfile)
            maxlength = 1
            sumlength = 0
            minlength = 32
            count = 0
            for item in data:
                count = count + 1
                #把json数据中list列表转换成想要的格式
                if len(item['query']) < minlength:
                    # print(len(item['query']))
                    # sumlength = sumlength + len(item['query'])
                    # maxlength = len(item['query'])
                    minlength = len(item['query'])
                    print(item['query'])
            print(count)
            # print(sumlength)
            # avglenth = sumlength/count
            # print(avglenth)
            print(minlength)
    readfile.close()
    return 0
# __B2D__data('KUAKE-QIC_dev.json','cblue_dev.csv')
# sbert4softmax('KUAKE-QIC_train.json','cblue4softmax_train.csv')
# sbert4Triplet('KUAKE-QIC_dev.json','cblue4triplet_dev.csv')
# dev2test('KUAKE-QIC_dev.json','KUAKE-QIC_dev2test.json')
# findwrong('KUAKE-QIC_dev.json','KUAKE-QIC_test_sbert.json','sbert_result.json')
count11x10x00('KUAKE-QIC_dev.json','KUAKE-QIC_test_sbert.json','KUAKE-QIC_test_bert.json','statistic.json')
# showcblue('KUAKE-QIC_test.json')