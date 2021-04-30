# coding=gbk
import os
import time
import json
import requests
 
def getManyPages(pages):
    params=[]
    for i in range(0, 12*pages+12, 12):
        params.append({
            'resource_id': 28266,
            'from_mid': 1,
            'format': 'json',
            'ie': 'utf-8',
            'oe': 'utf-8',
            'query': '��������',
            'sort_key': '',
            'sort_type': 1,
            'stat0': '',
            'stat1': '����',
            'stat2': '',
            'stat3': '',
            'pn': i,
            'rn': 12
                  })
    url = 'https://sp0.baidu.com/8aQDcjqpAAV3otqbppnN2DJv/api.php'
#    names = []
#    img_results = []
    x = 0
    f = open('F:\\Race_Classification\\spider\\starName_africa.txt', 'w', encoding='utf-8')
    for param in params:
        try:
            res = requests.get(url, params=param)
            js = json.loads(res.text)
            results = js.get('data')[0].get('result')
        except AttributeError as e:
            print(e)
            continue
        for result in results:
            img_name = result['ename']
#            img_url = result['pic_4n_78']
#            img_result =  [img_name,img_url]
#            img_results.append(img_result)
            f.write(img_name+'\n')
    #        names.append(img_name)
 
        if x % 10 == 0:
            print('��%dҳ......'%x)
        x += 1
    f.close()
 
if __name__ == '__main__':
    getManyPages(400)