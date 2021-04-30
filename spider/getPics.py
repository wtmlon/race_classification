# coding=gbk
import os
from icrawler.builtin import BingImageCrawler
path = r'F:\\Race_Classification\\spider\\starName_cauca'
f = open('F:\\Race_Classification\\spider\\starName_cauca.txt', 'r', encoding='utf-8')
lines = f.readlines()
for i, line in enumerate(lines):
    name = line.strip('\n')
    file_path = os.path.join(path, name)
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    bing_storage = {'root_dir': file_path}
    bing_crawler = BingImageCrawler(parser_threads=2, downloader_threads=4, storage=bing_storage)
    bing_crawler.crawl(keyword=name, max_num=50)
    print('��{}λ���ǣ�{}'.format(i, name))
'''
path = r'F:\\Race_Classification\\spider\\starName_black'
if not os.path.exists(path):
    os.makedirs(path)
bing_storage = {'root_dir': path}
bing_crawler = BingImageCrawler(parser_threads=2, downloader_threads=4, storage=bing_storage)
bing_crawler.crawl(keyword='black man', max_num=40000)
'''