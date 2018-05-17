#coding=utf-8

#urllib模块提供了读取Web页面数据的接口
import urllib.request
#re模块主要包含了正则表达式
import re
import os
#定义一个getHtml()函数
def getHtml(url):
    page = urllib.request.urlopen(url)  #urllib.urlopen()方法用于打开一个URL地址
    html = page.read() #read()方法用于读取URL上的数据
    html = html.decode('UTF-8')
    return html

def getImg(auto, html, x):
    imgre = re.compile('thumbURL.*?\.jpg')
    imglist = imgre.findall(html)
    path = 'D:\\tmp\\baidu_images'
    if not os.path.isdir(path):
        os.makedirs(path)
    paths = path + '\\'  # 保存在test路径下
    for imgurl in imglist:
        fileName = auto + '.'+ str(x)
        url = imgurl.split('":"')[1]
        urllib.request.urlretrieve(url, '{}{}.jpg'.format(paths, fileName))
        x = x + 1
    return x

def start():
    auto_dictionary = {
        'Benz_GLA': ['https://image.baidu.com/search/index?tn=baiduimage&ipn=r&ct=201326592&cl=2&lm=-1&st=-1&fm=result&fr=&sf=1&fmq=1513330277586_R&pv=&ic=0&nc=1&z=&se=1&showtab=0&fb=0&width=&height=&face=0&istype=2&ie=utf-8&word=%E5%A5%94%E9%A9%B0gla'],
        'Toyota_Camry': ['https://image.baidu.com/search/index?ct=201326592&cl=2&st=-1&lm=-1&nc=1&ie=utf-8&tn=baiduimage&ipn=r&rps=1&pv=&fm=rs1&word=%E4%B8%B0%E7%94%B0%E5%87%AF%E7%BE%8E%E7%91%9E2018&oriquery=%E5%87%AF%E7%BE%8E%E7%91%9E2018&ofr=%E5%87%AF%E7%BE%8E%E7%91%9E2018&sensitive=0'],
        'Honda_CR-V': ['https://image.baidu.com/search/index?tn=baiduimage&ipn=r&ct=201326592&cl=2&lm=-1&st=-1&fm=result&fr=&sf=1&fmq=1513330795310_R&pv=&ic=0&nc=1&z=&se=1&showtab=0&fb=0&width=&height=&face=0&istype=2&ie=utf-8&word=%E6%9C%AC%E7%94%B0crv2018%E6%AC%BE']
    }
    for auto in auto_dictionary.keys():
        print(auto)
        urls = auto_dictionary[auto]
        x = 1
        for url in urls:
            html = getHtml(url)
            x = getImg(auto, html, x)

start()
