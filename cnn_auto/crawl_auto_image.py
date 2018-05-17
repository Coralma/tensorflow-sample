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
    imgre = re.compile(r'<img src="(http://.+?)" ')
    imglist = imgre.findall(html)
    path = 'D:\\tmp\\images'
    if not os.path.isdir(path):
        os.makedirs(path)
    paths = path + '\\'  # 保存在test路径下
    for imgurl in imglist:
        fileName = auto + '.'+ str(x)
        urllib.request.urlretrieve(imgurl, '{}{}.jpg'.format(paths, fileName))
        x = x + 1
    return x

def start():
    auto_dictionary = {
        'Benz_GLA': ['http://photo.bitauto.com/modelmore/124769/6/1/#photoanchor',
                      'http://photo.bitauto.com/serial/4477/2017/c24861/6/1/#photoanchor',
                      'http://photo.bitauto.com/serial/4477/2017/c18971/6/1/#photoanchor',
                      'http://photo.bitauto.com/serialmore/4477/2017/14/1/#photoanchor'],
        'Toyota_Camry': ['http://photo.bitauto.com/modelmore/127165/6/1/#photoanchor',
                         'http://photo.bitauto.com/modelmore/127165/c15542/6/1/#photoanchor',
                         'http://photo.bitauto.com/serial/1991/2018/c25364/6/1/#photoanchor',
                         'http://photo.bitauto.com/serialmore/1991/2018/14/1/#photoanchor'],
        'Honda_CR-V': ['http://photo.bitauto.com/serial/1660/2017/c484/6/1/#photoanchor',
                        'http://photo.bitauto.com/serial/1660/2017/c24130/6/1/#photoanchor',
                        'http://photo.bitauto.com/serial/1660/2017/c24134/6/1/#photoanchor',
                        'http://photo.bitauto.com/serial/1660/2017/c18790/6/1/#photoanchor',
                        'http://photo.bitauto.com/serialmore/1660/2017/14/1/#photoanchor']
    }
    for auto in auto_dictionary.keys():
        print(auto)
        urls = auto_dictionary[auto]
        x = 1
        for url in urls:
            html = getHtml(url)
            x = getImg(auto, html, x)

start()
