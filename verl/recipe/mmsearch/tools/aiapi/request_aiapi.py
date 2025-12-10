# coding=utf-8
"""
注意点1：实际使用时请按需修改requestAIAPI中的请求参数(接口文档：https://iapi.baidu-int.com/apidoc/project-358102/api-4362170)
注意点2：BNS_NAME为线上bns，访问线上环境时时，AK/SK需要替换为自己的移动开发者中心申请下来的AK/SK，且要求请求环境已安装get_instance_by_service命令
注意点3：访问线下环境，可直接修改run()函数中HOST。
"""
import json
import hashlib
import bns
import time
import urllib.request
import socket

GET_INSTANCE_BY_SERVICE = '/usr/bin/get_instance_by_service'
BNS_NAME = 'bns://group.smartbns-from_product=api-default%group.flow-api-bfe.NWISE.all'
AK = ''
SK = ''
TIMEOUT = 10


def urlRequest(url, headers, body):
    """
    http 请求处理
    @param data 字符串类型
    @param header 请求头
    @param url 地址
    @param timeout 超时时间 
    """
    encode_body = json.dumps(body).encode('utf-8')
    req = urllib.request.Request(url, data=encode_body, headers=headers)

    return urllib.request.urlopen(req, timeout=TIMEOUT).read()


def api_auth_sign(params):
    """
    计算Api—Auth
    """
    # Concatenate all strings in the list
    concatenated = ''.join(params)
    
    # Create an MD5 hash of the concatenated string
    md5_hash = hashlib.md5(concatenated.encode('utf-8')).hexdigest()
    
    # Check if the length of the hash is less than 22
    if len(md5_hash) < 22:
        return ""
    
    # Specific indices from which to pick characters from the MD5 hex string
    auth_indices = [7, 3, 17, 13, 1, 21]
    
    # Build the result string using characters at the specified indices
    result = ''
    for idx in auth_indices:
        result += md5_hash[idx]
    
    return result


def requestAIAPI(host, word):
    """
    请求AIAPI
    """
    # url
    word = word.strip()
    hostname = socket.gethostname()
    ip = socket.gethostbyname(hostname)
    query = {
        'word': word,
        'clientip':ip,
    }
    encodeQuery = urllib.parse.urlencode(query)
    url = host + '/search/api?' + encodeQuery
    # header
    header = {
        'Api-Key': AK,
    }
    params = [AK, word, ip, SK]
    header['Api-Auth'] = api_auth_sign(params)
    # body
    body = {
        'size': {
            'main': 10
        }
    } 
    try:
        response = urlRequest(url, header, body)
    except Exception as e:
        print(e)
        return ''

    return response


def run():
    """
    入口函数
    """
    # 1、线上环境
    apibns = bns.BNS(BNS_NAME)
    # host = apibns.get_url() 
    host = 'http://10.103.44.13:8200/' # 使用调研环境ip:port访问，有限期至5.27
    
    word_list = ["适乐肤洗面奶", "电子琴价格", "西红柿炒鸡蛋怎么做", "勾股定理", "为什么要学习编程，少儿编程有用吗",
                "如何看洋流图", "如何学好数学", "每天都睡觉还是很困是什么原因"]
    start_time = time.time()
    
    
    for word in word_list:
        response = requestAIAPI(host, word)
        response_json = json.loads(response)
        
        print(response_json.keys())
        
        print(json.dumps(response_json, indent=4, ensure_ascii=False))
        # print(response_json)
        
    end_time = time.time()
    # print(f"平均耗时: {(end_time - start_time)/len(word_list):.2f}s")
    
if __name__ == '__main__':
    run()