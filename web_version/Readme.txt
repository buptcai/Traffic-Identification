Author:Cai
Date:2019/3/19
---------------------------------------------------------------------------
Support JPG and PNG format.
---------------------------------------------------------------------------
Start-up Server:
python web_server.py --port= --http_file= --database_path= --weights_path = 
---------------------------------------------------------------------------
Test:
curl localhost:8000/bridge -F "file=@**.jpg"
---------------------------------------------------------------------------
category_lists = ['biaoPai','gaiLiang','gongShangCeQiang','huLan','huPo',
	           'liang','qiaoDun','qiaoMian','qiaoTai','shenSuoFeng',
  	           'xieshuiKong','yiQiang','zhiZuo','zhuGongQuan','zhuiPo']