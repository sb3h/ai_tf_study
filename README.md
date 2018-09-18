# ai_tf_study
环境
python 3.5.1

linux下运行命令
vi ~/.pip/pip.conf
然后写入如下内容并保存
windows:
%HOMEPATH%\pip\pip.ini

-------------------------pip.ini-------------------------
 [global]
 trusted-host =  mirrors.aliyun.com
 index-url = https://mirrors.aliyun.com/pypi/simple
-------------------------pip.ini-------------------------

pip install tensorflow==1.5
pip install jieba
pip install jupyter
	启动命令:jupyter notebook
pip install matplotlib
	画图工具包