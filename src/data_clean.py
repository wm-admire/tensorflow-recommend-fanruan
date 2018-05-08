# encoding: utf-8
# author: yaoh.wu

# 进行数据清洗，拿出存在ip,username,userrole的数据，移除了无用的param,browser,sql参数，修正换行和乱码


import logging
import re
import sys

from src.utils import save_to_file, config_log

# 原始数据合法起始字段正则
data_pattern = r"^\"[0-9]{7}\"\t\""

# 原始数据路径
file_path = "../data/fr_data.txt"

# 存在ip,username,userrole的数据，移除了暂时无用的param,browser,sql参数
pure_data_file_path = "../data/fr_data.csv"

# 日志文件路径
log_file_path = "../log/log.log"

# 存储文件的窗口大小
page_size = 10000


def main():
    """
    main
    """
    config_log(log_file_path)


def clean_data():
    """
    先简单的处理一下数据，
    主要目的是数据清理，移除乱码的数据以及在文本中错误的换行修正
    """
    logging.info("clean data begin")

    # 临时数据列表
    tmp_data_list = []
    tmp_line = ""

    with open(file_path, "rb") as file:
        for line in file:
            # 由于其他数据会出现还行现象,不是以id开头的行是其他数据换行生成的数据，这些数据修正一下
            # 修正的思路是读到id之后不是立即处理该行数据，而是继续读取，追加后面行的内容，直到读到下一个id，再处理这些暂存起来的数据信息
            # 字节数据解析成字符串，并移除换行符
            line = line.decode(encoding="utf-8").replace("\r\n", "\n").replace("\n", "")
            # 粗略地处理数据
            if line.startswith("\"ID\"") is False and re.match(data_pattern, line) is None:
                # 既不是表头也不是以id开头，说明时上一项错误换行的内容，记录该内容并继续读下一行
                tmp_line += line
                continue
            else:
                # 读取到了下一个 id 的信息，处理一下上一次暂存的信息
                if tmp_line.strip():
                    data, need = __parse_data(tmp_line)
                    if need:
                        tmp_data_list.append(data)
                        # 分块写入文件,避免占用大量内存
                        if len(tmp_data_list) > page_size:
                            save_to_file(tmp_data_list, pure_data_file_path)
                            tmp_data_list.clear()
                # 下一个id的信息
                tmp_line = line.replace("\r\n", "\n").replace("\n", "")

    # 由于最后一条没有下一个 id 帮助触发数据处理了，因此在结尾处主动处理一下最后一条 id 的数据
    if tmp_line.strip():
        data, need = __parse_data(tmp_line)
        if need:
            tmp_data_list.append(data)
            # 分块写入文件,避免占用大量内存
            if len(tmp_data_list) > page_size:
                save_to_file(tmp_data_list, pure_data_file_path)
                tmp_data_list.clear()

    # 最后再处理一下末尾阶段不足1000条的数据
    save_to_file(tmp_data_list, pure_data_file_path)
    tmp_data_list.clear()
    logging.info("clean data end")


def __parse_data(line):
    """
    粗略处理一下数据，移除一些乱码的信息
    :param line: 字符串数据
    :return: ['id', 'tname', 'type', 'ip', 'username', 'userrole', 'time', 'logtime', 'memory']
    """
    # line 已经通过decode转成字符串了
    # 转小写,去除前后空格,去除双引号,拆分
    line = line.lower().strip().lstrip("\"").rstrip("\"").split("\"\t\"")
    # ['id', 'tname', 'type', 'param', 'ip', 'username', 'userrole', 'time', 'logtime', 'sql', 'browser', 'memory']
    # 删除param,sql,browser等信息
    # 部分模板路径以 / 开头，替换掉
    if line[1].startswith("/"):
        line[1] = line[1].replace("/", "", 1)
    # param
    del line[3]
    # sql
    del line[8]
    # browser
    del line[8]

    # ['id', 'tname', 'type', 'ip', 'username', 'userrole', 'time', 'logtime', 'memory']
    # 是否存在ip，username，userrole
    if line[4] != "" and line[5] != "" and line[6] != "":
        need = True
    else:
        need = False

    return line, need


if __name__ == '__main__':
    main()
    clean_data()
    sys.exit(0)
