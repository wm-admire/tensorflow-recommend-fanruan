# encoding: utf-8
# author: yaoh.wu

# encoding: utf-8
# author: yaoh.wu
import csv
import logging
import pickle


def config_log(log_file_path):
    # 配置日志
    logging.basicConfig(
        format='%(asctime)s %(pathname)s line:%(lineno)d %(message)s ',
        # 将日期格式化成正常模式
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file_path,
        level=logging.INFO
    )


def save_to_file(data, target_file_path):
    """
    存储到文件系统中
    :param data: 数据列表
    :param target_file_path: 文件路径
    """
    with open(target_file_path, "a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, dialect="excel")
        for item in data:
            try:
                writer.writerow(item)
            except UnicodeEncodeError as err:
                # 移除乱码的数据
                logging.debug("happened: " + err.reason)
                continue


def save_to_database(data):
    """
    保存到数据库中
    :param data: 数据
    :return: 保存成功的数据
    """
    # todo 存储到数据库中
    return data


def save_params(params):
    """
    Save parameters to file
    """
    pickle.dump(params, open('params.p', 'wb'))


def load_params():
    """
    Load parameters from file
    """
    return pickle.load(open('params.p', mode='rb'))


def get_batch(xs, ys, batch_size):
    for start in range(0, len(xs), batch_size):
        end = min(start + batch_size, len(ys))
        yield xs[start:end], ys[start:end]
