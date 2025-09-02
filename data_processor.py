# sentiment_analyzer/data_processor.py
import pandas as pd
import re
import jieba
import os

# 1. 数据加载与预处理
def load_data(train_path, dev_path=None, test_path=None):
    """
    加载数据集
    """
    # 如果只提供了一个文件路径（单文件模式）
    if dev_path is None and test_path is None:
        try:
            # 使用传入的路径参数
            data = pd.read_csv(train_path, sep='\t')
        except:
            # 如果失败，尝试自动检测分隔符
            data = pd.read_csv(train_path, sep=None, engine='python')
            
        if 'label' not in data.columns and 'text_a' not in data.columns:
            # 尝试检测并调整列名
            if len(data.columns) == 2:
                data.columns = ['label', 'text_a']
            elif 'text' in data.columns:
                data['text_a'] = data['text']
                if 'label' not in data.columns:
                    data['label'] = -1  # 默认标签
        
        print(f"数据集大小: {len(data)}")
        return data
    
    # 多文件模式（训练集+验证集+测试集）
    try:
        train_data = pd.read_csv(train_path, sep='\t')
    except:
        # 如果失败，尝试自动检测分隔符
        train_data = pd.read_csv(train_path, sep=None, engine='python')
    
    # 处理验证集
    if dev_path:
        try:
            dev_data = pd.read_csv(dev_path, sep='\t')
        except:
            # 如果失败，尝试自动检测分隔符
            dev_data = pd.read_csv(dev_path, sep=None, engine='python')
    else:
        # 如果没有提供验证集，创建空的DataFrame
        dev_data = pd.DataFrame(columns=['qid', 'label', 'text_a'])
    
    # 处理测试集
    if test_path:
        try:
            test_data = pd.read_csv(test_path, sep='\t')
        except:
            # 如果失败，尝试自动检测分隔符
            test_data = pd.read_csv(test_path, sep=None, engine='python')
    else:
        # 如果没有提供测试集，创建空的DataFrame
        test_data = pd.DataFrame(columns=['qid', 'text_a', 'label'])
    
    # 根据提供的数据集格式调整列名
    if 'label' in train_data.columns and 'text_a' in train_data.columns:
        # 训练集格式正确
        pass
    else:
        # 调整训练集列名
        if len(train_data.columns) >= 2:
            train_data.columns = ['label', 'text_a'] + list(train_data.columns[2:])
    
    # 处理验证集列名
    if len(dev_data) > 0:  # 只有当验证集不为空时才处理
        if 'qid' in dev_data.columns and 'label' in dev_data.columns and 'text_a' in dev_data.columns:
            # 验证集格式正确
            pass
        else:
            # 调整验证集列名
            if len(dev_data.columns) >= 3:
                dev_data.columns = ['qid', 'label', 'text_a'] + list(dev_data.columns[3:])
            elif len(dev_data.columns) >= 2:
                dev_data.columns = ['label', 'text_a'] + list(dev_data.columns[2:])
                dev_data.insert(0, 'qid', range(len(dev_data)))
    
    # 处理测试集列名
    if len(test_data) > 0:  # 只有当测试集不为空时才处理
        if 'qid' in test_data.columns and 'text_a' in test_data.columns:
            # 测试集格式正确
            pass
        else:
            # 调整测试集列名
            if len(test_data.columns) >= 2:
                test_data.columns = ['qid', 'text_a'] + list(test_data.columns[2:])
            elif len(test_data.columns) >= 1:
                test_data.columns = ['text_a'] + list(test_data.columns[1:])
                test_data.insert(0, 'qid', range(len(test_data)))
        
        # 如果测试集没有标签，创建一个虚拟标签（-1）用于后续处理
        if 'label' not in test_data.columns:
            test_data['label'] = -1
    
    print(f"训练集大小: {len(train_data)}")
    print(f"验证集大小: {len(dev_data)}")
    print(f"测试集大小: {len(test_data)}")
    
    # 始终返回三个数据集
    return train_data, dev_data, test_data


# 2. 数据清洗与预处理
def clean_text(text):
    """
    清洗文本数据
    """
    if isinstance(text, str):
        # 去除HTML标签
        text = re.sub(r'<.*?>', '', text)
        # 去除URL
        text = re.sub(r'http\S+', '', text)
        # 去除数字
        text = re.sub(r'\d+', '', text)
        # 去除标点符号和特殊字符
        text = re.sub(r'[^\w\s]', '', text)
        # 去除多余空格
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    else:
        return ""

def tokenize(text):
    """
    使用jieba进行分词
    """
    return list(jieba.cut(text))

def preprocess_data(data):
    """
    对数据进行预处理
    """
    # 创建数据副本，避免修改原始数据
    data = data.copy()
    
    # 检查是否有text_a列
    if 'text_a' not in data.columns:
        print("警告: 数据中没有找到 'text_a' 列")
        return data
    
    # 清洗文本
    data['clean_text'] = data['text_a'].apply(clean_text)
    
    # 分词
    data['tokens'] = data['clean_text'].apply(tokenize)
    
    return data

def split_data(data, test_size=0.2, dev_size=0.1, random_state=42):
    """
    将数据集拆分为训练集、验证集和测试集
    """
    from sklearn.model_selection import train_test_split
    
    # 检查数据是否为空
    if len(data) == 0:
        print("警告: 输入数据为空")
        return data, pd.DataFrame(), pd.DataFrame()
    
    # 先分离出测试集
    train_dev_data, test_data = train_test_split(data, test_size=test_size, random_state=random_state)
    
    # 再从剩余数据中分离出验证集
    if dev_size > 0 and len(train_dev_data) > 1:
        train_data, dev_data = train_test_split(train_dev_data, test_size=dev_size/(1-test_size), random_state=random_state)
    else:
        train_data = train_dev_data
        dev_data = pd.DataFrame()
    
    print(f"训练集大小: {len(train_data)}")
    print(f"验证集大小: {len(dev_data)}")
    print(f"测试集大小: {len(test_data)}")
    
    return train_data, dev_data, test_data

def get_data_statistics(data):
    """
    获取数据统计信息
    """
    if len(data) == 0:
        return {"message": "数据集为空"}
    
    stats = {}
    stats['total_samples'] = len(data)
    stats['columns'] = list(data.columns)
    
    # 标签分布统计
    if 'label' in data.columns:
        label_counts = data['label'].value_counts().sort_index()
        stats['label_distribution'] = label_counts.to_dict()
    
    # 文本长度统计
    if 'text_a' in data.columns:
        text_lengths = data['text_a'].astype(str).str.len()
        stats['text_length'] = {
            'mean': text_lengths.mean(),
            'std': text_lengths.std(),
            'min': text_lengths.min(),
            'max': text_lengths.max()
        }
    
    return stats

def print_data_statistics(data, dataset_name="数据集"):
    """
    打印数据统计信息
    """
    print(f"\n📊 {dataset_name} 统计信息:")
    print("=" * 50)
    
    if len(data) == 0:
        print("数据集为空")
        return
    
    stats = get_data_statistics(data)
    
    print(f"总样本数: {stats['total_samples']}")
    print(f"列名: {', '.join(stats['columns'])}")
    
    if 'label_distribution' in stats:
        print(f"\n标签分布:")
        for label, count in stats['label_distribution'].items():
            percentage = count / stats['total_samples'] * 100
            print(f"  标签 {label}: {count} 条 ({percentage:.1f}%)")
    
    if 'text_length' in stats:
        text_stats = stats['text_length']
        print(f"\n文本长度统计:")
        print(f"  平均长度: {text_stats['mean']:.1f} 字符")
        print(f"  标准差: {text_stats['std']:.1f}")
        print(f"  最短: {text_stats['min']} 字符")
        print(f"  最长: {text_stats['max']} 字符")

def validate_data_format(data, dataset_type='train'):
    """
    验证数据格式
    """
    if len(data) == 0:
        print(f"⚠️ {dataset_type}数据集为空")
        return False
    
    required_columns = ['text_a']
    missing_columns = [col for col in required_columns if col not in data.columns]
    
    if missing_columns:
        print(f"❌ {dataset_type}数据集缺少必需列: {missing_columns}")
        return False
    
    # 检查空值
    null_count = data['text_a'].isnull().sum()
    if null_count > 0:
        print(f"⚠️ {dataset_type}数据集中有 {null_count} 个空值")
    
    print(f"✅ {dataset_type}数据集格式验证通过")
    return True


# 3. 使用示例
if __name__ == "__main__":
    # 设置数据文件路径
    train_path = r"C:\Users\cathy\Desktop\sentiment_analysis_project\data\train.tsv"
    dev_path = r"C:\Users\cathy\Desktop\sentiment_analysis_project\data\dev.tsv"
    test_path = r"C:\Users\cathy\Desktop\sentiment_analysis_project\data\test.tsv"
    
    # 加载数据
    print("正在加载数据...")
    train_data, dev_data, test_data = load_data(train_path, dev_path, test_path)
    
    # 验证数据格式
    validate_data_format(train_data, 'train')
    if len(dev_data) > 0:
        validate_data_format(dev_data, 'dev')
    if len(test_data) > 0:
        validate_data_format(test_data, 'test')
    
    # 预处理数据
    print("\n正在预处理训练数据...")
    train_data = preprocess_data(train_data)
    
    if len(dev_data) > 0:
        print("正在预处理验证数据...")
        dev_data = preprocess_data(dev_data)
    
    if len(test_data) > 0:
        print("正在预处理测试数据...")
        test_data = preprocess_data(test_data)
    
    # 显示数据样例
    print("\n训练数据样例:")
    print(train_data.head())
    
    if len(dev_data) > 0:
        print("\n验证数据样例:")
        print(dev_data.head())
    
    if len(test_data) > 0:
        print("\n测试数据样例:")
        print(test_data.head())
    
    # 显示数据统计信息
    print_data_statistics(train_data, "训练集")
    if len(dev_data) > 0:
        print_data_statistics(dev_data, "验证集")
    if len(test_data) > 0:
        print_data_statistics(test_data, "测试集")