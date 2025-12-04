# Copyright (c) 2025 Beijing Youcare Kechuang Pharmaceutical Technology Co. Ltd., All rights reserved.
# Author: Dawei Wang
# This code may not be used, modified, or distributed without prior written consent from Beijing Youcare Kechuang Pharmaceutical Technology Co. Ltd.

import pandas as pd
import re
import time
import subprocess
from datetime import timedelta
import logging
import os
import sys
from collections import defaultdict
import random
import numpy as np
from Bio import SeqIO
from Bio.Seq import Seq


SEED = 42

def check_sequence_characters(df):
    """
    检查DataFrame中sequence列的序列是否只包含AUGC字符
    
    参数:
    df -- pandas.DataFrame, 必须包含'sequence'列和'ID'列
    
    功能:
    1. 预处理：去除每条序列中的空格
    2. 如果所有序列都只包含AUGC字符，打印成功消息
    3. 如果有序列包含其他字符，打印包含异常字符的序列ID和所有异常字符
    """
    # 验证输入
    if 'Sequence' not in df.columns or 'ID' not in df.columns:
        raise ValueError("DataFrame必须包含'Sequence'和'ID'列")
    
    # 定义有效字符集
    valid_chars = {'A', 'U', 'G', 'C'}
    invalid_entries = []
    
    # 遍历每一行
    for index, row in df.iterrows():
        seq = row['Sequence']
        seq_id = row['ID']
        
        # 预处理：去除空格
        if isinstance(seq, str):
            # 去除所有空格（包括普通空格、制表符等）
            # cleaned_seq = ''.join(seq.split())
            cleaned_seq = re.sub(r'\s+', '', seq)
            df.at[index, 'Sequence'] = cleaned_seq
        else:
            cleaned_seq = seq
        
        # 跳过非字符串类型
        if not isinstance(cleaned_seq, str):
            invalid_chars = {f"非字符串类型({type(cleaned_seq).__name__})"}
            invalid_entries.append((seq_id, invalid_chars))
            continue
        
        # 检查每个字符
        invalid_chars = set()
        for char in cleaned_seq.upper():  # 转为大写检查
            if char not in valid_chars:
                invalid_chars.add(char)
        
        if invalid_chars:
            invalid_entries.append((seq_id, invalid_chars))
    
    # 输出结果
    if not invalid_entries:
        print("所有序列都只包含AUGC字符")
    else:
        print("以下序列包含异常字符:")
        for seq_id, invalid_chars in invalid_entries:
            sorted_chars = sorted(invalid_chars)  # 对异常字符排序
            chars_str = ", ".join(sorted_chars)
            print(f"ID: {seq_id}, 异常字符: {chars_str}")
    
    # 返回包含检查结果的DataFrame
    return df


def process_dataframe(df, wb_thrd, elisa_thrd):
    # 创建副本避免修改原DataFrame
    result_df = df.copy()
    
    # 1. 计算序列长度并存入新列
    result_df['Seq_len'] = result_df['Sequence'].str.len()
    
    # 2. 根据wb_thrd生成二值标签
    result_df['Label_wb'] = (result_df['Exp_value(WB)'] >= wb_thrd).astype(int)
    # result_df['Label_wb'] = (result_df['WB'] >= wb_thrd).astype(int)
    
    # 3. 根据elisa_thrd生成二值标签 
    result_df['Label_elisa'] = (result_df['Exp_value(ELISA)'] >= elisa_thrd).astype(int)
    # result_df['Label_elisa'] = (result_df['IgG'] >= elisa_thrd).astype(int)
    
    # 4. 组合含有wb和elisa双标签的ID
    result_df['ID_with_Label'] = (result_df['ID'].astype(str) + '|' + result_df['Label_wb'].astype(str) + '|' + result_df['Label_elisa'].astype(str))

    # 计算并打印Label_wb的正负样本绝对数量比例
    wb_counts = result_df['Label_wb'].value_counts()
    elisa_counts = result_df['Label_elisa'].value_counts()
    
    print("\nLabel_wb 正负样本绝对数量比例:")
    print(f"正样本(1):负样本(0) = {wb_counts.get(1, 0)} : {wb_counts.get(0, 0)}")
    print("\nLabel_elisa 正负样本绝对数量比例:")
    print(f"正样本(1):负样本(0) = {elisa_counts.get(1, 0)} : {elisa_counts.get(0, 0)}")

    return result_df

def write_fasta(df, output_file):
    # 打开文件并写入序列
    with open(output_file, 'w') as f:
        # 遍历DataFrame的每一行
        for _, row in df.iterrows():
            # 获取ID_with_Label和序列
            # seq_id = str(row['ID_with_Label'])
            seq_id = str(row['ID'])
            sequence = str(row['Sequence'])
            
            # 写入FASTA格式：ID行和序列行
            f.write(f">{seq_id}\n")
            f.write(f"{sequence}\n")  # 序列写入为单行，不添加额外换行符
    print(f'序列成功写出至 {output_file}')

def run_cdhit_est(input_fasta, output_prefix, c, n, M=16000, T=16, G=1):
    """
    使用 CD-HIT-EST 去除序列冗余
    
    参数:
    input_fasta: 输入FASTA文件路径
    output_prefix: 输出文件前缀（不含扩展名）
    c: 序列相似性阈值 (0-1, 默认0.9)
    n: 短词长度 (默认5)
    M: 最大内存(MB, 默认16000=16GB)
    T: CPU核心数 (默认4)
    G: 使用全局比对 (默认1=是)
    
    返回:
    输出FASTA文件路径
    """
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("CD-HIT")
    
    # 构建命令
    cmd = [
        "cd-hit-est",
        "-i", input_fasta,
        "-o", f"{output_prefix}.fasta",
        "-c", str(c),
        "-n", str(n),
        "-M", str(M),
        "-T", str(T),
        "-G", str(G),
        "-d", "0",  # 保留完整序列描述
    ]
    
    # # 添加覆盖率参数（推荐）
    # if 0.7 <= c < 0.95:
    #     cmd.extend(["-aS", "0.8", "-aL", "0.8"])
    # cmd.extend(["-aS", "0.8", "-aL", "0.8", "-s", "0.7", "-g", "1"])
    cmd.extend(["-g", "1"])
    
    logger.info(f"执行CD-HIT命令: {' '.join(cmd)}")
    
    # 运行CD-HIT
    try:
        result = subprocess.run(
            cmd, 
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # 检查输出文件
        output_file = f"{output_prefix}.fasta"
        if not os.path.exists(output_file):
            logger.error(f"输出文件未创建: {output_file}")
            raise RuntimeError("CD-HIT未生成输出文件")
        
        # 统计结果
        input_count = sum(1 for _ in open(input_fasta) if _.startswith(">"))
        output_count = sum(1 for _ in open(output_file) if _.startswith(">"))
        reduction = (1 - output_count/input_count) * 100
        
        logger.info(f"去冗余完成！输入序列: {input_count}, 输出序列: {output_count}, 冗余度: {reduction:.1f}%")
        
        return output_file
        
    except subprocess.CalledProcessError as e:
        logger.error(f"CD-HIT执行失败, 状态码: {e.returncode}")
        logger.error(f"错误信息:\n{e.stderr}")
        raise RuntimeError(f"CD-HIT执行失败: {e.stderr}")
    

def check_fasta_duplicates_simple(fasta_file):
    """
    简单检查FASTA文件中的重复ID和重复序列
    
    参数:
        fasta_file: FASTA文件路径
        
    返回:
        (duplicate_ids, duplicate_seqs)
        duplicate_ids: 重复ID的列表 [id1, id2, ...]
        duplicate_seqs: 字典 {序列: [包含该序列的ID列表]}
    """
    ids = set()
    duplicate_ids = []
    sequences = {}
    
    current_id = None
    current_seq = []
    
    with open(fasta_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                # 处理前一个序列
                if current_id:
                    seq_str = ''.join(current_seq).upper().replace(' ', '')
                    sequences.setdefault(seq_str, []).append(current_id)
                
                # 获取新ID
                header = line[1:]
                new_id = header.split()[0] if header.strip() else "unnamed"
                
                # 检查ID是否重复
                if new_id in ids:
                    duplicate_ids.append(new_id)
                else:
                    ids.add(new_id)
                
                current_id = new_id
                current_seq = []
            else:
                current_seq.append(line)
        
        # 处理最后一个序列
        if current_id:
            seq_str = ''.join(current_seq).upper().replace(' ', '')
            sequences.setdefault(seq_str, []).append(current_id)
    
    # 找出重复序列
    duplicate_seqs = {seq: ids for seq, ids in sequences.items() if len(ids) > 1}
    
    return duplicate_ids, duplicate_seqs



def stratified_split_fasta(input_fasta, train_output, val_output, trainval_output, test_output, 
                           random_seed=SEED, val_ratio=0.2, test_ratio=0.2, label_type=None):
    """
    按标签分层抽样，划分训练集、验证集和测试集。
    
    参数:
    input_fasta: 输入FASTA文件路径
    train_output: 训练集输出文件路径
    val_output: 验证集输出文件路径
    trainval_output: 训练和验证集（测试集剩余部分）输出文件路径
    test_output: 测试集输出文件路径
    random_seed: 随机种子，确保结果可重现
    val_ratio: 验证集比例（整个数据集），默认为0.2
    test_ratio: 测试集比例（整个数据集），默认为0.2
    label_type: 标签类型，'wb' 或 'elisa'，用于从描述中提取标签
    
    返回: 包含数量统计的字典
    """
    # 设置随机种子以确保结果可重现
    random.seed(random_seed)
    np.random.seed(random_seed)

    # 使用Biopython读取FASTA文件并按标签分类
    mrna_sequences = defaultdict(list)
    
    for record in SeqIO.parse(input_fasta, "fasta"):
        label = 'unknown'
        # 提取标签（如 mRNA-033|0|1）第一个是wb标签，第二个是elisa标签
        if '|' in record.description and label_type is not None:
            parts = record.description.split('|')
            if label_type == 'wb':
                label = parts[1].strip() if len(parts) >= 2 else 'unknown'
                print(record.description, label)
            elif label_type == 'elisa':
                label = parts[2].strip() if len(parts) >= 3 else 'unknown'
                print(record.description, label)
            
        # 存储序列ID和序列内容
        mrna_sequences[label].append((record.description, str(record.seq)))

    # 统计各类RNA的数量
    print("各标签的数量统计:")
    for rna_label, sequences in mrna_sequences.items():
        print(f"{rna_label}: {len(sequences)}")

    # 按类别分层抽样，划分训练集和测试集
    trainval_seqs = []
    train_seqs = []
    val_seqs = []
    test_seqs = []
    trainval_counts = defaultdict(int)
    train_counts = defaultdict(int)
    val_counts = defaultdict(int)
    test_counts = defaultdict(int)
    
    for rna_label, sequences in mrna_sequences.items():
        # 随机打乱当前类别的序列
        random.shuffle(sequences)
        
        # 计算验证集和测试集大小
        test_size = max(1, int(len(sequences) * test_ratio))  # 确保每类至少有一个测试样本
        val_size = max(1, int(len(sequences) * val_ratio))  # 确保每类至少有一个测试样本
        
        # 划分训练集，验证集和测试集
        test_set = sequences[:test_size]
        trainval_set = sequences[test_size:]

        val_set = trainval_set[:val_size]
        train_set = trainval_set[val_size:]

        # 添加到总集合
        test_seqs.extend(test_set)
        trainval_seqs.extend(trainval_set)
        val_seqs.extend(val_set)
        train_seqs.extend(train_set)
        
        # 记录数量
        test_counts[rna_label] = len(test_set)
        trainval_counts[rna_label] = len(trainval_set)
        val_counts[rna_label] = len(val_set)
        train_counts[rna_label] = len(train_set)

    # 全局打乱顺序
    random.shuffle(train_seqs)
    random.shuffle(val_seqs)
    random.shuffle(trainval_seqs)
    random.shuffle(test_seqs)
    
    # 手动写入训练集文件，确保不添加额外换行符
    with open(train_output, 'w') as f:
        for seq_id, sequence in train_seqs:
            f.write(f">{seq_id}\n")
            f.write(f"{sequence}\n")  # 序列写入为单行，不添加额外换行符
    
    # 手动写入验证集文件，确保不添加额外换行符
    with open(val_output, 'w') as f:
        for seq_id, sequence in val_seqs:
            f.write(f">{seq_id}\n")
            f.write(f"{sequence}\n")  # 序列写入为单行，不添加额外换行符
    
    # 手动写入训练验证集文件，确保不添加额外换行符
    with open(trainval_output, 'w') as f:
        for seq_id, sequence in trainval_seqs:
            f.write(f">{seq_id}\n")
            f.write(f"{sequence}\n")  # 序列写入为单行，不添加额外换行符
    
    # 手动写入测试集文件，确保不添加额外换行符
    with open(test_output, 'w') as f:
        for seq_id, sequence in test_seqs:
            f.write(f">{seq_id}\n")
            f.write(f"{sequence}\n")  # 序列写入为单行，不添加额外换行符
    
    # 报告训练集和测试集中各类RNA的数量
    print(f"\n训练集，验证和测试集划分比例: {1-(val_ratio + test_ratio)}: {val_ratio}: {test_ratio}")
    print("\n训练集中各标签的数量:")
    for rna_label, count in train_counts.items():
        print(f"{rna_label}: {count}")
    
    print("\n验证集中各标签的数量:")
    for rna_label, count in val_counts.items():
        print(f"{rna_label}: {count}")

    print("\n训练验证集中各标签的数量:")
    for rna_label, count in trainval_counts.items():
        print(f"{rna_label}: {count}")
    
    print("\n测试集中各标签的数量:")
    for rna_label, count in test_counts.items():
        print(f"{rna_label}: {count}")
    
    # 返回统计信息，便于进一步处理
    return {
        'total_counts': {k: len(v) for k, v in mrna_sequences.items()},
        'train_counts': dict(train_counts),
        'val_counts': dict(val_counts),
        'trainval_counts': dict(trainval_counts),
        'test_counts': dict(test_counts)
    }


if __name__ == "__main__":
    # 读取 Excel 文件
    file_path = '/data/wangdw/YKYY025AI/data_finetune/筛选池序列20251010版有更新序列104条.xlsx'
    df = pd.read_excel(file_path)
    # 显示前5行数据
    # print(df)

    # 删除野生序列行
    # print(len(df))
    df = df[df["ID"] != "mRNA-045"]
    print(len(df))

    # 检查异常字符s
    check_sequence_characters(df)

    # 生成序列长度和wb, elisa标签列，重新生成带标签的id
    # result_df = process_dataframe(df, wb_thrd=1, elisa_thrd=500)
    # Label_wb 正负样本绝对数量比例:
    # 正样本(1):负样本(0) = 260 : 240
    # Label_elisa 正负样本绝对数量比例:
    # 正样本(1):负样本(0) = 268 : 232
    # result_df.to_csv('data_finetune/preprocessed_finetune_data.csv', index=False)
    # print(result_df.head())

    # ============ 9.5 新增：过滤的标签阈值附近的样本 ============
    # print(len(result_df))
    # result_df = result_df[(result_df['Exp_value(WB)'] >= 1.1) | (result_df['Exp_value(WB)'] <= 0.9) & (result_df['Exp_value(ELISA)'] >= 600) | (result_df['Exp_value(ELISA)'] <= 400)]
    # result_df = result_df[
    # ( (result_df['Exp_value(WB)'] >= 1.1) | (result_df['Exp_value(WB)'] <= 0.9) ) &
    # ( (result_df['Exp_value(ELISA)'] >= 550) | (result_df['Exp_value(ELISA)'] <= 450))
    # ]
    # print(len(result_df))

    # =========================================================
    # 转换为fasta文件并写出
    output_file_path = 'inference_data.fasta'
    # write_fasta(result_df, output_file_path)
    write_fasta(df, output_file_path)

    # 尝试全局去冗余
    output_dir = 'remove_redundancy'
    os.makedirs(output_dir, exist_ok=True)
    start_time = time.time() 
    cdhit_output = run_cdhit_est(
        input_fasta = output_file_path,   
        output_prefix = os.path.join(output_dir, "RNA_central_merged_sequences_filtered_cut_cdhit"),
        c = 1.0,
        n = 5, 
        M = 64000, 
        T = 32, 
        G = 1
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    time_delta = timedelta(seconds=elapsed_time)
    print(f"✅ CD-HIT-EST去冗余成功: {cdhit_output}，耗时 {time_delta}")
    # INFO:CD-HIT:执行CD-HIT命令: cd-hit-est -i data_finetune/preprocessed_finetune_data.fasta -o data_finetune/remove_redundancy/RNA_central_merged_sequences_filtered_cut_cdhit.fasta -c 1 -n 5 -M 64000 -T 32 -G 1 -d 0 -g 1
    # INFO:CD-HIT:去冗余完成！输入序列: 100, 输出序列: 100, 冗余度: 0.0%
    # ✅ CD-HIT-EST去冗余成功: data_finetune/remove_redundancy/RNA_central_merged_sequences_filtered_cut_cdhit.fasta，耗时 0:00:00.184247

    # ==========================
    # 检查重复序列和重复标签
    # fasta_file =  output_file_path
    # dup_ids, dup_seqs = check_fasta_duplicates_simple(fasta_file)
    # if dup_ids:
    #     print(f"重复ID: {', '.join(dup_ids)}")
    # else:
    #     print("没有重复ID")
    # if dup_seqs:
    #     print("\n重复序列:")
    #     for seq, ids in dup_seqs.items():
    #         print(f"序列长度: {len(seq)}bp, 出现在ID: {', '.join(ids)}")
    # else:
    #     print("\n没有重复序列")
    # sys.exit(1)

    # #====== WB data split ======
    # # 划分训练集和测试集
    # cdhit_out_dir = "data_finetune/split_data_by_WB_500/"
    # os.makedirs(cdhit_out_dir, exist_ok=True)
    # stats = stratified_split_fasta(
    #     # input_fasta='data_finetune/remove_redundancy/RNA_central_merged_sequences_filtered_cut_cdhit.fasta',
    #     input_fasta='/data/wangdw/YKYY025AI/data_finetune/alignment_501/aligned_seqs/mafft_aligned_auto_withoutWT.fasta',
    #     train_output='data_finetune/split_data_by_WB_500/WB_finetune_train_set.fasta',
    #     val_output='data_finetune/split_data_by_WB_500/WB_finetune_val_set.fasta',
    #     trainval_output='data_finetune/split_data_by_WB_500/WB_finetune_trainval_set.fasta',
    #     test_output='data_finetune/split_data_by_WB_500/WB_finetune_test_set.fasta',
    #     random_seed=SEED,
    #     val_ratio=0.2,  
    #     test_ratio=0.2,  
    #     label_type='wb'
    # )
    # # 各标签的数量统计:
    # # 0: 49
    # # 1: 51

    # # 训练集，验证和测试集划分比例: 0.6: 0.2: 0.2

    # # 训练集中各标签的数量:
    # # 0: 31
    # # 1: 31

    # # 验证集中各标签的数量:
    # # 0: 9
    # # 1: 10

    # # 训练验证集中各标签的数量:
    # # 0: 40
    # # 1: 41

    # # 测试集中各标签的数量:
    # # 0: 9
    # # 1: 10

    # #====== ELISA data split ======
    # print('==============')
    # cdhit_out_dir = "data_finetune/split_data_by_ELISA_500/"
    # os.makedirs(cdhit_out_dir, exist_ok=True)
    # stats = stratified_split_fasta(
    #     # input_fasta='data_finetune/remove_redundancy/RNA_central_merged_sequences_filtered_cut_cdhit.fasta',
    #     input_fasta='/data/wangdw/YKYY025AI/data_finetune/alignment_501/aligned_seqs/mafft_aligned_auto_withoutWT.fasta',
    #     train_output='data_finetune/split_data_by_ELISA_500/ELISA_finetune_train_set.fasta',
    #     val_output='data_finetune/split_data_by_ELISA_500/ELISA_finetune_val_set.fasta',
    #     trainval_output='data_finetune/split_data_by_ELISA_500/ELISA_finetune_trainval_set.fasta',
    #     test_output='data_finetune/split_data_by_ELISA_500/ELISA_finetune_test_set.fasta',
    #     random_seed=SEED,
    #     val_ratio=0.2,  
    #     test_ratio=0.2,  # 20%作为测试集
    #     label_type='elisa'
    # )
    # # 各标签的数量统计:
    # # 0: 46
    # # 1: 54

    # # 训练集，验证和测试集划分比例: 0.6: 0.2: 0.2

    # # 训练集中各标签的数量:
    # # 0: 28
    # # 1: 34

    # # 验证集中各标签的数量:
    # # 0: 9
    # # 1: 10

    # # 训练验证集中各标签的数量:
    # # 0: 37
    # # 1: 44

    # # 测试集中各标签的数量:
    # # 0: 9
    # # 1: 10

