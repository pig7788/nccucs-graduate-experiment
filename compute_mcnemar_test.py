# -*- coding: UTF-8 -*-
import sys


def read_and_sort(data_path):
    data = list()
    with open(data_path, 'r') as reader:
        for line in reader:
            splitted_line = line.strip().split('\t')
            ground_truth = splitted_line[0]
            prediction = splitted_line[-1]
            data.append((ground_truth, prediction))

    data.sort(key=lambda item: item[0])

    return data


def transform_to_binary(data):
    binary_records = list()

    for ground_truth, prediction in data:
        binary_records.append(ground_truth == prediction)

    return binary_records


def transform_to_contingency_table(first_binary_data, second_binary_data):
    contingency_table = [[0, 0], [0, 0]]
    for first_item, second_item in zip(first_binary_data, second_binary_data):
        if first_item and second_item:
            contingency_table[0][0] += 1
        elif first_item and not second_item:
            contingency_table[0][1] += 1
        elif not first_item and second_item:
            contingency_table[1][0] += 1
        elif not first_item and not second_item:
            contingency_table[1][1] += 1

    return contingency_table


def compute_mcnemar_test(contingency_table):
    true_false = contingency_table[0][1]
    false_true = contingency_table[1][0]

    try :
        return ((abs(true_false - false_true) - 1) ** 2) / (true_false + false_true)
    except ZeroDivisionError:
        return 0


if __name__ == '__main__':
    first_path = sys.argv[1]
    second_path = sys.argv[2]

    first_data = read_and_sort(first_path)
    second_data = read_and_sort(second_path)

    first_binary_data = transform_to_binary(first_data)
    second_binary_data = transform_to_binary(second_data)

    contingency_table = transform_to_contingency_table(
        first_binary_data, second_binary_data)
    statistic = compute_mcnemar_test(contingency_table)

    print('contingency_table:', contingency_table)
    print('statistic:', '{:.4f}'.format(statistic))
