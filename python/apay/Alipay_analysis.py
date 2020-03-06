# coding=utf-8

import csv

RAW_RECORD = '1YearPay.csv'  ## 填写你下载的文件名称，需要包括 .csv 后缀
CLEANED_RECORD = RAW_RECORD.split('.')[0] + '_cleaned.csv'

def clean_csv(input, output):
    with open(input, encoding='gbk') as f, open(output, 'w') as o:
        lines = f.readlines() ## 注意这行代码会一次性读取整个文件，若文件非常大时不推荐使用
        #lines = lines[4:-7] ## 去掉开头结尾非 CSV 的内容
        lines = [line.replace(' ', '') for line in lines] ## 去掉空格
        lines = [line.replace('\t','') for line in lines]
        o.writelines(lines)

def analyze_income_expense(record):
    """统计支出和收入的总和
    """
    income = 0
    pay = 0
    with open(record) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            try:
                temp = float(row['收入'])
                income += temp
            except ValueError:
                pass
            try:
                temp = float(row['支出'])
                pay -= temp
            except ValueError:
                pass
    print(f"总收入为 {income:.2f} 元，总支出为 {pay:.2f} 元")


def analyze_max_expense_item(record):
    """统计花费最高的事项
    """
    max_pay = 0
    max_record = ""
    with open(record) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            try:
                temp = float(row['支出'])
                if -temp > max_pay:
                    max_pay = -temp
                    max_record = row
            except ValueError:
                pass
    print(f"单笔最高支出 {max_pay:.2f}元")


def analyze_max_expense_date(record):
    """统计花费最多的是哪一天
    """
    date_expense = {}
    with open(record) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            try:
                date = row['时间'][:10]
                temp = float(row['支出'])
                if date in date_expense:
                    date_expense[date] -= temp
                else:
                    date_expense[date] = -temp
            except ValueError:
                pass
    max_expense = 0
    max_date = ""
    for key,value in date_expense.items():
        if value > max_expense:
            max_expense = value
            max_date = key
    print(f"单天最高支出 {max_expense:.2f} 日期为 {max_date}")


if __name__ == "__main__":
    clean_csv(RAW_RECORD, CLEANED_RECORD)
    analyze_income_expense(CLEANED_RECORD)
    analyze_max_expense_item(CLEANED_RECORD)
    analyze_max_expense_date(CLEANED_RECORD)