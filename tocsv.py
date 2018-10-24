import csv

result_file = open(r"D:\python project\data\dis.csv", "w", newline="")
writer = csv.writer(result_file)
with open(r"D:\python project\data\dis.txt", "r") as f:
    for item in f:
        f_list = item.strip().split(" ")
        k = f_list[2].strip(",")
        acc = f_list[5]
        writer.writerow([int(k), float(acc)])
result_file.close()