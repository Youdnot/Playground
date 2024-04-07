import pypdf
 
wk_in_file_name = 'test.pdf'
writer = pypdf.PdfWriter()
input1 = open(wk_in_file_name, "rb")
writer.append(input1)
 
f = open('dir.txt', 'r', encoding='utf8')
lines = f.readlines()  # 读取所有行
num_lines = len(lines)  # 标题的总个数
 
txt = []
for line in lines:
    line = line.strip()  # 去掉末尾的'\n'
    pline = line.split(' ')  # 根据line中' '进行分割
    level = line.count('.')  # 有n个'.'就是n+1级标题
 
    if level == 0:
        bookmark_parent_0 = writer.add_outline_item(title=pline[0] + pline[1], page_number=int(pline[-1]), parent=None)
    elif level == 1:
        bookmark_parent_1 = writer.add_outline_item(title=pline[0] + pline[1], page_number=int(pline[-1]),
                                                    parent=bookmark_parent_0)
    else:
        writer.add_outline_item(title=pline[0] + pline[1], page_number=int(pline[-1]), parent=bookmark_parent_1)
 
 
# Write to an output PDF document
output = open('04_'+wk_in_file_name, "wb")
writer.write(output)
 
# Close File Descriptors
writer.close()
output.close()
 
f.close()  # 关闭文件
print('f.closed=', f.closed)