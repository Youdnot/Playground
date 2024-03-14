import pypdf
import sys

# 读取输入文档
filename = input("Please input the PDF file name:")

# 识别粘贴文本是否带有拓展名并处理为纯标题
if ".pdf" in filename:
    filename = filename[:-4]
else:
    filename = filename
format = ".pdf"
full_filename = f"{filename}{format}"
input = open(full_filename, "rb")



# 初始化writer
writer = pypdf.PdfWriter()
writer.append(input)

# 搜索标题
reader = pypdf.PdfReader()
number_of_pages = len(reader.pages)
page = reader.pages[0]
text = page.extract_text()

# 添加书签
writer.add_outline_item(title="Test", page_number=5, parent=None)

# 写入输出文档
output = open(f"{file}_addoutline{format}", "wb")
writer.write(output)

# 关闭文件
writer.close()
output.close()

print('pypdf.__version__:', pypdf.__version__)
print('sys.version:', sys.version)

pass