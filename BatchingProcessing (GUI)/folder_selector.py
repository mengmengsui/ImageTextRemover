# 用于独立进程中显示文件夹选择对话框（修复编码和空值问题）
import tkinter as tk
from tkinter import filedialog

def select_folder():
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口
    folder_path = filedialog.askdirectory(title="选择待处理图片文件夹")
    root.destroy()
    return folder_path if folder_path else ""  # 未选择时返回空字符串而非None

if __name__ == '__main__':
    print(select_folder())  # 确保输出无多余空格