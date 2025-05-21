# SelectionBoxDeletion
目前最新最全的单个图片处理版本，支持：
- 上传一张图片，在页面上展示图片
- 在图片上框选要去除的文字区域
- 删除框选文字的选择框
- 调用Lama大模型，将文字去除，并填充适当的颜色
- 预览处理后的图片
- 下载处理后的图片
- 清空所选区域
- 清空当前所有
</br>

**文件说明**
- backend.py：后端主程序
- core.py：核心程序，lama大模型在此调用
- index.html：前端界面


**使用说明**
- 下载lama_fp32.onnx大模型，并与Python和HTML文件放在同一文件夹下
- 建议创建虚拟环境运行程序
- 在虚拟环境中，安装必要的Python依赖包（pip install）
  - flask
  - cv2
  - torch
  - onnxruntime
  - 其他
- Python解释器修改为虚拟环境
- 调试并运行程序，打开页面即可（127.0.0.1:5000）




****

# SelectionBoxOnly
SelectionBoxOnly的前序版本，仅能增加框选区域和删除全部框选区域，不能逐个删除

# BatchProcessing
图片批处理版本，支持：
- 


# SingleProcessing
单个图片处理版本，无前端，只能通过改代码中图片路径的形式处理。该版本属预研阶段的版本，实际已无用处，仅作存档。