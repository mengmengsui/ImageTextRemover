**说明：标*号的为当前最新版本**

[*SelectionBoxDeletion：框选可删](#selectionboxdeletion)</br>
[SelectionBox：框选](#selectionbox)</br>
[*BatchProcessing (GUI)：批处理界面操作](#batchprocessing-gui)</br>
[*BatchProcessing V1：批处理无界面](#batchprocessing-v1)</br>
[SingleProcessing：单个处理](#singleprocessing)</br>




# *SelectionBoxDeletion
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
1. 下载lama_fp32.onnx大模型，并与Python和HTML文件放在同一文件夹下。
2. 建议创建虚拟环境运行程序。
3. 在虚拟环境中，安装必要的Python依赖包（pip install）。
  - flask
  - cv2
  - torch
  - onnxruntime
  - 其他
4. Python解释器修改为虚拟环境。
5. 调试并运行程序，打开页面即可（127.0.0.1:5000）。




****

# SelectionBox
SelectionBoxDeletion的前序版本，仅能增加框选区域和删除全部框选区域，不能逐个删除。
</br>

**文件说明**
- backend.py：后端主程序
- core.py：核心程序，调用lama大模型处理图片
- index.html：前端界面
</br>

**使用说明**
1. 运行backend.py，打开网页。
2. 选择图片，页面将展示所选图片。
3. 在图片上框选要去除的文字。
4. 单击“开始去除文字”，开始处理。处理完成的图片将显示在下方。
5. 单击“清除所有区域”，可清除图片上所有的框选。
6. 单击“清空所有”，将清空当前任务，初始化程序。
7. 图片处理完成后，单击“下载”按钮，可下载处理后的图片。


****

# *BatchProcessing (GUI)
批处理当前最新版本。支持：
- 在网页上选择待处理的图片批量存储的文件夹
- 显示每张图片的处理进度和结果
- 显示所有图片的处理进度和结果统计
- 所有图片处理完成后，支持重置任务
</br>

**文件说明**
- multiple.py：后端主程序，运行自此开始
- multiple_core.py：核心程序，调用V1版本批处理程序处理图片
- example_multiple.py：V1版本批处理程序
- folder_selector.py：文件夹选择辅助代码
- /templates/multiple_index.html：前端页面
</br>

**使用说明**
1. 运行multiple.py，打开网页。
2. 单击“选择待处理文件夹”，选择待处理图片批量存放的文件夹。
3. 选择文件夹后，即开始处理图片。界面上实时展示当前处理的图片进度和结果，以及所有图片数量、正在处理数量、成功数量和失败数量的统计。
4. 所有图片处理完成后，单击“重置”按钮，可重置任务，程序和界面均初始化。

**注意**：当前“重置”按钮，仅支持所有图片处理完成后重置（任务初始化），不支持处理过程中的中断任务。


****

# *BatchProcessing V1
图片批处理早期版本，支持将一个文件夹中的图片批量删除文字并填充颜色。
</br>

**文件说明**
- example_multiple.py：唯一程序，直接运行即可
</br>

**使用说明**
1. 在Python程序同一文件夹下，创建imgs文件夹，并在该文件夹下创建一个子文件夹，然后将所有待处理的图片放入该子文件夹中。例如，子文件夹名为input2，则文件层级为：
  - example_multiple.py
  - imgs
    - inputs
      - 所有待处理的图片
2. 将main函数中的input_folder = "imgs/input2" ，其中的input2，修改为imgs中子文件夹的名字。
3. 直接运行程序，等待处理完成即可。


****

# SingleProcessing
单个图片处理版本，无前端，只能通过改代码中图片路径的形式处理。该版本属预研阶段的版本，实际已无用处，仅作存档。
</br>

**文件说明**
- example_single：唯一程序，直接运行就可以
</br>

**使用说明**
1. 下载大模型，安装相关依赖（略）。
2. 在Python程序统一文件夹下，创建名为imgs的子文件夹，并在其中创建一个与待处理图片同名的子文件夹，将图片放入该子文件夹中；同时在该子文件夹中，再创建一个名为cache的空文件夹。例如：假设待处理的图片名为t4.png，则文件层级如下：
  - example_single.py
  - imgs
    - t4
      - t4.png
      - cache（空文件夹）
2. 将main函数中的target = "t4"，修改为对应文件夹及图片的名称。
3. 运行程序，等待处理完成即可。