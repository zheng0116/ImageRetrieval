# DINOv2图像检索系统
 <strong>[中文](./README_zh.md) |
    English</strong>
## 🌟 概述
本项目是基于DINOv2模型的图像检索系统，允许用户上传图片，并从预定义的图像数据库中检索相似的图像。
    
## 🚀 特征

- 使用DINOv2模型进行图像特征提取
- 支持不同大小的DINOv2模型（小型、基础型、大型、巨型）
- 基于余弦相似度的图像检索
- 使用FastAPI构建的Web界面
- 对大规模图像数据库进行特征缓存

## 用户界面

![DINOv2图像检索系统界面](./images/image.png)

## 安装

1. 克隆仓库:

```bash
git clone https://github.com/zheng0116/ImageRetrieval.git
cd ImageRetrieval
```

2. 安装依赖项:

```bash
sh run.sh install
```
3. 下载DINOv2模型:
```bash
https://pan.baidu.com/s/1fBVgg_o8PTFEu_2vtLY25Q 
提取码: f9ww 
```

## 使用方法

1.准备您的图像数据库，将图片放置在`qurary`文件夹中（或者指定一个自定义文件夹）。

2. 运行应用程序:

```bash
sh run.sh start
```

3. 打开浏览器 `http://localhost:5999`。

4. 上传图片并点击“开始检索”以找到相似图像.

## 配置

运行应用程序时，您可以配置以下参数:

- `--model_path`: DINOv2模型的路径（默认：“./Dinov2_model/dinov2-small”）
- `--model_size`: DINOv2模型的大小（选择：small, base, large, giant；默认：small）
- `--database_folder`: 图像数据库文件夹的路径（默认：“./qurary”）


## 项目结构

- `main.py`：主应用程序文件，包含FastAPI服务器
- `model/Diniv2.py`：DINOv2模型加载器
- `utils/image_preprocessor.py`：图像预处理
- `utils/retrieval_processor.py`: 图像检索
- `static/index.html`：图像检索逻辑


## 配置需要

- Python 3.7+
- FastAPI
- Uvicorn
- PyTorch
- Transformers
- Pillow
- NumPy

## License

[MIT License](LICENSE)

## 致谢

- [DINOv2](https://github.com/facebookresearch/dinov2) by Facebook Research
- [FastAPI](https://fastapi.tiangolo.com/)