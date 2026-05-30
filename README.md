# Face Recognition

一个轻量级 Python 人脸注册与 1:N 识别项目，基于 InsightFace `buffalo_l` 模型，支持人员注册、相似度识别、人员列表、信息修改和删除。项目使用本地文件系统保存人员档案，适合门禁原型、机器人访客识别、实验室 Demo 和小型离线人脸库验证。

## 特性

- 人脸注册：从单人照片中提取 512 维人脸特征并保存档案
- 1:N 识别：将待识别人脸与本地人员库做余弦相似度匹配
- 本地存储：每个人一个目录，包含 `meta.json`、`embedding.npy`、`photo.jpg`
- 人员管理：支持列出、修改、删除已注册人员
- 自动裁脸：注册时额外保存一张基于检测框裁剪的人脸图 `face.jpg`
- GPU 优先：默认使用 ONNXRuntime CUDA Provider，CPU 作为备用
- 隐私友好：业务数据默认写入本地 `business_data/`，已在 `.gitignore` 中忽略

## 项目结构

```text
face_recognition/
├── face/
│   ├── __init__.py       # 对外导出 API
│   ├── face_model.py     # InsightFace 模型封装
│   ├── register.py       # 注册逻辑
│   ├── recognize.py      # 识别逻辑
│   ├── orientation.py    # 自动尝试 0/180/90/270 度检测
│   ├── storage.py        # 本地人员库读写
│   ├── manage.py         # 列表、修改、删除便捷函数
│   └── capture.py        # 采集接口预留
├── scripts/
│   └── backfill_face_crops.py  # 为旧数据补生成 face.jpg
├── requirements.txt
└── README.md
```

运行后会自动生成：

```text
business_data/
├── state.json
└── owners/
    └── 000001/
        ├── meta.json
        ├── embedding.npy
        ├── photo.jpg
        └── face.jpg
```

## 环境要求

- Python 3.10+
- Linux / WSL 推荐
- NVIDIA GPU 可选，但推荐用于更快推理
- 已正确安装 CUDA 相关运行库时，可使用 `onnxruntime-gpu`

安装依赖：

```bash
pip install -r requirements.txt
```

如果你使用 conda：

```bash
conda create -n face python=3.12
conda activate face
pip install -r requirements.txt
```

首次运行时，InsightFace 会在用户目录下下载或读取模型文件，例如：

```text
~/.insightface/models/buffalo_l/
```

## 快速开始

### 注册人员

```python
from face import register

result = register(
    "person.jpg",
    name="张三",
    room="301",
    phone="13800000000",
)

print(result)
```

注册成功后会返回人员信息、人脸框和当前库大小：

```python
{
    "owner_id": 1,
    "name": "张三",
    "room": "301",
    "phone": "13800000000",
    "registered_at": "2026-05-30T18:44:08",
    "face_bbox": [...],
    "det_score": 0.91,
    "rotation_degrees": 0,
    "gallery_size": 1,
}
```

### 识别人脸

```python
from face import recognize

result = recognize("person.jpg")
print(result["decision"])

if result["decision"] == "matched":
    print(result["owner"]["name"])
    print(result["similarity"])
```

识别结果的 `decision` 可能是：

| decision | 含义 |
| --- | --- |
| `matched` | 匹配到已注册人员 |
| `stranger` | 检测到人脸，但相似度低于阈值 |
| `no_face` | 未检测到人脸 |
| `empty_gallery` | 人员库为空 |
| `decode_error` | 图片无法解码 |

默认识别阈值是 `0.30`，可以自行传入：

```python
result = recognize("person.jpg", threshold=0.45)
```

## 人员管理

### 列出所有人员

```python
from face import list_owners

owners = list_owners()
print(owners)
```

### 修改人员信息

```python
from face import update_owner

owner = update_owner(1, name="李四")
print(owner)

owner = update_owner(1, room="502", phone="13900000000")
print(owner)
```

传入 `None` 可以清空字段：

```python
update_owner(1, phone=None)
```

### 删除人员

```python
from face import delete_owner

deleted = delete_owner(1)
print(deleted)
```

删除会移除对应目录：

```text
business_data/owners/000001/
```

`state.json` 中的自增 ID 不会回退，删除后的 ID 不会复用。

## 多人脸策略

注册时要求图片中只能有一张人脸：

- 0 张脸：抛出 `NoFaceError`
- 1 张脸：正常注册
- 多张脸：抛出 `MultipleFacesError`

这是为了避免把错误的人脸注册到某个姓名下。

识别时允许图片中有多张人脸，但当前只识别面积最大的一张：

- 0 张脸：返回 `decision = "no_face"`
- 1 张脸：识别这一张
- 多张脸：选择最大人脸进行 1:N 匹配

如果需要识别合照中的所有人，可以在现有 `recognize.py` 基础上扩展 `recognize_all()`。

注册和识别都会自动尝试常见直角旋转：

```text
0° → 180° → 90° → 270°
```

如果原图方向检测不到人脸，会继续尝试其他方向。返回结果中的 `rotation_degrees` 表示最终使用的旋转角度，例如倒置照片通常会返回 `180`。

## 异常处理

注册时可能抛出的业务异常：

```python
from face import register
from face.register import (
    DecodeError,
    NoFaceError,
    MultipleFacesError,
    FaceTooSmallError,
    DuplicateOwnerError,
)

try:
    result = register("person.jpg", name="张三")
    print(result)
except MultipleFacesError as e:
    print(f"检测到 {e.n} 张人脸，请使用单人照片")
except DuplicateOwnerError as e:
    print(f"疑似重复注册，已有 owner_id={e.owner_id}, sim={e.similarity:.3f}")
```

## 存储说明

每个注册人员都会保存为一个独立目录：

```text
business_data/owners/000001/
├── meta.json       # 姓名、房间、电话、注册时间
├── embedding.npy   # 512 维人脸特征
├── photo.jpg       # 注册原图
└── face.jpg        # 根据注册时的人脸框裁剪出的人脸图
```

`Storage()` 初始化时会扫描 `business_data/owners/`：

- 读取 `meta.json` 到内存中的 `self._meta`
- 读取 `embedding.npy` 到内存中的 `self.feats`
- 保存人员 ID 到内存中的 `self.ids`
- 不会把 `photo.jpg` 或 `face.jpg` 图片内容加载到内存

如果你在旧版本中已经注册过人员，可以运行下面的脚本为已有人员补生成 `face.jpg`：

```bash
python scripts/backfill_face_crops.py
```

## 性能建议

如果连续注册或识别多张图片，建议复用同一个模型和存储实例，避免重复加载模型：

```python
from face import FaceModel, Storage, register, recognize

model = FaceModel()
storage = Storage()

register("a.jpg", name="张三", model=model, storage=storage)
register("b.jpg", name="李四", model=model, storage=storage)

result = recognize("query.jpg", model=model, storage=storage)
print(result)
```

## Git 提交建议

项目已经默认忽略以下本地数据：

- `business_data/`
- `*.jpg` / `*.png` / `*.npy`
- `__pycache__/`
- `test_*.py`
- IDE 和本地工具配置

上传到 GitHub 前，请确认不要提交真实人脸照片、人员姓名、手机号等敏感信息。

## 当前状态

- `capture()` 是预留接口，暂未接入摄像头或机器人视频流
- 当前识别逻辑只返回最大人脸的匹配结果
- 本项目适合作为本地原型或二次开发基础，生产使用前建议增加权限控制、审计日志、数据加密和更完整的测试

## License

如果你计划开源发布，建议补充一个明确的许可证，例如 MIT、Apache-2.0 或仅限内部使用的私有声明。
