
#https://blog.csdn.net/weixin_44263674/article/details/125559389
# BP（反向传播）神经网络中的FashionMNIST（时装分类）模型
 
# 导入pyplot并命名为plt
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
 
# =================================================================
# =================================================================
# 一、
# 下载训练数据
print("一、下载训练、测试数据集\n=============================================")
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),  # 修改样本
)
 
# 下载测试数据
test_data = datasets.FashionMNIST(
    root="data",  # root存储训练/测试数据的路径
    train=False,  # 指定训练/测试数据集
    download=True,  # 如果数据在根目录下不可用，则从Internet下载
    transform=ToTensor(),  # 修改标签
)
# transform.ToTensor()作用是将原始的数据，格式化为张量类型
 
# 对于FashionMNIST数据集中的示例可视化展现
# labels_map = {
#     0: "T-Shirt",
#     1: "Trouser",
#     2: "Pullover",
#     3: "Dress",
#     4: "Coat",
#     5: "Sandal",
#     6: "Shirt",
#     7: "Sneaker",
#     8: "Bag",
#     9: "Ankle Boot",
# }
# figure = plt.figure(figsize=(8, 8))
# cols, rows = 3, 3
# for i in range(1, cols * rows + 1):
#     sample_idx = torch.randint(len(training_data), size=(1,)).item()
#     img, label = training_data[sample_idx]
#     figure.add_subplot(rows, cols, i)
#     plt.title(labels_map[label])
#     plt.axis("off")
#     plt.imshow(img.squeeze(), cmap="gray")
# plt.show()
 
# ======================================================================
# ======================================================================
# 二、使用dataloader为训练准备数据
# Data Loader 是一个迭代器，快速处理原数据并加载整合成batch用于后续训练。每次迭代都会返回train_features和train_labels（分别包含batch_size=64特征和标签）
print("二、使用dataloader为训练准备数据\n=============================================")
batch_size = 64
 
# Create data loaders。
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

# for X, y in test_dataloader:
#     print(f"Shape of X [N, C, H, W]:{X.shape}")
#     print(f"Shape of y:{y.shape} {y.dtype}")
#     break
 
 
# 遍历DataLoader。每次迭代都会返回一个特征和标签
# # Display image and label
# train_features, train_labels = next(iter(train_dataloader))
# print(f"Feature batch shape: {train_features.size()}")          # 特征的张量
# print(f"Labels batch shape: {train_labels.size()}")             # 类型的张量
# img = train_features[0].squeeze()
# label = train_labels[0]
# plt.imshow(img, cmap="gray")
# plt.show()
# print(f"Label: {label}")
 
 
# =================================================================
# =================================================================
# 三、创建模型
print("三、创建时装分类模型")
device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"Using {device} device")
 
 
# 定义自己的神经网络NeuralNetwork，继承自nn.Module,并用__init__初始化网络层
# define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

# nn.ReLU()激活函数。类似还有Sig moid函数（值域0-1，常用二分类）、Tanh函数（至于-1——1，缓解梯度消失问题）
 
# 每个神经网络子类都可在forward方法中实现输入数据的操作
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
 
"""  
model = NeuralNetwork().to(device)
print("模型如下：")
print(model)
 
# print("输入一个测试示例，并打印结果：")
# # 输入一个示例X，调用模型，返回一个10维张量，从中获得预测概率
# X = torch.rand(1, 28, 28, device=device)
# logits = model(X)
# pred_probab = nn.Softmax(dim=1)(logits)
# y_pred = pred_probab.argmax(1)
# # nn.Softmax将从神经网络层最后一个线性层返回的值，缩放到【0，1】，即表示模型对该输入示例，判定为每个类别的预测概率
# # pred_probab: tensor([[0.0526, 0.0035, 0.1659, 0.0145, 0.1206, 0.0494, 0.1321, 0.0178, 0.3762,0.0674]]。即表示X是tensor[0]类别的概率为0.0526，为tensor[1]类别的概率为0.0035
# # pred_probab.argmax（）函数，输出pred_probab中最大值所在的位置。
#
# # f及表示使用format函数，格式化字符串，使用后可在字符串中使用用花括号括起来的变量和表达式。等于”Predicted class: {}“.format（y_pred）
# print(f"预测类别: {y_pred}。一般{y_pred}即代表Bag类")
# print("=============================================")
 
# ===========================================================
# ===========================================================
# 四、优化模型参数
 
print("四、优化模型参数")
# 训练模型，需要损失函数loss_fn()和优化器optimizer
# 损失函数：衡量得到的结果与目标值的相异程度。训练目标使其最小
# 优化器： 在每个训练步骤中，调整模型参数以减少误差
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)  # 此处使用 SGD优化器，随机梯度下降
 
 
# 单训练循环中，分批输入，模型对训练集进行预测，并反向传播预测误差以调整模型参数
# 定义训练函数
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
 
        # Compute prediction error  计算预测误差
        pred = model(X)
        loss = loss_fn(pred, y)
 
        # Backpropagation   反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
 
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            # loss：记录100个batch累计的损失
            # current：当前是第几个batch*每个batch的图片数，得到当前已经训练了多张图片
            # size：数据集有多少张图片
            print(f"每100个batch的累计损失: {loss:>7f}  进度：[{current:>5d}/{size:>5d}]")
 
 
# 导入测试数据集，检查模型性能，确保模型在学习
# 定义测试函数
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"测试误差: \n 准确率: {(100 * correct):>0.1f}%, 平均损失: {test_loss:>8f} \n")
 

print("对模型进行迭代训练，并输出测试结果")
# 多次迭代
epochs = 5
for t in range(epochs):
    print(f"第 {t + 1} 次迭代\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("迭代完成")
print("=============================================")
# ===============================================================
# ===============================================================
# 五、保存模型
print("五、保存模型")
torch.save(model.state_dict(), "model_1.pth")
print("已保存模型到 model_1.pth")
print("=============================================")
"""
# ===============================================================
# ===============================================================
# 六、加载模型：重新创建模型结构，并将状态字典加载到其中
print("六、加载模型：重新创建模型结构，并将状态字典加载到其中")
model = NeuralNetwork()
model.load_state_dict(torch.load("model_1.pth"))
 
classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]
print("输入示例到模型，并打印预测结果")
model.eval()
x, y = test_data[9999][0], test_data[9999][1]
with torch.no_grad():
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'预测类别: "{predicted}", 实际类别: "{actual}"')
 
# 显示输入图片
# img, label = test_data[0]
    # plt.title('picture：Ankle boot')
    # plt.axis("off")
    # plt.imshow(img.squeeze(), cmap="gray")  # 画图
    # plt.show()                              # 显图