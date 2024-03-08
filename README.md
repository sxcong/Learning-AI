# Learning-AI
for beginner

about deep learning,artificial intelligence,machine learning. tensorflow and pytorch code. python and c++

1 

(PyTorch）神经网络模型快速上手——FashionMNIST 

https://blog.csdn.net/weixin_44263674/article/details/125559389

2 

基于tensorflow2.3.0的模型保存与恢复（以Fashion MNIST为数据集）

https://blog.csdn.net/wchwdog13/article/details/110441818

3 

Deep Learning Papers Reading Roadmap

https://github.com/floodsung/Deep-Learning-Papers-Reading-Roadmap

4

converts a trained keras model into a ready-for-inference TensorFlow model(将.h5文件转化为.pb文件)

https://github.com/amir-abdi/keras_to_tensorflow

5 

c++ tensorflow

https://blog.csdn.net/qq_41754894/article/details/93888450

https://blog.csdn.net/qq_37541097/article/details/90257985?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_baidulandingword~default-0-90257985-blog-93888450.235^v43^pc_blog_bottom_relevance_base3&spm=1001.2101.3001.4242.1&utm_relevant_index=3

https://blog.csdn.net/weixin_30687587/article/details/98936841?spm=1001.2101.3001.6650.1&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-1-98936841-blog-90257985.235%5Ev43%5Epc_blog_bottom_relevance_base3&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-1-98936841-blog-90257985.235%5Ev43%5Epc_blog_bottom_relevance_base3&utm_relevant_index=2

tensorflow::Tensor input_tensor(DT_FLOAT, TensorShape({ 1, height, width, 3 })); 

Session* session;

Status status = NewSession(SessionOptions(), &session);

GraphDef graph_def;

status = ReadBinaryProto(Env::Default(), model_path, &graph_def);

status = session->Create(graph_def);

std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = { { "Placeholder",input_tensor } };

std::vector<tensorflow::Tensor> outputs;

status = session->Run(inputs, { "Conv2D","side_3/conv2d_transpose","side_4/conv2d_transpose","side_5/conv2d_transpose" }, {}, &outputs);




