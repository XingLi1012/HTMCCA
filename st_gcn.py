import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# from net.utils.tgcn import ConvTemporalGraphical
# from net.utils.graph import Graph


if(1==1):
    import numpy as np

    class Graph():
        """ The Graph to model the skeletons extracted by the openpose

        Args:
            strategy (string): must be one of the follow candidates
            - uniform: Uniform Labeling
            - distance: Distance Partitioning
            - spatial: Spatial Configuration
            For more information, please refer to the section 'Partition Strategies'
                in our paper (https://arxiv.org/abs/1801.07455).

            layout (string): must be one of the follow candidates
            - openpose: Is consists of 18 joints. For more information, please
                refer to https://github.com/CMU-Perceptual-Computing-Lab/openpose#output
            - ntu-rgb+d: Is consists of 25 joints. For more information, please
                refer to https://github.com/shahroudy/NTURGB-D

            max_hop (int): the maximal distance between two connected nodes
            dilation (int): controls the spacing between the kernel points

        """

        def __init__(self,
                     layout='openpose',
                     strategy='uniform',
                     max_hop=1,#连接点最大距离（确定领域范围）
                     dilation=1):
            self.max_hop = max_hop
            self.dilation = dilation

            self.get_edge(layout)
            self.hop_dis = get_hop_distance(
                self.num_node, self.edge, max_hop=max_hop)
            self.get_adjacency(strategy)

        def __str__(self):
            return self.A

        def get_edge(self, layout):#骨骼点连接对[(0, 0), (1, 1),....,(23, 24), (24, 11)]
            if layout == 'openpose':
                self.num_node = 18
                self_link = [(i, i) for i in range(self.num_node)]
                neighbor_link = [(4, 3), (3, 2), (7, 6), (6, 5), (13, 12), (12,
                                                                            11),
                                 (10, 9), (9, 8), (11, 5), (8, 2), (5, 1), (2, 1),
                                 (0, 1), (15, 0), (14, 0), (17, 15), (16, 14)]
                self.edge = self_link + neighbor_link
                self.center = 1#中间点，取脖子
            elif layout == 'ntu-rgb+d':
                self.num_node = 25
                self_link = [(i, i) for i in range(self.num_node)]
                # print(self_link)
                # print('1111111111')
                neighbor_1base = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21),
                                  (6, 5), (7, 6), (8, 7), (9, 21), (10, 9),
                                  (11, 10), (12, 11), (13, 1), (14, 13), (15, 14),
                                  (16, 15), (17, 1), (18, 17), (19, 18), (20, 19),
                                  (22, 23), (23, 8), (24, 25), (25, 12)]
                neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
                # print(neighbor_link)
                # print('1111111111')
                self.edge = self_link + neighbor_link
                # print(self.edge)
                self.center = 21 - 1##中间点，取肩膀中间点
            elif layout == 'ntu_edge':
                self.num_node = 24
                self_link = [(i, i) for i in range(self.num_node)]
                neighbor_1base = [(1, 2), (3, 2), (4, 3), (5, 2), (6, 5), (7, 6),
                                  (8, 7), (9, 2), (10, 9), (11, 10), (12, 11),
                                  (13, 1), (14, 13), (15, 14), (16, 15), (17, 1),
                                  (18, 17), (19, 18), (20, 19), (21, 22), (22, 8),
                                  (23, 24), (24, 12)]
                neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
                self.edge = self_link + neighbor_link
                self.center = 2
            elif layout == 'ex':
                self.num_node = 7
                self_link = [(i, i) for i in range(self.num_node)]
                # print(self_link)
                # print('1111111111')
                neighbor_1base = [(1, 3), (2, 3), (3, 4), (3, 5), (5, 6),
                                  (5, 7)]
                neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
                # print(neighbor_link)
                # print('1111111111')
                self.edge = self_link + neighbor_link
                # print(self.edge)
                self.center = 3 - 1    
            # elif layout=='customer settings'
            #     pass
            else:
                raise ValueError("Do Not Exist This Layout.")

        #计算邻接矩阵A
        def get_adjacency(self, strategy):
            valid_hop = range(0, self.max_hop + 1, self.dilation)  #range(start,stop,step)
            
            adjacency = np.zeros((self.num_node, self.num_node))
            # print(adjacency)
            for hop in valid_hop:
                adjacency[self.hop_dis == hop] = 1#adjacency=transfer_mat[1](有连接是1包括自相连，无连接是0)
                
            # print(adjacency)    
            normalize_adjacency = normalize_digraph(adjacency)# （adjacency）*（adjacency的对角矩阵Dl的逆矩阵Dn）即adjacency的每个元素除以所在列的列和，normalize_adjacency的每一列列和为1
            # print(normalize_adjacency)
            if strategy == 'uniform':#矩阵不变，只多一层[[[  ]]]
                A = np.zeros((1, self.num_node, self.num_node))
                A[0] = normalize_adjacency
                # print(A)
                self.A = A
            elif strategy == 'distance':
                A = np.zeros((len(valid_hop), self.num_node, self.num_node))
                # print(len(valid_hop))
                for i, hop in enumerate(valid_hop):#enumerate同时列出下标和内容
                    # print(i)
                    # print(hop)
                    A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis ==
                                                                    hop]
                # print(A)                                                    
                self.A = A
            elif strategy == 'spatial':
                A = []
                for hop in valid_hop:
                    a_root = np.zeros((self.num_node, self.num_node))
                    a_close = np.zeros((self.num_node, self.num_node))
                    a_further = np.zeros((self.num_node, self.num_node))
                    for i in range(self.num_node):
                        for j in range(self.num_node):
                            if self.hop_dis[j, i] == hop:
                                if self.hop_dis[j, self.center] == self.hop_dis[
                                        i, self.center]:
                                    a_root[j, i] = normalize_adjacency[j, i]
                                elif self.hop_dis[j, self.
                                                  center] > self.hop_dis[i, self.
                                                                         center]:
                                    a_close[j, i] = normalize_adjacency[j, i]
                                else:
                                    a_further[j, i] = normalize_adjacency[j, i]
                    if hop == 0:
                        A.append(a_root)
                    else:
                        A.append(a_root + a_close)
                        A.append(a_further)
                A = np.stack(A)
                # print(A)
                self.A = A
            else:
                raise ValueError("Do Not Exist This Strategy")

    # 此函数的返回值hop_dis就是图的邻接矩阵(相连是1,自己是0,不相连是inf正无穷)
    def get_hop_distance(num_node, edge, max_hop=1):
        A = np.zeros((num_node, num_node))#A邻接矩阵，1代表有连接。
        for i, j in edge:
            A[j, i] = 1
            A[i, j] = 1
        # print(A)
        # print('1111')
        # compute hop steps
        hop_dis = np.zeros((num_node, num_node)) + np.inf  # np.inf 表示一个无穷大的正数
        
        # np.linalg.matrix_power(A, d)求矩阵A的d幂次方,transfer_mat矩阵(I,A)是一个将A矩阵拼接max_hop+1次的矩阵
        #A的0次方是单位矩阵I，一次方是原矩阵
        transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]#可达矩阵，0次方自连接，1次方距离为1的连接，2次方距离为了的连接。
        # print(transfer_mat[1])
        # (np.stack(transfer_mat) > 0)矩阵中大于0的返回Ture,小于0的返回False,最终arrive_mat是一个布尔矩阵,大小与transfer_mat一样
        arrive_mat = (np.stack(transfer_mat) > 0)#
        # print(arrive_mat)
        # range(start,stop,step) step=-1表示倒着取
        for d in range(max_hop, -1, -1):
            # 将arrive_mat[d]矩阵中为True的对应于hop_dis[]位置的数设置为d
            hop_dis[arrive_mat[d]] = d
        # print(hop_dis)    
        return hop_dis

    # （A）*（A的对角矩阵Dl的逆矩阵Dn）即A的每个元素除以所在列的列和，AD的每一列列和为1
    def normalize_digraph(A):
        Dl = np.sum(A, 0) #A的列和,即A的度矩阵的对角序列
        # print(Dl)
        num_node = A.shape[0]
        Dn = np.zeros((num_node, num_node))
        for i in range(num_node):
            if Dl[i] > 0:
                Dn[i, i] = Dl[i]**(-1)#Dn是A的对角矩阵Dl的逆矩阵
        # print(Dn)
        AD = np.dot(A, Dn)
        # print(AD)
        return AD


    def normalize_undigraph(A):
        Dl = np.sum(A, 0)
        num_node = A.shape[0]
        Dn = np.zeros((num_node, num_node))
        for i in range(num_node):
            if Dl[i] > 0:
                Dn[i, i] = Dl[i]**(-0.5)
        DAD = np.dot(np.dot(Dn, A), Dn)
        return DAD
        
    # gg=Graph(layout='ex', strategy='spatial')#**graph_args 拆开字典
if(1==1):
    # The based unit of graph convolutional networks.

    import torch
    import torch.nn as nn

    class ConvTemporalGraphical(nn.Module):

        r"""The basic module for applying a graph convolution.

        Args:
            in_channels (int): Number of channels in the input sequence data
            out_channels (int): Number of channels produced by the convolution
            kernel_size (int): Size of the graph convolving kernel
            t_kernel_size (int): Size of the temporal convolving kernel
            t_stride (int, optional): Stride of the temporal convolution. Default: 1
            t_padding (int, optional): Temporal zero-padding added to both sides of
                the input. Default: 0
            t_dilation (int, optional): Spacing between temporal kernel elements.
                Default: 1
            bias (bool, optional): If ``True``, adds a learnable bias to the output.
                Default: ``True``

        Shape:
            - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
            - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
            - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
            - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format

            where
                :math:`N` is a batch size,
                :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
                :math:`T_{in}/T_{out}` is a length of input/output sequence,
                :math:`V` is the number of graph nodes. 
        """

        def __init__(self,
                     in_channels,
                     out_channels,
                     kernel_size,
                     t_kernel_size=1,
                     t_stride=1,
                     t_padding=0,
                     t_dilation=1,
                     bias=True):
            super().__init__()

            self.kernel_size = kernel_size
            self.conv = nn.Conv2d(
                in_channels,
                out_channels * kernel_size,
                kernel_size=(t_kernel_size, 1),
                padding=(t_padding, 0),
                stride=(t_stride, 1),
                dilation=(t_dilation, 1),
                bias=bias)


        # forward()函数完成图卷积操作,x由（64,3,300,18）变成（64,64,300,18）
        def forward(self, x, A):
            assert A.size(0) == self.kernel_size#A.size=（3,25,25）
            # print(x.shape)#([120, 3, 300, 25])
            x = self.conv(x)#([120, 192=输出通道数64*3, 300, 25])
            # print(x.shape)
            n, kc, t, v = x.size()
            # (64,3,64,300,18)
            x = x.view(n, self.kernel_size, kc//self.kernel_size, t, v)#([120, 3, 64, 300, 25])
            # (64,64,300,18)
            # 此处的k消失的原因：在k维度上进行了求和操作,也即是x在邻接矩阵A的3个不同的子集上进行乘机操作再进行求和,对应于论文中的公式10
            x = torch.einsum('nkctv,kvw->nctw', (x, A))#nkctv首先看做n个样本，每个样本都有3个k的值，每个对应于一种A的分区方式vw，ctv*vw，再将三种k对应的三个ctv*vw相加 
            # contiguous()把tensor x变成在内存中连续分布的形式
            return x.contiguous(), A

class Model(nn.Module):
    r"""Spatial temporal graph convolutional networks.

    Args:
        in_channels (int): Number of channels in the input data
        num_class (int): Number of classes for the classification task
        graph_args (dict): The arguments for building the graph
        edge_importance_weighting (bool): If ``True``, adds a learnable
            importance weighting to the edges of the graph
        **kwargs (optional): Other parameters for graph convolution units

    Shape:
        - Input: :math:`(N, in_channels, T_{in}, V_{in}, M_{in})`
        - Output: :math:`(N, num_class)` where
            :math:`N` is a batch size,
            :math:`T_{in}` is a length of input sequence,
            :math:`V_{in}` is the number of graph nodes,
            :math:`M_{in}` is the number of instance in a frame.
    """

    def __init__(self, in_channels, num_class, graph_args,
                 edge_importance_weighting, **kwargs):#{'in_channels': 3, 'num_class': 60, 'dropout': 0.5, 'edge_importance_weighting': True, 'graph_args': {'layout': 'ntu-rgb+d', 'strategy': 'spatial'}}
        super().__init__()


        # print(graph_args)#{'layout': 'ntu-rgb+d', 'strategy': 'spatial'}
        # print(kwargs)#{'dropout': 0.5}
        # load graph
        self.graph = Graph(**graph_args)#**graph_args 拆开字典
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        # print(A)#就是graph返回的A
        self.register_buffer('A', A)  #缓存区,可通过A访问数据

        # build networks
        spatial_kernel_size = A.size(0)
        # print(spatial_kernel_size)#当'strategy': 'spatial'时spatial_kernel_size为3
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        
        self.data_bn =  nn.BatchNorm1d(in_channels * A.size(1))#A.size(1)就是节点数
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
        # print(kwargs0)
        self.st_gcn_networks = nn.ModuleList((
            st_gcn(in_channels, 64, kernel_size, 1, residual=False, **kwargs0),
            st_gcn(64, 64, kernel_size, 1, **kwargs),
            st_gcn(64, 64, kernel_size, 1, **kwargs),
            st_gcn(64, 64, kernel_size, 1, **kwargs),
            st_gcn(64, 128, kernel_size, 2, **kwargs),
            st_gcn(128, 128, kernel_size, 1, **kwargs),
            st_gcn(128, 128, kernel_size, 1, **kwargs),
            st_gcn(128, 256, kernel_size, 2, **kwargs),
            st_gcn(256, 256, kernel_size, 1, **kwargs),
            st_gcn(256, 256, kernel_size, 1, **kwargs),
        ))

        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)
        # print(self.edge_importance)
        # print([1] * len(self.st_gcn_networks))#[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        
        
        # for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            # print(gcn)
            # print(importance)
        # fcn for prediction
        self.fcn = nn.Conv2d(256, num_class, kernel_size=1)

    def forward(self, x):
        # print(x.size())
        # data normalization
        N, C, T, V, M = x.size() #(60,3,300,25,2)
        # N, C, T, V, M = x.shape #(60,3,300,25,2)
        # print(x.shape)
        # permute()将tensor的维度换位
        x = x.permute(0, 4, 3, 1, 2).contiguous() #(60,2,25,3,300)
        x = x.view(N * M, V * C, T) #(120,75,300)
        x = self.data_bn(x) # 将某一个节点的（X,Y,C）中每一个数值在时间维度上分别进行归一化
        x = x.view(N, M, V, C, T) #(60,2,25,3,300)
        x = x.permute(0, 1, 3, 4, 2).contiguous() #（60,2,3,300,25）
        x = x.view(N * M, C, T, V) #（120,3,300,25）
        print(x.shape)
        print('666')
        # forward
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            # print(importance)#torch.ones(self.A.size())
            x, _ = gcn(x, self.A * importance)
            print(x.shape)
        # global pooling
        # 此处的x是运行完所有的卷积层之后在进行平均池化之后的维度x=(64,256,1)
        x = F.avg_pool2d(x, x.size()[2:]) # pool层的大小是(300,25),对300*25个数求平均
        print('777')
        print(x.shape)
        # print(x.view(N, M, -1, 1, 1).shape)
        # （64,256,1,1）
        x = x.view(N, M, -1, 1, 1).mean(dim=1)#torch.Size([60, 2, 256, 1, 1])求两人骨架的平均数
        # 
        print('888')
        print(x.shape)
        # prediction
        x = self.fcn(x)
        print('999')
        print(x.shape)
        # （64,400）
        x = x.view(x.size(0), -1)
        print('111')
        print(x.shape)
        return x

    def extract_feature(self, x):

        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # forward
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance) #（64,256,300,18）

        _, c, t, v = x.size() #（64,256,300,18）
        # feature的维度是（64,256,300,18,1）
        feature = x.view(N, M, c, t, v).permute(0, 2, 3, 4, 1)

        # prediction
        # (64,400,300,18)
        x = self.fcn(x)
        # output: (64,400,300,18,1)
        output = x.view(N, M, -1, t, v).permute(0, 2, 3, 4, 1)

        return output, feature


class st_gcn(nn.Module):
    r"""Applies a spatial temporal graph convolution over an input graph sequence.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format

        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.

    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dropout=0,
                 residual=True):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)#//向下取整除法
        # print(padding)#(4,0)
        self.gcn = ConvTemporalGraphical(in_channels, out_channels,
                                         kernel_size[1])#kernel_size[1]空间核尺寸
        # self.tcn()没有改变变量x.size()
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )
        # print(residual)
        # print('000')
        if(1==1):#
            #class Model中共调用10次st_gcn，次部分功能是为每次调用在orward中选不通的residual函数即res = self.residual(x)
            #第一次直接返回0
            #当输入通道数等于输出通道数时，原样返回输入数据
            #当输入通道数不等于输出通道数时，数据经过2D卷积后输出
            if not residual:
                self.residual = lambda x: 0
                # print(self.residual)
                # print('999')
            elif (in_channels == out_channels) and (stride == 1):
                self.residual = lambda x: x
                # print(self.residual)
                # print('888')
            else:
                self.residual = nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=1,
                        stride=(stride, 1)),
                    nn.BatchNorm2d(out_channels),
                )
                # print(self.residual)
                # print('777')
            # print('--------')

        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):
        # print('111')
        # print(x.shape)
        # print(A.shape)
        res = self.residual(x)
        # print(res)
        # print('111')
        x, A = self.gcn(x, A)
        # print('222')
        x = self.tcn(x) + res #(64,64,300,18)
        # (64,64,300,18)
        # print('113')
        return self.relu(x), A



if(1==1):#数据读取        
    tr_data = np.load('D:\\Desktop\\ST-GCN重构\\04_NTU\\xsub\\train_data.npy', mmap_mode='r')[0:60]
    with open('D:\\Desktop\\ST-GCN重构\\04_NTU\\xsub\\train_label.pkl', 'rb') as f:
            tr_sample_name, tr_label = pickle.load(f)


    
print(tr_data.shape) 
tr_data=torch.tensor(tr_data,dtype=torch.float32)      
mm={'in_channels': 3, 'num_class': 60, 'dropout': 0.5, 'edge_importance_weighting': True, 'graph_args': {'layout': 'ntu-rgb+d', 'strategy': 'spatial'}}
model = Model(**mm)
output = model(tr_data)
print(output.shape)

