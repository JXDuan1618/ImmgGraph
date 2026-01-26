import torch
import dgl
import dgl.nn as dglnn
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(ResidualBlock, self).__init__()
        self.lin1 = nn.Linear(in_features, out_features)
        self.bn1 = nn.BatchNorm1d(out_features)
        self.lin2 = nn.Linear(out_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)

        # 如果输入输出维度不同，需要投影层
        self.downsample = None
        if in_features != out_features:
            self.downsample = nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features)
            )

    def forward(self, x):
        identity = x

        out = self.lin1(x)
        out = self.bn1(out)
        out = F.leaky_relu(out, negative_slope=0.2)

        out = self.lin2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.leaky_relu(out, negative_slope=0.2)

        return out

class Classifier(torch.nn.Module):
    def __init__(self, dna_num_classes=2, rna_num_classes=2, pro_num_classes=2):
        super(Classifier, self).__init__()
        # RNA分支
        self.rna_layers = nn.Sequential(
            nn.BatchNorm1d(9084),
            nn.Linear(9084, 1024),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.45),  # 增加 Dropout 比例
            ResidualBlock(1024, 512),
            nn.Dropout(0.45),
            ResidualBlock(512, 256),
            nn.Dropout(0.45),
            nn.Linear(256, dna_num_classes)
        )



        # DNA分支
        self.dna_layers = nn.Sequential(
            nn.BatchNorm1d(9084),
            nn.Linear(9084, 1024),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.45),  # 增加 Dropout 比例
            ResidualBlock(1024, 512),
            nn.Dropout(0.45),
            ResidualBlock(512, 256),
            nn.Dropout(0.45),
            nn.Linear(256, dna_num_classes)
        )


        # 蛋白质分支
        self.pro_layers = nn.Sequential(
            torch.nn.BatchNorm1d(9084),
            torch.nn.Linear(9084, 1024),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(1024),
            torch.nn.Dropout(0.3),

            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(512),
            torch.nn.Dropout(0.3),

            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(256),
            torch.nn.Dropout(0.3),

            torch.nn.Linear(256, pro_num_classes)
        )

    # number=number of nodes, transform the out put, from number of nodes to number of classes/patients
    def forward(self, x):
        #print(f"Input to DNA BatchNorm1d: {x.shape}")
        x_dna = self.dna_layers(x)

        #print(f"Input to Protein BatchNorm1d: {x.shape}")
        x_pro = self.pro_layers(x)

        #print(f"Input to RNA BatchNorm1d: {x.shape}")

        x_rna = self.rna_layers(x)

        rna_weight = 1
        dna_weight = 1.5


        return rna_weight * x_rna, dna_weight * x_dna, x_pro



class HeteroRGCNLayer(nn.Module):
    def __init__(self, in_size, out_size, etypes):
        super(HeteroRGCNLayer, self).__init__()
        # W_r for each relation
        self.attenton = nn.ModuleDict(
            {name: nn.Sequential(nn.Conv2d(in_size, out_size, kernel_size=3, stride=1, padding=1))
             # nn.Conv2d(8, out_size, kernel_size=3, stride=1, padding=1) )
             for name in etypes}
        )

    def forward(self, G, feat_dict):
        funcs = {}
        Wh = {}
        for srctype, etype, dsttype in G.canonical_etypes:
            #print(f"{etype}: num_edges = {G.num_edges(etype)}")  # 放这里
            G.apply_edges(fn.u_add_v('feat', 'feat', etype))
            Wh[etype] = self.attenton[etype](G.edges[etype].data[etype].unsqueeze(0).to(device))
            G.edges[etype].data[etype] = Wh[etype].squeeze(0)
            funcs[etype] = (fn.u_add_e("feat", etype, 'm'), fn.mean("m", "h"))
        G.multi_update_all(funcs, "mean")
        return {ntype: G.nodes[ntype].data["h"] for ntype in G.ntypes}, Wh




class HeteroRGCN(nn.Module):
    def __init__(self, G, in_size=1, hidden_size=1, out_size=1):
        super(HeteroRGCN, self).__init__()
        # create layers
        self.layer1 = HeteroRGCNLayer(in_size, hidden_size, G.etypes)
        self.layer2 = HeteroRGCNLayer(hidden_size, out_size, G.etypes)
    def forward(self, G):
        input_dict = {ntype: G.nodes[ntype].data["feat"] for ntype in G.ntypes}
        h_dict, _ = self.layer1(G, input_dict)
        # for k, h in h_dict.items():
        # print("*********kkk********",k)
        # print("*********hhh********",h)
        h_dict = {k: F.leaky_relu(h) for k, h in h_dict.items()}
        h_dict, edge_weight = self.layer2(G, h_dict)
        return h_dict, edge_weight


class clf_graph(nn.Module):

    def __init__(self, graph):
        super().__init__()
        self.clf = Classifier()
        self.h_model = HeteroRGCN(G=graph)

    def forward(self, graph):

        # h_model = self.HeteroRGCN(graph, in_size=1,hidden_size=1,out_size=1).to(device)

        # model_rgcn = RGCN(nodes_rna.shape[1], nodes_rna.shape[1], nodes_rna.shape[1], graph.etypes).to(device)

        # node_features = {'rna': fea_, 'patch': patch_fea}


        final_graph, edge_weight = self.h_model(graph)
        logits_dna = final_graph['dna']
        logits_rna = final_graph['rna']
        logits_pro = final_graph['protein']

        logits = torch.cat((logits_dna, logits_rna, logits_pro), dim=0)
        out_dna, out_rna, out_pro = self.clf(logits.t())

        return out_dna, out_rna, out_pro, edge_weight, logits_dna, logits_rna, logits_pro


if __name__ == "__main__":
    feature_extractor = clf_graph()
    f = feature_extractor()








