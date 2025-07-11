import pandas as pd
import dgl
import torch
import numpy as np
from collections import Counter
from collections import defaultdict
from sklearn.utils.class_weight import compute_class_weight
import torch.optim as optim
from dgl.nn import NNConv
import torch.nn as nn
from sklearn.metrics import classification_report
from itertools import combinations
import optuna

#CSV読み込み
print("csv読み込み開始")
conn_df=pd.read_csv("connection_log.csv")
mes_df=pd.read_csv("message_log.csv",quoting=0, on_bad_lines='skip')
com_df=pd.read_csv("command_log.csv",quoting=0, on_bad_lines='skip')

# ノード作成
all_nodes = pd.Index(list(conn_df['uuid'].unique()))  # すべてのuuidを対象に
node_id_map = {node: i for i, node in enumerate(all_nodes)}

# ノードID対応辞書
id_to_node = {v: k for k, v in node_id_map.items()}


#変換用のmap作成
uuid_to_mcid=(
    conn_df
        .sort_values("id")
        .groupby("uuid")
        .tail(1)
        .set_index("uuid")["mcid"]
        .to_dict()
)

mcid_to_uuid = (
    conn_df
    .sort_values("id")
    .groupby("mcid")
    .tail(1)
    .set_index("mcid")["uuid"]
    .to_dict()
)

# ip_to_uuids=defaultdict(set)
# for _,row in conn_df.iterrows():
#     ip=row["ip"]
#     uuid=row["uuid"]
#     ip_to_uuids[ip].add(uuid)

print("教師ラベル読み込み開始")
no_relation_label_df=pd.read_csv("train_no_relation_labels.csv")
friend_label_df=pd.read_csv("train_friend_labels.csv")
alt_label_df=pd.read_csv("train_alt_labels.csv")
print("ラベル結合")
supervised_label_df = pd.concat([no_relation_label_df, friend_label_df, alt_label_df], ignore_index=True)
print("ノード変換")
supervised_label_df["src"]=supervised_label_df["mcid_a"].map(mcid_to_uuid)
supervised_label_df["dst"]=supervised_label_df["mcid_b"].map(mcid_to_uuid)
supervised_label_df["src_id"] = supervised_label_df["src"].map(node_id_map)
supervised_label_df["dst_id"] = supervised_label_df["dst"].map(node_id_map)
print("不正ラベル消去")
supervised_label_df = supervised_label_df.dropna(subset=["src_id", "dst_id"])
supervised_label_df=supervised_label_df.dropna(subset=["label"])
supervised_label_df["src_id"] = supervised_label_df["src_id"].astype(int)
supervised_label_df["dst_id"] = supervised_label_df["dst_id"].astype(int)
supervised_label_df["label"] = supervised_label_df["label"].astype(int)

supervised_src_ids = supervised_label_df["src_id"].values
supervised_dst_ids = supervised_label_df["dst_id"].values
supervised_labels = torch.tensor(supervised_label_df["label"].values, dtype=torch.long)


print("テストデータ読み込み")
test_no_relation_label_df=pd.read_csv("test_no_relation_labels.csv")
test_friend_label_df=pd.read_csv("test_friend_labels.csv")
test_alt_label_df=pd.read_csv("test_alt_labels.csv")
print("ラベル結合")
test_df = pd.concat([test_no_relation_label_df, test_friend_label_df, test_alt_label_df], ignore_index=True)

test_df["src"] = test_df["mcid_a"].map(mcid_to_uuid)
test_df["dst"] = test_df["mcid_b"].map(mcid_to_uuid)
test_df = test_df.dropna(subset=["src", "dst", "label"])

test_df["src_id"] = test_df["src"].map(node_id_map).astype(int)
test_df["dst_id"] = test_df["dst"].map(node_id_map).astype(int)
test_df["label"] = test_df["label"].astype(int)

test_src_ids = test_df["src_id"].values
test_dst_ids = test_df["dst_id"].values
test_labels = torch.tensor(test_df["label"].values, dtype=torch.long)


print("エッジ特徴量計算開始")
#関係のあるコマンド群を登録
target_commands = [ "/pay",
                    "/mpay",
                    "/mre acceptuser",
                    "/tell",
                    "/m",
                    "/donjara join",
                    "/poker join",
                    "/bjp join",
                    "/mdsend",
                    "/w",
                    "/ipay",
                    "/mlend",
                    "/thank",
                    "/fuck"]

def extract_target_name(command):
    for prefix in target_commands:
        if command.startswith(prefix):
            args = command[len(prefix):].strip().split()
            if len(args) >= 1:
                return args[0]  # プレイヤー名と思しき第1引数を返す
    return None





target_com_df = com_df[com_df["command"].apply(lambda x: any(x.startswith(prefix) for prefix in target_commands))]
target_com_df["target_name"]=target_com_df["command"].apply(extract_target_name)
target_com_df["target_uuid"]=target_com_df["target_name"].map(mcid_to_uuid)

# # コマンドタイプ分類
# command_types = {
#     "/pay": "money",
#     "/mpay": "money",
#     "/mlend": "money",
#     "/mre acceptuser": "mre",
#     "/mdsend": "item",
#     "/ipay": "item",
#     "/tell": "msg",
#     "/m": "msg",
#     "/w": "msg",
#     "/r": "msg",
#     "/thank": "good_reaction",
#     "/fuck": "bad_reaction",
# }

# type_keys = ["money", "mre","item", "msg", "good_reaction","bad_reaction"]
counts = conn_df.groupby(["uuid", "ip"]).size().reset_index(name="count")
uuid_to_ipcount = defaultdict(list)
for _, row in counts.iterrows():
    uuid_to_ipcount[row["uuid"]].append((row["ip"], row["count"]))
# 通常のdictに変換（必要であれば）
uuid_to_ipcount = dict(uuid_to_ipcount)
uuid_to_ipset = {uuid: set(ip for ip, _ in ipcounts) for uuid, ipcounts in uuid_to_ipcount.items()}

def ipDupRatio(uuidA,uuidB):
    # IPチェック
    dupilicatedIpCount=0
    sumOfConnection=0
    if uuidA not in uuid_to_ipset or uuidB not in uuid_to_ipset:return 0
    if not (uuid_to_ipset[uuidA] & uuid_to_ipset[uuidB]): return 0
    for _,count in uuid_to_ipcount[uuidA]:
        sumOfConnection+=count

    for ipA,countA in uuid_to_ipcount[uuidA]:
        for ipB,_ in uuid_to_ipcount[uuidB]:
            if ipA==ipB:
                dupilicatedIpCount+=countA
                break

    return dupilicatedIpCount/sumOfConnection

def aggregate_edge_command_counts():
    # 集計用の辞書を作る
    data = []

    #この処理アホみたいに遅いのでむり
    processed_pairs = set()
    # for ip, uuid_list in ip_to_uuids.items():
    #     for src, dst in combinations(uuid_list, 2):
    #         if src == dst:
    #             continue
    #         # 重複ペアの処理防止（順序付きにしておく）
    #         pair = (src, dst)
    #         if pair in processed_pairs:
    #             continue

    #         group = target_com_df[(target_com_df["uuid"] == src) & (target_com_df["target_uuid"] == dst)]
    #         total = len(group)

    #         counts = {}
    #         for cmd in target_commands:
    #             count = group["command"].str.startswith(cmd + " ").sum()
    #             counts[cmd + "_amount"] = count
    #             counts[cmd + "_ratio"] = count / total if total > 0 else np.nan

    #         counts["src"] = src
    #         counts["dst"] = dst
    #         counts["ipdup_ratio"] = ipDupRatio(src, dst)

    #         data.append(counts)
    #         processed_pairs.add(pair)

    print("trainデータの特徴量計算開始")
    print("合計"+str(len(supervised_label_df))+"個")

    for i,row in supervised_label_df.iterrows():
        print(str(i)+"個目")
        counts = {}
        src=row["src"]
        dst=row["dst"]
        group=target_com_df[(target_com_df["uuid"]==src)&(target_com_df["target_uuid"]==dst)]
        total=len(group)
        for cmd in target_commands:
            count=group["command"].str.startswith(cmd+" ").sum()
            counts[cmd+"_amount"] =count
            counts[cmd+"_ratio"]=count/total if total>0 else np.nan
        counts["src"] = src
        counts["dst"] = dst
        counts["ipdup_ratio"]=ipDupRatio(src,dst)
        data.append(counts)

        counts = {}
        group=target_com_df[(target_com_df["uuid"]==dst)&(target_com_df["target_uuid"]==src)]
        total=len(group)
        for cmd in target_commands:
            count=group["command"].str.startswith(cmd+" ").sum()
            counts[cmd+"_amount"] =count
            counts[cmd+"_ratio"]=count/total if total>0 else np.nan
        counts["src"] = dst
        counts["dst"] = src
        counts["ipdup_ratio"]=ipDupRatio(dst,src)
        data.append(counts)
        processed_pairs.add((src,dst))
        processed_pairs.add((dst,src))

    print("testデータの特徴量計算開始")
    print("合計"+str(len(test_df))+"個")

    for i,row in test_df.iterrows():
        print(str(i)+"個目")
        counts = {}
        src=row["src"]
        dst=row["dst"]
        group=target_com_df[(target_com_df["uuid"]==src)&(target_com_df["target_uuid"]==dst)]
        total=len(group)
        for cmd in target_commands:
            count=group["command"].str.startswith(cmd+" ").sum()
            counts[cmd+"_amount"] =count
            counts[cmd+"_ratio"]=count/total if total>0 else np.nan
        counts["src"] = src
        counts["dst"] = dst
        counts["ipdup_ratio"]=ipDupRatio(src,dst)
        data.append(counts)

        counts = {}
        group=target_com_df[(target_com_df["uuid"]==dst)&(target_com_df["target_uuid"]==src)]
        total=len(group)
        for cmd in target_commands:
            count=group["command"].str.startswith(cmd+" ").sum()
            counts[cmd+"_amount"] =count
            counts[cmd+"_ratio"]=count/total if total>0 else np.nan
        counts["src"] = dst
        counts["dst"] = src
        counts["ipdup_ratio"]=ipDupRatio(dst,src)
        data.append(counts)
        processed_pairs.add((src,dst))
        processed_pairs.add((dst,src))


    print("残りのエッジの特徴量計算開始")

    for (src, dst), group in target_com_df.groupby(["uuid", "target_uuid"]):
        if (src, dst) in processed_pairs:continue
        counts = {}
        total=len(group)
        for cmd in target_commands:
            count=group["command"].str.startswith(cmd+" ").sum()
            counts[cmd+"_amount"] =count
            counts[cmd+"_ratio"]=count/total if total>0 else np.nan
        counts["src"] = src
        counts["dst"] = dst
        counts["ipdup_ratio"]=ipDupRatio(src,dst)
        data.append(counts)
    
    return pd.DataFrame(data)


edge_feature_df = aggregate_edge_command_counts()
edge_feature_df = edge_feature_df.fillna(0)
print("ノード特徴量計算開始")
# ノードID列挙

# 1. チャット回数
chat_counts = mes_df.groupby("uuid").size()

# 2. コマンド回数
command_counts = com_df.groupby("uuid").size()

# 3. ログイン合計時間
conn_df["connection_seconds"] = pd.to_numeric(conn_df["connection_seconds"], errors='coerce').fillna(0)
login_times = conn_df.groupby("uuid")["connection_seconds"].sum()

# 4 ログイン回数
login_counts=conn_df.groupby("uuid").size()

# 5. 異なるIP数
ip_counts = pd.Series({uuid: len(ip_set) for uuid, ip_set in uuid_to_ipset.items()})

# 6. 時間帯別ログイン時間割合（例：朝6-12時、昼12-18時、夜18-24時、深夜0-6時）
conn_df["hour"] = pd.to_datetime(conn_df["connected_time"]).dt.hour
def time_slot(hour):
    if 6 <= hour < 12:
        return "morning"
    elif 12 <= hour < 18:
        return "afternoon"
    elif 18 <= hour < 24:
        return "evening"
    else:
        return "night"
conn_df["time_slot"] = conn_df["hour"].apply(time_slot)

time_slot_sum = conn_df.groupby(["uuid", "time_slot"])["connection_seconds"].sum().unstack(fill_value=0)
time_slot_ratio = time_slot_sum.div(time_slot_sum.sum(axis=1), axis=0).fillna(0)

# 7. 特徴行列作成
node_features = []
for node in all_nodes:
    c = chat_counts.get(node, 0)
    cm = command_counts.get(node, 0)
    lt = login_times.get(node, 0)
    lc = login_counts.get(node,0)
    ipn = ip_counts.get(node, 0)
    tsr = time_slot_ratio.loc[node] if node in time_slot_ratio.index else pd.Series([0,0,0,0], index=["morning","afternoon","evening","night"])
    feat_vec = [c, cm, lt, lc, ipn] + tsr.tolist()
    node_features.append(feat_vec)

node_features = np.array(node_features, dtype=np.float32)
print("グラフ作成開始")
src_ids = edge_feature_df["src"].map(node_id_map).values.astype(np.int64)
dst_ids = edge_feature_df["dst"].map(node_id_map).values.astype(np.int64)
graph = dgl.graph((src_ids, dst_ids), num_nodes=len(all_nodes))
graph.ndata['feat'] = torch.tensor(node_features, dtype=torch.float32)
edge_feats = edge_feature_df.drop(columns=["src", "dst"]).to_numpy()
graph.edata['feat'] = torch.tensor(edge_feats, dtype=torch.float32)
print(graph)
print("Node features shape:", graph.ndata['feat'].shape)
print("Edge features shape:", graph.edata['feat'].shape)

label_edge_pairs = list(zip(src_ids, dst_ids))
label_index_map = {(s,d): i for i, (s,d) in enumerate(label_edge_pairs)}


#損失関数の作成
class_weights = compute_class_weight(class_weight='balanced',
                                     classes=np.array([0, 1, 2]),
                                     y=supervised_labels.numpy())
class_weights = torch.tensor(class_weights, dtype=torch.float32)
loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

print("学習モデル構築開始")
class NNConvNet(nn.Module):
    def __init__(self, in_feats, edge_feats, hidden_feats, out_feats,edge_mlp_hidden,dropout):
        super().__init__()
        self.edge_network = nn.Sequential(
            nn.Linear(edge_feats, edge_mlp_hidden * in_feats),
            nn.ReLU(),
            nn.Linear(edge_mlp_hidden * in_feats, hidden_feats * in_feats)
        )
        self.conv = NNConv(in_feats, hidden_feats, self.edge_network)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.edge_classifier = nn.Sequential(
            nn.Linear(hidden_feats * 2 + edge_feats, hidden_feats),
            nn.ReLU(),
            nn.Linear(hidden_feats, out_feats)
        )

    def forward(self, g, node_feats, edge_feats, edge_indices):
        h = self.relu(self.conv(g, node_feats, edge_feats))
        h = self.dropout(h)
        src, dst = g.find_edges(edge_indices)
        src_h = h[src]
        dst_h = h[dst]
        e_feat = edge_feats[edge_indices]
        edge_input = torch.cat([src_h, dst_h, e_feat], dim=1)
        logits = self.edge_classifier(edge_input)
        return logits
    # def __init__(self, in_feats, edge_feats, hidden_feats, out_feats,lr,edge_mlp_hidden,dropout):
    #     super().__init__()
    #     # エッジ重み生成ネットワーク（MLP）
    #     self.edge_network = nn.Sequential(
    #         nn.Linear(edge_feats, hidden_feats * in_feats),
    #         nn.ReLU(),
    #         nn.Linear(hidden_feats * in_feats, hidden_feats * in_feats)
    #     )
    #     # NNConvレイヤー（ノード埋め込みを生成）
    #     self.conv = NNConv(in_feats, hidden_feats, self.edge_network)
    #     self.relu = nn.ReLU()
    #     self.dropout = nn.Dropout(dropout)

    #     # エッジ分類MLP：src, dstノード埋め込み + エッジ特徴量 を入力
    #     self.edge_classifier = nn.Sequential(
    #         nn.Linear(hidden_feats * 2 + edge_feats, hidden_feats),
    #         nn.ReLU(),
    #         nn.Linear(hidden_feats, out_feats)
    #     )

    # def forward(self, g, node_feats, edge_feats, edge_indices):
    #     # ノード埋め込み
    #     h = self.relu(self.conv(g, node_feats, edge_feats))

    #     # ラベル付きエッジのsrc/dstインデックス
    #     src, dst = g.find_edges(edge_indices)

    #     # 各ラベル付きエッジに対し、ノード埋め込みとエッジ特徴を取り出す
    #     src_h = h[src]
    #     dst_h = h[dst]
    #     e_feat = edge_feats[edge_indices]

    #     # エッジ特徴量とノード埋め込みを連結して分類器に渡す
    #     edge_input = torch.cat([src_h, dst_h, e_feat], dim=1)
    #     logits = self.edge_classifier(edge_input)

    #     return logits
supervised_label_edge_indices = graph.edge_ids(supervised_src_ids, supervised_dst_ids)
model = NNConvNet(in_feats=node_features.shape[1],edge_feats=edge_feats.shape[1], hidden_feats=50, out_feats=3,dropout=0.4550633917745034,edge_mlp_hidden=57)
logits = model(graph, graph.ndata['feat'],graph.edata["feat"],supervised_label_edge_indices)

optimizer = torch.optim.Adam(model.parameters(), lr=0.007554552733655542)

#モデルのチェック
print(np.isnan(node_features).any())
print(np.isnan(edge_feats).any())
print(np.isnan(supervised_labels).any())
print(torch.isnan(logits).any())

#過学習対策
best_loss = float('inf')
patience = 5

#連続でロスの更新がされなかった回数をカウント
counter = 0
def evaluate_validation_loss(model, graph, edge_idx, labels, loss_fn):
    model.eval()
    with torch.no_grad():
        logits = model(graph, graph.ndata['feat'], graph.edata['feat'], edge_idx)
        val_loss = loss_fn(logits, labels)
        return val_loss.item()

print("学習開始")
for epoch in range(160):
    model.train()
    optimizer.zero_grad()

    logits = model(graph, graph.ndata['feat'], graph.edata['feat'],supervised_label_edge_indices)
    loss = loss_fn(logits, supervised_labels)

    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch}, loss: {loss.item()}")
    val_loss = evaluate_validation_loss(model,graph,supervised_label_edge_indices,supervised_labels,loss_fn)

    if val_loss < best_loss:
        best_loss = val_loss
        counter = 0
    else:
        counter += 1

    if counter >= patience:
        print(f"Early stopping at epoch {epoch}")
        break

test_edge_indices = graph.edge_ids(test_src_ids, test_dst_ids)

model.eval()
with torch.no_grad():
    test_logits = model(graph, graph.ndata['feat'], graph.edata['feat'], test_edge_indices)
    test_pred = torch.argmax(test_logits, dim=1)

print("=== Test Classification Report ===")
print(classification_report(test_labels.numpy(), test_pred.numpy(), labels=[0,1,2]))

torch.save(model.state_dict(), "best_model.pt")

# ハイパーパラメータ選定するやつ
# print("optunaでやってみよう")
# def objective(trial):
#     # 最適化対象のハイパーパラメータ
#     hidden_feats = trial.suggest_int("hidden_feats", 16, 128)
#     edge_mlp_hidden = trial.suggest_int("edge_mlp_hidden", 16, 128)
#     lr = trial.suggest_loguniform("lr", 1e-4, 1e-2)
#     dropout_rate = trial.suggest_uniform("dropout", 0.0, 0.5)

#     class NNConvNet(nn.Module):
#         def __init__(self, in_feats, edge_feats, hidden_feats, out_feats):
#             super().__init__()
#             self.edge_network = nn.Sequential(
#                 nn.Linear(edge_feats, edge_mlp_hidden * in_feats),
#                 nn.ReLU(),
#                 nn.Linear(edge_mlp_hidden * in_feats, hidden_feats * in_feats)
#             )
#             self.conv = NNConv(in_feats, hidden_feats, self.edge_network)
#             self.relu = nn.ReLU()
#             self.dropout = nn.Dropout(dropout_rate)
#             self.edge_classifier = nn.Sequential(
#                 nn.Linear(hidden_feats * 2 + edge_feats, hidden_feats),
#                 nn.ReLU(),
#                 nn.Linear(hidden_feats, out_feats)
#             )

#         def forward(self, g, node_feats, edge_feats, edge_indices):
#             h = self.relu(self.conv(g, node_feats, edge_feats))
#             h = self.dropout(h)
#             src, dst = g.find_edges(edge_indices)
#             src_h = h[src]
#             dst_h = h[dst]
#             e_feat = edge_feats[edge_indices]
#             edge_input = torch.cat([src_h, dst_h, e_feat], dim=1)
#             logits = self.edge_classifier(edge_input)
#             return logits

#     model = NNConvNet(
#         in_feats=node_features.shape[1],
#         edge_feats=edge_feats.shape[1],
#         hidden_feats=hidden_feats,
#         out_feats=3
#     )

#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)

#     # Early stopping
#     best_loss = float('inf')
#     counter = 0
#     patience = 5

#     for epoch in range(100):  # 最大100エポック
#         model.train()
#         optimizer.zero_grad()
#         logits = model(graph, graph.ndata['feat'], graph.edata['feat'], supervised_label_edge_indices)
#         loss = loss_fn(logits, supervised_labels)
#         loss.backward()
#         optimizer.step()

#         val_loss = evaluate_validation_loss(model, graph, supervised_label_edge_indices, supervised_labels, loss_fn)
#         if val_loss < best_loss:
#             best_loss = val_loss
#             counter = 0
#         else:
#             counter += 1
#         if counter >= patience:
#             break

#     return best_loss

# study = optuna.create_study(direction="minimize")
# study.optimize(objective, n_trials=30)  # 30回試す

# print("Best params:", study.best_params)