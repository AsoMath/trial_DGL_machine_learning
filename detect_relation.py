import pandas as pd
import dgl
import torch
import numpy as np
from collections import Counter
from collections import defaultdict
import torch.optim as optim
from dgl.nn import NNConv
import torch.nn as nn
from itertools import combinations

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
    print("エッジ特徴量計算開始")

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



model = NNConvNet(
    in_feats=graph.ndata['feat'].shape[1],
    edge_feats=graph.edata['feat'].shape[1],
    hidden_feats=107,  #対象のモデルをトレーニングしたときと同じ値
    out_feats=3,
    edge_mlp_hidden=70,
    dropout=0.15117248304538475
)

model.load_state_dict(torch.load("best_model_acr85.pt"))
model.eval()

def predict_relation(mcidA, mcidB):
    if mcidA not in mcid_to_uuid or mcidB not in mcid_to_uuid:
        print("指定されたMCIDが存在しません")
        return
    uuidA, uuidB = mcid_to_uuid[mcidA], mcid_to_uuid[mcidB]
    if uuidA not in node_id_map or uuidB not in node_id_map:
        print("指定されたユーザーがノードとして存在しません")
        return
    src_id = node_id_map[uuidA]
    dst_id = node_id_map[uuidB]
    try:
        edge_id = graph.edge_ids(src_id, dst_id)
        if edge_id.dim() == 0:edge_id = edge_id.unsqueeze(0)
    except:
        print("この2人の間にはエッジが存在しません")
        return

    model.eval()
    with torch.no_grad():
        logits = model(graph, graph.ndata['feat'], graph.edata['feat'], edge_id)
        pred = torch.argmax(logits, dim=1).item()
        label_map = {0: "無関係", 1: "友人", 2: "サブアカ"}
        print(f"{mcidA} と {mcidB} の関係は: {label_map[pred]}")

def enumerate_relations(mcid, target_label):
    if mcid not in mcid_to_uuid:
        print("指定されたMCIDが存在しません")
        return
    uuid = mcid_to_uuid[mcid]
    if uuid not in node_id_map:
        print("ノードとして存在しません")
        return
    src_id = node_id_map[uuid]
    label_num = {"friend": 1, "alt": 2}.get(target_label)
    if label_num is None:
        print("指定されたラベルは 'friend' または 'alt' のみです")
        return

    results = []
    model.eval()
    src_id = node_id_map[uuid]  # 例: あるノードID
    edge_ids = graph.out_edges(src_id, form='eid') 
    for edge_id in edge_ids:
        dst_id=graph.edges()[1][edge_id].item()
        logits = model(graph, graph.ndata['feat'], graph.edata['feat'], torch.tensor([edge_id], dtype=torch.long))
        pred = torch.argmax(logits, dim=1).item()
        if pred == label_num:
            dst_uuid = id_to_node[dst_id]
            dst_mcid = uuid_to_mcid.get(dst_uuid)
            results.append(dst_mcid)
    # with torch.no_grad():
    #     for dst_id in range(graph.num_nodes()):
    #         if src_id == dst_id:continue
    #         if not graph.has_edges_between(src_id, dst_id):continue
    #         edge_id = graph.edge_ids(src_id, dst_id)
    #         edge_id = torch.tensor([edge_id], dtype=torch.long)
    #         logits = model(graph, graph.ndata['feat'], graph.edata['feat'], edge_id)
    #         pred = torch.argmax(logits, dim=1).item()
    #         if pred == label_num:
    #             dst_uuid = id_to_node[dst_id]
    #             dst_mcid = uuid_to_mcid.get(dst_uuid)
    #             results.append(dst_mcid)

    if results:
        print(f"{mcid} に対して {target_label} と判定された相手:")
        for m in results:
            print("  -", m)
    else:
        print(f"{mcid} に対して {target_label} と判定された相手はいません")


def interactive_loop():
    print("入力を待っています。例: relation <mcidA> <mcidB>  または  enumerate <mcid> <friend|alt>")
    print("終了するには exit と入力してください。")
    while True:
        try:
            cmd = input(">>> ").strip()
            if cmd == "":
                continue
            if cmd == "exit":
                break
            parts = cmd.split()
            if parts[0] == "relation" and len(parts) == 3:
                predict_relation(parts[1], parts[2])
            elif parts[0] == "enumerate" and len(parts) == 3:
                enumerate_relations(parts[1], parts[2])
            else:
                print("不明なコマンドです")
        except KeyboardInterrupt:
            break

interactive_loop()