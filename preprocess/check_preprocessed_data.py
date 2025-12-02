import torch

path = "/home/danielamrhein/master_thesis/UIP_dataset/UIP_DB_Dataset/test.pt"
data = torch.load(path, map_location=torch.device("cpu"))

acc_list = data["acc"]
ori_list = data["ori"]
vuwb_list = data["vuwb"]
pose = data["pose"]

print("Type acc:", type(acc_list), "Length acc list:", len(acc_list))
print("Type ori:", type(ori_list), "Length ori list:", len(ori_list))
print("Type vuwb:", type(vuwb_list), "Length vuwb list:", len(vuwb_list))
print("Type pose:", type(pose), "Length pose list:", len(pose))

# Peek at the first sequence
print("First acc sequence shape:", torch.tensor(acc_list[0]).shape)
print("First ori sequence shape:", torch.tensor(ori_list[0]).shape)
print("First vuwb sequence shape:", torch.tensor(vuwb_list[0]).shape)
print("First pose sequence shape:", torch.tensor(pose[0]).shape)
